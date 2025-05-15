import sys
import logging
import io
import soundfile as sf
import math
try: 
    import torch
except ImportError: 
    torch = None
from typing import List
import numpy as np
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)

class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(self.start + offset, self.end + offset, self.text)

    def __repr__(self):
        return f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"

    def load_model(self, modelsize, cache_dir, model_dir):
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped as the backend."""
    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped

        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            verbose=None,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return result

    def ts_words(self, r) -> List[ASRToken]:
        """
        Converts the whisper_timestamped result to a list of ASRToken objects.
        """
        tokens = []
        for segment in r["segments"]:
            for word in segment["words"]:
                token = ASRToken(word["start"], word["end"], word["text"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [segment["end"] for segment in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        import os

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. "
                         f"modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("Either modelsize or model_dir must be set")
        
        # Enhanced device detection and configuration
        device = "cpu"
        compute_type = "int8"
        
        if torch:
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"  # Default to float16 for CUDA
                torch.cuda.empty_cache()  # Clear CUDA memory before loading model
                
                # Check available GPU memory and adjust settings accordingly
                try:
                    free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                    logger.info(f"Available GPU memory: {free_memory:.2f}GB of {total_memory:.2f}GB total")
                    
                    # For high-end GPUs like A100, use float16 for accuracy 
                    if total_memory >= 40:  # A100 40GB+
                        compute_type = "float16"
                        logger.info(f"Using {compute_type} precision on high-memory GPU")
                    # For devices with limited memory, use int8 quantization
                    elif free_memory < 2.0 and "large" in str(model_size_or_path).lower():
                        compute_type = "int8"
                        logger.info(f"Using {compute_type} precision for {model_size_or_path} due to limited GPU memory")
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {e}, using default settings")
            
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                compute_type = "float16"
                
        # Optimize for model size and task
        beam_size = 3  # Default
        if "tiny" in str(model_size_or_path).lower() or "base" in str(model_size_or_path).lower():
            beam_size = 2  # Faster for small models
        elif "large" in str(model_size_or_path).lower() or "distil" in str(model_size_or_path).lower():
            beam_size = 3  # Better accuracy for large models
        
        # Advanced configuration with adaptive parameters
        best_of = None  # Default from faster-whisper
        if device == "cuda":
            # Use more threads on GPU for parallel processing
            cpu_threads = 4
            # Use more workers for loading when GPU is available
            num_workers = 1 if "large" not in str(model_size_or_path).lower() else 2
        else:
            # Use more CPU threads when running on CPU
            cpu_threads = min(8, os.cpu_count() or 4)
            num_workers = 1  # Fewer workers on CPU to avoid contention
            
        # Cache model parameters for future reference
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.model_size = model_size_or_path
        
        # Memory cleanup
        import gc
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
                              
        logger.info(f"Loading model on {device} with {compute_type} precision")
        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            local_files_only=False  # Allow downloading if needed
        )
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        # Optimize transcription parameters based on audio length
        audio_length = len(audio) / 16000  # in seconds
        
        # Adjust beam size based on audio length for speed/accuracy tradeoff
        adaptive_beam_size = self.beam_size
        if audio_length > 15.0:  # For longer segments
            adaptive_beam_size = max(2, self.beam_size - 1)  # Reduce beam size
            
        # For very short segments, we can use more aggressive settings
        if audio_length < 2.0:
            adaptive_beam_size = 1  # Fastest setting

        # Optimize VAD parameters if enabled
        vad_parameters = {}
        if self.transcribe_kargs.get('vad_filter', False):
            vad_parameters = {
                'vad_filter': True,
                'vad_parameters': {
                    'min_silence_duration_ms': 300,  # Shorter silence detection
                    'threshold': 0.45,  # Slightly more sensitive
                }
            }

        # Apply optimized transcribe parameters
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=adaptive_beam_size,
            word_timestamps=True,
            condition_on_previous_text=True,
            temperature=0.0,  # Use greedy decoding for consistency
            compression_ratio_threshold=2.2,  # Slightly higher to filter out nonsense
            log_prob_threshold=-0.8,  # Slightly higher to filter low confidence
            no_speech_threshold=0.6,  # Slightly more aggressive
            **vad_parameters,
            **{k: v for k, v in self.transcribe_kargs.items() if k != 'vad_filter'}  # Other params
        )
        return list(segments)
    
    def transcribe_with_params(self, audio: np.ndarray, init_prompt: str = "", **kwargs) -> list:
        """
        Transcribe with custom parameters for dynamic optimization.
        This allows for runtime parameter adjustments to reduce lag during long sessions.
        """
        # Set base parameters
        beam_size = kwargs.get('beam_size', self.beam_size)
        vad_filter = kwargs.get('vad_filter', self.transcribe_kargs.get('vad_filter', False))
        
        # Build VAD parameters
        vad_parameters = {}
        if vad_filter:
            vad_parameters = {
                'vad_filter': True,
                'vad_parameters': {
                    'min_silence_duration_ms': 300,
                    'threshold': 0.45,
                }
            }
            
        # Apply optimized transcribe parameters with custom overrides
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=beam_size,
            word_timestamps=True,
            condition_on_previous_text=kwargs.get('condition_on_previous_text', True),
            temperature=kwargs.get('temperature', 0.0),
            compression_ratio_threshold=kwargs.get('compression_ratio_threshold', 2.2),
            log_prob_threshold=kwargs.get('log_prob_threshold', -0.8),
            no_speech_threshold=kwargs.get('no_speech_threshold', 0.6),
            **vad_parameters,
            **{k: v for k, v in self.transcribe_kargs.items() 
               if k != 'vad_filter' and k not in kwargs}
        )
        
        # Force memory cleanup for long running sessions
        if torch and torch.cuda.is_available() and self.device == "cuda":
            # Only run garbage collection every 10 calls to avoid overhead
            if not hasattr(self, '_gc_counter'):
                self._gc_counter = 0
            self._gc_counter += 1
            
            if self._gc_counter % 10 == 0:
                torch.cuda.empty_cache()
                
        return list(segments)
        
    def transcribe_batch(self, audio_batch: list) -> list:
        """Process a batch of audio samples efficiently using a single model.
        
        This method is optimized for A100 GPUs and large models like distil-whisper-v3.
        """
        if not torch or not torch.cuda.is_available() or self.device != "cuda":
            logger.warning("Batch processing requires CUDA. Falling back to sequential processing.")
            return [self.transcribe(audio) for audio in audio_batch]
            
        batch_size = len(audio_batch)
        results = []
        
        try:
            # Process in smaller sub-batches if needed for very large batches
            max_sub_batch = 16
            
            # For A100 with 40GB+, we can handle larger sub-batches
            if torch.cuda.get_device_properties(0).total_memory > 40 * (1024**3):
                max_sub_batch = 24
            
            # For distil-whisper-v3 on A100 80GB+, we can go even larger
            if "distil" in str(self.model_size).lower() and torch.cuda.get_device_properties(0).total_memory > 80 * (1024**3):
                max_sub_batch = 32
                
            logger.debug(f"Processing batch of {batch_size} audio samples with max sub-batch size {max_sub_batch}")
            
            # Process in sub-batches
            for i in range(0, batch_size, max_sub_batch):
                sub_batch = audio_batch[i:i+max_sub_batch]
                sub_results = []
                
                # Use speed-optimized parameters for batch processing to reduce lag
                beam_size = 1  # Use fastest beam size for batches
                
                # Use common parameters for all samples in the batch
                for audio in sub_batch:
                    audio_length = len(audio) / 16000
                    # Even more simplified processing for batches to reduce lag
                    if audio_length > 10.0:
                        # Use fastest settings for long audio in batches
                        segments = self.transcribe_with_params(
                            audio, 
                            beam_size=1,
                            condition_on_previous_text=False,  # Faster processing
                            temperature=0.0
                        )
                    else:
                        segments = self.transcribe(audio)
                        
                    sub_results.append({
                        "segments": segments,
                        "tokens": self.ts_words(segments)
                    })
                    
                results.extend(sub_results)
                
                # Clean up GPU memory after each sub-batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            return results
                
        except Exception as e:
            logger.error(f"Error in batch transcription: {e}")
            # Fall back to sequential processing
            return [{"error": str(e), "segments": [], "tokens": []}] * batch_size

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            # Skip segments with high no_speech probability
            if hasattr(segment, 'no_speech_prob') and segment.no_speech_prob > 0.85:  # More aggressive filtering
                continue
                
            # Process words with confidence filtering for better accuracy
            for word in segment.words:
                # Skip low-confidence words
                if hasattr(word, 'probability') and word.probability < 0.25:  # Reduced threshold to retain more words
                    continue
                    
                token = ASRToken(word.start, word.end, word.word, probability=word.probability)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper optimized for Apple Silicon.
    """
    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
            logger.debug(f"Loading whisper model {modelsize}. You use mlx whisper, so {model_size_or_path} will be used.")
        else:
            raise ValueError("Either modelsize or model_dir must be set")

        self.model_size_or_path = model_size_or_path
        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }
        mlx_model_path = model_mapping.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
        )
        return segments.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                token = ASRToken(word["start"], word["end"], word["word"], probability=word["probability"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for transcription."""
    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments) -> List[ASRToken]:
        """
        Converts OpenAI API response words into ASRToken objects while
        optionally skipping words that fall into no-speech segments.
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment.no_speech_prob > 0.8:
                    no_speech_segments.append((segment.start, segment.end))
        tokens = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            tokens.append(ASRToken(start, end, word.word))
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt
        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"


class ParakeetTDTASR(ASRBase):
    """Uses NVIDIA's Parakeet TDT model as the backend through NeMo toolkit."""
    sep = " "  # Changed from empty string to space for proper word separation

    def __init__(self, lan=None, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        # Call parent's init but keep our own parameters
        super().__init__(lan, modelsize, cache_dir, model_dir, logfile)
        
        # Add a basic sentence tokenizer for when using "sentence" buffer trimming mode
        self.tokenizer = self._create_basic_tokenizer()
        
    def _create_basic_tokenizer(self):
        """Create a simple tokenizer for sentence segmentation since Parakeet TDT doesn't provide one"""
        class BasicTokenizer:
            def split(self, text):
                # Simple regex-based sentence splitting
                import re
                text = text.strip()
                # Split on common sentence endings (period, question mark, exclamation mark) followed by space
                sentences = re.split(r'(?<=[.!?])\s+', text)
                return [s for s in sentences if s.strip()]  # Remove empty strings
                
        return BasicTokenizer()

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        try:
            # First, try to properly import NeMo
            try:
                import nemo.collections.asr as nemo_asr
                logger.info("Successfully imported NVIDIA NeMo toolkit")
            except ImportError:
                logger.error("NVIDIA NeMo toolkit is not installed. Installing required packages...")
                
                # Try to install NeMo dependencies
                import subprocess
                import sys
                
                # Install minimal dependencies needed for inference
                subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                      "nemo_toolkit[asr]>=2.0.0", "--no-deps"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                      "torch>=2.0.0", "torchvision", "torchaudio"])
                
                # Try to import again after installation
                import nemo.collections.asr as nemo_asr
                logger.info("Successfully installed and imported NVIDIA NeMo toolkit")
            
            # Set model name for Parakeet TDT
            model_name = "nvidia/parakeet-tdt-0.6b-v2"
            if modelsize and modelsize.startswith("nvidia/parakeet"):
                model_name = modelsize
            # model_name = "nvidia/stt_en_fastconformer_transducer_large"
            # if modelsize and modelsize.startswith("nvidia/canary"):
            #     model_name = modelsize
            logger.info(f"Loading Parakeet TDT model: {model_name}")
            
            # Clear CUDA cache before loading model if using GPU
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # Check available GPU memory
                try:
                    free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                    logger.info(f"Available GPU memory: {free_memory:.2f}GB of {total_memory:.2f}GB total")
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {e}")
            
            # Load model directly from NeMo hub
            logger.info(f"Attempting to load model {model_name} from NeMo hub")
            self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            
            # Load the model from NeMo hub
            # NeMo doesn't accept cache_dir parameter, so we don't pass it
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name
            )
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Store model information
            self.model = model
            self.model_size = model_name
            
            logger.info(f"Successfully loaded Parakeet TDT model on {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Parakeet TDT model: {e}")
            
            # Fallback to a Whisper model if Parakeet fails
            try:
                logger.warning("Falling back to Whisper model...")
                from faster_whisper import WhisperModel
                
                fallback_model = "large-v3"
                logger.info(f"Loading fallback Whisper model: {fallback_model}")
                
                # Configure device and compute type
                device = "cuda" if torch and torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                # Load the model
                model = WhisperModel(
                    fallback_model,
                    device=device,
                    compute_type=compute_type,
                    download_root=cache_dir
                )
                
                # Record information about the fallback model
                self.model = model
                self.fallback = True
                self.model_size = fallback_model
                self.device = device
                
                logger.info(f"Successfully loaded fallback Whisper model on {device}")
                return model
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError(f"Failed to load Parakeet TDT model: {e}. Fallback also failed: {fallback_error}")
            
            raise RuntimeError(f"Failed to load Parakeet TDT model: {e}")

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        """
        Transcribe audio using the Parakeet TDT model.
        """
        try:
            # Check if we're using the fallback model
            if hasattr(self, 'fallback') and self.fallback:
                # Using faster-whisper fallback
                segments, _ = self.model.transcribe(
                    audio,
                    language=self.original_language,
                    initial_prompt=init_prompt,
                    word_timestamps=True
                )
                return list(segments)
            
            # Create a temporary WAV file to use with NeMo
            import tempfile
            import soundfile as sf
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_path = temp_audio.name
            
            # Save the audio to the temporary file
            sf.write(temp_path, audio, samplerate=16000, format='WAV', subtype='PCM_16')
            
            # Try transcribing with timestamps first
            try:
                result = self.model.transcribe([temp_path], timestamps=True)
                # Process the result directly without calling _process_nemo_result
                segments = self._create_segments_from_nemo_result(result[0])
                
            except Exception as timestamp_error:
                if "timestamps are not supported" in str(timestamp_error):
                    logger.info(f"Model {self.model_size} doesn't support timestamps. Using fallback approach with approximate timestamps.")
                    # Transcribe without timestamps
                    result = self.model.transcribe([temp_path])
                    # Generate synthetic timestamps
                    segments = self._generate_synthetic_timestamps(result[0])
                else:
                    # Different error, re-raise
                    raise
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error during Parakeet TDT transcription: {e}")
            # Return empty result on error
            return []

    def _create_segments_from_nemo_result(self, result):
        """Create segments from NeMo result - replaces _process_nemo_result."""
        from dataclasses import dataclass
        
        @dataclass
        class Segment:
            start: float
            end: float
            text: str
            words: list
            no_speech_prob: float = 0.0
            
        @dataclass
        class Word:
            start: float
            end: float
            word: str
            probability: float = 0.9
            
        segments = []
        
        # Check if results exist and have text
        if not hasattr(result, 'text') or not result.text:
            return segments
        
        # Extract word timestamps if available
        words = []
        if hasattr(result, 'timestamp') and 'word' in result.timestamp:
            for word_stamp in result.timestamp['word']:
                word = Word(
                    start=word_stamp['start'],
                    end=word_stamp['end'],
                    word=word_stamp['word'],
                    probability=0.9  # Default confidence
                )
                words.append(word)
        
        # If we have words with timestamps
        if words:
            # Create a segment with all the words
            segment = Segment(
                start=words[0].start,
                end=words[-1].end,
                text=result.text,
                words=words
            )
            segments.append(segment)
        else:
            # If no word timestamps, create a segment with estimated duration
            audio_duration = getattr(result, 'duration', 5.0)  # Default 5 seconds if unknown
            segment = Segment(
                start=0.0,
                end=audio_duration,
                text=result.text,
                words=[]  # No word-level timestamps
            )
            segments.append(segment)
        
        return segments

    def _generate_synthetic_timestamps(self, result):
        """Generate synthetic timestamps for models that don't support them."""
        from dataclasses import dataclass
        import re
        
        @dataclass
        class Segment:
            start: float
            end: float
            text: str
            words: list
            no_speech_prob: float = 0.0
            
        @dataclass
        class Word:
            start: float
            end: float
            word: str
            probability: float = 0.9
            
        segments = []
        
        if not hasattr(result, 'text') or not result.text:
            return segments
            
        # Split text into tokens/words
        text = result.text.strip()
        words = re.findall(r'\S+|\s+', text)
        
        # Filter out whitespace-only tokens
        words = [w for w in words if w.strip()]
        
        # Calculate approximate duration
        audio_duration = getattr(result, 'duration', len(words) * 0.3)  # Fallback to estimate if duration not available
        
        # Estimate word durations - average English word takes ~0.3 seconds to speak
        # We'll distribute words evenly across the audio duration
        avg_word_duration = audio_duration / max(1, len(words))
        
        # Create synthetic word timestamps
        synthetic_words = []
        current_time = 0.0
        
        for word_text in words:
            # Adjust duration based on word length (longer words take more time)
            word_duration = avg_word_duration * (0.5 + 0.5 * len(word_text) / 5)  # Adjust based on word length
            
            word = Word(
                start=current_time,
                end=current_time + word_duration,
                word=word_text,
                probability=0.9  # Default confidence
            )
            synthetic_words.append(word)
            current_time += word_duration
        
        # Create a single segment containing all words
        if synthetic_words:
            segment = Segment(
                start=0.0,
                end=audio_duration,
                text=text,
                words=synthetic_words
            )
            segments.append(segment)
        
        return segments

    def transcribe_batch(self, audio_batch: list) -> list:
        """Process a batch of audio samples."""
        results = []
        
        # Check if we're using the fallback model
        if hasattr(self, 'fallback') and self.fallback:
            # Process in small batches to avoid memory issues
            for audio in audio_batch:
                segments = self.transcribe(audio)
                results.append({
                    "segments": segments,
                    "tokens": self.ts_words(segments)
                })
            return results
        
        # Using NeMo model - process through temporary files
        import tempfile
        import soundfile as sf
        import os
        
        try:
            # Create temporary WAV files for batch processing
            temp_files = []
            for i, audio in enumerate(audio_batch):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_path = temp_audio.name
                    temp_files.append(temp_path)
                    sf.write(temp_path, audio, samplerate=16000, format='WAV', subtype='PCM_16')
            
            # Transcribe all files in one batch
            batch_results = self.model.transcribe(temp_files, timestamps=True)
            
            # Process each result and add to results list
            for i, result in enumerate(batch_results):
                segments = self._create_segments_from_nemo_result(result)
                results.append({
                    "segments": segments,
                    "tokens": self.ts_words(segments)
                })
                
                # Delete temporary file
                try:
                    os.unlink(temp_files[i])
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_files[i]}: {e}")
            
        except Exception as e:
            logger.error(f"Error in batch transcription: {e}")
            # Return empty results
            results = [{"segments": [], "tokens": []}] * len(audio_batch)
            
        return results

    def ts_words(self, segments) -> List[ASRToken]:
        """Extract word-level timestamps from processed output."""
        tokens = []
        for segment in segments:
            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word, probability=getattr(word, 'probability', 0.9))
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        """Return the end timestamps of all segments."""
        return [segment.end for segment in segments]

    def use_vad(self):
        """VAD is built into Parakeet TDT model."""
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        """Set task to translation if supported."""
        logger.warning("Translation not supported by Parakeet TDT model, ignoring.")
        self.transcribe_kargs["task"] = "translate"