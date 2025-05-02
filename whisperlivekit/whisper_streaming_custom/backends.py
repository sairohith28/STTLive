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
                compute_type = "float16"
                torch.cuda.empty_cache()  # Clear CUDA memory before loading model
                
                # Check available GPU memory and adjust settings accordingly
                try:
                    free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
                    logger.info(f"Available GPU memory: {free_memory:.2f} GB")
                    
                    # For devices with limited memory, use int8 quantization
                    if free_memory < 2.0 and "large" in str(model_size_or_path).lower():
                        compute_type = "int8"
                        logger.info(f"Using {compute_type} precision for {model_size_or_path} due to limited GPU memory")
                    # For devices with 8+ GB memory, use float16 for better accuracy
                    elif free_memory > 8.0:
                        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
                        logger.info("Enabling CUDNN benchmark mode for faster processing")
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {e}, using default settings")
            
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                compute_type = "float16"
                
        # Optimize for model size and task
        beam_size = 3  # Default
        if "tiny" in str(model_size_or_path).lower() or "base" in str(model_size_or_path).lower():
            beam_size = 2  # Faster for small models
        elif "large" in str(model_size_or_path).lower():
            beam_size = 4  # Better accuracy for large models
        
        # Advanced configuration with adaptive parameters
        best_of = None  # Default from faster-whisper
        if device == "cuda":
            # Use more threads on GPU for parallel processing
            cpu_threads = 2
            # Use more workers for loading when GPU is available
            num_workers = 3 if "large" not in str(model_size_or_path).lower() else 2
        else:
            # Use more CPU threads when running on CPU
            cpu_threads = min(8, os.cpu_count() or 4)
            num_workers = 1  # Fewer workers on CPU to avoid contention
            
        # Cache model parameters for future reference
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
                              
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

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            # Skip segments with high no_speech probability
            if segment.no_speech_prob > 0.85:  # More aggressive filtering
                continue
                
            # Process words with confidence filtering for better accuracy
            for word in segment.words:
                # Skip low-confidence words
                if hasattr(word, 'probability') and word.probability < 0.4:
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