try:
    from whisperlivekit.whisper_streaming_custom.whisper_online import backend_factory, warmup_asr
except ImportError:
    from .whisper_streaming_custom.whisper_online import backend_factory, warmup_asr
from argparse import Namespace, ArgumentParser
import asyncio
import logging
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)

# Create a thread pool for CPU-intensive operations - shared across all instances
_global_thread_pool = ThreadPoolExecutor(max_workers=4)

class BatchTranscriptionRequest:
    """Represents a request for batch transcription processing"""
    
    def __init__(self, client_id: str, audio_data: np.ndarray, request_time: float):
        self.client_id = client_id
        self.audio_data = audio_data  
        self.request_time = request_time
        self.processing_complete = asyncio.Event()
        self.result = None
        
    def set_result(self, result):
        """Set the result and mark the request as complete"""
        self.result = result
        self.processing_complete.set()

class BatchTranscriptionService:
    """Manages batched transcription requests for multiple clients using a single model"""
    
    def __init__(self, asr=None, tokenizer=None, batch_size=8, max_wait_time=0.1):
        """Initialize the batch processing service with shared model"""
        self.asr = asr
        self.tokenizer = tokenizer
        self.batch_size = batch_size  # Maximum batch size
        self.max_wait_time = max_wait_time  # Maximum time to wait before processing
        
        # Request management
        self.pending_requests = asyncio.Queue()
        self.client_buffers: Dict[str, List[np.ndarray]] = {}
        self.client_last_request: Dict[str, float] = {}
        self.current_batch: List[BatchTranscriptionRequest] = []
        
        # Performance metrics
        self.total_processed = 0
        self.batch_sizes = []
        self.processing_times = []
        
        # Batch processing task
        self.is_running = True
        self.processing_task = None
        
        # Start the background task if asr is available
        if self.asr:
            self.start()
    
    def start(self):
        """Start the batch processing background task"""
        if not self.processing_task:
            self.processing_task = asyncio.create_task(self._process_batches())
            logger.info("Batch transcription service started")
    
    async def stop(self):
        """Stop the batch processing background task"""
        if self.processing_task:
            self.is_running = False
            await self.pending_requests.put(None)  # Signal termination
            await self.processing_task
            self.processing_task = None
            logger.info("Batch transcription service stopped")
    
    async def submit(self, client_id: str, audio_data: np.ndarray) -> Any:
        """Submit audio data for batch processing and wait for results"""
        if not self.asr:
            raise ValueError("No ASR model available for transcription")
            
        # Create a request
        request = BatchTranscriptionRequest(
            client_id=client_id, 
            audio_data=audio_data,
            request_time=time.time()
        )
        
        # Submit to queue
        await self.pending_requests.put(request)
        
        # Wait for processing to complete
        await request.processing_complete.wait()
        return request.result
    
    async def _process_batches(self):
        """Background task that processes batches of requests"""
        while self.is_running:
            try:
                # Initialize current batch
                self.current_batch = []
                batch_start_time = time.time()
                
                # Get at least one request to process
                request = await self.pending_requests.get()
                if request is None:  # Stop signal
                    break
                    
                self.current_batch.append(request)
                
                # Try to batch more requests if available
                max_wait_end_time = time.time() + self.max_wait_time
                while (len(self.current_batch) < self.batch_size and 
                       time.time() < max_wait_end_time):
                    try:
                        # Non-blocking get with timeout
                        next_request = await asyncio.wait_for(
                            self.pending_requests.get(), 
                            timeout=max(0, max_wait_end_time - time.time())
                        )
                        if next_request is None:  # Stop signal
                            break
                        self.current_batch.append(next_request)
                    except asyncio.TimeoutError:
                        break
                
                # Exit if we received stop signal
                if None in self.current_batch:
                    self.current_batch.remove(None)
                    break
                
                if not self.current_batch:
                    continue
                
                # Process the batch
                batch_size = len(self.current_batch)
                logger.debug(f"Processing batch of {batch_size} requests")
                self.batch_sizes.append(batch_size)
                
                process_start = time.time()
                
                # Run transcription in thread pool to avoid blocking the event loop
                results = await self._process_transcription_batch()
                
                # Record processing time
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                self.total_processed += batch_size
                
                # Return results to clients
                for req, result in zip(self.current_batch, results):
                    req.set_result(result)
                    
                # Log performance metrics periodically
                if self.total_processed % 50 == 0:
                    avg_batch = sum(self.batch_sizes[-50:]) / len(self.batch_sizes[-50:])
                    avg_time = sum(self.processing_times[-50:]) / len(self.processing_times[-50:])
                    logger.info(f"Transcription stats: avg batch size={avg_batch:.2f}, " 
                               f"avg process time={avg_time:.4f}s, total processed={self.total_processed}")
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # Return error to clients waiting in current batch
                for req in self.current_batch:
                    req.set_result({"error": str(e)})
                    
    async def _process_transcription_batch(self) -> List[Any]:
        """Process a batch of transcription requests using the model"""
        if not self.current_batch:
            return []
            
        # For single request, process directly
        if len(self.current_batch) == 1:
            req = self.current_batch[0]
            return [await self._transcribe_single(req.audio_data)]
            
        # For multiple requests, batch process if backend supports it
        loop = asyncio.get_event_loop()
        
        # Process using thread pool to avoid blocking
        try:
            # Create a list of audio data from all requests
            audio_batch = [req.audio_data for req in self.current_batch]
            
            # Process using FasterWhisperASR's batch processing if available
            if hasattr(self.asr, "transcribe_batch"):
                logger.debug(f"Using native batch transcription for {len(audio_batch)} samples")
                batch_results = await loop.run_in_executor(
                    _global_thread_pool, 
                    lambda: self.asr.transcribe_batch(audio_batch)
                )
                return batch_results
            else:
                # Fall back to sequential processing if batching not supported
                logger.debug(f"Using sequential processing for {len(audio_batch)} samples")
                results = []
                for audio in audio_batch:
                    result = await self._transcribe_single(audio)
                    results.append(result)
                return results
                
        except Exception as e:
            logger.error(f"Error in batch transcription: {e}")
            return [{"error": str(e)}] * len(self.current_batch)
    
    async def _transcribe_single(self, audio_data: np.ndarray) -> Any:
        """Process a single transcription request"""
        loop = asyncio.get_event_loop()
        try:
            # Process in thread pool
            segments = await loop.run_in_executor(
                _global_thread_pool,
                lambda: self.asr.transcribe(audio_data)
            )
            
            # Extract word timestamps
            tokens = await loop.run_in_executor(
                _global_thread_pool,
                lambda: self.asr.ts_words(segments)
            )
            
            return {"segments": segments, "tokens": tokens}
            
        except Exception as e:
            logger.error(f"Error in single transcription: {e}")
            return {"error": str(e)}

def parse_args():
    parser = ArgumentParser(description="Whisper FastAPI Online Server")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="The host address to bind the server to.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port number to bind the server to."
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=1,
        help="Number of ASR model instances to load.",
    )
    parser.add_argument(
        "--warmup-file",
        type=str,
        default=None,
        dest="warmup_file",
        help="""
        The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.
        If not set, uses https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav.
        If False, no warmup is performed.
        """,
    )

    parser.add_argument(
        "--confidence-validation",
        action="store_true",
        help="Accelerates validation of tokens using confidence scores. Transcription will be faster but punctuation might be less accurate.",
    )

    parser.add_argument(
        "--diarization",
        action="store_true",
        default=False,
        help="Enable speaker diarization.",
    )

    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Disable transcription to only see live diarization results.",
    )
    
    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=0.5,
        help="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        help="Name size of the Whisper model to use (default: tiny). Suggested values: tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo. The model is automatically downloaded from the model hub if not present in model cache dir.",
    )
    
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Overriding the default model cache dir where models downloaded from the hub are saved",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    parser.add_argument(
        "--lan",
        "--language",
        type=str,
        default="auto",
        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Transcribe or translate.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="faster-whisper",
        choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api", "parakeet-tdt"],
        help="Load only this backend for Whisper processing.",
    )
    parser.add_argument(
        "--vac",
        action="store_true",
        default=False,
        help="Use VAC = voice activity controller. Recommended. Requires torch.",
    )
    parser.add_argument(
        "--vac-chunk-size", type=float, default=0.04, help="VAC sample size in seconds."
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD (voice activity detection).",
    )
    
    parser.add_argument(
        "--buffer_trimming",
        type=str,
        default="segment",
        choices=["sentence", "segment"],
        help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.',
    )
    parser.add_argument(
        "--buffer_trimming_sec",
        type=float,
        default=15,
        help="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level",
        default="DEBUG",
    )
    parser.add_argument("--ssl-certfile", type=str, help="Path to the SSL certificate file.", default=None)
    parser.add_argument("--ssl-keyfile", type=str, help="Path to the SSL private key file.", default=None)


    args = parser.parse_args()
    
    args.transcription = not args.no_transcription
    args.vad = not args.no_vad    
    delattr(args, 'no_transcription')
    delattr(args, 'no_vad')
    
    return args

class WhisperLiveKit:
    _instance = None
    _initialized = False
    _service_index_lock = threading.Lock() # Lock for round-robin index
    _current_service_index = 0 # For round-robin assignment of batch services

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, **kwargs):
        if WhisperLiveKit._initialized and self._instance is not None:
            return
            
        default_args = vars(parse_args())
        
        merged_args = {**default_args, **kwargs}
        
        self.args = Namespace(**merged_args)
        
        self.asr_instances = []
        self.tokenizer_instances = []
        self.batch_services = []
        
        self.diarization = None

        if self.args.transcription:
            for i in range(self.args.num_models):
                logger.info(f"Initializing ASR model instance {i+1} of {self.args.num_models}...")
                asr_instance, tokenizer_instance = backend_factory(self.args)
                warmup_asr(asr_instance, self.args.warmup_file)
                self.asr_instances.append(asr_instance)
                self.tokenizer_instances.append(tokenizer_instance)

                optimal_batch_size = 16
                if "large" in self.args.model or "distil" in self.args.model:
                    if hasattr(asr_instance, "device") and asr_instance.device == "cuda":
                        try:
                            import torch
                            if torch.cuda.is_available():
                                gpu_mem = torch.cuda.get_device_properties(i % torch.cuda.device_count()).total_memory / (1024**3)
                                if gpu_mem > 40: optimal_batch_size = 24
                                if gpu_mem > 80: optimal_batch_size = 32
                                logger.info(f"Instance {i+1} on GPU {i % torch.cuda.device_count()} with {gpu_mem:.1f}GB RAM, batch size {optimal_batch_size}")
                        except Exception as e:
                            logger.warning(f"Could not detect GPU memory for instance {i+1}: {e}, using default batch size {optimal_batch_size}")
                
                batch_service_instance = BatchTranscriptionService(
                    asr=asr_instance,
                    tokenizer=tokenizer_instance,
                    batch_size=optimal_batch_size,
                    max_wait_time=0.1
                )
                batch_service_instance.start()
                self.batch_services.append(batch_service_instance)
            
            if not self.batch_services:
                logger.warning("No batch services were initialized. Transcription might not work.")
            else:
                logger.info(f"Initialized {len(self.batch_services)} ASR and batch service instances.")


        if self.args.diarization:
            from whisperlivekit.diarization.diarization_online import DiartDiarization
            self.diarization = DiartDiarization()
            
        WhisperLiveKit._initialized = True

    def get_next_batch_service(self) -> Optional[BatchTranscriptionService]:
        """Returns the next BatchTranscriptionService in a round-robin fashion."""
        if not self.batch_services:
            logger.error("No batch services available to assign.")
            return None
        
        with WhisperLiveKit._service_index_lock:
            service = self.batch_services[WhisperLiveKit._current_service_index]
            WhisperLiveKit._current_service_index = (WhisperLiveKit._current_service_index + 1) % len(self.batch_services)
            logger.info(f"Assigning batch service instance {WhisperLiveKit._current_service_index} (0-indexed) to new client.")
        return service

    def get_asr_and_tokenizer(self, batch_service_instance: BatchTranscriptionService) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Given a BatchTranscriptionService instance, find the corresponding ASR and Tokenizer.
        This assumes that the order in self.asr_instances, self.tokenizer_instances, 
        and self.batch_services is maintained.
        """
        if not batch_service_instance:
            return None, None
        try:
            idx = self.batch_services.index(batch_service_instance)
            asr = self.asr_instances[idx] if idx < len(self.asr_instances) else None
            tokenizer = self.tokenizer_instances[idx] if idx < len(self.tokenizer_instances) else None
            return asr, tokenizer
        except ValueError:
            logger.error("Provided batch_service_instance not found in the list.")
            return None, None
        except IndexError:
            logger.error("Mismatch in lengths of asr/tokenizer/batch_service lists.")
            return None, None


    def web_interface(self):
        import pkg_resources
        html_path = pkg_resources.resource_filename('whisperlivekit', 'web/live_transcription.html')
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        return html