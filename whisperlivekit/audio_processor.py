import asyncio
import numpy as np
import ffmpeg
from time import time, sleep
import math
import logging
import traceback
import uuid
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper_streaming_custom.whisper_online import online_factory
from whisperlivekit.core import WhisperLiveKit

# Set up logging once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a thread pool for CPU-intensive operations
thread_pool = ThreadPoolExecutor(max_workers=4)

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

class AudioProcessor:
    """
    Processes audio streams for transcription and diarization.
    Handles audio processing, state management, and result formatting.
    """
    
    # Added a class variable to keep track of clients per service instance
    _clients_per_service = {}
    _clients_per_service_lock = asyncio.Lock() # Lock for updating client counts
    MAX_CLIENTS_PER_INSTANCE = 2 # Max clients per ASR/BatchService instance

    def __init__(self):
        """Initialize the audio processor with configuration, models, and state."""
        
        self.kit_instance = WhisperLiveKit() # Get the singleton WhisperLiveKit instance
        
        # Generate a unique client ID for this processor instance
        self.client_id = str(uuid.uuid4())
        
        # Audio processing settings
        self.args = self.kit_instance.args
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 3  # Reduced from 5 to 3 seconds for faster processing
        self.last_ffmpeg_activity = time()
        self.ffmpeg_health_check_interval = 3
        self.ffmpeg_max_idle_time = 6
 
        # State management
        self.tokens = []
        self.buffer_transcription = ""
        self.buffer_diarization = ""
        self.full_transcription = ""
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep = " "  # Default separator
        self.last_response_content = ""
        
        # Models and processing
        # These will be set by assign_service()
        self.asr = None
        self.tokenizer = None
        self.assigned_batch_service = None 
        self.online = None # Will be initialized in assign_service

        # Diarization can be taken from the kit_instance directly if it's global
        self.diarization = self.kit_instance.diarization

        self.ffmpeg_process = self.start_ffmpeg_decoder()
        self.transcription_queue = asyncio.Queue(maxsize=10) if self.args.transcription else None
        self.diarization_queue = asyncio.Queue(maxsize=10) if self.args.diarization else None
        self.pcm_buffer = bytearray()
        
        # Performance optimizations
        self.use_parallel = True  # Enable parallel processing
        self.max_update_freq = 0.2  # Maximum update frequency in seconds
        self.last_update = 0

        logger.info(f"AudioProcessor initialized with client_id: {self.client_id}. Waiting for service assignment.")

    async def assign_service(self):
        """Assigns a suitable BatchTranscriptionService to this AudioProcessor."""
        assigned = False
        while not assigned:
            async with AudioProcessor._clients_per_service_lock:
                for service in self.kit_instance.batch_services:
                    current_clients = AudioProcessor._clients_per_service.get(service, 0)
                    if current_clients < AudioProcessor.MAX_CLIENTS_PER_INSTANCE:
                        # Tentatively assign the service to get its ASR/tokenizer
                        # self.assigned_batch_service = service # Assign later only on full success
                        _asr, _tokenizer = self.kit_instance.get_asr_and_tokenizer(service)

                        transcription_requirements_met = False
                        if self.args.transcription:
                            if _asr:  # ASR is always needed for transcription
                                if self.args.backend == "faster-whisper":
                                    transcription_requirements_met = True  # Assume faster-whisper ASR handles tokenization or online_factory adapts
                                    if not _tokenizer:
                                        logger.info(f"Client {self.client_id}: For faster-whisper backend, tokenizer from backend_factory is None. This is assumed to be handled by online_factory.")
                                elif _tokenizer:  # For other backends, tokenizer must be present
                                    transcription_requirements_met = True
                        else:  # Transcription is not enabled by args
                            transcription_requirements_met = True # No ASR/tokenizer needed for transcription part if transcription is off

                        if not transcription_requirements_met:
                            logger.error(f"Client {self.client_id}: Failed to meet transcription requirements for service with backend {self.args.backend}. ASR: {'OK' if _asr else 'Missing'}, Tokenizer: {'OK' if _tokenizer else 'Missing'}. Trying next service or retrying.")
                            # self.assigned_batch_service = None # Ensure it's cleared if we had tentatively set it
                            continue # Try next service in the for loop
                        
                        # If we reach here, requirements are met for this service
                        self.assigned_batch_service = service # Confirm assignment
                        AudioProcessor._clients_per_service[service] = current_clients + 1
                        self.asr = _asr
                        self.tokenizer = _tokenizer # This might be None for faster-whisper
                        assigned = True
                        logger.info(f"Client {self.client_id} assigned to BatchService. Service now has {AudioProcessor._clients_per_service[service]} clients. Backend: {self.args.backend}, ASR: OK, Tokenizer: {'OK' if self.tokenizer else 'None/Handled by backend'}.")
                        break # Exit the for loop (found a service)
                # End of for loop (iterating through services)
            # End of async with lock

            if not assigned:
                logger.warning(f"Client {self.client_id}: All ASR instances are currently at max capacity ({AudioProcessor.MAX_CLIENTS_PER_INSTANCE} clients) or failed to meet ASR/Tokenizer requirements for any available service. Waiting...")
                await asyncio.sleep(1) # Wait and retry

        # Initialize the online ASR engine if transcription is enabled and ASR is available
        if self.args.transcription and self.asr:
            # self.tokenizer might be None here if backend is faster-whisper and it returned None
            self.online = online_factory(self.args, self.asr, self.tokenizer)
            if self.online:
                logger.info(f"Client {self.client_id}: Online ASR engine initialized successfully for backend {self.args.backend} with ASR and Tokenizer ({'Present' if self.tokenizer else 'None/Handled by backend'}).")
            else:
                logger.error(f"Client {self.client_id}: online_factory failed to initialize for backend {self.args.backend}. This is critical for transcription.")
        elif self.args.transcription and not self.asr:
            logger.error(f"Client {self.client_id}: Online ASR engine NOT initialized because self.asr is missing, despite transcription being enabled.")
        else:
            logger.info(f"Client {self.client_id}: Transcription is disabled by arguments. Online ASR engine not initialized.")

    def release_service(self):
        """Releases the BatchTranscriptionService when client disconnects."""
        async def _release_service_async():
            if self.assigned_batch_service:
                async with AudioProcessor._clients_per_service_lock:
                    if self.assigned_batch_service in AudioProcessor._clients_per_service:
                        AudioProcessor._clients_per_service[self.assigned_batch_service] -= 1
                        logger.info(f"Client {self.client_id} released BatchService. Service now has {AudioProcessor._clients_per_service[self.assigned_batch_service]} clients.")
                        if AudioProcessor._clients_per_service[self.assigned_batch_service] == 0:
                            del AudioProcessor._clients_per_service[self.assigned_batch_service]
                    else:
                        logger.warning(f"Client {self.client_id}: Tried to release a service not in the tracking dict.")
                self.assigned_batch_service = None
                self.asr = None
                self.tokenizer = None
        # Run this in a new task to avoid blocking if called from a sync context or during cleanup
        asyncio.create_task(_release_service_async())

    def convert_pcm_to_float(self, pcm_buffer):
        """Convert PCM buffer in s16le format to normalized NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    def start_ffmpeg_decoder(self):
        """Start FFmpeg process for WebM to PCM conversion with optimized settings."""
        return (ffmpeg.input("pipe:0", format="webm")
                .output("pipe:1", format="s16le", acodec="pcm_s16le", 
                        ac=self.channels, ar=str(self.sample_rate),
                        # Add thread optimization for ffmpeg
                        **{'threads': '4'})
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True))

    async def restart_ffmpeg(self):
        """Restart the FFmpeg process after failure with improved error handling."""
        logger.warning("Restarting FFmpeg process...")
        
        if self.ffmpeg_process:
            try:
                # Check if process is still running
                if self.ffmpeg_process.poll() is None:
                    logger.info("Terminating existing FFmpeg process")
                    self.ffmpeg_process.stdin.close()
                    self.ffmpeg_process.terminate()
                    
                    # Wait for termination with timeout
                    try:
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait),
                            timeout=3.0  # Reduced from 5.0 to 3.0 seconds
                        )
                    except asyncio.TimeoutError:
                        logger.warning("FFmpeg process did not terminate, killing forcefully")
                        self.ffmpeg_process.kill()
                        await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait)
            except Exception as e:
                logger.error(f"Error during FFmpeg process termination: {e}")
                logger.error(traceback.format_exc())
        
        # Start new process
        try:
            logger.info("Starting new FFmpeg process")
            self.ffmpeg_process = self.start_ffmpeg_decoder()
            self.pcm_buffer = bytearray()
            self.last_ffmpeg_activity = time()
            logger.info("FFmpeg process restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart FFmpeg process: {e}")
            logger.error(traceback.format_exc())
            # Try again after shorter delay
            await asyncio.sleep(2)
            try:
                self.ffmpeg_process = self.start_ffmpeg_decoder()
                self.pcm_buffer = bytearray()
                self.last_ffmpeg_activity = time()
                logger.info("FFmpeg process restarted successfully on second attempt")
            except Exception as e2:
                logger.critical(f"Failed to restart FFmpeg process on second attempt: {e2}")
                logger.critical(traceback.format_exc())

    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        """Thread-safe update of transcription with new data."""
        # Rate limit updates to minimize contention
        current_time = time()
        if current_time - self.last_update < self.max_update_freq and not new_tokens:
            return
            
        self.last_update = current_time
        
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep
            
    async def update_diarization(self, end_attributed_speaker, buffer_diarization=""):
        """Thread-safe update of diarization with new data."""
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization
            
    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.beg_loop
            self.tokens.append(ASRToken(
                start=current_time, end=current_time + 1,
                text=".", speaker=-1, is_dummy=True
            ))
            
    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()
            
            # Calculate remaining times
            remaining_transcription = 0
            if self.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 2))
                
            remaining_diarization = 0
            if self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 2))
                
            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "remaining_time_transcription": remaining_transcription,
                "remaining_time_diarization": remaining_diarization
            }
            
    async def reset(self):
        """Reset all state variables to initial values."""
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = self.buffer_diarization = ""
            self.end_buffer = self.end_attributed_speaker = 0
            self.full_transcription = self.last_response_content = ""
            self.beg_loop = time()

    async def ffmpeg_stdout_reader(self):
        """Read audio data from FFmpeg stdout and process it with optimized buffer handling."""
        loop = asyncio.get_event_loop()
        beg = time()
        
        while True:
            try:
                current_time = time()
                elapsed_time = math.floor((current_time - beg) * 10) / 10
                # Optimize buffer size calculation for smoother processing
                buffer_size = min(max(int(16000 * elapsed_time), 2048), 16384)  # Cap at 16KB for more frequent updates
                beg = current_time

                # Detect idle state quickly
                if current_time - self.last_ffmpeg_activity > self.ffmpeg_max_idle_time:
                    logger.warning(f"FFmpeg process idle for {current_time - self.last_ffmpeg_activity:.2f}s. Restarting...")
                    await self.restart_ffmpeg()
                    beg = time()
                    self.last_ffmpeg_activity = time()
                    continue

                chunk = await loop.run_in_executor(None, self.ffmpeg_process.stdout.read, buffer_size)
                if chunk:
                    self.last_ffmpeg_activity = time()
                        
                if not chunk:
                    logger.info("FFmpeg stdout closed.")
                    break
                    
                self.pcm_buffer.extend(chunk)
                
                # Optimized PCM buffer processing with parallel queuing
                if len(self.pcm_buffer) >= self.bytes_per_sec:
                    if len(self.pcm_buffer) > self.max_bytes_per_sec:
                        logger.warning(
                            f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                            f"Consider using a smaller model."
                        )

                    # Process audio chunk
                    pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:self.max_bytes_per_sec])
                    self.pcm_buffer = self.pcm_buffer[self.max_bytes_per_sec:]
                    
                    # Send to transcription and diarization in parallel
                    tasks = []
                    
                    if self.args.transcription and self.transcription_queue:
                        # Use try/except with nowait to avoid blocking
                        try:
                            self.transcription_queue.put_nowait(pcm_array.copy())
                        except asyncio.QueueFull:
                            logger.warning("Transcription queue full, skipping chunk")
                    
                    if self.args.diarization and self.diarization_queue:
                        try:
                            self.diarization_queue.put_nowait(pcm_array.copy())
                        except asyncio.QueueFull:
                            logger.warning("Diarization queue full, skipping chunk")
                    
                    # Wait for all queue puts to complete
                    if tasks:
                        await asyncio.gather(*tasks)
                    
                    # Small delay to yield control to other tasks
                    if not self.args.transcription and not self.args.diarization:
                        await asyncio.sleep(0.05)  # Reduced from 0.1 to 0.05
                    
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                break

# In whisperlivekit/audio_processor.py

# ...existing code...

    async def transcription_processor(self):
        """Process audio chunks for transcription with optimized handling."""
        if not self.args.transcription or not self.online: # Removed check for self.assigned_batch_service here as we'll primarily use self.online
            logger.warning(f"Client {self.client_id}: Transcription processor cannot start. Transcription disabled or ASR (self.online) not initialized.")
            return

        self.full_transcription = ""
        # Ensure self.online and self.online.asr are valid before accessing sep
        if hasattr(self.online, 'asr') and self.online.asr is not None:
            self.sep = self.online.asr.sep
        else:
            logger.error(f"Client {self.client_id}: self.online.asr is not available. Cannot determine separator. Defaulting to space.")
            self.sep = " " # Default separator if ASR object isn't where expected
        
        while True:
            try:
                pcm_array = await self.transcription_queue.get()
                
                # Primarily use the self.online object for streaming transcription
                # This logic is similar to _process_transcription_locally
                def process_streaming_transcription_in_thread():
                    try:
                        # Ensure self.online is still valid
                        if not self.online:
                            logger.warning(f"Client {self.client_id}: self.online became None during processing.")
                            return []
                        self.online.insert_audio_chunk(pcm_array)
                        return self.online.process_iter()
                    except Exception as e:
                        logger.error(f"Client {self.client_id}: Error in streaming transcription thread (process_iter): {e}")
                        logger.error(traceback.format_exc())
                        return []

                new_tokens = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, process_streaming_transcription_in_thread
                )

                if new_tokens:
                    self.full_transcription += self.sep.join([t.text for t in new_tokens])
                    logger.debug(f"Client {self.client_id}: Received {len(new_tokens)} new tokens from self.online.process_iter().")
                # No "else" here for logging 0 tokens, as process_iter might yield empty list between segments.

                # Get buffer information from self.online
                _buffer = self.online.get_buffer() if self.online else None
                buffer_text = _buffer.text if _buffer else ""
                
                # Determine end_buffer: use new token's end, else buffer's end, else keep previous self.end_buffer
                if new_tokens:
                    current_segment_end_time = new_tokens[-1].end
                elif _buffer and _buffer.end:
                    current_segment_end_time = _buffer.end
                else:
                    current_segment_end_time = self.end_buffer 

                # Avoid duplicating buffer text if it's already at the end of full_transcription
                if buffer_text and self.full_transcription.endswith(buffer_text):
                    buffer_text = ""
            
                await self.update_transcription(
                    new_tokens if new_tokens is not None else [], # Ensure new_tokens is a list
                    buffer_text, 
                    current_segment_end_time, 
                    self.full_transcription, 
                    self.sep
                )
                
            except Exception as e:
                logger.warning(f"Client {self.client_id}: Exception in transcription_processor main loop: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
            finally:
                if hasattr(self.transcription_queue, 'task_done'):
                    self.transcription_queue.task_done()

    async def _process_transcription_locally(self, pcm_array):
        """Process transcription locally without batch service"""
        # Perform CPU-intensive transcription in thread pool
        def process_transcription():
            try:
                # Process audio and extract tokens - this is CPU intensive
                self.online.insert_audio_chunk(pcm_array)
                return self.online.process_iter()
            except Exception as e:
                logger.error(f"Error in transcription thread: {e}")
                return []
        
        # Run in thread pool to avoid blocking the event loop
        new_tokens = await asyncio.get_event_loop().run_in_executor(
            thread_pool, process_transcription
        )
        
        if new_tokens:
            self.full_transcription += self.sep.join([t.text for t in new_tokens])
            
        # Get buffer information
        _buffer = self.online.get_buffer()
        buffer = _buffer.text
        end_buffer = _buffer.end if _buffer.end else (
            new_tokens[-1].end if new_tokens else 0
        )
        
        # Avoid duplicating content
        if buffer in self.full_transcription:
            buffer = ""
            
        await self.update_transcription(
            new_tokens, buffer, end_buffer, self.full_transcription, self.sep
        )

    async def diarization_processor(self, diarization_obj):
        """Process audio chunks for speaker diarization with thread pool optimization."""
        buffer_diarization = ""
        
        while True:
            try:
                pcm_array = await self.diarization_queue.get()
                
                # Use thread pool to avoid blocking the event loop with CPU-intensive diarization
                async def process_diarization():
                    # Process diarization
                    await diarization_obj.diarize(pcm_array)
                    
                    # Get current state and update speakers
                    state = await self.get_current_state()
                    return diarization_obj.assign_speakers_to_tokens(
                        state["end_attributed_speaker"], state["tokens"]
                    )
                
                new_end = await process_diarization()
                await self.update_diarization(new_end, buffer_diarization)
                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
            finally:
                self.diarization_queue.task_done()

    async def results_formatter(self):
        """Format processing results for output with optimized update frequency."""
        last_yield_time = 0
        min_yield_interval = 0.1  # Minimum time between yields to client
        max_tokens_processed = 500  # Maximum number of tokens to process in one cycle
        last_state_hash = ""  # Track state changes to avoid redundant formatting
        
        while True:
            try:
                current_time = time()
                
                # Rate-limit updates to reduce CPU usage and network traffic
                if current_time - last_yield_time < min_yield_interval:
                    await asyncio.sleep(0.05)
                    continue
                
                # Get current state
                state = await self.get_current_state()
                
                # Create a simple hash to check if state has meaningfully changed
                simple_hash = f"{len(state['tokens'])}-{state['end_buffer']}-{state['end_attributed_speaker']}-{len(state['buffer_transcription'])}-{len(state['buffer_diarization'])}"
                
                # Skip processing if state hasn't changed significantly
                if simple_hash == last_state_hash:
                    await asyncio.sleep(0.05)
                    continue
                
                last_state_hash = simple_hash
                
                tokens = state["tokens"]
                buffer_transcription = state["buffer_transcription"]
                buffer_diarization = state["buffer_diarization"]
                end_attributed_speaker = state["end_attributed_speaker"]
                sep = state["sep"]
                
                # Add dummy tokens if needed
                if (not tokens or tokens[-1].is_dummy) and not self.args.transcription and self.args.diarization:
                    await self.add_dummy_token()
                    state = await self.get_current_state()
                    tokens = state["tokens"]
                
                # Limit tokens processing to prevent lag in very long sessions
                if len(tokens) > max_tokens_processed:
                    # Only process the most recent tokens
                    recent_tokens = tokens[-max_tokens_processed:]
                    logger.info(f"Limiting token processing: {len(tokens)} tokens reduced to {len(recent_tokens)}")
                    # Keep the first token to maintain speaker information
                    if tokens and len(tokens) > max_tokens_processed:
                        processed_tokens = [tokens[0]] + recent_tokens
                    else:
                        processed_tokens = recent_tokens
                else:
                    processed_tokens = tokens
                
                # Format output with optimized processing
                previous_speaker = -1
                lines = []
                last_end_diarized = 0
                undiarized_text = []
                
                # Process each token with vectorized operations where possible
                for token in processed_tokens:
                    speaker = token.speaker
                    
                    # Handle diarization
                    if self.args.diarization:
                        if (speaker in [-1, 0]) and token.end >= end_attributed_speaker:
                            undiarized_text.append(token.text)
                            continue
                        elif (speaker in [-1, 0]) and token.end < end_attributed_speaker:
                            speaker = previous_speaker if previous_speaker != -1 else 1
                        if speaker not in [-1, 0]:
                            last_end_diarized = max(token.end, last_end_diarized)

                    # Use more efficient string formatting
                    if speaker != previous_speaker or not lines:
                        lines.append({
                            "speaker": speaker,
                            "text": token.text,
                            "beg": format_time(token.start),
                            "end": format_time(token.end),
                            "diff": round(token.end - last_end_diarized, 2)
                        })
                        previous_speaker = speaker
                    elif token.text:  # Only append if text isn't empty
                        # More efficient string concatenation
                        if not lines[-1]["text"]:
                            lines[-1]["text"] = token.text
                        else:
                            lines[-1]["text"] += sep + token.text
                        lines[-1]["end"] = format_time(token.end)
                        lines[-1]["diff"] = round(token.end - last_end_diarized, 2)
                
                # Handle undiarized text with optimized string operations
                if undiarized_text:
                    # Use join instead of repeated concatenation
                    combined = sep.join(undiarized_text)
                    if buffer_transcription:
                        combined += sep
                    await self.update_diarization(end_attributed_speaker, combined)
                    buffer_diarization = combined
                
                # Create response object with reduced allocations
                if not lines:
                    lines = [{
                        "speaker": 1,
                        "text": "",
                        "beg": format_time(0),
                        "end": format_time(tokens[-1].end if tokens else 0),
                        "diff": 0
                    }]
                
                response = {
                    "lines": lines, 
                    "buffer_transcription": buffer_transcription,
                    "buffer_diarization": buffer_diarization,
                    "remaining_time_transcription": state["remaining_time_transcription"],
                    "remaining_time_diarization": state["remaining_time_diarization"]
                }
                
                # Compute response content hash for change detection
                # Use a more efficient hash calculation
                lines_hash = ''.join([f"{line['speaker']}-{line['text'][:10]}" for line in lines[:3]])
                buffers_hash = f"{buffer_transcription[:20]}-{buffer_diarization[:20]}"
                response_hash = f"{lines_hash}|{buffers_hash}"
                
                # Only yield if content has meaningfully changed
                if response_hash != getattr(self, '_last_response_hash', '') and (lines or buffer_transcription or buffer_diarization):
                    yield response
                    self._last_response_hash = response_hash
                    last_yield_time = current_time
                
                # Dynamic sleep time based on system activity and lag
                # Use shorter sleep for active transcription, longer when idle
                is_active = current_time - last_yield_time < 2.0
                sleep_time = 0.05 if is_active else 0.2
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                # Add exponential backoff on error
                await asyncio.sleep(min(0.5 * (1 + getattr(self, '_error_count', 0)), 2.0))
                self._error_count = getattr(self, '_error_count', 0) + 1
        
    async def create_tasks(self):
        """Create and start processing tasks with improved task management."""
        
        # Ensure service is assigned before creating tasks that depend on ASR/tokenizer
        await self.assign_service()
            
        tasks = []    
        if self.args.transcription and self.online:
            tasks.append(asyncio.create_task(self.transcription_processor()))
            
        if self.args.diarization and self.diarization:
            tasks.append(asyncio.create_task(self.diarization_processor(self.diarization)))
        
        tasks.append(asyncio.create_task(self.ffmpeg_stdout_reader()))
        
        # Monitor overall system health with optimized checks
        async def watchdog():
            while True:
                try:
                    await asyncio.sleep(5)  # Check every 5 seconds (reduced from 10)
                    
                    current_time = time()
                    # Check for stalled tasks
                    for i, task in enumerate(tasks):
                        if task.done():
                            exc = task.exception() if task.done() else None
                            task_name = task.get_name() if hasattr(task, 'get_name') else f"Task {i}"
                            logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                            
                            # Restart critical tasks that died unexpectedly
                            if "ffmpeg_stdout_reader" in str(task):
                                logger.warning("Restarting FFmpeg reader task")
                                tasks[i] = asyncio.create_task(self.ffmpeg_stdout_reader())
                            elif "transcription_processor" in str(task) and self.args.transcription and self.online:
                                logger.warning("Restarting transcription processor task")
                                tasks[i] = asyncio.create_task(self.transcription_processor())
                    
                    # Check for FFmpeg process health with shorter thresholds
                    ffmpeg_idle_time = current_time - self.last_ffmpeg_activity
                    if ffmpeg_idle_time > 10:  # 10 seconds instead of 15
                        logger.warning(f"FFmpeg idle for {ffmpeg_idle_time:.2f}s - may need attention")
                        
                        # Force restart after 20 seconds of inactivity
                        if ffmpeg_idle_time > 20:  # 20 seconds instead of 30
                            logger.error("FFmpeg idle for too long, forcing restart")
                            await self.restart_ffmpeg()
                            
                except Exception as e:
                    logger.error(f"Error in watchdog task: {e}")

        tasks.append(asyncio.create_task(watchdog()))
        self.tasks = tasks
        
        return self.results_formatter()
        
    async def cleanup(self):
        """Clean up resources when processing is complete."""
        self.release_service() # Release the service slot
        for task in getattr(self, 'tasks', []):
            task.cancel()
            
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
            
        if self.args.diarization and hasattr(self, 'diarization'):
            self.diarization.close()

    async def process_audio(self, message):
        """Process incoming audio data with optimized retry logic."""
        retry_count = 0
        max_retries = 2  # Reduced from 3 to 2 for faster failure recovery
        
        # Log periodic heartbeats showing ongoing audio proc
        current_time = time()
        if not hasattr(self, '_last_heartbeat') or current_time - self._last_heartbeat >= 10:
            logger.debug(f"Processing audio chunk, last FFmpeg activity: {current_time - self.last_ffmpeg_activity:.2f}s ago")
            self._last_heartbeat = current_time
        
        while retry_count < max_retries:
            try:
                if not self.ffmpeg_process or not hasattr(self.ffmpeg_process, 'stdin') or self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg process not available, restarting...")
                    await self.restart_ffmpeg()
                
                loop = asyncio.get_running_loop()                
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: self.ffmpeg_process.stdin.write(message)),
                        timeout=1.0  # Reduced from 2.0 to 1.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg write operation timed out, restarting...")
                    await self.restart_ffmpeg()
                    retry_count += 1
                    continue
                    
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, self.ffmpeg_process.stdin.flush),
                        timeout=1.0  # Reduced from 2.0 to 1.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg flush operation timed out, restarting...")
                    await self.restart_ffmpeg()
                    retry_count += 1
                    continue
                    
                self.last_ffmpeg_activity = time()
                return
                    
            except (BrokenPipeError, AttributeError, OSError) as e:
                retry_count += 1
                logger.warning(f"Error writing to FFmpeg: {e}. Retry {retry_count}/{max_retries}...")
                
                if retry_count < max_retries:
                    await self.restart_ffmpeg()
                    await asyncio.sleep(0.25)  # Reduced from 0.5 to 0.25
                else:
                    logger.error("Maximum retries reached for FFmpeg process")
                    await self.restart_ffmpeg()
                    return