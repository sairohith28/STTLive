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
    
    def __init__(self):
        """Initialize the audio processor with configuration, models, and state."""
        
        models = WhisperLiveKit()
        
        # Generate a unique client ID for this processor instance
        self.client_id = str(uuid.uuid4())
        
        # Audio processing settings
        self.args = models.args
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
        self.asr = models.asr
        self.tokenizer = models.tokenizer
        self.diarization = models.diarization
        self.batch_service = models.batch_service  # Use the shared batch service
        self.ffmpeg_process = self.start_ffmpeg_decoder()
        self.transcription_queue = asyncio.Queue(maxsize=10) if self.args.transcription else None
        self.diarization_queue = asyncio.Queue(maxsize=10) if self.args.diarization else None
        self.pcm_buffer = bytearray()
        
        # Initialize transcription engine if enabled
        if self.args.transcription:
            self.online = online_factory(self.args, models.asr, models.tokenizer)
            
        # Performance optimizations
        self.use_parallel = True  # Enable parallel processing
        self.max_update_freq = 0.2  # Maximum update frequency in seconds
        self.last_update = 0

        logger.info(f"AudioProcessor initialized with client_id: {self.client_id}")

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

    async def transcription_processor(self):
        """Process audio chunks for transcription with optimized handling."""
        self.full_transcription = ""
        self.sep = self.online.asr.sep
        
        while True:
            try:
                pcm_array = await self.transcription_queue.get()
                
                # If using batch service, submit to it
                if self.batch_service and hasattr(self.batch_service, 'submit'):
                    try:
                        # Submit to batch service and get results
                        batch_result = await self.batch_service.submit(self.client_id, pcm_array)
                        
                        # Process the results if valid
                        if batch_result and not batch_result.get("error"):
                            segments = batch_result.get("segments", [])
                            tokens = batch_result.get("tokens", [])
                            
                            # Process the tokens and update state
                            if tokens:
                                # Insert audio chunk into online ASR (needed for buffer management)
                                self.online.insert_audio_chunk(pcm_array)
                                
                                # Update with tokens from batch processing instead
                                # of using self.online.process_iter()
                                new_tokens = tokens
                                
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
                        else:
                            # Fall back to regular processing if batch processing failed
                            logger.warning(f"Batch processing error: {batch_result.get('error', 'Unknown error')}")
                            await self._process_transcription_locally(pcm_array)
                    except Exception as e:
                        logger.error(f"Error in batch transcription: {e}")
                        logger.error(traceback.format_exc())
                        # Fall back to local processing
                        await self._process_transcription_locally(pcm_array)
                else:
                    # Use regular processing if batch service is not available
                    await self._process_transcription_locally(pcm_array)
                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
            finally:
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
        for task in self.tasks:
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