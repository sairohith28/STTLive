import sys
import numpy as np
import logging
import time
from typing import List, Tuple, Optional
from whisperlivekit.timed_objects import ASRToken, Sentence, Transcript

logger = logging.getLogger(__name__)


class HypothesisBuffer:
    """
    Buffer to store and process ASR hypothesis tokens.

    It holds:
      - committed_in_buffer: tokens that have been confirmed (committed)
      - buffer: the last hypothesis that is not yet committed
      - new: new tokens coming from the recognizer
    """
    def __init__(self, logfile=sys.stderr, confidence_validation=False):
        self.confidence_validation = confidence_validation
        self.committed_in_buffer: List[ASRToken] = []
        self.buffer: List[ASRToken] = []
        self.new: List[ASRToken] = []
        self.last_committed_time = 0.0
        self.last_committed_word: Optional[str] = None
        self.logfile = logfile

    def insert(self, new_tokens: List[ASRToken], offset: float):
        """
        Insert new tokens (after applying a time offset) and compare them with the 
        already committed tokens. Only tokens that extend the committed hypothesis 
        are added.
        """
        # Apply the offset to each token.
        new_tokens = [token.with_offset(offset) for token in new_tokens]
        # Only keep tokens that are roughly “new”
        self.new = [token for token in new_tokens if token.start > self.last_committed_time - 0.1]

        if self.new:
            first_token = self.new[0]
            if abs(first_token.start - self.last_committed_time) < 1:
                if self.committed_in_buffer:
                    committed_len = len(self.committed_in_buffer)
                    new_len = len(self.new)
                    # Try to match 1 to 5 consecutive tokens
                    max_ngram = min(min(committed_len, new_len), 5)
                    for i in range(1, max_ngram + 1):
                        committed_ngram = " ".join(token.text for token in self.committed_in_buffer[-i:])
                        new_ngram = " ".join(token.text for token in self.new[:i])
                        if committed_ngram == new_ngram:
                            removed = []
                            for _ in range(i):
                                removed_token = self.new.pop(0)
                                removed.append(repr(removed_token))
                            logger.debug(f"Removing last {i} words: {' '.join(removed)}")
                            break

    def flush(self) -> List[ASRToken]:
        """
        Returns the committed chunk, defined as the longest common prefix
        between the previous hypothesis and the new tokens.
        """
        committed: List[ASRToken] = []
        while self.new:
            current_new = self.new[0]
            if self.confidence_validation and current_new.probability and current_new.probability > 0.95:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.new.pop(0)
                self.buffer.pop(0) if self.buffer else None
            elif not self.buffer:
                break
            elif current_new.text == self.buffer[0].text:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(committed)
        return committed

    def pop_committed(self, time: float):
        """
        Remove tokens (from the beginning) that have ended before `time`.
        """
        while self.committed_in_buffer and self.committed_in_buffer[0].end <= time:
            self.committed_in_buffer.pop(0)



class OnlineASRProcessor:
    """
    Processes incoming audio in a streaming fashion, calling the ASR system
    periodically, and uses a hypothesis buffer to commit and trim recognized text.
    
    The processor supports two types of buffer trimming:
      - "sentence": trims at sentence boundaries (using a sentence tokenizer)
      - "segment": trims at fixed segment durations.
    """
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        tokenize_method: Optional[callable] = None,
        buffer_trimming: Tuple[str, float] = ("segment", 15),
        confidence_validation = False,
        logfile=sys.stderr,
    ):
        """
        asr: An ASR system object (for example, a WhisperASR instance) that
             provides a `transcribe` method, a `ts_words` method (to extract tokens),
             a `segments_end_ts` method, and a separator attribute `sep`.
        tokenize_method: A function that receives text and returns a list of sentence strings.
        buffer_trimming: A tuple (option, seconds), where option is either "sentence" or "segment".
        """
        self.asr = asr
        self.tokenize = tokenize_method
        self.logfile = logfile
        self.confidence_validation = confidence_validation
        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming
        
        # Add buffer monitoring to prevent excessive growth
        self.last_trim_time = time.time()
        self.force_trim_interval = 8.0  # Force trimming every 8 seconds to prevent lag
        self.max_buffer_size = self.SAMPLING_RATE * 20  # Maximum 20 seconds of audio to prevent OOM
        
        if self.buffer_trimming_way not in ["sentence", "segment"]:
            raise ValueError("buffer_trimming must be either 'sentence' or 'segment'")
        if self.buffer_trimming_sec <= 0:
            raise ValueError("buffer_trimming_sec must be positive")
        elif self.buffer_trimming_sec > 30:
            logger.warning(
                f"buffer_trimming_sec is set to {self.buffer_trimming_sec}, which is very long. It may cause OOM."
            )
        
        # Adjust buffer trimming based on model size for better experience
        if hasattr(asr, 'model_size') and asr.model_size:
            model_size = str(asr.model_size).lower()
            if "large" in model_size or "distil" in model_size:
                # Use more aggressive trimming for large models
                self.buffer_trimming_sec = min(self.buffer_trimming_sec, 10.0)
                logger.info(f"Large model detected, adjusted buffer trimming to {self.buffer_trimming_sec}s")

    def init(self, offset: Optional[float] = None):
        """Initialize or reset the processing buffers."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile, confidence_validation=self.confidence_validation)
        self.buffer_time_offset = offset if offset is not None else 0.0
        self.transcript_buffer.last_committed_time = self.buffer_time_offset
        self.committed: List[ASRToken] = []
        self.last_trim_time = time.time()

    def insert_audio_chunk(self, audio: np.ndarray):
        """Append an audio chunk (a numpy array) to the current audio buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio)
        
        # Enforce maximum buffer size to prevent unbounded growth
        current_buffer_size = len(self.audio_buffer)
        if current_buffer_size > self.max_buffer_size:
            # Cut to half of max size to avoid frequent trimming
            excess = current_buffer_size - (self.max_buffer_size // 2)
            cutoff_time = excess / self.SAMPLING_RATE
            self.buffer_time_offset += cutoff_time
            self.audio_buffer = self.audio_buffer[excess:]
            logger.warning(f"Buffer exceeded maximum size, trimmed {cutoff_time:.2f}s from beginning")
    
    def prompt(self) -> Tuple[str, str]:
        """
        Get a prompt for the transcription (common transcription prefix).
        Returns a tuple (prompt_text, context_text).
        prompt_text includes the last chunk of committed tokens, to provide
        context for the transcription.
        """
        prompt_list = []
        context_text = ""
        # Prompt with max 16 tokens
        for token in list(reversed(self.committed))[-16:]:
            prompt_list.append(token.text)
            context_text = token.text + self.asr.sep + context_text
        return self.asr.sep.join(prompt_list[::-1]), context_text

    def get_buffer(self):
        """
        Get the unvalidated buffer in string format.
        """
        return self.concatenate_tokens(self.transcript_buffer.buffer)

    def process_iter(self) -> Transcript:
        """
        Processes the current audio buffer with periodic forced trimming to prevent lag.
        
        Returns a Transcript object representing the committed transcript.
        """
        # Check if we need to force trim based on time interval
        current_time = time.time()
        if current_time - self.last_trim_time > self.force_trim_interval:
            # Force trim even if not at natural boundary
            logger.debug(f"Forcing buffer trim after {self.force_trim_interval}s interval")
            buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
            if buffer_duration > 3.0:  # Only trim if we have at least 3 seconds
                trim_point = max(1.0, buffer_duration * 0.5)  # Trim half of buffer
                trim_time = self.buffer_time_offset + trim_point
                self.chunk_at(trim_time)
                self.last_trim_time = current_time
        
        prompt_text, _ = self.prompt()
        logger.debug(
            f"Transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:.2f} seconds from {self.buffer_time_offset:.2f}"
        )
        
        # Use time-based beam size adjustment for large buffers to reduce processing time
        init_kwargs = {}
        if hasattr(self.asr, 'transcribe_with_params'):
            buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
            if buffer_duration > 10.0:
                # Use faster parameters for large buffers
                init_kwargs['beam_size'] = 1
        
        # Perform transcription with potential parameters
        if hasattr(self.asr, 'transcribe_with_params') and init_kwargs:
            res = self.asr.transcribe_with_params(self.audio_buffer, init_prompt=prompt_text, **init_kwargs)
        else:
            res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt_text)
            
        tokens = self.asr.ts_words(res)  # Expecting List[ASRToken]
        self.transcript_buffer.insert(tokens, self.buffer_time_offset)
        committed_tokens = self.transcript_buffer.flush()
        self.committed.extend(committed_tokens)
        completed = self.concatenate_tokens(committed_tokens)
        logger.debug(f">>>> COMPLETE NOW: {completed.text}")
        incomp = self.concatenate_tokens(self.transcript_buffer.buffer)
        logger.debug(f"INCOMPLETE: {incomp.text}")

        # More aggressive buffer management to prevent lag
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
        if buffer_duration > self.buffer_trimming_sec * 0.7:  # Lower threshold for trimming
            if self.buffer_trimming_way == "sentence" and committed_tokens:
                self.chunk_completed_sentence()
            elif self.buffer_trimming_way == "segment":
                self.chunk_completed_segment(res)
                
        logger.debug(
            f"Length of audio buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f} seconds"
        )
        return committed_tokens

    def chunk_completed_sentence(self):
        """
        Aggressively trim the audio buffer to prevent lag buildup.
        If the committed tokens form at least one sentence, chunk at the 
        end of the last sentence, or aggressively trim if buffer is growing too large.
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE        
        if not self.committed:
            if buffer_duration > self.buffer_trimming_sec * 0.7:  # More aggressive
                chunk_time = self.buffer_time_offset + (buffer_duration / 3)
                logger.debug(f"--- No speech detected, forced chunking at {chunk_time:.2f}")
                self.chunk_at(chunk_time)
            return
        
        logger.debug("COMPLETED SENTENCE: " + " ".join(token.text for token in self.committed))
        sentences = self.words_to_sentences(self.committed)
        for sentence in sentences:
            logger.debug(f"\tSentence: {sentence.text}")
        
        chunk_done = False
        if len(sentences) >= 1:  # Reduced from 2 to 1 - chunk at any sentence
            if len(sentences) > 1:
                sentences.pop(0)  # Remove first sentence if we have multiple
            chunk_time = sentences[-1].end
            logger.debug(f"--- Sentence chunked at {chunk_time:.2f}")
            self.chunk_at(chunk_time)
            chunk_done = True
            self.last_trim_time = time.time()
        
        if not chunk_done and buffer_duration > self.buffer_trimming_sec * 0.8:  # More aggressive
            last_committed_time = self.committed[-1].end
            logger.debug(f"--- Not enough sentences, chunking at last committed time {last_committed_time:.2f}")
            self.chunk_at(last_committed_time)
            self.last_trim_time = time.time()

    def chunk_completed_segment(self, res):
        """
        Optimized chunking of the audio buffer based on segment-end timestamps.
        Improved to be more aggressive with trimming to reduce lag.
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE        
        if not self.committed:
            if buffer_duration > self.buffer_trimming_sec * 0.7:  # More aggressive
                # More aggressive chunking when no speech detected
                chunk_time = self.buffer_time_offset + (buffer_duration / 3)  # Chunk earlier at 1/3 of buffer
                logger.debug(f"--- No speech detected, aggressive chunking at {chunk_time:.2f}")
                self.chunk_at(chunk_time)
                self.last_trim_time = time.time()
            return
        
        logger.debug("Processing committed tokens for segmenting")
        ends = self.asr.segments_end_ts(res)
        last_committed_time = self.committed[-1].end        
        chunk_done = False
        
        # More aggressive chunking strategy
        if len(ends) > 0:  # Changed from 1 to 0 - trim even with one segment
            logger.debug("Segments available for chunking")
            
            # First try to find optimal chunking point
            optimal_point_found = False
            for i in range(1, len(ends) + 1):
                if i > len(ends):
                    break
                segment_end = ends[-i] + self.buffer_time_offset
                
                # More permissive chunking criteria
                if segment_end <= last_committed_time or buffer_duration > self.buffer_trimming_sec * 0.8:
                    logger.debug(f"--- Segment chunked at optimal point {segment_end:.2f}")
                    self.chunk_at(segment_end)
                    chunk_done = True
                    optimal_point_found = True
                    self.last_trim_time = time.time()
                    break
            
            # If no optimal point found but we have segments and the buffer is getting large
            if not optimal_point_found and buffer_duration > self.buffer_trimming_sec * 0.6:  # More aggressive
                # Use the latest segment even if not optimal
                if len(ends) > 0:
                    segment_end = ends[-1] + self.buffer_time_offset
                    if segment_end > 0:
                        logger.debug(f"--- Buffer growing, forced chunking at latest segment {segment_end:.2f}")
                        self.chunk_at(segment_end)
                        chunk_done = True
                        self.last_trim_time = time.time()
        else:
            logger.debug("--- Not enough segments to chunk")
        
        # Enhanced aggressive chunking for long buffers
        if not chunk_done:
            if buffer_duration > self.buffer_trimming_sec * 0.7:  # More aggressive
                # For buffers approaching the limit, chunk at last committed token
                logger.debug(f"--- Buffer approaching limit, chunking at last committed time {last_committed_time:.2f}")
                self.chunk_at(last_committed_time)
                self.last_trim_time = time.time()
            elif buffer_duration > self.buffer_trimming_sec * 0.9:  # More aggressive
                # For buffers exceeding the limit, take more drastic action
                # Chunk at midpoint between buffer start and last committed
                midpoint = self.buffer_time_offset + (last_committed_time - self.buffer_time_offset) * 0.7
                logger.debug(f"--- Buffer exceeded limit, emergency chunking at {midpoint:.2f}")
                self.chunk_at(midpoint)
                self.last_trim_time = time.time()
        
        logger.debug("Segment chunking complete")

    def chunk_at(self, chunk_timestamp: float):
        """
        Trim both the hypothesis and audio buffer at the given time.
        """
        logger.debug(f"Chunking at {chunk_timestamp:.2f}s")
        logger.debug(
            f"Audio buffer length before chunking: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f}s"
        )
        self.transcript_buffer.pop_committed(chunk_timestamp)
        cut_seconds = chunk_timestamp - self.buffer_time_offset
        
        # Safety check to prevent bad trimming
        if cut_seconds <= 0 or cut_seconds >= len(self.audio_buffer)/self.SAMPLING_RATE:
            logger.warning(f"Invalid chunk point {chunk_timestamp:.2f}s, offset: {self.buffer_time_offset:.2f}s")
            # If invalid, just trim a safe amount
            safe_trim = min(len(self.audio_buffer) // 2, int(self.SAMPLING_RATE * 5))
            if safe_trim > 0:
                self.audio_buffer = self.audio_buffer[safe_trim:]
                self.buffer_time_offset += safe_trim / self.SAMPLING_RATE
        else:
            trim_samples = int(cut_seconds * self.SAMPLING_RATE)
            self.audio_buffer = self.audio_buffer[trim_samples:]
            self.buffer_time_offset = chunk_timestamp
            
        logger.debug(
            f"Audio buffer length after chunking: {len(self.audio_buffer)/self.SAMPLING_RATE:.2f}s"
        )
        
        # Update last trim time
        self.last_trim_time = time.time()

    def words_to_sentences(self, tokens: List[ASRToken]) -> List[Sentence]:
        """
        Converts a list of tokens to a list of Sentence objects using the provided
        sentence tokenizer.
        """
        if not tokens:
            return []

        full_text = " ".join(token.text for token in tokens)

        if self.tokenize:
            try:
                sentence_texts = self.tokenize(full_text)
            except Exception as e:
                # Some tokenizers (e.g., MosesSentenceSplitter) expect a list input.
                try:
                    sentence_texts = self.tokenize([full_text])
                except Exception as e2:
                    raise ValueError("Tokenization failed") from e2
        else:
            sentence_texts = [full_text]

        sentences: List[Sentence] = []
        token_index = 0
        for sent_text in sentence_texts:
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            sent_tokens = []
            accumulated = ""
            # Accumulate tokens until roughly matching the length of the sentence text.
            while token_index < len(tokens) and len(accumulated) < len(sent_text):
                token = tokens[token_index]
                accumulated = (accumulated + " " + token.text).strip() if accumulated else token.text
                sent_tokens.append(token)
                token_index += 1
            if sent_tokens:
                sentence = Sentence(
                    start=sent_tokens[0].start,
                    end=sent_tokens[-1].end,
                    text=" ".join(t.text for t in sent_tokens),
                )
                sentences.append(sentence)
        return sentences
    def finish(self) -> Transcript:
        """
        Flush the remaining transcript when processing ends.
        """
        remaining_tokens = self.transcript_buffer.buffer
        final_transcript = self.concatenate_tokens(remaining_tokens)
        logger.debug(f"Final non-committed transcript: {final_transcript}")
        self.buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        return final_transcript

    def concatenate_tokens(
        self,
        tokens: List[ASRToken],
        sep: Optional[str] = None,
        offset: float = 0
    ) -> Transcript:
        sep = sep if sep is not None else self.asr.sep
        text = sep.join(token.text for token in tokens)
        probability = sum(token.probability for token in tokens if token.probability) / len(tokens) if tokens else None
        if tokens:
            start = offset + tokens[0].start
            end = offset + tokens[-1].end
        else:
            start = None
            end = None
        return Transcript(start, end, text, probability=probability)


class VACOnlineASRProcessor:
    """
    Wraps an OnlineASRProcessor with a Voice Activity Controller (VAC).
    
    It receives small chunks of audio, applies VAD (e.g. with Silero),
    and when the system detects a pause in speech (or end of an utterance)
    it finalizes the utterance immediately.
    """
    SAMPLING_RATE = 16000

    def __init__(
        self, 
        online_chunk_size: float,
        asr,
        tokenize_method: Optional[callable] = None,
        buffer_trimming: Tuple[str, float] = ("segment", 15),
        confidence_validation = False,
        logfile=sys.stderr,
    ):
        """
        Initialize the VAC processor with the ASR model and configuration parameters.
        
        Args:
            online_chunk_size: Minimum audio chunk size in seconds.
            asr: An ASR system object that provides transcribe and ts_words methods.
            tokenize_method: A function for sentence tokenization.
            buffer_trimming: Buffer trimming configuration.
            confidence_validation: Whether to use confidence scores for validation.
            logfile: Log file or stream.
        """
        self.online_chunk_size = online_chunk_size
        self.asr = asr  # Store the ASR model directly
        self.tokenize = tokenize_method
        self.buffer_trimming = buffer_trimming
        self.confidence_validation = confidence_validation
        self.logfile = logfile
        
        # Create the associated online processor
        self.online = OnlineASRProcessor(
            self.asr,
            self.tokenize,
            buffer_trimming=buffer_trimming,
            confidence_validation=confidence_validation,
            logfile=logfile
        )
        
        # Load a VAD model (e.g. Silero VAD)
        import torch
        model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        from .silero_vad_iterator import FixedVADIterator
        
        self.vac = FixedVADIterator(model)
        self.init()

    def init(self, offset: Optional[float] = None):
        """Initialize or reset the processing buffers."""
        self.online.init(offset)
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        self.status: Optional[str] = None  # "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        """Clear the audio buffer and update buffer offset."""
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio: np.ndarray):
        """
        Process an incoming small audio chunk:
          - run VAD on the chunk,
          - decide whether to send the audio to the online ASR processor immediately,
          - and/or to mark the current utterance as finished.
        """
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            try:
                # VAD returned a result; adjust the frame number
                frame = list(res.values())[0] - self.buffer_offset
                if "start" in res and "end" not in res:
                    self.status = "voice"
                    send_audio = self.audio_buffer[frame:]
                    self.online.init(offset=(frame + self.buffer_offset) / self.SAMPLING_RATE)
                    self.online.insert_audio_chunk(send_audio)
                    self.current_online_chunk_buffer_size += len(send_audio)
                    self.clear_buffer()
                elif "end" in res and "start" not in res:
                    self.status = "nonvoice"
                    send_audio = self.audio_buffer[:frame]
                    self.online.insert_audio_chunk(send_audio)
                    self.current_online_chunk_buffer_size += len(send_audio)
                    self.is_currently_final = True
                    self.clear_buffer()
                else:
                    beg = res["start"] - self.buffer_offset
                    end = res["end"] - self.buffer_offset
                    self.status = "nonvoice"
                    send_audio = self.audio_buffer[beg:end]
                    self.online.init(offset=(beg + self.buffer_offset) / self.SAMPLING_RATE)
                    self.online.insert_audio_chunk(send_audio)
                    self.current_online_chunk_buffer_size += len(send_audio)
                    self.is_currently_final = True
                    self.clear_buffer()
            except Exception as e:
                logger.error(f"Error in VAC processing: {e}")
                # Fall back to regular processing on error
                self.online.insert_audio_chunk(audio)
        else:
            if self.status == "voice":
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # Keep 1 second worth of audio in case VAD later detects voice,
                # but trim to avoid unbounded memory usage.
                self.buffer_offset += max(0, len(self.audio_buffer) - self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]

    def process_iter(self) -> Transcript:
        """
        Depending on the VAD status and the amount of accumulated audio,
        process the current audio chunk.
        """
        try:
            if self.is_currently_final:
                result = self.finish()
                return result
            elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
                self.current_online_chunk_buffer_size = 0
                return self.online.process_iter()
            else:
                logger.debug("No online update, only VAD")
                return Transcript(None, None, "")
        except Exception as e:
            logger.error(f"Error in VAC process_iter: {e}")
            # Return empty transcript on error
            return Transcript(None, None, "")

    def finish(self) -> Transcript:
        """Finish processing by flushing any remaining text."""
        result = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return result
    
    def get_buffer(self):
        """Get the unvalidated buffer in string format."""
        return self.online.get_buffer()
