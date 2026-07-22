"""Kyutai TTS Handler

Streaming text-to-speech handler for Kyutai's Delayed-Streams-Modeling (DSM)
TTS models, served through the ``moshi`` library.

The default model is ``kyutai/tts-1.6b-en_fr`` (English + French), a
CFG-distilled, multi-speaker model conditioned on voice embeddings pulled from
the ``kyutai/tts-voices`` repository.

Works on both CUDA and CPU (bfloat16 on CUDA, float32 on CPU by default). The
model produces mimi codec frames at 12.5 Hz which are decoded incrementally to
24kHz PCM and resampled to the 16kHz pipeline rate.
"""

from __future__ import annotations

import logging
import queue as queue_module
import threading
from threading import Event
from time import perf_counter
from typing import Any, Iterator, Optional

import numpy as np
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from rich.console import Console

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MODEL_REPO = "kyutai/tts-1.6b-en_fr"
DEFAULT_VOICE_REPO = "kyutai/tts-voices"
DEFAULT_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
DEFAULT_CFG_COEF = 2.0
DEFAULT_N_Q = 32
DEFAULT_TEMP = 0.6
DEFAULT_PADDING_BETWEEN = 1
PIPELINE_SR = 16000

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


class _GenerationAborted(Exception):
    """Raised inside the on_frame callback to stop generation early on interruption."""


class KyutaiTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """TTS handler for Kyutai DSM models (e.g. ``kyutai/tts-1.6b-en_fr``)."""

    def setup(
        self,
        should_listen: Event,
        model_repo: str = DEFAULT_MODEL_REPO,
        voice_repo: str = DEFAULT_VOICE_REPO,
        voice: str = DEFAULT_VOICE,
        device: str = "cuda",
        dtype: Optional[str] = None,
        n_q: int = DEFAULT_N_Q,
        temp: float = DEFAULT_TEMP,
        cfg_coef: float = DEFAULT_CFG_COEF,
        padding_between: int = DEFAULT_PADDING_BETWEEN,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,  # accepted for pipeline compatibility
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.model_repo = model_repo
        self.voice_repo = voice_repo
        self.voice = voice
        self.device = device
        self.n_q = n_q
        self.temp = temp
        self.cfg_coef = cfg_coef
        self.padding_between = padding_between
        self.blocksize = blocksize

        self.dtype = self._normalize_dtype(dtype, device)

        # Bridge flag: set to request early termination of the background
        # generation thread when a turn is interrupted.
        self._abort_generation = False
        # Cache of voice name -> ConditionAttributes so we do not recompute
        # speaker embeddings for every utterance.
        self._condition_cache: dict[str, Any] = {}
        self._initial_voice = voice

        self._load_model()

        # mimi codec output rate (typically 24000 Hz).
        self.model_sample_rate = int(getattr(self.model.mimi, "sample_rate", 24000))

        self._default_condition_attributes = self._resolve_condition_attributes(self.voice)

        self.warmup()

    def _normalize_dtype(self, dtype: Optional[str], device: str) -> str:
        if dtype:
            normalized = _DTYPE_ALIASES.get(str(dtype).strip().lower())
            if normalized is None:
                valid = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
                raise ValueError(f"Unsupported kyutai_tts_dtype '{dtype}'. Valid values: {valid}.")
            return normalized
        # bfloat16 is the moshi default on GPU; CPU inference is more reliable in float32.
        return "float32" if device == "cpu" else "bfloat16"

    def _load_model(self) -> None:
        try:
            import torch
            from moshi.models.loaders import CheckpointInfo
            from moshi.models.tts import TTSModel
        except ImportError as e:
            raise ImportError(
                "moshi is required for Kyutai TTS. "
                'Install it with `pip install "speech-to-speech[kyutai]"`.'
            ) from e

        torch_dtype = getattr(torch, self.dtype)
        logger.info("Loading Kyutai TTS model %s on %s (%s)", self.model_repo, self.device, self.dtype)

        checkpoint_info = CheckpointInfo.from_hf_repo(self.model_repo)
        self.model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            voice_repo=self.voice_repo,
            n_q=self.n_q,
            temp=self.temp,
            device=torch.device(self.device),
            dtype=torch_dtype,
        )

        # Validate the requested CFG coefficient against the model's distilled
        # conditionings, when the model exposes them.
        valid_cfg = getattr(self.model, "valid_cfg_conditionings", None)
        if valid_cfg and self.cfg_coef not in valid_cfg:
            valids = ", ".join(str(x) for x in valid_cfg)
            raise ValueError(f"Unsupported kyutai_tts_cfg_coef {self.cfg_coef}. Valid values: {valids}.")

    def _resolve_condition_attributes(self, voice: str) -> Any:
        """Build (and cache) the speaker ConditionAttributes for a voice name."""
        cached = self._condition_cache.get(voice)
        if cached is not None:
            return cached

        voice_path = self.model.get_voice_path(voice)
        condition_attributes = self.model.make_condition_attributes([voice_path], cfg_coef=self.cfg_coef)
        self._condition_cache[voice] = condition_attributes
        return condition_attributes

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        try:
            self.model.warmup([self._default_condition_attributes])
        except Exception as e:
            logger.warning("Kyutai-TTS backend warmup failed: %s", e)

        try:
            for _ in self._synthesize("Hello, this is a warmup.", self._default_condition_attributes):
                pass
            logger.info("%s warmed up", self.__class__.__name__)
        except Exception as e:
            logger.warning("Warmup generation failed: %s", e)

    def _to_int16(self, audio: np.ndarray) -> np.ndarray:
        return np.clip(audio * 32768, -32768, 32767).astype(np.int16)

    def _resample_to_pipeline_sr(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == PIPELINE_SR:
            return audio
        from scipy.signal import resample_poly

        gcd = np.gcd(PIPELINE_SR, sr)
        return resample_poly(audio, up=PIPELINE_SR // gcd, down=sr // gcd)

    def _pcm_frames(self, text: str, condition_attributes: Any) -> Iterator[np.ndarray]:
        """Yield float32 mono PCM frames at ``self.model_sample_rate``.

        The moshi ``generate`` call is callback-based, so we run it in a
        background thread and bridge decoded frames through a queue. Decoding
        happens inside the ``on_frame`` callback (same thread as generation),
        wrapped in ``mimi.streaming`` so codec state is preserved incrementally.
        """
        import torch

        entries = self.model.prepare_script([text], padding_between=self.padding_between)
        mimi = self.model.mimi

        frame_queue: queue_module.Queue = queue_module.Queue()
        sentinel = object()
        self._abort_generation = False

        def _on_frame(frame: Any) -> None:
            if self._abort_generation:
                raise _GenerationAborted()
            # frame shape is [batch, 1 + n_q, 1]; -1 marks ungenerated tokens
            # (the initial delay steps), which we skip.
            if (frame != -1).all():
                with torch.no_grad():
                    pcm = mimi.decode(frame[:, 1:, :])
                pcm_np = pcm.cpu().numpy()[0, 0]
                frame_queue.put(np.clip(pcm_np, -1.0, 1.0).astype(np.float32))

        def _run() -> None:
            try:
                with mimi.streaming(1), torch.no_grad():
                    self.model.generate([entries], [condition_attributes], on_frame=_on_frame)
            except _GenerationAborted:
                pass
            except Exception as e:  # surface generation errors to the consumer
                frame_queue.put(e)
            finally:
                frame_queue.put(sentinel)

        thread = threading.Thread(target=_run, name="KyutaiTTSGen", daemon=True)
        thread.start()
        try:
            while True:
                item = frame_queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Ensure the background thread stops if the consumer abandons us.
            self._abort_generation = True
            thread.join(timeout=5.0)

    def _synthesize(self, text: str, condition_attributes: Any) -> Iterator[np.ndarray]:
        """Stream int16 audio blocks for ``text`` at the 16kHz pipeline rate."""
        cancel_gen = self.cancel_scope.generation if self.cancel_scope else None
        start = perf_counter()
        total_samples = 0
        first_chunk = True
        found_speech = False
        leftover = np.array([], dtype=np.int16)

        for pcm in self._pcm_frames(text, condition_attributes):
            if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
                logger.info("TTS generation cancelled (interruption)")
                self._abort_generation = True
                return

            if pcm.size == 0:
                continue

            if first_chunk:
                logger.info("Kyutai-TTS TTFA: %.2fs", perf_counter() - start)
                first_chunk = False

            audio_chunk = self._resample_to_pipeline_sr(pcm, self.model_sample_rate)
            audio_chunk = self._to_int16(audio_chunk)

            # Trim the initial silent ramp-up, but keep a short preroll so soft
            # initial phonemes are not shaved off.
            if not found_speech:
                threshold = int(32768 * 0.01)
                above = np.abs(audio_chunk) > threshold
                if not np.any(above):
                    continue
                start_idx = max(0, int(np.argmax(above)) - int(PIPELINE_SR * 0.040))
                audio_chunk = audio_chunk[start_idx:]
                found_speech = True

            audio_chunk = np.concatenate([leftover, audio_chunk])

            n = (len(audio_chunk) // self.blocksize) * self.blocksize
            for i in range(0, n, self.blocksize):
                yield audio_chunk[i : i + self.blocksize]
                total_samples += self.blocksize
            leftover = audio_chunk[n:]

        if len(leftover) > 0:
            chunk = np.pad(leftover, (0, self.blocksize - len(leftover)))
            yield chunk
            total_samples += len(leftover)

        generation_time = perf_counter() - start
        audio_duration = total_samples / PIPELINE_SR
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info(
            "Kyutai-TTS generated %.2fs audio in %.2fs (RTF: %.2f)",
            audio_duration,
            generation_time,
            rtf,
        )

    def _coalesce_pending_tts_input(self, current_input: TTSInput) -> tuple[str, Optional[str], bool]:
        """Combine already-queued text chunks before the next TTS synthesis call."""
        if not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return current_input.text, current_input.language_code, False

        text = current_input.text
        language_code = current_input.language_code

        parts = [text.strip()] if text and text.strip() else []
        saw_end_of_response = False

        with self.queue_in.mutex:
            while self.queue_in.queue:
                next_item = self.queue_in.queue[0]
                if is_control_message(next_item, SESSION_END.kind):
                    break
                if isinstance(next_item, bytes) and next_item == PIPELINE_END:
                    break
                if isinstance(next_item, EndOfResponse):
                    saw_end_of_response = True
                    break
                if not isinstance(next_item, TTSInput):
                    break
                if current_input.turn_id != next_item.turn_id or current_input.turn_revision != next_item.turn_revision:
                    break
                if (
                    language_code is not None
                    and next_item.language_code is not None
                    and next_item.language_code != language_code
                ):
                    break

                self.queue_in.queue.popleft()
                if next_item.text.strip():
                    parts.append(next_item.text.strip())
                if language_code is None:
                    language_code = next_item.language_code

        combined_text = " ".join(parts).strip()
        return combined_text, language_code, saw_end_of_response

    def _apply_session_voice_override(
        self,
        runtime_config: RuntimeConfig | None,
        response: RealtimeResponseCreateParams | None,
    ) -> None:
        """Switch the active voice from a realtime session/response voice, if valid."""
        voice = None
        try:
            if response is not None:
                voice = getattr(getattr(response, "audio", None), "voice", None)
            if voice is None and runtime_config is not None:
                session = getattr(runtime_config, "session", None)
                audio = getattr(session, "audio", None)
                output = getattr(audio, "output", None)
                voice = getattr(output, "voice", None)
        except Exception:
            voice = None

        if not voice or not isinstance(voice, str):
            return
        if voice == self.voice:
            return

        # Kyutai voices are repository paths (e.g. "expresso/....wav"), not the
        # generic OpenAI voice names. Only accept path-like overrides.
        if "/" not in voice and not voice.endswith((".wav", ".safetensors")):
            logger.warning(
                "Ignoring Kyutai-TTS session voice override %r: expected a kyutai/tts-voices path.",
                voice,
            )
            return

        try:
            condition_attributes = self._resolve_condition_attributes(voice)
        except Exception as e:
            logger.warning("Failed to load Kyutai-TTS voice override %r: %s", voice, e)
            return

        self.voice = voice
        self._default_condition_attributes = condition_attributes
        logger.info("Kyutai-TTS switched to session voice %r", voice)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        runtime_config = tts_input.runtime_config
        response = tts_input.response

        self._apply_session_voice_override(runtime_config, response)

        coalesced_text, _language_code, _saw_end_of_response = self._coalesce_pending_tts_input(tts_input)
        text = coalesced_text or "Hello."

        console.print(f"[green]ASSISTANT: {text}")

        try:
            first_audio = True
            for audio_chunk in self._synthesize(text, self._default_condition_attributes):
                if first_audio:
                    self._log_first_audio_latency(tts_input)
                    first_audio = False
                yield audio_chunk
        except Exception as e:
            logger.error("Error during Kyutai-TTS generation: %s", e, exc_info=True)

    def _log_first_audio_latency(self, tts_input: TTSInput) -> None:
        if tts_input.speech_stopped_at_s is None:
            return
        latency_s = perf_counter() - tts_input.speech_stopped_at_s
        if latency_s < 0:
            return
        logger.info(
            "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
            latency_s,
            tts_input.turn_id,
            tts_input.turn_revision,
        )

    def on_session_end(self) -> None:
        self.voice = self._initial_voice
        try:
            self._default_condition_attributes = self._resolve_condition_attributes(self.voice)
        except Exception as e:
            logger.warning("Failed to restore Kyutai-TTS initial voice %r: %s", self.voice, e)
        logger.debug("Kyutai-TTS session state reset")

    def cleanup(self) -> None:
        self._abort_generation = True
        try:
            del self.model
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("Kyutai-TTS cleanup error: %s", e)
