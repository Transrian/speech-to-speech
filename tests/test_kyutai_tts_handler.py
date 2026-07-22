import logging
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

import speech_to_speech.TTS.kyutai_tts_handler as kyutai_module
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.kyutai_tts_handler import KyutaiTTSHandler


def _new_handler():
    handler = object.__new__(KyutaiTTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.blocksize = 512
    handler.model_sample_rate = 24000
    handler.voice = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    handler._initial_voice = handler.voice
    handler._condition_cache = {}
    handler._abort_generation = False
    handler._default_condition_attributes = object()
    handler.queue_in = Queue()
    return handler


# --------------------------------------------------------------------------- #
# setup: dtype + cfg validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", "float32"), ("cuda", "bfloat16"), ("mps", "bfloat16")],
)
def test_normalize_dtype_defaults_per_device(device, expected):
    handler = object.__new__(KyutaiTTSHandler)
    assert handler._normalize_dtype(None, device) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("bf16", "bfloat16"), ("float16", "float16"), ("FP32", "float32"), ("half", "float16")],
)
def test_normalize_dtype_accepts_aliases(value, expected):
    handler = object.__new__(KyutaiTTSHandler)
    assert handler._normalize_dtype(value, "cuda") == expected


def test_normalize_dtype_rejects_unknown():
    handler = object.__new__(KyutaiTTSHandler)
    with pytest.raises(ValueError, match="Unsupported kyutai_tts_dtype"):
        handler._normalize_dtype("int8", "cuda")


def test_setup_loads_model_and_resolves_voice(monkeypatch):
    recorded = {}

    def _load_model(self):
        self.model = SimpleNamespace(
            mimi=SimpleNamespace(sample_rate=24000),
            valid_cfg_conditionings=[1.0, 1.5, 2.0],
        )
        recorded["loaded"] = True

    def _resolve(self, voice):
        recorded["voice"] = voice
        return SimpleNamespace(name=voice)

    monkeypatch.setattr(KyutaiTTSHandler, "_load_model", _load_model)
    monkeypatch.setattr(KyutaiTTSHandler, "_resolve_condition_attributes", _resolve)
    monkeypatch.setattr(KyutaiTTSHandler, "warmup", lambda self: None)

    handler = object.__new__(KyutaiTTSHandler)
    handler.setup(Event(), device="cpu", voice="expresso/foo.wav")

    assert recorded["loaded"] is True
    assert handler.dtype == "float32"
    assert handler.model_sample_rate == 24000
    assert handler.voice == "expresso/foo.wav"
    assert recorded["voice"] == "expresso/foo.wav"
    assert handler._initial_voice == "expresso/foo.wav"


def test_setup_rejects_invalid_cfg_coef(monkeypatch):
    def _load_model(self):
        # Mirror _load_model raising when cfg_coef is unsupported.
        valid = [1.0, 1.5, 2.0]
        if self.cfg_coef not in valid:
            raise ValueError(f"Unsupported kyutai_tts_cfg_coef {self.cfg_coef}. Valid values: 1.0, 1.5, 2.0.")

    monkeypatch.setattr(KyutaiTTSHandler, "_load_model", _load_model)
    monkeypatch.setattr(KyutaiTTSHandler, "warmup", lambda self: None)

    handler = object.__new__(KyutaiTTSHandler)
    with pytest.raises(ValueError, match="Unsupported kyutai_tts_cfg_coef"):
        handler.setup(Event(), device="cpu", cfg_coef=3.7)


# --------------------------------------------------------------------------- #
# voice resolution + caching
# --------------------------------------------------------------------------- #


def test_resolve_condition_attributes_caches():
    handler = _new_handler()
    calls = []
    handler.cfg_coef = 2.0
    handler.model = SimpleNamespace(
        get_voice_path=lambda v: calls.append(("path", v)) or f"/voices/{v}",
        make_condition_attributes=lambda voices, cfg_coef: calls.append(("cond", voices, cfg_coef))
        or SimpleNamespace(voices=voices),
    )
    handler._condition_cache = {}

    first = handler._resolve_condition_attributes("expresso/a.wav")
    second = handler._resolve_condition_attributes("expresso/a.wav")

    assert first is second
    # get_voice_path + make_condition_attributes called only once (cached second time)
    assert [c[0] for c in calls] == ["path", "cond"]


# --------------------------------------------------------------------------- #
# warmup
# --------------------------------------------------------------------------- #


def test_warmup_logs_backend_neutral_failure(caplog):
    handler = _new_handler()
    handler.model = SimpleNamespace(
        warmup=lambda attrs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    handler._synthesize = lambda text, cond: iter(())

    with caplog.at_level(logging.WARNING):
        handler.warmup()

    assert "Kyutai-TTS backend warmup failed: boom" in caplog.text


# --------------------------------------------------------------------------- #
# _pcm_frames: callback -> iterator bridge
# --------------------------------------------------------------------------- #


def _fake_streaming_cm():
    class _CM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _CM()


def _make_streaming_model(frames):
    """A fake TTSModel whose generate() replays `frames` through on_frame."""

    def _generate(all_entries, attributes, on_frame=None, **kwargs):
        for frame in frames:
            if on_frame is not None:
                on_frame(frame)

    def _decode(codes):
        # codes is frame[:, 1:, :]; return a tensor-like with cpu().numpy()
        n = 240
        value = float(codes[0, 0, 0])
        arr = np.full((1, 1, n), value, dtype=np.float32)
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: arr))

    mimi = SimpleNamespace(
        sample_rate=24000,
        streaming=lambda batch_size: _fake_streaming_cm(),
        decode=_decode,
    )
    return SimpleNamespace(
        mimi=mimi,
        prepare_script=lambda script, padding_between: [("entry", script)],
        generate=_generate,
    )


def test_pcm_frames_skips_ungenerated_and_yields_valid(monkeypatch):
    # Ensure torch import inside _pcm_frames doesn't require a real device.
    handler = _new_handler()
    handler.padding_between = 1
    # Two ungenerated frames (contain -1) then two valid frames.
    frames = [
        np.array([[[-1], [-1]]], dtype=np.float32),
        np.array([[[0.0], [0.2]]], dtype=np.float32),
        np.array([[[0.0], [0.5]]], dtype=np.float32),
    ]
    handler.model = _make_streaming_model(frames)

    out = list(handler._pcm_frames("hi", object()))

    # Only the 2 valid frames are decoded/yielded.
    assert len(out) == 2
    assert all(isinstance(x, np.ndarray) for x in out)
    assert out[0].shape == (240,)


def test_pcm_frames_propagates_generation_error():
    handler = _new_handler()
    handler.padding_between = 1

    def _generate(all_entries, attributes, on_frame=None, **kwargs):
        raise RuntimeError("gen failed")

    mimi = SimpleNamespace(sample_rate=24000, streaming=lambda b: _fake_streaming_cm(), decode=lambda c: None)
    handler.model = SimpleNamespace(mimi=mimi, prepare_script=lambda s, padding_between: [("e", s)], generate=_generate)
    # generate lives on the model in _pcm_frames via self.model.generate
    handler.model.generate = _generate

    with pytest.raises(RuntimeError, match="gen failed"):
        list(handler._pcm_frames("hi", object()))


# --------------------------------------------------------------------------- #
# _synthesize: resample + int16 + blocksize chunking
# --------------------------------------------------------------------------- #


def test_synthesize_yields_int16_blocks_of_blocksize(monkeypatch):
    handler = _new_handler()
    handler.blocksize = 512
    handler.model_sample_rate = 24000

    # One second of a loud tone at 24kHz -> resampled to 16kHz -> ~16000 samples.
    tone = (0.5 * np.ones(24000, dtype=np.float32))
    monkeypatch.setattr(handler, "_pcm_frames", lambda text, cond: iter([tone]))

    blocks = list(handler._synthesize("hello", object()))

    assert len(blocks) > 0
    assert all(b.dtype == np.int16 for b in blocks)
    assert all(len(b) == handler.blocksize for b in blocks)
    # Audio is audible (not all zeros).
    assert any(np.abs(b).max() > 0 for b in blocks)


def test_synthesize_skips_leading_silence(monkeypatch):
    handler = _new_handler()
    handler.blocksize = 512
    handler.model_sample_rate = 16000  # no resampling

    silence = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(handler, "_pcm_frames", lambda text, cond: iter([silence]))

    blocks = list(handler._synthesize("hello", object()))
    # Pure silence never crosses the speech threshold -> nothing emitted.
    assert blocks == []


# --------------------------------------------------------------------------- #
# process(): EndOfResponse, staleness, commit, latency
# --------------------------------------------------------------------------- #


def test_process_end_of_response_yields_audio_done():
    handler = _new_handler()
    assert list(handler.process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]


def test_process_only_reenables_listening_after_end_of_response(monkeypatch):
    handler = _new_handler()
    handler._apply_session_voice_override = lambda runtime_config, response: None
    handler._synthesize = lambda text, cond: iter([np.zeros(512, dtype=np.int16)])

    monkeypatch.setattr(kyutai_module.console, "print", lambda *a, **k: None)

    outputs = list(handler.process(TTSInput(text="Hi there.", runtime_config=RuntimeConfig())))
    assert len(outputs) == 1
    assert handler.should_listen.is_set() is False

    end_outputs = list(handler.process(EndOfResponse()))
    assert end_outputs == [AUDIO_RESPONSE_DONE]


def test_process_commits_turn_before_generating_audio(monkeypatch, caplog):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    handler = _new_handler()
    handler.speculative_turns = tracker
    handler._apply_session_voice_override = lambda runtime_config, response: None

    def _synth(text, cond):
        assert tracker.is_committed("turn_1", 0)
        yield np.zeros(512, dtype=np.int16)

    handler._synthesize = _synth
    monkeypatch.setattr(kyutai_module.console, "print", lambda *a, **k: None)

    with caplog.at_level(logging.INFO, logger="speech_to_speech.TTS.kyutai_tts_handler"):
        outputs = list(
            handler.process(
                TTSInput(
                    text="Hello there.",
                    turn_id="turn_1",
                    turn_revision=0,
                    speech_stopped_at_s=kyutai_module.perf_counter() - 1.0,
                )
            )
        )

    assert len(outputs) == 1
    assert tracker.is_committed("turn_1", 0)
    assert "Last speech detected to first speech out:" in caplog.text


def test_process_drops_stale_input():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.start_reopen_grace("turn_1", 0, grace_s=0.5)
    handler = _new_handler()
    handler.speculative_turns = tracker
    done = Event()
    outputs = []

    def run_process():
        outputs.extend(handler.process(TTSInput(text="stale", turn_id="turn_1", turn_revision=0)))
        done.set()

    thread = Thread(target=run_process)
    thread.start()
    assert not done.wait(0.05)
    candidate = tracker.begin_reopen_candidate("turn_1", 0)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate)
    assert done.wait(1.0)
    thread.join(timeout=1.0)
    assert outputs == []


def test_process_swallows_generation_errors(monkeypatch):
    handler = _new_handler()
    handler._apply_session_voice_override = lambda runtime_config, response: None

    def _boom(text, cond):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    handler._synthesize = _boom
    monkeypatch.setattr(kyutai_module.console, "print", lambda *a, **k: None)

    outputs = list(handler.process(TTSInput(text="Hello.")))
    assert outputs == []
    assert handler.should_listen.is_set() is False


# --------------------------------------------------------------------------- #
# text coalescing
# --------------------------------------------------------------------------- #


def test_coalesce_merges_same_turn_text():
    handler = _new_handler()
    handler.queue_in.put(TTSInput(text="world", turn_id="t", turn_revision=0))
    handler.queue_in.put(EndOfResponse(turn_id="t", turn_revision=0))

    text, lang, saw_end = handler._coalesce_pending_tts_input(
        TTSInput(text="hello", turn_id="t", turn_revision=0)
    )

    assert text == "hello world"
    assert saw_end is True


def test_coalesce_stops_at_different_turn():
    handler = _new_handler()
    handler.queue_in.put(TTSInput(text="other", turn_id="t2", turn_revision=0))

    text, _lang, saw_end = handler._coalesce_pending_tts_input(
        TTSInput(text="hello", turn_id="t1", turn_revision=0)
    )

    assert text == "hello"
    assert saw_end is False
    # The other-turn item is left in the queue.
    assert handler.queue_in.qsize() == 1


# --------------------------------------------------------------------------- #
# session voice override
# --------------------------------------------------------------------------- #


def test_session_voice_override_accepts_path_like():
    handler = _new_handler()
    resolved = SimpleNamespace(tag="new")
    handler._resolve_condition_attributes = lambda v: resolved
    cfg = SimpleNamespace(session=SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(voice="vctk/p225.wav"))))

    handler._apply_session_voice_override(cfg, None)

    assert handler.voice == "vctk/p225.wav"
    assert handler._default_condition_attributes is resolved


def test_session_voice_override_ignores_generic_name(caplog):
    handler = _new_handler()
    original = handler._default_condition_attributes
    handler._resolve_condition_attributes = lambda v: pytest.fail("should not resolve generic voice")
    cfg = SimpleNamespace(session=SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(voice="alloy"))))

    with caplog.at_level(logging.WARNING):
        handler._apply_session_voice_override(cfg, None)

    assert handler.voice == "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    assert handler._default_condition_attributes is original
    assert "Ignoring Kyutai-TTS session voice override" in caplog.text


def test_on_session_end_restores_initial_voice():
    handler = _new_handler()
    handler.voice = "vctk/p225.wav"
    handler._resolve_condition_attributes = lambda v: SimpleNamespace(v=v)

    handler.on_session_end()

    assert handler.voice == "expresso/ex03-ex01_happy_001_channel1_334s.wav"
