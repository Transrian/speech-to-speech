"""Real end-to-end tests for the Kyutai TTS handler.

These load the actual ``kyutai/tts-1.6b-en_fr`` model through ``moshi`` and
synthesize a short utterance, verifying the handler produces audible, correctly
shaped int16 audio blocks at the 16kHz pipeline rate on both CPU and CUDA.

They are heavy (they download several GB the first time and run real
inference), so they are skipped unless ``RUN_KYUTAI_TTS_E2E=1`` is set:

    RUN_KYUTAI_TTS_E2E=1 pytest tests/test_kyutai_tts_e2e.py -v -s
"""

from __future__ import annotations

import os
from threading import Event

import numpy as np
import pytest

RUN_E2E = os.environ.get("RUN_KYUTAI_TTS_E2E") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="Set RUN_KYUTAI_TTS_E2E=1 to run the real Kyutai TTS end-to-end tests (downloads the model).",
)

moshi = pytest.importorskip("moshi", reason="moshi is required for Kyutai TTS e2e tests")
import torch  # noqa: E402

from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput  # noqa: E402
from speech_to_speech.TTS.kyutai_tts_handler import KyutaiTTSHandler  # noqa: E402

_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")


def _make_handler(device: str) -> KyutaiTTSHandler:
    handler = KyutaiTTSHandler(
        Event(),  # stop_event (BaseHandler positional)
        queue_in=None,
        queue_out=None,
        setup_args=(Event(),),  # should_listen
        setup_kwargs={"device": device, "blocksize": 512},
    )
    return handler


@pytest.fixture(scope="module", params=_DEVICES)
def handler(request):
    h = _make_handler(request.param)
    yield h
    try:
        h.cleanup()
    except Exception:
        pass


def test_synthesize_produces_audible_int16_blocks(handler):
    blocks = list(handler._synthesize("Hello, this is a real test.", handler._default_condition_attributes))

    assert len(blocks) > 0, "expected at least one audio block"
    for b in blocks:
        assert isinstance(b, np.ndarray)
        assert b.dtype == np.int16
        assert len(b) == handler.blocksize

    audio = np.concatenate(blocks).astype(np.float32)
    peak = np.abs(audio).max()
    assert peak > 500, f"audio should be audible, got peak={peak}"

    # At least ~0.3s of audio at 16kHz for this sentence.
    assert len(audio) >= 16000 * 0.3


def test_process_streams_then_end_of_response(handler):
    outputs = list(handler.process(TTSInput(text="Bonjour, ceci est un test.")))
    assert len(outputs) > 0
    assert all(isinstance(o, np.ndarray) and o.dtype == np.int16 for o in outputs)
    # process() must not flip should_listen; the streamer does that via AUDIO_RESPONSE_DONE.
    assert handler.should_listen.is_set() is False

    end_outputs = list(handler.process(EndOfResponse()))
    assert end_outputs == [AUDIO_RESPONSE_DONE]
