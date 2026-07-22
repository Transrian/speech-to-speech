from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KyutaiTTSHandlerArguments:
    kyutai_tts_model_repo: str = field(
        default="kyutai/tts-1.6b-en_fr",
        metadata={
            "help": "HuggingFace repo for the Kyutai DSM TTS model. Default is 'kyutai/tts-1.6b-en_fr' "
            "(English + French). The model is served through the 'moshi' library."
        },
    )
    kyutai_tts_voice_repo: str = field(
        default="kyutai/tts-voices",
        metadata={
            "help": "HuggingFace repo containing the voice embeddings. Default is 'kyutai/tts-voices'."
        },
    )
    kyutai_tts_voice: str = field(
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        metadata={
            "help": "Voice to use, given as a path within the voice repo (e.g. "
            "'expresso/ex03-ex01_happy_001_channel1_334s.wav'). Can also be a 'hf://REPO/PATH' reference or a "
            "local '.safetensors' voice-embedding file. Default is the Expresso 'happy' voice."
        },
    )
    kyutai_tts_device: str = field(
        default="cuda",
        metadata={"help": "Device to run the model on. Options: 'cuda', 'cpu', 'mps'. Default is 'cuda'."},
    )
    kyutai_tts_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Compute dtype. Options: 'bfloat16', 'float16', 'float32'. Default is None, which selects "
            "'bfloat16' on CUDA/MPS and 'float32' on CPU."
        },
    )
    kyutai_tts_n_q: int = field(
        default=32,
        metadata={"help": "Number of mimi codec codebooks to generate. Default is 32."},
    )
    kyutai_tts_temp: float = field(
        default=0.6,
        metadata={"help": "Sampling temperature for generation. Default is 0.6."},
    )
    kyutai_tts_cfg_coef: float = field(
        default=2.0,
        metadata={
            "help": "Classifier-free guidance coefficient (the model was trained with CFG distillation). Higher "
            "values follow the voice/text conditioning more strictly at some cost to audio quality. Valid values "
            "are typically 1.0-4.0 in 0.5 increments. Default is 2.0."
        },
    )
    kyutai_tts_padding_between: int = field(
        default=1,
        metadata={"help": "Padding (in codec frames) inserted between script words. Default is 1."},
    )
    kyutai_tts_blocksize: int = field(
        default=512,
        metadata={
            "help": "Size of audio blocks to yield for streaming. Must match the audio streamer blocksize. "
            "Default is 512."
        },
    )
