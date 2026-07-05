"""Speaker embedding module (ONNX, torch-free).

Provides a frozen ONNX speaker encoder that maps a reference waveform to a single global
192-d speaker embedding, used as a non-decaying global conditioning anchor for
VieNeu-TTS v3 Turbo voice cloning. The encoder is never trained — only the small
`xvec_proj` projection inside the TTS model is learned. The fbank front-end is a NumPy
port of Kaldi's fbank (``fbank_np``), so nothing here imports torch.
"""
from .onnx_extractor import OnnxSpeakerEncoder, EMBED_DIM
from .fbank_np import (
    _SPEAKER_FBANK_N_MELS,
    _SPEAKER_FBANK_SAMPLE_RATE,
    extract_speaker_fbank,
)

__all__ = [
    "OnnxSpeakerEncoder",
    "EMBED_DIM",
    "extract_speaker_fbank",
    "_SPEAKER_FBANK_N_MELS",
    "_SPEAKER_FBANK_SAMPLE_RATE",
]
