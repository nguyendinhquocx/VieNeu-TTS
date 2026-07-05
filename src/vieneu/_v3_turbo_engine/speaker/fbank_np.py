"""
Torch-free Kaldi fbank front-end (numpy) for the ONNX speaker encoder.
=====================================================================
A faithful NumPy port of ``torchaudio.compliance.kaldi.fbank`` for exactly the
options the speaker encoder uses (80 mel bins, 16 kHz, povey window, pre-emphasis
0.97, remove-DC, snip-edges, power spectrum, log, per-utterance mean-norm). This
lets voice cloning run with no torch/torchaudio — only numpy + soxr (resampling).

Verified against the torch path: x-vector cosine ≈ 1.0 on real speech.
"""
from __future__ import annotations

import numpy as np

_SPEAKER_FBANK_SAMPLE_RATE = 16000
_SPEAKER_FBANK_N_MELS = 80
EPSILON = np.float64(1.1920928955078125e-07)  # numeric_limits<float>::epsilon()


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _mel_scale(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _get_mel_banks(num_bins: int, window_length_padded: int, sample_freq: float,
                   low_freq: float, high_freq: float) -> np.ndarray:
    """Triangular mel filterbank → (num_bins, num_fft_bins) (matches Kaldi/torchaudio)."""
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_freq / window_length_padded
    mel_low = _mel_scale(low_freq)
    mel_high = _mel_scale(high_freq)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)

    bin_idx = np.arange(num_bins, dtype=np.float64)[:, None]
    left_mel = mel_low + bin_idx * mel_delta
    center_mel = mel_low + (bin_idx + 1.0) * mel_delta
    right_mel = mel_low + (bin_idx + 2.0) * mel_delta

    mel = _mel_scale(fft_bin_width * np.arange(num_fft_bins, dtype=np.float64))[None, :]
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    bins = np.maximum(0.0, np.minimum(up_slope, down_slope))     # (num_bins, num_fft_bins)
    return bins


def _povey_window(window_size: int) -> np.ndarray:
    n = np.arange(window_size, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (window_size - 1))  # periodic=False
    return hann ** 0.85


def compute_fbank(waveform: np.ndarray, *, sample_rate: int, num_mel_bins: int = _SPEAKER_FBANK_N_MELS,
                  frame_length: float = 25.0, frame_shift: float = 10.0, low_freq: float = 20.0,
                  high_freq: float = 0.0, preemphasis: float = 0.97, remove_dc: bool = True,
                  mean_norm: bool = True) -> np.ndarray:
    """1-D waveform @ ``sample_rate`` → (T, num_mel_bins) log-mel fbank (Kaldi-compatible)."""
    wav = np.asarray(waveform, dtype=np.float64).reshape(-1)
    window_shift = int(sample_rate * frame_shift * 0.001)         # 160
    window_size = int(sample_rate * frame_length * 0.001)         # 400
    padded = _next_power_of_2(window_size)                        # 512
    if wav.shape[0] < window_size:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    # snip_edges framing → (m, window_size)
    m = 1 + (wav.shape[0] - window_size) // window_shift
    idx = np.arange(window_size)[None, :] + window_shift * np.arange(m)[:, None]
    frames = wav[idx]                                            # (m, window_size)

    if remove_dc:
        frames = frames - frames.mean(axis=1, keepdims=True)

    if preemphasis != 0.0:
        offset = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)  # replicate-pad left
        frames = frames - preemphasis * offset

    frames = frames * _povey_window(window_size)[None, :]

    if padded != window_size:
        frames = np.pad(frames, ((0, 0), (0, padded - window_size)))

    spectrum = np.abs(np.fft.rfft(frames, n=padded, axis=1)) ** 2.0    # (m, padded//2+1) power

    mel = _get_mel_banks(num_mel_bins, padded, sample_rate, low_freq, high_freq)  # (n_mels, padded//2)
    mel = np.pad(mel, ((0, 0), (0, 1)))                          # (n_mels, padded//2+1)
    mel_energies = spectrum @ mel.T                             # (m, n_mels)
    feat = np.log(np.maximum(mel_energies, EPSILON))

    if mean_norm:
        feat = feat - feat.mean(axis=0, keepdims=True)
    return feat.astype(np.float32)


def _resample(wav: np.ndarray, sr: int, target: int = _SPEAKER_FBANK_SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return np.asarray(wav, dtype=np.float32)
    import soxr
    return soxr.resample(np.asarray(wav, dtype=np.float32), sr, target).astype(np.float32)


def extract_speaker_fbank(waveform, *, sample_rate: int) -> np.ndarray:
    """Mono waveform (any sr) → (T, 80) mean-normed log-mel fbank @ 16 kHz, torch-free."""
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if sample_rate != _SPEAKER_FBANK_SAMPLE_RATE:
        wav = _resample(wav, sample_rate, _SPEAKER_FBANK_SAMPLE_RATE)
    return compute_fbank(wav, sample_rate=_SPEAKER_FBANK_SAMPLE_RATE, num_mel_bins=_SPEAKER_FBANK_N_MELS)
