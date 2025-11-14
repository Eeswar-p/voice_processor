from pathlib import Path
from typing import Tuple

import numpy as np


def _resample_naive(data: np.ndarray, src_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if src_sr == target_sr:
        return data.astype(np.float32), src_sr
    import math
    ratio = target_sr / src_sr
    new_len = int(math.ceil(len(data) * ratio))
    x = np.linspace(0, 1, len(data), endpoint=False)
    xi = np.linspace(0, 1, new_len, endpoint=False)
    data = np.interp(xi, x, data).astype(np.float32)
    return data, target_sr


def load_mono_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 [-1,1] at target_sr.
    Tries soundfile for WAV/FLAC, then librosa for MP3/others, finally raw wave module.
    """
    ext = path.suffix.lower().lstrip(".")
    # 1) Try soundfile first (fast, high quality)
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=False)
        if hasattr(data, "ndim") and data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
    except Exception:
        data = None
        sr = 0

    # 2) If soundfile failed or file is MP3, try librosa (uses audioread/ffmpeg)
    if data is None or ext == "mp3":
        try:
            import librosa  # type: ignore

            y, sr2 = librosa.load(str(path), sr=None, mono=True)
            data = y.astype(np.float32)
            sr = int(sr2)
        except Exception:
            pass

    # 3) Last resort: wave (WAV PCM only)
    if data is None:
        try:
            import wave

            with wave.open(str(path), "rb") as w:
                sr = w.getframerate()
                n = w.getnframes()
                raw = w.readframes(n)
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            data = x
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {path} ({e})")

    data = data.astype(np.float32)
    data, sr = _resample_naive(data, sr, target_sr)
    return data, sr


def write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), sr)
