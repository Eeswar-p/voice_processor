from typing import List, Tuple

import numpy as np


def _frame_generator(wav: np.ndarray, sr: int, frame_ms: int):
    frame_len = int(sr * frame_ms / 1000)
    for start in range(0, len(wav), frame_len):
        end = min(start + frame_len, len(wav))
        yield start, end, wav[start:end]


def _merge_intervals(intervals: List[Tuple[float, float]], min_gap: float = 0.15) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for s, e in intervals:
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s - pe <= min_gap:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
    return merged


def _vad_webrtc(wav: np.ndarray, sr: int, frame_ms: int, aggressiveness: int) -> List[Tuple[float, float]]:
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    intervals: List[Tuple[float, float]] = []
    active = False
    seg_start = 0
    for start, end, frame in _frame_generator(wav, sr, frame_ms):
        pcm16 = np.clip(frame * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        is_speech = vad.is_speech(pcm16, sr)
        if is_speech and not active:
            active = True
            seg_start = start
        elif not is_speech and active:
            active = False
            intervals.append((seg_start / sr, end / sr))
    if active:
        intervals.append((seg_start / sr, len(wav) / sr))
    return _merge_intervals(intervals)


def _vad_silero(wav: np.ndarray, sr: int, frame_ms: int) -> List[Tuple[float, float]]:
    import torch
    # Use cached model, don't force reload
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', 
        model='silero_vad', 
        force_reload=False,
        verbose=False  # Suppress download progress
    )
    (get_speech_timestamps, _, _, _, _, _) = utils
    wav_t = torch.from_numpy(wav).float()
    if wav_t.ndim > 1:
        wav_t = wav_t.mean(dim=0)
    timestamps = get_speech_timestamps(wav_t, model, sampling_rate=sr)
    intervals = []
    for t in timestamps:
        s = t['start'] / sr
        e = t['end'] / sr
        intervals.append((float(s), float(e)))
    return _merge_intervals(intervals)


def _vad_energy(wav: np.ndarray, sr: int, frame_ms: int, rms_thresh: float = 0.01) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    active = False
    seg_start_s: float = 0.0
    for start, end, frame in _frame_generator(wav, sr, frame_ms):
        if len(frame) == 0:
            continue
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        is_speech = rms >= rms_thresh
        if is_speech and not active:
            active = True
            seg_start_s = start / sr
        elif not is_speech and active:
            active = False
            intervals.append((seg_start_s, end / sr))
    if active:
        intervals.append((seg_start_s, len(wav) / sr))
    return _merge_intervals(intervals)


def detect_speech_intervals(
    wav: np.ndarray, sr: int, frame_ms: int = 30, aggressiveness: int = 2
) -> List[Tuple[float, float]]:
    """
    VAD with graceful fallback:
    - Try webrtcvad (fast, lightweight but needs compiled wheel)
    - Fallback to Silero VAD via torch.hub (pure Python-friendly)
    Returns list of (start_sec, end_sec).
    """
    try:
        import webrtcvad  # noqa: F401
        return _vad_webrtc(wav, sr, frame_ms, aggressiveness)
    except Exception:
        try:
            return _vad_silero(wav, sr, frame_ms)
        except Exception:
            return _vad_energy(wav, sr, frame_ms)
