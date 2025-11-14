from typing import List, Tuple, Dict

import numpy as np

from app.pipeline.embedding import get_speaker_embedding, cosine_sim


Segment = Tuple[float, float]  # (start_sec, end_sec)


def label_segments_by_similarity(
    wav: np.ndarray,
    sr: int,
    intervals: List[Segment],
    target_emb: np.ndarray,
    threshold: float = 0.6,
    device: str = "cpu",
) -> List[Dict]:
    labeled: List[Dict] = []
    for (s, e) in intervals:
        s_i = int(s * sr)
        e_i = int(e * sr)
        seg = wav[s_i:e_i]
        try:
            emb = get_speaker_embedding(seg, sr, device=device)
            score = cosine_sim(emb, target_emb)
        except Exception:
            score = 0.0
        speaker = "Target" if score >= threshold else "Other"
        labeled.append({
            "speaker": speaker,
            "start": float(s),
            "end": float(e),
            "score": float(score),
        })
    return labeled


def assemble_audio(wav: np.ndarray, sr: int, labeled: List[Dict], speaker_label: str = "Target") -> np.ndarray:
    chunks = []
    for item in labeled:
        if item.get("speaker") != speaker_label:
            continue
        s = int(item["start"] * sr)
        e = int(item["end"] * sr)
        chunks.append(wav[s:e])
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)
