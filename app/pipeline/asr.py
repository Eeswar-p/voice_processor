from typing import List, Dict

import numpy as np

_WHISPER_MODELS = {}


def _get_whisper_model(model_size: str):
    import whisper

    if model_size not in _WHISPER_MODELS:
        _WHISPER_MODELS[model_size] = whisper.load_model(model_size)
    return _WHISPER_MODELS[model_size]


def _asr_whisper_segment(wav_seg: np.ndarray, sr: int, model_size: str = "tiny") -> Dict:
    import tempfile
    import numpy as np
    import soundfile as sf
    import os
    
    model = _get_whisper_model(model_size)
    
    # Ensure mono audio
    if wav_seg.ndim > 1:
        wav_seg = wav_seg.mean(axis=1)
    
    # Ensure float32
    wav_seg = wav_seg.astype(np.float32)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        import math
        ratio = 16000 / sr
        new_len = int(math.ceil(len(wav_seg) * ratio))
        x = np.linspace(0, 1, len(wav_seg), endpoint=False)
        xi = np.linspace(0, 1, new_len, endpoint=False)
        wav_seg = np.interp(xi, x, wav_seg).astype(np.float32)
        sr = 16000

    # Create temp file with delete=False to keep it available for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, wav_seg, sr)
    
    try:
        result = model.transcribe(tmp_path)
        text = result.get("text", "").strip()
    finally:
        # Clean up temp file after transcription
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    return {"text": text, "confidence": 0.0}


def transcribe_segments(
    wav: np.ndarray,
    sr: int,
    labeled: List[Dict],
    backend: str = "whisper",
    model_size: str = "tiny",
    min_duration: float = 0.5,
) -> List[Dict]:
    import logging
    logger = logging.getLogger(__name__)
    
    entries: List[Dict] = []
    total = len(labeled)
    
    for idx, item in enumerate(labeled, 1):
        duration = item["end"] - item["start"]
        
        # Skip very short segments to save time
        if duration < min_duration:
            entries.append({
                "speaker": item.get("speaker", "Unknown"),
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": "",
                "confidence": 0.0,
            })
            continue
        
        s = int(item["start"] * sr)
        e = int(item["end"] * sr)
        seg = wav[s:e].astype(np.float32)
        text = ""
        conf = 0.0
        
        if backend == "whisper":
            try:
                if idx % 10 == 0:
                    logger.info(f"Transcribing segment {idx}/{total} ({duration:.1f}s)")
                r = _asr_whisper_segment(seg, sr, model_size=model_size)
                text = r.get("text", "")
                conf = float(r.get("confidence", 0.0))
            except Exception as ex:
                logger.warning(f"ASR failed for segment {item['start']:.2f}-{item['end']:.2f}s: {ex}")
                text = ""
                conf = 0.0
        
        entries.append(
            {
                "speaker": item.get("speaker", "Unknown"),
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": text,
                "confidence": conf,
            }
        )
    
    logger.info(f"Transcribed {len([e for e in entries if e['text']])} segments with text")
    return entries
