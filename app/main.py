import argparse
import json
from pathlib import Path
from typing import List, Tuple

from app.audio.io import load_mono_audio, write_wav
from app.pipeline.config import PipelineConfig
from app.pipeline.embedding import get_speaker_embedding
from app.pipeline.vad import detect_speech_intervals
from app.pipeline.diarization import label_segments_by_similarity, assemble_audio
from app.pipeline.asr import transcribe_segments
from app.utils.logging import get_logger


log = get_logger(__name__)


def run_pipeline(mixture_path: Path, target_path: Path, out_dir: Path, cfg: PipelineConfig) -> None:
    import time
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    log.info("Loading audio files...")
    wav_mix, sr_mix = load_mono_audio(mixture_path, target_sr=cfg.sample_rate)
    wav_tgt, sr_tgt = load_mono_audio(target_path, target_sr=cfg.sample_rate)
    log.info(f"Audio loaded ({time.time()-start_time:.1f}s) - Mixture: {len(wav_mix)/cfg.sample_rate:.1f}s, Target: {len(wav_tgt)/cfg.sample_rate:.1f}s")

    if sr_mix != cfg.sample_rate:
        log.warning(f"Mixture resampled to {cfg.sample_rate} Hz")
    if sr_tgt != cfg.sample_rate:
        log.warning(f"Target resampled to {cfg.sample_rate} Hz")

    step_start = time.time()
    log.info("Computing target speaker embedding...")
    tgt_emb = get_speaker_embedding(wav_tgt, cfg.sample_rate, device=cfg.device)
    log.info(f"Embedding computed ({time.time()-step_start:.1f}s)")

    step_start = time.time()
    log.info("Detecting speech intervals (VAD)...")
    intervals = detect_speech_intervals(
        wav_mix, cfg.sample_rate, frame_ms=cfg.vad_frame_ms, aggressiveness=cfg.vad_aggressiveness
    )
    if not intervals:
        log.warning("No speech detected in mixture")
    log.info(f"VAD complete ({time.time()-step_start:.1f}s) - Found {len(intervals)} intervals")

    step_start = time.time()
    log.info("Scoring segments by target similarity...")
    labeled = label_segments_by_similarity(
        wav_mix, cfg.sample_rate, intervals, tgt_emb, threshold=cfg.target_threshold, device=cfg.device
    )
    target_count = len([s for s in labeled if s["speaker"] == "Target"])
    log.info(f"Diarization complete ({time.time()-step_start:.1f}s) - {target_count} Target, {len(labeled)-target_count} Other")

    step_start = time.time()
    log.info("Assembling target speaker audio...")
    tgt_audio = assemble_audio(wav_mix, cfg.sample_rate, labeled, speaker_label="Target")
    target_out = out_dir / "target_speaker.wav"
    write_wav(target_out, tgt_audio, cfg.sample_rate)
    log.info(f"Wrote {target_out}")

    log.info("Transcribing per segment with ASR")
    # Optionally filter to only transcribe target speaker (much faster)
    segments_to_transcribe = [s for s in labeled if s["speaker"] == "Target"] if cfg.transcribe_only_target else labeled
    log.info(f"Transcribing {len(segments_to_transcribe)} segments ({len([s for s in segments_to_transcribe if s['speaker']=='Target'])} Target)")
    
    diarization_entries = transcribe_segments(
        wav_mix, cfg.sample_rate, segments_to_transcribe, backend=cfg.asr_backend, model_size=cfg.asr_model
    )
    
    # If we only transcribed target, add back Other segments with empty text
    if cfg.transcribe_only_target:
        other_segments = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": "", "confidence": 0.0} 
                         for s in labeled if s["speaker"] != "Target"]
        diarization_entries = sorted(diarization_entries + other_segments, key=lambda x: x["start"])

    diar_out = out_dir / "diarization.json"
    diar_out.write_text(json.dumps(diarization_entries, indent=2), encoding="utf-8")
    log.info(f"Wrote {diar_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Target Speaker Diarization + ASR (baseline)")
    p.add_argument("mixture", type=Path, help="Path to multi-speaker WAV file")
    p.add_argument("target", type=Path, help="Path to target speaker reference WAV (3-10s)")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    p.add_argument("--asr-backend", default="whisper", choices=["whisper"], help="ASR backend")
    p.add_argument("--asr-model", default="tiny", help="Whisper model size (e.g., tiny, base, small)")
    p.add_argument("--device", default="cpu", help="torch device: cpu or cuda")
    p.add_argument("--threshold", type=float, default=0.6, help="Target similarity threshold [0-1]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        device=args.device,
        target_threshold=args.threshold,
    )
    run_pipeline(args.mixture, args.target, args.out, cfg)


if __name__ == "__main__":
    main()
