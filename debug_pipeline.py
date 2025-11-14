import logging
import json
from pathlib import Path

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from app.main import run_pipeline
from app.pipeline.config import PipelineConfig

# Create config
cfg = PipelineConfig(
    target_threshold=0.6,
    asr_model="tiny"
)

# Run pipeline with logging
run_pipeline(
    mixture_path=Path('outputs/ui_run/_tmp/mixture.wav'),
    target_path=Path('outputs/ui_run/_tmp/target.wav'),
    out_dir=Path('outputs/debug_run'),
    cfg=cfg
)

print(f"\n{'='*60}")
print("PIPELINE COMPLETED")
print(f"{'='*60}")

# Load and check results
with open('outputs/debug_run/diarization.json', 'r') as f:
    segments = json.load(f)

print(f"\nTotal segments: {len(segments)}")
with_text = [s for s in segments if s['text'].strip()]
print(f"Segments with text: {len(with_text)}")

if with_text:
    print("\nTranscribed segments:")
    for i, seg in enumerate(with_text[:10]):
        print(f"{i+1}. [{seg['speaker']}] {seg['start']:.2f}-{seg['end']:.2f}s: {seg['text']}")
else:
    print("\n⚠️ NO TRANSCRIPTIONS FOUND")
