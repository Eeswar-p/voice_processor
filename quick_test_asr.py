import soundfile as sf
import numpy as np
from app.pipeline.asr import _asr_whisper_segment

# Load a test audio segment
audio, sr = sf.read('outputs/ui_run/_tmp/mixture.wav')

# Take a 3-second segment from the middle
start = int(sr * 10)
end = int(sr * 13)
segment = audio[start:end]

print(f"Testing ASR on {len(segment)/sr:.2f}s segment...")
print(f"Sample rate: {sr} Hz")

# Test Whisper
result = _asr_whisper_segment(segment, sr, model_size="tiny")

print(f"\n✓ ASR Result:")
print(f"  Text: '{result['text']}'")
print(f"  Confidence: {result['confidence']}")

if result['text'].strip():
    print("\n✅ SUCCESS: Whisper is working!")
else:
    print("\n⚠️ WARNING: No text transcribed (might be silence)")
