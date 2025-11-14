import json
import soundfile as sf
import numpy as np
import whisper

# Load diarization results
with open('outputs/ui_run/diarization.json', 'r') as f:
    segments = json.load(f)

# Load the mixture audio
mixture_path = 'outputs/ui_run/_tmp/mixture.wav'
audio, sr = sf.read(mixture_path)

# Get first few Target segments that are longer than 1 second
target_segments = [s for s in segments if s['speaker'] == 'Target' and (s['end'] - s['start']) > 1.0]

print(f"Found {len(target_segments)} Target segments > 1s")

if target_segments:
    # Test first segment
    seg = target_segments[0]
    print(f"\nTesting segment: {seg['start']:.2f}-{seg['end']:.2f}s ({seg['end']-seg['start']:.2f}s)")
    
    # Extract audio segment
    start_sample = int(seg['start'] * sr)
    end_sample = int(seg['end'] * sr)
    audio_seg = audio[start_sample:end_sample]
    
    # Save test segment
    test_path = 'outputs/ui_run/test_segment.wav'
    sf.write(test_path, audio_seg, sr)
    print(f"Saved test segment to {test_path}")
    
    # Try Whisper transcription
    print("\nLoading Whisper model...")
    model = whisper.load_model("tiny")
    
    print("Transcribing...")
    result = model.transcribe(test_path)
    
    print(f"\nTranscription result:")
    print(f"Text: '{result['text']}'")
    print(f"Language: {result.get('language', 'N/A')}")
    
    if 'segments' in result:
        print(f"Whisper segments: {len(result['segments'])}")
        for i, wseg in enumerate(result['segments'][:3]):
            print(f"  {i+1}. {wseg.get('start', 0):.2f}-{wseg.get('end', 0):.2f}s: '{wseg.get('text', '')}'")
else:
    print("No Target segments longer than 1s found")
