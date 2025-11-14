import json
from pathlib import Path

# Load the diarization results
diar_file = Path("outputs/ui_run/diarization.json")

if not diar_file.exists():
    print("❌ No diarization results found. Please run the pipeline first.")
    exit(1)

with open(diar_file, 'r', encoding='utf-8') as f:
    segments = json.load(f)

# Filter segments with text
transcribed = [s for s in segments if s['text'].strip()]

print(f"\n{'='*80}")
print(f"TRANSCRIPT - Voice to Text")
print(f"{'='*80}\n")

if not transcribed:
    print("⚠️ No transcribed text found.")
    print("\nPossible reasons:")
    print("- The audio files were processed with the old version (before ASR fix)")
    print("- Re-run the pipeline in Streamlit to get transcriptions")
    print(f"\nTotal segments: {len(segments)}")
    target_segs = [s for s in segments if s['speaker'] == 'Target']
    print(f"Target speaker segments: {len(target_segs)}")
    print(f"Other speaker segments: {len(segments) - len(target_segs)}")
else:
    print(f"Found {len(transcribed)} segments with text:\n")
    
    # Group by speaker for better readability
    current_speaker = None
    for i, seg in enumerate(transcribed, 1):
        time_str = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
        
        if seg['speaker'] != current_speaker:
            current_speaker = seg['speaker']
            print(f"\n{'─'*80}")
            print(f"🎙️ {current_speaker} Speaker")
            print(f"{'─'*80}\n")
        
        print(f"{time_str:20} {seg['text']}")
    
    print(f"\n{'='*80}")
    print(f"Total transcribed: {len(transcribed)} segments")
    print(f"{'='*80}\n")
    
    # Summary statistics
    target_text = [s for s in transcribed if s['speaker'] == 'Target']
    other_text = [s for s in transcribed if s['speaker'] == 'Other']
    
    print(f"📊 Statistics:")
    print(f"   Target speaker: {len(target_text)} segments transcribed")
    print(f"   Other speakers: {len(other_text)} segments transcribed")
    
    # Full transcript by speaker
    print(f"\n📝 Full Transcript by Speaker:\n")
    
    if target_text:
        print("TARGET SPEAKER:")
        print(" ".join([s['text'] for s in target_text]))
        print()
    
    if other_text:
        print("OTHER SPEAKERS:")
        print(" ".join([s['text'] for s in other_text]))
