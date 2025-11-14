import json

with open('outputs/ui_run/diarization.json', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total segments: {len(data)}')
target = [s for s in data if s['speaker'] == 'Target']
other = [s for s in data if s['speaker'] == 'Other']
print(f'Target segments: {len(target)}')
print(f'Other segments: {len(other)}')

non_empty = [s for s in data if s['text'].strip()]
print(f'Segments with transcribed text: {len(non_empty)}')

if target:
    avg_target_dur = sum(s["end"] - s["start"] for s in target) / len(target)
    total_target_time = sum(s["end"] - s["start"] for s in target)
    print(f'Average Target segment duration: {avg_target_dur:.2f}s')
    print(f'Total Target speaker time: {total_target_time:.2f}s')

if other:
    avg_other_dur = sum(s["end"] - s["start"] for s in other) / len(other)
    total_other_time = sum(s["end"] - s["start"] for s in other)
    print(f'Average Other segment duration: {avg_other_dur:.2f}s')
    print(f'Total Other speaker time: {total_other_time:.2f}s')

print('\nFirst 10 segments:')
for i, s in enumerate(data[:10]):
    print(f"{i+1}. [{s['speaker']}] {s['start']:.2f}-{s['end']:.2f}s ({s['end']-s['start']:.2f}s): '{s['text']}'")

if non_empty:
    print(f'\nSegments with text ({len(non_empty)} total):')
    for i, s in enumerate(non_empty[:20]):
        print(f"{i+1}. [{s['speaker']}] {s['start']:.2f}-{s['end']:.2f}s: {s['text']}")
else:
    print('\n⚠️ No segments contain transcribed text. All text fields are empty.')
    print('This suggests Whisper did not detect speech in the segments.')
    print('\nPossible reasons:')
    print('- Audio segments are too short for Whisper to process')
    print('- Audio quality or volume is too low')
    print('- VAD detected segments but they contain no actual speech')
