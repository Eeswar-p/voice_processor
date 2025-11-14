Design Overview

Pipeline Stages (Baseline)
- Ingest: load mono 16kHz audio for mixture and target.
- VAD: segment mixture into speech intervals (webrtcvad, 30 ms frames, agg=2).
- Embeddings: SpeechBrain ECAPA-TDNN vectors for target and each segment.
- Scoring: cosine similarity; label `Target` when score >= threshold (0.6 default).
- ASR: Whisper per segment; aggregate into JSON with timestamps and labels.
- Output: assemble target-only audio by concatenating `Target` segments.

Why This Design
- Modular: each stage is a small file with focused responsibility.
- Replaceable: plug in SOTA blocks (PyAnnote, MossFormer2, Paraformer) later.
- Practical: runs locally with accessible open-source deps.

Future Extensions
- Noise suppression (e.g., UVR-MDX-Net), overlap detection, better diarization.
- Real-time streaming via WebSocket with buffered VAD/ASR.
- Punctuation restoration (deepmultilingualpunctuation) and language tags.
