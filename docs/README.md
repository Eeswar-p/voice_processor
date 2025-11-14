Unified Neural Pipeline (Baseline)

Objective
- Target-speaker diarization and ASR for multi-speaker audio.
 Inputs: `mixture_audio.wav|.mp3` (conversation), `target_sample.wav|.mp3` (3–10s).
- Outputs: `outputs/target_speaker.wav`, `outputs/diarization.json`.

What’s Implemented (Baseline)
- VAD: `webrtcvad` to segment speech intervals.
- Target matching: SpeechBrain ECAPA-TDNN embeddings + cosine similarity.
- MP3 decoding uses `librosa` (audioread). On Windows you may need FFmpeg.
   - Windows (Chocolatey): `choco install ffmpeg -y`
   - Or install FFmpeg manually and ensure it’s on PATH.
- Diarization: label each VAD segment as `Target` or `Other` by threshold.
- ASR: Whisper (`tiny` by default) on each segment; JSON with timestamps.
- API: FastAPI endpoint for offline `/run` job.

Quickstart
1) Create/activate a Python environment, then install deps:
   ```powershell
   Push-Location "d:\Projects\AIML projects\vs  code"
   C:/Users/eshwa/AppData/Local/Programs/Python/Python313/python.exe -m pip install -r requirements.txt
   Pop-Location
   ```
2) Run CLI (writes to `outputs/`):
   ```powershell
   Push-Location "d:\Projects\AIML projects\vs  code"
   C:/Users/eshwa/AppData/Local/Programs/Python/Python313/python.exe -m app.main mixture_audio.wav target_sample.wav --out outputs --asr-model tiny
   Pop-Location
   ```
3) Run Streamlit UI:
   ```powershell
   Push-Location "d:\Projects\AIML projects\vs  code"
   C:/Users/eshwa/AppData/Local/Programs/Python/Python313/python.exe -m streamlit run app/ui/streamlit_app.py
   Pop-Location
   ```
3) Run API:
   ```powershell
   Push-Location "d:\Projects\AIML projects\vs  code"
   C:/Users/eshwa/AppData/Local/Programs/Python/Python313/python.exe -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000
   Pop-Location
   ```

Notes & Assumptions
- Whisper will download a model on first run; use `--asr-model base|small` for better accuracy.
- Embedding threshold defaults to `0.6`; tune per data.
- Resampling is a simple interpolation placeholder; for production use torchaudio or librosa.
- Real SOTA blocks (e.g., CAM++, MossFormer2, PyAnnote) can be plugged in behind current interfaces.

Outputs
- `outputs/target_speaker.wav`: concatenated segments labeled as `Target`.
- `outputs/diarization.json`: list of `{speaker, start, end, text, confidence}`.
