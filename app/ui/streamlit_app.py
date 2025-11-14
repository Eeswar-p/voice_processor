import io
import json
from pathlib import Path

import streamlit as st

from app.pipeline.config import PipelineConfig
from app.main import run_pipeline


st.set_page_config(page_title="Voice Processor", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice Processor")
st.caption("AI-powered speaker diarization and voice-to-text transcription")


@st.cache_resource(show_spinner="Preloading models...")
def preload_models():
    """Preload models to speed up first run"""
    import torch
    # Preload Whisper
    import whisper
    whisper.load_model("tiny")
    # Preload Silero VAD
    torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    return True


def save_uploaded_file(uploaded, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = uploaded.read()
    path.write_bytes(data)
    return path


with st.sidebar:
    st.header("Settings")
    asr_backend = st.selectbox("ASR backend", ["whisper"], index=0)
    asr_model = st.selectbox("Whisper model", ["tiny", "base", "small"], index=0)
    threshold = st.slider("Target similarity threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    transcribe_only_target = st.checkbox("Transcribe only Target speaker (faster)", value=True)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    out_dir = Path("outputs/ui_run")
    st.text(f"Output dir: {out_dir}")
    
    st.info("⏱️ **Performance Tip**: Transcription processes ~1 segment/second on CPU. Segments < 0.5s are skipped automatically.", icon="ℹ️")

# Preload models on first access
preload_models()

st.markdown("### 📤 Upload Audio Files")
mixture = st.file_uploader("Multi-speaker audio (WAV/MP3)", type=["wav", "mp3"], accept_multiple_files=False, help="Upload the audio file with multiple speakers")
target = st.file_uploader("Target speaker sample (3-10 seconds)", type=["wav", "mp3"], accept_multiple_files=False, help="Upload a short sample of the target speaker's voice")

run_clicked = st.button("Run Pipeline", type="primary", disabled=not (mixture and target))

if run_clicked and mixture and target:
    status = st.status("Running pipeline...", expanded=True)
    tmp = out_dir / "_tmp"
    mix_path = save_uploaded_file(mixture, tmp / "mixture.wav")
    tgt_path = save_uploaded_file(target, tmp / "target.wav")

    cfg = PipelineConfig(
        asr_backend=asr_backend,
        asr_model=asr_model,
        device=device,
        target_threshold=threshold,
        transcribe_only_target=transcribe_only_target,
    )
    try:
        status.update(label="Loading audio...", state="running")
        run_pipeline(mix_path, tgt_path, out_dir, cfg)
        status.update(label="✅ Pipeline complete!", state="complete")
        st.success("Done! Scroll down to see results.")
    except Exception as e:
        status.update(label="❌ Pipeline failed", state="error")
        st.error(f"Pipeline failed: {e}")

if out_dir.exists():
    diar_json = out_dir / "diarization.json"
    if diar_json.exists():
        try:
            data = json.loads(diar_json.read_text(encoding="utf-8"))
        except Exception:
            data = []
        
        # Filter segments with transcribed text
        transcribed = [s for s in data if s.get('text', '').strip()]
        target_segments = [s for s in transcribed if s.get('speaker') == 'Target']
        other_segments = [s for s in transcribed if s.get('speaker') == 'Other']
        
        # Analysis Section
        st.header("📊 Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Segments", len(data))
        with col2:
            st.metric("Transcribed", len(transcribed))
        with col3:
            st.metric("Target Speaker", len(target_segments))
        with col4:
            st.metric("Other Speakers", len(other_segments))
        
        # Transcript Section
        if transcribed:
            st.header("📝 Transcript")
            
            # Show full transcript by speaker
            if target_segments:
                with st.expander("🎙️ Target Speaker Transcript", expanded=True):
                    for seg in target_segments:
                        time_str = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
                        st.markdown(f"**{time_str}** {seg['text']}")
                
                st.subheader("📄 Complete Transcript")
                full_target_text = " ".join([s['text'] for s in target_segments])
                st.text_area("Target Speaker Full Text", full_target_text, height=200, label_visibility="collapsed")
            
            if other_segments:
                with st.expander("👥 Other Speakers Transcript"):
                    for seg in other_segments:
                        time_str = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
                        st.markdown(f"**{time_str}** {seg['text']}")
        else:
            st.warning("No transcribed text found. The segments may be too short or contain no speech.")
        
        # Download section
        st.markdown("---")
        st.subheader("📥 Downloads")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📄 Download Transcript (JSON)", 
                diar_json.read_bytes(), 
                file_name="diarization.json",
                mime="application/json"
            )
        with col2:
            tgt_audio = out_dir / "target_speaker.wav"
            if tgt_audio.exists():
                st.download_button(
                    "🔊 Download Target Audio (WAV)", 
                    tgt_audio.read_bytes(), 
                    file_name="target_speaker.wav",
                    mime="audio/wav"
                )

st.markdown("---")
st.caption("🚀 Voice Processor v1.0 | Powered by SpeechBrain, Whisper & Streamlit")
