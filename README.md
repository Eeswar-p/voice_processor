# 🎙️ Voice Processor

**AI-powered Speaker Diarization and Voice-to-Text Transcription System**

> Built with SpeechBrain, OpenAI Whisper, and Streamlit

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Technical Architecture](#technical-architecture)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Technologies Used](#technologies-used)

---

## 🎯 Overview

Voice Processor is an end-to-end AI system that performs **speaker diarization** (identifying "who spoke when") and **automatic speech recognition** (converting speech to text). The system can isolate a specific target speaker from multi-speaker audio and generate accurate transcripts.

### Use Cases
- Meeting transcription with speaker identification
- Podcast/interview processing
- Voice command extraction from conversations
- Multi-speaker audio analysis

---

## ✨ Key Features

1. **Target Speaker Isolation** - Identify and extract a specific speaker's voice from group conversations
2. **Voice Activity Detection (VAD)** - Multi-tier fallback system (Silero AI → Energy-based)
3. **Speaker Embeddings** - Deep learning-based speaker recognition using ECAPA-TDNN
4. **Speech-to-Text** - State-of-the-art Whisper ASR for accurate transcription
5. **Multi-Format Support** - Handles WAV and MP3 audio files
6. **Web Interface** - User-friendly Streamlit UI with real-time processing
7. **Performance Optimized** - Model caching, smart segment filtering, parallel processing

---

## 📁 Project Structure

```
voice-processor/
│
├── app/                          # Core Application
│   ├── __init__.py              # Package init with compatibility patches
│   ├── main.py                  # Pipeline orchestrator (CLI entry point)
│   │
│   ├── audio/                   # Audio Processing Module
│   │   ├── __init__.py
│   │   └── io.py               # Audio loading/saving (WAV/MP3)
│   │
│   ├── pipeline/               # ML Pipeline Components
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration dataclass
│   │   ├── vad.py             # Voice Activity Detection
│   │   ├── embedding.py       # Speaker embedding extraction
│   │   ├── diarization.py     # Speaker diarization logic
│   │   └── asr.py             # Automatic Speech Recognition
│   │
│   ├── ui/                     # User Interface
│   │   ├── __init__.py
│   │   └── streamlit_app.py   # Streamlit web application
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       └── logging.py          # Logging configuration
│
├── tests/                       # Unit Tests
│   ├── __init__.py
│   └── test_audio_io.py
│
├── docs/                        # Documentation
│   └── Awe_hackaton.pdf       # Project requirements
│
├── outputs/                     # Generated Outputs (gitignored)
│   └── ui_run/
│       ├── diarization.json    # Transcription with timestamps
│       ├── target_speaker.wav  # Isolated target audio
│       └── _tmp/               # Temporary files
│
├── requirements.txt             # Python dependencies
└── README.md                   # Project documentation
```

---

## 🏗️ Technical Architecture

### Pipeline Components

```
┌─────────────────┐
│  Audio Input    │ (Mixture + Target Sample)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Loading   │ → Supports WAV/MP3, auto-resampling to 16kHz
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │ → SpeechBrain ECAPA-TDNN (target speaker)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      VAD        │ → Silero (PyTorch) / Energy-based fallback
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Diarization    │ → Cosine similarity-based speaker matching
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      ASR        │ → OpenAI Whisper (tiny/base/small)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ JSON + Audio    │ → Transcript + Isolated speaker audio
└─────────────────┘
```

### Algorithm Details

1. **VAD (Voice Activity Detection)**
   - Primary: Silero VAD (PyTorch Hub)
   - Fallback: Energy-based RMS threshold
   - Output: Speech intervals [(start, end), ...]

2. **Speaker Embedding**
   - Model: SpeechBrain ECAPA-TDNN
   - Pretrained: VoxCeleb dataset
   - Output: 192-dimensional L2-normalized vector

3. **Diarization**
   - Algorithm: Cosine similarity scoring
   - Threshold: 0.6 (configurable)
   - Labels: "Target" vs "Other"

4. **ASR (Automatic Speech Recognition)**
   - Model: OpenAI Whisper
   - Modes: tiny (fast), base, small (accurate)
   - Optimization: Skip segments < 0.5s

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.13+
- 4GB+ RAM
- FFmpeg (for MP3 support)

### Install Dependencies

```powershell
pip install -r requirements.txt
```

### First Run (downloads models ~500MB)
Models are cached automatically on first use:
- Whisper: ~/.cache/whisper/
- SpeechBrain: ~/.cache/huggingface/
- Silero VAD: ~/.cache/torch/hub/

---

## 💻 Usage

### Web Interface (Recommended)

```powershell
streamlit run app/ui/streamlit_app.py
```

Then open http://localhost:8501

**Steps:**
1. Upload multi-speaker audio file (WAV/MP3)
2. Upload target speaker sample (3-10 seconds)
3. Adjust settings (threshold, model size)
4. Click "Run Pipeline"
5. View transcript with analysis metrics

### Command Line Interface

```powershell
python -m app.main mixture.wav target.wav --out outputs/
```

**Options:**
```
--asr-model {tiny,base,small}  Whisper model (default: tiny)
--threshold FLOAT              Similarity threshold 0-1 (default: 0.6)
--device {cpu,cuda}            Processing device
```

---

## 🔄 Pipeline Workflow

### Step 1: Audio Loading
- Load mixture audio (multi-speaker)
- Load target speaker sample
- Resample to 16kHz mono

### Step 2: Embedding Extraction
- Process target sample through ECAPA-TDNN
- Generate 192-dim speaker embedding
- Normalize with L2 norm

### Step 3: VAD
- Detect speech segments in mixture
- Filter non-speech regions
- Output intervals with timestamps

### Step 4: Diarization
- For each speech segment:
  - Extract embedding
  - Compute cosine similarity with target
  - Label as "Target" or "Other" based on threshold

### Step 5: ASR
- Transcribe each segment with Whisper
- Optional: Process only target speaker (faster)
- Skip segments < 0.5s for performance

### Step 6: Output Generation
- **diarization.json**: Structured transcript
  ```json
  [{
    "speaker": "Target",
    "start": 10.1,
    "end": 11.9,
    "text": "transcribed text here",
    "confidence": 0.0
  }]
  ```
- **target_speaker.wav**: Concatenated target audio

---

## 🛠️ Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| Deep Learning | PyTorch 2.9 | Neural network framework |
| Speaker Recognition | SpeechBrain 1.0 | ECAPA-TDNN embeddings |
| Speech Recognition | OpenAI Whisper | Voice-to-text |
| VAD | Silero VAD | Speech detection |
| Audio I/O | soundfile, librosa | File handling |
| Web UI | Streamlit 1.50 | Interactive interface |
| Language | Python 3.13 | Core implementation |

---

## 📊 Performance Metrics

- **VAD Accuracy**: 3-5 seconds for 15-min audio
- **Embedding Speed**: ~30-50s per target sample
- **ASR Speed**: ~1 segment/second (CPU, tiny model)
- **Total Processing**: ~2-5 min for 15-min audio (target-only mode)

### Optimization Features
✅ Model caching (no reloading)  
✅ Skip short segments (< 0.5s)  
✅ Target-only transcription mode  
✅ Efficient audio resampling  

---

## 🎓 Interview Talking Points

1. **Problem Solved**: Multi-speaker audio transcription with speaker identification
2. **ML Models Used**:
   - ECAPA-TDNN for speaker embeddings
   - Silero VAD for speech detection
   - Whisper for transcription
3. **Key Challenges**:
   - Model compatibility across SpeechBrain versions
   - Performance optimization for real-time processing
   - Audio format handling (WAV/MP3)
4. **Technical Decisions**:
   - Cosine similarity for speaker matching
   - Multi-tier VAD fallback
   - Modular pipeline design
5. **Results**: Accurate speaker isolation + transcription with timestamps

---

## 📄 License

Educational/Hackathon Project

---

## 🙏 Credits

- **SpeechBrain** - Speaker recognition models
- **OpenAI** - Whisper ASR
- **Silero Team** - VAD model
- **Streamlit** - UI framework
