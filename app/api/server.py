from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from app.pipeline.config import PipelineConfig
from app.main import run_pipeline

app = FastAPI(title="Target Speaker Diarization + ASR (baseline)")


class RunResponse(BaseModel):
    target_audio: str
    diarization_json: str


@app.post("/run", response_model=RunResponse)
async def run(
    mixture: UploadFile = File(...),
    target: UploadFile = File(...),
    out_dir: Optional[str] = Form(None),
    asr_backend: str = Form("whisper"),
    asr_model: str = Form("tiny"),
    device: str = Form("cpu"),
    threshold: float = Form(0.6),
):
    """Run offline pipeline on uploaded files."""
    out = Path(out_dir or "outputs")
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    mix_path = tmp / "mixture.wav"
    tgt_path = tmp / "target.wav"
    mix_path.write_bytes(await mixture.read())
    tgt_path.write_bytes(await target.read())

    cfg = PipelineConfig(
        asr_backend=asr_backend,
        asr_model=asr_model,
        device=device,
        target_threshold=threshold,
    )
    run_pipeline(mix_path, tgt_path, out, cfg)
    return RunResponse(
        target_audio=str(out / "target_speaker.wav"),
        diarization_json=str(out / "diarization.json"),
    )
