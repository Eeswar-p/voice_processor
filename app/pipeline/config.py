from dataclasses import dataclass


@dataclass
class PipelineConfig:
    sample_rate: int = 16000
    vad_frame_ms: int = 30
    vad_aggressiveness: int = 2  # 0..3

    # Target speaker match
    target_threshold: float = 0.6

    # ASR
    asr_backend: str = "whisper"
    asr_model: str = "tiny"
    transcribe_only_target: bool = True  # Only transcribe target speaker (faster)

    # Torch
    device: str = "cpu"
