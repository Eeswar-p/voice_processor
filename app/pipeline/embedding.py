from typing import Optional

import numpy as np

_SB_CLASSIFIER = None


def _patch_torchaudio_backends() -> None:
    """Work around torchaudio API differences across versions.
    Some environments miss list_audio_backends/get_audio_backend.
    We provide safe no-op fallbacks before loading SpeechBrain.
    """
    try:
        import torchaudio  # type: ignore

        if not hasattr(torchaudio, "list_audio_backends"):
            def _list_audio_backends():
                return []

            torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]

        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda name: None  # type: ignore[attr-defined]
    except Exception:
        # If torchaudio import itself fails, SpeechBrain may still work
        # for embedding inference; ignore and proceed.
        pass


def _to_tensor(x: np.ndarray, device: str):
    import torch

    t = torch.from_numpy(x).float().to(device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def _resolve_encoder_classifier():
    """Dynamically resolve EncoderClassifier across SpeechBrain versions.
    Tries multiple module paths and returns the class if found.
    """
    import importlib
    for mod in (
        "speechbrain.pretrained",
        "speechbrain.inference.interfaces",
        "speechbrain.pretrained.interfaces",
        "speechbrain.inference.classifier",
    ):
        try:
            m = importlib.import_module(mod)
            EC = getattr(m, "EncoderClassifier", None)
            if EC is not None:
                return EC
        except Exception:
            continue
    raise ImportError("speechbrain EncoderClassifier not found in known modules")


def get_speaker_embedding(wav: np.ndarray, sr: int, device: str = "cpu") -> np.ndarray:
    """
    Compute a single-speaker embedding for the given mono waveform using SpeechBrain
    ECAPA-TDNN. Returns a L2-normalized vector as np.ndarray.
    """
    _patch_torchaudio_backends()
    EncoderClassifier = _resolve_encoder_classifier()
    import torch
    import numpy as np

    global _SB_CLASSIFIER
    if _SB_CLASSIFIER is None:
        _SB_CLASSIFIER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}, savedir=None
        )

    wav_t = _to_tensor(wav, device)
    with torch.no_grad():
        emb = _SB_CLASSIFIER.encode_batch(wav_t)
    emb_np = emb.squeeze(0).squeeze(0).cpu().numpy()
    norm = np.linalg.norm(emb_np) + 1e-9
    return emb_np / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))
