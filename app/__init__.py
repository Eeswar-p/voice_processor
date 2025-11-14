"""
App package initializer.

Applies torchaudio compatibility shims early so downstream imports
don't fail on missing backend attributes in certain builds.
"""

def _patch_torchaudio_backends() -> None:
    try:
        import types
        import torchaudio  # type: ignore

        if not hasattr(torchaudio, "list_audio_backends"):
            def _list_audio_backends():
                return []
            torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]

        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda name: None  # type: ignore[attr-defined]

        backend = getattr(torchaudio, "backend", None)
        if isinstance(backend, types.ModuleType):
            if not hasattr(backend, "list_audio_backends"):
                backend.list_audio_backends = torchaudio.list_audio_backends  # type: ignore[attr-defined]
            if not hasattr(backend, "get_audio_backend"):
                backend.get_audio_backend = torchaudio.get_audio_backend  # type: ignore[attr-defined]
            if not hasattr(backend, "set_audio_backend"):
                backend.set_audio_backend = torchaudio.set_audio_backend  # type: ignore[attr-defined]
    except Exception:
        pass


_patch_torchaudio_backends()
