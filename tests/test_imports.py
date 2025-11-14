import pytest


def test_import_pipeline_modules():
    # Skip if numpy not available to keep CI light without installing heavy deps
    pytest.importorskip("numpy")

    # Ensure modules import without heavy deps at import time
    import app.main  # noqa: F401
    import app.pipeline.config  # noqa: F401
    import app.pipeline.embedding  # noqa: F401
    import app.pipeline.vad  # noqa: F401
    import app.pipeline.diarization  # noqa: F401
    import app.pipeline.asr  # noqa: F401
    import app.audio.io  # noqa: F401
    import app.api.server  # noqa: F401
    import app.utils.logging  # noqa: F401
