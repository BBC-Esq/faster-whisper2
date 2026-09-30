import av
import pytest

from faster_whisper import decode_audio


@pytest.mark.parametrize("version, passes_metadata_errors", [("18.1.0", True), ("19.0.0", False)])
def test_metadata_errors_follows_pyav_version(monkeypatch, version, passes_metadata_errors):
    open_kwargs = {}

    def fake_open(*args, **kwargs):
        open_kwargs.update(kwargs)
        raise RuntimeError("stop before decoding")

    monkeypatch.setattr(av, "__version__", version)
    monkeypatch.setattr(av, "open", fake_open)

    with pytest.raises(RuntimeError, match="stop before decoding"):
        decode_audio("audio.wav")

    assert ("metadata_errors" in open_kwargs) == passes_metadata_errors
