import numpy as np

from faster_whisper.vad import collect_chunks


def test_collect_chunks_segments():
    sampling_rate = 16000
    audio = np.zeros(40 * sampling_rate, dtype=np.float32)
    chunks = [
        {"start": i * 10 * sampling_rate, "end": (i + 1) * 10 * sampling_rate} for i in range(4)
    ]

    _, chunks_metadata = collect_chunks(audio, chunks, max_duration=25)

    assert [metadata["segments"] for metadata in chunks_metadata] == [chunks[:2], chunks[2:]]
