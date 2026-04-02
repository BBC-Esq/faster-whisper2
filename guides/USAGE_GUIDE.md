# faster-whisper2 Usage Guide

## Contents

1. [Installation](#1-installation)
2. [Loading a Model](#2-loading-a-model)
3. [Basic Transcription](#3-basic-transcription)
4. [Batched Transcription](#4-batched-transcription)
5. [Word-Level Timestamps](#5-word-level-timestamps)
6. [Multilingual Transcription](#6-multilingual-transcription)
7. [VAD Filtering](#7-vad-filtering)
8. [Translation](#8-translation)
9. [Segment and Word Fields](#9-segment-and-word-fields)
10. [Model Parameters](#10-model-parameters)
11. [Transcription Parameters](#11-transcription-parameters)
12. [Available Models](#12-available-models)
13. [Logging](#13-logging)

---

## 1. Installation

```bash
pip install faster-whisper2
```

## 2. Loading a Model

```python
from faster_whisper import WhisperModel

# Load by model name (downloads automatically)
model = WhisperModel("large-v3")

# GPU with float16
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# GPU with int8 quantization (model is downloaded in the best
# source precision and converted to int8 at runtime)
model = WhisperModel("large-v3", device="cuda", compute_type="int8")

# CPU with int8
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

# Load from a local directory
model = WhisperModel("/path/to/model/directory")

# Load from a custom HuggingFace repo
model = WhisperModel("username/my-custom-whisper-ct2")
```

### Device and compute type selection

| Parameter | Options | Default |
| --- | --- | --- |
| `device` | `"auto"`, `"cpu"`, `"cuda"` | `"auto"` |
| `compute_type` | `"default"`, `"auto"`, `"float32"`, `"float16"`, `"bfloat16"`, `"int8"`, `"int8_float16"`, `"int8_float32"`, `"int8_bfloat16"`, `"int16"` | `"default"` |

Not all compute types work on all devices. See the [compatibility reference](../README.md) in the README for full details.

## 3. Basic Transcription

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

segments, info = model.transcribe("audio.mp3")

print("Detected language: %s (probability: %.2f)" % (info.language, info.language_probability))
print("Audio duration: %.2fs" % info.duration)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

> **Note:** `segments` is a generator. Transcription only starts when you iterate over it. To process everything at once:

```python
segments, info = model.transcribe("audio.mp3")
segments = list(segments)  # transcription runs here
```

### Audio input formats

The `audio` parameter accepts:

- A file path (string): `"audio.mp3"`, `"audio.wav"`, etc.
- A file-like object (binary): `open("audio.mp3", "rb")`
- A NumPy array (float32, 16kHz sample rate): pre-loaded audio waveform

## 4. Batched Transcription

Batched transcription processes multiple audio chunks in parallel for faster throughput.

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

segments, info = batched_model.transcribe("audio.mp3", batch_size=16)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

`BatchedInferencePipeline.transcribe` accepts the same parameters as `WhisperModel.transcribe`, plus `batch_size` (default 8). VAD filtering is enabled by default for batched transcription.

## 5. Word-Level Timestamps

```python
segments, info = model.transcribe("audio.mp3", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s (confidence: %.2f)" % (
            word.start, word.end, word.word, word.probability
        ))
```

Word timestamps work with both `WhisperModel` and `BatchedInferencePipeline`.

## 6. Multilingual Transcription

When `multilingual=True`, the model detects the language for each audio segment independently. Each segment includes a `language` field.

```python
segments, info = model.transcribe("audio.mp3", multilingual=True)

for segment in segments:
    print("[%.2fs -> %.2fs] (%s) %s" % (
        segment.start, segment.end, segment.language, segment.text
    ))
```

For single-language audio where you know the language, specify it directly to skip detection:

```python
segments, info = model.transcribe("audio.mp3", language="en")
```

## 7. VAD Filtering

Voice Activity Detection (VAD) uses the Silero VAD model to skip silent portions of audio before transcription. This can improve speed and accuracy.

```python
# Enable VAD with default settings
segments, info = model.transcribe("audio.mp3", vad_filter=True)

# Custom VAD parameters
segments, info = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500,
        speech_pad_ms=200,
    ),
)
```

### VAD parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `threshold` | 0.5 | Speech probability threshold |
| `min_speech_duration_ms` | 0 | Minimum speech chunk length |
| `max_speech_duration_s` | inf | Maximum speech chunk length |
| `min_silence_duration_ms` | 2000 | Silence duration to split speech |
| `speech_pad_ms` | 400 | Padding added to each side of speech |

## 8. Translation

Translate audio from any supported language to English:

```python
segments, info = model.transcribe("french_audio.mp3", task="translate")

for segment in segments:
    print(segment.text)  # English translation
```

## 9. Segment and Word Fields

### Segment

| Field | Type | Description |
| --- | --- | --- |
| `id` | int | Segment index (starting from 1) |
| `start` | float | Start time in seconds |
| `end` | float | End time in seconds |
| `text` | str | Transcribed text |
| `language` | str | Detected language code |
| `tokens` | list[int] | Token IDs |
| `avg_logprob` | float | Average log probability |
| `compression_ratio` | float | Gzip compression ratio of the text |
| `no_speech_prob` | float | Probability that the segment contains no speech |
| `words` | list[Word] or None | Word-level details (when `word_timestamps=True`) |
| `temperature` | float | Temperature used for this segment |

### Word

| Field | Type | Description |
| --- | --- | --- |
| `start` | float | Start time in seconds |
| `end` | float | End time in seconds |
| `word` | str | The word text |
| `probability` | float | Confidence score (0 to 1) |

### TranscriptionInfo

Returned as the second element of the `transcribe()` tuple:

| Field | Type | Description |
| --- | --- | --- |
| `language` | str | Detected language code |
| `language_probability` | float | Detection confidence |
| `duration` | float | Total audio duration in seconds |
| `duration_after_vad` | float | Duration after VAD filtering |
| `all_language_probs` | list or None | All detected languages with probabilities |

## 10. Model Parameters

Parameters for `WhisperModel()`:

| Parameter | Default | Description |
| --- | --- | --- |
| `model_size_or_path` | (required) | Model name, HF repo ID, or local path |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, or `"auto"` |
| `device_index` | `0` | GPU index (or list for multi-GPU) |
| `compute_type` | `"default"` | Precision/quantization for inference |
| `cpu_threads` | `0` | Number of CPU threads (0 = automatic) |
| `num_workers` | `1` | Number of workers for parallel processing |
| `download_root` | `None` | Custom cache directory for models |
| `local_files_only` | `False` | Only use cached models, don't download |

## 11. Transcription Parameters

Parameters for `model.transcribe()`:

| Parameter | Default | Description |
| --- | --- | --- |
| `audio` | (required) | File path, file-like object, or numpy array |
| `language` | `None` | Language code (auto-detected if not set) |
| `task` | `"transcribe"` | `"transcribe"` or `"translate"` |
| `beam_size` | `5` | Beam size for decoding |
| `best_of` | `5` | Candidates when sampling with non-zero temperature |
| `temperature` | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | Sampling temperature(s) with fallback |
| `vad_filter` | `False` | Enable Silero VAD to skip silence |
| `vad_parameters` | `None` | Dict or VadOptions to customize VAD |
| `word_timestamps` | `False` | Enable word-level timestamps |
| `multilingual` | `False` | Per-segment language detection |
| `initial_prompt` | `None` | Text or token IDs to condition the model |
| `hotwords` | `None` | Hint phrases to bias the model |
| `condition_on_previous_text` | `True` | Use previous output as prompt for next window |
| `no_speech_threshold` | `0.6` | Threshold for considering a segment silent |
| `compression_ratio_threshold` | `2.4` | Threshold for detecting failed decoding |
| `log_prob_threshold` | `-1.0` | Threshold for detecting failed decoding |
| `repetition_penalty` | `1` | Penalty for repeated tokens (>1 to penalize) |
| `no_repeat_ngram_size` | `0` | Prevent repeated ngrams of this size |
| `suppress_tokens` | `[-1]` | Token IDs to suppress (-1 for default set) |
| `without_timestamps` | `False` | Only sample text tokens |
| `max_new_tokens` | `None` | Maximum tokens to generate per chunk |
| `hallucination_silence_threshold` | `None` | Skip hallucinated segments in silent regions |
| `log_progress` | `False` | Show progress bar |

## 12. Available Models

| Name | Type | Languages |
| --- | --- | --- |
| `tiny` | Standard | Multilingual |
| `tiny.en` | Standard | English only |
| `base` | Standard | Multilingual |
| `base.en` | Standard | English only |
| `small` | Standard | Multilingual |
| `small.en` | Standard | English only |
| `medium` | Standard | Multilingual |
| `medium.en` | Standard | English only |
| `large-v3` | Standard | Multilingual |
| `large-v3-turbo` / `turbo` | Standard | Multilingual |
| `distil-small.en` | Distilled | English only |
| `distil-medium.en` | Distilled | English only |
| `distil-large-v3` | Distilled | Multilingual |
| `distil-large-v3.5` | Distilled | Multilingual |

Distilled models are smaller and faster while maintaining most of the accuracy of the full models. The `turbo` model offers a good balance between speed and accuracy.

You can also list available models programmatically:

```python
from faster_whisper import available_models
print(available_models())
```

## 13. Logging

```python
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
```

This shows detailed information about language detection, VAD filtering, and transcription progress.
