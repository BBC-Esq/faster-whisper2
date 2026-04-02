# Faster Whisper 2 - transcription with CTranslate2

**faster-whisper2** is a reimplementation of OpenAI's Whisper model using [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which is a fast inference engine for Transformer models.

This implementation is up to 4 times faster than [openai/whisper](https://github.com/openai/whisper) for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.

## Requirements

* Python 3.9 or greater

Unlike openai-whisper, FFmpeg does **not** need to be installed on the system. The audio is decoded with the Python library [PyAV](https://github.com/PyAV-Org/PyAV) which bundles the FFmpeg libraries in its package.

### GPU

GPU execution requires the following NVIDIA libraries to be installed:

* [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
* [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn)

**Note**: The latest versions of `ctranslate2` only support CUDA 12 and cuDNN 9. For CUDA 11 and cuDNN 8, the current workaround is downgrading to the `3.24.0` version of `ctranslate2`, for CUDA 12 and cuDNN 8, downgrade to the `4.4.0` version of `ctranslate2`, (This can be done with `pip install --force-reinstall ctranslate2==4.4.0` or specifying the version in a `requirements.txt`).

There are multiple ways to install the NVIDIA libraries mentioned above. The recommended way is described in the official NVIDIA documentation, but we also suggest other installation methods below.

<details>
<summary>Other installation methods (click to expand)</summary>


**Note:** For all these methods below, keep in mind the above note regarding CUDA versions. Depending on your setup, you may need to install the _CUDA 11_ versions of libraries that correspond to the CUDA 12 libraries listed in the instructions below.

#### Use Docker

The libraries (cuBLAS, cuDNN) are installed in this official NVIDIA CUDA Docker images: `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`.

#### Install with `pip` (Linux only)

On Linux these libraries can be installed with `pip`. Note that `LD_LIBRARY_PATH` must be set before launching Python.

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```


</details>

## Installation

The module can be installed from [PyPI](https://pypi.org/project/faster-whisper2/):

```bash
pip install faster-whisper2
```

<details>
<summary>Other installation methods (click to expand)</summary>

### Install the master branch

```bash
pip install --force-reinstall "faster-whisper2 @ https://github.com/BBC-Esq/faster-whisper2/archive/refs/heads/master.tar.gz"
```

### Install a specific commit

```bash
pip install --force-reinstall "faster-whisper2 @ https://github.com/BBC-Esq/faster-whisper2/archive/a4f1cc8f11433e454c3934442b5e1a4ed5e865c3.tar.gz"
```

</details>

<details>
<summary>Community integrations (click to expand)</summary>

Here is a non exhaustive list of open-source projects using faster-whisper. Feel free to add your project to the list!

* [speaches](https://github.com/speaches-ai/speaches) is an OpenAI compatible server using `faster-whisper`. It's easily deployable with Docker, works with OpenAI SDKs/CLI, supports streaming, and live transcription.
* [WhisperX](https://github.com/m-bain/whisperX) is an award-winning Python library that offers speaker diarization and accurate word-level timestamps using wav2vec2 alignment
* [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) is a command line client based on faster-whisper and compatible with the original client from openai/whisper.
* [whisper-diarize](https://github.com/MahmoudAshraf97/whisper-diarization) is a speaker diarization tool that is based on faster-whisper and NVIDIA NeMo.
* [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) Standalone CLI executables of faster-whisper for Windows, Linux & macOS.
* [asr-sd-pipeline](https://github.com/hedrergudene/asr-sd-pipeline) provides a scalable, modular, end to end multi-speaker speech to text solution implemented using AzureML pipelines.
* [Open-Lyrics](https://github.com/zh-plus/Open-Lyrics) is a Python library that transcribes voice files using faster-whisper, and translates/polishes the resulting text into `.lrc` files in the desired language using OpenAI-GPT.
* [wscribe](https://github.com/geekodour/wscribe) is a flexible transcript generation tool supporting faster-whisper, it can export word level transcript and the exported transcript then can be edited with [wscribe-editor](https://github.com/geekodour/wscribe-editor)
* [aTrain](https://github.com/BANDAS-Center/aTrain) is a graphical user interface implementation of faster-whisper developed at the BANDAS-Center at the University of Graz for transcription and diarization in Windows ([Windows Store App](https://apps.microsoft.com/detail/atrain/9N15Q44SZNS2)) and Linux.
* [Whisper-Streaming](https://github.com/ufal/whisper_streaming) implements real-time mode for offline Whisper-like speech-to-text models with faster-whisper as the most recommended back-end. It implements a streaming policy with self-adaptive latency based on the actual source complexity, and demonstrates the state of the art.
* [WhisperLive](https://github.com/collabora/WhisperLive) is a nearly-live implementation of OpenAI's Whisper which uses faster-whisper as the backend to transcribe audio in real-time.
* [Faster-Whisper-Transcriber](https://github.com/BBC-Esq/ctranslate2-faster-whisper-transcriber) is a simple but reliable voice transcriber that provides a user-friendly interface.
* [Open-dubbing](https://github.com/softcatala/open-dubbing) is open dubbing is an AI dubbing system which uses machine learning models to automatically translate and synchronize audio dialogue into different languages.
* [Whisper-FastAPI](https://github.com/heimoshuiyu/whisper-fastapi) whisper-fastapi is a very simple script that provides an API backend compatible with OpenAI, HomeAssistant, and Konele (Android voice typing) formats.

</details>

<details>
<summary>Compute type compatibility reference (click to expand)</summary>

When you specify a `compute_type`, faster-whisper2 validates that it is compatible with your device before downloading anything. If an incompatible combination is detected, a clear error is raised. For quantized types (e.g. `int8`), the library automatically downloads the best source precision for your hardware and CTranslate2 handles the runtime conversion.

### Supported compute types by device

| Compute Type | CPU | CUDA >= 6.1 | CUDA >= 7.0 | CUDA >= 8.0 |
| --- | --- | --- | --- | --- |
| `float32` | Yes | Yes | Yes | Yes |
| `float16` | No | No | Yes | Yes |
| `bfloat16` | No | No | No | Yes |
| `int8` | Yes | Yes | Yes | Yes |
| `int8_float16` | No | No | Yes | Yes |
| `int8_float32` | Yes | Yes | Yes | Yes |
| `int8_bfloat16` | No | No | No | Yes |
| `int16` | Yes (Intel MKL only) | No | No | No |
| `auto` | Yes | Yes | Yes | Yes |
| `default` | Yes | Yes | Yes | Yes |

### Which model precision is downloaded

| Device | Requested Compute Type | Downloaded Precision |
| --- | --- | --- |
| CPU | Any | float32 |
| CUDA (any) | `float32` | float32 |
| CUDA (any) | `float16` | float16 |
| CUDA (any) | `bfloat16` | bfloat16 |
| CUDA >= 8.0 (Ampere+) | `int8`, `int8_float16`, `int8_float32`, `int8_bfloat16`, `auto`, `default` | bfloat16 |
| CUDA < 8.0 (pre-Ampere) | `int8`, `int8_float16`, `int8_float32`, `int8_bfloat16`, `auto`, `default` | float16 |

### CUDA compute capability by GPU generation

| Generation | Compute Capability | Example GPUs |
| --- | --- | --- |
| Maxwell | 5.x | GTX 950, GTX 970, GTX 980 |
| Pascal | 6.x | GTX 1060, GTX 1070, GTX 1080 |
| Turing | 7.x | RTX 2060, RTX 2070, RTX 2080 |
| Ampere | 8.x | RTX 3060, RTX 3070, RTX 3080, A100 |
| Ada Lovelace | 8.9 | RTX 4060, RTX 4070, RTX 4080, RTX 4090 |
| Hopper | 9.0 | H100 |
| Blackwell | 10.0 | RTX 5070, RTX 5080, RTX 5090, B200 |

</details>

## Contributing

Pull requests are welcome! Before submitting, make sure your code passes the following checks. You can run them locally to catch issues early:

### 1. Black (code formatting)

[Black](https://github.com/psf/black) enforces a consistent code style. It handles things like indentation, trailing commas, quote style, and line wrapping. The project uses a max line length of 100 characters.

```bash
black --check .     # check for issues
black .             # auto-fix formatting
```

### 2. isort (import sorting)

[isort](https://github.com/PyCQA/isort) ensures imports are grouped (stdlib, third-party, local) and alphabetically sorted. It's configured to be compatible with Black.

```bash
isort --check-only .   # check for issues
isort .                # auto-fix import order
```

### 3. Flake8 (linting)

[Flake8](https://github.com/PyCQA/flake8) catches common issues like unused imports, undefined names, and style violations. Max line length is 100 characters.

```bash
flake8 .
```

### 4. Pytest (tests)

The test suite must pass. Some tests require model downloads and may take a while on the first run.

```bash
pytest -v tests/
```

### Quick setup

Install the dev dependencies and run all checks:

```bash
pip install -e ".[dev]"
black . && isort . && flake8 . && pytest -v tests/
```
