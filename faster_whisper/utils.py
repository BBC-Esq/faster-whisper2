import logging
import os
import re

from typing import List, Optional, Set, Union

import ctranslate2
import huggingface_hub

from tqdm.auto import tqdm

_MODELS = {
    "tiny.en": "whisper-tiny.en",
    "tiny": "whisper-tiny",
    "base.en": "whisper-base.en",
    "base": "whisper-base",
    "small.en": "whisper-small.en",
    "small": "whisper-small",
    "medium.en": "whisper-medium.en",
    "medium": "whisper-medium",
    "large-v3": "whisper-large-v3",
    "distil-small.en": "distil-whisper-small.en",
    "distil-medium.en": "distil-whisper-medium.en",
    "distil-large-v3": "distil-whisper-large-v3",
    "distil-large-v3.5": "whisper-distil-large-v3.5",
    "large-v3-turbo": "whisper-large-v3-turbo",
    "turbo": "whisper-large-v3-turbo",
}


def available_models() -> List[str]:
    """Returns the names of available models."""
    return list(_MODELS.keys())


def get_assets_path():
    """Returns the path to the assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_logger():
    """Returns the module logger."""
    return logging.getLogger("faster_whisper")


def _get_supported_compute_types(device: str) -> Set[str]:
    try:
        return ctranslate2.get_supported_compute_types(device)
    except Exception:
        return set()


def _gpu_supports_bfloat16(device_index=0):
    return "bfloat16" in _get_supported_compute_types("cuda")


def _select_download_precision(device, compute_type, device_index=0):
    if device == "cpu":
        return "float32"

    if compute_type in ("float32", "float16", "bfloat16"):
        return compute_type

    if _gpu_supports_bfloat16(device_index):
        return "bfloat16"
    return "float16"


def validate_compute_type(device, compute_type, device_index=0):
    if compute_type in ("auto", "default"):
        return

    supported = _get_supported_compute_types(device)

    if not supported:
        return

    if compute_type not in supported:
        sorted_types = sorted(supported)
        raise ValueError(
            f"Compute type '{compute_type}' is not supported on {device}. "
            f"Supported types: {', '.join(sorted_types)}."
        )


def _get_model_repo_id(model_name, precision):
    base_name = _MODELS[model_name]
    return f"ctranslate2-4you/{base_name}-ct2-{precision}"


def download_model(
    size_or_id: str,
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[str, bool]] = None,
    device: str = "cpu",
    compute_type: str = "default",
    device_index: int = 0,
):
    """Downloads a CTranslate2 Whisper model from the Hugging Face Hub.

    The model precision (float32, float16, bfloat16) is automatically selected
    based on the device and compute_type. For GPU with quantized compute types
    (int8, etc.), the best available precision is chosen as a source for runtime
    conversion.

    Args:
      size_or_id: Size of the model to download from ctranslate2-4you on HuggingFace
        (tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium,
        medium.en, distil-medium.en, large-v3, distil-large-v3, distil-large-v3.5,
        large-v3-turbo, turbo), or a CTranslate2-converted model ID from the
        Hugging Face Hub (e.g. ctranslate2-4you/whisper-large-v3-ct2-float16).
      output_dir: Directory where the model should be saved. If not set, the model
        is saved in the cache directory.
      local_files_only: If True, avoid downloading the file and return the path to
        the local cached file if it exists.
      cache_dir: Path to the folder where cached files are stored.
      revision: An optional Git revision id which can be a branch name, a tag, or a
        commit hash.
      use_auth_token: HuggingFace authentication token or True to use the token
        stored by the HuggingFace config folder.
      device: Device type ("cpu" or "cuda") used to select model precision.
      compute_type: Compute type used to select model precision.
      device_index: GPU device index, used to check bfloat16 support.

    Returns:
      The path to the downloaded model.

    Raises:
      ValueError: if the model size is invalid.
    """
    if re.match(r".*/.*", size_or_id):
        repo_id = size_or_id
    else:
        if size_or_id not in _MODELS:
            raise ValueError(
                "Invalid model size '%s', expected one of: %s"
                % (size_or_id, ", ".join(_MODELS.keys()))
            )
        precision = _select_download_precision(device, compute_type, device_index)
        repo_id = _get_model_repo_id(size_or_id, precision)

    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    kwargs = {
        "local_files_only": local_files_only,
        "allow_patterns": allow_patterns,
        "tqdm_class": disabled_tqdm,
        "revision": revision,
    }

    if output_dir is not None:
        kwargs["local_dir"] = output_dir

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    if use_auth_token is not None:
        kwargs["token"] = use_auth_token

    return huggingface_hub.snapshot_download(repo_id, **kwargs)


def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


class disabled_tqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )
