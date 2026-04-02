from unittest.mock import patch

import pytest

from faster_whisper.utils import (
    _MODELS,
    _get_model_repo_id,
    _select_download_precision,
    available_models,
)

ALL_MODEL_NAMES = list(_MODELS.keys())

COMPUTE_TYPES_EXPLICIT = ["float32", "float16", "bfloat16"]
COMPUTE_TYPES_QUANTIZED = ["int8", "int8_float16", "int16"]
COMPUTE_TYPES_AUTO = ["auto", "default"]


class TestSelectDownloadPrecision:
    def test_cpu_always_returns_float32(self):
        for ct in COMPUTE_TYPES_EXPLICIT + COMPUTE_TYPES_QUANTIZED + COMPUTE_TYPES_AUTO:
            result = _select_download_precision("cpu", ct)
            assert result == "float32", f"cpu + {ct} should yield float32, got {result}"

    def test_cuda_explicit_precision_returns_as_is(self):
        for ct in COMPUTE_TYPES_EXPLICIT:
            result = _select_download_precision("cuda", ct)
            assert result == ct, f"cuda + {ct} should yield {ct}, got {result}"

    @patch("faster_whisper.utils._gpu_supports_bfloat16", return_value=True)
    def test_cuda_quantized_with_ampere_gpu_returns_bfloat16(self, _mock):
        for ct in COMPUTE_TYPES_QUANTIZED + COMPUTE_TYPES_AUTO:
            result = _select_download_precision("cuda", ct)
            assert result == "bfloat16", f"cuda + {ct} + ampere should yield bfloat16, got {result}"

    @patch("faster_whisper.utils._gpu_supports_bfloat16", return_value=False)
    def test_cuda_quantized_with_pre_ampere_gpu_returns_float16(self, _mock):
        for ct in COMPUTE_TYPES_QUANTIZED + COMPUTE_TYPES_AUTO:
            result = _select_download_precision("cuda", ct)
            assert (
                result == "float16"
            ), f"cuda + {ct} + pre-ampere should yield float16, got {result}"


class TestGetModelRepoId:
    def test_all_models_with_all_precisions(self):
        for model_name in ALL_MODEL_NAMES:
            for precision in ["float32", "float16", "bfloat16"]:
                repo_id = _get_model_repo_id(model_name, precision)
                assert repo_id.startswith("ctranslate2-4you/")
                assert repo_id.endswith(f"-ct2-{precision}")

    def test_standard_model_repo_ids(self):
        assert _get_model_repo_id("tiny", "float16") == "ctranslate2-4you/whisper-tiny-ct2-float16"
        assert (
            _get_model_repo_id("tiny.en", "float32")
            == "ctranslate2-4you/whisper-tiny.en-ct2-float32"
        )
        assert (
            _get_model_repo_id("large-v3", "bfloat16")
            == "ctranslate2-4you/whisper-large-v3-ct2-bfloat16"
        )

    def test_distil_model_repo_ids(self):
        assert (
            _get_model_repo_id("distil-small.en", "float32")
            == "ctranslate2-4you/distil-whisper-small.en-ct2-float32"
        )
        assert (
            _get_model_repo_id("distil-medium.en", "float16")
            == "ctranslate2-4you/distil-whisper-medium.en-ct2-float16"
        )
        assert (
            _get_model_repo_id("distil-large-v3", "bfloat16")
            == "ctranslate2-4you/distil-whisper-large-v3-ct2-bfloat16"
        )
        assert (
            _get_model_repo_id("distil-large-v3.5", "float32")
            == "ctranslate2-4you/whisper-distil-large-v3.5-ct2-float32"
        )

    def test_turbo_alias(self):
        assert _get_model_repo_id("turbo", "float16") == _get_model_repo_id(
            "large-v3-turbo", "float16"
        )


class TestFullPermutations:
    @patch("faster_whisper.utils._gpu_supports_bfloat16", return_value=True)
    def test_all_models_cpu_all_compute_types(self, _mock):
        all_compute_types = COMPUTE_TYPES_EXPLICIT + COMPUTE_TYPES_QUANTIZED + COMPUTE_TYPES_AUTO
        for model in ALL_MODEL_NAMES:
            for ct in all_compute_types:
                precision = _select_download_precision("cpu", ct)
                repo_id = _get_model_repo_id(model, precision)
                assert (
                    repo_id == f"ctranslate2-4you/{_MODELS[model]}-ct2-float32"
                ), f"cpu + {model} + {ct}: expected float32 repo, got {repo_id}"

    @patch("faster_whisper.utils._gpu_supports_bfloat16", return_value=True)
    def test_all_models_cuda_ampere_all_compute_types(self, _mock):
        for model in ALL_MODEL_NAMES:
            for ct in COMPUTE_TYPES_EXPLICIT:
                precision = _select_download_precision("cuda", ct)
                repo_id = _get_model_repo_id(model, precision)
                assert repo_id.endswith(
                    f"-ct2-{ct}"
                ), f"cuda ampere + {model} + {ct}: expected {ct} repo, got {repo_id}"

            for ct in COMPUTE_TYPES_QUANTIZED + COMPUTE_TYPES_AUTO:
                precision = _select_download_precision("cuda", ct)
                repo_id = _get_model_repo_id(model, precision)
                assert repo_id.endswith(
                    "-ct2-bfloat16"
                ), f"cuda ampere + {model} + {ct}: expected bfloat16 repo, got {repo_id}"

    @patch("faster_whisper.utils._gpu_supports_bfloat16", return_value=False)
    def test_all_models_cuda_pre_ampere_all_compute_types(self, _mock):
        for model in ALL_MODEL_NAMES:
            for ct in COMPUTE_TYPES_EXPLICIT:
                precision = _select_download_precision("cuda", ct)
                repo_id = _get_model_repo_id(model, precision)
                assert repo_id.endswith(
                    f"-ct2-{ct}"
                ), f"cuda pre-ampere + {model} + {ct}: expected {ct} repo, got {repo_id}"

            for ct in COMPUTE_TYPES_QUANTIZED + COMPUTE_TYPES_AUTO:
                precision = _select_download_precision("cuda", ct)
                repo_id = _get_model_repo_id(model, precision)
                assert repo_id.endswith(
                    "-ct2-float16"
                ), f"cuda pre-ampere + {model} + {ct}: expected float16 repo, got {repo_id}"


class TestEdgeCases:
    def test_invalid_model_name_not_in_models(self):
        with pytest.raises(KeyError):
            _get_model_repo_id("nonexistent-model", "float32")

    def test_dropped_models_not_in_models(self):
        assert "large-v1" not in _MODELS
        assert "large-v2" not in _MODELS
        assert "distil-large-v2" not in _MODELS
        assert "large" not in _MODELS

    def test_available_models_matches_models_dict(self):
        assert set(available_models()) == set(_MODELS.keys())

    def test_device_index_passed_through(self):
        with patch("faster_whisper.utils._gpu_supports_bfloat16") as mock_bf16:
            mock_bf16.return_value = True
            _select_download_precision("cuda", "int8", device_index=2)
            mock_bf16.assert_called_once_with(2)
