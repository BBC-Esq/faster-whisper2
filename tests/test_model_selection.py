from unittest.mock import patch

import pytest

from faster_whisper.utils import (
    _MODELS,
    _get_model_repo_id,
    _select_download_precision,
    available_models,
    validate_compute_type,
)

ALL_MODEL_NAMES = list(_MODELS.keys())

COMPUTE_TYPES_EXPLICIT = ["float32", "float16", "bfloat16"]
COMPUTE_TYPES_QUANTIZED = ["int8", "int8_float16", "int8_float32", "int8_bfloat16", "int16"]
COMPUTE_TYPES_AUTO = ["auto", "default"]
ALL_COMPUTE_TYPES = COMPUTE_TYPES_EXPLICIT + COMPUTE_TYPES_QUANTIZED + COMPUTE_TYPES_AUTO

SCENARIOS = [
    ("cpu", None),
    ("cuda", True),
    ("cuda", False),
]

SCENARIO_LABELS = {
    ("cpu", None): "CPU",
    ("cuda", True): "CUDA (Ampere+)",
    ("cuda", False): "CUDA (pre-Ampere)",
}


def _expected_precision(device, compute_type, bf16_support):
    if device == "cpu":
        return "float32"
    if compute_type in ("float32", "float16", "bfloat16"):
        return compute_type
    if bf16_support:
        return "bfloat16"
    return "float16"


class TestSelectDownloadPrecision:
    def test_cpu_always_returns_float32(self):
        for ct in ALL_COMPUTE_TYPES:
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


class TestValidateComputeType:
    def test_auto_and_default_always_pass(self):
        for device in ("cpu", "cuda"):
            for ct in ("auto", "default"):
                validate_compute_type(device, ct)

    def test_cpu_valid_types(self):
        for ct in ("float32", "int8", "int8_float32", "int16"):
            validate_compute_type("cpu", ct)

    def test_cpu_rejects_float16(self):
        with pytest.raises(ValueError, match="not supported on CPU"):
            validate_compute_type("cpu", "float16")

    def test_cpu_rejects_bfloat16(self):
        with pytest.raises(ValueError, match="not supported on CPU"):
            validate_compute_type("cpu", "bfloat16")

    def test_cpu_rejects_int8_float16(self):
        with pytest.raises(ValueError, match="not supported on CPU"):
            validate_compute_type("cpu", "int8_float16")

    def test_cpu_rejects_int8_bfloat16(self):
        with pytest.raises(ValueError, match="not supported on CPU"):
            validate_compute_type("cpu", "int8_bfloat16")

    @patch("faster_whisper.utils._get_cuda_compute_capability", return_value=(8, 6))
    def test_cuda_ampere_accepts_all_cuda_types(self, _mock):
        for ct in (
            "float32",
            "float16",
            "bfloat16",
            "int8",
            "int8_float16",
            "int8_float32",
            "int8_bfloat16",
        ):
            validate_compute_type("cuda", ct)

    @patch("faster_whisper.utils._get_cuda_compute_capability", return_value=(7, 5))
    def test_cuda_turing_accepts_float16_and_int8(self, _mock):
        for ct in ("float32", "float16", "int8", "int8_float16", "int8_float32"):
            validate_compute_type("cuda", ct)

    @patch("faster_whisper.utils._get_cuda_compute_capability", return_value=(7, 5))
    def test_cuda_turing_rejects_bfloat16(self, _mock):
        with pytest.raises(ValueError, match="compute capability >= 8.0"):
            validate_compute_type("cuda", "bfloat16")

    @patch("faster_whisper.utils._get_cuda_compute_capability", return_value=(7, 5))
    def test_cuda_turing_rejects_int8_bfloat16(self, _mock):
        with pytest.raises(ValueError, match="compute capability >= 8.0"):
            validate_compute_type("cuda", "int8_bfloat16")

    @patch("faster_whisper.utils._get_cuda_compute_capability", return_value=(5, 2))
    def test_cuda_old_gpu_rejects_float16(self, _mock):
        with pytest.raises(ValueError, match="compute capability >= 7.0"):
            validate_compute_type("cuda", "float16")

    @patch("faster_whisper.utils._get_cuda_compute_capability", return_value=(5, 2))
    def test_cuda_old_gpu_rejects_int8(self, _mock):
        with pytest.raises(ValueError, match="compute capability >= 6.1"):
            validate_compute_type("cuda", "int8")

    @patch("faster_whisper.utils._get_cuda_compute_capability", return_value=(6, 1))
    def test_cuda_pascal_61_accepts_int8(self, _mock):
        validate_compute_type("cuda", "int8")

    def test_cuda_int16_always_rejected(self):
        with pytest.raises(ValueError, match="only supported on CPU"):
            validate_compute_type("cuda", "int16")

    @patch("faster_whisper.utils._get_cuda_compute_capability", return_value=(None, None))
    def test_cuda_unknown_gpu_skips_capability_checks(self, _mock):
        for ct in ("float32", "float16", "bfloat16", "int8", "int8_float16", "int8_bfloat16"):
            validate_compute_type("cuda", ct)


class TestValidationTable:
    def test_all_validation_permutations_with_table(self):
        gpu_scenarios = [
            ("cpu", None, None, "CPU"),
            ("cuda", 8, 6, "CUDA (Ampere 8.6)"),
            ("cuda", 7, 5, "CUDA (Turing 7.5)"),
            ("cuda", 6, 1, "CUDA (Pascal 6.1)"),
            ("cuda", 5, 2, "CUDA (Maxwell 5.2)"),
            ("cuda", None, None, "CUDA (unknown)"),
        ]

        header = f"{'Compute Type':<16} {'Device':<22} {'Expected':<12} {'Got':<12} {'Status'}"
        separator = "-" * len(header)
        lines = ["\n", separator, header, separator]
        total = 0
        passed = 0
        failed = 0

        for device, major, minor, label in gpu_scenarios:
            for ct in ALL_COMPUTE_TYPES:
                total += 1
                should_reject = _should_reject(device, ct, major)

                with patch(
                    "faster_whisper.utils._get_cuda_compute_capability",
                    return_value=(major, minor),
                ):
                    try:
                        validate_compute_type(device, ct)
                        got_rejected = False
                    except ValueError:
                        got_rejected = True

                ok = got_rejected == should_reject
                expected_str = "REJECT" if should_reject else "ACCEPT"
                got_str = "REJECT" if got_rejected else "ACCEPT"
                status = "PASS" if ok else "FAIL"

                if ok:
                    passed += 1
                else:
                    failed += 1

                lines.append(f"{ct:<16} {label:<22} {expected_str:<12} {got_str:<12} {status}")

        lines.append(separator)
        lines.append(f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
        lines.append(separator)

        print("\n".join(lines))
        assert failed == 0, f"{failed} validation permutations failed"


def _should_reject(device, compute_type, major):
    if compute_type in ("auto", "default"):
        return False

    if device == "cpu":
        return compute_type in ("float16", "bfloat16", "int8_float16", "int8_bfloat16")

    if compute_type == "int16":
        return True

    if major is None:
        return False

    if compute_type in ("float16", "int8_float16") and major < 7:
        return True
    if compute_type in ("bfloat16", "int8_bfloat16") and major < 8:
        return True
    if compute_type == "int8" and major < 6:
        return True

    return False


class TestFullPermutationTable:
    def test_all_permutations_with_table(self, capsys):
        header = (
            f"{'Model':<20} {'Compute Type':<16} {'Device':<18} "
            f"{'Expected':<12} {'Got':<12} {'Repo ID':<58} {'Status'}"
        )
        separator = "-" * len(header)
        lines = ["\n", separator, header, separator]
        total = 0
        passed = 0
        failed = 0

        for device, bf16_support in SCENARIOS:
            label = SCENARIO_LABELS[(device, bf16_support)]
            mock_target = "faster_whisper.utils._gpu_supports_bfloat16"

            for model in ALL_MODEL_NAMES:
                for ct in ALL_COMPUTE_TYPES:
                    total += 1
                    expected = _expected_precision(device, ct, bf16_support)

                    with patch(mock_target, return_value=bf16_support or False):
                        actual = _select_download_precision(device, ct)
                        repo_id = _get_model_repo_id(model, actual)

                    ok = actual == expected
                    status = "PASS" if ok else "FAIL"
                    if ok:
                        passed += 1
                    else:
                        failed += 1

                    lines.append(
                        f"{model:<20} {ct:<16} {label:<18} "
                        f"{expected:<12} {actual:<12} {repo_id:<58} {status}"
                    )

        lines.append(separator)
        lines.append(f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
        lines.append(separator)

        table = "\n".join(lines)
        print(table)

        assert failed == 0, f"{failed} permutations failed"
