"""Determinism.

The contract this library sells: same image, same config, same weights, byte-
identical metrics. A longitudinal tracker compares today's numbers against a
reading from months ago, so any run-to-run drift becomes an apparent change in
someone's skin.

These tests are the ones to run first when something feels wrong.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from skinlib import analyze
from skinlib.config import Config, config_fingerprint
from skinlib.types import METRIC_NAMES

from .conftest import FIXTURE_DIR, PORTRAITS, PRIMARY_PORTRAIT


def _assert_metrics_identical(first: dict[str, float], second: dict[str, float]) -> None:
    """Exact equality, with NaN counted as equal to NaN.

    Not approximate: `pytest.approx` would let real drift through, and drift is
    precisely what this is guarding against. NaN needs the special case because
    NaN != NaN, and NaN is a legitimate value here meaning "not measured".
    """
    assert set(first) == set(second)
    for name, value in first.items():
        other = second[name]
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(other), f"{name}: {value} vs {other}"
        else:
            assert value == other, f"{name}: {value!r} != {other!r}"


@pytest.fixture(scope="module")
def twice(full_config: Config, parser):
    path = FIXTURE_DIR / PRIMARY_PORTRAIT
    return (
        analyze(path, config=full_config, parser=parser),
        analyze(path, config=full_config, parser=parser),
    )


def test_same_image_twice_gives_identical_metrics(twice) -> None:
    first, second = twice
    assert first.metrics is not None and second.metrics is not None
    _assert_metrics_identical(first.metrics.global_, second.metrics.global_)
    for region in first.metrics.by_region:
        _assert_metrics_identical(
            first.metrics.by_region[region], second.metrics.by_region[region]
        )
    assert first.metrics.pixel_counts == second.metrics.pixel_counts


def test_same_image_twice_gives_identical_masks(twice) -> None:
    first, second = twice
    assert first.masks is not None and second.masks is not None
    assert np.array_equal(first.masks.skin, second.masks.skin)
    for region, mask in first.masks.regions.items():
        assert np.array_equal(mask, second.masks.regions[region])


def test_same_image_twice_gives_identical_spots(twice) -> None:
    first, second = twice
    assert first.spots == second.spots, "Spot records must compare exactly equal"


def test_same_image_twice_gives_identical_quality(twice) -> None:
    first, second = twice
    assert first.quality.flags == second.quality.flags
    assert first.quality.usable == second.quality.usable
    assert first.quality.unreliable_metrics == second.quality.unreliable_metrics
    _assert_metrics_identical(first.quality.measures, second.quality.measures)


def test_colour_correction_is_byte_identical(twice) -> None:
    first, second = twice
    assert first.color is not None and second.color is not None
    assert np.array_equal(first.color.image, second.color.image)
    assert first.color.gains == second.color.gains


@pytest.mark.parametrize("name", PORTRAITS)
def test_determinism_holds_across_fixtures(name: str, full_config: Config, parser) -> None:
    """Including captures the quality gate rejects.

    A short-circuited result is still a stored result, and it must be as stable
    as any other.
    """
    path = FIXTURE_DIR / name
    first = analyze(path, config=full_config, parser=parser)
    second = analyze(path, config=full_config, parser=parser)

    assert first.quality.flags == second.quality.flags
    assert first.comparable_key == second.comparable_key
    if first.metrics is None:
        assert second.metrics is None
        return
    _assert_metrics_identical(first.metrics.global_, second.metrics.global_)


def test_loading_from_path_and_array_agree(full_config: Config, parser) -> None:
    """The same pixels must measure the same however they were handed over."""
    import cv2

    path = FIXTURE_DIR / PRIMARY_PORTRAIT
    from_path = analyze(path, config=full_config, parser=parser)
    from_array = analyze(
        cv2.imread(str(path), cv2.IMREAD_COLOR), config=full_config, parser=parser
    )
    assert from_path.metrics is not None and from_array.metrics is not None
    _assert_metrics_identical(from_path.metrics.global_, from_array.metrics.global_)


# ---------------------------------------------------------------------------
# comparability
# ---------------------------------------------------------------------------


def test_result_carries_the_full_comparability_key(twice) -> None:
    result, _ = twice
    version, config_hash, weights, landmarker = result.comparable_key
    assert version == result.version and version
    assert len(config_hash) == 16
    # Computed from what actually ran, so a swapped model is detectable even if
    # nobody remembered to bump the version string.
    assert len(weights) == 16
    assert len(landmarker) == 16


def test_a_tuned_threshold_changes_the_config_hash(full_config: Config) -> None:
    """The whole point: comparability is computed, not remembered."""
    tweaked = replace(
        full_config,
        spots=replace(full_config.spots, threshold_sigma=2.3),
    )
    assert config_fingerprint(tweaked) != config_fingerprint(full_config)


def test_config_hash_ignores_where_the_weights_live(full_config: Config, tmp_path) -> None:
    """Moving a checkpoint is not a pipeline change; its content hash covers it."""
    moved = replace(
        full_config,
        parse=replace(full_config.parse, weights_path=tmp_path / "elsewhere.pth"),
    )
    assert config_fingerprint(moved) == config_fingerprint(full_config)


def test_config_hash_is_stable_across_processes(full_config: Config) -> None:
    """No dict ordering, object identity or PYTHONHASHSEED dependence."""
    import subprocess
    import sys

    script = (
        "import sys; sys.path.insert(0, '.');"
        "from skinlib.config import Config, config_fingerprint;"
        "print(config_fingerprint(Config()))"
    )
    for seed in ("0", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": seed},
        )
        assert completed.stdout.strip() == config_fingerprint(Config())


def test_changing_a_threshold_actually_moves_a_metric(full_config: Config, parser) -> None:
    """Guards the version-bump rule.

    If a config change could never move a metric, the bump rule would be
    theatre. This shows the coupling is real: the same photo measured under a
    different threshold is a different reading, which is why the config hash
    has to travel with the result.
    """
    loosened = replace(
        full_config,
        metrics=replace(full_config.metrics, erythema_percentile=90.0),
    )
    path = FIXTURE_DIR / PRIMARY_PORTRAIT
    base = analyze(path, config=full_config, parser=parser)
    other = analyze(path, config=loosened, parser=parser)
    assert base.metrics is not None and other.metrics is not None
    assert base.metrics.global_["erythema"] != other.metrics.global_["erythema"]
    assert base.config_hash != other.config_hash


def test_no_nondeterministic_metric_names_leak(twice) -> None:
    result, _ = twice
    assert result.metrics is not None
    assert list(result.metrics.global_) == list(METRIC_NAMES)
