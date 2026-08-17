"""Tests for haemoglobin-based inflammation metrics and lesion detection.

The claim under test is that this measures something the melanin family cannot.
A haemoglobin channel that merely tracked the melanin one would be redundant
rather than wrong, and would pass every smoke test, so the separation is
asserted directly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from skinlib.config import Config
from skinlib.metrics import compute_metrics
from skinlib.spots import (
    detect_lesions,
    detect_spots,
    hemoglobin_residual,
    melanin_residual,
)
from skinlib.types import Lesion


# ---------------------------------------------------------------------------
# the residual
# ---------------------------------------------------------------------------


def test_hemoglobin_residual_is_zero_mean_ish_on_flat_skin(analysed, full_config: Config) -> None:
    """A local residual of a smooth field is nothing much."""
    loaded, face, skin, _regions = analysed
    residual = hemoglobin_residual(loaded.image, skin, face, full_config)
    assert residual.shape == loaded.image.shape[:2]
    assert abs(float(np.median(residual[skin]))) < 0.05


def test_hemoglobin_and_melanin_residuals_are_not_the_same_signal(
    analysed, full_config: Config
) -> None:
    """If they were redundant, the second channel would buy nothing."""
    loaded, face, skin, _regions = analysed
    mel = melanin_residual(loaded.image, skin, face, full_config)[skin]
    hgb = hemoglobin_residual(loaded.image, skin, face, full_config)[skin]
    correlation = abs(float(np.corrcoef(mel, hgb)[0, 1]))
    assert correlation < 0.95, f"channels are near-duplicates (r={correlation:.3f})"


def test_diffuse_redness_does_not_register_as_inflammation(
    analysed, full_config: Config
) -> None:
    """A whole-face flush is smooth at the kernel scale, so it is background.

    This is deliberate. A lesion detector should find lesions; the diffuse
    component is what `erythema_index` and `hemoglobin_density` are for. If a
    broad flush leaked into the residual, every warm room would read as acne.
    """
    loaded, face, skin, regions = analysed
    flushed = loaded.image.astype(np.float64)
    flushed[:, :, 2] = np.clip(flushed[:, :, 2] * 1.12, 0, 255)  # redder, everywhere
    flushed = flushed.astype(np.uint8)

    before = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)
    after = compute_metrics(flushed, skin, regions, full_config, spots=[], face=face)

    baseline = before.global_["inflammation_burden"]
    assert abs(after.global_["inflammation_burden"] - baseline) < 0.25 * max(baseline, 1e-6)


def test_local_red_lesions_do_register(analysed, full_config: Config) -> None:
    """Sensitivity: discrete red marks must move the inflammation metrics."""
    loaded, face, skin, regions = analysed
    marked = loaded.image.copy()
    rows, cols = np.nonzero(skin)
    picks = np.linspace(0, len(rows) - 1, 40).astype(int)
    discs = np.zeros(skin.shape, dtype=np.uint8)
    for row, col in zip(rows[picks], cols[picks]):
        cv2.circle(discs, (int(col), int(row)), 6, 1, -1)
    spot = discs.astype(bool) & skin
    # Redder and slightly darker, which is what an inflamed papule looks like.
    patch = marked[spot].astype(np.float64)
    patch[:, 2] = np.clip(patch[:, 2] * 1.10, 0, 255)  # more red
    patch[:, 0] = patch[:, 0] * 0.80  # less blue
    patch[:, 1] = patch[:, 1] * 0.80  # less green
    marked[spot] = patch.astype(np.uint8)

    before = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)
    after = compute_metrics(marked, skin, regions, full_config, spots=[], face=face)
    assert after.global_["inflammation_burden"] > before.global_["inflammation_burden"]
    assert after.global_["inflammation_contrast"] > before.global_["inflammation_contrast"]


# ---------------------------------------------------------------------------
# metrics plumbing
# ---------------------------------------------------------------------------


def test_inflammation_metrics_need_a_face(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    without = compute_metrics(loaded.image, skin, regions, full_config, spots=[])
    assert np.isnan(without.global_["inflammation_burden"])
    assert np.isnan(without.global_["inflammation_contrast"])

    with_face = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)
    assert np.isfinite(with_face.global_["inflammation_burden"])
    assert 0.0 <= with_face.global_["inflammation_burden"] <= 1.0


def test_inflammation_differs_from_pigmentation_across_regions(
    analysed, full_config: Config
) -> None:
    """The two families must rank regions differently, or one is redundant.

    On real captures the forehead measured spot_burden 0.30 against
    inflammation_burden 0.03 — a tenfold split, which is what pigmentation
    without active inflammation looks like.
    """
    loaded, face, skin, regions = analysed
    result = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)
    pairs = [
        (v["spot_burden"], v["inflammation_burden"])
        for v in result.by_region.values()
        if np.isfinite(v["spot_burden"]) and np.isfinite(v["inflammation_burden"])
    ]
    assert len(pairs) >= 5
    spot = np.array([p[0] for p in pairs])
    inflammation = np.array([p[1] for p in pairs])
    assert abs(float(np.corrcoef(spot, inflammation)[0, 1])) < 0.98


# ---------------------------------------------------------------------------
# lesion detection
# ---------------------------------------------------------------------------


def test_detect_lesions_returns_lesions_not_spots(analysed, full_config: Config) -> None:
    """Separate types on purpose: the two must never be summed."""
    loaded, face, skin, regions = analysed
    lesions = detect_lesions(loaded.image, skin, face, regions, full_config)
    assert all(isinstance(item, Lesion) for item in lesions)
    for lesion in lesions:
        assert lesion.area_px > 0
        assert np.isfinite(lesion.mean_hemoglobin_density)
        assert lesion.region in set(regions) | {""}


def test_detect_lesions_is_deterministic(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    first = detect_lesions(loaded.image, skin, face, regions, full_config)
    second = detect_lesions(loaded.image, skin, face, regions, full_config)
    assert [x.centroid for x in first] == [x.centroid for x in second]
    assert [x.area_px for x in first] == [x.area_px for x in second]


def test_detect_lesions_handles_an_empty_mask(analysed, full_config: Config) -> None:
    loaded, face, _skin, regions = analysed
    empty = np.zeros(loaded.image.shape[:2], dtype=bool)
    assert detect_lesions(loaded.image, empty, face, regions, full_config) == []


def test_spot_and_lesion_detectors_disagree(analysed, full_config: Config) -> None:
    """Different chromophores should not return an identical set of objects."""
    loaded, face, skin, regions = analysed
    spots = detect_spots(loaded.image, skin, face, regions, full_config)
    lesions = detect_lesions(loaded.image, skin, face, regions, full_config)
    assert {s.centroid for s in spots} != {le.centroid for le in lesions}


def test_refactor_preserved_spot_detection(analysed, full_config: Config) -> None:
    """detect_spots moved onto the shared extractor; results must be unchanged.

    Pinned because the shared path is now also used by lesions, and a change
    made for one chromophore would otherwise silently move the other.
    """
    loaded, face, skin, regions = analysed
    spots = detect_spots(loaded.image, skin, face, regions, full_config)
    assert spots, "expected the primary fixture to yield spots"
    # Sorted by area descending, ties broken by position — fully specified.
    areas = [s.area_px for s in spots]
    assert areas == sorted(areas, reverse=True)
    for spot in spots:
        assert np.isfinite(spot.mean_melanin_index)
        assert np.isfinite(spot.contrast)
