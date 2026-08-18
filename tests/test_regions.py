"""Tests for the region map."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from skinlib import landmarks as lm
from skinlib.config import Config
from skinlib.detect import detect_face, load_image
from skinlib.regions import build_region_polygons, build_regions
from skinlib.types import REGION_NAMES

from .conftest import FIXTURE_DIR, PORTRAITS


def test_all_regions_present(analysed) -> None:
    _loaded, _face, _skin, regions = analysed
    assert set(regions) == set(REGION_NAMES)
    assert list(regions) == list(REGION_NAMES), "region order must be stable"


def test_regions_are_subsets_of_the_skin_mask(analysed) -> None:
    """The core invariant: no metric can read a pixel parsing rejected."""
    _loaded, _face, skin, regions = analysed
    for name, mask in regions.items():
        assert not (mask & ~skin).any(), f"{name} escapes the skin mask"


def test_regions_are_mutually_exclusive(analysed) -> None:
    """No pixel may be counted twice in per-region aggregates."""
    _loaded, _face, _skin, regions = analysed
    stacked = np.stack([mask for mask in regions.values()])
    overlap = stacked.sum(axis=0)
    assert overlap.max() <= 1, f"{int((overlap > 1).sum())} pixels claimed by 2+ regions"


def test_regions_cover_a_meaningful_share_of_the_skin(analysed) -> None:
    _loaded, _face, skin, regions = analysed
    covered = np.zeros_like(skin)
    for mask in regions.values():
        covered |= mask
    # Not 100%: the neck-adjacent jaw, the temples and the silhouette inset are
    # deliberately unassigned. But most of the face should land somewhere.
    assert covered.sum() / skin.sum() > 0.5


def test_periorbital_outranks_the_cheek_under_the_eye(analysed) -> None:
    """Under-eye skin is the dark-circle signal and must not be annexed.

    The cheek band reaches up under the eye, and the cheek is far larger, so a
    priority inversion here would silently average the signal away rather than
    produce any visible error.
    """
    _loaded, face, _skin, regions = analysed
    for eye, periorbital in (
        (lm.LEFT_EYE, "periorbital_left"),
        (lm.RIGHT_EYE, "periorbital_right"),
    ):
        eye_points = face.landmarks[list(eye)]
        centre_x = float(eye_points[:, 0].mean())
        bottom_y = float(eye_points[:, 1].max())
        # A short way below the lower lid, inside the periorbital ring.
        probe_y = int(round(bottom_y + 0.06 * face.bbox[3]))
        probe_x = int(round(centre_x))
        if regions[periorbital][probe_y, probe_x] or not any(
            regions[name][probe_y, probe_x] for name in ("left_cheek", "right_cheek")
        ):
            continue
        pytest.fail(f"under-eye probe was claimed by a cheek instead of {periorbital}")


def _centre_x(mask: np.ndarray) -> float:
    cols = np.nonzero(mask)[1]
    assert len(cols) > 0
    return float(cols.mean())


def test_side_labels_follow_the_eye_landmarks(analysed) -> None:
    """`left_cheek` is the cheek under the subject's left eye, by construction.

    Sides are derived from the eye-corner axis rather than from image columns,
    so head tilt and framing cannot flip them.
    """
    _loaded, face, _skin, regions = analysed
    left_eye_x = float(face.landmarks[list(lm.LEFT_EYE)][:, 0].mean())
    right_eye_x = float(face.landmarks[list(lm.RIGHT_EYE)][:, 0].mean())

    assert (_centre_x(regions["left_cheek"]) > _centre_x(regions["right_cheek"])) == (
        left_eye_x > right_eye_x
    )
    assert (
        _centre_x(regions["periorbital_left"]) > _centre_x(regions["periorbital_right"])
    ) == (left_eye_x > right_eye_x)


def test_mirrored_capture_swaps_the_side_labels(analysed, config: Config) -> None:
    """A KNOWN LIMITATION, pinned so it cannot change unnoticed.

    A mirrored photo is indistinguishable from an unmirrored photo of a
    mirror-image person, so the landmarker labels the apparent left eye as the
    left eye and every side label follows. Nothing downstream can recover the
    true handedness from pixels alone.

    This matters for longitudinal tracking: if a capture app mirrors its front
    camera in one session and not the next, `left_cheek` silently changes
    cheeks and manufactures a change that never happened. The caller must keep
    mirroring consistent — the library cannot detect it. See README.
    """
    loaded, _face, _skin, regions = analysed

    mirrored = np.ascontiguousarray(loaded.image[:, ::-1])
    face = detect_face(mirrored, config)
    assert face is not None
    flipped = build_region_polygons(face, config)

    original_left_is_right_of_frame = _centre_x(regions["left_cheek"]) > _centre_x(
        regions["right_cheek"]
    )
    flipped_left_is_right_of_frame = _centre_x(flipped["left_cheek"]) > _centre_x(
        flipped["right_cheek"]
    )
    # Same side of the FRAME in both: the label tracked the pixels, not the
    # person. That is the limitation being documented.
    assert original_left_is_right_of_frame == flipped_left_is_right_of_frame


@pytest.mark.parametrize("name", PORTRAITS)
def test_polygons_are_produced_for_every_pose(name: str, config: Config) -> None:
    """Polygon construction must not collapse on a turned head.

    A cheek boundary cut at the nose ala emptied the far cheek entirely under
    yaw; the midline split replaced it.
    """
    loaded = load_image(FIXTURE_DIR / name)
    face = detect_face(loaded.image, config)
    assert face is not None
    polygons = build_region_polygons(face, config)
    for region, mask in polygons.items():
        assert mask.sum() > 0, f"{name}: {region} polygon is empty before masking"


def test_regions_reject_a_mismatched_mask(analysed, config: Config) -> None:
    _loaded, face, skin, _regions = analysed
    with pytest.raises(ValueError, match="does not match"):
        build_regions(face, skin[: skin.shape[0] // 2], config)


def test_priority_must_cover_every_region(analysed, config: Config) -> None:
    """A typo in priority must fail loudly, not silently drop a region."""
    _loaded, face, skin, _regions = analysed
    broken = replace(
        config,
        regions=replace(config.regions, priority=("forehead", "nose")),
    )
    with pytest.raises(ValueError, match="priority"):
        build_regions(face, skin, broken)


def test_non_exclusive_mode_allows_overlap(analysed, config: Config) -> None:
    _loaded, face, skin, exclusive = analysed
    overlapping = build_regions(
        face, skin, replace(config, regions=replace(config.regions, exclusive=False))
    )
    total_exclusive = sum(int(mask.sum()) for mask in exclusive.values())
    total_overlapping = sum(int(mask.sum()) for mask in overlapping.values())
    assert total_overlapping >= total_exclusive


def test_regions_are_deterministic(analysed, config: Config) -> None:
    _loaded, face, skin, regions = analysed
    again = build_regions(face, skin, config)
    for name in REGION_NAMES:
        assert np.array_equal(regions[name], again[name])
