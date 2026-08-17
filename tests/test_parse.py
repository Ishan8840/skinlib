"""Tests for the skin mask."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skinlib import landmarks as lm
from skinlib.config import Config
from skinlib.detect import AssetNotFoundError, detect_face, load_image
from skinlib.parse import parse_skin, resolve_weights, weights_hash

from .conftest import FIXTURE_DIR, PORTRAITS


def _centroid(landmarks: np.ndarray, indices: tuple[int, ...]) -> tuple[int, int]:
    point = landmarks[list(indices)].mean(axis=0)
    return int(round(point[1])), int(round(point[0]))  # (row, col)


def test_mask_is_boolean_and_image_shaped(analysed) -> None:
    loaded, _face, skin, _regions = analysed
    assert skin.dtype == np.bool_
    assert skin.shape == loaded.working_size


def test_mask_covers_a_plausible_share_of_the_face(analysed) -> None:
    _loaded, face, skin, _regions = analysed
    box_area = face.bbox[2] * face.bbox[3]
    covered = skin.sum() / box_area
    # The landmark box bounds the face tightly, so a correct mask fills most of
    # it. Far outside this range means the mask latched onto something else.
    assert 0.4 < covered < 1.2, f"skin mask covers {covered:.2f} of the face box"


@pytest.mark.parametrize("feature", ["eyes", "lips"])
def test_mask_excludes_non_skin_features(analysed, feature: str) -> None:
    """Metrics must never read a pixel from an eye or a lip."""
    _loaded, face, skin, _regions = analysed
    index_sets = {
        "eyes": (lm.LEFT_EYE, lm.RIGHT_EYE),
        "lips": (lm.LIPS_OUTER,),
    }[feature]
    for indices in index_sets:
        row, col = _centroid(face.landmarks, indices)
        assert not skin[row, col], f"{feature} centre at ({row}, {col}) is inside the skin mask"


def test_mask_excludes_the_background(analysed) -> None:
    _loaded, _face, skin, _regions = analysed
    # Frame corners are background in every fixture.
    for row, col in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert not skin[row, col]


@pytest.mark.parametrize("name", PORTRAITS)
def test_mask_is_anchored_to_the_face_not_the_largest_blob(
    name: str, full_config: Config, parser
) -> None:
    """The mask must overlap the detected face.

    Selecting the largest connected skin component instead of the one on the
    face picks up bare arms, hands, or a flight suit that parsed as skin — one
    fixture produced a suit blob several times the size of the face, which
    yielded a mask with zero overlap with every region.
    """
    loaded = load_image(FIXTURE_DIR / name)
    face = detect_face(loaded.image, full_config)
    assert face is not None
    skin = parse_skin(loaded.image, face, full_config, parser=parser)

    x, y, w, h = face.bbox
    inside_box = skin[y : y + h, x : x + w].sum()
    assert inside_box > 0
    # The overwhelming majority of mask pixels must lie on the face itself.
    assert inside_box / skin.sum() > 0.85, f"{name}: mask is mostly off the face"


def test_nostrils_are_carved_out(analysed) -> None:
    """BiSeNet files nostrils under `nose`, so they need an explicit carve."""
    _loaded, face, skin, _regions = analysed
    for indices in (lm.LEFT_NOSTRIL, lm.RIGHT_NOSTRIL):
        row, col = _centroid(face.landmarks, indices)
        assert not skin[row, col], "nostril aperture is inside the skin mask"


def test_facial_hair_suppression_removes_pixels(full_config: Config, parser) -> None:
    """The beard fixture must lose lower-face pixels when suppression is on."""
    loaded = load_image(FIXTURE_DIR / "portrait_a.jpg")
    face = detect_face(loaded.image, full_config)
    assert face is not None

    with_suppression = parse_skin(loaded.image, face, full_config, parser=parser)
    without = parse_skin(
        loaded.image,
        face,
        replace(full_config, parse=replace(full_config.parse, suppress_facial_hair=False)),
        parser=parser,
    )
    assert with_suppression.sum() < without.sum()
    # Suppression only ever removes pixels; it must never add any.
    assert not (with_suppression & ~without).any()


def test_disabling_suppression_is_reflected_in_the_config_hash(full_config: Config) -> None:
    """A mask-changing switch must change the comparability key."""
    from skinlib.config import config_fingerprint

    other = replace(
        full_config, parse=replace(full_config.parse, suppress_facial_hair=False)
    )
    assert config_fingerprint(full_config) != config_fingerprint(other)


def test_parse_rejects_mismatched_image(analysed, full_config: Config, parser) -> None:
    """Landmarks and pixels must belong to the same image."""
    loaded, face, _skin, _regions = analysed
    smaller = loaded.image[: loaded.image.shape[0] // 2]
    with pytest.raises(ValueError, match="does not match"):
        parse_skin(smaller, face, full_config, parser=parser)


def test_missing_weights_raise(config: Config) -> None:
    with pytest.raises(AssetNotFoundError):
        resolve_weights(replace(config.parse, weights_path=Path("/nope/bisenet.pth")))


def test_weights_hash_is_stable(full_config: Config) -> None:
    assert weights_hash(full_config.parse) == weights_hash(full_config.parse)
    assert len(weights_hash(full_config.parse)) == 16


def test_parse_is_deterministic(analysed, full_config: Config, parser) -> None:
    loaded, face, skin, _regions = analysed
    again = parse_skin(loaded.image, face, full_config, parser=parser)
    assert np.array_equal(skin, again)
