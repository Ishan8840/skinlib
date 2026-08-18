"""The spot detector's floors are physical sizes, not pixel counts.

A pixel floor admits sub-pore noise on a close capture and rejects real marks on
a distant one. Scored against the first hand-labelled face, the old 8px floor
gave precision 0.13 against 27 false positives; a 1.2mm floor gives 0.50.
"""

from __future__ import annotations


import numpy as np
import pytest

from skinlib.config import Config, SpotsConfig
from skinlib.spots import mask_edge_margin, min_mark_area
from skinlib.types import Face


def _face(width: int) -> Face:
    return Face(
        landmarks=np.zeros((478, 2), np.float32),
        bbox=(0, 0, width, width),
        image_size=(width * 2, width * 2),
        n_faces=1,
        area_fraction=0.25,
    )


def test_minimum_mark_is_the_same_physical_size_at_every_distance() -> None:
    """The whole point: 1.2mm stays 1.2mm however close the subject stood."""
    config = SpotsConfig()
    for width in (400, 700, 1400):
        area = min_mark_area(_face(width), config)
        diameter_px = 2.0 * np.sqrt(area / np.pi)
        mm = diameter_px / (width / config.assumed_face_width_mm)
        assert mm == pytest.approx(config.min_mark_mm, rel=0.02), width


def test_area_floor_grows_with_apparent_face_size() -> None:
    config = SpotsConfig()
    small, large = min_mark_area(_face(400), config), min_mark_area(_face(1400), config)
    # Area scales with the square of apparent size.
    assert large / small == pytest.approx((1400 / 400) ** 2, rel=0.05)


def test_edge_margin_is_face_relative() -> None:
    config = SpotsConfig()
    assert mask_edge_margin(_face(1400), config) > mask_edge_margin(_face(400), config)


def test_both_fall_back_without_a_face() -> None:
    """A caller with no face measurement gets the absolute floors, not a crash."""
    config = SpotsConfig()
    assert min_mark_area(None, config) == float(config.min_area_px)
    assert mask_edge_margin(None, config) == config.mask_edge_margin_px


def test_default_rejects_sub_millimetre_noise() -> None:
    """The failure this replaced: 0.4mm specks at the lash line counted as marks."""
    config = Config().spots
    face = _face(1400)                      # a full-resolution capture
    px_per_mm = 1400 / config.assumed_face_width_mm
    speck = np.pi / 4 * (0.4 * px_per_mm) ** 2
    assert speck < min_mark_area(face, config)
