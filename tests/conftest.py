"""Shared test fixtures.

Image fixtures are public-domain NASA portraits (see fixtures/SOURCES.md),
chosen to span a range of skin tones — a measurement library that is only ever
exercised on one tone is a library whose failures on other tones are invisible.

Derived fixtures (multi-face, no-face) are synthesised from the committed
images rather than downloaded, so they are reproducible and carry no extra
binary blobs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from skinlib.config import Config

FIXTURE_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent

# Single-face portraits, in roughly descending melanin order.
PORTRAITS = ("portrait_a.jpg", "portrait_b.jpg", "portrait_d.jpg", "portrait_c.jpg")
# Deliberately framed too far away; used by the quality tests.
TOO_FAR_PORTRAIT = "portrait_d.jpg"
# The canonical "good capture" used wherever one image will do.
PRIMARY_PORTRAIT = "portrait_a.jpg"


def _default_landmarker() -> Path | None:
    """The asset checked into ``models/`` during development, if present."""
    candidate = REPO_ROOT / "models" / "face_landmarker.task"
    return candidate if candidate.is_file() else None


@pytest.fixture(scope="session")
def landmarker_asset() -> Path:
    import os

    from skinlib.detect import AssetNotFoundError, resolve_landmarker_asset

    env = os.environ.get("SKINLIB_FACE_LANDMARKER")
    if env:
        try:
            return resolve_landmarker_asset(Config().detect)
        except AssetNotFoundError as exc:
            pytest.skip(str(exc))
    asset = _default_landmarker()
    if asset is None:
        pytest.skip(
            "face_landmarker.task not found; set $SKINLIB_FACE_LANDMARKER "
            "or place it in models/ (see README)"
        )
    return asset


@pytest.fixture(scope="session")
def bisenet_weights() -> Path:
    import os

    env = os.environ.get("SKINLIB_BISENET_WEIGHTS")
    candidate = Path(env) if env else REPO_ROOT / "models" / "bisenet_79999_iter.pth"
    if not candidate.is_file():
        pytest.skip(
            "BiSeNet checkpoint not found; set $SKINLIB_BISENET_WEIGHTS or place "
            "it in models/ (see README)"
        )
    return candidate


@pytest.fixture(scope="session")
def config(landmarker_asset: Path) -> Config:
    """Default config with the landmarker resolved. No parsing weights.

    Session-scoped and frozen, so no test can mutate what another test sees.
    """
    base = Config()
    return replace(base, detect=replace(base.detect, landmarker_asset_path=landmarker_asset))


@pytest.fixture(scope="session")
def full_config(config: Config, bisenet_weights: Path) -> Config:
    """Config with every model asset resolved: the whole pipeline can run."""
    return replace(config, parse=replace(config.parse, weights_path=bisenet_weights))


@pytest.fixture(scope="session")
def parser(full_config: Config):
    """One loaded BiSeNet for the whole session.

    Hoisted because the checkpoint is 53MB and the library deliberately does
    not cache it in a module global.
    """
    from skinlib.parse import load_parser

    return load_parser(full_config)


@pytest.fixture(scope="session")
def analysed(full_config: Config, parser):
    """Detect + parse + regions for the primary portrait, computed once.

    Returns ``(loaded, face, skin_mask, regions)``.
    """
    from skinlib.detect import detect_face, load_image
    from skinlib.parse import parse_skin
    from skinlib.regions import build_regions

    loaded = load_image(FIXTURE_DIR / PRIMARY_PORTRAIT)
    face = detect_face(loaded.image, full_config)
    assert face is not None
    skin = parse_skin(loaded.image, face, full_config, parser=parser)
    regions = build_regions(face, skin, full_config)
    return loaded, face, skin, regions


@pytest.fixture(scope="session")
def portrait_path() -> Path:
    return FIXTURE_DIR / PRIMARY_PORTRAIT


@pytest.fixture(scope="session")
def portrait_bgr(portrait_path: Path) -> np.ndarray:
    image = cv2.imread(str(portrait_path), cv2.IMREAD_COLOR)
    assert image is not None, f"fixture failed to decode: {portrait_path}"
    return image


@pytest.fixture(scope="session")
def multi_face_bgr() -> np.ndarray:
    """Two faces in one frame, built by tiling two portraits side by side.

    Synthesised because real group photos put the faces small enough that
    detection at working resolution is unreliable — which would make the test
    assert on the detector's distance limits rather than on the multiple-faces
    branch it is meant to cover.
    """
    left = cv2.imread(str(FIXTURE_DIR / "portrait_a.jpg"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(FIXTURE_DIR / "portrait_c.jpg"), cv2.IMREAD_COLOR)
    assert left is not None and right is not None
    height = min(left.shape[0], right.shape[0])
    left = left[:height]
    right = right[:height]
    return np.ascontiguousarray(np.hstack([left, right]))


@pytest.fixture(scope="session")
def no_face_bgr() -> np.ndarray:
    """A face-free image with realistic texture and colour.

    A flat grey block would pass a no-face assertion for the wrong reason; this
    has structure the detector could plausibly latch onto.
    """
    from skimage import data

    return cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)
