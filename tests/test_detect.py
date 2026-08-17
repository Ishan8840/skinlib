"""Tests for image loading and face detection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skinlib.config import Config, IOConfig
from skinlib.detect import (
    AssetNotFoundError,
    ImageLoadError,
    detect_face,
    file_hash,
    load_image,
    resolve_landmarker_asset,
)
from skinlib.types import Face

from .conftest import FIXTURE_DIR, PORTRAITS


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_load_downscales_to_working_resolution(portrait_path: Path) -> None:
    loaded = load_image(portrait_path, IOConfig(max_long_edge_px=800))
    assert max(loaded.working_size) == 800
    assert loaded.scale < 1.0
    # Aspect ratio survives the resize.
    source_ratio = loaded.source_size[0] / loaded.source_size[1]
    working_ratio = loaded.working_size[0] / loaded.working_size[1]
    assert working_ratio == pytest.approx(source_ratio, abs=0.01)


def test_load_never_upscales(portrait_path: Path) -> None:
    """A small source keeps its native size.

    This is the guard against invented high-frequency content: an upsampled
    image would carry resampling ringing into the texture band and be measured
    as roughness that the skin does not have.
    """
    loaded = load_image(portrait_path, IOConfig(max_long_edge_px=4000))
    assert loaded.working_size == loaded.source_size
    assert loaded.scale == 1.0


def test_load_none_limit_keeps_native(portrait_path: Path) -> None:
    loaded = load_image(portrait_path, IOConfig(max_long_edge_px=None))
    assert loaded.working_size == loaded.source_size
    assert loaded.scale == 1.0


def test_load_accepts_array(portrait_bgr: np.ndarray) -> None:
    loaded = load_image(portrait_bgr, IOConfig(max_long_edge_px=None))
    assert np.array_equal(loaded.image, portrait_bgr)


def test_load_rejects_bad_input() -> None:
    with pytest.raises(ImageLoadError):
        load_image(FIXTURE_DIR / "does_not_exist.jpg")
    with pytest.raises(ImageLoadError):
        load_image(np.zeros((10, 10), dtype=np.uint8))  # not 3-channel
    with pytest.raises(ImageLoadError):
        load_image(np.zeros((10, 10, 3), dtype=np.float32))  # not uint8


def test_load_is_deterministic(portrait_path: Path) -> None:
    first = load_image(portrait_path)
    second = load_image(portrait_path)
    assert np.array_equal(first.image, second.image)


def test_unsupported_interpolation_raises(portrait_path: Path) -> None:
    """Silently accepting another kernel would be a silent longitudinal break."""
    bad = IOConfig(max_long_edge_px=400)
    object.__setattr__(bad, "downscale_interpolation", "linear")
    with pytest.raises(ValueError, match="texture band"):
        load_image(portrait_path, bad)


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------


def test_missing_asset_raises_not_returns_none() -> None:
    """A misconfigured deployment must be loud, not a `no_face` flag."""
    config = Config().detect
    with pytest.raises(AssetNotFoundError):
        resolve_landmarker_asset(replace(config, landmarker_asset_path=Path("/nope/x.task")))


def test_file_hash_is_stable_and_content_derived(landmarker_asset: Path, tmp_path: Path) -> None:
    assert file_hash(landmarker_asset) == file_hash(landmarker_asset)
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"model-weights-v1")
    b.write_bytes(b"model-weights-v2")
    assert file_hash(a) != file_hash(b)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PORTRAITS)
def test_detects_single_face_across_skin_tones(name: str, config: Config) -> None:
    loaded = load_image(FIXTURE_DIR / name)
    face = detect_face(loaded.image, config)
    assert face is not None, f"no face detected in {name}"
    assert face.n_faces == 1
    assert face.landmarks.shape == (478, 2)
    assert face.landmarks.dtype == np.float32


def test_landmarks_lie_inside_the_frame(portrait_bgr: np.ndarray, config: Config) -> None:
    loaded = load_image(portrait_bgr)
    face = detect_face(loaded.image, config)
    assert face is not None
    height, width = loaded.working_size
    assert face.landmarks[:, 0].min() >= -1.0
    assert face.landmarks[:, 1].min() >= -1.0
    assert face.landmarks[:, 0].max() <= width + 1.0
    assert face.landmarks[:, 1].max() <= height + 1.0


def test_bbox_bounds_the_landmarks(portrait_bgr: np.ndarray, config: Config) -> None:
    loaded = load_image(portrait_bgr)
    face = detect_face(loaded.image, config)
    assert face is not None
    x, y, w, h = face.bbox
    assert w > 0 and h > 0
    assert x <= face.landmarks[:, 0].min() + 1
    assert y <= face.landmarks[:, 1].min() + 1
    assert x + w >= face.landmarks[:, 0].max() - 1
    assert y + h >= face.landmarks[:, 1].max() - 1
    assert 0.0 < face.area_fraction <= 1.0


def test_no_face_returns_none(no_face_bgr: np.ndarray, config: Config) -> None:
    """Absence of a face is data, not an error."""
    loaded = load_image(no_face_bgr)
    assert detect_face(loaded.image, config) is None


def test_multiple_faces_are_counted(multi_face_bgr: np.ndarray, config: Config) -> None:
    """n_faces reports the truth so the quality gate can flag it.

    Without this the library would silently analyse whichever face the detector
    happened to return first.
    """
    loaded = load_image(multi_face_bgr)
    face = detect_face(loaded.image, config)
    assert face is not None
    assert face.n_faces >= 2


def test_primary_face_selection_picks_the_larger(config: Config) -> None:
    """The subject is the near face, not the one that sorted first."""
    left = load_image(FIXTURE_DIR / "portrait_a.jpg", IOConfig(max_long_edge_px=None)).image
    small = load_image(FIXTURE_DIR / "portrait_c.jpg", IOConfig(max_long_edge_px=None)).image
    # Shrink one face into a corner of a canvas the other fills.
    height = left.shape[0]
    scaled = cv2_resize_half(small, height)
    canvas = np.zeros((height, left.shape[1] + scaled.shape[1], 3), dtype=np.uint8)
    canvas[:, : left.shape[1]] = left
    canvas[: scaled.shape[0], left.shape[1] :] = scaled

    # No downscale: the composite is wide, and shrinking it to the default
    # working resolution would push the smaller face under the detector's size
    # limit, turning this into a distance test instead of a selection test.
    loaded = load_image(canvas, IOConfig(max_long_edge_px=None))
    face = detect_face(loaded.image, config)
    assert face is not None
    assert face.n_faces >= 2
    # The chosen face must be the large one: its bbox centre sits in the left
    # half of the canvas.
    centre_x = face.bbox[0] + face.bbox[2] / 2
    assert centre_x < loaded.working_size[1] * 0.5


def cv2_resize_half(image: np.ndarray, target_height: int) -> np.ndarray:
    import cv2

    scale = (target_height * 0.55) / image.shape[0]
    return cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def test_detection_is_deterministic(portrait_bgr: np.ndarray, config: Config) -> None:
    """Byte-identical landmarks across runs, or nothing downstream is stable."""
    loaded = load_image(portrait_bgr)
    first = detect_face(loaded.image, config)
    second = detect_face(loaded.image, config)
    assert first is not None and second is not None
    assert np.array_equal(first.landmarks, second.landmarks)
    assert first.bbox == second.bbox


def test_face_width_property(portrait_bgr: np.ndarray, config: Config) -> None:
    loaded = load_image(portrait_bgr)
    face = detect_face(loaded.image, config)
    assert isinstance(face, Face)
    assert face.width == face.bbox[2]
