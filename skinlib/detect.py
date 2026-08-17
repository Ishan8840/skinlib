"""Image loading, face detection and landmarks.

MediaPipe >= 1.0 removed ``mediapipe.solutions.face_mesh``; this uses the Tasks
API ``FaceLandmarker``, which produces the same 478-point mesh (468 mesh points
plus 10 iris points when refinement is on).

Determinism: the landmarker runs in IMAGE mode, which is stateless — no
tracking carried between calls. A fresh landmarker is built per call rather
than cached in a module global, so there is no shared mutable state and no
call-ordering effect on output.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import Config, DetectConfig, IOConfig
from .types import Face, LoadedImage

__all__ = [
    "AssetNotFoundError",
    "ImageLoadError",
    "detect_face",
    "load_image",
    "resolve_landmarker_asset",
    "file_hash",
]

# Landmark index constants live in landmarks.py, alongside the rest of the
# fixed FaceMesh topology.

_ENV_LANDMARKER = "SKINLIB_FACE_LANDMARKER"


class AssetNotFoundError(FileNotFoundError):
    """A required model asset is missing. Raised, never swallowed."""


class ImageLoadError(ValueError):
    """The image could not be decoded."""


# Content hashes, memoised on (path, mtime_ns, size). The comparability
# guarantee needs the CONTENT hash, but content that has not changed does not
# need re-reading: hashing 57MB of checkpoints cost 60ms on every `analyze`.
# A checkpoint swapped mid-process changes mtime or size and invalidates the
# entry, so the longitudinal guarantee is intact.
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def file_hash(path: Path) -> str:
    """Content hash of a model asset, for the result's comparability key."""
    stat = Path(path).stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()[:16]
    _HASH_CACHE[key] = value
    return value


def resolve_landmarker_asset(config: DetectConfig) -> Path:
    """Locate ``face_landmarker.task``.

    Order: explicit config, then ``$SKINLIB_FACE_LANDMARKER``. Never downloads
    — a download at analysis time could silently swap the model between two
    sessions of a longitudinal series.
    """
    candidate = config.landmarker_asset_path
    if candidate is None:
        env = os.environ.get(_ENV_LANDMARKER)
        candidate = Path(env) if env else None
    if candidate is None:
        raise AssetNotFoundError(
            "MediaPipe face landmarker asset not configured. Set "
            "DetectConfig.landmarker_asset_path or $" + _ENV_LANDMARKER + ". "
            "Download: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
    candidate = Path(candidate)
    if not candidate.is_file():
        raise AssetNotFoundError(f"face landmarker asset not found at {candidate}")
    return candidate


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_image(source: str | Path | np.ndarray, config: IOConfig | None = None) -> LoadedImage:
    """Decode and bring an image to working resolution.

    Downscale only. A source smaller than the working resolution keeps its
    native size: upsampling would invent high-frequency content, inflating
    ``roughness`` and manufacturing spot detections out of resampling ringing.
    The quality gate flags such an image ``low_resolution`` instead.
    """
    config = config or IOConfig()

    if isinstance(source, np.ndarray):
        image = source
        if image.ndim != 3 or image.shape[2] != 3:
            raise ImageLoadError(f"expected an (H, W, 3) BGR array, got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ImageLoadError(f"expected uint8, got {image.dtype}")
        image = np.ascontiguousarray(image)
    else:
        path = Path(source)
        if not path.is_file():
            raise ImageLoadError(f"no such image: {path}")
        # cv2 honours the EXIF orientation tag by default; IMREAD_IGNORE_ORIENTATION
        # opts out. Either way the choice is explicit and recorded in config.
        flags = cv2.IMREAD_COLOR
        if not config.apply_exif_orientation:
            flags |= cv2.IMREAD_IGNORE_ORIENTATION
        image = cv2.imread(str(path), flags)
        if image is None:
            raise ImageLoadError(f"could not decode image: {path}")

    source_size = (image.shape[0], image.shape[1])
    long_edge = max(source_size)
    limit = config.max_long_edge_px

    if limit is not None and long_edge > limit:
        scale = limit / long_edge
        # Round rather than truncate so a 2:3 portrait keeps its aspect ratio
        # to within a pixel.
        new_size = (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        )
        image = cv2.resize(image, new_size, interpolation=_interpolation(config))
    else:
        scale = 1.0

    return LoadedImage(
        image=np.ascontiguousarray(image),
        source_size=source_size,
        working_size=(image.shape[0], image.shape[1]),
        scale=scale,
    )


def _interpolation(config: IOConfig) -> int:
    # Only INTER_AREA is permitted: it integrates over the source footprint
    # instead of point-sampling, which is what keeps the texture band stable
    # across capture resolutions. Config types this as a literal, so anything
    # else means the config was constructed by hand and bypassed typing.
    if config.downscale_interpolation != "area":
        raise ValueError(
            f"unsupported downscale_interpolation {config.downscale_interpolation!r}; "
            "only 'area' preserves the texture band across resolutions"
        )
    return cv2.INTER_AREA


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _landmarker_options(asset: str, max_faces: int, min_confidence: float):
    """Cached options object.

    Only the immutable options are cached — the landmarker itself is built per
    call. Caching an inference object would be shared mutable state.
    """
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    return vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=asset),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max_faces,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        # Off: they are learned expression coefficients, not measurements, and
        # they cost time we do not need to spend.
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )


def load_landmarker(config: Config | None = None):
    """Load a MediaPipe FaceLandmarker, for callers analysing many images.

    Mirrors ``load_parser``: the handle is returned explicitly and owned by the
    caller rather than cached in a module global, so there is no shared mutable
    state hidden inside the library.

    Building one costs ~182ms against ~24ms to actually detect, so a caller that
    lets ``detect_face`` build its own pays 8x the inference cost per frame.
    Reuse is safe: RunningMode.IMAGE is stateless by MediaPipe's contract, and
    18 detections across four fixtures in four interleaved orders were verified
    bit-identical against freshly built landmarkers.

    NOT documented thread-safe — use one landmarker per worker.

    The caller owns closing it. Used as a context manager or left to the
    interpreter, both are fine; it holds no OS resource beyond the model.
    """
    config = config or Config()
    from mediapipe.tasks.python import vision

    asset = resolve_landmarker_asset(config.detect)
    options = _landmarker_options(
        str(asset), config.detect.max_faces, config.detect.min_detection_confidence
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_face(
    image: np.ndarray, config: Config | None = None, landmarker=None
) -> Face | None:
    """Detect the primary face and its landmarks.

    Returns ``None`` when no face is found — the quality gate turns that into a
    ``no_face`` flag. Asset and decode problems raise instead: they are bugs in
    the deployment, not properties of the photo.

    Pass ``landmarker`` (from ``load_landmarker``) when analysing many images.
    Without it a fresh one is built and closed per call, which costs ~182ms
    against ~24ms of actual inference.
    """
    config = config or Config()
    detect_config = config.detect
    asset = resolve_landmarker_asset(detect_config)

    import mediapipe as mp
    from mediapipe.tasks.python import vision

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    if landmarker is not None:
        detection = landmarker.detect(mp_image)
    else:
        options = _landmarker_options(
            str(asset), detect_config.max_faces, detect_config.min_detection_confidence
        )
        with vision.FaceLandmarker.create_from_options(options) as owned:
            detection = owned.detect(mp_image)

    faces = detection.face_landmarks
    if not faces:
        return None

    all_points = [_to_pixels(face, width, height) for face in faces]
    index = _primary_index(all_points, detect_config)
    points = all_points[index]

    bbox = _bbox(points, width, height)
    return Face(
        landmarks=points,
        landmarks_z=_to_depth(faces[index], width),
        bbox=bbox,
        image_size=(height, width),
        n_faces=len(faces),
        area_fraction=(bbox[2] * bbox[3]) / float(width * height),
    )


def _to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    """Normalised landmarks -> (N, 2) float32 pixel coordinates.

    Kept unrounded: region polygons are rasterised from these, and rounding
    here would quantise every boundary to the pixel grid twice.
    """
    return np.array(
        [(point.x * width, point.y * height) for point in landmarks],
        dtype=np.float32,
    )


def _to_depth(landmarks, width: int) -> np.ndarray:
    """Landmark depth as an (N,) float32 array, in the same units as x.

    MediaPipe reports z on roughly the x scale, measured from an origin near the
    head centre, negative toward the camera. It is a LEARNED estimate from a
    single view, not a measurement: good for the gross shape of a face — the
    nose stands proud, the cheeks fall away — and useless for relief at the
    scale of a wrinkle. Scaled by width alone, never by height, so the aspect
    ratio of the working image cannot distort it.
    """
    return np.array([point.z * width for point in landmarks], dtype=np.float32)


def _primary_index(all_points: list[np.ndarray], config: DetectConfig) -> int:
    """Pick the analysed face when several were detected.

    Deterministic in both modes: "first" is MediaPipe's own order, and
    "largest" breaks ties by that same order via argmax's first-max rule.
    """
    if config.primary_face == "first" or len(all_points) == 1:
        return 0
    areas = [
        float(np.ptp(points[:, 0]) * np.ptp(points[:, 1])) for points in all_points
    ]
    return int(np.argmax(areas))


def _bbox(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Axis-aligned landmark bounds, clipped to the frame, as (x, y, w, h)."""
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    x0 = int(np.clip(np.floor(x0), 0, width - 1))
    y0 = int(np.clip(np.floor(y0), 0, height - 1))
    x1 = int(np.clip(np.ceil(x1), x0 + 1, width))
    y1 = int(np.clip(np.ceil(y1), y0 + 1, height))
    return (x0, y0, x1 - x0, y1 - y0)
