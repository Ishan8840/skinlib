"""Landmark-derived region masks.

Two entry points:

``build_region_polygons`` needs only landmarks. It is used by the parsing stage
(which must know where the lower face is before it can suppress facial hair)
and by anything that wants region geometry without a skin mask.

``build_regions`` intersects those polygons with the skin mask and resolves
overlaps by priority, producing the mutually exclusive masks the metrics run
on. Every returned region is a subset of the skin mask, so no metric can read a
pixel that parsing rejected.

Regions are constructed in the anatomical frame from ``landmarks.py``, not in
image axes: a tilted capture must yield the same anatomy.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import landmarks as lm
from .config import Config
from .types import REGION_NAMES, Face

__all__ = ["build_region_polygons", "build_regions"]


# ---------------------------------------------------------------------------
# rasterisation helpers
# ---------------------------------------------------------------------------


def _blank(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def _fill_hull(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Rasterise the convex hull of a point set."""
    canvas = _blank(shape)
    if len(points) < 3:
        return canvas.astype(bool)
    hull = cv2.convexHull(np.round(points).astype(np.int32))
    cv2.fillConvexPoly(canvas, hull, 1)
    return canvas.astype(bool)


def _fill_polygon(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Rasterise a (possibly concave) closed polygon in landmark order."""
    canvas = _blank(shape)
    if len(points) < 3:
        return canvas.astype(bool)
    cv2.fillPoly(canvas, [np.round(points).astype(np.int32)], 1)
    return canvas.astype(bool)


def _morph(mask: np.ndarray, radius: int, dilate: bool) -> np.ndarray:
    """Dilate or erode by a disc of the given pixel radius."""
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    op = cv2.dilate if dilate else cv2.erode
    return op(mask.astype(np.uint8), kernel).astype(bool)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return _morph(mask, radius, dilate=True)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    return _morph(mask, radius, dilate=False)


class _Frame2D:
    """Adapter over the 2D anatomical frame: index-based projection, pixel units."""

    def __init__(self, frame: lm.FaceFrame, points: np.ndarray) -> None:
        self._frame = frame
        self._points = points

    def project(self, indices) -> tuple[np.ndarray, np.ndarray]:
        return self._frame.project(np.atleast_2d(self._points[list(indices)]))

    @property
    def height(self) -> float:
        return float(self._frame.height)


class _Frame3D:
    """Adapter over the canonical frame: index-based projection, interocular units.

    Same interface as ``_Frame2D`` so the region code below reads identically
    either way — which is the point. The two differ only in whether a threshold
    is a distance in the image plane or on the face itself.
    """

    def __init__(self, frame, points3d: np.ndarray) -> None:
        self._frame = frame
        self._points = points3d

    def project(self, indices) -> tuple[np.ndarray, np.ndarray]:
        return self._frame.project(np.atleast_2d(self._points[list(indices)]))

    @property
    def height(self) -> float:
        vertical = self.project([lm.FOREHEAD_TOP, lm.CHIN_BOTTOM])[1]
        return float(abs(vertical[1] - vertical[0]))


def _coordinate_fields(shape: tuple[int, int], frame: lm.FaceFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel (lateral, vertical) coordinates in the anatomical frame."""
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width]
    dx = xs - frame.origin[0]
    dy = ys - frame.origin[1]
    lateral = dx * frame.lat[0] + dy * frame.lat[1]
    vertical = dx * frame.down[0] + dy * frame.down[1]
    return lateral, vertical


# ---------------------------------------------------------------------------
# polygons
# ---------------------------------------------------------------------------


# Two-entry memo for the polygon build, which is the single most expensive step
# in a frame at ~253ms and was being run TWICE per frame: `_suppress_facial_hair`
# needs region geometry to know where the lower face is, and `build_regions`
# then rebuilds the same polygons from the same face and config moments later.
#
# Keyed on object identity, and the entry holds strong references to both keys.
# That is what makes id() safe here — a live reference cannot be collected, so
# its id cannot be recycled onto a different object while the entry stands.
# Two entries rather than one so a caller alternating between two configs (the
# A/B pattern used throughout this library's own tooling) still hits.
_POLYGON_MEMO: list[tuple[Face, Config, dict[str, np.ndarray]]] = []
_POLYGON_MEMO_SIZE = 2


def build_region_polygons(face: Face, config: Config | None = None) -> dict[str, np.ndarray]:
    """Region geometry from landmarks alone, before any skin-mask intersection.

    Returns one boolean mask per name in ``REGION_NAMES``. Overlaps are still
    present at this stage; ``build_regions`` resolves them.

    Memoised on the identity of ``face`` and ``config``; see ``_POLYGON_MEMO``.
    The returned masks are treated as read-only by every caller in this package.
    """
    config = config or Config()

    for cached_face, cached_config, polygons in _POLYGON_MEMO:
        if cached_face is face and cached_config is config:
            return polygons

    polygons = _build_region_polygons(face, config)
    _POLYGON_MEMO.append((face, config, polygons))
    del _POLYGON_MEMO[:-_POLYGON_MEMO_SIZE]
    return polygons


def _build_region_polygons(face: Face, config: Config) -> dict[str, np.ndarray]:
    """The actual build. Uncached, so tests can measure it directly."""
    region_config = config.regions

    shape = face.image_size
    points = face.landmarks.astype(np.float64)
    frame = lm.face_frame(face.landmarks)

    # Canonical (3D) coordinates when depth is available and enabled: a basis
    # carried by the face itself does not foreshorten when the head turns, so a
    # region threshold keeps landing on the same anatomy. The 2D frame is an
    # image-plane projection and does not. Falls back silently when there is no
    # depth, so old detections still work.
    projector: _Frame2D | _Frame3D = _Frame2D(frame, points)
    if region_config.canonical_frame:
        from .canonical import canonical_fields, canonical_frame as _canonical_frame
        from .canonical import _points_3d

        canonical = _canonical_frame(face)
        fields = canonical_fields(face, shape) if canonical is not None else None
        points_3d = _points_3d(face)
        if canonical is not None and fields is not None and points_3d is not None:
            lateral, vertical = fields
            projector = _Frame3D(canonical, points_3d)
        else:
            lateral, vertical = _coordinate_fields(shape, frame)
    else:
        lateral, vertical = _coordinate_fields(shape, frame)

    oval = _fill_polygon(shape, points[list(lm.FACE_OVAL)])

    left_eye = _fill_hull(shape, points[list(lm.LEFT_EYE)])
    right_eye = _fill_hull(shape, points[list(lm.RIGHT_EYE)])
    left_brow = _fill_hull(shape, points[list(lm.LEFT_BROW)])
    right_brow = _fill_hull(shape, points[list(lm.RIGHT_BROW)])
    lips = _fill_hull(shape, points[list(lm.LIPS_OUTER)])
    nose = _fill_hull(shape, points[list(lm.NOSE_HULL)])

    regions: dict[str, np.ndarray] = {}

    brow_top_vert = float(projector.project(list(lm.LEFT_BROW) + list(lm.RIGHT_BROW))[1].min())

    # -- periorbital: a ring around the eye, excluding the eye and the brow --
    ring_radius = max(1, int(round(region_config.periorbital_outer_frac * frame.interocular)))
    for name, eye, brow in (
        ("periorbital_left", left_eye, left_brow),
        ("periorbital_right", right_eye, right_brow),
    ):
        ring = dilate(eye, ring_radius) & ~eye & ~dilate(brow, 2)
        # Stop at the brow line. Skin above the brow is forehead: it is lit
        # differently, ages differently, and pooling it with under-eye skin
        # would dilute the dark-circle signal this region exists to capture.
        regions[name] = ring & oval & (vertical >= brow_top_vert)

    # -- glabella: between the inner brow ends, down to the nasion --
    inner_lat, inner_vert = projector.project([lm.BROW_INNER_RIGHT, lm.BROW_INNER_LEFT])
    centre_lat = float(inner_lat.mean())
    half_width = 0.5 * abs(float(inner_lat[1] - inner_lat[0])) * region_config.glabella_width_frac
    brow_vert = float(inner_vert.min())
    nasion_vert = float(projector.project([lm.NASION])[1][0])
    # Extend below the nasion by the configured fraction of the brow-to-nasion
    # span, so the glabella is a band rather than a hairline sliver.
    span = max(nasion_vert - brow_vert, 1.0)
    glabella = (
        (np.abs(lateral - centre_lat) <= half_width)
        & (vertical >= brow_vert - span * region_config.glabella_height_frac)
        & (vertical <= nasion_vert)
    )
    regions["glabella"] = glabella & oval & ~left_brow & ~right_brow

    # -- forehead: extrapolated upward from the brow line --
    # FaceMesh has no landmarks above the brows, so the top edge is geometric.
    # The real upper bound is the hairline, which the skin mask supplies.
    forehead = (vertical <= brow_top_vert) & (
        vertical >= brow_top_vert - region_config.forehead_height_frac * projector.height
    )
    regions["forehead"] = forehead & oval

    # -- nose --
    regions["nose"] = nose & oval

    # -- perioral: a ring around the lips --
    ring = max(1, int(round(region_config.perioral_ring_frac * frame.mouth_width)))
    regions["perioral"] = dilate(lips, ring) & ~lips & oval

    # -- chin: below the lower lip --
    lip_bottom_vert = float(projector.project([lm.LIP_BOTTOM])[1][0])
    regions["chin"] = (vertical > lip_bottom_vert) & oval & ~lips

    # -- cheeks: below the eyes, above the mouth, either side of the midline --
    lid_vert = float(projector.project([lm.LOWER_LID_RIGHT, lm.LOWER_LID_LEFT])[1].max())
    # Split on the facial midline and subtract the nose, rather than cutting at
    # the nose alae. Under head yaw the far ala projects laterally past the far
    # cheek, so an ala-based boundary empties that cheek entirely — a turned
    # head produced a zero-pixel cheek on a fixture that plainly had one.
    midline_lat = float(projector.project([lm.NASION, 1, lm.LIP_BOTTOM])[0].mean())
    nose_block = dilate(nose, 2)
    inset = max(0, int(round(region_config.cheek_inset_frac * frame.height)))
    # Inset from the silhouette: grazing light at the face edge produces a
    # luminance falloff that is geometry, not skin.
    oval_inset = erode(oval, inset)
    band = (vertical >= lid_vert) & (vertical <= lip_bottom_vert) & oval_inset & ~nose_block
    regions["left_cheek"] = band & (lateral > midline_lat)
    regions["right_cheek"] = band & (lateral < midline_lat)

    # Trim every polygon so adjacent regions do not share a boundary pixel and
    # so no region hugs an anatomical edge it was not meant to include.
    trim = region_config.polygon_erode_px
    if trim > 0:
        regions = {name: erode(mask, trim) for name, mask in regions.items()}

    missing = set(REGION_NAMES) - set(regions)
    if missing:  # pragma: no cover - guards a future edit, not a runtime path
        raise AssertionError(f"region builder produced no mask for {sorted(missing)}")
    return {name: regions[name] for name in REGION_NAMES}


def nostril_polygons(face: Face, config: Config | None = None) -> np.ndarray:
    """Mask bracketing both nostril openings, for the parsing stage to carve."""
    points = face.landmarks.astype(np.float64)
    shape = face.image_size
    left = _fill_hull(shape, points[list(lm.LEFT_NOSTRIL)])
    right = _fill_hull(shape, points[list(lm.RIGHT_NOSTRIL)])
    return left | right


def feature_polygons(face: Face) -> np.ndarray:
    """Eyes, brows and lips from landmarks, for the parsing stage to carve."""
    points = face.landmarks.astype(np.float64)
    shape = face.image_size
    mask = np.zeros(shape, dtype=bool)
    for indices in (lm.LEFT_EYE, lm.RIGHT_EYE, lm.LEFT_BROW, lm.RIGHT_BROW, lm.LIPS_OUTER):
        mask |= _fill_hull(shape, points[list(indices)])
    return mask


def face_oval_polygon(face: Face) -> np.ndarray:
    """The landmark face silhouette: forehead and jaw in, scalp and neck out."""
    points = face.landmarks.astype(np.float64)
    return _fill_polygon(face.image_size, points[list(lm.FACE_OVAL)])


# ---------------------------------------------------------------------------
# final masks
# ---------------------------------------------------------------------------


def build_regions(
    face: Face,
    skin_mask: np.ndarray,
    config: Config | None = None,
) -> dict[str, np.ndarray]:
    """Region masks, intersected with the skin mask and made exclusive.

    Priority order comes from ``RegionConfig.priority``: the first region to
    claim a pixel keeps it. Periorbital outranks the cheeks deliberately — the
    cheek band reaches up under the eye, and under-eye skin is the dark-circle
    signal. Letting the much larger cheek win would average that signal away.
    """
    config = config or Config()
    region_config = config.regions

    if skin_mask.shape != face.image_size:
        raise ValueError(
            f"skin mask shape {skin_mask.shape} does not match image size {face.image_size}"
        )

    polygons = build_region_polygons(face, config)
    regions = {name: mask & skin_mask for name, mask in polygons.items()}

    if not region_config.exclusive:
        return regions

    unknown = set(region_config.priority) ^ set(REGION_NAMES)
    if unknown:
        raise ValueError(
            f"RegionConfig.priority must cover exactly the known regions; differs by {sorted(unknown)}"
        )

    claimed = np.zeros(skin_mask.shape, dtype=bool)
    exclusive: dict[str, np.ndarray] = {}
    for name in region_config.priority:
        mask = regions[name] & ~claimed
        exclusive[name] = mask
        claimed |= mask

    return {name: exclusive[name] for name in REGION_NAMES}
