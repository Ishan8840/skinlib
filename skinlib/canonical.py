"""A face-intrinsic 3D frame, and pose-invariant coordinate fields built on it.

``regions.py`` cuts its regions with thresholds on a **2D** projected frame. That
frame is anatomical in the image plane, but it is still a projection: turn the
head and the projection foreshortens asymmetrically, so a threshold like
``lateral > midline_lat`` lands on different anatomy than it did before. The
metric then moves because the region moved, not because the skin changed.

That is the leading explanation for the angle penalty in the error budget —
`spot_burden` costs 5.9x and `uniformity` 6.3x under varied head angle — and it
is why correcting by surface incidence failed: incidence correlates with the
metrics (mean |r| = 0.51) but with an inconsistent SIGN between regions, which is
the signature of masks sliding rather than of a photometric factor.

The fix is to stop measuring position in the image plane. A basis built from the
face's own landmarks in 3D rotates WITH the head, so a coordinate expressed in it
does not change when the head turns:

* **lateral** — outer eye corner to outer eye corner;
* **vertical** — nasion to chin, orthogonalised against lateral;
* **normal** — their cross product.

Coordinates are divided by interocular distance, so they are also invariant to
capture distance and to the working resolution.

Per-pixel coordinates come from barycentric interpolation over the landmark
triangulation: a pixel inside a triangle takes the weighted canonical coordinate
of its three vertices. The triangulation is computed on the CANONICAL points, so
its topology is a property of face anatomy rather than of this particular pose.

Scope: this fixes *where a region is*. It does not correct foreshortened
sampling density, and it cannot recover skin the pose has hidden.
"""

from __future__ import annotations

import numpy as np

from . import landmarks as lm
from .types import Face

__all__ = ["CanonicalFrame", "canonical_fields", "canonical_frame"]


class CanonicalFrame:
    """An orthonormal basis carried by the face itself.

    ``project`` maps image-space landmark indices to (lateral, vertical) in units
    of interocular distance, with the origin at the nasion. Positive lateral is
    the subject's left, positive vertical is downward — matching the existing 2D
    frame's conventions so region code reads the same in either.
    """

    __slots__ = ("origin", "lateral_axis", "vertical_axis", "normal", "scale")

    def __init__(
        self,
        origin: np.ndarray,
        lateral_axis: np.ndarray,
        vertical_axis: np.ndarray,
        normal: np.ndarray,
        scale: float,
    ) -> None:
        self.origin = origin
        self.lateral_axis = lateral_axis
        self.vertical_axis = vertical_axis
        self.normal = normal
        self.scale = scale

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) points -> (lateral, vertical), both scale-normalised."""
        centred = np.asarray(points, dtype=np.float64) - self.origin
        return (
            (centred @ self.lateral_axis) / self.scale,
            (centred @ self.vertical_axis) / self.scale,
        )


def _points_3d(face: Face) -> np.ndarray | None:
    """(N, 3) landmarks, or None when the detection carried no depth."""
    if face.landmarks_z is None or face.landmarks_z.size == 0:
        return None
    count = min(len(face.landmarks), len(face.landmarks_z))
    return np.column_stack(
        [face.landmarks[:count].astype(np.float64), face.landmarks_z[:count].astype(np.float64)]
    )


def canonical_frame(face: Face) -> CanonicalFrame | None:
    """Build the face's own 3D basis. None when depth is unavailable.

    Gram-Schmidt rather than raw landmark vectors: the eye line and the
    nasion-chin line are not exactly perpendicular on a real face, and a
    non-orthogonal basis would shear the coordinates by an amount that varies
    from person to person.
    """
    points = _points_3d(face)
    if points is None or len(points) <= max(lm.CHIN_BOTTOM, lm.LEFT_EYE_OUTER):
        return None

    right_eye = points[lm.RIGHT_EYE_OUTER]
    left_eye = points[lm.LEFT_EYE_OUTER]
    nasion = points[lm.NASION]
    chin = points[lm.CHIN_BOTTOM]

    lateral = left_eye - right_eye
    scale = float(np.linalg.norm(lateral))
    if scale < 1e-6:
        return None
    lateral = lateral / scale

    down = chin - nasion
    # Remove the component along lateral, leaving a true perpendicular.
    vertical = down - (down @ lateral) * lateral
    magnitude = float(np.linalg.norm(vertical))
    if magnitude < 1e-6:
        return None
    vertical = vertical / magnitude

    normal = np.cross(lateral, vertical)
    normal_magnitude = float(np.linalg.norm(normal))
    if normal_magnitude < 1e-6:
        return None

    return CanonicalFrame(nasion, lateral, vertical, normal / normal_magnitude, scale)


def canonical_fields(
    face: Face, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-pixel ``(lateral, vertical)`` in the face's own frame.

    Returns NaN outside the landmark hull, where there is no triangle to
    interpolate over and therefore no anatomical coordinate to report. Callers
    must treat NaN as "not on the face" — a comparison against NaN is False,
    which gives the right answer for every region threshold by construction.

    Returns None when depth is unavailable, so the caller can fall back to the
    2D frame rather than fail.
    """
    frame = canonical_frame(face)
    points = _points_3d(face)
    if frame is None or points is None:
        return None

    lateral_values, vertical_values = frame.project(points)

    from scipy.interpolate import LinearNDInterpolator

    height, width = shape
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)

    # Barycentric interpolation over the landmark triangulation in IMAGE space,
    # carrying CANONICAL values. Locating a pixel has to happen where pixels
    # live; what gets interpolated is the pose-invariant coordinate. Both fields
    # go through one interpolator so the triangulation is built once.
    interpolator = LinearNDInterpolator(
        points[:, :2], np.column_stack([lateral_values, vertical_values])
    )
    values = interpolator(pixels)

    return (
        values[:, 0].reshape(shape),
        values[:, 1].reshape(shape),
    )
