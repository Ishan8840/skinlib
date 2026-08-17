"""FaceMesh landmark index constants and the face coordinate frame.

Index sets are properties of the FaceMesh topology, which is fixed. They are
constants, not tunables, so they live here rather than in ``Config``.

LEFT AND RIGHT ARE ANATOMICAL — the subject's left and right, matching
MediaPipe's own naming. In an unmirrored frontal photo the subject's left
appears on the right of the image. This matters for longitudinal tracking: a
region labelled ``left_cheek`` must mean the same cheek in every session, no
matter how the capture was framed or whether the front camera mirrored it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FACE_OVAL",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_BROW",
    "RIGHT_BROW",
    "LEFT_IRIS",
    "RIGHT_IRIS",
    "LIPS_OUTER",
    "NOSE_HULL",
    "LEFT_NOSTRIL",
    "RIGHT_NOSTRIL",
    "N_LANDMARKS_BASE",
    "N_LANDMARKS_REFINED",
    "FaceFrame",
    "face_frame",
]

N_LANDMARKS_BASE = 468
N_LANDMARKS_REFINED = 478

# Iris landmarks exist only when DetectConfig.refine_landmarks is on. They are
# what makes the sclera reference possible: without them the iris cannot be
# excluded, and the iris is emphatically not neutral.
LEFT_IRIS: tuple[int, ...] = (473, 474, 475, 476, 477)
RIGHT_IRIS: tuple[int, ...] = (468, 469, 470, 471, 472)

# Silhouette, clockwise from the forehead midpoint.
FACE_OVAL: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)

# Eyelid contours (subject's left / right).
LEFT_EYE: tuple[int, ...] = (
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
)
RIGHT_EYE: tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
)

LEFT_BROW: tuple[int, ...] = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
RIGHT_BROW: tuple[int, ...] = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)

LIPS_OUTER: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185,
)

# Bridge midline + alae + nostril rims. Taken as a convex hull, this gives the
# external nose without reaching onto the cheeks.
NOSE_HULL: tuple[int, ...] = (
    6, 197, 195, 5, 4, 1, 19, 94, 2, 122, 351, 115, 344, 45, 275, 220, 440,
    129, 358, 98, 327,
)

# Nostril openings. Verified by rendering indices over a fixture rather than
# copied from memory: these bracket the opening on each side, and the carve
# intersects the hull with a darkness test so only the actual dark aperture is
# removed rather than a fixed blob of ala skin.
#
# The two sets are exact mirror pairs across the mesh midline
# (98/327, 97/326, 99/328, 240/460, 75/305, 60/290, 20/250, 242/462, 241/461,
#  238/458, 239/459, 237/457, 220/440, 45/275, 44/274).
RIGHT_NOSTRIL: tuple[int, ...] = (
    98, 97, 99, 240, 75, 60, 20, 242, 241, 238, 239, 237, 220, 45, 44,
)
LEFT_NOSTRIL: tuple[int, ...] = (
    327, 326, 328, 460, 305, 290, 250, 462, 461, 458, 459, 457, 440, 275, 274,
)

# Single-point anchors used to build the face frame.
CHIN_BOTTOM = 152
FOREHEAD_TOP = 10
RIGHT_EYE_OUTER = 33
LEFT_EYE_OUTER = 263
NASION = 6
BROW_INNER_RIGHT = 55
BROW_INNER_LEFT = 285
LIP_BOTTOM = 17
ALA_RIGHT = 129
ALA_LEFT = 358
LOWER_LID_RIGHT = 145
LOWER_LID_LEFT = 374


@dataclass(frozen=True, eq=False)
class FaceFrame:
    """An anatomical coordinate frame derived from the landmarks.

    Regions are defined in this frame rather than in image axes, so a tilted
    capture yields the same anatomy. ``lat`` points toward the subject's LEFT,
    ``down`` toward the chin; both are unit vectors in pixel space.
    """

    origin: np.ndarray  # (2,) the face centre
    lat: np.ndarray  # (2,) unit, toward subject's left
    down: np.ndarray  # (2,) unit, toward the chin
    height: float  # forehead-top to chin, px
    interocular: float  # eye centre to eye centre, px
    mouth_width: float  # px

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates -> (lateral, vertical) frame coordinates."""
        delta = points - self.origin
        return delta @ self.lat, delta @ self.down


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.hypot(vector[0], vector[1]))
    if norm < 1e-6:
        raise ValueError("degenerate landmark geometry: zero-length face axis")
    return (vector / norm).astype(np.float64)


def face_frame(landmarks: np.ndarray) -> FaceFrame:
    """Build the anatomical frame from a landmark array."""
    points = landmarks.astype(np.float64)

    # Lateral axis from eye corner to eye corner: robust to head tilt, and its
    # sign is anatomical rather than image-relative.
    lat = _unit(points[LEFT_EYE_OUTER] - points[RIGHT_EYE_OUTER])
    # Vertical axis forced perpendicular to lateral, so the frame is
    # orthonormal and the two projections stay independent.
    raw_down = points[CHIN_BOTTOM] - points[FOREHEAD_TOP]
    down = _unit(raw_down - (raw_down @ lat) * lat)

    left_eye_centre = points[list(LEFT_EYE)].mean(axis=0)
    right_eye_centre = points[list(RIGHT_EYE)].mean(axis=0)

    return FaceFrame(
        origin=points[list(FACE_OVAL)].mean(axis=0),
        lat=lat,
        down=down,
        height=float(np.linalg.norm(points[CHIN_BOTTOM] - points[FOREHEAD_TOP])),
        interocular=float(np.linalg.norm(left_eye_centre - right_eye_centre)),
        mouth_width=float(np.linalg.norm(points[291] - points[61])),
    )
