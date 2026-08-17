"""Face surface geometry from landmark depth, and what it is good for.

MediaPipe returns a z for every landmark and the library used to throw it away.
It carries the gross shape of a face — the nose stands proud, the cheeks fall
away, the jaw turns under — which is enough to answer two questions no
two-dimensional signal can:

* **which way is a patch of skin facing?** A cheek at 40 degrees to the camera is
  foreshortened, sampled at fewer pixels per square millimetre, and lit at a
  different angle than the forehead beside it. Every per-region metric silently
  assumes these are comparable, and they are not.
* **where is the light?** Given normals and observed brightness, the illuminant
  direction is a least-squares fit rather than a guess, which turns `side_lit`
  from a flag into a number with a direction attached.

**What this is not.** MediaPipe's z is a LEARNED estimate from a single view,
not a measurement. It is smooth and sparse — 478 vertices over a whole face —
so it describes curvature at the scale of a cheek and carries nothing at the
scale of a pore or a wrinkle. Anything claiming to measure relief, line depth or
lesion volume needs real depth (a TrueDepth stream, photometric stereo, or
multi-view), and this module deliberately does not pretend otherwise.

Scale is arbitrary. z arrives in units of image width, so quantities here are
directions and cosines, never millimetres.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Face

__all__ = [
    "depth_surface",
    "estimate_light_direction",
    "incidence_map",
    "region_incidence",
    "surface_normals",
]

# Measured limit of what this buys, on the 13-frame `angle` capture set.
#
# Incidence IS a real covariate of the per-region metrics: mean |r| across
# regions came to 0.51 for spot_burden, 0.46 for inflammation_burden and 0.27
# for roughness, so roughly a quarter of the angle-set variance moves with it.
#
# It is NOT a correctable one. The SIGN is inconsistent between regions —
#
#   forehead   +0.50    nose             -0.85
#   perioral   +0.61    periorbital_left -0.80
#   chin       -0.54    periorbital_right+0.53
#
# — and a physical factor acting through foreshortening or shading would push
# the same way everywhere. Dividing by the cosine would therefore improve some
# regions and actively damage others.
#
# The likeliest explanation is that the dominant angle penalty is not
# photometric at all: `build_regions` places regions from 2D landmark geometry,
# so under out-of-plane rotation each mask slides over slightly different
# anatomy. The metric changes because the region moved, not because the light
# did. If so the real fix is to define regions in a 3D canonical face frame,
# which this depth field makes possible and which is a larger change than a
# correction factor.
#
# Use incidence to WEIGHT or to flag an obliquely-viewed region. Do not use it
# as a divisor.


def region_incidence(
    face: Face,
    regions: dict[str, np.ndarray],
    shape: tuple[int, int],
    direction: np.ndarray | None = None,
    min_pixels: int = 250,
) -> dict[str, float]:
    """Mean cosine between each region's normals and a direction.

    With the default camera axis this is how squarely a region faces the lens:
    1.0 dead-on, falling as it curves away. Measured on a real face the forehead
    reads 0.95 and the chin 0.63, which is the flat plane against the surface
    that turns under the jaw.

    Intended as a **weight or a flag**, not a divisor — see the note above on why
    correcting by it would help some regions and hurt others.
    """
    normals = surface_normals(face, shape)
    if normals is None:
        return {}
    axis = np.array([0.0, 0.0, 1.0]) if direction is None else np.asarray(direction)
    cosine = incidence_map(normals, axis)
    return {
        name: float(cosine[mask].mean())
        for name, mask in regions.items()
        if int(mask.sum()) >= min_pixels
    }


def depth_surface(face: Face, shape: tuple[int, int]) -> np.ndarray | None:
    """Dense depth over the face, interpolated from the landmark mesh.

    Linear interpolation over the Delaunay triangulation of the landmarks, with
    the convex hull filled and everything outside it left at the nearest edge
    value. Linear rather than cubic on purpose: a cubic fit through sparse,
    noisy vertices overshoots into ripples that then read as curvature the face
    does not have.

    Returns None when the detection carried no depth.
    """
    if face.landmarks_z is None or face.landmarks_z.size == 0:
        return None

    height, width = shape
    points = face.landmarks[: len(face.landmarks_z)].astype(np.float64)
    values = face.landmarks_z.astype(np.float64)

    from scipy.interpolate import griddata

    grid_y, grid_x = np.mgrid[0:height, 0:width]
    depth = griddata(points, values, (grid_x, grid_y), method="linear")
    # Outside the hull griddata leaves NaN. Nearest-neighbour fills it so the
    # gradient operator below never meets a hole; those pixels are outside the
    # skin mask anyway and are never measured.
    holes = ~np.isfinite(depth)
    if holes.any():
        filled = griddata(points, values, (grid_x, grid_y), method="nearest")
        depth[holes] = filled[holes]
    return depth.astype(np.float32)


def surface_normals(
    face: Face, shape: tuple[int, int], smooth_sigma: float | None = None
) -> np.ndarray | None:
    """Per-pixel unit surface normal, as an (H, W, 3) float32 array.

    Computed from the gradient of the interpolated depth: a surface z = f(x, y)
    has normal proportional to (-dz/dx, -dz/dy, 1). Sign convention is +z toward
    the camera, so a patch facing the lens has normal near (0, 0, 1).

    The depth is smoothed first, at a scale proportional to face width. The
    interpolation is piecewise-linear over triangles, so its raw gradient is
    piecewise-constant and discontinuous at every triangle edge — differentiating
    it unsmoothed would return the mesh topology rather than the face.
    """
    depth = depth_surface(face, shape)
    if depth is None:
        return None

    sigma = smooth_sigma if smooth_sigma is not None else max(face.width * 0.02, 1.0)
    smoothed = cv2.GaussianBlur(depth, (0, 0), sigma)

    dz_dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dz_dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(dz_dx)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals / np.maximum(norm, 1e-9)).astype(np.float32)


def estimate_light_direction(
    image: np.ndarray, normals: np.ndarray, skin_mask: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fit a single distant light direction to the observed shading.

    Returns ``(unit_direction, r_squared)``.

    Under a Lambertian model with roughly constant albedo, observed intensity is
    ``I = a * (n . l) + ambient``. With the normals known that is linear in the
    unknowns, so a least-squares solve over the skin pixels recovers the light
    direction and how much of the shading it actually explains.

    **``r_squared`` is the honest part.** Skin is not Lambertian, albedo is not
    constant — that variation is the very signal the rest of the library
    measures — and a room usually holds more than one light. A low value means
    the single-light model does not describe this capture, and the direction it
    returns should not be trusted.
    """
    if not skin_mask.any():
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), float("nan")

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    intensity = grey[skin_mask] / 255.0
    design = np.column_stack(
        [normals[skin_mask].astype(np.float64), np.ones(intensity.size)]
    )

    solution, *_ = np.linalg.lstsq(design, intensity, rcond=None)
    direction = solution[:3]
    magnitude = float(np.linalg.norm(direction))
    if magnitude < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0

    residual = intensity - design @ solution
    variance = float(np.var(intensity))
    r_squared = 1.0 - float(np.var(residual)) / variance if variance > 1e-12 else 0.0
    return direction / magnitude, r_squared


def incidence_map(normals: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """``cos`` of the angle between each normal and a direction, clipped at 0.

    With the camera axis this is foreshortening: 1 where skin faces the lens
    squarely, falling toward 0 as it turns away, and it is the factor by which a
    patch is sampled at fewer pixels per unit area. With the estimated light
    direction it is the Lambertian shading term.

    Clipped rather than signed because a surface turned past 90 degrees is not
    lit negatively, it is not lit at all — and it is not visible either.
    """
    cosine = normals @ np.asarray(direction, dtype=np.float32)
    return np.clip(cosine, 0.0, 1.0).astype(np.float32)
