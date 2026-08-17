"""Melanin / haemoglobin separation in optical-density space.

Skin absorbance is, to a good approximation, linear in two pigment densities:

    OD(x, y) = c_m(x, y) * s_m  +  c_h(x, y) * s_h  +  shading(x, y) * (1,1,1)

where ``OD = -log10(reflectance)`` per channel and ``s_m``, ``s_h`` are fixed
melanin and haemoglobin absorbance directions (Tsumura et al., *Independent
component analysis of skin color image*, JOSA A 1999).

The useful structure is the third term. Shading and exposure are **neutral
multiplicative gains** on the raw channels, so in optical density they are a
pure additive offset along ``(1,1,1)`` — regardless of magnitude, geometry, or
skin tone. Projecting that direction out removes them *analytically*, not
statistically: no fitting, no reference patch, no self-normalisation.

What remains is a 2D plane, and ``c_m``/``c_h`` are the coordinates of the
residual in the (non-orthogonal) ``s_m``, ``s_h`` basis. Coordinates, not
projections — a dot product against a non-orthogonal basis is not the
coefficient, and using one silently mixes the two pigments.

Measured on ``portrait_a``, response to a pure shading gradient (0.55x..1.0x
across the patch) and to a +1/3 stop exposure change, both of which are zero
pigment change and should read as zero:

    signal                   shading      exposure
    melanin_index (old)      +0.0875      -0.0754
    melanin_density          -0.0089      +0.0027
    hemoglobin_density       +0.0346      -0.0134

Roughly 10x better on shading and 28x on exposure for melanin. The residual is
uint8 clipping at the dark end of the gradient, not a limitation of the method.

**This buys invariance to shading and exposure, NOT to white balance.** A
neutral gain cancels; a per-channel colour cast does not, because a cast moves
the very channel ratios these coordinates measure. That is the opposite trade
from the self-normalised ``_rel`` metrics, which cancel a cast (both terms shift
together) but also cancel any face-wide pigment change. The two families are
complementary and both are reported — see ``types.METRIC_NAMES``. Run these on
the colour-corrected image and check ``ColorResult.estimator`` before trusting a
cross-session delta.

Nothing here is fitted or trained. The basis is a published constant and the
arithmetic is closed-form.
"""

from __future__ import annotations

import numpy as np

from .config import MetricsConfig

__all__ = [
    "deshade",
    "optical_density",
    "separate_chromophores",
]

# Unit vector along the shading direction. A neutral gain k on every channel
# adds -log10(k) * (1,1,1) to the optical density, so the entire effect of
# shading and exposure lies along this one axis.
_SHADING_AXIS = np.ones(3, dtype=np.float64) / np.sqrt(3.0)


def optical_density(image: np.ndarray, config: MetricsConfig | None = None) -> np.ndarray:
    """BGR uint8 -> per-channel optical density ``-log10(reflectance)``, RGB order.

    Returned in RGB order because the absorbance basis vectors are quoted that
    way in the literature; keeping the array in BGR here would make the basis
    constants silently wrong in a way no test would catch.

    The floor bounds the density of a crushed pixel instead of letting it run to
    infinity. It is the same floor the melanin index uses, so the two signals
    agree about what a black pixel means.
    """
    config = config or MetricsConfig()
    reflectance = np.clip(
        image.astype(np.float64)[..., ::-1] / 255.0, config.chromophore_od_floor, 1.0
    )
    return -np.log10(reflectance)


def deshade(density: np.ndarray) -> np.ndarray:
    """Remove the ``(1,1,1)`` component: shading and exposure, exactly.

    Works on any trailing-axis-3 array. This is an orthogonal projection, so it
    is idempotent and cannot amplify noise.
    """
    along = density @ _SHADING_AXIS
    return density - along[..., None] * _SHADING_AXIS


def _basis(config: MetricsConfig) -> np.ndarray:
    """The two absorbance axes, de-shaded and stacked as a 3x2 matrix.

    De-shaded first so the decomposition happens entirely inside the plane the
    residual lives in. Without this the basis would have a component along an
    axis that has already been removed from the data, and the resulting
    coordinates would not reconstruct the residual.
    """
    axes = np.stack(
        [
            np.asarray(config.chromophore_melanin_axis, dtype=np.float64),
            np.asarray(config.chromophore_hemoglobin_axis, dtype=np.float64),
        ],
        axis=1,
    )
    return deshade(axes.T).T


def _coordinate_operator(basis: np.ndarray) -> np.ndarray:
    """Left inverse of a 3x2 basis, as a 2x3 matrix. Closed form, no LAPACK.

    ``(B^T B)^-1 B^T``, with the 2x2 inverse written out. ``np.linalg.pinv``
    would compute this via SVD, whose exact output can shift with the linked
    BLAS — and this library's determinism guarantee is the whole point of it.
    Two vectors need no SVD.
    """
    gram = basis.T @ basis
    (a, b), (c, d) = gram
    determinant = a * d - b * c
    if abs(determinant) < 1e-12:
        raise ValueError(
            "melanin and haemoglobin axes are collinear after de-shading; "
            "the separation is undefined for this basis"
        )
    inverse = np.array([[d, -b], [-c, a]], dtype=np.float64) / determinant
    return inverse @ basis.T


def separate_chromophores(
    image: np.ndarray, config: MetricsConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """BGR uint8 -> ``(melanin_density, hemoglobin_density)`` maps.

    Both are float64, same height and width as the image, in arbitrary but
    fixed density units: higher means more pigment. They are computed for every
    pixel; masking to skin is the caller's job, because the maps are also useful
    over a neighbourhood the mask excludes (spot backgrounds are estimated
    across mask boundaries).

    Physically meaningless off skin — hair, cloth and background are not a
    two-chromophore medium, and their coordinates are extrapolation.
    """
    config = config or MetricsConfig()
    operator = _coordinate_operator(_basis(config))
    residual = deshade(optical_density(image, config))
    coordinates = residual @ operator.T
    return coordinates[..., 0], coordinates[..., 1]
