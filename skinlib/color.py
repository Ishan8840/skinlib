"""Color constancy.

Two estimators of the scene illuminant, with precedence:

* **Sclera-referenced** (primary when confident). Eye-white is an actual
  neutral surface in the frame, so it measures the illuminant instead of
  assuming anything about the scene. It fails on closed, squinting or heavily
  shadowed eyes, which is why it reports a confidence.
* **Shades-of-grey, Minkowski p=6** (fallback, always computed). Assumes the
  scene averages to grey. That assumption breaks against a strongly coloured
  wall — common where these photos actually get taken — which is exactly why
  it is the fallback rather than the primary.

``ColorResult.estimator`` records which one produced the applied gains. When a
tracked metric jumps between sessions, that field is the first thing to check.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import landmarks as lm
from . import regions as regions_module
from .config import ColorConfig, Config
from .types import ColorResult, Face

__all__ = ["apply_gains", "correct_color", "sclera_white_balance", "shades_of_gray"]


def _normalise_gains(
    illuminant: np.ndarray, clamp: tuple[float, float]
) -> tuple[float, float, float]:
    """Illuminant estimate -> per-channel gains that preserve overall exposure.

    Gains are scaled so their mean is 1. Normalising to a fixed channel instead
    would drag overall brightness around with the illuminant and corrupt every
    luminance-dependent metric.
    """
    illuminant = np.maximum(illuminant.astype(np.float64), 1e-6)
    gains = float(illuminant.mean()) / illuminant
    gains = np.clip(gains, clamp[0], clamp[1])
    return (float(gains[0]), float(gains[1]), float(gains[2]))


def shades_of_gray(
    image: np.ndarray,
    config: ColorConfig | None = None,
    skin_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Minkowski-norm illuminant estimate. Returns BGR gains.

    With ``sog_estimate_from == "skin"`` the estimate is taken from skin pixels
    only. That is available but not the default: forcing mean skin colour
    toward grey would erase the erythema and melanin signal being measured.
    """
    config = config or ColorConfig()
    pixels = image.reshape(-1, 3).astype(np.float64)

    valid = np.all(
        (pixels >= config.sog_exclude_below) & (pixels <= config.sog_exclude_above), axis=1
    )
    if config.sog_estimate_from == "skin":
        if skin_mask is None:
            raise ValueError("sog_estimate_from='skin' requires a skin mask")
        valid &= skin_mask.reshape(-1)

    selected = pixels[valid]
    if len(selected) < 32:
        # Nothing usable to estimate from — a unity gain is the honest answer,
        # rather than an estimate from a handful of clipped pixels.
        return (1.0, 1.0, 1.0)

    p = config.minkowski_p
    illuminant = np.power(np.mean(np.power(selected, p), axis=0), 1.0 / p)
    return _normalise_gains(illuminant, config.sog_gain_clamp)


def _sclera_pixels(
    image: np.ndarray, face: Face, config: ColorConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate eye-white pixels, and the per-eye membership used for scoring.

    Returns ``(pixels, eye_index)`` where ``eye_index`` is 0 for the subject's
    left eye and 1 for the right, so the two estimates can be cross-checked.
    """
    shape = face.image_size
    points = face.landmarks.astype(np.float64)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    # The iris is not neutral. Refined landmarks give it directly; without them
    # there is no reliable way to exclude it, so sclera correction is skipped.
    has_iris = len(points) >= 478
    if not has_iris:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int8)

    collected = []
    indices = []
    for eye_id, (eye, iris) in enumerate(
        ((lm.LEFT_EYE, lm.LEFT_IRIS), (lm.RIGHT_EYE, lm.RIGHT_IRIS))
    ):
        eye_mask = regions_module._fill_hull(shape, points[list(eye)])
        # Erode: the eyelid margin and lash line sit just inside the contour.
        eye_mask = regions_module.erode(eye_mask, 2)
        iris_mask = regions_module.dilate(
            regions_module._fill_hull(shape, points[list(iris)]), 3
        )
        candidate = (
            eye_mask
            & ~iris_mask
            & (value >= config.sclera_v_min)
            & (saturation <= config.sclera_s_max)
        )
        if not candidate.any():
            continue
        # Trimmed percentile band on V: rejects residual lash shadow at the
        # bottom and specular catchlights at the top, both of which are far
        # from neutral.
        candidate_values = value[candidate]
        low, high = np.percentile(candidate_values, config.sclera_percentile_band)
        keep = candidate & (value >= low) & (value <= high)
        if not keep.any():
            continue
        collected.append(image[keep].astype(np.float64))
        indices.append(np.full(int(keep.sum()), eye_id, dtype=np.int8))

    if not collected:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int8)
    return np.concatenate(collected), np.concatenate(indices)


def sclera_white_balance(
    image: np.ndarray,
    face: Face,
    config: ColorConfig | None = None,
) -> tuple[tuple[float, float, float], float, int, str | None]:
    """Estimate the illuminant from eye-white.

    Returns ``(gains, confidence, pixel_count, failure_reason)``. The gains are
    always returned so they can be inspected; the caller decides whether the
    confidence clears the bar. ``failure_reason`` is None when the estimate is
    sound.

    Confidence combines three independent ways this fails: too few pixels
    (squint, closed eyes), sclera too dark (shadowed), and the two eyes
    disagreeing (one lit, one shaded — which is the case where a single-eye
    estimate would be confidently wrong).
    """
    config = config or ColorConfig()
    pixels, eye_index = _sclera_pixels(image, face, config)
    count = int(len(pixels))

    if count < config.sclera_min_pixels:
        reason = "eyes_closed_or_no_sclera" if count == 0 else f"only {count} sclera pixels"
        return (1.0, 1.0, 1.0), 0.0, count, reason

    gains = _normalise_gains(pixels.mean(axis=0), config.sclera_gain_clamp)

    # -- confidence --
    count_factor = float(np.clip(count / float(config.sclera_saturation_pixels), 0.0, 1.0))

    brightness = float(pixels.max(axis=1).mean() / 255.0)
    span = max(1.0 - config.sclera_v_min, 1e-6)
    brightness_factor = float(np.clip((brightness - config.sclera_v_min) / span, 0.0, 1.0))

    both_eyes = np.unique(eye_index)
    if len(both_eyes) < 2:
        # One eye only: nothing to cross-check against, so this estimate cannot
        # be verified. Capped rather than rejected.
        agreement_factor = 0.5
    else:
        per_eye = np.stack(
            [pixels[eye_index == eye].mean(axis=0) for eye in both_eyes]
        )
        per_eye = per_eye / np.maximum(per_eye.mean(axis=1, keepdims=True), 1e-6)
        disagreement = float(np.abs(per_eye[0] - per_eye[1]).max())
        agreement_factor = float(np.clip(1.0 - disagreement / 0.25, 0.0, 1.0))

    confidence = float(count_factor * brightness_factor * agreement_factor)
    reason = None if confidence >= config.sclera_min_confidence else (
        f"confidence {confidence:.2f} < {config.sclera_min_confidence:.2f}"
    )
    return gains, confidence, count, reason


def apply_gains(image: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    """Apply per-channel BGR gains, rounding half away from zero.

    numpy's default rounding is banker's rounding, which would make the output
    depend on the exact float representation at .5 boundaries. Explicit
    floor(x + 0.5) keeps this reproducible.
    """
    scaled = image.astype(np.float64) * np.asarray(gains, dtype=np.float64)
    return np.floor(np.clip(scaled, 0.0, 255.0) + 0.5).astype(np.uint8)


def correct_color(
    image: np.ndarray,
    face: Face | None = None,
    skin_mask: np.ndarray | None = None,
    config: Config | None = None,
) -> ColorResult:
    """Color-correct the image, preferring the sclera estimate when confident."""
    config = config or Config()
    color_config = config.color

    sog_gains = shades_of_gray(image, color_config, skin_mask)

    if not color_config.sclera_enabled or face is None:
        reason = None if not color_config.sclera_enabled else "no_face"
        return ColorResult(
            image=apply_gains(image, sog_gains),
            gains=sog_gains,
            estimator="shades_of_gray",
            fallback_reason=reason,
            sclera_confidence=None,
            shades_of_gray_gains=sog_gains,
        )

    sclera_gains, confidence, count, reason = sclera_white_balance(image, face, color_config)
    use_sclera = (
        color_config.sclera_as_primary
        and reason is None
        and confidence >= color_config.sclera_min_confidence
    )

    gains = sclera_gains if use_sclera else sog_gains
    return ColorResult(
        image=apply_gains(image, gains),
        gains=gains,
        estimator="sclera" if use_sclera else "shades_of_gray",
        fallback_reason=None if use_sclera else (reason or "sclera_not_primary"),
        sclera_confidence=confidence,
        sclera_pixel_count=count,
        shades_of_gray_gains=sog_gains,
    )
