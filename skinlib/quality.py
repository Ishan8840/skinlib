"""Quality gate.

Never raises. Every problem with the photo comes back as a flag, because a
photo being bad is data about the photo, not an error in the library. Genuine
faults — a missing checkpoint, an undecodable file — still raise, from the
stages that own them.

Two tiers of verdict:

* ``usable`` covers the hard blocks: captures that cannot be measured at all.
* ``unreliable_metrics`` names the metrics undermined by softer problems.
  Metrics still compute, and the caller decides what to trust. Specular
  highlights wreck texture while barely touching ITA; a beauty filter destroys
  pores while leaving colour roughly intact. A single boolean would throw away
  the good half of both captures.

Runs on the raw working image, before color correction: correcting first would
mask the very exposure and white-balance problems being detected.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config, QualityConfig
from .metrics import band_pass, lab_channels, log_luminance
from .types import Face, LoadedImage, QualityFlag, QualityResult

__all__ = ["check_quality"]


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def check_quality(
    loaded: LoadedImage,
    face: Face | None = None,
    skin_mask: np.ndarray | None = None,
    regions: dict[str, np.ndarray] | None = None,
    config: Config | None = None,
) -> QualityResult:
    """Assess whether the capture can be measured, and what to distrust.

    ``face``, ``skin_mask`` and ``regions`` may be None when an earlier stage
    produced nothing; the checks that need them are skipped rather than guessed
    at. A check that cannot be evaluated never fires — an unmeasurable
    condition is not a passed condition, and flagging it either way would be a
    fabrication.
    """
    config = config or Config()
    quality_config = config.quality

    flags: list[str] = []
    measures: dict[str, float] = {}

    # -- resolution: independent of every other stage --
    measures["source_long_edge"] = float(loaded.source_long_edge)
    if loaded.source_long_edge < quality_config.min_long_edge_px:
        flags.append(QualityFlag.LOW_RESOLUTION)

    # -- face presence --
    if face is None:
        flags.append(QualityFlag.NO_FACE)
        return _finalise(flags, measures, quality_config)

    measures["n_faces"] = float(face.n_faces)
    if face.n_faces > 1:
        flags.append(QualityFlag.MULTIPLE_FACES)

    # -- framing --
    measures["face_area_fraction"] = float(face.area_fraction)
    low, high = quality_config.face_area_frac_band
    if face.area_fraction < low:
        flags.append(QualityFlag.TOO_FAR)
    elif face.area_fraction > high:
        flags.append(QualityFlag.TOO_CLOSE)

    if skin_mask is None or not skin_mask.any():
        flags.append(QualityFlag.MASK_TOO_SMALL)
        measures["skin_pixels"] = 0.0
        return _finalise(flags, measures, quality_config)

    skin_pixels = int(skin_mask.sum())
    measures["skin_pixels"] = float(skin_pixels)
    if skin_pixels < config.parse.min_skin_pixels:
        flags.append(QualityFlag.MASK_TOO_SMALL)
        return _finalise(flags, measures, quality_config)

    image = loaded.image
    lightness, _a_star, _b_star = lab_channels(image)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # -- exposure --
    #
    # `too_dark` fires on INFORMATION LOSS, not on darkness.
    #
    # An absolute luminance floor cannot separate "underexposed" from "dark
    # skin", and thresholding mean L* got the answer wrong for the darkest
    # tones by construction. Inverting the ITA scale against published skin
    # colorimetry (Chardon/Del Bino) shows what the old 32.0 floor rejected:
    #
    #   ITA class        b*=12   b*=16   b*=20      <- mean L* of CORRECTLY
    #   dark (<-30)       43.1    40.8    38.5         exposed skin
    #   dark deep (~-50)  35.7    30.9    26.2      <- partly rejected
    #   dark deepest      24.3    15.7     7.1      <- always rejected
    #
    # The floor began rejecting at ITA -42 to -56 depending on b*, while
    # `monk_ita_edges` in MetricsConfig reaches -30 — the library claimed to
    # CLASSIFY skin its own gate refused to MEASURE. That is not a tuning
    # error, it is the wrong quantity.
    #
    # Shadow clipping is the tone-independent form of the same question: real
    # underexposure crushes pixels against black and destroys information,
    # whereas dark skin merely reflects less. It is measured first because the
    # flag now depends on it.
    measures["shadow_clipped_fraction"] = float(
        (image.min(axis=2)[skin_mask] <= config.metrics.clipped_below).mean()
    )
    measures["highlight_clipped_fraction"] = float(
        (image.max(axis=2)[skin_mask] >= config.metrics.clipped_above).mean()
    )

    mean_lightness = float(lightness[skin_mask].mean())
    measures["mean_lightness"] = mean_lightness
    dark_limit, bright_limit = quality_config.luminance_band

    # The absolute floor survives only as a backstop for a frame with no signal
    # at all. It sits far below any real skin tone, so it cannot fire on
    # pigmentation.
    # Both tails, symmetrically. 8.0.0 fixed only `too_dark`, but the argument
    # was never one-sided: mean L* rises with skin lightness as much as with
    # exposure, so an absolute ceiling rejects very light skin the same way the
    # floor rejected very deep skin. Measured, the old L* > 82 bar was wrong in
    # BOTH directions at once — it would reject correctly exposed ITA > 55 skin
    # (L* 84.6 at b* = 20) while staying silent on a capture with 17% of the
    # skin already blown out.
    if (
        measures["shadow_clipped_fraction"] > quality_config.shadow_clipped_max
        or mean_lightness < dark_limit
    ):
        flags.append(QualityFlag.TOO_DARK)
    elif (
        measures["highlight_clipped_fraction"] > quality_config.highlight_clipped_max
        or mean_lightness > bright_limit
    ):
        flags.append(QualityFlag.TOO_BRIGHT)

    # -- directional lighting --
    side_lit_ratio = _side_lit_ratio(lightness, regions, config.metrics.min_region_pixels)
    measures["side_lit_ratio"] = _finite(side_lit_ratio)
    if np.isfinite(side_lit_ratio) and side_lit_ratio > quality_config.side_lit_max_frac:
        flags.append(QualityFlag.SIDE_LIT)

    # -- focus --
    #
    # Measured on LOG luminance, for the same reason `roughness` is: a
    # multiplicative brightness change becomes an additive constant in the log
    # domain, and the Laplacian removes constants exactly.
    #
    # On linear intensity the Laplacian's amplitude scales with contrast, and
    # contrast scales with how much light the skin reflects — so an absolute
    # threshold on it rejects darker skin as blurred. Measured on one unchanged
    # photo scaled toward deeper tones, sharpness identical throughout:
    #
    #   scale   linear var   log var
    #    1.00       68.8     0.000944
    #    0.45       15.6     0.000865
    #    0.25        5.9     0.000813
    #
    # An 11.7x swing on the linear measure against 1.16x on the log one. That is
    # the same defect `too_dark` had, in a different flag: an absolute bar on a
    # quantity that scales with skin brightness.
    #
    # Genuine blur still separates cleanly — a sigma=1 Gaussian takes the log
    # measure from 0.001125 to 0.000082, a 13.7x drop.
    laplacian = cv2.Laplacian(log_luminance(image), cv2.CV_64F)
    blur_metric = float(laplacian[skin_mask].var())
    measures["log_laplacian_variance"] = blur_metric
    # Retained for inspection and for comparison with pre-9.0.0 stored results,
    # but no longer what the flag keys on. See QualityConfig.compute_diagnostics.
    if quality_config.compute_diagnostics:
        measures["laplacian_variance"] = float(
            cv2.Laplacian(grey, cv2.CV_64F)[skin_mask].var()
        )
    if blur_metric < quality_config.blur_log_laplacian_var_min:
        flags.append(QualityFlag.BLURRY)

    # -- specular highlights --
    #
    # A highlight is bright RELATIVE to the diffuse skin around it, not in
    # absolute terms: it is light reflected off the surface before entering the
    # skin, so it carries the illuminant's colour (hence desaturated) and stands
    # well above the local diffuse level. On deep skin that level is low, so the
    # highlight is dimmer in absolute terms while being just as much of a
    # highlight.
    #
    # The old absolute bar (V >= 0.92) did not merely shift on darker skin, it
    # STOPPED WORKING. Measured on one unchanged photo scaled toward deeper
    # tones, the specular fraction went 0.00628 -> exactly 0.00000 and stayed
    # there: the brightest channel never reaches 0.92 once median V is 0.59, so
    # shine was undetectable on anything but light skin.
    #
    # Referenced to the skin's own median V, the same photo holds 0.00688 ->
    # 0.00471 across the same range, and 1.25x reproduces the old measure's
    # value on the unchanged image (0.00688 against 0.00628).
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float64) / 255.0
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0

    median_value = float(np.median(value[skin_mask])) if skin_mask.any() else 0.0
    measures["median_value"] = median_value
    # The absolute floor survives only to stop a near-black region manufacturing
    # "highlights" out of quantisation noise; it sits below any real skin.
    specular_level = max(
        quality_config.specular_v_ratio * median_value, quality_config.specular_v_min
    )
    measures["specular_level"] = specular_level
    specular = (
        skin_mask
        & (value > specular_level)
        & (saturation <= quality_config.specular_s_max)
    )
    specular_fraction = float(specular.sum() / skin_pixels)
    measures["specular_fraction"] = specular_fraction
    if specular_fraction > quality_config.specular_frac_max:
        flags.append(QualityFlag.HIGH_SPECULAR)

    # -- smoothing filter --
    hf_energy, scale_ratio, texture_ratio = _filter_signals(
        lightness, skin_mask, regions, quality_config, config.metrics.min_region_pixels
    )
    measures["hf_energy"] = _finite(hf_energy)
    measures["scale_ratio"] = _finite(scale_ratio)
    if quality_config.compute_diagnostics:
        measures["texture_ratio"] = _finite(texture_ratio)
    # BOTH conditions, deliberately. Low high-frequency energy alone is also
    # what a soft lens, a distant subject or a dark exposure looks like. It is
    # the combination with a collapsed fine/coarse ratio — detail missing at
    # pore scale while facial structure survives — that distinguishes a
    # smoothing filter from a merely soft photo.
    low_detail = np.isfinite(hf_energy) and hf_energy < quality_config.filtered_hf_energy_min
    scale_selective = (
        np.isfinite(scale_ratio) and scale_ratio < quality_config.filtered_scale_ratio_min
    )
    if low_detail and scale_selective:
        flags.append(QualityFlag.POSSIBLY_FILTERED)

    return _finalise(flags, measures, quality_config)


def _side_lit_ratio(
    lightness: np.ndarray,
    regions: dict[str, np.ndarray] | None,
    min_pixels: int,
) -> float:
    """Relative left/right cheek luminance difference, or NaN if unmeasurable."""
    if regions is None:
        return float("nan")
    left = regions.get("left_cheek")
    right = regions.get("right_cheek")
    if left is None or right is None:
        return float("nan")
    # A turned head leaves one cheek as a sliver. Comparing a sliver against a
    # full cheek measures pose, not lighting, so the check declines to fire.
    if int(left.sum()) < min_pixels or int(right.sum()) < min_pixels:
        return float("nan")

    left_mean = float(lightness[left].mean())
    right_mean = float(lightness[right].mean())
    average = 0.5 * (left_mean + right_mean)
    if average <= 1e-6:
        return float("nan")
    return abs(left_mean - right_mean) / average


def _filter_signals(
    lightness: np.ndarray,
    skin_mask: np.ndarray,
    regions: dict[str, np.ndarray] | None,
    config: QualityConfig,
    min_pixels: int,
) -> tuple[float, float, float]:
    """(high-frequency energy, fine/coarse scale ratio, nose/cheek ratio).

    Energy is contrast-normalised so it measures detail rather than exposure:
    a correctly exposed dark-skinned face has lower absolute high-frequency
    amplitude than a light-skinned one at identical real detail, and an
    un-normalised threshold would flag it as filtered.

    The scale ratio compares a pore-scale band against a structure-scale band.
    It is a ratio of two bands of the same image, so it is invariant to overall
    contrast and to skin tone — both bands scale together — which an absolute
    texture threshold is not.
    """
    blurred = cv2.GaussianBlur(lightness, (0, 0), config.filtered_hf_sigma)
    detail = np.abs(lightness - blurred)
    local_level = np.maximum(blurred[skin_mask], 1e-6)
    hf_energy = float(np.mean(detail[skin_mask] / local_level))

    fine = band_pass(lightness, config.filtered_fine_sigma, config.filtered_band_ratio)
    coarse = band_pass(lightness, config.filtered_coarse_sigma, config.filtered_band_ratio)
    coarse_std = float(np.std(coarse[skin_mask]))
    scale_ratio = (
        float(np.std(fine[skin_mask]) / coarse_std) if coarse_std > 1e-9 else float("nan")
    )

    texture_ratio = float("nan")
    if regions is not None:
        first, second = config.filtered_texture_pair
        mask_a = regions.get(first)
        mask_b = regions.get(second)
        if (
            mask_a is not None
            and mask_b is not None
            and int(mask_a.sum()) >= min_pixels
            and int(mask_b.sum()) >= min_pixels
        ):
            texture = band_pass(lightness, config.filtered_hf_sigma, 2.0)
            std_a = float(np.std(texture[mask_a]))
            std_b = float(np.std(texture[mask_b]))
            if std_b > 1e-9:
                texture_ratio = std_a / std_b

    return hf_energy, scale_ratio, texture_ratio


def _finalise(
    flags: list[str],
    measures: dict[str, float],
    config: QualityConfig,
) -> QualityResult:
    """Assemble the verdict. Flag order is stable for reproducible output."""
    ordered = [str(flag) for flag in flags]
    blocking = set(config.blocking_flags)
    usable = not any(flag in blocking for flag in ordered)
    return QualityResult(
        usable=usable,
        flags=ordered,
        unreliable_metrics=config.unreliable_metrics_for(ordered),
        measures=measures,
    )
