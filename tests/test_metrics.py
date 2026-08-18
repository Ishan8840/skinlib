"""Tests for the metrics, colour constancy and spot detection."""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np
import pytest

from skinlib.color import apply_gains, correct_color, sclera_white_balance, shades_of_gray
from skinlib.config import ColorConfig, Config
from skinlib.metrics import (
    band_pass,
    compute_metrics,
    ita_degrees,
    lab_channels,
    melanin_index_map,
    monk_bin_from_ita,
)
from skinlib.spots import detect_spots
from skinlib.types import METRIC_NAMES, REGION_NAMES


# ---------------------------------------------------------------------------
# metric primitives
# ---------------------------------------------------------------------------


def test_melanin_index_rises_as_reflectance_falls() -> None:
    """Higher index means darker, and a mid grey sits where the maths says."""
    bright = np.full((4, 4, 3), 200, dtype=np.uint8)
    dark = np.full((4, 4, 3), 60, dtype=np.uint8)
    assert melanin_index_map(dark).mean() > melanin_index_map(bright).mean()
    expected = math.log10(1.0 / (200 / 255))
    assert melanin_index_map(bright).mean() == pytest.approx(expected, abs=1e-9)


def test_melanin_index_floor_prevents_infinity() -> None:
    """Pure black must not produce inf and poison every downstream mean."""
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    values = melanin_index_map(black)
    assert np.all(np.isfinite(values))


def test_lab_channels_are_in_cielab_units() -> None:
    grey = np.full((4, 4, 3), 128, dtype=np.uint8)
    lightness, a_star, b_star = lab_channels(grey)
    assert 0.0 <= lightness.mean() <= 100.0
    # Neutral grey has no chroma.
    assert abs(float(a_star.mean())) < 2.0
    assert abs(float(b_star.mean())) < 2.0


def test_band_pass_removes_a_smooth_gradient() -> None:
    """Roughness must measure texture, not shading.

    Measured away from the frame edge: both Gaussians reflect at the border, so
    a ramp leaves a residual there. It is irrelevant in practice — the skin
    mask never reaches the image border — but it would swamp this assertion.
    """
    ramp = np.tile(np.linspace(0, 100, 200, dtype=np.float64), (200, 1))
    interior = band_pass(ramp, 3.0, 2.0)[20:-20, 20:-20]
    assert float(np.std(interior)) < 0.05


def test_band_pass_responds_to_texture() -> None:
    rng = np.random.default_rng(0)  # fixed seed: test input, not library behaviour
    noisy = np.tile(np.linspace(0, 100, 200), (200, 1)) + rng.normal(0, 5, (200, 200))
    smooth = np.tile(np.linspace(0, 100, 200, dtype=np.float64), (200, 1))
    assert float(np.std(band_pass(noisy, 3.0, 2.0))) > float(
        np.std(band_pass(smooth, 3.0, 2.0))
    )


def test_ita_is_higher_for_lighter_skin() -> None:
    light = np.full((8, 8, 3), (170, 195, 225), dtype=np.uint8)  # BGR
    dark = np.full((8, 8, 3), (60, 80, 110), dtype=np.uint8)
    light_l, _, light_b = lab_channels(light)
    dark_l, _, dark_b = lab_channels(dark)
    assert ita_degrees(light_l, light_b) > ita_degrees(dark_l, dark_b)


def test_monk_bin_spans_the_scale_and_orders_correctly() -> None:
    """Bin 1 is lightest, 10 darkest, and every ITA lands somewhere valid."""
    bins = [monk_bin_from_ita(ita) for ita in (80, 55, 45, 38, 30, 22, 15, 0, -20, -60)]
    assert bins[0] == 1
    assert bins[-1] == 10
    assert bins == sorted(bins), "monk_bin must increase as ITA decreases"
    assert all(1 <= value <= 10 for value in bins)


def test_monk_bin_is_nan_for_undefined_ita() -> None:
    assert math.isnan(monk_bin_from_ita(float("nan")))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_all_metrics_present_globally_and_per_region(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    result = compute_metrics(loaded.image, skin, regions, full_config, spots=[])

    assert set(result.global_) == set(METRIC_NAMES)
    assert set(result.by_region) == set(REGION_NAMES)
    for name, values in result.by_region.items():
        assert set(values) == set(METRIC_NAMES), f"{name} is missing metrics"


def test_thin_regions_are_nan_not_noise(analysed, full_config: Config) -> None:
    """NaN says "not measured", which is not the same claim as a number."""
    loaded, face, skin, regions = analysed
    starved = dict(regions)
    tiny = np.zeros_like(skin)
    tiny[:5, :5] = True
    starved["chin"] = tiny

    result = compute_metrics(loaded.image, skin, starved, full_config, spots=[])
    assert all(math.isnan(value) for value in result.by_region["chin"].values())
    assert result.pixel_counts["chin"] < full_config.metrics.min_region_pixels


def test_spot_columns_are_nan_when_spots_were_not_run(analysed, full_config: Config) -> None:
    """Not measured (NaN) must not masquerade as measured-and-none-found (0)."""
    loaded, _face, skin, regions = analysed
    result = compute_metrics(loaded.image, skin, regions, full_config, spots=None)
    assert math.isnan(result.global_["spot_count"])
    assert math.isnan(result.global_["spot_area_fraction"])


def test_uniformity_falls_as_pigment_varies(analysed, full_config: Config) -> None:
    loaded, _face, skin, regions = analysed
    speckled = loaded.image.copy()
    ys, xs = np.nonzero(skin)
    speckled[ys[::7], xs[::7]] = (20, 20, 20)

    even = compute_metrics(loaded.image, skin, regions, full_config, spots=[])
    uneven = compute_metrics(speckled, skin, regions, full_config, spots=[])
    assert uneven.global_["uniformity"] < even.global_["uniformity"]


def test_erythema_rises_with_redness(analysed, full_config: Config) -> None:
    loaded, _face, skin, regions = analysed
    redder = loaded.image.astype(np.int32)
    redder[:, :, 2] = np.clip(redder[:, :, 2] + 25, 0, 255)
    redder = redder.astype(np.uint8)

    base = compute_metrics(loaded.image, skin, regions, full_config, spots=[])
    warm = compute_metrics(redder, skin, regions, full_config, spots=[])
    assert warm.global_["erythema"] > base.global_["erythema"]
    assert warm.global_["erythema_mean"] > base.global_["erythema_mean"]


def test_metrics_read_only_inside_the_mask(analysed, full_config: Config) -> None:
    """Changing pixels outside the mask must not move a single metric.

    ``face`` is passed so the burden metrics are exercised too. They are a
    genuine test of this invariant rather than a formality: they read the
    melanin residual, whose background is a large-kernel median that reaches
    well past the mask edge. It only survives because ``_background`` floods
    everything outside the mask with the in-mask median before blurring.
    """
    loaded, face, skin, regions = analysed
    tampered = loaded.image.copy()
    tampered[~skin] = (255, 0, 255)

    before = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)
    after = compute_metrics(tampered, skin, regions, full_config, spots=[], face=face)

    for name in METRIC_NAMES:
        if name in ("roughness",):
            # The texture band is a neighbourhood operator, so mask-adjacent
            # pixels legitimately see the change. Everything else must not.
            continue
        first, second = before.global_[name], after.global_[name]
        # NaN means "not measured", and two not-measured readings agree. approx
        # would call them unequal, which is the wrong claim about this library.
        if math.isnan(first) and math.isnan(second):
            continue
        assert first == pytest.approx(second, abs=1e-9), name


def test_clipped_pixels_are_excluded(analysed, full_config: Config) -> None:
    loaded, _face, skin, regions = analysed
    blown = loaded.image.copy()
    ys, xs = np.nonzero(skin)
    blown[ys[: len(ys) // 4], xs[: len(xs) // 4]] = (255, 255, 255)

    excluded = compute_metrics(blown, skin, regions, full_config, spots=[])
    kept = compute_metrics(
        blown, skin, regions,
        replace(full_config, metrics=replace(full_config.metrics, exclude_clipped=False)),
        spots=[],
    )
    assert excluded.pixel_counts["global"] < kept.pixel_counts["global"]
    assert excluded.global_["erythema_mean"] != kept.global_["erythema_mean"]


# ---------------------------------------------------------------------------
# colour constancy
# ---------------------------------------------------------------------------


def test_shades_of_gray_neutralises_a_colour_cast() -> None:
    """A blue-cast grey scene must come back closer to neutral."""
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    scene[:] = (170, 120, 100)  # BGR: strongly blue
    gains = shades_of_gray(scene, ColorConfig())
    corrected = apply_gains(scene, gains).astype(np.float64)
    spread_before = float(np.ptp(scene.reshape(-1, 3).mean(axis=0)))
    spread_after = float(np.ptp(corrected.reshape(-1, 3).mean(axis=0)))
    assert spread_after < spread_before


def test_shades_of_gray_gains_are_clamped() -> None:
    extreme = np.zeros((32, 32, 3), dtype=np.uint8)
    extreme[:, :, 0] = 240  # blue only
    extreme[:, :, 1] = 10
    extreme[:, :, 2] = 10
    low, high = ColorConfig().sog_gain_clamp
    for gain in shades_of_gray(extreme, ColorConfig()):
        assert low <= gain <= high


def test_shades_of_gray_returns_unity_when_nothing_is_usable() -> None:
    """An all-black frame yields no estimate, and says so by not changing anything."""
    assert shades_of_gray(np.zeros((32, 32, 3), dtype=np.uint8), ColorConfig()) == (1.0, 1.0, 1.0)


def test_sclera_reports_low_confidence_when_eyes_are_unusable(analysed, full_config: Config) -> None:
    """Closed or shadowed eyes must lower confidence, not produce silent gains."""
    loaded, face, _skin, _regions = analysed
    darkened = (loaded.image * 0.15).astype(np.uint8)
    _gains, confidence, _count, reason = sclera_white_balance(
        darkened, face, full_config.color
    )
    assert confidence < full_config.color.sclera_min_confidence
    assert reason is not None


def test_colour_falls_back_to_shades_of_gray_and_records_why(
    analysed, full_config: Config
) -> None:
    """Provenance is the point: which estimator ran must always be recoverable."""
    loaded, face, skin, _regions = analysed
    darkened = (loaded.image * 0.15).astype(np.uint8)
    result = correct_color(darkened, face, skin, full_config)
    assert result.estimator == "shades_of_gray"
    assert result.fallback_reason
    assert result.gains == result.shades_of_gray_gains


def test_disabling_sclera_uses_shades_of_gray(analysed, full_config: Config) -> None:
    loaded, face, skin, _regions = analysed
    config = replace(full_config, color=replace(full_config.color, sclera_enabled=False))
    result = correct_color(loaded.image, face, skin, config)
    assert result.estimator == "shades_of_gray"
    assert result.sclera_confidence is None


def test_colour_correction_is_deterministic(analysed, full_config: Config) -> None:
    loaded, face, skin, _regions = analysed
    first = correct_color(loaded.image, face, skin, full_config)
    second = correct_color(loaded.image, face, skin, full_config)
    assert np.array_equal(first.image, second.image)
    assert first.gains == second.gains
    assert first.estimator == second.estimator


# ---------------------------------------------------------------------------
# spots
# ---------------------------------------------------------------------------


def test_spots_lie_inside_the_skin_mask(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    spots = detect_spots(loaded.image, skin, face, regions, full_config)
    for spot in spots:
        col, row = int(round(spot.centroid[0])), int(round(spot.centroid[1]))
        assert skin[row, col], "spot centroid outside the skin mask"


def test_spot_records_are_well_formed(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    spots = detect_spots(loaded.image, skin, face, regions, full_config)
    if not spots:
        pytest.skip("no spots on this fixture")
    for spot in spots:
        x, y, w, h = spot.bbox
        assert w > 0 and h > 0
        assert x <= spot.centroid[0] <= x + w
        assert y <= spot.centroid[1] <= y + h
        assert spot.area_px > 0
        assert 0.0 < spot.area_fraction < 1.0
        assert spot.region in set(REGION_NAMES) | {""}
        assert np.isfinite(spot.mean_melanin_index)


def test_injected_dark_spot_is_detected(analysed, full_config: Config) -> None:
    """A synthetic pigment blob on the cheek must be found."""
    loaded, face, skin, regions = analysed
    planted = loaded.image.copy()
    ys, xs = np.nonzero(regions["left_cheek"])
    centre_index = len(ys) // 2
    centre = (int(xs[centre_index]), int(ys[centre_index]))
    cv2.circle(planted, centre, 9, (0, 0, 0), -1)
    # Blend so it reads as pigment rather than a hard-edged paste.
    planted = cv2.GaussianBlur(planted, (0, 0), 1.5)

    spots = detect_spots(planted, skin, face, regions, full_config)
    assert spots, "planted spot was not detected at all"
    distances = [math.dist(spot.centroid, centre) for spot in spots]
    assert min(distances) < 15, "planted spot was not among the detections"


def test_spot_ordering_is_fully_determined(analysed, full_config: Config) -> None:
    """Ties must not resolve differently between runs."""
    loaded, face, skin, regions = analysed
    first = detect_spots(loaded.image, skin, face, regions, full_config)
    second = detect_spots(loaded.image, skin, face, regions, full_config)
    assert [s.centroid for s in first] == [s.centroid for s in second]
    areas = [spot.area_px for spot in first]
    assert areas == sorted(areas, reverse=True)


def test_area_filters_are_applied(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    strict = replace(
        full_config, spots=replace(full_config.spots, min_area_frac=1e-2)
    )
    assert detect_spots(loaded.image, skin, face, strict.regions and regions, strict) == []


def test_spots_carry_region_labels(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    spots = detect_spots(loaded.image, skin, face, regions, full_config)
    labelled = [spot for spot in spots if spot.region]
    if not labelled:
        pytest.skip("no spots landed inside a named region")
    for spot in labelled:
        col, row = int(round(spot.centroid[0])), int(round(spot.centroid[1]))
        assert regions[spot.region][row, col]


def test_empty_mask_yields_no_spots(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    empty = np.zeros_like(skin)
    assert detect_spots(loaded.image, empty, face, regions, full_config) == []


# ---------------------------------------------------------------------------
# precision and filtering (0.2.0)
# ---------------------------------------------------------------------------


def test_lab_is_not_quantised_to_integers(analysed) -> None:
    """a* must have finer resolution than 1 unit.

    The 8-bit Lab path gave the entire skin mask 17 distinct a* values, so
    `erythema` (a percentile of a*) could only ever return an integer — a
    resolution ceiling roughly 8x the measured session-to-session noise.
    """
    loaded, _face, skin, _regions = analysed
    _lightness, a_star, _b_star = lab_channels(loaded.image)
    distinct = len(np.unique(a_star[skin]))
    assert distinct > 200, f"a* has only {distinct} distinct values; is Lab quantised?"


def test_erythema_is_not_an_integer_grid(analysed, full_config: Config) -> None:
    loaded, _face, skin, regions = analysed
    result = compute_metrics(loaded.image, skin, regions, full_config, spots=[])
    values = [result.by_region[r]["erythema"] for r in REGION_NAMES]
    values = [v for v in values if not math.isnan(v)]
    assert any(abs(v - round(v)) > 1e-6 for v in values), "every erythema landed on an integer"


def test_mad_threshold_resists_a_bright_artifact(analysed, full_config: Config) -> None:
    """A large high-contrast artifact must not raise the bar for real spots.

    A plain standard deviation is inflated by exactly the artifacts being
    filtered out (hairline shadow, stubble), which raises the threshold and
    hides the low-contrast spots underneath.
    """
    from skinlib.spots import _threshold

    rng = np.random.default_rng(0)
    clean = rng.normal(0, 0.01, 20000)
    contaminated = np.concatenate([clean, rng.normal(0.5, 0.05, 400)])

    mad_config = full_config.spots
    sigma_config = replace(mad_config, threshold_mode="sigma")

    mad_shift = _threshold(contaminated, mad_config) / _threshold(clean, mad_config)
    sigma_shift = _threshold(contaminated, sigma_config) / _threshold(clean, sigma_config)
    assert mad_shift < sigma_shift, "MAD should be less disturbed than sigma"
    assert mad_shift < 1.5


def test_shape_filters_are_skipped_on_tiny_components(analysed, full_config: Config) -> None:
    """Eccentricity is meaningless on a handful of pixels.

    A 3-pixel diagonal scores ~1.0 and would be discarded as a hair strand; on
    a real photo this rejected 59 of 102 candidates with median area 6px.
    """
    assert full_config.spots.shape_min_area_px > full_config.spots.min_area_px


def test_area_floor_is_absolute_not_only_relative(analysed, full_config: Config) -> None:
    """On a small capture the relative floor collapsed to under 2 pixels."""
    loaded, _face, skin, _regions = analysed
    relative_floor = full_config.spots.min_area_frac * int(skin.sum())
    effective = max(full_config.spots.min_area_px, relative_floor)
    assert effective >= full_config.spots.min_area_px


def test_boundary_rejection_uses_the_whole_component(
    analysed, full_config: Config
) -> None:
    """A blob straddling the mask edge centres well inside it.

    An observed 229px false positive sat on the hairline with its centroid
    comfortably within the mask, which a centroid-only test waved through.
    """
    loaded, face, skin, regions = analysed
    planted = loaded.image.copy()

    # Paint a dark blob half on, half off the mask boundary.
    edge = skin & ~erode_mask(skin, 6)
    ys, xs = np.nonzero(edge)
    index = len(ys) // 2
    cv2.circle(planted, (int(xs[index]), int(ys[index])), 7, (0, 0, 0), -1)
    planted = cv2.GaussianBlur(planted, (0, 0), 1.0)

    spots = detect_spots(planted, skin, face, regions, full_config)
    for spot in spots:
        assert math.dist(spot.centroid, (float(xs[index]), float(ys[index]))) > 8


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    from skinlib.regions import erode

    return erode(mask, radius)


def test_shadow_rejection_is_opt_in(analysed, full_config: Config) -> None:
    """Off by default: it is not yet validated as tone-neutral.

    On a dark-skinned fixture the chroma-residual spread was 4.5x larger,
    compressing every z-score, and a z>=2.5 cut removed every detection.
    Enabling it unvalidated would degrade darker skin specifically.
    """
    assert full_config.spots.reject_neutral_shadows is False

    loaded, face, skin, regions = analysed
    baseline = detect_spots(loaded.image, skin, face, regions, full_config)
    strict = detect_spots(
        loaded.image, skin, face, regions,
        replace(full_config, spots=replace(full_config.spots, reject_neutral_shadows=True)),
    )
    # It may only ever remove detections, never invent them.
    assert len(strict) <= len(baseline)
    assert all(spot in baseline for spot in strict)
