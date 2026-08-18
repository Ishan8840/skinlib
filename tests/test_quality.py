"""Tests for the quality gate."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from skinlib.config import Config, IOConfig
from skinlib.detect import detect_face, load_image
from skinlib.quality import check_quality
from skinlib.types import QualityFlag

from .conftest import FIXTURE_DIR, TOO_FAR_PORTRAIT


@pytest.fixture(scope="module")
def gate(analysed, full_config: Config):
    loaded, face, skin, regions = analysed
    return lambda cfg=None: check_quality(loaded, face, skin, regions, cfg or full_config)


def test_good_capture_is_usable(gate) -> None:
    result = gate()
    assert result.usable
    assert not result.unreliable_metrics


def test_gate_never_raises_on_a_face_free_image(no_face_bgr, config: Config) -> None:
    """A bad photo is data about the photo, not an exception."""
    loaded = load_image(no_face_bgr)
    result = check_quality(loaded, None, None, None, config)
    assert not result.usable
    assert result.has(QualityFlag.NO_FACE)


def test_multiple_faces_flagged(multi_face_bgr, full_config: Config, parser) -> None:
    from skinlib.parse import parse_skin
    from skinlib.regions import build_regions

    loaded = load_image(multi_face_bgr, IOConfig(max_long_edge_px=None))
    face = detect_face(loaded.image, full_config)
    assert face is not None and face.n_faces >= 2
    skin = parse_skin(loaded.image, face, full_config, parser=parser)
    regions = build_regions(face, skin, full_config)
    result = check_quality(loaded, face, skin, regions, full_config)
    assert result.has(QualityFlag.MULTIPLE_FACES)
    assert not result.usable


def test_too_far_flagged(full_config: Config, parser) -> None:
    from skinlib.parse import parse_skin
    from skinlib.regions import build_regions

    loaded = load_image(FIXTURE_DIR / TOO_FAR_PORTRAIT)
    face = detect_face(loaded.image, full_config)
    assert face is not None
    skin = parse_skin(loaded.image, face, full_config, parser=parser)
    regions = build_regions(face, skin, full_config)
    result = check_quality(loaded, face, skin, regions, full_config)
    assert result.has(QualityFlag.TOO_FAR)
    assert not result.usable


def test_low_resolution_flagged_without_upscaling(full_config: Config, parser) -> None:
    """A small source is flagged, not resampled up to hide the problem."""
    from skinlib.parse import parse_skin
    from skinlib.regions import build_regions

    original = cv2.imread(str(FIXTURE_DIR / "portrait_a.jpg"))
    small = cv2.resize(original, (300, 401), interpolation=cv2.INTER_AREA)
    loaded = load_image(small)
    assert loaded.working_size == (401, 300), "a small source must not be upscaled"

    face = detect_face(loaded.image, full_config)
    if face is None:
        pytest.skip("face not detectable at this size; the resolution check needs a face")
    skin = parse_skin(loaded.image, face, full_config, parser=parser)
    regions = build_regions(face, skin, full_config)
    result = check_quality(loaded, face, skin, regions, full_config)
    assert result.has(QualityFlag.LOW_RESOLUTION)


def test_blur_is_detected(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    blurred = cv2.GaussianBlur(loaded.image, (0, 0), 6.0)
    blurred_loaded = load_image(blurred, IOConfig(max_long_edge_px=None))

    sharp = check_quality(loaded, face, skin, regions, full_config)
    soft = check_quality(blurred_loaded, face, skin, regions, full_config)

    assert soft.measures["laplacian_variance"] < sharp.measures["laplacian_variance"]
    assert soft.has(QualityFlag.BLURRY)
    assert not sharp.has(QualityFlag.BLURRY)


def test_darker_skin_is_not_blur(analysed, full_config: Config) -> None:
    """The second flag with the same defect, pinned.

    Laplacian variance scales with contrast and contrast scales with how much
    light skin reflects, so an absolute bar on the LINEAR measure rejects darker
    skin as out of focus. Measured on one unchanged photo scaled toward deeper
    tones, the linear variance swung 11.7x (68.8 -> 5.9) while the log-domain
    measure moved 1.16x (0.000944 -> 0.000813).

    Nothing about the photo changed but its brightness.
    """
    loaded, face, skin, regions = analysed
    for scale in (0.7, 0.45, 0.25):
        darker = np.clip(
            loaded.image.astype(np.float64) * scale + 8.0, 0, 255
        ).astype(np.uint8)
        dark = load_image(darker, IOConfig(max_long_edge_px=None))
        result = check_quality(dark, face, skin, regions, full_config)
        assert not result.has(QualityFlag.BLURRY), (
            f"an unchanged photo scaled to {scale} was called blurry"
        )


def test_genuine_blur_is_still_caught(analysed, full_config: Config) -> None:
    """Brightness invariance must not have cost the flag its job."""
    loaded, face, skin, regions = analysed
    blurred = load_image(
        cv2.GaussianBlur(loaded.image, (0, 0), 2.0), IOConfig(max_long_edge_px=None)
    )
    result = check_quality(blurred, face, skin, regions, full_config)
    assert result.has(QualityFlag.BLURRY)


def test_specular_detection_survives_darker_skin(analysed, full_config: Config) -> None:
    """The third flag with the same defect — and the worst of them.

    `V >= 0.92` did not merely shift on darker skin, it STOPPED WORKING: the
    specular fraction measured 0.00628 -> exactly 0.00000 as one unchanged photo
    was scaled toward deeper tones, because the brightest channel never reaches
    0.92 once median V is 0.59. Shine was undetectable on anything but light
    skin.

    A highlight is bright RELATIVE to the diffuse level around it, so the
    threshold is now a multiple of the skin's own median V.
    """
    loaded, face, skin, regions = analysed
    fractions = []
    for scale in (1.0, 0.6, 0.35, 0.25):
        darker = np.clip(
            loaded.image.astype(np.float64) * scale + 8.0, 0, 255
        ).astype(np.uint8)
        result = check_quality(
            load_image(darker, IOConfig(max_long_edge_px=None)),
            face, skin, regions, full_config,
        )
        fractions.append(result.measures["specular_fraction"])

    assert all(f > 0.0 for f in fractions), (
        f"specular detection died on darker skin: {fractions}"
    )
    # Measured 0.00688 -> 0.00471 across this range; 2x is a generous bound.
    assert max(fractions) / min(fractions) < 2.0


def test_blown_highlights_flag_too_bright(analysed, full_config: Config) -> None:
    """The other tail of the same argument, which 8.0.0 left unfixed.

    The old bar was mean L* > 82, which measured wrong in BOTH directions at
    once: it would reject correctly exposed very light skin (ITA > 55 reaches
    L* 84.6 at b* = 20) while staying silent on a capture with 17% of the skin
    already at the ceiling.
    """
    loaded, face, skin, regions = analysed
    blown = np.clip(loaded.image.astype(np.int32) + 40, 0, 255).astype(np.uint8)
    result = check_quality(
        load_image(blown, IOConfig(max_long_edge_px=None)), face, skin, regions, full_config
    )
    assert result.measures["highlight_clipped_fraction"] > full_config.quality.highlight_clipped_max
    assert result.has(QualityFlag.TOO_BRIGHT)


def test_light_skin_is_not_overexposure(analysed, full_config: Config) -> None:
    """A light face with nothing clipped is a lighter person, not a worse photo."""
    loaded, face, skin, regions = analysed
    # Range-compressed and lifted so mean L* clears the old 82.0 ceiling while
    # NOTHING reaches the clip point. L* is strongly non-linear in sRGB, so a
    # plain scale-and-offset overshoots the ceiling long before it lifts the
    # mean; measured, this lands at L* 86.6 with highlight clipping at exactly
    # zero.
    lighter = np.clip(
        loaded.image.astype(np.float64) * 0.25 + 185.0, 0, 255
    ).astype(np.uint8)
    result = check_quality(
        load_image(lighter, IOConfig(max_long_edge_px=None)), face, skin, regions, full_config
    )
    assert result.measures["mean_lightness"] > 82.0, "should exceed the OLD threshold"
    assert result.measures["highlight_clipped_fraction"] <= full_config.quality.highlight_clipped_max
    assert not result.has(QualityFlag.TOO_BRIGHT)


def test_crushed_shadows_flag_too_dark(analysed, full_config: Config) -> None:
    """`too_dark` means information was destroyed, not that skin is dark.

    Simulated as a genuine underexposure: scaled down far enough that a large
    share of skin pixels land on or below the black point, which is what
    destroys information no correction can restore.
    """
    loaded, face, skin, regions = analysed
    crushed = load_image(
        (loaded.image * 0.02).astype(np.uint8), IOConfig(max_long_edge_px=None)
    )
    result = check_quality(crushed, face, skin, regions, full_config)
    assert result.has(QualityFlag.TOO_DARK)
    assert not result.usable


def test_dark_skin_is_not_underexposure(analysed, full_config: Config) -> None:
    """The fairness property, pinned.

    A uniformly darker face with NOTHING clipped is a darker person, not a worse
    photo. The old gate thresholded mean L* at 32.0 and therefore rejected
    correctly exposed deep skin by construction: inverting the ITA scale, L* 32
    is ITA -42 to -56 depending on b*, well inside the "dark" class that
    `monk_ita_edges` claims to classify. The library cannot both classify a tone
    and refuse to measure it.
    """
    loaded, face, skin, regions = analysed
    # Scaled to land in the deep-skin L* range while staying clear of the black
    # point, so no information is lost.
    darker = np.clip(loaded.image.astype(np.float64) * 0.45 + 8.0, 0, 255).astype(np.uint8)
    dark = load_image(darker, IOConfig(max_long_edge_px=None))
    result = check_quality(dark, face, skin, regions, full_config)

    assert result.measures["shadow_clipped_fraction"] <= full_config.quality.shadow_clipped_max
    assert not result.has(QualityFlag.TOO_DARK), (
        "a correctly exposed dark face must not be rejected as a bad capture"
    )


def test_brightening_the_image_flags_too_bright(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    bright = load_image(
        np.clip(loaded.image.astype(np.int32) + 110, 0, 255).astype(np.uint8),
        IOConfig(max_long_edge_px=None),
    )
    result = check_quality(bright, face, skin, regions, full_config)
    assert result.has(QualityFlag.TOO_BRIGHT)


def test_specular_highlights_are_detected(analysed, full_config: Config) -> None:
    """Painted-in highlights must fire the specular check."""
    loaded, face, skin, regions = analysed
    speckled = loaded.image.copy()
    # Blow out a patch of cheek to near-white, which is what a hard flash does.
    cheek = regions["left_cheek"]
    ys, xs = np.nonzero(cheek)
    take = slice(0, len(ys) // 2)
    speckled[ys[take], xs[take]] = (252, 252, 252)

    result = check_quality(
        load_image(speckled, IOConfig(max_long_edge_px=None)), face, skin, regions, full_config
    )
    assert result.has(QualityFlag.HIGH_SPECULAR)


def test_side_lighting_is_detected(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    lit = loaded.image.astype(np.float64)
    # Darken one cheek only: a pure left/right luminance imbalance.
    lit[regions["right_cheek"]] *= 0.55
    result = check_quality(
        load_image(lit.astype(np.uint8), IOConfig(max_long_edge_px=None)),
        face, skin, regions, full_config,
    )
    assert result.has(QualityFlag.SIDE_LIT)
    assert result.measures["side_lit_ratio"] > full_config.quality.side_lit_max_frac


def test_side_lit_declines_to_fire_when_a_cheek_is_too_small(
    analysed, full_config: Config
) -> None:
    """An unmeasurable check must not fire either way.

    On a turned head one cheek is a sliver; comparing it against a full cheek
    measures pose, not lighting. Reporting NaN and no flag is the honest
    outcome — inventing a verdict from 200 pixels would not be.
    """
    loaded, face, skin, regions = analysed
    starved = dict(regions)
    starved["right_cheek"] = np.zeros_like(regions["right_cheek"])
    result = check_quality(loaded, face, skin, starved, full_config)
    assert np.isnan(result.measures["side_lit_ratio"])
    assert not result.has(QualityFlag.SIDE_LIT)


def test_smoothing_filter_is_detected(analysed, full_config: Config) -> None:
    """A beauty-filter imitation: edge-preserving smoothing that kills pores.

    Bilateral rather than Gaussian on purpose — it is what a real filter does,
    keeping facial structure while erasing pore-scale detail. That asymmetry is
    the thing the check keys on.
    """
    loaded, face, skin, regions = analysed
    smoothed = cv2.bilateralFilter(loaded.image, 25, 150, 150)
    result = check_quality(
        load_image(smoothed, IOConfig(max_long_edge_px=None)), face, skin, regions, full_config
    )
    assert result.has(QualityFlag.POSSIBLY_FILTERED)
    # Advisory, not blocking: colour survives smoothing, so the capture is
    # still worth something.
    assert "roughness" in result.unreliable_metrics
    assert "erythema" not in result.unreliable_metrics


def test_unfiltered_photo_is_not_flagged_as_filtered(gate) -> None:
    """False positives here accuse a user of faking their photo. Keep it clean."""
    result = gate()
    assert not result.has(QualityFlag.POSSIBLY_FILTERED)
    assert result.measures["scale_ratio"] > full_scale_floor()


def full_scale_floor() -> float:
    from skinlib.config import QualityConfig

    return QualityConfig().filtered_scale_ratio_min


def test_filtered_photo_is_flagged_while_still_sharp(analysed, full_config: Config) -> None:
    """The case that matters: sharp to the eye, but pore detail is gone.

    A beauty filter leaves edges intact, so focus checks pass and the photo
    looks fine. Only the scale-selective loss of fine detail gives it away.
    Ordinary defocus trips `blurry` as well, so it is already caught; a
    filtered photo would sail through without this check.
    """
    loaded, face, skin, regions = analysed
    filtered = cv2.bilateralFilter(loaded.image, 25, 150, 150)
    soft = cv2.GaussianBlur(loaded.image, (0, 0), 3.0)

    def gate_on(image: np.ndarray):
        return check_quality(
            load_image(image, IOConfig(max_long_edge_px=None)),
            face, skin, regions, full_config,
        )

    filtered_result = gate_on(filtered)
    soft_result = gate_on(soft)

    assert filtered_result.has(QualityFlag.POSSIBLY_FILTERED)
    # Still sharp by the focus metric — which is exactly why the extra check
    # has to exist.
    assert not filtered_result.has(QualityFlag.BLURRY)
    assert (
        filtered_result.measures["laplacian_variance"]
        > soft_result.measures["laplacian_variance"] * 5
    )
    assert soft_result.has(QualityFlag.BLURRY)


def test_flags_map_to_unreliable_metrics(full_config: Config) -> None:
    """Per-metric reliability, not a single blunt verdict."""
    mapping = full_config.quality.unreliable_metrics_for(["high_specular"])
    assert "roughness" in mapping
    assert "spot_count" in mapping
    # Specular highlights barely move ITA, so it stays trusted.
    assert "ita" not in mapping


def test_advisory_flags_do_not_block(analysed, full_config: Config) -> None:
    """A capture with soft problems still yields metrics worth keeping."""
    loaded, face, skin, regions = analysed
    smoothed = cv2.GaussianBlur(loaded.image, (0, 0), 1.2)
    result = check_quality(
        load_image(smoothed, IOConfig(max_long_edge_px=None)), face, skin, regions, full_config
    )
    advisory = {"side_lit", "high_specular", "possibly_filtered", "low_resolution"}
    if set(result.flags) <= advisory and result.flags:
        assert result.usable


def test_empty_mask_flags_rather_than_crashes(analysed, full_config: Config) -> None:
    loaded, face, _skin, regions = analysed
    empty = np.zeros(loaded.working_size, dtype=bool)
    result = check_quality(loaded, face, empty, regions, full_config)
    assert result.has(QualityFlag.MASK_TOO_SMALL)
    assert not result.usable


def test_clipping_measures_are_reported(gate) -> None:
    """Exposure evidence that does not depend on skin tone."""
    measures = gate().measures
    assert 0.0 <= measures["shadow_clipped_fraction"] <= 1.0
    assert 0.0 <= measures["highlight_clipped_fraction"] <= 1.0


def test_blocking_flags_are_configurable(analysed, full_config: Config) -> None:
    loaded, face, skin, regions = analysed
    relaxed = replace(
        full_config, quality=replace(full_config.quality, blocking_flags=())
    )
    dark = load_image((loaded.image * 0.02).astype(np.uint8), IOConfig(max_long_edge_px=None))
    result = check_quality(dark, face, skin, regions, relaxed)
    assert result.has(QualityFlag.TOO_DARK)
    assert result.usable, "with no blocking flags configured, nothing blocks"


def test_quality_is_deterministic(gate) -> None:
    first, second = gate(), gate()
    assert first.flags == second.flags
    assert first.usable == second.usable
    assert first.measures == second.measures
