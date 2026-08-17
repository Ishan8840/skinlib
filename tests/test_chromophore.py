"""Tests for melanin / haemoglobin separation and the face-relative texture band.

Two properties are load-bearing here and each has a test that fails loudly if it
regresses:

* the chromophore densities are invariant to shading and exposure, because that
  invariance is the entire reason they exist alongside ``melanin_index``;
* ``roughness`` no longer depends on how close the subject stood, because a
  metric that does is a tracker reporting capture geometry as a skin change.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from skinlib.chromophore import (
    _basis,
    _coordinate_operator,
    deshade,
    optical_density,
    separate_chromophores,
)
from skinlib.config import Config, MetricsConfig
from skinlib.metrics import (
    band_pass,
    compute_metrics,
    log_luminance,
    melanin_index_map,
    roughness_sigma_for,
)

from .conftest import FIXTURE_DIR, PRIMARY_PORTRAIT


# ---------------------------------------------------------------------------
# optical density and the shading projection
# ---------------------------------------------------------------------------


def test_optical_density_is_returned_in_rgb_order() -> None:
    """The basis constants are quoted RGB; a BGR array would silently swap them."""
    # Pure blue in BGR is (255, 0, 0).
    blue = np.zeros((2, 2, 3), dtype=np.uint8)
    blue[:, :, 0] = 255
    density = optical_density(blue)
    # Blue reflects fully -> zero density in the BLUE (last, in RGB) channel.
    assert density[0, 0, 2] == pytest.approx(0.0, abs=1e-9)
    assert density[0, 0, 0] > 1.0


def test_deshade_removes_the_neutral_axis_and_is_idempotent() -> None:
    rng = np.random.default_rng(0)
    density = rng.normal(size=(8, 8, 3))
    once = deshade(density)
    assert np.abs(once.sum(axis=-1)).max() < 1e-12
    assert np.allclose(once, deshade(once), atol=1e-12)


def test_a_neutral_gain_is_exactly_a_shift_along_the_shading_axis() -> None:
    """The premise the whole separation rests on."""
    rng = np.random.default_rng(1)
    image = rng.integers(40, 220, size=(16, 16, 3), dtype=np.uint8)
    dimmed = (image.astype(np.float64) * 0.6).astype(np.uint8)
    difference = optical_density(image) - optical_density(dimmed)
    # Every channel moved by the same amount: the difference is along (1,1,1).
    assert float(difference.std(axis=-1).max()) < 0.01


def test_coordinate_operator_is_a_left_inverse() -> None:
    """Coordinates, not projections. A dot product would fail this."""
    basis = _basis(MetricsConfig())
    operator = _coordinate_operator(basis)
    assert np.allclose(operator @ basis, np.eye(2), atol=1e-12)


def test_coordinates_reconstruct_the_deshaded_density() -> None:
    image = cv2.imread(str(FIXTURE_DIR / PRIMARY_PORTRAIT))
    patch = image[200:280, 200:280]
    config = MetricsConfig()
    melanin, hemoglobin = separate_chromophores(patch, config)
    basis = _basis(config)
    rebuilt = melanin[..., None] * basis[:, 0] + hemoglobin[..., None] * basis[:, 1]
    assert np.allclose(rebuilt, deshade(optical_density(patch, config)), atol=1e-9)


def test_collinear_axes_raise_rather_than_returning_nonsense() -> None:
    config = replace(
        MetricsConfig(),
        chromophore_melanin_axis=(0.5, 0.6, 0.7),
        chromophore_hemoglobin_axis=(1.0, 1.2, 1.4),
    )
    with pytest.raises(ValueError, match="collinear"):
        separate_chromophores(np.full((4, 4, 3), 128, dtype=np.uint8), config)


# ---------------------------------------------------------------------------
# the invariance that justifies the metric
# ---------------------------------------------------------------------------


def _patch() -> np.ndarray:
    image = cv2.imread(str(FIXTURE_DIR / PRIMARY_PORTRAIT))
    height, width = image.shape[:2]
    return image[height // 3 : height // 3 + 400, width // 3 : width // 3 + 400]


def test_melanin_density_beats_melanin_index_against_exposure() -> None:
    """A brighter photo of the same skin is not more pigmented skin."""
    patch = _patch().astype(np.float64)
    brighter = np.clip(patch * 1.26, 0, 255).astype(np.uint8)
    base = patch.astype(np.uint8)

    index_shift = abs(float(np.mean(melanin_index_map(brighter) - melanin_index_map(base))))
    density_shift = abs(
        float(np.mean(separate_chromophores(brighter)[0] - separate_chromophores(base)[0]))
    )
    assert density_shift < 0.01
    # Measured 0.0754 vs 0.0027; the margin is wide enough that 10x is a floor,
    # not a threshold tuned to the fixture.
    assert density_shift * 10 < index_shift


def test_melanin_density_beats_melanin_index_against_shading() -> None:
    """A cheek turning away from the light is geometry, not pigment."""
    patch = _patch().astype(np.float64)
    rows, cols = patch.shape[:2]
    _, columns = np.mgrid[0:rows, 0:cols]
    gradient = (0.55 + 0.45 * (columns / (cols - 1.0)))[..., None]
    shaded = np.clip(patch * gradient, 0, 255).astype(np.uint8)
    base = patch.astype(np.uint8)

    index_shift = abs(float(np.mean(melanin_index_map(shaded) - melanin_index_map(base))))
    density_shift = abs(
        float(np.mean(separate_chromophores(shaded)[0] - separate_chromophores(base)[0]))
    )
    assert density_shift < 0.02
    assert density_shift < index_shift


def _with_extra_melanin(image: np.ndarray, delta: float, config: MetricsConfig) -> np.ndarray:
    """Add a uniform melanin density to every pixel, per the absorbance model.

    Injected along the melanin axis in optical density and converted back, which
    is what "more pigment, evenly, everywhere" actually means. A plain channel
    tint would be a colour cast as much as a pigment change and would not
    isolate the property under test.
    """
    axis = np.asarray(config.chromophore_melanin_axis, dtype=np.float64)
    density = optical_density(image, config) + delta * axis
    reflectance = np.power(10.0, -density)
    return np.clip(reflectance[..., ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)


def test_density_recovers_a_face_wide_shift_that_the_rel_metrics_cannot_see(
    analysed, full_config: Config
) -> None:
    """Why both families are reported.

    ``melanin_index_rel`` subtracts the face-wide median, so a change affecting
    the WHOLE face cancels — which is precisely the result a skincare tracker
    exists to show. The density is an absolute, and it does not merely *notice*
    the change: it returns the injected quantity.

    Measured, injected -> recovered by melanin_density, and the same change as
    seen by melanin_index_rel:

        0.010 -> 0.01004    rel moved 0.00047
        0.020 -> 0.01946    rel moved 0.00024
        0.050 -> 0.05024    rel moved 0.00033

    The _rel column is flat at its noise floor regardless of magnitude, which is
    the signature of a metric that is structurally blind rather than merely
    insensitive.
    """
    loaded, face, skin, regions = analysed
    before = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)

    for delta in (0.01, 0.05):
        shifted = _with_extra_melanin(loaded.image, delta, full_config.metrics)
        after = compute_metrics(shifted, skin, regions, full_config, spots=[], face=face)

        recovered = after.global_["melanin_density"] - before.global_["melanin_density"]
        rel_move = abs(
            after.global_["melanin_index_rel"] - before.global_["melanin_index_rel"]
        )
        # 5% tolerance covers uint8 round-tripping through reflectance.
        assert recovered == pytest.approx(delta, rel=0.05)
        assert rel_move < delta / 10.0


def test_densities_are_deterministic() -> None:
    """No SVD, no randomness — the closed-form solve must be bit-stable."""
    patch = _patch()
    first = separate_chromophores(patch)
    second = separate_chromophores(patch)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


# ---------------------------------------------------------------------------
# roughness must not measure capture distance
# ---------------------------------------------------------------------------


def test_roughness_sigma_falls_back_when_face_width_is_unknown() -> None:
    config = MetricsConfig()
    assert roughness_sigma_for(config, None) == config.roughness_sigma
    assert roughness_sigma_for(config, float("nan")) == config.roughness_sigma
    assert roughness_sigma_for(config, 0.0) == config.roughness_sigma
    assert roughness_sigma_for(replace(config, roughness_sigma_face_frac=None), 375.0) == (
        config.roughness_sigma
    )


def test_roughness_sigma_reproduces_the_historical_default_at_a_typical_face() -> None:
    """0.008 * 375px == 3.0px, so mid-range framing is unchanged by the switch."""
    assert roughness_sigma_for(MetricsConfig(), 375.0) == pytest.approx(3.0, abs=1e-9)


def test_roughness_sigma_has_a_floor() -> None:
    """A tiny face must degrade gracefully, not band-pass pure sensor noise."""
    assert roughness_sigma_for(MetricsConfig(), 1.0) == 0.5


def test_face_relative_sigma_makes_roughness_survive_capture_distance() -> None:
    """The bug this release exists to fix.

    Same skin, resampled across the linear range `face_area_frac_band` admits.
    A fixed pixel sigma reported the far end as roughly twice as rough as the
    near end; scaling the band with apparent face size holds it flat.
    """
    patch = _patch()
    scales = (1.0, 0.75, 0.58, 0.45, 0.33)

    fixed, relative = [], []
    for scale in scales:
        resized = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        grey = log_luminance(resized)
        fixed.append(float(np.std(band_pass(grey, 3.0, 2.0))))
        relative.append(float(np.std(band_pass(grey, 3.0 * scale, 2.0))))

    def spread(values: list[float]) -> float:
        return max(values) / min(values)

    # Pin both sides: the fix must hold, and the bug must stay demonstrated so
    # nobody "simplifies" the sigma back to a constant.
    assert spread(fixed) > 2.0
    assert spread(relative) < 1.1


# ---------------------------------------------------------------------------
# continuous pigmentation burden
# ---------------------------------------------------------------------------


def test_burden_metrics_are_nan_without_a_face(analysed, full_config: Config) -> None:
    """No face means no residual, which is not measured rather than zero."""
    loaded, _face, skin, regions = analysed
    result = compute_metrics(loaded.image, skin, regions, full_config, spots=[])
    assert np.isnan(result.global_["spot_burden"])
    assert np.isnan(result.global_["spot_contrast"])


def test_burden_metrics_do_not_need_the_spot_detector(
    analysed, full_config: Config
) -> None:
    """They read the residual directly, so `spots=None` must not blank them.

    This is half the point: `spot_count` and `spot_area_fraction` go NaN without
    a detector run, while burden and contrast are pure residual statistics.
    """
    loaded, face, skin, regions = analysed
    result = compute_metrics(loaded.image, skin, regions, full_config, spots=None, face=face)
    assert np.isnan(result.global_["spot_count"])
    assert np.isfinite(result.global_["spot_burden"])
    assert np.isfinite(result.global_["spot_contrast"])


def test_burden_is_a_fraction_and_contrast_is_positive(
    analysed, full_config: Config
) -> None:
    loaded, face, skin, regions = analysed
    result = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)
    assert 0.0 <= result.global_["spot_burden"] <= 1.0
    # The upper percentile of the residual sits above its median by definition.
    assert result.global_["spot_contrast"] > 0.0


def test_burden_separates_regions(analysed, full_config: Config) -> None:
    """Regions must be comparable, which is why the reference is face-wide.

    Measured on real captures the burden spans ~76x across the regions of one
    face. A per-region reference would renormalise each region to look average
    and collapse exactly this spread.
    """
    loaded, face, skin, regions = analysed
    result = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)
    values = [
        v["spot_burden"] for v in result.by_region.values() if np.isfinite(v["spot_burden"])
    ]
    assert len(values) >= 5
    assert max(values) > 3 * max(min(values), 1e-6)


def test_burden_rises_with_injected_pigment(analysed, full_config: Config) -> None:
    """Sensitivity: a real dark mark must move it in the right direction."""
    loaded, face, skin, regions = analysed
    marked = loaded.image.copy()
    rows, cols = np.nonzero(skin)
    # Index rows and cols TOGETHER — sampling them independently and zipping
    # yields coordinate pairs that are not skin pixels at all.
    picks = np.linspace(0, len(rows) - 1, 40).astype(int)
    discs = np.zeros(skin.shape, dtype=np.uint8)
    for row, col in zip(rows[picks], cols[picks]):
        cv2.circle(discs, (int(col), int(row)), 6, 1, -1)
    # Darken existing skin rather than painting a flat colour. Painting black
    # would put every disc below `clipped_below`, so `_valid_pixels` would drop
    # the whole injected signal before a metric ever saw it.
    spot = discs.astype(bool) & skin
    marked[spot] = (marked[spot].astype(np.float64) * 0.65).astype(np.uint8)

    before = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)
    after = compute_metrics(marked, skin, regions, full_config, spots=[], face=face)
    assert after.global_["spot_burden"] > before.global_["spot_burden"]
    assert after.global_["spot_contrast"] > before.global_["spot_contrast"]


def test_compute_metrics_scales_the_texture_band_to_the_detected_face(
    analysed, full_config: Config
) -> None:
    """compute_metrics must actually use the face it is handed."""
    loaded, face, skin, regions = analysed

    with_face = compute_metrics(loaded.image, skin, regions, full_config, spots=[], face=face)
    without = compute_metrics(loaded.image, skin, regions, full_config, spots=[])

    # Fixture faces are 274-392px, so the face-relative sigma is never 3.0 and
    # the two must differ. Equality would mean `face` was silently dropped.
    assert with_face.global_["roughness"] != without.global_["roughness"]
