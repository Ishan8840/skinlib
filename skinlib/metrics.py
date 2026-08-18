"""Skin metrics, computed globally and per region.

Everything here runs on the color-corrected image, inside the skin mask, and
nowhere else. Every function is a pure transform of pixels plus config — no
learned components, no randomness.

A region with too few pixels yields NaN for every metric rather than a number
derived from noise. NaN is a load-bearing value here: it says "not measured",
which a longitudinal tracker must be able to distinguish from a real reading.
"""

from __future__ import annotations

import cv2
import numpy as np

from .chromophore import separate_chromophores
from .config import Config, MetricsConfig
from .types import METRIC_NAMES, Face, MetricsResult, Spot

__all__ = [
    "band_pass",
    "compute_metrics",
    "ita_degrees",
    "lab_channels",
    "melanin_index_map",
    "monk_bin_from_ita",
    "region_metrics",
    "unclipped_mask",
    "roughness_sigma_for",
]


def lab_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BGR uint8 -> (L*, a*, b*) in CIELAB units.

    Converted in float32, NOT through OpenCV's 8-bit Lab. The 8-bit path packs
    a* and b* into one byte each, quantising them to integer steps: measured on
    a real photo, the entire skin mask held just 17 distinct a* values.

    That quantisation is not noise, it is a resolution ceiling, and it lands
    directly on `erythema` (a percentile of a*, so it can only ever return an
    integer). On the same face the step size was ~1.0 a* units while the
    session-to-session variation in mean a* was 0.13 — the ceiling was roughly
    8x the signal it was supposed to resolve. Float conversion takes the same
    image to 799 distinct values at no cost.
    """
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return (
        lab[:, :, 0].astype(np.float64),
        lab[:, :, 1].astype(np.float64),
        lab[:, :, 2].astype(np.float64),
    )


def erythema_index_map(image: np.ndarray, config: MetricsConfig | None = None) -> np.ndarray:
    """log10(R / G): redness as a log ratio, invariant to exposure.

    CIELAB a* is not a log ratio, so an exposure change does not become a
    uniform additive offset in it and subtracting the face median only
    approximately cancels — self-normalising a* would be normalisation in name
    only. A log ratio of two channels cancels a neutral exposure change
    exactly: scaling both R and G by k leaves log10(kR / kG) unchanged.

    Higher means redder. Equivalent to log10(1/G) - log10(1/R), i.e. the
    difference of two melanin-style indices.
    """
    config = config or MetricsConfig()
    # +1 before the log keeps crushed pixels finite without distorting mid-tones.
    red = image[:, :, 2].astype(np.float64) + 1.0
    green = image[:, :, 1].astype(np.float64) + 1.0
    return np.log10(red / green)


def log_luminance(image: np.ndarray) -> np.ndarray:
    """log10 of grey level.

    Texture is measured here rather than on linear intensity because a
    multiplicative exposure change becomes an additive constant in the log
    domain, and a band-pass filter removes constants exactly. On linear
    intensity the band-pass amplitude scales with exposure, so a brighter
    photo of the same skin simply reads as rougher.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) + 1.0
    return np.log10(grey)


def melanin_index_map(image: np.ndarray, config: MetricsConfig | None = None) -> np.ndarray:
    """log10(1 / R_normalised), the standard reflectance-based melanin index.

    Higher means darker. The red channel is used because melanin absorbs
    broadly while haemoglobin absorbs strongly in green — red is where
    pigmentation dominates the reflectance.
    """
    config = config or MetricsConfig()
    red = image[:, :, 2].astype(np.float64) / 255.0
    red = np.clip(red, config.melanin_r_floor, 1.0)
    return np.log10(1.0 / red)


def band_pass(grey: np.ndarray, sigma: float, ratio: float) -> np.ndarray:
    """Difference of Gaussians: keeps the texture band, drops tone and shading.

    The low-frequency term removes illumination gradients and overall tone; the
    high-frequency term removes sensor noise. What remains is pore- and
    fine-wrinkle scale.
    """
    inner = cv2.GaussianBlur(grey, (0, 0), sigma)
    outer = cv2.GaussianBlur(grey, (0, 0), sigma * ratio)
    return inner - outer


def roughness_sigma_for(config: MetricsConfig, face_width: float | None) -> float:
    """Band-pass sigma in pixels for a face of this apparent width.

    A fixed pixel sigma measures a different physical scale at every capture
    distance. Within the distance band `QualityConfig.face_area_frac_band`
    admits, that alone swings roughness by a factor of 2.2 on an unchanged skin
    patch — which a longitudinal tracker reports as the skin getting rougher
    when the subject merely stood closer. Scaling with face width holds the same
    patch flat to 3.4%.

    Falls back to the fixed sigma when the face width is unknown, which restores
    the old distance-sensitive behaviour rather than guessing a scale. Callers
    that have a face should pass it.
    """
    frac = config.roughness_sigma_face_frac
    if frac is None or face_width is None or not np.isfinite(face_width) or face_width <= 0:
        return float(config.roughness_sigma)
    # A sigma below half a pixel band-passes nothing but sensor noise; the floor
    # keeps a very distant face degrading gracefully instead of returning noise
    # with a plausible-looking magnitude.
    return float(max(frac * float(face_width), 0.5))


def ita_degrees(lightness: np.ndarray, b_star: np.ndarray) -> float:
    """Individual Typology Angle, in degrees, from medians.

    Medians rather than means: a specular highlight or a missed hair strand
    shifts a mean but not a median.
    """
    if lightness.size == 0:
        return float("nan")
    median_b = float(np.median(b_star))
    return float(np.degrees(np.arctan2(float(np.median(lightness)) - 50.0, median_b)))


def monk_bin_from_ita(ita: float, config: MetricsConfig | None = None) -> float:
    """ITA -> Monk Skin Tone bin, 1 (lightest) to 10 (darkest).

    DISPLAY VALUE ONLY. ITA is the source of truth; this is an approximation
    with no official standing. Never use it for fairness or bias evaluation —
    that audits tone fairness against this very mapping, which is circular.
    Fairness work needs human-assigned Monk labels or raw ITA bands.
    """
    config = config or MetricsConfig()
    if not np.isfinite(ita):
        return float("nan")
    # Edges descend; the first edge the value falls below advances the bin.
    return float(1 + int(np.sum(np.asarray(config.monk_ita_edges) > ita)))


def unclipped_mask(image: np.ndarray, config: MetricsConfig) -> np.ndarray | None:
    """Pixels that are neither blown nor crushed. None when the check is off.

    A whole-image reduction, so it is computed once and reused across every
    region rather than recomputed per call — with ten masks per image the naive
    form scanned the full frame ten times for an answer that never changes.
    """
    if not config.exclude_clipped:
        return None
    channel_max = image.max(axis=2)
    channel_min = image.min(axis=2)
    return (channel_max <= config.clipped_above) & (channel_min >= config.clipped_below)


def _valid_pixels(
    image: np.ndarray,
    mask: np.ndarray,
    config: MetricsConfig,
    unclipped: np.ndarray | None = None,
) -> np.ndarray:
    """Mask with clipped pixels dropped.

    A blown or crushed pixel carries no chroma: its a* and b* are driven by the
    clip, not by skin, so including it biases every colour metric toward zero
    chroma.
    """
    if not config.exclude_clipped:
        return mask
    if unclipped is None:
        unclipped = unclipped_mask(image, config)
    return mask & unclipped


def _nan_metrics() -> dict[str, float]:
    return {name: float("nan") for name in METRIC_NAMES}


# Burden metric families: the field holding the residual, the reference keys it
# is measured against, and the two metric names it produces.
#
# Identical machinery over two different chromophores. Melanin excess is a dark
# mark — a lentigo, or the post-inflammatory stain acne leaves behind.
# Haemoglobin excess is an *active* lesion, which the melanin map structurally
# cannot see because acne is red and melanin is brown.
_BURDEN_FAMILIES: tuple[tuple[str, str, str, str], ...] = (
    ("residual", "residual", "spot_burden", "spot_contrast"),
    (
        "hemoglobin_residual",
        "hemoglobin_residual",
        "inflammation_burden",
        "inflammation_contrast",
    ),
)


def _burden_metrics(
    fields: dict[str, np.ndarray],
    usable: np.ndarray,
    config: MetricsConfig,
    reference: dict[str, float] | None,
) -> dict[str, float]:
    """Continuous burden and contrast for every chromophore family.

    ``*_burden`` is the fraction of the mask's pixels whose residual clears a
    face-wide threshold — extent. ``*_contrast`` is how far the upper percentile
    of that residual sits above the face-wide median — intensity.

    Both reference the FACE-WIDE median and sigma rather than the region's own.
    A per-region reference would renormalise every region to look average and
    destroy exactly the between-region differences these exist to show; the
    shared reference is also estimated from far more pixels, so it is steadier
    than the quantity being measured against it.

    Prefer these over the corresponding counts. Counting discrete objects near a
    detection boundary is Poisson, so a count's relative noise cannot beat
    1/sqrt(N); these have no discretisation step and measured 30-50x more
    repeatable. Note the two halves do not degrade equally under varied capture:
    burden holds up, contrast does not. See README.
    """
    out: dict[str, float] = {}
    for field_name, reference_key, burden_name, contrast_name in _BURDEN_FAMILIES:
        residual = fields.get(field_name)
        median = (reference or {}).get(f"{reference_key}_median", float("nan"))
        sigma = (reference or {}).get(f"{reference_key}_sigma", float("nan"))
        values = residual[usable] if residual is not None else None

        if (
            values is None
            or values.size == 0
            or not (np.isfinite(median) and np.isfinite(sigma))
        ):
            out[burden_name] = float("nan")
            out[contrast_name] = float("nan")
            continue

        out[burden_name] = float(
            (values > median + config.spot_burden_sigma * sigma).mean()
        )
        out[contrast_name] = float(
            np.percentile(values, config.spot_contrast_percentile) - median
        )
    return out


def region_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    config: MetricsConfig | None = None,
    spots: list[Spot] | None = None,
    spot_reference_area: int | None = None,
    precomputed: dict[str, np.ndarray] | None = None,
    reference: dict[str, float] | None = None,
    face_width: float | None = None,
    unclipped: np.ndarray | None = None,
) -> dict[str, float]:
    """Every metric for one mask.

    ``spots`` are those already assigned to this mask; ``spot_reference_area``
    is the pixel count the area fraction is taken against.

    That reference must be the RAW mask area, not the unclipped count. Spot
    components are found on the raw skin mask, so their pixels include any that
    the clipping filter would drop; dividing that numerator by an unclipped
    denominator mixes two different pixel populations and biases the fraction
    upward exactly where the capture is worst. Small in magnitude, but this is a
    column a longitudinal tracker reads. ``reference`` carries the face-wide
    medians the ``_rel`` variants are measured against; without it those come
    back NaN, because a relative metric with no reference is not a number.

    ``face_width`` scales the texture band so roughness measures skin rather
    than capture distance. Ignored when ``precomputed`` is supplied, since the
    band has already been built by then.
    """
    config = config or MetricsConfig()

    fields = (
        precomputed if precomputed is not None else _precompute(image, config, face_width)
    )
    usable = _valid_pixels(image, mask, config, unclipped)
    count = int(usable.sum())

    if count < config.min_region_pixels:
        return _nan_metrics()

    a_values = fields["a_star"][usable]
    b_values = fields["b_star"][usable]
    lightness_values = fields["lightness"][usable]
    melanin_values = fields["melanin"][usable]
    erythema_values = fields["erythema_index"][usable]
    texture_values = fields["texture"][usable]
    melanin_density_values = fields["melanin_density"][usable]
    hemoglobin_density_values = fields["hemoglobin_density"][usable]

    ita = ita_degrees(lightness_values, b_values)
    melanin_std = float(np.std(melanin_values))
    melanin_mean = float(np.mean(melanin_values))
    erythema_mean_log = float(np.mean(erythema_values))
    roughness = float(np.std(texture_values))
    uniformity = float(1.0 / (1.0 + melanin_std))

    area = spot_reference_area if spot_reference_area is not None else count
    spot_list = spots or []
    spot_area = float(sum(spot.area_px for spot in spot_list))

    values: dict[str, float] = {
        # -- exposure-invariant, safe to display and to track --
        "erythema_index": float(np.percentile(erythema_values, config.erythema_percentile)),
        "erythema_index_mean": erythema_mean_log,
        "uniformity": uniformity,
        "roughness": roughness,
        "spot_count": float(len(spot_list)),
        "spot_area_fraction": float(spot_area / area) if area > 0 else float("nan"),
        # Medians, not means: these are absolutes with no self-normalisation to
        # blunt an outlier, so a surviving specular highlight or stray hair
        # would move a mean straight into the reported value.
        "melanin_density": float(np.median(melanin_density_values)),
        "hemoglobin_density": float(np.median(hemoglobin_density_values)),
        **_burden_metrics(fields, usable, config, reference),
        # -- INTERNAL ONLY: exposure-dependent, see types.INTERNAL_ONLY_METRICS --
        "melanin_index": melanin_mean,
        "erythema": float(np.percentile(a_values, config.erythema_percentile)),
        "erythema_mean": float(np.mean(a_values)),
        "ita": ita,
        "monk_bin": monk_bin_from_ita(ita, config),
    }

    # -- self-normalised variants --
    if reference is None:
        for name in ("melanin_index_rel", "erythema_index_rel", "roughness_rel", "uniformity_rel"):
            values[name] = float("nan")
    else:
        values["melanin_index_rel"] = melanin_mean - reference["melanin"]
        values["erythema_index_rel"] = erythema_mean_log - reference["erythema_index"]
        # Roughness and uniformity are already exposure-invariant (a band-pass
        # of a log image, and a standard deviation, both kill a constant
        # offset). Their _rel form is regional contrast against the whole face,
        # not an exposure correction.
        values["roughness_rel"] = roughness - reference["texture"]
        values["uniformity_rel"] = uniformity - reference["uniformity"]

    return {name: values[name] for name in METRIC_NAMES}


def _precompute(
    image: np.ndarray,
    config: MetricsConfig,
    face_width: float | None = None,
    residual: np.ndarray | None = None,
    hemoglobin_residual_map: np.ndarray | None = None,
    chromophores: tuple[np.ndarray, np.ndarray] | None = None,
    melanin: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Whole-image derived fields, computed once and shared by every region.

    Computed on the full image rather than per region so that the Gaussians in
    the texture band see real neighbouring pixels instead of a mask edge, which
    would register as an enormous fake gradient at every region boundary.

    ``face_width`` sets the texture band's physical scale. Without it the band
    falls back to a fixed pixel sigma, which conflates skin texture with capture
    distance — see ``roughness_sigma_for``.
    """
    lightness, a_star, b_star = lab_channels(image)
    melanin_density, hemoglobin_density = (
        chromophores if chromophores is not None else separate_chromophores(image, config)
    )
    fields: dict[str, np.ndarray] = {
        "lightness": lightness,
        "a_star": a_star,
        "b_star": b_star,
        "melanin": melanin if melanin is not None else melanin_index_map(image, config),
        "erythema_index": erythema_index_map(image, config),
        # Shading and exposure projected out analytically, not self-normalised.
        "melanin_density": melanin_density,
        "hemoglobin_density": hemoglobin_density,
        # Band-passed LOG luminance: exposure becomes an additive constant,
        # which the band-pass removes. See log_luminance.
        "texture": band_pass(
            log_luminance(image),
            roughness_sigma_for(config, face_width),
            config.roughness_sigma_ratio,
        ),
    }
    # The residual keys are ABSENT rather than None when the caller had no face
    # or mask to build them from. Consumers reach them through `.get`, so a
    # missing key reads as "not available" and the burden metrics report NaN —
    # which keeps the dict honestly typed as arrays all the way down.
    if residual is not None:
        fields["residual"] = residual
    if hemoglobin_residual_map is not None:
        fields["hemoglobin_residual"] = hemoglobin_residual_map
    return fields


# Metrics whose regional value is self-normalised by subtracting the face-wide
# median of the map they come from, and the map each one draws on.
#
# Difference, never ratio. These maps are already logarithmic, so a difference
# IS the log of the linear-reflectance ratio — the physically meaningful
# quantity. A ratio of two logs has no physical meaning and blows up whenever
# the denominator approaches zero.
_RELATIVE_SOURCES: dict[str, str] = {
    "melanin_index_rel": "melanin",
    "erythema_index_rel": "erythema_index",
}


def _face_reference(fields: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    """Face-wide median of each self-normalisation source map.

    The median over the whole skin mask, not a per-region statistic: it is
    computed from tens of thousands of pixels, which makes the reference itself
    far more stable than any small in-frame patch.
    """
    reference: dict[str, float] = {}
    for source in set(_RELATIVE_SOURCES.values()):
        values = fields[source][mask]
        reference[source] = float(np.median(values)) if values.size else float("nan")
    if mask.any():
        # Imported here, not at module scope: spots imports melanin_index_map
        # from this module, so a top-level import would be circular.
        from .spots import residual_reference

        for key in ("residual", "hemoglobin_residual"):
            residual = fields.get(key)
            if residual is None:
                continue
            median, sigma = residual_reference(residual, mask)
            reference[f"{key}_median"] = median
            reference[f"{key}_sigma"] = sigma
    reference["texture"] = float(np.std(fields["texture"][mask])) if mask.any() else float("nan")
    reference["uniformity"] = (
        float(1.0 / (1.0 + np.std(fields["melanin"][mask]))) if mask.any() else float("nan")
    )
    return reference


def compute_metrics(
    image: np.ndarray,
    skin_mask: np.ndarray,
    regions: dict[str, np.ndarray],
    config: Config | None = None,
    spots: list[Spot] | None = None,
    face: Face | None = None,
    melanin_residual_map: np.ndarray | None = None,
    hemoglobin_residual_map: np.ndarray | None = None,
    chromophores: tuple[np.ndarray, np.ndarray] | None = None,
    melanin: np.ndarray | None = None,
) -> MetricsResult:
    """All metrics, globally and per region.

    ``spots`` fills the ``spot_count`` and ``spot_area_fraction`` columns. Call
    ``detect_spots`` first and pass the result; without it those two entries are
    reported as NaN (not measured) rather than 0 (measured, none found), which
    are very different claims.

    ``face`` scales the texture band to the subject's apparent size. Omitting it
    leaves ``roughness`` and ``roughness_rel`` on a fixed pixel sigma, where
    they track capture distance as much as skin — accepted only because a
    caller may legitimately have masks without a face. ``analyze`` always passes
    it.
    """
    config = config or Config()
    metrics_config = config.metrics

    # Imported here rather than at module scope: spots imports melanin_index_map
    # from this module, so a top-level import would be circular.
    from .spots import hemoglobin_residual, melanin_residual

    # The burden metrics need a local residual, which needs both a face (to
    # scale the background kernel) and the skin mask. Without a face they stay
    # None and the metrics report NaN — not measured, rather than measured as
    # zero.
    measurable = face is not None and bool(skin_mask.any())
    # All three are accepted precomputed. The chromophore separation and the
    # large-kernel medians behind the residuals are the library's two most
    # expensive operations, and the spot and lesion detectors need the very same
    # maps — so `analyze` builds each once and passes it to all three callers.
    residual = melanin_residual_map
    hemoglobin = hemoglobin_residual_map
    if face is not None and measurable:
        if residual is None:
            residual = melanin_residual(image, skin_mask, face, config)
        if hemoglobin is None:
            hemoglobin = hemoglobin_residual(image, skin_mask, face, config)
    fields = _precompute(
        image,
        metrics_config,
        face.width if face is not None else None,
        residual,
        hemoglobin,
        chromophores,
        melanin,
    )
    unclipped = unclipped_mask(image, metrics_config)
    usable_skin = _valid_pixels(image, skin_mask, metrics_config, unclipped)
    skin_area = int(usable_skin.sum())
    reference = _face_reference(fields, usable_skin) if skin_area else None

    def metrics_for(mask: np.ndarray, own_spots: list[Spot] | None, area: int) -> dict[str, float]:
        values = region_metrics(
            image, mask, metrics_config, own_spots, area,
            precomputed=fields, reference=reference, unclipped=unclipped,
        )
        if spots is None:
            values["spot_count"] = float("nan")
            values["spot_area_fraction"] = float("nan")
        return values

    # Raw mask area, matching the population the spot detector ran on.
    global_metrics = metrics_for(skin_mask, spots, int(skin_mask.sum()))

    by_region: dict[str, dict[str, float]] = {}
    pixel_counts: dict[str, int] = {"global": skin_area}
    for name, mask in regions.items():
        # pixel_counts reports what was MEASURED (unclipped); the spot fraction
        # is referenced to the raw mask, which is what spots were found on.
        pixel_counts[name] = int(_valid_pixels(image, mask, metrics_config, unclipped).sum())
        own = None if spots is None else [spot for spot in spots if spot.region == name]
        by_region[name] = metrics_for(mask, own, int(mask.sum()))

    # D(x, y) = M(x, y) - median(M over the face).
    #
    # Kept as a map, not collapsed to the per-region scalars above, because the
    # scalars discard everything spatial: a new localized spot, a diffuse shift
    # across one region, cheek-vs-cheek asymmetry, pigmentation haloing around
    # a lesion. All of those are visible here and invisible in a regional mean.
    # Outside the skin mask it is NaN — not zero, which would read as
    # "measured, no deviation".
    normalized = np.full(skin_mask.shape, np.nan, dtype=np.float32)
    if reference is not None:
        normalized[usable_skin] = (
            fields["melanin"][usable_skin] - reference["melanin"]
        ).astype(np.float32)

    return MetricsResult(
        global_=global_metrics,
        by_region=by_region,
        pixel_counts=pixel_counts,
        normalized_map=normalized,
        face_reference=dict(reference) if reference is not None else {},
    )
