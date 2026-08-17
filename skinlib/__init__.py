"""skinlib — deterministic facial skin measurement.

    from skinlib import analyze, Config
    result = analyze("photo.jpg", config=Config())

Same image plus same config plus same weights always produces identical
metrics. Every stage is also exposed individually for debugging and reuse:
``detect_face``, ``parse_skin``, ``build_regions``, ``check_quality``,
``correct_color``, ``compute_metrics``, ``detect_spots``, ``detect_lesions``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .chromophore import deshade, optical_density, separate_chromophores
from .color import correct_color, sclera_white_balance, shades_of_gray
from .config import (
    Config,
    ColorConfig,
    DetectConfig,
    IOConfig,
    MetricsConfig,
    ParseConfig,
    QualityConfig,
    RegionConfig,
    SessionConfig,
    SpotsConfig,
    config_fingerprint,
)
from .derived import asymmetry, periorbital_decomposition
from .detect import detect_face, load_image, load_landmarker
from .geometry import (
    depth_surface,
    estimate_light_direction,
    incidence_map,
    region_incidence,
    surface_normals,
)
from .metrics import (
    compute_metrics,
    erythema_index_map,
    log_luminance,
    melanin_index_map,
    roughness_sigma_for,
)
from .parse import load_parser, parse_skin, weights_hash
from .quality import check_quality
from .regions import build_regions
from .session import analyze_session, register_to, sharpness_of
from .spots import (
    detect_lesions,
    detect_spots,
    hemoglobin_residual,
    local_residual,
    melanin_residual,
    residual_reference,
)
from .types import (
    DISPLAY_SAFE_METRICS,
    INTERNAL_ONLY_METRICS,
    METRIC_NAMES,
    REGION_NAMES,
    AnalysisResult,
    ColorResult,
    Face,
    FrameReport,
    Lesion,
    LoadedImage,
    Masks,
    MetricsResult,
    QualityFlag,
    QualityResult,
    SessionResult,
    Spot,
)
from .version import PREPROCESSING_VERSION

__all__ = [
    "melanin_index_map",
    "log_luminance",
    "erythema_index_map",
    "deshade",
    "optical_density",
    "separate_chromophores",
    "roughness_sigma_for",
    "INTERNAL_ONLY_METRICS",
    "DISPLAY_SAFE_METRICS",
    "AnalysisResult",
    "ColorConfig",
    "ColorResult",
    "Config",
    "DetectConfig",
    "Face",
    "IOConfig",
    "Lesion",
    "LoadedImage",
    "METRIC_NAMES",
    "Masks",
    "MetricsConfig",
    "MetricsResult",
    "PREPROCESSING_VERSION",
    "ParseConfig",
    "QualityConfig",
    "QualityFlag",
    "QualityResult",
    "REGION_NAMES",
    "RegionConfig",
    "Spot",
    "SpotsConfig",
    "analyze",
    "analyze_session",
    "register_to",
    "sharpness_of",
    "SessionConfig",
    "SessionResult",
    "FrameReport",
    "build_regions",
    "check_quality",
    "compute_metrics",
    "config_fingerprint",
    "correct_color",
    "asymmetry",
    "periorbital_decomposition",
    "depth_surface",
    "estimate_light_direction",
    "incidence_map",
    "region_incidence",
    "surface_normals",
    "detect_face",
    "load_landmarker",
    "detect_lesions",
    "detect_spots",
    "hemoglobin_residual",
    "local_residual",
    "melanin_residual",
    "residual_reference",
    "load_image",
    "load_parser",
    "parse_skin",
    "sclera_white_balance",
    "shades_of_gray",
    "weights_hash",
]


def analyze(
    source: str | Path | np.ndarray,
    config: Config | None = None,
    parser=None,
    landmarker=None,
) -> AnalysisResult:
    """Run the full pipeline.

    Order is load -> detect -> parse -> regions -> quality -> color -> spots ->
    lesions -> metrics. Spot detection runs before metrics because
    ``spot_count`` and ``spot_area_fraction`` are metric columns.

    ``spots`` are melanin excess (dark marks) and ``lesions`` are haemoglobin
    excess (active inflammation). Both are detected; they are different
    chromophores and answer different questions.

    The quality gate short-circuits on a hard block: an unusable capture
    returns with masks and flags populated but ``metrics`` set to None, so
    nothing downstream can mistake a number computed from an unmeasurable photo
    for a real reading. Set ``QualityConfig.short_circuit_when_unusable`` to
    False to compute anyway when debugging.

    Pass ``parser`` (from ``load_parser``) and ``landmarker`` (from
    ``load_landmarker``) when analysing many images. Both are built per call
    otherwise, and the landmarker costs ~182ms to build against ~24ms to run.
    """
    config = config or Config()

    loaded = load_image(source, config.io)
    face = detect_face(loaded.image, config, landmarker=landmarker)

    fingerprint = config_fingerprint(config)
    landmarker_fingerprint = _landmarker_hash(config)

    if face is None:
        quality = check_quality(loaded, None, None, None, config)
        return AnalysisResult(
            quality=quality,
            version=PREPROCESSING_VERSION,
            config_hash=fingerprint,
            weights_hash="",
            landmarker_hash=landmarker_fingerprint,
            image=loaded.image,
        )

    model = parser if parser is not None else load_parser(config)
    skin = parse_skin(loaded.image, face, config, parser=model)
    regions = build_regions(face, skin, config)
    masks = Masks(skin=skin, regions=regions)

    quality = check_quality(loaded, face, skin, regions, config)

    result = AnalysisResult(
        quality=quality,
        version=PREPROCESSING_VERSION,
        config_hash=fingerprint,
        weights_hash=weights_hash(config.parse),
        landmarker_hash=landmarker_fingerprint,
        masks=masks,
        face=face,
        image=loaded.image,
    )

    if not quality.usable and config.quality.short_circuit_when_unusable:
        return result

    color = correct_color(loaded.image, face, skin, config)

    # Build each expensive map ONCE and hand it to all three consumers. The
    # chromophore separation and the two large-kernel medians dominate a frame's
    # cost, and the spot detector, the lesion detector and the burden metrics
    # all want the same maps. Computing them independently ran the separation
    # three times and the median four times per frame.
    from .chromophore import separate_chromophores as _separate
    from .metrics import melanin_index_map as _melanin_map
    from .spots import local_residual as _local_residual

    melanin_map = _melanin_map(color.image, config.metrics)
    chromophores = _separate(color.image, config.metrics)
    melanin_res = _local_residual(melanin_map, skin, face, config)
    hemoglobin_res = _local_residual(chromophores[1], skin, face, config)

    spots = detect_spots(
        color.image, skin, face, regions, config,
        melanin=melanin_map, residual=melanin_res,
    )
    lesions = detect_lesions(
        color.image, skin, face, regions, config,
        hemoglobin=chromophores[1], residual=hemoglobin_res,
    )
    metrics = compute_metrics(
        color.image, skin, regions, config, spots=spots, face=face,
        melanin_residual_map=melanin_res,
        hemoglobin_residual_map=hemoglobin_res,
        chromophores=chromophores,
    )

    return AnalysisResult(
        quality=quality,
        version=PREPROCESSING_VERSION,
        config_hash=fingerprint,
        weights_hash=result.weights_hash,
        landmarker_hash=landmarker_fingerprint,
        metrics=metrics,
        spots=spots,
        lesions=lesions,
        masks=masks,
        face=face,
        color=color,
        image=loaded.image,
    )


def _landmarker_hash(config: Config) -> str:
    """Content hash of the landmarker asset, for the comparability key."""
    from .detect import file_hash, resolve_landmarker_asset

    return file_hash(resolve_landmarker_asset(config.detect))
