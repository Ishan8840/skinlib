"""Dark spot detection.

Pipeline: melanin index map -> subtract a large-kernel median (removes global
tone and shading gradients, leaving only local excess pigment) -> threshold ->
connected components -> shape and position filters.

The filters exist because the residual is not selective on its own. Nostril
rims, lash lines, cast shadows, stray hair and the mask boundary itself all
produce local dark excess, and every one of them would otherwise be reported as
a pigmented lesion.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import regions as regions_module
from .config import Config, SpotsConfig
from .chromophore import separate_chromophores
from .metrics import melanin_index_map
from .types import Face, Lesion, Spot

__all__ = [
    "detect_lesions",
    "mask_edge_margin",
    "min_mark_area",
    "detect_spots",
    "hemoglobin_residual",
    "local_residual",
    "melanin_residual",
    "residual_reference",
]


def min_mark_area(face: Face | None, config: SpotsConfig) -> float:
    """Minimum component area in pixels, from a physical mark size.

    A mark of ``min_mark_mm`` across, converted through the face's apparent
    width, so the same physical size is required at every capture distance.
    Falls back to the absolute pixel floor when the face carries no width.
    """
    if face is None or not face.width:
        return float(config.min_area_px)
    px_per_mm = float(face.width) / config.assumed_face_width_mm
    diameter = config.min_mark_mm * px_per_mm
    return max(float(config.min_area_px), np.pi / 4.0 * diameter * diameter)


def mask_edge_margin(face: Face | None, config: SpotsConfig) -> int:
    """Boundary clearance in pixels, as a fraction of face width."""
    if face is None or not face.width:
        return int(config.mask_edge_margin_px)
    return max(1, int(round(config.mask_edge_margin_face_frac * float(face.width))))


def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _background(melanin: np.ndarray, mask: np.ndarray, kernel: int) -> np.ndarray:
    """Large-kernel median of the melanin map: tone and shading, without spots.

    OpenCV's large-kernel median only accepts uint8, so the map is quantised
    against its own in-mask range rather than a fixed scale. That keeps the
    quantisation step around 0.001 melanin units instead of the 0.01 a fixed
    0..255 mapping would give, which matters because the residual threshold is
    a small multiple of the residual's own standard deviation.
    """
    inside = melanin[mask]
    low = float(inside.min())
    high = float(inside.max())
    span = max(high - low, 1e-6)

    scaled = np.clip((melanin - low) / span * 255.0, 0.0, 255.0).astype(np.uint8)
    # Outside the mask the map is meaningless; flooding it with the in-mask
    # median stops background pixels from dragging the median near the edge.
    median_level = np.uint8(np.clip((float(np.median(inside)) - low) / span * 255.0, 0, 255))
    scaled[~mask] = median_level

    blurred = cv2.medianBlur(scaled, kernel)
    return blurred.astype(np.float64) / 255.0 * span + low


def _chroma_residual_z(
    image: np.ndarray, skin_mask: np.ndarray, kernel: int
) -> tuple[np.ndarray, float, float]:
    """Shading-invariant chromaticity residual, and its robust centre/scale.

    ``log10(R) - log10(B)`` is unchanged when every channel is scaled by the
    same factor, which is what a shadow does. Pigment is not neutral, so it
    moves this quantity. Differenced against the same large-kernel median as
    the melanin map, it isolates colour change from shading change.
    """
    # +1 keeps the logs finite on crushed pixels without distorting mid-tones.
    channels = image.astype(np.float64) + 1.0
    chroma = np.log10(channels[:, :, 2]) - np.log10(channels[:, :, 0])
    residual = chroma - _background(chroma, skin_mask, kernel)

    inside = residual[skin_mask]
    median = float(np.median(inside))
    robust_sigma = float(np.median(np.abs(inside - median))) * 1.4826
    return residual, median, max(robust_sigma, 1e-9)


def _threshold(inside: np.ndarray, config: SpotsConfig) -> float:
    """Residual level above which a pixel is a spot candidate.

    The MAD path measures spread about the median with a robust estimator, so
    the artifacts being filtered out (hairline shadow, stubble) do not set the
    bar for the spots being kept. The sigma path uses a plain standard
    deviation, which those same artifacts inflate.
    """
    if config.threshold_mode == "absolute":
        return float(config.threshold_absolute)
    if config.threshold_mode == "sigma":
        return float(config.threshold_sigma * np.std(inside))

    median = float(np.median(inside))
    mad = float(np.median(np.abs(inside - median)))
    robust_sigma = mad * 1.4826
    if robust_sigma <= 1e-9:
        # A degenerate residual (a flat or posterised region) would otherwise
        # make every pixel a candidate.
        return float(config.threshold_absolute)
    return float(median + config.threshold_mad * robust_sigma)


def local_residual(
    density: np.ndarray,
    skin_mask: np.ndarray,
    face: Face,
    config: Config | None = None,
) -> np.ndarray:
    """Local excess of any density map: the map minus its large-kernel median.

    Positive means more of whatever the map measures than the surrounding skin.
    The large-kernel median carries away global level and smooth gradients, so
    what survives is local — which is what makes this work equally for a
    pigmented mark on a melanin map and an inflamed lesion on a haemoglobin one.

    Diffuse redness is removed by the same mechanism that removes shading: a
    whole-cheek flush is smooth at the kernel's scale, so it lands in the
    background rather than the residual. That is the intended behaviour — a
    lesion detector should find lesions, and the diffuse component is already
    reported by `erythema_index` and `hemoglobin_density`.
    """
    config = config or Config()
    # Kernel scales with face size so the same physical feature is background at
    # every capture distance.
    kernel = _odd(int(round(config.spots.median_kernel_face_frac * face.width)))
    return density - _background(density, skin_mask, kernel)


def melanin_residual(
    image: np.ndarray,
    skin_mask: np.ndarray,
    face: Face,
    config: Config | None = None,
) -> np.ndarray:
    """Local excess pigment, for dark spots and post-inflammatory marks.

    Exposed because the continuous burden metrics need exactly this map and must
    not depend on the component detector having run. Spot *detection* thresholds
    and componentises it; ``spot_burden`` and ``spot_contrast`` do not, which is
    why they avoid the counting-statistics noise floor that bounds ``spot_count``.
    """
    config = config or Config()
    return local_residual(
        melanin_index_map(image, config.metrics), skin_mask, face, config
    )


def hemoglobin_residual(
    image: np.ndarray,
    skin_mask: np.ndarray,
    face: Face,
    config: Config | None = None,
) -> np.ndarray:
    """Local excess haemoglobin, for active inflammatory lesions.

    The right instrument for papules and pustules, which a melanin-residual
    detector structurally cannot see: active acne is red, and melanin is brown.
    The melanin map catches what acne leaves *behind* — post-inflammatory
    hyperpigmentation — not the lesion itself.

    Built on the separated haemoglobin density rather than a raw channel ratio,
    so shading and exposure are already projected out before the local residual
    is taken (see chromophore.py).
    """
    config = config or Config()
    _, hemoglobin = separate_chromophores(image, config.metrics)
    return local_residual(hemoglobin, skin_mask, face, config)


def residual_reference(residual: np.ndarray, skin_mask: np.ndarray) -> tuple[float, float]:
    """Face-wide ``(median, robust sigma)`` of the residual.

    Taken over the whole skin mask, never per region: a shared reference is what
    makes two regions' burden figures comparable, and it is estimated from tens
    of thousands of pixels rather than a few hundred.
    """
    inside = residual[skin_mask]
    if inside.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(inside))
    sigma = float(np.median(np.abs(inside - median))) * 1.4826
    return median, max(sigma, 1e-9)


def _components(
    residual: np.ndarray,
    intensity: np.ndarray,
    skin_mask: np.ndarray,
    spots_config: SpotsConfig,
    regions: dict[str, np.ndarray] | None,
    extra_reject=None,
    face: Face | None = None,
) -> list[tuple]:
    """Threshold a residual, componentise it, and apply the shape/position filters.

    Shared by the melanin and haemoglobin detectors. Everything here is
    chromophore-agnostic: the filters reject things that are the wrong *shape*
    or in the wrong *place*, which are the same failure modes on either map.
    Only the residual and the intensity map reported per component differ.

    Yields ``(centroid, bbox, area, mean_intensity, contrast, eccentricity,
    region)`` tuples for the caller to wrap in its own record type.
    """
    if not skin_mask.any():
        return []

    threshold = _threshold(residual[skin_mask], spots_config)
    candidates = skin_mask & (residual > threshold)
    if not candidates.any():
        return []

    # Components must sit clear of the mask boundary. The boundary is where
    # nostril rims, the lash line and the hairline meet the mask, and all three
    # produce residual indistinguishable from a real feature.
    interior = regions_module.erode(skin_mask, mask_edge_margin(face, spots_config))

    count, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    if count <= 1:
        return []

    skin_area = int(skin_mask.sum())
    min_area = max(
        min_mark_area(face, spots_config),
        spots_config.min_area_frac * skin_area,
    )
    max_area = spots_config.max_area_frac * skin_area

    from skimage.measure import regionprops

    out: list[tuple] = []
    for prop in regionprops(labels, intensity_image=intensity):
        area = int(prop.area)
        if area < min_area or area > max_area:
            continue

        # Shape tests only where the shape means something. On a handful of
        # pixels eccentricity is dominated by the pixel grid, not the feature.
        if area >= spots_config.shape_min_area_px:
            # Hair strands are long and thin; shadow edges are ragged.
            if float(prop.eccentricity) > spots_config.max_eccentricity:
                continue
            if float(prop.solidity) < spots_config.min_solidity:
                continue

        # Judge boundary proximity on the whole component, not its centroid: a
        # blob straddling the hairline centres well inside the mask.
        pixels = prop.image
        if float(interior[prop.slice][pixels].mean()) < spots_config.interior_overlap_min:
            continue

        if extra_reject is not None and extra_reject(prop, pixels):
            continue

        row, col = prop.centroid
        min_row, min_col, max_row, max_col = prop.bbox
        out.append(
            (
                (float(col), float(row)),
                (int(min_col), int(min_row), int(max_col - min_col), int(max_row - min_row)),
                area,
                area / skin_area if skin_area else float("nan"),
                float(prop.intensity_mean),
                float(residual[prop.slice][pixels].mean()),
                float(prop.eccentricity),
                _region_at(regions, int(round(row)), int(round(col)), interior.shape),
            )
        )
    return out


def detect_lesions(
    image: np.ndarray,
    skin_mask: np.ndarray,
    face: Face,
    regions: dict[str, np.ndarray] | None = None,
    config: Config | None = None,
    hemoglobin: np.ndarray | None = None,
    residual: np.ndarray | None = None,
) -> list[Lesion]:
    """Detect active inflammatory lesions inside the skin mask.

    Papules and pustules as local haemoglobin excess — the signal ``detect_spots``
    structurally cannot see, since it looks at melanin and active acne is red.
    Run both: together they separate active lesions from the marks left behind.

    ``image`` should be the color-corrected image.

    Caveats worth knowing before trusting a count. Any local vascular feature is
    haemoglobin excess too, so telangiectasia, a healing scratch and the
    vascular component of under-eye darkening all qualify; nothing here
    distinguishes them from acne. Diffuse flushing is *not* picked up, because
    the large-kernel background removes it by design — that component is
    reported by ``erythema_index`` and ``hemoglobin_density`` instead.

    Thresholds are inherited from ``SpotsConfig`` and have not been calibrated
    against labelled lesions. They are a starting point, not an operating point.
    """
    config = config or Config()
    if not skin_mask.any():
        return []

    # Both maps are accepted precomputed. The chromophore separation and the
    # large-kernel median behind the residual are the two most expensive
    # operations in the library, and `analyze` needs the same two maps for the
    # burden metrics — computing them once and passing them down cuts a frame's
    # cost roughly in half.
    if hemoglobin is None:
        _, hemoglobin = separate_chromophores(image, config.metrics)
    if residual is None:
        residual = local_residual(hemoglobin, skin_mask, face, config)

    records = [
        Lesion(
            centroid=centroid,
            bbox=bbox,
            area_px=area,
            area_fraction=area_fraction,
            mean_hemoglobin_density=intensity,
            contrast=contrast,
            eccentricity=eccentricity,
            region=region,
        )
        for centroid, bbox, area, area_fraction, intensity, contrast, eccentricity, region
        in _components(residual, hemoglobin, skin_mask, config.spots, regions, face=face)
    ]

    # Fully specified ordering, as for spots: area alone leaves ties, and tied
    # records would otherwise come back in labeller order.
    if config.spots.sort_by == "area_desc":
        records.sort(key=lambda s: (-s.area_px, s.centroid[1], s.centroid[0]))
    else:
        records.sort(key=lambda s: (s.centroid[1], s.centroid[0]))
    return records


def detect_spots(
    image: np.ndarray,
    skin_mask: np.ndarray,
    face: Face,
    regions: dict[str, np.ndarray] | None = None,
    config: Config | None = None,
    melanin: np.ndarray | None = None,
    residual: np.ndarray | None = None,
) -> list[Spot]:
    """Detect dark spots inside the skin mask.

    Melanin excess: lentigines, freckles, and the post-inflammatory marks acne
    leaves behind. For *active* lesions use ``detect_lesions``, which reads the
    haemoglobin map instead — this detector structurally cannot see them.

    ``image`` should be the color-corrected image. ``regions`` supplies each
    spot's region label; without it labels come back empty.
    """
    config = config or Config()
    spots_config = config.spots

    if not skin_mask.any():
        return []

    if melanin is None:
        melanin = melanin_index_map(image, config.metrics)
    kernel = _odd(int(round(spots_config.median_kernel_face_frac * face.width)))
    if residual is None:
        residual = local_residual(melanin, skin_mask, face, config)

    # Optional: drop components that are merely darker, without the colour shift
    # real pigment produces. See SpotsConfig for why this is off by default.
    reject = None
    if spots_config.reject_neutral_shadows:
        residual_map, centre, scale = _chroma_residual_z(image, skin_mask, kernel)

        def reject(prop, pixels) -> bool:
            score = (float(residual_map[prop.slice][pixels].mean()) - centre) / scale
            return score < spots_config.shadow_chroma_z_min

    records: list[Spot] = [
        Spot(
            centroid=centroid,
            bbox=bbox,
            area_px=area,
            area_fraction=area_fraction,
            mean_melanin_index=intensity,
            contrast=contrast,
            eccentricity=eccentricity,
            region=region,
        )
        for centroid, bbox, area, area_fraction, intensity, contrast, eccentricity, region
        in _components(residual, melanin, skin_mask, spots_config, regions, reject, face=face)
    ]

    # Fully specified ordering. Area alone leaves ties, and tied records would
    # otherwise come back in whatever order the labeller happened to produce.
    if spots_config.sort_by == "area_desc":
        records.sort(key=lambda s: (-s.area_px, s.centroid[1], s.centroid[0]))
    else:
        records.sort(key=lambda s: (s.centroid[1], s.centroid[0]))
    return records


def _region_at(
    regions: dict[str, np.ndarray] | None,
    row: int,
    col: int,
    shape: tuple[int, int],
) -> str:
    """Region containing a point; "" when it belongs to none.

    Regions are mutually exclusive, so at most one can claim the pixel.
    """
    if regions is None:
        return ""
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return ""
    for name, mask in regions.items():
        if mask[row, col]:
            return name
    return ""
