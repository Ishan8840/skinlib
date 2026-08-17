"""Result dataclasses.

Structured returns only — no bare dicts crossing a stage boundary, except the
two metric maps (``global_``, ``by_region``) where the key set is data, not
schema.

Dataclasses holding numpy arrays use ``eq=False``: the generated ``__eq__``
would compare arrays and raise on the ambiguous truth value. Compare metrics,
not result objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Literal

import numpy as np

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

REGION_NAMES: Final[tuple[str, ...]] = (
    "forehead",
    "glabella",
    "left_cheek",
    "right_cheek",
    "nose",
    "perioral",
    "chin",
    "periorbital_left",
    "periorbital_right",
)

METRIC_NAMES: Final[tuple[str, ...]] = (
    # -- self-normalised: region value minus the face-wide median of the map it
    #    comes from. Exposure cancels, so these are the tracking metrics.
    #
    #    Blind to face-wide change by construction: subtracting the face median
    #    means a treatment that lightens the WHOLE face reports zero. Pair them
    #    with the chromophore densities below, which see exactly that.
    "melanin_index_rel",
    "erythema_index_rel",
    "roughness_rel",
    "uniformity_rel",
    # -- chromophore densities: absolute, and invariant to shading and exposure
    #    analytically rather than by self-normalisation (see chromophore.py).
    #    These are the only metrics that can see a uniform face-wide shift.
    #
    #    The trade against the _rel family is white balance: a neutral gain
    #    cancels here, a colour cast does not. The _rel metrics are the reverse.
    #    Check ColorResult.estimator before trusting a cross-session delta.
    "melanin_density",
    "hemoglobin_density",
    # -- exposure-invariant absolutes --
    "erythema_index",
    "erythema_index_mean",
    "uniformity",
    "roughness",
    # -- continuous pigmentation burden. Prefer these over spot_count, which
    #    has an irreducible 1/sqrt(N) counting-noise floor (CV 0.206 measured
    #    against 0.007 and 0.004 for these two). Burden is extent, contrast is
    #    intensity; neither needs the component detector to have run.
    "spot_burden",
    "spot_contrast",
    "spot_count",
    "spot_area_fraction",
    # -- ACTIVE inflammation, from local haemoglobin excess. Distinct from the
    #    spot family, which measures melanin: acne is red while a healed mark is
    #    brown, so a melanin detector sees the scar and not the lesion. Same
    #    machinery, other chromophore.
    "inflammation_burden",
    "inflammation_contrast",
    # -- internal only, see INTERNAL_ONLY_METRICS --
    "melanin_index",
    "erythema",
    "erythema_mean",
    "ita",
    "monk_bin",
)

# Metrics that measure the capture as much as the skin, and must not be shown
# to a user or compared between sessions.
#
# Measured on nine casual captures of one face: ITA correlates r = 0.94 with
# mean skin luminance and melanin_index r = -0.97. Exposure varying 17 L* units
# accounted for 34 of the 36 degrees of observed ITA range. That is not a
# cross-session caveat — within a single capture these are reporting how bright
# the room was, so they are compromised immediately, not merely over time.
#
# `monk_bin` additionally must never be used for fairness or bias evaluation:
# it is derived from an unofficial ITA mapping, so auditing tone fairness with
# it tests the approximation against itself and is circular.
#
# `melanin_index` is SUPERSEDED by `melanin_density`, which measures the same
# pigment with exposure removed analytically instead of not at all (10x better
# against shading, 28x against exposure — see chromophore.py). It is retained
# because stored results reference it and because it is the map spot detection
# still runs on. Track `melanin_density`.
INTERNAL_ONLY_METRICS: Final[frozenset[str]] = frozenset(
    {"melanin_index", "erythema", "erythema_mean", "ita", "monk_bin"}
)

# Safe to surface and to track.
DISPLAY_SAFE_METRICS: Final[tuple[str, ...]] = tuple(
    name for name in METRIC_NAMES if name not in INTERNAL_ONLY_METRICS
)


class QualityFlag(StrEnum):
    """Flag vocabulary. ``QualityResult.flags`` is ``list[str]`` of these values."""

    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    SIDE_LIT = "side_lit"
    TOO_FAR = "too_far"
    TOO_CLOSE = "too_close"
    BLURRY = "blurry"
    HIGH_SPECULAR = "high_specular"
    POSSIBLY_FILTERED = "possibly_filtered"
    # Source long edge below the working resolution. The image is NOT upscaled
    # (that would invent high-frequency content), so texture-band metrics are
    # measured at reduced detail.
    LOW_RESOLUTION = "low_resolution"
    # Not in the original spec, but the parse stage can produce a mask too
    # small to measure (heavy occlusion, extreme pose). Surfaced as a flag
    # rather than an exception, per the no-silent-failure rule.
    MASK_TOO_SMALL = "mask_too_small"


# --------------------------------------------------------------------------
# stage outputs
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LoadedImage:
    """Output of ``load_image``: the working image plus its provenance.

    The working image is at most ``IOConfig.max_long_edge_px`` on its long
    edge, and is NEVER upscaled. ``source_size`` is retained so the quality
    gate can flag a source that was too small to begin with — information the
    resized array itself no longer carries.
    """

    image: np.ndarray  # uint8 BGR, working resolution
    source_size: tuple[int, int]  # (height, width) as decoded
    working_size: tuple[int, int]  # (height, width) after resize
    # working_long_edge / source_long_edge. Exactly 1.0 when untouched, and
    # always <= 1.0 — upscaling is not a thing this library does.
    scale: float

    @property
    def source_long_edge(self) -> int:
        return max(self.source_size)


@dataclass(frozen=True, eq=False)
class Face:
    """Output of ``detect_face``."""

    # (N, 2) float32 pixel coordinates in the working image. N == 478 when
    # refine_landmarks is on, else 468.
    landmarks: np.ndarray
    # (x, y, w, h) integer bbox derived from the landmark hull.
    bbox: tuple[int, int, int, int]
    # (height, width) of the working image the landmarks index into.
    image_size: tuple[int, int]
    # How many faces MediaPipe returned, before primary-face selection.
    n_faces: int
    # bbox area / frame area, precomputed for the quality gate.
    area_fraction: float
    # (N,) float32 landmark depth, same units as x, negative toward the camera.
    # A LEARNED single-view estimate, not a measurement: it carries the gross
    # shape of a face and nothing at the scale of a wrinkle. Empty when
    # detection predates this field.
    landmarks_z: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    @property
    def width(self) -> int:
        """Face bbox width; the scale reference for size-relative kernels."""
        return self.bbox[2]


@dataclass(frozen=True, eq=False)
class Masks:
    """Output of ``parse_skin`` + ``build_regions``."""

    skin: np.ndarray  # bool, (H, W)
    regions: dict[str, np.ndarray] = field(default_factory=dict)  # bool, (H, W) each

    def region(self, name: str) -> np.ndarray:
        """Region mask by name, raising a clear error on a typo."""
        try:
            return self.regions[name]
        except KeyError:
            raise KeyError(f"unknown region {name!r}; known: {sorted(self.regions)}") from None


@dataclass(frozen=True)
class QualityResult:
    """Output of ``check_quality``. Never raises; failures become flags.

    ``usable`` covers only the hard blocks — captures that cannot be measured
    at all. Everything softer is expressed through ``unreliable_metrics``:
    metrics still compute, and the caller decides what to trust. A capture with
    specular highlights has sound colour metrics and worthless texture metrics,
    and a longitudinal tracker should be able to keep the former.
    """

    usable: bool
    flags: list[str]
    # Metric names made untrustworthy by the flags that fired. Empty when
    # nothing advisory fired. Names match types.METRIC_NAMES.
    unreliable_metrics: frozenset[str] = frozenset()
    # Raw values behind each check, so a threshold can be tuned against real
    # numbers instead of guessed at. Keys are stable across versions.
    measures: dict[str, float] = field(default_factory=dict)

    def has(self, flag: str) -> bool:
        return str(flag) in self.flags

    def trusted(self, metric: str) -> bool:
        """True when this metric is not undermined by any flag that fired."""
        return metric not in self.unreliable_metrics


@dataclass(frozen=True, eq=False)
class ColorResult:
    """Output of ``correct_color``.

    ``estimator`` is the provenance field: it says which estimator produced the
    gains that were actually applied. When a tracked metric jumps between two
    sessions, the first thing to check is whether the illuminant estimate
    changed hands.
    """

    image: np.ndarray  # uint8 BGR, color-corrected
    # Per-channel gains applied, in BGR order.
    gains: tuple[float, float, float]
    # Which estimator produced `gains`.
    estimator: Literal["sclera", "shades_of_gray"]
    # Why sclera was not used, when it was not. None when sclera was applied or
    # never enabled. E.g. "confidence 0.31 < 0.45", "eyes_closed", "no_pixels".
    fallback_reason: str | None = None
    # 0..1, None when sclera was disabled entirely. Reported even when the
    # sclera gains were discarded — a low value is diagnostic.
    sclera_confidence: float | None = None
    sclera_pixel_count: int = 0
    # Always populated, even when sclera won: the baseline is cheap and having
    # both makes disagreement between estimators visible.
    shades_of_gray_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Spot:
    """One detected dark spot. Coordinates are in working-image pixels."""

    centroid: tuple[float, float]  # (x, y)
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    area_px: int
    area_fraction: float  # of skin mask area
    mean_melanin_index: float
    # Contrast against the local background (the median-blur residual).
    contrast: float
    eccentricity: float
    # Region containing the centroid; "" when it falls outside every region.
    region: str


@dataclass(frozen=True)
class Lesion:
    """One detected inflammatory lesion. Coordinates are in working-image pixels.

    Separate from :class:`Spot` because the two measure different chromophores
    and must never be summed or compared: a Spot is melanin excess (a dark mark,
    including the stain acne leaves behind) and a Lesion is haemoglobin excess
    (an active, red one). A single type with a mode flag would make mixing them
    an easy mistake to make silently.

    As with ``spot_count``, ``len(lesions)`` carries a 1/sqrt(N) counting-noise
    floor. Use these records to SHOW someone where their lesions are, and
    ``inflammation_burden`` to track whether they are improving.
    """

    centroid: tuple[float, float]  # (x, y)
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    area_px: int
    area_fraction: float  # of skin mask area
    mean_hemoglobin_density: float
    # Contrast against the local background (the median-blur residual).
    contrast: float
    eccentricity: float
    # Region containing the centroid; "" when it falls outside every region.
    region: str


@dataclass(frozen=True, eq=False)
class FrameReport:
    """One frame's fate within a burst.

    Kept frames contribute to the session estimate; rejected ones are still
    reported, because *why* frames were dropped is how a capture protocol gets
    debugged. A session that silently discarded half its frames looks identical
    to a clean one otherwise.
    """

    name: str
    kept: bool
    # "" when kept. One of: unusable, blurred, moved, no_face.
    rejected: str = ""
    flags: tuple[str, ...] = ()
    # Variance of the Laplacian inside the skin mask. Higher is sharper.
    sharpness: float = float("nan")
    face_width: float = float("nan")
    sclera_confidence: float = float("nan")
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class SessionResult:
    """Aggregate of a burst: one estimate per metric, with its own error bar.

    A single frame gives a number with no indication of how much to trust it.
    A burst gives the number AND the spread that produced it, measured on this
    capture rather than assumed from a reference table — so a tracker can refuse
    to report a change smaller than the session's own noise.

    ``metrics`` are medians across kept frames. ``noise`` is the robust spread of
    a SINGLE frame; ``standard_error`` is the uncertainty of the median, which
    falls as 1/sqrt(n); ``detectable_change`` is the smallest difference between
    two such sessions that is not noise, at 95%.
    """

    metrics: dict[str, float]
    by_region: dict[str, dict[str, float]]
    # Robust spread of a single frame, per metric.
    noise: dict[str, float] = field(default_factory=dict)
    # Uncertainty of the reported median. ~1.253 * noise / sqrt(n).
    standard_error: dict[str, float] = field(default_factory=dict)
    # 2.77 * standard_error: two sessions must differ by more than this.
    detectable_change: dict[str, float] = field(default_factory=dict)

    frames: list[FrameReport] = field(default_factory=list)
    # Session-level warnings: ae_not_locked, subject_moved, high_frame_loss,
    # single_frame.
    flags: list[str] = field(default_factory=list)

    # Per-pixel view-dependent (specular) component, recovered from how much
    # each pixel varies across the burst once frames are aligned. NaN outside
    # the skin mask, None when registration did not run.
    specular_map: np.ndarray | None = None
    # Temporal median of the aligned frames: the same face with glare removed.
    composite: np.ndarray | None = None

    # Detections from the REFERENCE frame, for showing a user where their marks
    # are. Not aggregated: a count carries a 1/sqrt(N) noise floor, so these are
    # for display and `spot_burden`/`inflammation_burden` are for tracking.
    spots: list[Spot] = field(default_factory=list)
    lesions: list[Lesion] = field(default_factory=list)

    version: str = ""
    config_hash: str = ""
    weights_hash: str = ""
    landmarker_hash: str = ""

    @property
    def n_kept(self) -> int:
        return sum(1 for frame in self.frames if frame.kept)

    @property
    def comparable_key(self) -> tuple[str, str, str, str]:
        return (self.version, self.config_hash, self.weights_hash, self.landmarker_hash)

    def trusted_change(self, metric: str, other: "SessionResult") -> bool:
        """Is the difference in ``metric`` between two sessions real?

        The question every longitudinal tracker actually needs answered, and the
        one a bare float cannot answer. Uses the larger of the two sessions'
        detectable changes, since a comparison is only as sharp as its blunter
        half.
        """
        first, second = self.metrics.get(metric), other.metrics.get(metric)
        if first is None or second is None:
            return False
        if not (np.isfinite(first) and np.isfinite(second)):
            return False
        bar = max(
            self.detectable_change.get(metric, float("nan")),
            other.detectable_change.get(metric, float("nan")),
        )
        return bool(np.isfinite(bar) and abs(first - second) > bar)


@dataclass(frozen=True, eq=False)
class MetricsResult:
    """Output of ``compute_metrics``.

    A region whose pixel count is below ``MetricsConfig.min_region_pixels``
    gets NaN for every metric rather than a number computed from noise.
    """

    global_: dict[str, float]
    by_region: dict[str, dict[str, float]]
    # Pixels actually measured, per region plus "global". Context for NaNs.
    pixel_counts: dict[str, int] = field(default_factory=dict)
    # D(x, y) = melanin(x, y) - median(melanin over the face), float32.
    # NaN outside the skin mask — not zero, which would read as "measured, no
    # deviation". Retained because the per-region scalars discard everything
    # spatial: new localized spots, diffuse regional shifts, cheek-vs-cheek
    # asymmetry, haloing around a lesion.
    normalized_map: np.ndarray | None = None
    # The face-wide medians the _rel metrics were measured against. Stored so a
    # stored result can be re-derived or re-referenced later.
    face_reference: dict[str, float] = field(default_factory=dict)

    def display(self, region: str | None = None) -> dict[str, float]:
        """Metrics safe to surface, with the exposure-compromised ones removed."""
        source = self.global_ if region is None else self.by_region[region]
        return {name: source[name] for name in DISPLAY_SAFE_METRICS if name in source}


@dataclass(frozen=True, eq=False)
class AnalysisResult:
    """Output of ``analyze``.

    Comparability between two stored results is
    ``version + config_hash + weights_hash``. The version string is a
    hand-maintained changelog and is not load-bearing: the two hashes are
    computed from what actually ran, so a tuned threshold or a swapped
    checkpoint is detectable even if nobody bumped the version.

    ``metrics`` is None when the quality gate short-circuited. ``quality``,
    ``version`` and the hashes are always populated.
    """

    quality: QualityResult
    version: str
    # Hash of the fully-resolved Config (config.config_fingerprint).
    config_hash: str
    # Content hash of the BiSeNet checkpoint; "" when parsing never ran.
    weights_hash: str
    # Content hash of the MediaPipe face_landmarker asset; "" when detection
    # never ran. Separate from weights_hash because it is a separate model with
    # its own release cadence: a landmarker update moves every region boundary,
    # which moves every per-region metric, and that must be as detectable as a
    # BiSeNet swap.
    landmarker_hash: str = ""
    metrics: MetricsResult | None = None
    # Melanin excess: dark marks, including post-inflammatory ones.
    spots: list[Spot] = field(default_factory=list)
    # Haemoglobin excess: ACTIVE inflammatory lesions. A separate list, and a
    # separate type, because the two measure different chromophores and must
    # never be summed — a healed mark and an active papule are not the same
    # finding, and treating them as one would report a healing face as a
    # worsening one.
    lesions: list[Lesion] = field(default_factory=list)
    masks: Masks | None = None
    face: Face | None = None
    color: ColorResult | None = None
    # Working image after IO resize/EXIF, before color correction. Kept so
    # tools/visualize.py can overlay without redoing the load.
    image: np.ndarray | None = None

    @property
    def comparable_key(self) -> tuple[str, str, str, str]:
        """Identity of the pipeline that produced this result.

        Two results are only comparable longitudinally when this key matches.
        Every component is computed from what actually ran, except ``version``,
        which is the human-readable changelog.
        """
        return (self.version, self.config_hash, self.weights_hash, self.landmarker_hash)
