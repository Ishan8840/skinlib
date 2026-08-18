"""Thresholds and tunables.

Every number that could change a metric value lives here. Nothing in this
package reads a magic constant out of a function body.

All config dataclasses are frozen: a ``Config`` instance is a value, safe to
share between calls and threads. There is no global mutable state — pass a
``Config`` explicitly or accept the default.

Changing any default here shifts output values. ``PREPROCESSING_VERSION``
should be bumped when that happens, but comparability does not depend on
remembering to: :func:`config_fingerprint` hashes the resolved config, and the
result carries that hash. See README "Comparability".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from collections.abc import Iterable, Mapping

# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IOConfig:
    """Image loading. Affects pixel values, so it is versioned config."""

    apply_exif_orientation: bool = True

    # Long-edge working resolution. DOWNSCALE ONLY — a smaller source is never
    # resampled up, because upsampling invents high-frequency content that
    # corrupts `roughness` and inflates spot detection. A source below
    # QualityConfig.min_long_edge_px keeps its native size and raises the
    # `low_resolution` flag.
    max_long_edge_px: int | None = 1536

    # Pinned deliberately: different resampling kernels put different energy in
    # the texture band, which is a silent longitudinal break. INTER_AREA is the
    # correct choice for downscale (it integrates over the source footprint
    # instead of point-sampling). Changing this invalidates stored roughness
    # and spot metrics.
    downscale_interpolation: Literal["area"] = "area"


@dataclass(frozen=True)
class DetectConfig:
    """MediaPipe FaceMesh (Tasks API ``FaceLandmarker``).

    MediaPipe >= 1.0 removed the legacy ``solutions.face_mesh`` module, so the
    landmarker model is an external asset like the BiSeNet checkpoint. Same 478
    landmarks, same topology.
    """

    # Not auto-downloaded. Resolution order: this field, then
    # $SKINLIB_FACE_LANDMARKER. Excluded from the config fingerprint — the
    # asset's content hash is stamped on the result instead.
    landmarker_asset_path: Path | None = None

    # Detect more than one so `multiple_faces` can be flagged rather than
    # silently analysing whichever face MediaPipe returned first.
    max_faces: int = 3
    min_detection_confidence: float = 0.5
    # Iris/eyelid refinement: needed for periorbital regions and for locating
    # sclera. Costs ~2ms, adds landmarks 468..477.
    refine_landmarks: bool = True
    # Always True here: video mode carries state between calls and would break
    # determinism. Exposed so it is visible, not so it is changed.
    static_image_mode: bool = True
    # If several faces pass detection, this picks the analysed one when the
    # caller opts to continue anyway (largest bbox = the subject).
    primary_face: Literal["largest", "first"] = "largest"


@dataclass(frozen=True)
class ParseConfig:
    """BiSeNet face parsing -> binary skin mask."""

    # Not auto-downloaded. Resolution order: this field, then
    # $SKINLIB_BISENET_WEIGHTS. Excluded from the config fingerprint (it is a
    # filesystem location); the checkpoint's content hash is stamped on the
    # result separately.
    weights_path: Path | None = None
    input_size: int = 512
    device: Literal["cpu", "cuda"] = "cpu"

    # CelebAMask-HQ 19-class label map (zllrunning/face-parsing.PyTorch):
    #   0 background  1 skin      2 l_brow  3 r_brow  4 l_eye   5 r_eye
    #   6 eyeglasses  7 l_ear     8 r_ear   9 earring 10 nose  11 mouth
    #  12 u_lip      13 l_lip    14 neck   15 necklace 16 cloth 17 hair 18 hat
    #
    # NOTE: nose (10) is skin and is required as a region, so it is kept.
    # Ears (7,8) are dropped: they are not facial skin and are lit differently.
    skin_classes: tuple[int, ...] = (1, 10)
    # Grown before subtraction so anti-aliased class boundaries do not leak
    # lash/lip/hair pixels into the skin mask.
    exclude_classes: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18)
    exclude_dilate_px: int = 5
    # Pulls the mask off the face silhouette, where background bleeds in.
    skin_erode_px: int = 3
    # Below this the mask is unusable; raised as a quality flag, not an exception.
    min_skin_pixels: int = 5000

    # Clip the mask to the landmark face oval. BiSeNet's `skin` class is
    # anatomically correct but not facial: on a bald or shaven head the scalp
    # parses as skin and would be measured as face. Forehead, temples and jaw
    # all survive the clip; scalp, neck and ears do not.
    limit_to_face_oval: bool = True
    face_oval_dilate_px: int = 8

    # Carve eyes, brows and lips from landmark geometry in addition to the
    # BiSeNet classes. The parser's class boundaries are soft and it
    # occasionally files an eyelid or lash pixel as skin; landmarks do not
    # drift that way. Belt and braces, because a single sclera or lip pixel
    # inside the mask is a large outlier in a* and in melanin index.
    carve_landmark_features: bool = True
    feature_dilate_px: int = 4

    # BiSeNet has no nostril class (nostrils fall inside `nose`). Carved out
    # from FaceMesh nostril landmarks, which are reliable.
    suppress_nostrils: bool = True
    nostril_dilate_px: int = 3

    # BiSeNet has no beard class either. Facial hair is suppressed by local
    # darkness + high texture within the lower-face regions only.
    #
    # Tuned to OVER-suppress on purpose: discarding some real chin skin costs
    # far less than counting beard shadow as pigmentation, which would inflate
    # melanin_index and manufacture spots.
    suppress_facial_hair: bool = True
    # Pixel is hair-like if L* is this many sigma below the region median.
    # Lower = more aggressive.
    facial_hair_darkness_sigma: float = 1.2
    facial_hair_texture_min: float = 0.04
    # Hair-like pixels are grown before removal, to catch the soft shadow halo
    # around each stubble cluster.
    facial_hair_dilate_px: int = 4
    facial_hair_regions: tuple[str, ...] = ("perioral", "chin")


@dataclass(frozen=True)
class RegionConfig:
    """Landmark -> region polygons.

    Regions are built from FaceMesh landmarks and then intersected with the
    skin mask, so every region mask is a subset of the skin mask.
    """

    # Cut regions in the face's own 3D frame instead of the 2D image-plane
    # projection. A projected frame foreshortens when the head turns, so a
    # threshold slides across different anatomy and the metric moves because the
    # REGION moved rather than because the skin did.
    #
    # Measured on the 13-frame `angle` set against the `constant` baseline, as
    # median per-region angle penalty:
    #
    #                        2D     3D
    #   spot_burden         4.63x  3.87x   1.20x better
    #   inflammation_burden 4.58x  3.79x   1.21x better
    #   roughness           4.33x  3.61x   1.20x better
    #   uniformity          5.46x  5.04x   1.08x better
    #   melanin_density     6.19x  6.40x   0.97x (no gain)
    #
    # So region drift is REAL but explains only about a fifth of the angle
    # penalty; 3.87x remains after removing it. The rest is most likely
    # irreducible — at a different angle a different patch of skin is visible,
    # sampled at a different density, and no choice of coordinate frame recovers
    # what the pose has hidden.
    #
    # On by default anyway: it is the more correct measurement (anatomy measured
    # in an anatomical frame), costs ~50ms of a ~3s pipeline, and changes total
    # region area by 1%. Falls back to the 2D frame when a detection carries no
    # landmark depth.
    canonical_frame: bool = True

    # FaceMesh has no landmarks above the brows. Forehead is extrapolated
    # upward from the brow line by this fraction of face height, then clipped
    # by the face oval and the skin mask (which stops at the hairline).
    #
    # Set generously on purpose: over-reaching is free because both clips are
    # hard, whereas under-reaching silently discards good forehead skin and
    # leaves the region measuring only the brow-adjacent band.
    forehead_height_frac: float = 0.45
    # Ring around the eye, as a fraction of interocular (eye-centre to
    # eye-centre) distance. Kept tight: periorbital outranks every neighbouring
    # region, so an over-wide ring silently annexes the glabella, the nose
    # bridge and the upper cheeks rather than merely overlapping them.
    periorbital_outer_frac: float = 0.20
    # Glabella box between the brows, as a fraction of interbrow distance.
    glabella_width_frac: float = 1.0
    glabella_height_frac: float = 0.55
    # Cheeks are inset from the silhouette to avoid grazing-light falloff.
    cheek_inset_frac: float = 0.06
    # Perioral ring around the lips, as a fraction of mouth width.
    perioral_ring_frac: float = 0.30
    # Applied to every region polygon before mask intersection.
    polygon_erode_px: int = 2

    # Regions are made mutually exclusive in this order (first wins) so a pixel
    # is never counted twice in per-region aggregates.
    #
    # Periorbital MUST outrank the cheeks: the cheek polygon extends up under
    # the eye, and under-eye skin is the dark-circle signal. If cheek won, that
    # signal would be averaged away into a much larger region.
    exclusive: bool = True
    priority: tuple[str, ...] = (
        "periorbital_left",
        "periorbital_right",
        "glabella",
        "nose",
        "perioral",
        "forehead",
        "left_cheek",
        "right_cheek",
        "chin",
    )


# Flag -> metrics that flag makes untrustworthy.
#
# A flag is not a verdict on the whole capture. Specular highlights wreck
# texture and spot detection while barely moving ITA; a beauty filter destroys
# pores while leaving colour roughly intact; side lighting mainly breaks
# left/right comparability of tone. The caller keeps what survives.
#
# Pairs rather than a dict: the config must stay immutable AND survive
# `dataclasses.asdict`, which deep-copies unrecognised values (a
# MappingProxyType cannot be deep-copied) and mutates a plain dict in place.
_DEFAULT_UNRELIABLE: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Blown highlights: local brightness is the illuminant, not the skin.
    ("high_specular", ("roughness", "spot_count", "spot_area_fraction", "uniformity")),
    # Smoothing filters flatten pores and erase small pigment features.
    ("possibly_filtered", ("roughness", "spot_count", "spot_area_fraction", "uniformity")),
    # A luminance gradient across the face masquerades as a tone gradient.
    ("side_lit", ("uniformity", "melanin_index", "ita", "monk_bin")),
    # Too few pixels per pore/spot for the texture band to mean anything.
    ("low_resolution", ("roughness", "spot_count", "spot_area_fraction")),
)


@dataclass(frozen=True)
class QualityConfig:
    """Quality gate. Runs before color correction, on raw pixels."""

    # Mean L* (0..100) inside the skin mask.
    #
    # BOTH bounds are BACKSTOPS ONLY — they catch a frame with no signal at
    # either end, and both must sit outside any real skin tone. Neither is the
    # exposure test any more; `too_dark` and `too_bright` fire on clipping.
    #
    # The lower bound was 32.0, which rejected correctly exposed deep skin:
    # inverting the ITA scale, mean L* of 32 is ITA -42 to -56 depending on b*,
    # well inside the "dark" class `monk_ita_edges` claims to classify.
    #
    # The upper bound was 82.0, which is the same error mirrored: ITA > 55 skin
    # reaches L* 84.6 at b* = 20, so the ceiling rejected the lightest skin
    # exactly as the floor rejected the deepest. See quality.py.
    luminance_band: tuple[float, float] = (12.0, 95.0)

    # Fraction of skin pixels crushed against black before a capture counts as
    # underexposed. THIS is the real `too_dark` test, and it is tone-independent:
    # losing information is not the same as reflecting less light.
    #
    # 0.02 sits ~10x above the worst observed good capture (0.00199, from the
    # `distance` set; every other set measured below 0.0001) and does catch the
    # genuinely dark clinical image observed at 0.027.
    shadow_clipped_max: float = 0.02

    # The same test at the other end, and for the same reason. Mean L* rises
    # with skin lightness as well as with exposure, so the old `too_bright` bar
    # at L* > 82 was wrong in both directions at once: it would reject correctly
    # exposed very light skin (ITA > 55 reaches L* 84.6 at b* = 20) while
    # staying silent on a capture measured with 17% of the skin already blown.
    #
    # 0.02 mirrors `shadow_clipped_max` and sits ~5x above the worst observed
    # good capture (0.00406, from the `flash` set, where a bright on-axis source
    # legitimately puts a little of the skin at the ceiling).
    highlight_clipped_max: float = 0.02
    # |L_left - L_right| / mean(L) above this -> side_lit.
    side_lit_max_frac: float = 0.15
    # Face bbox area / frame area.
    face_area_frac_band: tuple[float, float] = (0.06, 0.55)
    # Variance of the Laplacian of LOG luminance, inside the skin mask.
    #
    # The log domain makes this brightness-invariant; an absolute threshold on
    # the linear Laplacian rejected darker skin as blurred. See quality.py.
    #
    # 0.0004 sits 2.2x below the worst good capture across all five sets
    # (0.00087, from `distance`) and 4.9x above a sigma=1 Gaussian blur
    # (0.000082). The genuinely blurred frames in the `lighting` set measure
    # down to 0.000332 and are correctly rejected.
    blur_log_laplacian_var_min: float = 0.0004
    # Superseded by the above. Kept so a stored config still resolves, and so
    # the linear measure remains available in QualityResult.measures.
    blur_laplacian_var_min: float = 55.0
    # Source long edge below this -> low_resolution (and no upscale happens).
    min_long_edge_px: int = 900
    # Specular pixel: HSV V above `specular_v_ratio * median(V over skin)`, and
    # S <= s_max. Both normalised 0..1.
    #
    # RELATIVE to the skin's own brightness, because a highlight is bright
    # relative to the diffuse level around it. The old absolute bar of 0.92 did
    # not shift on darker skin, it stopped working: the specular fraction went
    # 0.00628 -> exactly 0.00000 as one unchanged photo was scaled toward deeper
    # tones, so shine was undetectable on anything but light skin.
    #
    # 1.25 reproduces the old measure on the unchanged image (0.00688 against
    # 0.00628) and holds within 1.46x across the same brightness range.
    specular_v_ratio: float = 1.25
    # Absolute floor, a backstop only: it stops a near-black region inventing
    # highlights from quantisation noise. Far below any real skin tone.
    specular_v_min: float = 0.15
    specular_s_max: float = 0.18
    specular_frac_max: float = 0.05

    # possibly_filtered: BOTH conditions must hold.
    #  (a) high-frequency energy in the skin band is anomalously low
    #      (mean |img - gaussian(img, sigma)| inside mask, contrast-normalised)
    filtered_hf_sigma: float = 1.5
    filtered_hf_energy_min: float = 0.012
    #  (b) the fine texture band has collapsed relative to the coarse band.
    #
    # This replaces a nose-vs-cheek texture comparison, which measurement
    # showed runs BACKWARDS. Smoothing raises the nose/cheek ratio rather than
    # flattening it (cheek texture falls to the noise floor while
    # edge-preserving filters explicitly protect the nose's structural edges),
    # and an unfiltered fixture measured 1.17 against a 1.18 "uniform"
    # threshold — so the check fired on clean photos and stayed silent on
    # filtered ones. Measured nose/cheek ratios, original -> heavy bilateral:
    # 1.17 -> 1.99, 1.54, 1.96 across filters.
    #
    # Scale selectivity is what actually distinguishes a beauty filter: it
    # removes pore-scale detail while leaving coarse facial structure intact,
    # so fine/coarse collapses. Measured on three fixtures, original -> heavy
    # bilateral: 0.591 -> 0.259, 0.397 -> 0.191, 0.348 -> 0.149.
    filtered_fine_sigma: float = 1.0
    filtered_coarse_sigma: float = 4.0
    filtered_band_ratio: float = 2.0
    filtered_scale_ratio_min: float = 0.32
    # Still computed and reported in QualityResult.measures for inspection,
    # since it is a genuine descriptor of the face — just not a filter
    # detector. Nothing keys off it.
    filtered_texture_pair: tuple[str, str] = ("nose", "left_cheek")

    # Hard blocks: the capture cannot be measured at all, so metrics are not
    # computed. Everything else is advisory — metrics compute, and the affected
    # ones are named in QualityResult.unreliable_metrics.
    blocking_flags: tuple[str, ...] = (
        "no_face",
        "multiple_faces",
        "too_dark",
        "too_bright",
        "blurry",
        "too_far",
        "too_close",
        "mask_too_small",
    )
    unreliable_metrics_by_flag: tuple[tuple[str, tuple[str, ...]], ...] = _DEFAULT_UNRELIABLE

    # If False, metrics are computed even when usable is False (debugging).
    short_circuit_when_unusable: bool = True

    def unreliable_metrics_for(self, flags: Iterable[str]) -> frozenset[str]:
        """Union of the metrics the given flags make untrustworthy."""
        lookup = dict(self.unreliable_metrics_by_flag)
        return frozenset(
            metric for flag in flags for metric in lookup.get(str(flag), ())
        )


@dataclass(frozen=True)
class ColorConfig:
    """Color constancy.

    Two estimators, with precedence. Sclera is PRIMARY when it clears its
    confidence threshold: eye-white is an actual neutral reference, whereas
    whole-frame shades-of-grey assumes the scene averages to grey and fails
    against a strongly coloured wall (common in bathrooms, where these photos
    get taken). Shades-of-grey is the fallback.

    ColorResult.estimator records which one produced the final gains.
    """

    # -- shades of grey (fallback, and the always-computed baseline) --
    minkowski_p: int = 6
    # Illuminant is estimated from the WHOLE frame, not from skin pixels.
    # Estimating from skin alone would drive mean skin colour toward grey and
    # destroy the erythema/melanin signal we are trying to measure.
    sog_estimate_from: Literal["frame", "skin"] = "frame"
    # Saturated and near-black pixels bias the Minkowski mean; dropped.
    sog_exclude_above: int = 250
    sog_exclude_below: int = 8
    # Per-channel gain clamp, guards against a wild estimate on a coloured wall.
    sog_gain_clamp: tuple[float, float] = (0.5, 2.0)

    # -- sclera reference (primary when confident) --
    sclera_enabled: bool = True
    sclera_as_primary: bool = True
    sclera_min_pixels: int = 60
    # Pixel count at which the count term of the confidence saturates.
    #
    # Calibrated against observed captures rather than assumed: nine real
    # photos yielded 32 to 136 sclera pixels, typically around 100. The
    # previous formula saturated at 4x sclera_min_pixels = 240, so a typical
    # capture scored at most 0.41 on that term alone and could never clear the
    # 0.45 bar however clean the sclera was. The gate never opened — sclera was
    # found in eight of nine and used in none.
    sclera_saturation_pixels: int = 110
    # Sclera candidate inside the eye polygon: bright and desaturated.
    sclera_v_min: float = 0.55
    sclera_s_max: float = 0.30
    # Trimmed percentile band used as the neutral reference (rejects lashes,
    # limbus, catchlights).
    sclera_percentile_band: tuple[float, float] = (60.0, 90.0)
    sclera_gain_clamp: tuple[float, float] = (0.75, 1.35)
    # Below this confidence the sclera gains are discarded and shades-of-grey
    # is used instead. The confidence is reported either way.
    sclera_min_confidence: float = 0.45


@dataclass(frozen=True)
class MetricsConfig:
    """Metric definitions."""

    erythema_percentile: float = 75.0
    # Band-pass = gaussian(sigma) - gaussian(sigma * ratio), on L*-derived grey.
    #
    # Sigma is a fraction of FACE WIDTH, not a fixed pixel count. A fixed sigma
    # band-passes a different physical scale at every capture distance, so it
    # reports how close the subject stood rather than how their skin looks.
    # `face_area_frac_band` admits a 9.2x area range (~3x linear), and measured
    # across that range on a fixed skin patch a fixed sigma=3.0 swings
    # roughness 0.0473 -> 0.1031, a factor of 2.2. Scaling with face width holds
    # the same patch to 0.0473 -> 0.0457, a 3.4% residual which is genuine
    # detail lost to downsampling.
    #
    # 0.008 reproduces the historical sigma=3.0 at a 375px face, the middle of
    # the 274-392px range the fixtures actually occupy at 1536px working size.
    roughness_sigma_face_frac: float | None = 0.008
    # Fallback when face width is unknown (region_metrics called without a
    # face). Carries the old fixed-pixel behaviour and its distance sensitivity.
    roughness_sigma: float = 3.0
    roughness_sigma_ratio: float = 2.0
    # melanin_index = log10(1 / R_norm); R_norm = clip(R/255, floor, 1).
    melanin_r_floor: float = 1.0 / 255.0

    # -- chromophore separation (see chromophore.py) --
    #
    # Melanin and haemoglobin absorbance directions in optical-density space,
    # RGB order (Tsumura et al., JOSA A 1999). Published constants, not fitted
    # here. They sit 69.8 degrees apart after de-shading (condition number 1.7),
    # so the 2x2 solve is well posed.
    #
    # These are the one place where a literature value directly sets a reported
    # metric, so they are config rather than module constants: a lab with its
    # own measured basis should be able to supply it, and config_hash will
    # record that they did.
    chromophore_melanin_axis: tuple[float, float, float] = (0.4143, 0.8697, 0.2843)
    chromophore_hemoglobin_axis: tuple[float, float, float] = (0.2988, 0.6838, 0.6657)
    # Reflectance floor before the log. Matches melanin_r_floor so both signals
    # treat a crushed pixel identically.
    chromophore_od_floor: float = 1.0 / 255.0
    # -- continuous pigmentation burden --
    #
    # `spot_count` has an IRREDUCIBLE noise floor. Counting discrete objects
    # near a detection boundary is a Poisson process: measured across 12
    # identical captures the observed CV tracked 1/sqrt(N) at every filter stage
    # (ratio 1.02, 0.85, 1.11, 1.46), so at N=50 the count cannot do better than
    # +-7 however good the detector is. Raising the area floor makes it WORSE,
    # because smaller N means larger 1/sqrt(N) — measured 0.21 -> 0.30 -> 0.74 as
    # the count fell 50 -> 5 -> 2.
    #
    # These two measure the same signal without the discretisation step, and on
    # the same 12 captures came out 30-50x more repeatable:
    #
    #   spot_count           CV 0.206
    #   spot_area_fraction   CV 0.130
    #   spot_burden          CV 0.007
    #   spot_contrast        CV 0.004
    #
    # Sensitivity is intact: burden spans 76x across regions of one face (180x
    # its own noise floor) and 2.4x across four different faces.
    #
    # Burden is extent (how much of the skin is affected), contrast is intensity
    # (how dark the affected part is). Clinical pigmentation scales such as MASI
    # are also an area term times a darkness term, for the same reason: one
    # number cannot separate a few dark marks from many faint ones.
    #
    # Multiples of the face-wide robust sigma of the melanin residual. Separate
    # from SpotsConfig.threshold_mad so that tuning spot DETECTION does not
    # silently move a tracked metric.
    spot_burden_sigma: float = 2.2
    spot_contrast_percentile: float = 95.0

    # Below this pixel count a region's metrics are NaN rather than noise.
    min_region_pixels: int = 250
    # Clipped pixels are excluded from colour metrics (they carry no chroma).
    exclude_clipped: bool = True
    clipped_above: int = 252
    clipped_below: int = 3

    # ITA (deg) -> Monk Skin Tone bin 1..10, descending edges. There is no
    # official ITA->Monk mapping; this subdivides the standard ITA classes
    # (very light >55, light 41-55, intermediate 28-41, tan 10-28,
    # brown -30..10, dark <-30) into ten bins.
    #
    # !! ITA IS THE SOURCE OF TRUTH. `monk_bin` is a DERIVED DISPLAY VALUE.
    # !! Never use monk_bin for fairness or bias evaluation. Doing so audits
    # !! tone fairness against this approximation rather than against real skin
    # !! tone, which makes the result circular — the mapping under test is the
    # !! same mapping generating the labels. Fairness work needs human-assigned
    # !! Monk labels, or raw ITA bands.
    monk_ita_edges: tuple[float, ...] = (58.0, 48.0, 41.0, 34.5, 28.0, 19.0, 10.0, -10.0, -30.0)


@dataclass(frozen=True)
class SessionConfig:
    """Burst capture: frame selection, aggregation and the specular recovery.

    A burst costs the user nothing — ten frames over two seconds feels like one
    photo — and buys three things a single frame cannot give at any price:
    averaging (noise falls as 1/sqrt(n)), a choice of frames, and a view of what
    changes BETWEEN frames, which is how glare is told apart from pigment.
    """

    # -- frame selection --
    #
    # Sharpness is judged RELATIVE to the sharpest frame in the burst, never
    # against an absolute number. Laplacian variance depends on how much detail
    # the face itself has, so an absolute bar would reject every frame of a
    # smooth face and accept every frame of a stubbled one.
    sharpness_min_fraction: float = 0.6
    # Frames where the face changed apparent size are frames where the subject
    # moved. A fraction of the burst's median face width.
    face_width_tolerance: float = 0.06
    # Below this many kept frames, aggregation still runs but the session is
    # flagged: an error bar from three frames is barely an error bar.
    min_frames_for_noise: int = 4
    # More than this fraction of frames rejected -> high_frame_loss.
    max_rejected_fraction: float = 0.4

    # -- colour --
    #
    # Estimate the illuminant ONCE for the burst and apply it to every frame,
    # rather than re-estimating per frame. Per-frame estimation injects the
    # estimator's own noise into every colour metric, and the illuminant did not
    # change during a two-second capture. Measured across 12 captures the
    # shades-of-grey gains varied by ~3%, which lands directly on the pigment
    # metrics because they are sensitive to a colour cast by construction.
    shared_white_balance: bool = True

    # Temporal median of the aligned frames. Real and cheap: a per-pixel median
    # across a burst suppresses sensor noise and transient artifacts without the
    # spatial blurring a single-frame denoiser would cause. Returned for display
    # and as the substrate for future multi-frame work; metrics are NOT computed
    # on it, because denoising changes what a texture metric means.
    build_composite: bool = True

    # -- specular recovery: EXPERIMENTAL, OFF, and measured not to work --
    #
    # The idea: specular reflection is view-dependent and chromophores are not,
    # so hand tremor should move a highlight while leaving pigment in place,
    # making the between-frame variation the glare. It would have given an
    # oiliness measure with no flash/no-flash pair.
    #
    # It does not survive contact with data. On a 12-frame burst the recovered
    # map tracked EDGES, not brightness:
    #
    #   r(signal, edge gradient) = +0.551   <- registration residual
    #   r(signal, luminance)     = -0.221   <- glare would be POSITIVE
    #   r(signal, saturation)    = +0.124   <- glare would be NEGATIVE
    #   edgiest 10% of skin scored 0.0185 against 0.0096 for the brightest 10%
    #
    # It was measuring imperfect alignment at high-contrast boundaries — the
    # lash line, brows, nostril rims — and calling it shine. The T-zone/cheek
    # ratio was 1.27x with `nose`, normally the shiniest region on a face,
    # ranking 7th of 9.
    #
    # Gating on bright AND desaturated AND away from edges removes the artifact
    # (r(edge) 0.551 -> 0.134) and leaves nothing behind: the T-zone ratio FALLS
    # to 1.08x and nose still does not rank. Whatever glare is in these frames,
    # a 6%-face-width hand tremor does not move it enough to recover.
    #
    # Enable only to reproduce the above. Do not build an oiliness metric on it
    # without re-validating against a capture with a harsh point source, where
    # there is a real highlight to move. The flash/no-flash pair (capture set G)
    # remains the sound route: it separates the same two components by changing
    # the illumination rather than the viewpoint, which is a far larger lever.
    recover_specular: bool = False
    # Percentile of the per-pixel temporal distribution taken as the specular
    # excess, measured against the temporal median.
    specular_percentile: float = 90.0
    # Registration needs enough frames for a temporal median to mean anything.
    min_frames_for_specular: int = 5

    # -- aggregation --
    #
    # Standard error of a median is ~1.253x that of a mean for Gaussian data.
    # The median is used anyway: a burst's failure mode is one bad frame, which
    # is exactly what a median ignores and a mean absorbs.
    median_se_factor: float = 1.253
    # 1.96 * sqrt(2), the Bland-Altman repeatability coefficient.
    repeatability_factor: float = 2.77


@dataclass(frozen=True)
class SpotsConfig:
    """Dark spot detection on the melanin index map."""

    # Median blur kernel as a fraction of face width (forced odd, >= 3).
    # Scaling to face size keeps behaviour stable across capture distances.
    median_kernel_face_frac: float = 0.075

    # Threshold on (melanin - median_blur(melanin)) inside the skin mask.
    #
    # "mad" is the default rather than "sigma" because the residual's standard
    # deviation is itself inflated by the artifacts being rejected. A hairline
    # shadow and a patch of stubble are large, high-contrast residuals; they
    # drag the std up, which raises the threshold, which hides the genuine
    # low-contrast spots underneath. Median absolute deviation ignores those
    # outliers by construction, so the threshold reflects the skin rather than
    # the artifacts sitting on it. "sigma" is retained for comparison.
    threshold_mode: Literal["mad", "sigma", "absolute"] = "mad"
    # Multiples of a robust sigma (MAD * 1.4826), measured about the median.
    #
    # 2.2 rather than 3.0: scored against the only hand-labelled face available
    # (9 marks), 3.0 found 3 of 9 at precision 0.50 (F1 0.40) while 2.2 found
    # 7 of 9 at precision 0.78 (F1 0.78). Recall more than doubled.
    #
    # CAVEAT: one face, nine labels, and the F1 curve is a narrow peak
    # (2.5 -> 0.52, 2.2 -> 0.78, 2.0 -> 0.70), not a plateau. A narrow optimum
    # on a single image is a prime candidate for overfitting. Re-run
    # tools/evaluate.py across several labelled faces before trusting this
    # number, and expect to move it.
    threshold_mad: float = 2.2
    threshold_sigma: float = 2.2
    threshold_absolute: float = 0.02

    # Area filters, as a fraction of skin mask area, with an absolute floor.
    # The fraction alone scales sensibly on a full-resolution capture but
    # collapses on a small one: on a 768px photo it worked out to 1.9px, so
    # 3-pixel components were reaching the shape tests.
    min_area_frac: float = 2.5e-5
    min_area_px: int = 8
    max_area_frac: float = 6.0e-3

    # Shape filters: rejects hair strands (elongated) and ragged shadow edges.
    max_eccentricity: float = 0.88
    min_solidity: float = 0.55
    # Shape statistics are meaningless on a handful of pixels — a 3-pixel
    # diagonal scores eccentricity ~1.0 and is discarded as a "hair strand".
    # Measured on a real photo, eccentricity rejected 59 of 102 candidates
    # whose median area was 6px. Below this area the shape tests are skipped
    # and the component is judged on area and position alone.
    shape_min_area_px: int = 12

    # Mask-boundary rejection. The boundary is where the hairline, lash line
    # and nostril rims meet the mask, and all three produce residual that looks
    # exactly like a spot.
    mask_edge_margin_px: int = 4
    # Fraction of a component's pixels that must lie clear of that boundary.
    # Testing the centroid alone is not enough: an observed 229px false
    # positive straddling the hairline had its centroid comfortably inside the
    # mask while most of its mass was the hairline itself.
    interior_overlap_min: float = 0.9
    # -- neutral-shadow rejection (EXPERIMENTAL, OFF BY DEFAULT) --
    #
    # Physics: a shadow attenuates R, G and B by the same factor, so it leaves
    # channel ratios untouched. Melanin absorbs more strongly at shorter
    # wavelengths, so real pigment shifts them. Differencing a shading-
    # invariant chromaticity (log R - log B) against the same median
    # background therefore responds to pigment but not to shading.
    #
    # It works on the concave shadows that survive the melanin residual — the
    # inner eye corner, the nasal crease, the brow furrow. Measured on one
    # photo, the two clearest shadow false positives scored z=2.13 and z=1.68
    # while genuine marks scored z=3.88 to 4.73. Note the melanin residual
    # alone ranked the WORST false positive highest, so this is real
    # information the current pipeline does not use.
    #
    # OFF BY DEFAULT because it is not tone-neutral as implemented. On a
    # dark-skinned fixture the chroma-residual spread was 4.5x larger
    # (robust sigma 0.0385 vs 0.0085), compressing every z-score: a z>=2.5 cut
    # removed 2 of 3 false positives on one face and ALL EIGHT detections on
    # the other. Enabling this without validating across skin tones would
    # silently make spot detection worse for darker skin.
    #
    # Validate on tone-diverse real captures before turning it on.
    reject_neutral_shadows: bool = False
    shadow_chroma_z_min: float = 2.5

    # Deterministic ordering of the returned records.
    sort_by: Literal["area_desc", "raster"] = "area_desc"


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Root config. ``Config()`` is the versioned default."""

    io: IOConfig = field(default_factory=IOConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    parse: ParseConfig = field(default_factory=ParseConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    spots: SpotsConfig = field(default_factory=SpotsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------

# Fields excluded from the fingerprint, as (section, field). Only exclude
# things that cannot change output values.
_FINGERPRINT_EXCLUDE: frozenset[tuple[str, str]] = frozenset(
    {
        # Filesystem paths. Model CONTENT is hashed separately and stamped on
        # the result, so moving a file must not change the config hash.
        ("parse", "weights_path"),
        ("detect", "landmarker_asset_path"),
    }
)


def _jsonable(value: Any) -> Any:
    """Stable, type-tagged JSON form.

    Tuples and lists must not collapse to the same representation: a config
    change from ``(1, 10)`` to ``[1, 10]`` is not a value change, but a change
    from ``0.5`` to ``"0.5"`` is, and both must be distinguishable from each
    other rather than merged by loose stringification.
    """
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # repr round-trips exactly for float; str() would lose digits on 3.12-.
        return repr(value)
    return str(value)


def config_fingerprint(config: Config) -> str:
    """Stable hash of the fully-resolved config.

    Comparability of two stored results is
    ``version + config_hash + weights_hash`` — computed, not remembered. A
    tuned threshold changes this hash whether or not anyone bumped
    ``PREPROCESSING_VERSION``.

    Stable across processes and machines: no ``hash()``, no dict ordering
    dependence, no object identity.
    """
    payload = asdict(config)
    for section, key in _FINGERPRINT_EXCLUDE:
        payload.get(section, {}).pop(key, None)
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
