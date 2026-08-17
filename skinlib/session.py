"""Burst analysis: many frames of one moment, aggregated into one measurement.

    from skinlib import analyze_session
    session = analyze_session(sorted(Path("burst").glob("*.JPG")))

    session.metrics["spot_burden"]             # median across kept frames
    session.detectable_change["spot_burden"]   # what a real change must exceed
    session.trusted_change("spot_burden", last_week)

A single photo yields a number with no indication of how far to trust it. Ten
frames over two seconds feel identical to the user and yield three things a
single frame cannot buy at any price:

* **averaging** — noise falls as 1/sqrt(n). Measured on real captures, a
  ten-frame burst takes ``spot_count`` from +-10.4 to +-3.3 and ``spot_burden``
  from CV 0.007 to 0.002;
* **a choice of frames** — the sharpest one, the one where the eyes are open,
  the ones where the subject had not drifted;
* **a view of what changes BETWEEN frames**, which is how glare is separated
  from pigment. Specular reflection is view-dependent and chromophores are not,
  so hand tremor moves the highlight and leaves the melanin where it was.

The error bar is measured on THIS capture rather than assumed from a reference
table, which matters because a session's noise depends on how steady the hands
were that morning.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from .color import apply_gains, correct_color
from .config import Config, config_fingerprint
from .detect import detect_face, load_image, load_landmarker
from .metrics import compute_metrics
from .parse import load_parser, parse_skin, weights_hash
from .quality import check_quality
from .regions import build_regions
from .chromophore import separate_chromophores
from .metrics import melanin_index_map
from .spots import detect_lesions, detect_spots, local_residual
from .types import METRIC_NAMES, Face, FrameReport, SessionResult
from .version import PREPROCESSING_VERSION

__all__ = ["analyze_session", "register_to", "sharpness_of"]


# --------------------------------------------------------------------------
# frame selection
# --------------------------------------------------------------------------


def sharpness_of(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian, inside the mask when one is given.

    Measured on skin rather than the whole frame: a sharp patterned background
    behind a blurred face scores well on a whole-frame measure, which is exactly
    backwards for choosing which frame to measure a face from.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    values = laplacian[mask] if mask is not None and mask.any() else laplacian.ravel()
    return float(values.var()) if values.size else float("nan")


def _select(reports: list[FrameReport], config: Config) -> list[FrameReport]:
    """Mark which frames contribute, and why the others do not.

    Both criteria are RELATIVE to the burst. Laplacian variance depends on how
    much detail a face has, so an absolute sharpness bar would reject every
    frame of a smooth face and accept every frame of a stubbled one; face width
    is compared against the burst's own median because what matters is whether
    the subject moved during this capture, not where they stood.
    """
    session = config.session
    live = [r for r in reports if r.kept]
    if not live:
        return reports

    sharpest = max(
        (r.sharpness for r in live if np.isfinite(r.sharpness)), default=float("nan")
    )
    widths = [r.face_width for r in live if np.isfinite(r.face_width)]
    median_width = float(np.median(widths)) if widths else float("nan")

    out: list[FrameReport] = []
    for report in reports:
        if not report.kept:
            out.append(report)
            continue

        rejected = ""
        if (
            np.isfinite(sharpest)
            and np.isfinite(report.sharpness)
            and sharpest > 0
            and report.sharpness < session.sharpness_min_fraction * sharpest
        ):
            rejected = "blurred"
        elif (
            np.isfinite(median_width)
            and np.isfinite(report.face_width)
            and median_width > 0
            and abs(report.face_width - median_width) / median_width
            > session.face_width_tolerance
        ):
            rejected = "moved"

        out.append(
            FrameReport(
                name=report.name,
                kept=not rejected,
                rejected=rejected,
                flags=report.flags,
                sharpness=report.sharpness,
                face_width=report.face_width,
                sclera_confidence=report.sclera_confidence,
                metrics=report.metrics,
            )
        )
    return out


# --------------------------------------------------------------------------
# registration and specular recovery
# --------------------------------------------------------------------------


def register_to(image: np.ndarray, source: Face, target: Face) -> np.ndarray:
    """Warp ``image`` so its landmarks sit where ``target``'s do.

    A partial affine (rotation, uniform scale, translation) estimated from the
    shared landmarks. Deliberately not a full affine or a homography: over the
    couple of seconds a burst spans, the face is the same shape and only the
    camera moved, so the extra degrees of freedom would fit expression and
    landmark jitter instead of camera motion — and a warp that absorbed a real
    expression change would erase the very differences being measured.
    """
    count = min(len(source.landmarks), len(target.landmarks))
    matrix, _ = cv2.estimateAffinePartial2D(
        source.landmarks[:count].astype(np.float32),
        target.landmarks[:count].astype(np.float32),
        method=cv2.LMEDS,
    )
    if matrix is None:
        return image
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _specular(
    stack: np.ndarray, skin_mask: np.ndarray, config: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal median of an aligned stack, and the upper-percentile excess above it.

    Returns ``(specular_map, composite)``.

    The composite is sound: a pigmented mark sits in the same place with the
    same darkness in every frame, so a per-pixel median across the burst
    suppresses sensor noise and transients without spatial blurring.

    The excess map was intended as glare — a highlight should slide across the
    skin as the camera moves while pigment stays put. **Measured, it does not
    work**: it tracks edge gradient (r = +0.551) rather than luminance
    (r = -0.221, the wrong sign), because imperfect alignment at the lash line
    and nostril rims varies far more between frames than any highlight does.
    ``SessionConfig.recover_specular`` is False accordingly, and the full
    numbers are recorded there.
    """
    composite = np.median(stack, axis=0)
    upper = np.percentile(stack, config.session.specular_percentile, axis=0)

    # Luminance only. A highlight takes the illuminant's colour, so its
    # signature is brightness that appears and vanishes, not a hue change.
    excess = upper.mean(axis=2) - composite.mean(axis=2)

    specular_map = np.full(excess.shape, np.nan, dtype=np.float32)
    specular_map[skin_mask] = (excess[skin_mask] / 255.0).astype(np.float32)
    return specular_map, composite.astype(np.uint8)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def _robust_sigma(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)) * 1.4826)


def _aggregate(
    per_frame: list[dict[str, float]], config: Config
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Median, single-frame noise, standard error of the median, and the bar.

    NaN is dropped per metric rather than per frame: a region can be too small
    to measure in one frame and fine in the next, and discarding the whole frame
    over one NaN column would throw away good measurements of everything else.
    """
    session = config.session
    metrics: dict[str, float] = {}
    noise: dict[str, float] = {}
    error: dict[str, float] = {}
    bar: dict[str, float] = {}

    for name in METRIC_NAMES:
        raw = np.array([frame.get(name, np.nan) for frame in per_frame], dtype=np.float64)
        values = raw[np.isfinite(raw)]
        if values.size == 0:
            metrics[name] = noise[name] = error[name] = bar[name] = float("nan")
            continue

        metrics[name] = float(np.median(values))
        sigma = _robust_sigma(values)
        noise[name] = sigma
        if np.isfinite(sigma):
            standard_error = session.median_se_factor * sigma / np.sqrt(values.size)
            error[name] = float(standard_error)
            bar[name] = float(session.repeatability_factor * standard_error)
        else:
            error[name] = bar[name] = float("nan")
    return metrics, noise, error, bar


def _aggregate_regions(
    per_frame: list[dict[str, dict[str, float]]], config: Config
) -> dict[str, dict[str, float]]:
    names = sorted({region for frame in per_frame for region in frame})
    return {
        region: _aggregate([frame.get(region, {}) for frame in per_frame], config)[0]
        for region in names
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def analyze_session(
    sources: list[str | Path | np.ndarray],
    config: Config | None = None,
    parser=None,
    landmarker=None,
) -> SessionResult:
    """Analyse a burst of frames captured as one moment.

    The frames must be the same face, seconds apart, with the skin unchanged.
    That is what licenses treating the spread between them as measurement noise;
    pass captures from different days and the "noise" will be real change.

    Returns medians, the noise that produced them, and the smallest difference
    another session would have to show before it means anything.
    """
    config = config or Config()
    session_config = config.session
    model = parser if parser is not None else load_parser(config)
    # One landmarker for the whole burst. Building one costs ~182ms against
    # ~24ms to detect, so a ten-frame burst that rebuilt per frame spent ~1.8s
    # on construction alone. Closed in the finally below when we built it;
    # a caller-supplied one is the caller's to close.
    with ExitStack() as stack:
        if landmarker is None:
            landmarker = stack.enter_context(load_landmarker(config))
        return _analyze_session(sources, config, model, landmarker)


def _analyze_session(sources, config: Config, model, landmarker) -> SessionResult:
    """The body of ``analyze_session``, with both handles guaranteed live."""
    session_config = config.session

    # Keyed on the REPORT INDEX, never on the frame name. A name is
    # `Path(source).name`, or the literal "<array>" for an ndarray, so names are
    # not unique: every ndarray in a burst shares one, and `a/IMG_0001.jpg`
    # collides with `b/IMG_0001.jpg`. Joining the selection verdict back to
    # pixel data by name therefore admitted frames that `_select` had just
    # rejected, and they went on to contaminate `noise`, `standard_error` and
    # `detectable_change` — defeating the one guarantee the burst path exists
    # to provide. Names are for humans reading a FrameReport.
    loaded_frames: list[tuple[int, np.ndarray, Face, np.ndarray, dict]] = []
    reports: list[FrameReport] = []

    for source in sources:
        name = Path(source).name if isinstance(source, (str, Path)) else "<array>"
        loaded = load_image(source, config.io)
        face = detect_face(loaded.image, config, landmarker=landmarker)
        if face is None:
            reports.append(FrameReport(name=name, kept=False, rejected="no_face"))
            continue

        skin = parse_skin(loaded.image, face, config, parser=model)
        regions = build_regions(face, skin, config)
        quality = check_quality(loaded, face, skin, regions, config)
        if not quality.usable:
            reports.append(
                FrameReport(
                    name=name,
                    kept=False,
                    rejected="unusable",
                    flags=tuple(quality.flags),
                    face_width=float(face.width),
                )
            )
            continue

        reports.append(
            FrameReport(
                name=name,
                kept=True,
                flags=tuple(quality.flags),
                sharpness=sharpness_of(loaded.image, skin),
                face_width=float(face.width),
            )
        )
        loaded_frames.append((len(reports) - 1, loaded.image, face, skin, regions))

    reports = _select(reports, config)
    kept = [frame for frame in loaded_frames if reports[frame[0]].kept]

    fingerprints = dict(
        version=PREPROCESSING_VERSION,
        config_hash=config_fingerprint(config),
        weights_hash=weights_hash(config.parse),
        landmarker_hash=_landmarker_hash(config),
    )

    if not kept:
        return SessionResult(
            metrics={name: float("nan") for name in METRIC_NAMES},
            by_region={},
            frames=reports,
            flags=["no_usable_frames"],
            **fingerprints,
        )

    # -- one illuminant for the whole burst ------------------------------
    #
    # Re-estimating per frame would inject the estimator's own noise into every
    # colour metric, and the light did not change during a two-second capture.
    # The frame with the most confident sclera sets the gains for all of them,
    # because sclera is a real neutral surface where shades-of-grey is only an
    # assumption about the scene.
    corrections = [correct_color(image, face, skin, config) for _, image, face, skin, _ in kept]
    for report_index, correction in enumerate(corrections):
        index = kept[report_index][0]
        reports[index] = FrameReport(
            name=reports[index].name,
            kept=reports[index].kept,
            rejected=reports[index].rejected,
            flags=reports[index].flags,
            sharpness=reports[index].sharpness,
            face_width=reports[index].face_width,
            sclera_confidence=(
                float(correction.sclera_confidence)
                if correction.sclera_confidence is not None
                else float("nan")
            ),
            metrics=reports[index].metrics,
        )

    if session_config.shared_white_balance:
        best = max(
            range(len(corrections)),
            key=lambda i: (
                corrections[i].sclera_confidence
                if corrections[i].sclera_confidence is not None
                else -1.0
            ),
        )
        gains = corrections[best].gains
        corrected = [apply_gains(image, gains) for _, image, _, _, _ in kept]
    else:
        corrected = [correction.image for correction in corrections]

    # -- register, then build the composite ------------------------------
    #
    # The specular map is off by default and measured not to work: it recovers
    # registration residual at high-contrast edges rather than glare. See
    # SessionConfig for the numbers. The composite is sound and kept.
    specular_map = composite = None
    wants_stack = session_config.build_composite or session_config.recover_specular
    if wants_stack and len(kept) >= session_config.min_frames_for_specular:
        reference_face = kept[0][2]
        aligned = [corrected[0]] + [
            register_to(corrected[i], kept[i][2], reference_face)
            for i in range(1, len(kept))
        ]
        stack = np.stack(aligned).astype(np.float64)
        recovered, composite = _specular(stack, kept[0][3], config)
        if session_config.recover_specular:
            specular_map = recovered

    # -- per-frame metrics, then aggregate -------------------------------
    per_frame: list[dict[str, float]] = []
    per_frame_regions: list[dict[str, dict[str, float]]] = []
    # Detection records from the reference frame, for display. The counts carry
    # a 1/sqrt(N) noise floor and are not the tracking signal — spot_burden and
    # inflammation_burden are — but a user still wants to see WHERE their marks
    # are, and that needs one frame's records rather than an aggregate.
    lesion_counts: list[int] = []
    first_frame_lesions: list = []
    first_frame_spots: list = []
    for index, (report_index, _image, face, skin, regions) in enumerate(kept):
        image = corrected[index]
        # Same map sharing as `analyze`: the chromophore separation and the two
        # large-kernel medians dominate per-frame cost, and every consumer below
        # wants the same maps. This matters more here than anywhere else, since
        # a session pays it once per frame.
        melanin_map = melanin_index_map(image, config.metrics)
        chromophores = separate_chromophores(image, config.metrics)
        melanin_res = local_residual(melanin_map, skin, face, config)
        hemoglobin_res = local_residual(chromophores[1], skin, face, config)

        spots = detect_spots(
            image, skin, face, regions, config,
            melanin=melanin_map, residual=melanin_res,
        )
        lesions = detect_lesions(
            image, skin, face, regions, config,
            hemoglobin=chromophores[1], residual=hemoglobin_res,
        )
        lesion_counts.append(len(lesions))
        if index == 0:
            first_frame_lesions = lesions
            first_frame_spots = spots
        result = compute_metrics(
            image, skin, regions, config, spots=spots, face=face,
            melanin_residual_map=melanin_res,
            hemoglobin_residual_map=hemoglobin_res,
            chromophores=chromophores,
        )
        per_frame.append(result.global_)
        per_frame_regions.append(result.by_region)

        position = report_index
        reports[position] = FrameReport(
            name=reports[position].name,
            kept=True,
            rejected="",
            flags=reports[position].flags,
            sharpness=reports[position].sharpness,
            face_width=reports[position].face_width,
            sclera_confidence=reports[position].sclera_confidence,
            metrics=dict(result.global_),
        )

    metrics, noise, error, bar = _aggregate(per_frame, config)

    return SessionResult(
        metrics=metrics,
        by_region=_aggregate_regions(per_frame_regions, config),
        noise=noise,
        standard_error=error,
        detectable_change=bar,
        frames=reports,
        flags=_session_flags(reports, kept, config),
        specular_map=specular_map,
        composite=composite,
        spots=first_frame_spots,
        lesions=first_frame_lesions,
        **fingerprints,
    )


def _session_flags(
    reports: list[FrameReport], kept: list, config: Config
) -> list[str]:
    """Warnings about the capture rather than the skin.

    Deliberately not folded into the metrics. A session whose error bar is wide
    because three frames survived is different from one that is wide because the
    skin is genuinely variable, and only the flags can tell them apart.
    """
    session = config.session
    flags: list[str] = []

    if len(kept) < session.min_frames_for_noise:
        # An error bar from a handful of frames is barely an error bar.
        flags.append("few_frames")
    if len(kept) == 1:
        flags.append("single_frame")

    rejected = sum(1 for r in reports if not r.kept)
    if reports and rejected / len(reports) > session.max_rejected_fraction:
        flags.append("high_frame_loss")

    widths = [r.face_width for r in reports if r.kept and np.isfinite(r.face_width)]
    if len(widths) >= 2:
        spread = (max(widths) - min(widths)) / max(float(np.median(widths)), 1e-9)
        if spread > session.face_width_tolerance:
            flags.append("subject_moved")

    return flags


def _landmarker_hash(config: Config) -> str:
    from .detect import file_hash, resolve_landmarker_asset

    return file_hash(resolve_landmarker_asset(config.detect))
