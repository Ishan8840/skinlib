"""Tests for burst analysis.

The session API's whole claim is that N frames beat one, so the tests assert the
arithmetic that claim rests on: the error bar must shrink as 1/sqrt(n), bad
frames must be excluded rather than averaged in, and a change smaller than the
session's own noise must not be reported as a change.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from skinlib.config import Config
from skinlib.session import _aggregate, _select, analyze_session, register_to, sharpness_of
from skinlib.types import METRIC_NAMES, FrameReport, SessionResult


# ---------------------------------------------------------------------------
# frame selection
# ---------------------------------------------------------------------------


def test_sharpness_falls_when_an_image_is_blurred() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
    assert sharpness_of(blurred) < sharpness_of(sharp)


def test_sharpness_is_measured_inside_the_mask() -> None:
    """A detailed background must not vouch for a smooth face."""
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(1)
    image[:32] = rng.integers(0, 255, size=(32, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[32:] = True  # the flat half
    assert sharpness_of(image, mask) < sharpness_of(image)


def _frame(name: str, sharpness: float, width: float) -> FrameReport:
    return FrameReport(name=name, kept=True, sharpness=sharpness, face_width=width)


def test_blurred_frames_are_rejected_relative_to_the_burst() -> None:
    """Relative, not absolute: Laplacian variance depends on the face."""
    config = Config()
    reports = _select(
        [_frame("a", 100.0, 500.0), _frame("b", 95.0, 500.0), _frame("c", 20.0, 500.0)],
        config,
    )
    assert [r.kept for r in reports] == [True, True, False]
    assert reports[2].rejected == "blurred"


def test_frames_where_the_subject_moved_are_rejected() -> None:
    config = Config()
    reports = _select(
        [_frame("a", 100.0, 500.0), _frame("b", 100.0, 505.0), _frame("c", 100.0, 700.0)],
        config,
    )
    assert reports[2].rejected == "moved"


def test_selection_scales_with_the_burst_not_an_absolute_bar() -> None:
    """A uniformly soft burst must not reject everything."""
    config = Config()
    reports = _select(
        [_frame("a", 12.0, 500.0), _frame("b", 11.0, 500.0), _frame("c", 10.0, 500.0)],
        config,
    )
    assert all(r.kept for r in reports)


# ---------------------------------------------------------------------------
# aggregation arithmetic
# ---------------------------------------------------------------------------


def test_error_bar_shrinks_as_root_n() -> None:
    """The entire justification for capturing a burst."""
    config = Config()
    rng = np.random.default_rng(2)

    def bar(n: int) -> float:
        frames = [
            {name: float(v) for name in METRIC_NAMES}
            for v in rng.normal(1.0, 0.1, size=n)
        ]
        return _aggregate(frames, config)[3]["roughness"]

    small = np.median([bar(4) for _ in range(40)])
    large = np.median([bar(36) for _ in range(40)])
    # 9x the frames should be ~3x tighter. Generous bounds: this is a median of
    # a robust sigma of a random draw, so it is noisy in its own right.
    assert 2.0 < small / large < 4.5


def test_aggregate_uses_the_median_not_the_mean() -> None:
    """One wild frame must not drag the session estimate."""
    config = Config()
    frames = [{"roughness": v} for v in (1.0, 1.0, 1.0, 1.0, 99.0)]
    metrics, _noise, _error, _bar = _aggregate(frames, config)
    assert metrics["roughness"] == pytest.approx(1.0)


def test_nan_is_dropped_per_metric_not_per_frame() -> None:
    """A region too small in one frame must not discard that frame's others."""
    config = Config()
    frames = [
        {"roughness": 1.0, "spot_burden": float("nan")},
        {"roughness": 2.0, "spot_burden": 0.5},
        {"roughness": 3.0, "spot_burden": 0.5},
    ]
    metrics, _noise, _error, _bar = _aggregate(frames, config)
    assert metrics["roughness"] == pytest.approx(2.0)
    assert metrics["spot_burden"] == pytest.approx(0.5)


def test_a_single_frame_has_no_error_bar() -> None:
    """Honest: one measurement cannot estimate its own spread."""
    config = Config()
    metrics, noise, _error, bar = _aggregate([{"roughness": 1.0}], config)
    assert metrics["roughness"] == pytest.approx(1.0)
    assert np.isnan(noise["roughness"])
    assert np.isnan(bar["roughness"])


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_register_to_undoes_a_known_translation(analysed) -> None:
    loaded, face, _skin, _regions = analysed
    shift = np.float32([[1, 0, 12], [0, 1, -7]])
    moved = cv2.warpAffine(
        loaded.image, shift, (loaded.image.shape[1], loaded.image.shape[0])
    )
    moved_face = replace(face, landmarks=face.landmarks + np.array([12.0, -7.0]))
    restored = register_to(moved, moved_face, face)

    centre = (slice(200, -200), slice(200, -200))
    before = float(np.abs(moved[centre].astype(float) - loaded.image[centre].astype(float)).mean())
    after = float(
        np.abs(restored[centre].astype(float) - loaded.image[centre].astype(float)).mean()
    )
    assert after < before / 3


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def burst(full_config: Config, parser):
    """The four fixture portraits are DIFFERENT faces, not a burst.

    Used only to exercise plumbing. A real burst is one face seconds apart, and
    nothing here asserts anything about the aggregated values, which would be
    meaningless across four people.
    """
    from tests.conftest import FIXTURE_DIR, PORTRAITS

    return [FIXTURE_DIR / name for name in PORTRAITS]


def test_analyze_session_reports_every_frame(burst, full_config: Config, parser) -> None:
    session = analyze_session(burst, config=full_config, parser=parser)
    assert isinstance(session, SessionResult)
    assert len(session.frames) == len(burst)
    # Rejections carry a reason; kept frames do not.
    for frame in session.frames:
        assert bool(frame.rejected) != frame.kept


def test_analyze_session_on_one_frame_flags_itself(burst, full_config: Config, parser) -> None:
    session = analyze_session(burst[:1], config=full_config, parser=parser)
    if session.n_kept:
        assert "single_frame" in session.flags
        assert np.isnan(session.detectable_change["roughness"])


def test_analyze_session_carries_the_comparability_key(
    burst, full_config: Config, parser
) -> None:
    session = analyze_session(burst, config=full_config, parser=parser)
    version, config_hash, weights, landmarker = session.comparable_key
    assert version and config_hash and weights and landmarker


def test_trusted_change_refuses_a_difference_inside_the_noise() -> None:
    """The question a bare float cannot answer."""
    base = SessionResult(
        metrics={"spot_burden": 0.100},
        by_region={},
        detectable_change={"spot_burden": 0.010},
    )
    tiny = SessionResult(
        metrics={"spot_burden": 0.103},
        by_region={},
        detectable_change={"spot_burden": 0.010},
    )
    real = SessionResult(
        metrics={"spot_burden": 0.130},
        by_region={},
        detectable_change={"spot_burden": 0.010},
    )
    assert not base.trusted_change("spot_burden", tiny)
    assert base.trusted_change("spot_burden", real)


def test_trusted_change_uses_the_blunter_of_the_two_sessions() -> None:
    sharp = SessionResult(
        metrics={"spot_burden": 0.100}, by_region={}, detectable_change={"spot_burden": 0.001}
    )
    blunt = SessionResult(
        metrics={"spot_burden": 0.115}, by_region={}, detectable_change={"spot_burden": 0.050}
    )
    # 0.015 clears the sharp session's bar but not the blunt one's.
    assert not sharp.trusted_change("spot_burden", blunt)


def test_rejected_frames_do_not_reach_the_metrics(analysed, full_config, parser) -> None:
    """Selection must actually exclude, even when every frame shares a name.

    Frames were rejoined to their pixel data BY NAME. An ndarray source is named
    "<array>", so a whole burst shared one name and the set-membership test
    admitted frames selection had just rejected — which then contaminated
    `noise`, `standard_error` and `detectable_change`, the very numbers the
    burst path exists to produce. `a/IMG_0001.jpg` and `b/IMG_0001.jpg` collided
    identically.

    Four still frames plus one zoomed frame that `_select` rejects as "moved".
    """
    loaded, _face, _skin, _regions = analysed
    image = loaded.image
    height, width = image.shape[:2]

    # A centre crop resized back: same face, visibly larger, so face_width moves
    # well past `face_width_tolerance`.
    margin_y, margin_x = height // 6, width // 6
    zoomed = cv2.resize(
        image[margin_y : height - margin_y, margin_x : width - margin_x],
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    session = analyze_session(
        [image, image, image, image, zoomed], config=full_config, parser=parser
    )

    rejected = [r for r in session.frames if not r.kept]
    assert rejected, "the zoomed frame should have been rejected"
    # Every frame is named "<array>"; the join must not rely on that.
    assert len({r.name for r in session.frames}) == 1

    # The count that actually matters: only kept frames may contribute.
    contributing = sum(1 for r in session.frames if r.kept and r.metrics)
    assert contributing == session.n_kept
    assert contributing == len(session.frames) - len(rejected)


def test_per_frame_metrics_land_on_the_right_report(analysed, full_config, parser) -> None:
    """The name lookup always resolved to index 0, so all metrics piled onto it."""
    loaded, _face, _skin, _regions = analysed
    session = analyze_session(
        [loaded.image, loaded.image, loaded.image], config=full_config, parser=parser
    )
    populated = [r for r in session.frames if r.kept and r.metrics]
    assert len(populated) == session.n_kept, (
        "each kept frame must carry its own metrics, not just the first"
    )


def test_specular_recovery_is_off_by_default() -> None:
    """It measured registration residual, not glare. See SessionConfig."""
    assert Config().session.recover_specular is False
