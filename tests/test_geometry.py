"""Tests for surface geometry and the metrics derived from the metric table.

The geometry comes from a LEARNED depth estimate, so the tests assert what that
estimate can actually support — a normal field of unit vectors, the right
ordering of which regions face the camera, an exactly recoverable light on
synthetic data — and deliberately assert nothing about fine relief, which
sparse landmark depth cannot carry.
"""

from __future__ import annotations

import numpy as np
import pytest

from skinlib.config import Config
from skinlib.derived import ASYMMETRY_PAIRS, asymmetry, periorbital_decomposition
from skinlib.geometry import (
    depth_surface,
    estimate_light_direction,
    incidence_map,
    surface_normals,
)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_landmark_depth_is_retained(analysed) -> None:
    """The signal the library used to discard."""
    _loaded, face, _skin, _regions = analysed
    assert face.landmarks_z.size == len(face.landmarks)
    assert np.isfinite(face.landmarks_z).all()
    # A face has real relief; a constant z would mean the field is dead.
    assert face.landmarks_z.std() > 1.0


def test_depth_surface_covers_the_frame_without_holes(analysed) -> None:
    loaded, face, _skin, _regions = analysed
    depth = depth_surface(face, loaded.image.shape[:2])
    assert depth.shape == loaded.image.shape[:2]
    assert np.isfinite(depth).all(), "a hole would break the gradient operator"


def test_surface_normals_are_unit_vectors(analysed) -> None:
    loaded, face, skin, _regions = analysed
    normals = surface_normals(face, loaded.image.shape[:2])
    lengths = np.linalg.norm(normals[skin], axis=-1)
    assert np.allclose(lengths, 1.0, atol=1e-4)


def test_normals_face_the_camera(analysed) -> None:
    """Sign convention: +z toward the lens, so visible skin has positive cosine."""
    loaded, face, skin, _regions = analysed
    normals = surface_normals(face, loaded.image.shape[:2])
    cosine = incidence_map(normals, np.array([0.0, 0.0, 1.0]))
    assert float(cosine[skin].mean()) > 0.5


def test_flat_regions_face_the_camera_more_than_curved_ones(analysed) -> None:
    """The forehead is a plane; the chin turns under. Measured 0.95 vs 0.63."""
    loaded, face, skin, regions = analysed
    normals = surface_normals(face, loaded.image.shape[:2])
    cosine = incidence_map(normals, np.array([0.0, 0.0, 1.0]))

    def mean_for(name: str) -> float:
        mask = regions[name] & skin
        return float(cosine[mask].mean()) if mask.sum() > 250 else float("nan")

    forehead, chin = mean_for("forehead"), mean_for("chin")
    if np.isfinite(forehead) and np.isfinite(chin):
        assert forehead > chin


def test_no_depth_yields_no_normals(analysed) -> None:
    """Detections predating landmarks_z must degrade, not crash."""
    from dataclasses import replace

    loaded, face, _skin, _regions = analysed
    stripped = replace(face, landmarks_z=np.empty(0, dtype=np.float32))
    assert surface_normals(stripped, loaded.image.shape[:2]) is None
    assert depth_surface(stripped, loaded.image.shape[:2]) is None


def test_light_direction_is_recovered_exactly_on_lambertian_data() -> None:
    """On data that matches the model, the fit must be exact.

    Establishes that a poor r-squared on a real face is the face departing from
    the model, not the solver being wrong.
    """
    rng = np.random.default_rng(0)
    normals = rng.normal(size=(64, 64, 3))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    truth = np.array([0.3, -0.5, 0.8])
    truth = truth / np.linalg.norm(truth)

    shading = np.clip(normals @ truth, 0, 1)
    grey = np.clip(shading * 200.0 + 20.0, 0, 255).astype(np.uint8)
    image = np.repeat(grey[:, :, None], 3, axis=2)
    mask = shading > 0.05  # where the linear model actually holds

    direction, r_squared = estimate_light_direction(image, normals.astype(np.float32), mask)
    assert r_squared > 0.98
    assert float(np.dot(direction, truth)) > 0.98


def test_incidence_is_clipped_at_zero() -> None:
    """Skin turned past 90 degrees is unlit, not negatively lit."""
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    normals[..., 2] = -1.0  # facing directly away
    assert float(incidence_map(normals, np.array([0.0, 0.0, 1.0])).max()) == 0.0


# ---------------------------------------------------------------------------
# derived
# ---------------------------------------------------------------------------


def test_asymmetry_is_signed_left_minus_right() -> None:
    by_region = {
        "left_cheek": {"spot_burden": 0.30},
        "right_cheek": {"spot_burden": 0.10},
    }
    out = asymmetry(by_region, metrics=("spot_burden",))
    assert out["cheek_spot_burden_asymmetry"] == pytest.approx(0.20)


def test_asymmetry_is_nan_when_a_side_is_unmeasured() -> None:
    """A difference against a missing half is not a small asymmetry."""
    by_region = {
        "left_cheek": {"spot_burden": 0.30},
        "right_cheek": {"spot_burden": float("nan")},
    }
    out = asymmetry(by_region, metrics=("spot_burden",))
    assert np.isnan(out["cheek_spot_burden_asymmetry"])


def test_only_paired_regions_appear() -> None:
    """Midline regions have no counterpart and must not be invented."""
    names = {name for name, _left, _right in ASYMMETRY_PAIRS}
    assert names == {"cheek", "periorbital"}
    out = asymmetry({}, metrics=("spot_burden",))
    assert all(key.startswith(("cheek_", "periorbital_")) for key in out)


def test_periorbital_components_reference_the_cheek_below() -> None:
    by_region = {
        "periorbital_left": {"melanin_density": 0.20, "hemoglobin_density": 0.50},
        "left_cheek": {"melanin_density": 0.15, "hemoglobin_density": 0.30},
        "periorbital_right": {},
        "right_cheek": {},
    }
    out = periorbital_decomposition(by_region)
    assert out["periorbital_left_pigment"] == pytest.approx(0.05)
    assert out["periorbital_left_vascular"] == pytest.approx(0.20)
    # Vascular dominates 0.20 to 0.05, so it should carry 4/5 of the share.
    assert out["periorbital_left_vascular_share"] == pytest.approx(0.8)
    assert out["periorbital_left_pigment_share"] == pytest.approx(0.2)


def test_a_lighter_than_cheek_component_contributes_no_share() -> None:
    """A component reading lighter is not a negative part of the darkness."""
    by_region = {
        "periorbital_left": {"melanin_density": 0.10, "hemoglobin_density": 0.50},
        "left_cheek": {"melanin_density": 0.15, "hemoglobin_density": 0.30},
    }
    out = periorbital_decomposition(by_region)
    assert out["periorbital_left_pigment"] < 0
    assert out["periorbital_left_pigment_share"] == pytest.approx(0.0)
    assert out["periorbital_left_vascular_share"] == pytest.approx(1.0)


def test_structural_term_needs_incidence() -> None:
    by_region = {
        "periorbital_left": {"melanin_density": 0.2, "hemoglobin_density": 0.5},
        "left_cheek": {"melanin_density": 0.15, "hemoglobin_density": 0.3},
    }
    assert np.isnan(periorbital_decomposition(by_region)["periorbital_left_structural"])
    with_geometry = periorbital_decomposition(
        by_region, incidence={"periorbital_left": 0.70, "left_cheek": 0.85}
    )
    # The orbit faces the camera less directly: a hollow.
    assert with_geometry["periorbital_left_structural"] == pytest.approx(0.15)
