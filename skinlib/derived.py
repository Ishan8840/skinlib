"""Quantities derived from metrics already computed, not from pixels.

Nothing here touches an image. These are rearrangements of ``MetricsResult``
that answer questions the per-region table holds but does not state, and they
are separated from ``metrics.py`` for exactly that reason: a derived value
inherits its inputs' noise and cannot be more trustworthy than they are, which
is easier to remember when it lives somewhere else.

Two families:

* **asymmetry** — left versus right, which matters because a face is nearly
  symmetric and a real one-sided finding is therefore conspicuous, while sun
  exposure, sleeping side and habitual phone side all produce genuine
  lateralisation worth tracking;
* **periorbital decomposition** — under-eye darkness split into the pigment,
  the vasculature and the shadow that produce it. These have different causes
  and different remedies, and no single-channel measurement can tell them
  apart. This library separates melanin from haemoglobin analytically and
  measures surface geometry, so it can.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ASYMMETRY_PAIRS", "asymmetry", "periorbital_decomposition"]

# Anatomically paired regions, subject's left first. Unpaired regions
# (forehead, glabella, nose, perioral, chin) straddle the midline and have no
# counterpart, so they are absent by construction rather than by omission.
ASYMMETRY_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("cheek", "left_cheek", "right_cheek"),
    ("periorbital", "periorbital_left", "periorbital_right"),
)


def asymmetry(
    by_region: dict[str, dict[str, float]], metrics: tuple[str, ...] | None = None
) -> dict[str, float]:
    """Signed left-minus-right difference for each paired region and metric.

    Keys are ``{pair}_{metric}_asymmetry``; positive means the subject's LEFT
    reads higher. Signed, not absolute: which side is worse is the clinically
    interesting part, and an absolute difference also cannot average to zero
    across sessions, so noise would accumulate into a fake finding.

    NaN when either side is unmeasured. A difference against a missing half is
    not a small asymmetry, it is no measurement.

    **This inherits both regions' noise**, so its noise floor is roughly sqrt(2)
    times a single region's. Compare against that, not against zero.
    """
    if metrics is None:
        metrics = ("spot_burden", "inflammation_burden", "roughness", "melanin_density")

    out: dict[str, float] = {}
    for name, left_region, right_region in ASYMMETRY_PAIRS:
        left = by_region.get(left_region, {})
        right = by_region.get(right_region, {})
        for metric in metrics:
            first = left.get(metric, float("nan"))
            second = right.get(metric, float("nan"))
            out[f"{name}_{metric}_asymmetry"] = (
                float(first - second)
                if np.isfinite(first) and np.isfinite(second)
                else float("nan")
            )
    return out


def periorbital_decomposition(
    by_region: dict[str, dict[str, float]],
    face_reference: dict[str, float] | None = None,
    incidence: dict[str, float] | None = None,
) -> dict[str, float]:
    """Split under-eye darkness into pigment, vascular and structural parts.

    Dark circles are three different conditions wearing the same appearance:

    * **pigmented** — melanin, from post-inflammatory change or genetics;
    * **vascular** — haemoglobin in thin periorbital skin, the classic blue-grey;
    * **structural** — a tear-trough hollow casting a shadow, which is geometry
      and not skin at all.

    They respond to different things, so a single "darkness" number is close to
    useless. Each component here is the periorbital region's value MINUS the
    cheek beneath it, since the cheek is the same person's skin under the same
    light and is the only fair reference available.

    Returned per side, plus the shares each component contributes. **The shares
    are descriptive, not diagnostic**: they say which signal dominates, not what
    is causing it, and the structural term is inferred from coarse landmark
    geometry that cannot resolve a tear trough directly. Validate before showing
    any of this to a person.
    """
    out: dict[str, float] = {}

    for side, orbit, cheek in (
        ("left", "periorbital_left", "left_cheek"),
        ("right", "periorbital_right", "right_cheek"),
    ):
        eye = by_region.get(orbit, {})
        reference = by_region.get(cheek, {})

        pigment = _difference(eye, reference, "melanin_density")
        vascular = _difference(eye, reference, "hemoglobin_density")

        # Structural: how much less directly the orbit faces the camera than the
        # cheek. A hollow turns away from the lens and away from the light, so a
        # deficit here is a shadow rather than a pigment.
        structural = float("nan")
        if incidence is not None:
            orbit_cos = incidence.get(orbit, float("nan"))
            cheek_cos = incidence.get(cheek, float("nan"))
            if np.isfinite(orbit_cos) and np.isfinite(cheek_cos):
                structural = float(cheek_cos - orbit_cos)

        out[f"periorbital_{side}_pigment"] = pigment
        out[f"periorbital_{side}_vascular"] = vascular
        out[f"periorbital_{side}_structural"] = structural

        # Shares over whichever components were measured. Only the positive part
        # of each contributes: a component that reads LIGHTER than the cheek is
        # not a negative share of the darkness, it is not part of it.
        parts = {
            "pigment": max(pigment, 0.0) if np.isfinite(pigment) else float("nan"),
            "vascular": max(vascular, 0.0) if np.isfinite(vascular) else float("nan"),
        }
        total = sum(v for v in parts.values() if np.isfinite(v))
        for label, value in parts.items():
            out[f"periorbital_{side}_{label}_share"] = (
                float(value / total) if np.isfinite(value) and total > 1e-9 else float("nan")
            )

    return out


def _difference(region: dict[str, float], reference: dict[str, float], metric: str) -> float:
    first = region.get(metric, float("nan"))
    second = reference.get(metric, float("nan"))
    if not (np.isfinite(first) and np.isfinite(second)):
        return float("nan")
    return float(first - second)
