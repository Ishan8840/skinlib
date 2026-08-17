"""Per-metric noise floor from a set of same-session captures.

    python tools/repeatability.py /path/to/captures [-o report.json]

Point this at photos of one face taken in one sitting, where the skin genuinely
did not change. Everything the library then reports as variation is measurement
noise, and that number is what makes every other number interpretable: without
it, a tracker cannot tell a treatment working from the subject standing closer.

Two columns carry the weight.

``noise`` is a robust spread (MAD * 1.4826) rather than a standard deviation.
With a dozen captures a single bad frame sets a standard deviation, which would
report the outlier rather than the floor.

``r(face_w)`` is the correlation between the metric and apparent face width. A
metric that measures skin should be near zero here. A metric that measures how
close the subject stood will not be, and no amount of averaging fixes that —
it is bias, not noise. This column is the one that catches a scale bug.

The detectable-change guide is ``2.77 * noise``: the repeatability coefficient,
the difference two single measurements must exceed before it is distinguishable
from noise at 95%. Quote it, or quote nothing.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skinlib import Config, analyze, load_parser
from skinlib.types import DISPLAY_SAFE_METRICS, INTERNAL_ONLY_METRICS, METRIC_NAMES

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic")


@dataclass(frozen=True)
class Capture:
    """One analysed photo, reduced to what the report needs."""

    name: str
    usable: bool
    flags: tuple[str, ...]
    unreliable: frozenset[str]
    face_width: float
    metrics: dict[str, float]


def _robust_spread(values: np.ndarray) -> float:
    """MAD * 1.4826 — a standard deviation's worth of scale, outlier-resistant.

    Matches the estimator the spot threshold already uses, so "noise" means the
    same thing in both places.
    """
    if values.size < 2:
        return float("nan")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)) * 1.4826)


def _correlation(values: np.ndarray, other: np.ndarray) -> float:
    """Pearson r, or NaN when either side is constant.

    A constant input makes r undefined rather than zero, and reporting zero
    there would read as "no distance dependence" on evidence that shows
    nothing at all.
    """
    if values.size < 3:
        return float("nan")
    if float(np.std(values)) < 1e-12 or float(np.std(other)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(values, other)[0, 1])


def collect(paths: list[Path], config: Config) -> list[Capture]:
    parser = load_parser(config)
    captures: list[Capture] = []
    for path in paths:
        result = analyze(path, config=config, parser=parser)
        captures.append(
            Capture(
                name=path.name,
                usable=result.quality.usable,
                flags=tuple(result.quality.flags),
                unreliable=result.quality.unreliable_metrics,
                face_width=float(result.face.width) if result.face is not None else float("nan"),
                metrics=dict(result.metrics.global_) if result.metrics is not None else {},
            )
        )
    return captures


def summarise(captures: list[Capture], metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Per-metric centre, noise floor, detectable change and distance coupling.

    Only usable captures contribute. An unusable capture is one the library has
    already declined to measure, and folding it in would let a photo the gate
    rejected set the noise floor for every photo it accepts.
    """
    usable = [capture for capture in captures if capture.usable and capture.metrics]
    widths = np.array([capture.face_width for capture in usable], dtype=np.float64)

    report: dict[str, dict[str, float]] = {}
    for name in metrics:
        raw = np.array(
            [capture.metrics.get(name, float("nan")) for capture in usable], dtype=np.float64
        )
        finite = np.isfinite(raw)
        values = raw[finite]
        if values.size < 2:
            report[name] = {"n": float(values.size)}
            continue

        noise = _robust_spread(values)
        median = float(np.median(values))
        report[name] = {
            "n": float(values.size),
            "median": median,
            "noise": noise,
            # Bland-Altman repeatability coefficient: 1.96 * sqrt(2) * sigma.
            "detectable_change": 2.77 * noise,
            # Noise as a fraction of the level. Meaningless for a metric whose
            # zero is arbitrary (the _rel family straddles zero), so it is
            # reported but must not be compared across families.
            "cv": abs(noise / median) if abs(median) > 1e-12 else float("nan"),
            "min": float(values.min()),
            "max": float(values.max()),
            "r_face_width": _correlation(values, widths[finite]),
        }
    return report


def _format(report: dict[str, dict[str, float]], captures: list[Capture]) -> str:
    lines: list[str] = []
    usable = [capture for capture in captures if capture.usable]
    widths = [capture.face_width for capture in usable]

    lines.append(f"captures: {len(captures)}   usable: {len(usable)}")
    if widths:
        lines.append(
            f"face width: {min(widths):.0f}-{max(widths):.0f}px "
            f"({max(widths) / max(min(widths), 1e-9):.2f}x linear range)"
        )

    counts: dict[str, int] = {}
    for capture in captures:
        for flag in capture.flags:
            counts[flag] = counts.get(flag, 0) + 1
    if counts:
        lines.append("flags: " + ", ".join(
            f"{flag} {count}/{len(captures)}" for flag, count in sorted(counts.items())
        ))

    lines.append("")
    header = f"{'metric':22s} {'n':>3s} {'median':>10s} {'noise':>10s} {'detect':>10s} {'r(face_w)':>10s}"
    lines.append(header)
    lines.append("-" * len(header))

    for name in METRIC_NAMES:
        row = report.get(name, {})
        if row.get("n", 0) < 2:
            lines.append(f"{name:22s} {int(row.get('n', 0)):3d}   (not measured)")
            continue
        marker = " *" if name in INTERNAL_ONLY_METRICS else ""
        correlation = row["r_face_width"]
        # Flag a metric still coupled to capture distance; that is bias, and
        # averaging more captures will not remove it.
        warn = " <-- distance-coupled" if abs(correlation) > 0.6 else ""
        lines.append(
            f"{name + marker:22s} {int(row['n']):3d} {row['median']:10.4f} "
            f"{row['noise']:10.4f} {row['detectable_change']:10.4f} "
            f"{correlation:10.2f}{warn}"
        )

    lines.append("")
    lines.append("* internal-only: measures the capture as much as the skin, do not track.")
    lines.append("noise = MAD*1.4826 across captures. detect = 2.77*noise, the smallest")
    lines.append("difference between two single captures that is not just noise (95%).")
    lines.append("r(face_w) near zero is what a skin metric should show.")
    return "\n".join(lines)


def _budget(reports: dict[str, dict[str, dict[str, float]]], baseline: str) -> str:
    """Each set's noise as a multiple of the baseline set's.

    The ratio is the whole point of shooting more than one set. A metric whose
    noise under varied pose is 1.2x its noise under fixed pose is robust to
    pose; one at 8x is not, and the number says how much there is to win before
    any work is done. The baseline is the instrument floor, so a ratio near 1.0
    means that condition costs nothing and no amount of effort will improve it.
    """
    others = [name for name in reports if name != baseline]
    lines: list[str] = []
    header = f"{'metric':22s} {baseline + ' noise':>14s}" + "".join(
        f"{name:>12s}" for name in others
    )
    lines.append(header)
    lines.append("-" * len(header))

    for metric in METRIC_NAMES:
        base = reports[baseline].get(metric, {})
        floor = base.get("noise", float("nan"))
        if not math.isfinite(floor) or floor <= 0:
            continue
        marker = " *" if metric in INTERNAL_ONLY_METRICS else ""
        row = f"{metric + marker:22s} {floor:14.5f}"
        for name in others:
            noise = reports[name].get(metric, {}).get("noise", float("nan"))
            row += f"{noise / floor:11.1f}x" if math.isfinite(noise) else f"{'-':>12s}"
        lines.append(row)

    lines.append("")
    lines.append(f"Columns are each set's noise divided by {baseline}'s.")
    lines.append("1.0x = that condition costs nothing. Large = that is where the error lives.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "directories",
        type=Path,
        nargs="+",
        help="one or more directories of same-session captures; the FIRST is the "
        "baseline every other set is reported as a multiple of",
    )
    parser.add_argument("-o", "--output", type=Path, help="write the full report as JSON")
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="include internal-only metrics in the JSON (they are always shown in the table)",
    )
    parser.add_argument(
        "--session",
        action="store_true",
        help="also aggregate each directory as a burst and print what a session "
        "estimate would resolve, versus a single frame",
    )
    args = parser.parse_args()

    config = Config()
    names = METRIC_NAMES if args.all_metrics else DISPLAY_SAFE_METRICS
    reports: dict[str, dict[str, dict[str, float]]] = {}
    everything: dict[str, list[Capture]] = {}

    for directory in args.directories:
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise SystemExit(f"no images found in {directory}")

        label = directory.name
        captures = collect(paths, config)
        report = summarise(captures, METRIC_NAMES)
        reports[label] = report
        everything[label] = captures

        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
        print(_format(report, captures))

        if args.session:
            from skinlib import analyze_session

            session = analyze_session(paths, config=config)
            print(
                f"\nas a burst: {session.n_kept}/{len(paths)} frames kept"
                + (f", flags {session.flags}" if session.flags else "")
            )
            header = (
                f"  {'metric':22s} {'session':>10s} {'1-frame bar':>13s} {'session bar':>13s} {'gain':>7s}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for name in DISPLAY_SAFE_METRICS:
                single = report.get(name, {}).get("detectable_change", float("nan"))
                pooled = session.detectable_change.get(name, float("nan"))
                if not (math.isfinite(single) and math.isfinite(pooled) and pooled > 0):
                    continue
                print(
                    f"  {name:22s} {session.metrics[name]:10.4f} {single:13.5f} "
                    f"{pooled:13.5f} {single / pooled:6.1f}x"
                )

    if len(reports) > 1:
        baseline = args.directories[0].name
        print(f"\n{'=' * 72}\nERROR BUDGET (baseline: {baseline})\n{'=' * 72}")
        print(_budget(reports, baseline))

    if args.output:
        payload = {
            label: {
                "captures": [
                    {
                        "name": capture.name,
                        "usable": capture.usable,
                        "flags": list(capture.flags),
                        "face_width": capture.face_width,
                        "metrics": capture.metrics,
                    }
                    for capture in everything[label]
                ],
                "noise_floor": {n: reports[label][n] for n in names if n in reports[label]},
            }
            for label in reports
        }
        args.output.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
