"""Score skinlib's metrics against a folder-per-class labelled dataset.

    python tools/benchmark.py /path/to/dataset --per-class 100 -o bench.json
    python tools/benchmark.py /path/to/dataset --compare bench.json

Every threshold in this library was set on a handful of images of one face. That
is enough to catch a default that is wrong by a factor of two, and nowhere near
enough to know whether a metric measures what its name claims. This runs the
pipeline over a labelled corpus and asks two questions the single-face
calibration cannot:

* **Separability.** For each metric, how well does it distinguish its target
  class from every other class? Reported as one-vs-rest AUC, which is
  threshold-free and unaffected by the classes having different sizes. 0.5 is
  a coin flip; below 0.5 means the metric runs BACKWARDS on this label set.
* **Ranking.** Does the metric take its highest median on the class it is
  named for? A metric that peaks elsewhere is measuring something else.

**Read the face-detection rate first.** skinlib needs 478 landmarks, so any
image that is a macro crop of skin rather than a face is dropped — and the drop
rate varies by class (measured 0% on blackheads, 72% on redness). Classes are
therefore represented by their DETECTABLE SUBSET, and whatever makes a face
detectable may itself correlate with the label. Treat cross-class comparisons
as indicative, and a class below `--min-detected` as not measured at all.

**A disagreement does not automatically indict the metric.** Web-collected
corpora draw each class from different sources, so class correlates with camera
and lighting. Colour metrics are known to drift 5.2x their own error bar between
two sessions in one room; across unrelated cameras that variation swamps skin
entirely. Texture is largely immune, which is why it is the family that scores.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from skinlib import Config, analyze, load_landmarker, load_parser

# Metric -> the class name substring it should score highest on. Only metrics
# with a defensible target belong here; a metric with no expected peak cannot
# pass or fail a ranking test.
TARGETS: dict[str, str] = {
    "inflammation_burden": "inflammatory acne",
    "spot_burden": "dark spots",
    "spot_contrast": "dark spots",
    "erythema_index": "redness",
    "hemoglobin_density": "redness",
    "roughness": "wrinkles",
    "melanin_density": "pigmentation",
}

REPORTED = list(dict.fromkeys(list(TARGETS) + ["uniformity", "melanin_index_rel"]))


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """One-vs-rest AUC via the rank-sum identity. No threshold, no class balance.

    Equivalent to the probability that a randomly drawn positive scores above a
    randomly drawn negative, which is exactly the question "does this metric
    separate this class" and is the reason a raw median comparison is not enough.
    """
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    joined = np.concatenate([positive, negative])
    order = joined.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, joined.size + 1)
    # Average ranks over ties, or ties inflate the score.
    _, inverse, counts = np.unique(joined, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = ranks[: positive.size].sum()
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2) / (positive.size * negative.size)
    )


def collect(root: Path, per_class: int, seed: int, config: Config,
            ignore_quality: bool = False) -> dict:
    classes = sorted(d for d in root.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"no class folders in {root}")

    parser = load_parser(config)
    generator = random.Random(seed)
    out: dict[str, dict] = {}

    with load_landmarker(config) as landmarker:
        for folder in classes:
            files = sorted(
                f for f in folder.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
            )
            generator.shuffle(files)

            values: dict[str, list[float]] = {m: [] for m in REPORTED}
            attempted = analysed = faces = 0
            flags: dict[str, int] = {}
            for path in files:
                if analysed >= per_class:
                    break
                attempted += 1
                try:
                    result = analyze(path, config=config, parser=parser, landmarker=landmarker)
                except Exception:
                    continue
                if result.face is not None:
                    faces += 1
                for flag in result.quality.flags:
                    flags[flag] = flags.get(flag, 0) + 1
                if result.metrics is None:
                    continue
                analysed += 1
                for metric in REPORTED:
                    value = result.metrics.global_.get(metric, float("nan"))
                    if np.isfinite(value):
                        values[metric].append(float(value))

            out[folder.name] = {
                "analysed": analysed,
                "attempted": attempted,
                "faces": faces,
                # Two different failures, reported separately: no face at all
                # (a macro crop of skin), versus a face the quality gate then
                # refused (blurry, badly framed, low resolution). Collapsing
                # them hides which one a dataset actually suffers from.
                "face_rate": faces / attempted if attempted else 0.0,
                "pass_rate": analysed / attempted if attempted else 0.0,
                "flags": dict(sorted(flags.items(), key=lambda kv: -kv[1])[:6]),
                "values": values,
            }
            top = ", ".join(f"{k} {v}" for k, v in list(out[folder.name]["flags"].items())[:3])
            print(
                f"  {folder.name:36s} face {out[folder.name]['face_rate']:4.0%} "
                f"pass {out[folder.name]['pass_rate']:4.0%}  n={analysed:3d}   {top}",
                flush=True,
            )
    return out


def report(data: dict, min_detected: int) -> str:
    lines: list[str] = []
    usable = {k: v for k, v in data.items() if v["analysed"] >= min_detected}
    dropped = {k: v for k, v in data.items() if v["analysed"] < min_detected}

    lines.append(f"{len(usable)} classes with >= {min_detected} analysed images")
    if dropped:
        lines.append("NOT MEASURED (too few faces detected): " + ", ".join(
            f"{k} ({v['analysed']})" for k, v in dropped.items()))
    lines.append("")

    header = f"{'metric':22s} {'target class':22s} {'AUC':>7s} {'rank':>7s}  verdict"
    lines.append(header)
    lines.append("-" * len(header))

    for metric, want in TARGETS.items():
        target = next((k for k in usable if want.lower() in k.lower()), None)
        if target is None:
            lines.append(f"{metric:22s} {want:22s} {'-':>7s} {'-':>7s}  target class not measured")
            continue

        positive = np.array(usable[target]["values"][metric], dtype=np.float64)
        negative = np.concatenate(
            [np.array(v["values"][metric], dtype=np.float64) for k, v in usable.items() if k != target]
            or [np.array([])]
        )
        score = auc(positive, negative)

        medians = {k: (np.median(v["values"][metric]) if v["values"][metric] else -np.inf)
                   for k, v in usable.items()}
        order = sorted(medians, key=medians.get, reverse=True)
        place = order.index(target) + 1

        if not np.isfinite(score):
            verdict = "no data"
        elif score >= 0.70 and place == 1:
            verdict = "PASS"
        elif score >= 0.60:
            verdict = "weak"
        elif score < 0.45:
            verdict = "BACKWARDS"
        else:
            verdict = "FAIL"
        lines.append(
            f"{metric:22s} {target[:22]:22s} {score:7.3f} {place:4d}/{len(usable)}  {verdict}"
        )

    lines.append("")
    lines.append("AUC is one-vs-rest: 0.5 is chance, <0.5 means the metric runs backwards.")
    lines.append("PASS needs AUC >= 0.70 AND the highest median on the target class.")
    lines.append("Detection rates differ by class, so each class is its DETECTABLE subset.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", type=Path, help="a directory of class folders")
    parser.add_argument("--per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-detected", type=int, default=25)
    parser.add_argument("--ignore-quality", action="store_true",
                        help="measure every image with a detectable face, even ones the "
                             "quality gate would reject. The gate is calibrated for "
                             "controlled captures and rejects most in-the-wild photos, "
                             "which biases a benchmark far more than an imperfect capture does")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--compare", type=Path,
                        help="a previous JSON, to show what moved since")
    args = parser.parse_args()

    config = Config()
    if args.ignore_quality:
        config = replace(config, quality=replace(config.quality,
                                                 short_circuit_when_unusable=False))
    print(f"running over {args.dataset}"
          + ("  (quality gate bypassed)" if args.ignore_quality else ""))
    data = collect(args.dataset, args.per_class, args.seed, config, args.ignore_quality)
    print()
    print(report(data, args.min_detected))

    if args.output:
        args.output.write_text(json.dumps(
            {"per_class": args.per_class, "seed": args.seed, "classes": data}, indent=2))
        print(f"\nwrote {args.output}")

    if args.compare and args.compare.exists():
        previous = json.loads(args.compare.read_text())["classes"]
        print("\nchange since " + str(args.compare))
        for metric in TARGETS:
            for name, current in data.items():
                if name not in previous:
                    continue
                now = current["values"][metric]
                was = previous[name]["values"].get(metric, [])
                if now and was:
                    delta = np.median(now) - np.median(was)
                    if abs(delta) > 1e-9:
                        print(f"  {metric:22s} {name[:22]:22s} {delta:+.5f}")


if __name__ == "__main__":
    main()
