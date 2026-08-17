"""Score spot detection against a hand-labelled image.

Labelling protocol: open the photo in any editor, draw a closed outline in
saturated red around each mark, save as PNG. Outlines only — do not fill them.
Any red that is not a stroke (lips, for instance) is rejected by saturation, so
lip colour will not be mistaken for an annotation.

    python tools/evaluate.py photo.jpg labelled.png
    python tools/evaluate.py photo.jpg labelled.png --sweep

The labelled copy does not need to be the same size as the photo. Landmarks are
used to register one to the other, and the fit is reported so a bad
registration cannot quietly corrupt the scores.

This exists because thresholds tuned by eye are guesses. One labelled face is
already enough to show a default was wrong by a factor of two in recall; a
handful would be enough to set one honestly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace  # noqa: E402

from skinlib import Config, analyze, detect_face, load_image  # noqa: E402

# Annotation strokes are saturated; skin and lips are not. Measured separation
# on a real labelled image: strokes R-G = 100..120, lips R-G = 57..67.
MIN_RED_EXCESS = 85
MIN_RED = 120
MIN_STROKE_AREA = 12


def extract_labels(labelled: np.ndarray) -> list[tuple[float, float]]:
    """Centroids of each red outline, in the labelled image's own coordinates."""
    blue, green, red = (labelled[:, :, i].astype(int) for i in range(3))
    strokes = (red > MIN_RED) & (red - green > MIN_RED_EXCESS) & (red - blue > MIN_RED_EXCESS)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(strokes.astype(np.uint8), 8)
    centres: list[tuple[float, float]] = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < MIN_STROKE_AREA:
            continue
        # Fill the outline: the mark is what the stroke encloses, not the ink.
        contours, _ = cv2.findContours(
            (labels == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros(labelled.shape[:2], np.uint8)
        cv2.drawContours(filled, contours, -1, 1, -1)
        ys, xs = np.nonzero(filled)
        centres.append((float(xs.mean()), float(ys.mean())))
    return centres


def register(labelled: np.ndarray, working: np.ndarray, config: Config) -> np.ndarray:
    """Similarity transform mapping labelled coordinates onto the working image."""
    source = detect_face(labelled, config)
    target = detect_face(working, config)
    if source is None or target is None:
        raise SystemExit("a face must be detectable in both images to register them")

    matrix, inliers = cv2.estimateAffinePartial2D(
        source.landmarks, target.landmarks, method=cv2.RANSAC
    )
    if matrix is None:
        raise SystemExit("could not register the labelled image against the photo")

    projected = cv2.transform(source.landmarks[None], matrix)[0]
    residual = np.linalg.norm(projected - target.landmarks, axis=1)
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    print(
        f"registration: scale={scale:.4f}  inliers={int(inliers.sum())}/{len(inliers)}  "
        f"median residual={np.median(residual):.2f}px  max={residual.max():.2f}px"
    )
    if np.median(residual) > 3.0:
        print("  WARNING: poor registration; scores below are unreliable")
    return matrix


def score(detections, truth: np.ndarray, radius: float) -> dict[str, float]:
    """Greedy nearest-neighbour matching, one detection per labelled mark."""
    taken: set[int] = set()
    matched: list[int] = []
    for point in truth:
        best, best_distance = None, float("inf")
        for index, spot in enumerate(detections):
            if index in taken:
                continue
            distance = float(np.hypot(spot.centroid[0] - point[0], spot.centroid[1] - point[1]))
            if distance < best_distance:
                best, best_distance = index, distance
        if best is not None and best_distance <= radius:
            taken.add(best)
            matched.append(best)

    true_positive = len(matched)
    precision = true_positive / len(detections) if detections else 0.0
    recall = true_positive / len(truth) if len(truth) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "fp": len(detections) - true_positive,
        "fn": len(truth) - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("labelled", type=Path)
    parser.add_argument("--radius", type=float, default=15.0, help="match distance in px")
    parser.add_argument("--sweep", action="store_true", help="sweep threshold_mad")
    args = parser.parse_args(argv)

    config = Config()
    labelled = cv2.imread(str(args.labelled), cv2.IMREAD_COLOR)
    if labelled is None:
        raise SystemExit(f"could not read {args.labelled}")

    working = load_image(args.photo, config.io).image
    truth_local = extract_labels(labelled)
    if not truth_local:
        raise SystemExit("no red outlines found; are the annotations saturated enough?")

    matrix = register(labelled, working, config)
    truth = cv2.transform(np.array(truth_local, np.float32)[:, None, :], matrix).reshape(-1, 2)
    print(f"labelled marks: {len(truth)}   match radius: {args.radius:.0f}px\n")

    result = analyze(args.photo, config=config)
    outcome = score(result.spots, truth, args.radius)
    matched = set(outcome["matched"])

    print(f"threshold_mad={config.spots.threshold_mad}: {len(result.spots)} detections")
    print(
        f"  TP={outcome['tp']}  FP={outcome['fp']}  FN={outcome['fn']}   "
        f"precision={outcome['precision']:.2f}  recall={outcome['recall']:.2f}  "
        f"F1={outcome['f1']:.2f}"
    )
    for index, spot in enumerate(result.spots):
        tag = "TP" if index in matched else "FP"
        print(f"   [{tag}] {spot.region or '(none)':18s} {spot.area_px:4d}px")

    if args.sweep:
        print(f"\n{'mad':>5s} {'det':>4s} {'TP':>3s} {'FP':>3s} {'FN':>3s} "
              f"{'prec':>5s} {'rec':>5s} {'F1':>5s}")
        for mad in (3.0, 2.8, 2.5, 2.2, 2.0, 1.8, 1.6, 1.4):
            tuned = replace(config, spots=replace(config.spots, threshold_mad=mad))
            spots = analyze(args.photo, config=tuned).spots
            row = score(spots, truth, args.radius)
            print(
                f"{mad:5.1f} {len(spots):4d} {row['tp']:3d} {row['fp']:3d} {row['fn']:3d} "
                f"{row['precision']:5.2f} {row['recall']:5.2f} {row['f1']:5.2f}"
            )
        print("\nA narrow F1 peak means the optimum is fragile. Prefer a value that sits")
        print("on a plateau across SEVERAL labelled faces over the best score on one.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
