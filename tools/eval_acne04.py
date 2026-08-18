"""Score skinlib against ACNE04-v2's dermatologist lesion circles.

ACNE04-v2 (Gazeau et al., MICCAI 2024) re-annotated the ICCV-2019 ACNE04 images
with 32,443 lesion circles — centre and radius, which is exactly the shape of a
hand label from ``tools/label.py``. That makes it ground truth we did not have
to draw, on 304 faces rather than the dozen labelled by hand.

Two questions, deliberately separated:

  per-lesion  — does a detection land on a circle? This is the metric that has
                capped at F1 ~0.42 on our own faces, and it is NOT what the
                product needs.
  regional    — do the regions carrying the most lesions rank the same way in
                the detector as in the annotation? This is what "tell me where
                the marks are" actually asks for.

Annotation coordinates are in the ORIGINAL capture resolution (mostly
3112x3456); the mirrored images are downscaled to 1024px, so every coordinate
is rescaled by the true width ratio rather than an assumed constant.

ACNE04 is free for academic use only — this validates the pipeline, it does not
license a model trained on it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from skinlib import Config, analyze, load_landmarker, load_parser


def load_annotations(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text())
    by_id = {im["id"]: im for im in raw["images"]}
    out: dict[str, dict] = {}
    for im in raw["images"]:
        out[im["file_name"]] = {"width": im["width"], "height": im["height"], "circles": []}
    for a in raw["annotations"]:
        im = by_id.get(a["image_id"])
        if im is None:
            continue
        x, y = a["coordinates"]
        out[im["file_name"]]["circles"].append((float(x), float(y), float(a["radius"])))
    return out


def per_lesion(dets, circles, scale: float) -> tuple[int, int, int]:
    """A detection is a hit if its centroid falls inside an unclaimed circle."""
    claimed = [False] * len(circles)
    tp = 0
    for d in dets:
        cx, cy = d.centroid
        best, best_dist = -1, None
        for i, (x, y, r) in enumerate(circles):
            if claimed[i]:
                continue
            dist = np.hypot(cx - x * scale, cy - y * scale)
            if dist <= max(r * scale, 4.0) and (best_dist is None or dist < best_dist):
                best, best_dist = i, dist
        if best >= 0:
            claimed[best] = True
            tp += 1
    return tp, len(dets) - tp, len(circles) - tp


def region_densities(masks, points) -> dict[str, float]:
    out = {}
    for name, m in masks.regions.items():
        if name == "skin" or m.sum() < 500:
            continue
        h, w = m.shape
        n = sum(
            1
            for px, py in points
            if m[min(max(int(py), 0), h - 1), min(max(int(px), 0), w - 1)]
        )
        out[name] = n / (m.sum() / 1e5)
    return out


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return order.astype(float)

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", type=Path)
    ap.add_argument("annotations", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ann = load_annotations(args.annotations)
    config = replace(
        Config(),
        quality=replace(Config().quality, short_circuit_when_unusable=False),
    )
    parser = load_parser(config)

    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    if args.limit:
        files = files[: args.limit]

    tot = defaultdict(int)
    rhos: list[float] = []
    top1 = top2 = scored = 0
    rows = []
    with load_landmarker(config) as lm:
        for i, path in enumerate(files):
            meta = ann.get(path.name)
            if meta is None:
                continue
            try:
                result = analyze(path, config=config, parser=parser, landmarker=lm)
            except Exception as exc:  # a bad decode is data, not a bug
                print(f"  skip {path.name}: {exc}", flush=True)
                continue
            if result.metrics is None:
                continue
            scale = result.image.shape[1] / float(meta["width"])
            dets = list(result.spots) + list(result.lesions)
            tp, fp, fn = per_lesion(dets, meta["circles"], scale)
            tot["tp"] += tp
            tot["fp"] += fp
            tot["fn"] += fn

            truth = region_densities(
                result.masks, [(x * scale, y * scale) for x, y, _ in meta["circles"]]
            )
            pred = region_densities(result.masks, [d.centroid for d in dets])
            shared = sorted(set(truth) & set(pred))
            if len(shared) >= 4 and sum(truth[k] for k in shared) > 0:
                rho = spearman([truth[k] for k in shared], [pred[k] for k in shared])
                if not np.isnan(rho):
                    rhos.append(rho)
                    worst_true = max(shared, key=lambda k: truth[k])
                    ranked = sorted(shared, key=lambda k: pred[k], reverse=True)
                    top1 += worst_true == ranked[0]
                    top2 += worst_true in ranked[:2]
                    scored += 1
            rows.append(
                {
                    "image": path.name,
                    "circles": len(meta["circles"]),
                    "detections": len(dets),
                    "tp": tp,
                }
            )
            if i % 25 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    prec = tot["tp"] / max(tot["tp"] + tot["fp"], 1)
    rec = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\nfaces scored: {len(rows)}  annotated lesions: {tot['tp'] + tot['fn']}")
    print("\n=== per-lesion (the hard metric, not what the product needs) ===")
    print(f"  precision {prec:.2f}  recall {rec:.2f}  F1 {f1:.2f}")
    print("\n=== regional (what 'where are the marks' actually asks) ===")
    if rhos:
        arr = np.array(rhos)
        print(f"  Spearman rho per face: mean {arr.mean():+.3f}  median {np.median(arr):+.3f}")
        print(f"  worst region ranked #1: {top1}/{scored} ({top1 / scored:.0%})")
        print(f"  worst region in top 2:  {top2}/{scored} ({top2 / scored:.0%})")
    else:
        print("  no face had enough regions to rank")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
