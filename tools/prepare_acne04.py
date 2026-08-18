"""Cache face crops and lesion density targets from ACNE04-v2.

Runs MediaPipe once per image and writes a single .npz, so training can iterate
without paying for face detection every epoch. The crop is the face box grown by
a margin and squared off, then resized — the model never sees the background,
and every face arrives at the same scale, which is the same normalisation the
classical pipeline relies on.

The target is a density map, not a box list. The product question is "where are
the marks", and a coarse density map answers it directly; per-lesion boxes would
be answering a harder question and then throwing the extra precision away.
Each annotated circle is splatted as a unit-mass Gaussian whose sigma follows
the circle's own radius, so the map integrates to the lesion count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from skinlib import Config, load_landmarker
from skinlib.detect import detect_face

CROP = 256
GRID = 32


def splat(points: list[tuple[float, float, float]], grid: int, crop: int) -> np.ndarray:
    """Unit-mass Gaussians on a `grid` x `grid` map, in crop pixel coordinates."""
    out = np.zeros((grid, grid), dtype=np.float32)
    cell = crop / grid
    ys, xs = np.mgrid[0:grid, 0:grid].astype(np.float32)
    for cx, cy, r in points:
        gx, gy = cx / cell, cy / cell
        sigma = max(r / cell, 0.6)
        d2 = (xs - gx + 0.5) ** 2 + (ys - gy + 0.5) ** 2
        g = np.exp(-d2 / (2 * sigma * sigma))
        total = g.sum()
        if total > 0:
            out += (g / total).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", type=Path)
    ap.add_argument("annotations", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/acne04.npz"))
    ap.add_argument("--margin", type=float, default=0.15)
    args = ap.parse_args()

    raw = json.loads(args.annotations.read_text())
    by_id = {im["id"]: im["file_name"] for im in raw["images"]}
    circles: dict[str, list] = {im["file_name"]: [] for im in raw["images"]}
    for a in raw["annotations"]:
        name = by_id.get(a["image_id"])
        if name is None:
            continue
        x, y = a["coordinates"]
        circles[name].append((float(x), float(y), float(a["radius"])))

    config = Config()
    crops: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    counts: list[int] = []
    skipped = {"missing": 0, "no_face": 0, "small": 0}

    with load_landmarker(config) as lm:
        for i, (name, pts) in enumerate(sorted(circles.items())):
            path = args.images / name
            if not path.exists():
                skipped["missing"] += 1
                continue
            image = cv2.imread(str(path))
            if image is None:
                skipped["missing"] += 1
                continue
            face = detect_face(image, config, landmarker=lm)
            if face is None:
                skipped["no_face"] += 1
                continue
            x, y, w, h = face.bbox
            side = max(w, h) * (1.0 + 2 * args.margin)
            if side < 200:
                skipped["small"] += 1
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            x0, y0 = cx - side / 2.0, cy - side / 2.0
            # Pad rather than clamp: clamping would shift the face inside the
            # crop, and then the same landmark would map to different grid cells
            # on different images.
            pad = (
                int(max(0, -x0, -y0, x0 + side - image.shape[1], y0 + side - image.shape[0]))
                + 1
            )
            padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            xi, yi, si = int(x0) + pad, int(y0) + pad, int(side)
            patch = padded[yi : yi + si, xi : xi + si]
            if patch.shape[0] < 10 or patch.shape[1] < 10:
                skipped["small"] += 1
                continue
            scale = CROP / float(si)
            crop = cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)

            local = []
            for px, py, r in pts:
                qx = (px + pad - xi) * scale
                qy = (py + pad - yi) * scale
                if 0 <= qx < CROP and 0 <= qy < CROP:
                    local.append((qx, qy, r * scale))
            crops.append(crop)
            targets.append(splat(local, GRID, CROP))
            names.append(name)
            counts.append(len(local))
            if i % 100 == 0:
                print(f"  {i}/{len(circles)} kept={len(crops)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        crops=np.stack(crops),
        targets=np.stack(targets),
        names=np.array(names),
        counts=np.array(counts),
    )
    inside = int(np.sum(counts))
    total = sum(len(v) for v in circles.values())
    print(f"\nwrote {args.out}: {len(crops)} faces, {inside}/{total} circles inside the crops")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
