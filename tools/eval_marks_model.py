"""Score the trained density model on real anatomical regions.

``train_marks.py`` validates on a 4x4 block grid, which is cheap but is not what
the product says. This scores the same held-out faces on skinlib's actual
regions — forehead, cheeks, chin, nose — so the number is directly comparable to
the classical pipeline's 27% top-1 / 47% top-2 from version 14.0.0.

The split is recomputed with the training script's own seed and stratification,
so the faces scored here are the faces the model never saw.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from skinlib import Config, analyze, load_landmarker, load_parser
from tools.train_marks import DensityNet, spearman_rows, to_tensor


def val_indices(names: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = np.array([str(s).split("_")[0] for s in names])
    mask = np.zeros(len(names), dtype=bool)
    for lv in np.unique(level):
        idx = np.flatnonzero(level == lv)
        rng.shuffle(idx)
        mask[idx[: int(round(len(idx) * val_frac))]] = True
    return np.flatnonzero(mask)


def region_density(mask: np.ndarray, field: np.ndarray) -> float:
    return float(field[mask].sum() / (mask.sum() / 1e5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/acne04.npz"))
    ap.add_argument("--model", type=Path, default=Path("models/marks_density.pt"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/bisenet_79999_iter.pth"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = np.load(args.data, allow_pickle=False)
    names, crops, targets = blob["names"], blob["crops"], blob["targets"]
    va = val_indices(names, args.val_frac, args.seed)
    if args.limit:
        va = va[: args.limit]
    print(f"scoring {len(va)} held-out faces")

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = DensityNet(args.checkpoint, grid=ckpt["grid"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    config = replace(
        Config(), quality=replace(Config().quality, short_circuit_when_unusable=False)
    )
    parser = load_parser(config)

    rhos_m, rhos_c = [], []
    t1m = t2m = t1c = t2c = scored = 0
    with load_landmarker(config) as lm, torch.no_grad():
        for k, j in enumerate(va):
            name = str(names[j])
            path = args.images / name
            if not path.exists():
                continue
            result = analyze(path, config=config, parser=parser, landmarker=lm)
            if result.metrics is None:
                continue
            # The cache crop is the face box + margin, squared; rebuild it here so
            # the predicted grid can be laid back over the original pixels.
            face = result.face
            if face is None:
                continue
            x, y, w, h = face.bbox
            side = max(w, h) * 1.30
            cx, cy = x + w / 2.0, y + h / 2.0
            x0, y0 = cx - side / 2.0, cy - side / 2.0

            pred = torch.exp(model(to_tensor(crops[j : j + 1], ckpt["size"])))[0].numpy()
            true = targets[j]

            fields = {}
            for key, grid in (("model", pred), ("truth", true)):
                big = cv2.resize(grid, (int(side), int(side)), interpolation=cv2.INTER_LINEAR)
                big *= grid.sum() / max(big.sum(), 1e-9)  # resampling must conserve count
                canvas = np.zeros(result.image.shape[:2], dtype=np.float32)
                sx, sy = int(round(x0)), int(round(y0))
                gx0, gy0 = max(sx, 0), max(sy, 0)
                gx1 = min(sx + big.shape[1], canvas.shape[1])
                gy1 = min(sy + big.shape[0], canvas.shape[0])
                if gx1 <= gx0 or gy1 <= gy0:
                    continue
                canvas[gy0:gy1, gx0:gx1] = big[gy0 - sy : gy1 - sy, gx0 - sx : gx1 - sx]
                fields[key] = canvas
            if len(fields) != 2:
                continue

            regions = [
                (n, m)
                for n, m in result.masks.regions.items()
                if n != "skin" and m.sum() >= 500
            ]
            if len(regions) < 4:
                continue
            truth_v = np.array([[region_density(m, fields["truth"]) for _, m in regions]])
            model_v = np.array([[region_density(m, fields["model"]) for _, m in regions]])
            dets = list(result.spots) + list(result.lesions)
            cls_v = np.array(
                [
                    [
                        sum(
                            1
                            for d in dets
                            if m[
                                min(int(d.centroid[1]), m.shape[0] - 1),
                                min(int(d.centroid[0]), m.shape[1] - 1),
                            ]
                        )
                        / (m.sum() / 1e5)
                        for _, m in regions
                    ]
                ]
            )
            if truth_v.sum() <= 0:
                continue
            rm = spearman_rows(truth_v, model_v)[0]
            rc = spearman_rows(truth_v, cls_v)[0]
            if np.isnan(rm):
                continue
            rhos_m.append(rm)
            rhos_c.append(0.0 if np.isnan(rc) else rc)
            worst = int(np.argmax(truth_v[0]))
            om = np.argsort(-model_v[0])[:2]
            oc = np.argsort(-cls_v[0])[:2]
            t1m += worst == om[0]
            t2m += worst in om
            t1c += worst == oc[0]
            t2c += worst in oc
            scored += 1
            if k % 25 == 0:
                print(f"  {k}/{len(va)}", flush=True)

    print(f"\n{scored} faces scored on anatomical regions\n")
    print(f"{'':10s} {'rho':>8s} {'top-1':>8s} {'top-2':>8s}")
    print(f"{'model':10s} {np.mean(rhos_m):+8.3f} {t1m / scored:8.0%} {t2m / scored:8.0%}")
    print(f"{'classical':10s} {np.mean(rhos_c):+8.3f} {t1c / scored:8.0%} {t2c / scored:8.0%}")


if __name__ == "__main__":
    main()
