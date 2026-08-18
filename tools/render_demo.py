"""Render what the app would actually show: photo, heat overlay, zone ranking.

Emits a JSON bundle of base64 images and numbers for the demo page. Everything
here comes from the real pipeline — quality gate, face parse, learned density
model — so the page is a screenshot of the system, not an illustration of it.
"""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from skinlib import Config, analyze, load_landmarker, load_parser
from tools.eval_marks_model import ZONES
from tools.train_marks import DensityNet, to_tensor

LABELS = {
    "forehead": "Forehead",
    "left_cheek": "Left cheek",
    "right_cheek": "Right cheek",
    "nose": "Nose",
    "mouth_chin": "Mouth & chin",
}


def encode(image: np.ndarray, width: int = 900, quality: int = 82) -> str:
    if image.shape[1] > width:
        scale = width / image.shape[1]
        image = cv2.resize(
            image, (width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def heat_overlay(
    image: np.ndarray, field: np.ndarray, skin: np.ndarray, cell: float
) -> np.ndarray:
    """Paint only what is ABOVE this face's own baseline.

    An absolute heat ramp paints every face red, because every face has some
    density everywhere; what a user needs to see is where THEY are worse than
    their own average. So the median over their skin is subtracted first and
    the remainder is what gets colour — the same face-relative reasoning the
    metrics use.
    """
    # Blur by roughly one grid cell. The model predicts a 32x32 map over the
    # face; showing it sharper than that draws boxes it did not draw.
    f = cv2.GaussianBlur(field, (0, 0), max(cell * 0.75, 2.0))
    inside = f[skin]
    if inside.size == 0:
        return image.copy()
    excess = np.clip(f - np.median(inside), 0, None)
    top = float(excess[skin].max())
    if top <= 0:
        return image.copy()
    norm = np.clip(excess / top, 0, 1) ** 1.6  # gamma: only the real peaks carry colour
    norm[~skin] = 0
    # Feather the mask edge. A hard skin-mask boundary reads as a paint smear,
    # which is a claim about a straight line the model never made.
    feather = cv2.GaussianBlur(
        skin.astype(np.float32), (0, 0), max(image.shape[1] / 200.0, 1.0)
    )
    norm *= feather
    ramp = np.zeros((*norm.shape, 3), dtype=np.float32)
    ramp[..., 2] = 255.0  # red channel (BGR)
    ramp[..., 1] = 210.0 * np.clip(1.0 - norm * 1.5, 0, 1)  # amber where it is mild
    ramp[..., 0] = 40.0
    alpha = (norm * 0.52)[..., None]
    out = image.astype(np.float32) * (1 - alpha) + ramp * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", type=Path, nargs="+")
    ap.add_argument("--model", type=Path, default=Path("models/marks_density.pt"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/bisenet_79999_iter.pth"))
    ap.add_argument("--out", type=Path, default=Path("data/demo.json"))
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = DensityNet(args.checkpoint, grid=ckpt["grid"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    config = replace(
        Config(), quality=replace(Config().quality, short_circuit_when_unusable=False)
    )
    parser = load_parser(config)

    cards = []
    with load_landmarker(config) as lm, torch.no_grad():
        for path in args.images:
            result = analyze(path, config=config, parser=parser, landmarker=lm)
            # analyze() may downscale; every mask and box below is in ITS frame.
            image = result.image
            face = result.face
            if face is None or result.metrics is None:
                print(f"skip {path.name}: no usable face")
                continue

            x, y, w, h = face.bbox
            side = max(w, h) * 1.30
            cx, cy = x + w / 2.0, y + h / 2.0
            x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
            pad = (
                int(max(0, -x0, -y0, x0 + side - image.shape[1], y0 + side - image.shape[0]))
                + 1
            )
            padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            patch = padded[y0 + pad : y0 + pad + int(side), x0 + pad : x0 + pad + int(side)]
            crop = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA)

            grid = torch.exp(model(to_tensor(crop[None], ckpt["size"])))[0].numpy()
            big = cv2.resize(grid, (int(side), int(side)), interpolation=cv2.INTER_CUBIC)
            big = np.clip(big, 0, None)
            big *= grid.sum() / max(big.sum(), 1e-9)
            canvas = np.zeros(image.shape[:2], dtype=np.float32)
            gx0, gy0 = max(x0, 0), max(y0, 0)
            gx1 = min(x0 + big.shape[1], canvas.shape[1])
            gy1 = min(y0 + big.shape[0], canvas.shape[0])
            canvas[gy0:gy1, gx0:gx1] = big[gy0 - y0 : gy1 - y0, gx0 - x0 : gx1 - x0]

            skin = result.masks.skin
            zones = []
            for key, members in ZONES.items():
                mask = np.zeros_like(skin)
                for name in members:
                    m = result.masks.regions.get(name)
                    if m is not None:
                        mask |= m
                if mask.sum() < 500:
                    continue
                zones.append(
                    {
                        "key": key,
                        "label": LABELS[key],
                        "density": float(canvas[mask].sum() / (mask.sum() / 1e5)),
                        "share": float(canvas[mask].sum()),
                    }
                )
            total = sum(z["share"] for z in zones) or 1.0
            peak = max(z["density"] for z in zones) or 1.0
            for z in zones:
                z["share"] = z["share"] / total
                z["level"] = z["density"] / peak
            zones.sort(key=lambda z: -z["density"])

            flags = sorted(result.quality.flags) if result.quality else []
            cards.append(
                {
                    "name": path.stem,
                    "photo": encode(image),
                    "overlay": encode(heat_overlay(image, canvas, skin, side / ckpt["grid"])),
                    "zones": zones,
                    "estimated_count": float(canvas[skin].sum()),
                    "flags": flags,
                    "face_px": int(face.width),
                    "metrics": {
                        "roughness": float(
                            result.metrics.global_.get("roughness", float("nan"))
                        ),
                        "erythema_index": float(
                            result.metrics.global_.get("erythema_index", float("nan"))
                        ),
                        "melanin_density": float(
                            result.metrics.global_.get("melanin_density", float("nan"))
                        ),
                    },
                }
            )
            print(f"{path.name}: {len(zones)} zones, worst {zones[0]['label']}, flags {flags}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cards))
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size:.1f} MB, {len(cards)} faces)")


if __name__ == "__main__":
    main()
