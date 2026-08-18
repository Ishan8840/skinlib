"""Train a lesion-density model on ACNE04-v2 circles.

The classical detector was measured at 27% top-region accuracy on 281 faces it
had never seen (see version 14.0.0). That is 2.2x chance and nowhere near
useful, and it does not move with threshold tuning — hence a learned model.

Design, and why each piece is what it is:

encoder    The BiSeNet face-parsing ResNet18 already shipped in ``models/``.
           It was trained on CelebAMask-HQ, so its features are face features,
           not ImageNet's dog-and-boat features. It also costs no new download
           and no torchvision, which does not have a build matching the pinned
           torch.
head       Predicts a log-rate per cell, upsampled to a 32x32 density grid over
           the face crop.
loss       Poisson negative log-likelihood. The target is a count density, and
           counts are Poisson; MSE would treat a miss on a dense cheek the same
           as a miss on a clear forehead.
frozen     conv1 and layer1 stay frozen. They are generic edge filters, they are
           the most expensive layers to backprop through, and this machine is a
           laptop with no GPU.

Validation reports the metric the product is judged on — can it rank areas of
the face by how marked they are — over a 4x4 block grid, plus the count error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from skinlib._bisenet import Resnet18

BLOCKS = 4


def load_encoder(checkpoint: Path) -> Resnet18:
    model = Resnet18()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    prefix = "cp.resnet."
    sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(sub, strict=False)
    if unexpected:
        raise ValueError(f"unexpected encoder keys: {list(unexpected)[:3]}")
    if missing:
        raise ValueError(f"encoder half-loaded, missing {len(missing)}: {list(missing)[:3]}")
    return model


class DensityNet(nn.Module):
    def __init__(self, checkpoint: Path, grid: int = 32) -> None:
        super().__init__()
        self.grid = grid
        self.encoder = load_encoder(checkpoint)
        for name, param in self.encoder.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1")):
                param.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat8, feat16, _ = self.encoder(x)
        up = F.interpolate(feat16, size=feat8.shape[-2:], mode="bilinear", align_corners=False)
        log_rate = self.head(torch.cat([feat8, up], dim=1))
        return F.interpolate(
            log_rate, size=(self.grid, self.grid), mode="bilinear", align_corners=False
        ).squeeze(1)


def blocks(x: torch.Tensor) -> torch.Tensor:
    """Sum a (N, G, G) density into (N, BLOCKS*BLOCKS) face areas."""
    n, g, _ = x.shape
    step = g // BLOCKS
    return (
        x[:, : step * BLOCKS, : step * BLOCKS]
        .reshape(n, BLOCKS, step, BLOCKS, step)
        .sum(dim=(2, 4))
        .reshape(n, -1)
    )


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def rank(v: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(v, axis=1), axis=1).astype(np.float64)

    ra, rb = rank(a), rank(b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    denom = np.sqrt((ra**2).sum(1) * (rb**2).sum(1))
    return np.where(denom > 0, (ra * rb).sum(1) / np.maximum(denom, 1e-9), np.nan)


def augment(crop: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
        target = target[:, ::-1]
    gain = np.float32(rng.uniform(0.85, 1.15))
    bias = np.float32(rng.uniform(-12, 12))
    crop = np.clip(crop.astype(np.float32) * gain + bias, 0, 255)
    return np.ascontiguousarray(crop), np.ascontiguousarray(target)


def to_tensor(crops: np.ndarray, size: int) -> torch.Tensor:
    x = torch.from_numpy(crops.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1)  # BGR, as cv2 read it
    std = torch.tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1)
    return (x - mean) / std


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/acne04.npz"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/bisenet_79999_iter.pth"))
    ap.add_argument("--out", type=Path, default=Path("models/marks_density.pt"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=6, help="kept below core count on purpose")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    blob = np.load(args.data, allow_pickle=False)
    crops, targets, names = blob["crops"], blob["targets"], blob["names"]
    n = len(crops)

    # Split by severity level, which the filename carries. A random split would
    # let the val set drift lighter or heavier than train and make the numbers
    # unreadable.
    level = np.array([str(s).split("_")[0] for s in names])
    val_mask = np.zeros(n, dtype=bool)
    for lv in np.unique(level):
        idx = np.flatnonzero(level == lv)
        rng.shuffle(idx)
        val_mask[idx[: int(round(len(idx) * args.val_frac))]] = True
    tr, va = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    print(
        f"{n} faces: {len(tr)} train / {len(va)} val, levels {dict(zip(*np.unique(level, return_counts=True), strict=True))}"
    )

    model = DensityNet(args.checkpoint, grid=targets.shape[-1])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -np.inf
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        total, steps = 0.0, 0
        for s in range(0, len(order), args.batch):
            idx = order[s : s + args.batch]
            batch_c, batch_t = [], []
            for j in idx:
                c, t = augment(crops[j], targets[j], rng)
                batch_c.append(c)
                batch_t.append(t)
            x = to_tensor(np.stack(batch_c), args.size)
            y = torch.from_numpy(np.stack(batch_t))
            log_rate = model(x)
            loss = F.poisson_nll_loss(log_rate, y, log_input=True, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
            if steps % 20 == 0:
                print(
                    f"  epoch {epoch} step {steps}/{-(-len(order) // args.batch)} loss {total / steps:.4f}",
                    flush=True,
                )
        sched.step()

        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for s in range(0, len(va), args.batch):
                idx = va[s : s + args.batch]
                x = to_tensor(crops[idx], args.size)
                preds.append(torch.exp(model(x)))
                truth.append(torch.from_numpy(targets[idx]))
        p = torch.cat(preds)
        t = torch.cat(truth)
        pb, tb = blocks(p).numpy(), blocks(t).numpy()
        keep = tb.sum(1) > 0
        rho = np.nanmean(spearman_rows(tb[keep], pb[keep]))
        top1 = float(np.mean(np.argmax(pb[keep], 1) == np.argmax(tb[keep], 1)))
        order2 = np.argsort(-pb[keep], axis=1)[:, :2]
        top2 = float(np.mean([np.argmax(tb[keep][i]) in order2[i] for i in range(keep.sum())]))
        count_mae = float(np.mean(np.abs(p.sum((1, 2)).numpy() - t.sum((1, 2)).numpy())))
        print(
            f"epoch {epoch}: train {total / max(steps, 1):.4f} | "
            f"val rho {rho:+.3f} top1 {top1:.0%} top2 {top2:.0%} count MAE {count_mae:.1f}",
            flush=True,
        )
        if rho > best:
            best = rho
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "grid": int(targets.shape[-1]),
                    "size": args.size,
                    "rho": float(rho),
                },
                args.out,
            )
            print(f"  saved {args.out} (rho {rho:+.3f})", flush=True)

    print(f"\nbest val rho {best:+.3f}")


if __name__ == "__main__":
    main()
