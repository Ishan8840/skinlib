"""Overlay masks, regions and spots on an image.

A debugging aid, not part of the library surface. Visual inspection catches
what tests do not: a region that has drifted onto the lip, a skin mask eating
the hairline, spots clustering along a shadow edge.

    python tools/visualize.py photo.jpg -o out.png

Renders a contact sheet: original, skin mask, regions, spots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skinlib.config import Config  # noqa: E402
from skinlib.types import AnalysisResult, Spot  # noqa: E402

# Distinct hues per region, BGR. Chosen to stay distinguishable when alpha
# blended over skin tones, which are themselves warm and mid-luminance.
REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "forehead": (255, 128, 0),
    "glabella": (255, 0, 255),
    "left_cheek": (0, 200, 255),
    "right_cheek": (0, 128, 255),
    "nose": (0, 255, 128),
    "perioral": (200, 0, 200),
    "chin": (128, 255, 0),
    "periorbital_left": (255, 255, 0),
    "periorbital_right": (180, 180, 0),
}

_ALPHA = 0.45


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = _ALPHA,
    outline: bool = True,
) -> np.ndarray:
    """Alpha-blend a single boolean mask over the image."""
    out = image.copy()
    if not mask.any():
        return out
    tint = np.zeros_like(out)
    tint[:] = color
    out[mask] = cv2.addWeighted(out, 1 - alpha, tint, alpha, 0)[mask]
    if outline:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 1, cv2.LINE_AA)
    return out


def overlay_regions(image: np.ndarray, regions: dict[str, np.ndarray]) -> np.ndarray:
    """Colour every region. Overlaps are impossible once regions are exclusive,
    so any blended colour visible here is a bug worth seeing."""
    out = image.copy()
    for name, mask in regions.items():
        out = overlay_mask(out, mask, REGION_COLORS.get(name, (255, 255, 255)))
    return out


def overlay_spots(image: np.ndarray, spots: list[Spot]) -> np.ndarray:
    """Circle each detected spot, sized to its own bounding box."""
    out = image.copy()
    for spot in spots:
        x, y, w, h = spot.bbox
        radius = max(3, int(round(max(w, h) * 0.7)))
        centre = (int(round(spot.centroid[0])), int(round(spot.centroid[1])))
        cv2.circle(out, centre, radius, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _legend(image: np.ndarray, regions: dict[str, np.ndarray]) -> np.ndarray:
    """Region names in their own colours, with pixel counts.

    The count is the useful half: a region reading zero pixels looks identical
    to a correct one in an overlay, but is obviously broken in a legend.
    """
    out = image.copy()
    y = 52
    for name, mask in regions.items():
        color = REGION_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(out, (8, y - 10), (24, y + 2), color, -1)
        cv2.putText(
            out, f"{name} ({int(mask.sum())}px)", (30, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += 20
    return out


def render_panel(result: AnalysisResult) -> np.ndarray:
    """Contact sheet of every stage's output, side by side."""
    if result.image is None:
        raise ValueError("result carries no image; nothing to visualise")
    base = result.image

    panes = [_label(base, "original")]

    if result.masks is not None:
        skin = _label(overlay_mask(base, result.masks.skin, (0, 255, 0)), "skin mask")
        panes.append(skin)
        regions = overlay_regions(base, result.masks.regions)
        panes.append(_legend(_label(regions, "regions"), result.masks.regions))

    # "0 spots" and "spots were never computed" look identical in an overlay,
    # and the gate short-circuits on an unusable capture. Say which it is.
    if result.metrics is None:
        spot_label = "spots (not computed - gate blocked)"
    else:
        spot_label = f"spots ({len(result.spots)})"
    panes.append(_label(overlay_spots(base, result.spots), spot_label))

    flags = ",".join(result.quality.flags) or "none"
    status = f"usable={result.quality.usable} flags={flags}"
    sheet = np.hstack(panes)
    footer = np.zeros((34, sheet.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        footer, f"{status}  |  {result.version}  cfg:{result.config_hash}",
        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([sheet, footer])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("visualization.png"))
    parser.add_argument(
        "--max-width", type=int, default=2400,
        help="downscale the contact sheet to this width for viewing",
    )
    args = parser.parse_args(argv)

    from skinlib import analyze

    result = analyze(args.image, config=Config())
    sheet = render_panel(result)

    if sheet.shape[1] > args.max_width:
        scale = args.max_width / sheet.shape[1]
        sheet = cv2.resize(
            sheet,
            (args.max_width, int(round(sheet.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    cv2.imwrite(str(args.output), sheet)
    print(f"wrote {args.output}")
    print(f"usable={result.quality.usable} flags={result.quality.flags}")
    if result.quality.unreliable_metrics:
        print(f"unreliable={sorted(result.quality.unreliable_metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
