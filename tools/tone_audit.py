"""Does the quality gate reject darker skin more often?

    python tools/tone_audit.py --out /tmp/fitz --per-type 100

The one question this library has never been able to answer. Every threshold in
it was set on one person in one room, and README "Known limitations" says
plainly that ``too_dark`` thresholds mean skin lightness, so a correctly exposed
deep skin tone has a genuinely low mean L* and a bar tuned on light skin will
call it a bad photo.

That is a fairness claim, and it needs a tone-diverse set to test. This audits
against **Fitzpatrick17k** (Groh et al., CVPR 2021), which exists for exactly
this purpose.

**Licensing.** Fitzpatrick17k is CC BY-NC-SA 3.0 — non-commercial, share-alike.
It may be used to VALIDATE this library and must never be used to train, tune or
fit anything that ships. Nothing here writes a threshold; it only measures.

**What the numbers can and cannot support.** These are clinical atlas images:
~400px, arbitrary framing, real skin disease, and a different camera per photo.
They are nothing like a locked-AE iPhone burst, so the ABSOLUTE flag rates here
say nothing about your capture path. What transfers is the COMPARISON between
tones, because framing and resolution problems are tone-independent while
``too_dark`` is precisely what is under test. Read the columns against each
other, never against your own captures.

Only the Atlas Dermatologico half of the dataset is reachable: the DermaAmin
host serves a stub to non-browser clients, which is a deliberate access control
and is left alone. Atlas is 3,752 labelled images and is better balanced across
tones than the full set.
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from skinlib import Config, analyze, load_parser

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

# Fitzpatrick labels come from two independent annotation services. They
# disagree often enough that the dataset's own follow-up paper is about the
# disagreement, so both are carried and the primary is reported.
PRIMARY = "fitzpatrick_scale"
SECONDARY = "fitzpatrick_centaur"


# Conditions that present on the FACE. Without this filter the sample is
# dominated by close-ups of limbs and trunk: a first run over all Atlas rows
# found a face in 1-7% of images, which put every per-tone rate on a sample of
# one to six and made the audit meaningless.
FACIAL_CONDITIONS = (
    "rosacea",
    "acne",
    "melasma",
    "perioral dermatitis",
    "seborrheic dermatitis",
    "lupus",
)


def load_rows(csv_path: Path, facial_only: bool = True) -> list[dict]:
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if "atlasdermatologico" in row["url"] and row[PRIMARY] not in ("-1", "")
    ]
    if facial_only:
        rows = [
            row
            for row in rows
            if any(word in row["label"].lower() for word in FACIAL_CONDITIONS)
        ]
    return rows


def sample(rows: list[dict], per_type: int, seed: int = 0) -> list[dict]:
    """Stratify by Fitzpatrick type so no tone dominates the comparison.

    The dataset is skewed; drawing at random would mean the light-skin flag rate
    was estimated from ten times as many images as the dark-skin one, and the
    difference between them would then be mostly sampling noise.
    """
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        buckets[row[PRIMARY]].append(row)

    generator = random.Random(seed)
    out: list[dict] = []
    for key in sorted(buckets):
        pool = buckets[key][:]
        generator.shuffle(pool)
        out.extend(pool[:per_type])
    return out


def fetch(row: dict, directory: Path) -> Path | None:
    destination = directory / f"{row['md5hash']}.jpg"
    if destination.exists():
        return destination
    try:
        request = urllib.request.Request(row["url"], headers=HEADERS)
        data = urllib.request.urlopen(request, timeout=20).read()
    except Exception:
        return None
    # A stub or an error page decodes to nothing; check rather than trust size.
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
        return None
    destination.write_bytes(data)
    return destination


def audit(rows: list[dict], directory: Path, config: Config) -> list[dict]:
    parser = load_parser(config)
    records: list[dict] = []
    for row in rows:
        path = fetch(row, directory)
        if path is None:
            continue
        try:
            result = analyze(path, config=config, parser=parser)
        except Exception:
            continue
        records.append(
            {
                "fitzpatrick": row[PRIMARY],
                "centaur": row[SECONDARY],
                "label": row["label"],
                "face": result.face is not None,
                "usable": result.quality.usable,
                "flags": list(result.quality.flags),
                "mean_lightness": result.quality.measures.get("mean_lightness", float("nan")),
                "shadow_clipped": result.quality.measures.get(
                    "shadow_clipped_fraction", float("nan")
                ),
            }
        )
    return records


def report(records: list[dict]) -> str:
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        by_type[record["fitzpatrick"]].append(record)

    lines: list[str] = []
    lines.append(f"analysed {len(records)} images that decoded and reached the pipeline")
    lines.append("")
    header = (
        f"{'fitz':>4s} {'n':>4s} {'face':>6s} {'usable':>7s} "
        f"{'too_dark':>9s} {'mean L*':>8s} {'shadow clip':>12s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for key in sorted(by_type):
        group = by_type[key]
        found = [r for r in group if r["face"]]
        if not found:
            lines.append(f"{key:>4s} {len(group):4d}   no face detected")
            continue
        dark = sum(1 for r in found if "too_dark" in r["flags"])
        lightness = [r["mean_lightness"] for r in found if np.isfinite(r["mean_lightness"])]
        clipped = [r["shadow_clipped"] for r in found if np.isfinite(r["shadow_clipped"])]
        lines.append(
            f"{key:>4s} {len(group):4d} {len(found) / len(group):6.0%} "
            f"{sum(1 for r in found if r['usable']) / len(found):7.0%} "
            f"{dark / len(found):9.0%} "
            f"{np.median(lightness) if lightness else float('nan'):8.1f} "
            f"{np.median(clipped) if clipped else float('nan'):12.3f}"
        )

    lines.append("")
    lines.append("`too_dark` thresholds mean skin L*, so a correctly exposed deep tone")
    lines.append("reads low through no fault of the photo. `shadow clip` is the")
    lines.append("tone-INDEPENDENT part of the same question: real underexposure crushes")
    lines.append("pixels against black and destroys information, dark skin does not. A")
    lines.append("too_dark rate that climbs with Fitzpatrick type while shadow clipping")
    lines.append("stays flat is the gate mistaking skin tone for a bad capture.")

    counts: collections.Counter = collections.Counter()
    for record in records:
        counts.update(record["flags"])
    if counts:
        lines.append("")
        lines.append("all flags: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", type=Path, required=True, help="fitzpatrick17k.csv")
    parser.add_argument("--out", type=Path, required=True, help="image cache directory")
    parser.add_argument("--per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="do not restrict to facial conditions (yields ~1-7%% face detection)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.csv, facial_only=not args.all_conditions)
    chosen = sample(rows, args.per_type, args.seed)
    print(f"{len(rows)} reachable rows; sampling {len(chosen)} stratified by tone")

    records = audit(chosen, args.out, Config())
    print()
    print(report(records))


if __name__ == "__main__":
    main()
