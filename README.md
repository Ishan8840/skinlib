# skinlib

Deterministic facial skin measurement. Photos in, structured metrics out —
redness, pigmentation, active inflammation, texture, and per-region breakdowns,
each with an error bar measured on the capture itself.

**Nothing is trained here.** Beyond two off-the-shelf models (face detection and
face parsing), every metric is plain signal processing: no fitted weights, no
training data, and no dataset licence to inherit. The same photo with the same
config always produces byte-identical numbers.

Built for **tracking skin over time**, which is a harder problem than measuring
it once. Most of the work in this library is about telling a real change apart
from a different room, a different distance, or a slightly turned head.

---

## Install

Python 3.11+. MediaPipe has no wheels for 3.14, so pin the environment:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
```

The CPU torch index is worth the extra line — the default wheel pulls ~2.5GB of
CUDA runtime this library never uses.

## Model assets

Two pretrained models, ~57MB total. Neither is vendored and neither is
downloaded at runtime: a model silently swapped between two sessions of a
longitudinal series is the one failure this library exists to prevent.

```bash
mkdir -p models

# BiSeNet face parser (CelebAMask-HQ, 19 classes, ~53MB)
curl -L -o models/bisenet_79999_iter.pth \
  https://huggingface.co/ManyOtherFunctions/face-parse-bisent/resolve/main/79999_iter.pth

# MediaPipe face landmarker (~3.8MB)
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Point the library at them, by environment or config:

```bash
export SKINLIB_BISENET_WEIGHTS=models/bisenet_79999_iter.pth
export SKINLIB_FACE_LANDMARKER=models/face_landmarker.task
```

A missing asset raises `AssetNotFoundError` — a deployment fault, never folded
into a quality flag.

> **Licence note.** The BiSeNet checkpoint is trained on CelebAMask-HQ, whose
> terms are research/non-commercial. Verify before shipping commercially. The
> parser is the one swappable component; any model producing a skin mask drops
> in.

---

## Usage

### One photo

```python
from skinlib import analyze, Config, load_parser

parser = load_parser(Config())          # 53MB — load once, reuse
result = analyze("photo.jpg", parser=parser)

if not result.quality.usable:
    print("retake:", result.quality.flags)     # e.g. ['blurry', 'too_far']
else:
    print(result.metrics.display())            # metrics safe to show a user
    print(result.metrics.by_region["left_cheek"])
    print(len(result.spots), "dark marks", len(result.lesions), "active lesions")
```

### A burst — recommended

Ten frames over two seconds feel like one photo to the user and buy three things
a single frame cannot: averaging, a choice of frames, and **an error bar
measured on this capture** rather than assumed.

```python
from skinlib import analyze_session, load_landmarker

with load_landmarker(Config()) as landmarker:
    session = analyze_session(paths, parser=parser, landmarker=landmarker)

session.metrics["spot_burden"]              # median across kept frames
session.detectable_change["spot_burden"]    # what a real change must exceed
session.frames                              # per-frame verdicts, and why any were dropped
```

Measured on a real burst, this takes `spot_count` from ±28.7 to ±11.4 and
`spot_burden` from 0.0030 to 0.0010.

### Tracking change

The question a bare number cannot answer:

```python
if this_week.trusted_change("spot_burden", last_week):
    show_change(...)
else:
    show("no measurable change")     # the honest default
```

---

## What you get

| | |
|---|---|
| **Pigmentation** | `spot_burden` (extent), `spot_contrast` (intensity), `melanin_density` |
| **Active inflammation** | `inflammation_burden`, `inflammation_contrast`, `result.lesions` |
| **Redness** | `erythema_index`, `hemoglobin_density` |
| **Texture** | `roughness`, `uniformity` |
| **Regions** | 9 mutually exclusive: forehead, glabella, cheeks, nose, perioral, chin, periorbital |
| **Derived** | left/right asymmetry, under-eye split into pigment / vascular / structural |
| **Geometry** | per-pixel surface normals, light direction, per-region incidence |

Two chromophores are separated analytically, so **dark marks and active acne are
measured independently** — a face whose lesions are healing into fresh
post-inflammatory marks shows falling inflammation and rising spot burden at the
same time, which no single channel can represent.

## Three rules for using it correctly

**1. Respect the quality gate.** `result.quality.usable` means *do not measure*.
When a capture is merely imperfect, `quality.unreliable_metrics` names which
metrics to drop, and `quality.trusted(name)` answers per metric. Surface these
as retake prompts.

**2. Only show `DISPLAY_SAFE_METRICS`.** `ita`, `monk_bin`, `melanin_index`,
`erythema` and `erythema_mean` measure the room as much as the skin — `ita`
correlates r = 0.94 with skin luminance. `result.metrics.display()` filters them
for you.

**3. Store `comparable_key` with every result.** Version plus config hash plus
both model content hashes. Two stored results are comparable only when all four
match; without it you will eventually plot a threshold change as a skin change.

---

## Capture protocol

Measurement quality is dominated by capture, not by algorithm. See
[`CAPTURE.md`](CAPTURE.md) for the full protocol; the short version:

- **Lock AE/AF** — tap and hold until `AE/AF LOCK` appears. Auto-exposure varying
  across a burst was the single largest error source measured.
- **Native camera, no messaging apps.** A round-trip through a chat app arrived
  at 640px and 0.07 bytes/px against 3088×2316 and 0.35 over USB.
- **Fixed distance**, one lamp, no windows.
- **Never change the front-camera mirror setting** — the library cannot detect
  mirroring, and flipping it silently swaps left and right cheek.

## Tools

```bash
python tools/visualize.py photo.jpg -o panel.png    # look before trusting numbers
python tools/repeatability.py DIR [DIR ...]         # noise floors + error budget
python tools/tone_audit.py --csv ... --out ...      # skin-tone fairness audit
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

192 tests. They skip cleanly when model assets are absent. `-n 2` parallelises;
`-n auto` will OOM, since each worker loads its own torch and BiSeNet.

---

## Going deeper

- **[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)** — why every threshold is what
  it is, the measurements behind each one, and the ideas that measurement killed.
- **`skinlib/version.py`** — change history, with empirical justification for
  every version bump.
- **`skinlib/config.py`** — every tunable in the library, each carrying the data
  that set it. No magic constants live in function bodies.

## Known limitations

Read these before trusting a number:

- **Thresholds are calibrated on one person, one room, one session.** Flag rates
  on a real user base are unknown.
- **Skin-tone fairness is partly unvalidated.** Four quality flags were found to
  reject dark skin and are fixed; whether the *detector and parser* handle darker
  faces equally is upstream of this library and unmeasured.
- **Session-to-session reproducibility is unmeasured.** Every error bar describes
  precision within one sitting.
- **Head angle costs 2–6× the noise floor** and only ~20% of that proved
  addressable. Hold pose steady.
- **Spot detection needs 3000px+ captures** to mean anything.

`docs/METHODOLOGY.md` covers each of these in full.
