# skinlib

Deterministic facial skin measurement. One photo in, structured metrics out.

No learned components beyond off-the-shelf face detection and face parsing.
Nothing is trained here. Everything downstream of the two pretrained models is
plain, reproducible signal processing.

```python
from skinlib import analyze, Config

result = analyze("photo.jpg", config=Config())

result.quality.usable               # bool
result.quality.flags                # list[str]
result.quality.unreliable_metrics   # frozenset[str] - which metrics to distrust
result.metrics.global_              # dict[str, float]
result.metrics.by_region            # dict[region, dict[metric, float]]
result.spots                        # list[Spot]
result.masks.skin                   # np.ndarray bool
result.masks.regions                # dict[region, np.ndarray bool]
result.version                      # preprocessing version string
result.comparable_key               # (version, config_hash, weights_hash, landmarker_hash)
```

Stages are exposed individually for debugging and reuse:

```python
from skinlib import (
    detect_face, parse_skin, build_regions, check_quality,
    correct_color, compute_metrics, detect_spots, load_parser,
)
```

`analyze` loads the BiSeNet checkpoint on every call. When processing many
images, hoist it:

```python
parser = load_parser(config)
for path in paths:
    result = analyze(path, config=config, parser=parser)
```

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

Two pretrained models, neither vendored and neither downloaded at runtime. A
download during analysis could silently swap a model between two sessions of a
longitudinal series, which is the one failure this library exists to prevent.

**BiSeNet face parser** (CelebAMask-HQ, 19 classes, ~53MB). Original:
[zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
(`79999_iter.pth`, hosted on Google Drive). A direct-download mirror:

```bash
curl -L -o models/bisenet_79999_iter.pth \
  https://huggingface.co/ManyOtherFunctions/face-parse-bisent/resolve/main/79999_iter.pth
```

**MediaPipe face landmarker** (~3.8MB). MediaPipe ≥ 1.0 removed the legacy
`solutions.face_mesh` API, so the landmarker is now an external asset too:

```bash
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Point the library at them by config or environment:

```bash
export SKINLIB_BISENET_WEIGHTS=models/bisenet_79999_iter.pth
export SKINLIB_FACE_LANDMARKER=models/face_landmarker.task
```

```python
config = Config(
    parse=ParseConfig(weights_path=Path("models/bisenet_79999_iter.pth")),
    detect=DetectConfig(landmarker_asset_path=Path("models/face_landmarker.task")),
)
```

A missing asset raises `AssetNotFoundError`. It is a deployment fault, not a
property of the photo, so it is never folded into a quality flag.

## Pipeline

```
image
  -> load           EXIF orientation, downscale-only to 1536px long edge
  -> detect         MediaPipe FaceLandmarker, 478 landmarks
  -> parse          BiSeNet -> skin mask, then nostril/facial-hair/oval cleanup
  -> regions        landmark geometry -> 9 mutually exclusive region masks
  -> quality        flags + per-metric reliability (never raises)
  -> color          sclera reference, falling back to shades-of-grey p=6
  -> spots          melanin residual -> connected components -> shape filters
  -> metrics        per region and global
```

The quality gate runs before colour correction, on raw pixels: correcting first
would mask the exposure and white-balance problems it exists to detect.

Spots run before metrics because `spot_count` and `spot_area_fraction` are
metric columns. Calling `compute_metrics` without spots reports those two as
NaN — "not measured", which is a different claim from 0, "measured, none
found".

## Surface geometry

MediaPipe returns a z for every landmark. `Face.landmarks_z` keeps it, and
`geometry.py` turns it into a depth surface, per-pixel normals, and a
least-squares estimate of the light direction.

```python
from skinlib import region_incidence, surface_normals, estimate_light_direction

incidence = region_incidence(face, regions, image.shape[:2])   # cos vs camera
direction, r_squared = estimate_light_direction(image, normals, skin)
```

Validated: normals are unit vectors, the forehead reads cos 0.95 against the
chin's 0.63 — a flat plane against the surface that turns under the jaw — and
the light solver recovers a synthetic Lambertian illuminant to r² > 0.98, which
establishes that a poor fit on a real face is the face departing from the model
rather than the solver failing.

**What landmark depth cannot do.** It is a *learned* single-view estimate over
478 sparse vertices. It describes curvature at the scale of a cheek and carries
nothing at the scale of a pore or a wrinkle. Line depth, lesion volume and
relief need real depth — a TrueDepth stream, photometric stereo, or multi-view —
and nothing here pretends otherwise.

### Incidence is a weight, not a divisor

The obvious use would be correcting metrics for how obliquely a region is
viewed. **Measured on the 13-frame `angle` set, that does not work.**

Incidence is a genuine covariate — mean |r| across regions is 0.51 for
`spot_burden`, 0.46 for `inflammation_burden`, 0.27 for `roughness` — but its
**sign is inconsistent between regions**:

```
forehead  +0.50      nose              -0.85
perioral  +0.61      periorbital_left  -0.80
chin      -0.54      periorbital_right +0.53
```

A factor acting through foreshortening or shading would push the same way
everywhere. Dividing by the cosine would improve some regions and damage others.

The likeliest explanation is that the angle penalty is not photometric at all.
`build_regions` places regions from **2D** landmark geometry, so under
out-of-plane rotation each mask slides across slightly different anatomy — the
metric moves because the region moved, not because the light did. If that is
right, the real fix is defining regions in a 3D canonical face frame, which this
depth field makes possible and which is a larger change than a correction
factor.

## Derived quantities

`derived.py` rearranges the metric table; it never touches pixels. A derived
value inherits its inputs' noise and cannot be more trustworthy than they are.

**Asymmetry** — signed left-minus-right for paired regions (`cheek`,
`periorbital`). Signed rather than absolute because which side is worse is the
interesting part, and an absolute difference cannot average to zero across
sessions, so noise would accumulate into a fake finding. Its noise floor is
roughly √2 × a single region's.

**Periorbital decomposition** — under-eye darkness split into three conditions
that look alike and respond differently: **pigmented** (melanin), **vascular**
(haemoglobin in thin skin, the classic blue-grey), and **structural** (a hollow
casting a shadow). Each is measured against the cheek directly below, being the
same person's skin under the same light. This library can attempt the split only
because it separates melanin from haemoglobin analytically and measures surface
geometry.

The component shares are **descriptive, not diagnostic** — they say which signal
dominates, not what causes it, and the structural term is inferred from coarse
landmark geometry that cannot resolve a tear trough directly.

## Bursts

```python
from skinlib import analyze_session

session = analyze_session(sorted(Path("burst").glob("*.JPG")))

session.metrics["spot_burden"]            # median across kept frames
session.detectable_change["spot_burden"]  # what a real change must exceed
session.trusted_change("spot_burden", last_week)   # bool
```

A single photo yields a number with no indication of how far to trust it. Ten
frames over two seconds feel identical to the user and buy three things one
frame cannot at any price.

**Averaging.** Noise falls as 1/√n. Measured on a real 12-frame burst:

| | single frame | 10-frame session |
|---|---|---|
| `spot_burden` detectable change | 0.00302 | **0.00102** |
| `spot_count` detectable change | 28.7 | **11.4** |

**A choice of frames.** The sharpest, the ones with eyes open, the ones before
the subject drifted. Both criteria are judged **relative to the burst**, never
against an absolute bar — Laplacian variance depends on how much detail a face
has, so an absolute threshold would reject every frame of a smooth face and
accept every frame of a stubbled one. Rejected frames are still reported with a
reason, because *why* frames were dropped is how a capture protocol gets
debugged.

**An error bar measured on this capture**, not assumed from a table. A session's
noise depends on how steady the hands were that morning, and `trusted_change`
uses the blunter of two sessions' bars — a comparison is only as sharp as its
worse half.

One illuminant is estimated for the whole burst, from the frame with the most
confident sclera. Re-estimating per frame injects the estimator's own noise into
every colour metric, and the light did not change during a two-second capture.

Session flags (`few_frames`, `single_frame`, `high_frame_loss`, `subject_moved`)
are deliberately kept out of the metrics: a wide error bar because three frames
survived is a different thing from a wide error bar because the skin is
genuinely variable, and only the flags distinguish them.

### Specular recovery does not work — measured, not assumed

The idea was appealing: glare is view-dependent and chromophores are not, so
hand tremor should move a highlight while pigment stays put, making the
between-frame variation an oiliness measure with no flash pair needed.

It measures registration residual instead. On the 12-frame burst:

```
r(signal, edge gradient) = +0.551    <- misalignment at the lash line
r(signal, luminance)     = -0.221    <- glare would be POSITIVE
r(signal, saturation)    = +0.124    <- glare would be NEGATIVE
```

The edgiest 10% of skin scored 0.0185 against 0.0096 for the brightest 10%, and
`nose` — normally the shiniest region on a face — ranked 7th of 9. Gating on
bright *and* desaturated *and* away from edges removes the artifact
(r(edge) 0.551 → 0.134) and leaves nothing: the T-zone/cheek ratio *falls* to
1.08×.

`SessionConfig.recover_specular` is therefore False. The flash/no-flash pair
remains the sound route to oiliness — it separates the same two components by
changing the illumination rather than the viewpoint, which is a far larger
lever than a 6% hand tremor.

The temporal median composite **is** kept and is sound. Metrics are not computed
on it, because denoising changes what a texture metric means.

## Regions

`forehead`, `glabella`, `left_cheek`, `right_cheek`, `nose`, `perioral`,
`chin`, `periorbital_left`, `periorbital_right`.

Built in an anatomical coordinate frame derived from the landmarks, not in
image axes, so head tilt does not move the anatomy. Every region is intersected
with the skin mask and made mutually exclusive by a configured priority order,
so no pixel is counted twice.

**Priority matters.** `periorbital` outranks the cheeks deliberately: the cheek
band reaches up under the eye, and under-eye skin is the dark-circle signal.
With the cheek winning, that signal would be averaged into a region five times
its size and silently disappear — no error, just a wrong number.

**Left and right are anatomical** — the subject's left and right, following
MediaPipe's convention. See the mirroring limitation below.

## Metrics

| Metric | Definition |
|---|---|
| `erythema` | 75th percentile of CIELAB a\* |
| `erythema_mean` | mean of a\* |
| `melanin_index` | mean of log₁₀(1 / R_normalised) |
| `melanin_density` | median melanin coordinate in optical-density space |
| `hemoglobin_density` | median haemoglobin coordinate in optical-density space |
| `uniformity` | 1 / (1 + std(melanin_index)) |
| `roughness` | std of a band-passed grey (difference of Gaussians, σ = 0.008 × face width) |
| `ita` | degrees(arctan2(median(L\*) − 50, median(b\*))) |
| `monk_bin` | ITA mapped to a Monk Skin Tone bin, 1–10 |
| `spot_burden` | fraction of skin whose melanin residual clears a face-wide threshold |
| `spot_contrast` | 95th percentile of the residual, minus its face-wide median |
| `spot_count` | detected dark spots — see the noise floor warning below |
| `spot_area_fraction` | total spot area / skin mask area |
| `inflammation_burden` | fraction of skin whose haemoglobin residual clears a face-wide threshold |
| `inflammation_contrast` | 95th percentile of the haemoglobin residual, minus its face-wide median |

All computed on the colour-corrected image, inside the skin mask, and nowhere
else.

### Chromophore densities

Skin absorbance is approximately linear in two pigment densities, and shading
and exposure are **neutral** gains — so in optical density (−log₁₀ reflectance)
they are a pure additive offset along (1,1,1), whatever the geometry or skin
tone. Projecting that axis out removes them analytically; the remaining plane is
spanned by the melanin and haemoglobin absorbance directions, and the two
densities are the *coordinates* of the residual in that basis. Not projections —
the basis is not orthogonal (the axes sit 69.8° apart), so a dot product would
silently mix the pigments.

Nothing is fitted. The basis is a published constant (Tsumura et al., JOSA A
1999), the arithmetic is a closed-form 2×2 solve, and there is no ICA and no
SVD — `np.linalg.pinv` would have put LAPACK inside the determinism guarantee
for no benefit at two dimensions.

Measured on `portrait_a` against changes that are zero pigment change:

| signal | shading gradient | +⅓ stop exposure |
|---|---|---|
| `melanin_index` | +0.0875 | −0.0754 |
| `melanin_density` | −0.0089 | +0.0027 |
| `hemoglobin_density` | +0.0346 | −0.0134 |

Roughly 10× better against shading and 28× against exposure. The residual is
uint8 clipping at the dark end of the gradient, not a limit of the method.

**This buys invariance to shading and exposure, not to white balance.** A
neutral gain cancels; a colour cast does not, because a cast moves the very
channel ratios the coordinates measure. That is the *opposite* trade from the
`_rel` family, which cancels a cast (both terms shift together) but also cancels
any face-wide pigment change — see below. The two are complementary, which is
why both are reported. Check `result.color.estimator` before trusting a
cross-session density delta.

### Why both `_rel` and the densities

`melanin_index_rel` is a region's value minus the **face-wide median**. That
cancels exposure, but it also cancels any change affecting the whole face — and
"my skin got lighter overall" is exactly the result a skincare tracker exists to
show.

Injecting a uniform melanin density across the whole face (added along the
melanin absorbance axis in optical density, which is what "more pigment
everywhere" physically means) and measuring what each metric reports:

| injected | `melanin_density` recovers | `melanin_index_rel` moves |
|---|---|---|
| 0.010 | 0.01004 | 0.00047 |
| 0.020 | 0.01946 | 0.00024 |
| 0.050 | 0.05024 | 0.00033 |

The density does not merely notice the change, it returns the injected
quantity. The `_rel` column is flat at its noise floor whatever the magnitude —
the signature of a metric that is structurally blind rather than insensitive.

Neither family is sufficient alone. Track both.

A region with fewer than `min_region_pixels` usable pixels returns NaN for
every metric rather than a number derived from noise. Treat NaN as "not
measured" — it is load-bearing, and a tracker must not plot it as zero.

### Active inflammation vs. the marks it leaves

Two chromophores, two families, same machinery:

| | melanin excess | haemoglobin excess |
|---|---|---|
| what it is | a dark mark: lentigo, freckle, post-inflammatory stain | an **active** lesion: papule, pustule |
| tracked by | `spot_burden`, `spot_contrast` | `inflammation_burden`, `inflammation_contrast` |
| records | `result.spots` (`Spot`) | `result.lesions` (`Lesion`) |
| detector | `detect_spots` | `detect_lesions` |

A melanin detector structurally cannot see active acne — acne is red and melanin
is brown, so what the spot family catches is what acne leaves *behind*. Running
both separates "healing" from "worsening", which one channel cannot do: a face
whose lesions are resolving into fresh post-inflammatory marks has falling
inflammation and *rising* spot burden at the same time.

`Spot` and `Lesion` are deliberately separate types, and `spots` and `lesions`
separate lists. They must never be summed.

Measured on the AE-locked capture sets, the inflammation family is noisier at
rest but substantially more **robust**, because haemoglobin density already has
shading and exposure projected out analytically before the local residual is
taken:

| metric | baseline noise | varied distance | varied angle |
|---|---|---|---|
| `spot_burden` | 0.00109 | 3.0× | 5.9× |
| `spot_contrast` | 0.00044 | **60.6×** | **30.3×** |
| `inflammation_burden` | 0.00196 | **2.1×** | **2.5×** |
| `inflammation_contrast` | 0.00234 | **3.6×** | **2.0×** |

Under varied distance `inflammation_contrast` ends up roughly 2.5× more reliable
than `spot_contrast` despite starting 5× noisier. Precision at rest is not
robustness in use.

**Diffuse redness is deliberately excluded.** A whole-cheek flush is smooth at
the background kernel's scale, so it lands in the background rather than the
residual — otherwise every warm room would read as acne. The diffuse component
is what `erythema_index` and `hemoglobin_density` are for.

**Caveats.** Any local vascular feature is haemoglobin excess: telangiectasia, a
healing scratch, and the vascular part of under-eye darkening all register, and
nothing here distinguishes them from acne. Thresholds are inherited from
`SpotsConfig` and have **not** been calibrated against labelled lesions — they
are a starting point, not an operating point.

### `spot_count` has an irreducible noise floor — prefer the burden metrics

Counting discrete objects near a detection boundary is a Poisson process, so the
count's relative noise is bounded below by 1/√N no matter how good the detector
is. Measured across 12 identical AE-locked captures of one face, the observed
spread tracked that bound at every stage of the filter chain:

| stage | N | observed CV | 1/√N | ratio |
|---|---|---|---|---|
| components | 891.5 | 0.034 | 0.033 | 1.02 |
| after area filter | 110.5 | 0.081 | 0.095 | 0.85 |
| after shape filter | 64.0 | 0.139 | 0.125 | 1.11 |
| after boundary filter | 50.5 | 0.206 | 0.141 | 1.46 |

At N ≈ 50 the count cannot beat ±7 spots. Tuning confirms it: hysteresis
seeding measured 0.204 against the current 0.206, and **raising the area floor
makes it worse**, because a smaller N means a larger 1/√N — CV went
0.21 → 0.30 → 0.74 as the count fell 50 → 5 → 2. There is no threshold that
fixes this, because the threshold is not what is broken.

Removing the discretisation step removes the floor:

| metric | CV over 12 identical captures |
|---|---|
| `spot_count` | 0.206 |
| `spot_area_fraction` | 0.130 |
| `spot_burden` | **0.007** |
| `spot_contrast` | **0.004** |

Sensitivity is intact — `spot_burden` spans 76× across the regions of one face
(180× its own noise floor) and 2.4× across four different faces. Stability alone
would prove nothing; a metric that never moves is repeatable and useless.

**Burden is extent, contrast is intensity.** Clinical pigmentation scales such
as MASI are likewise an area term times a darkness term, because one number
cannot separate a few dark marks from many faint ones.

Both reference the **face-wide** median and robust sigma of the residual, never
a per-region one. A per-region reference would renormalise every region to look
average and destroy the between-region differences they exist to show.

Neither needs `detect_spots` to have run — they read the melanin residual
directly, so they are finite even when `spots=None` leaves `spot_count` NaN.

`spot_count` and the individual `Spot` records are still the right output for
*showing a user where their marks are*. They are the wrong output for tracking
whether those marks are improving.

**The two do not degrade equally.** Repeatability under fixed conditions is not
robustness, and the error budget separates them:

| metric | fixed (CV) | varied distance | varied head angle |
|---|---|---|---|
| `spot_burden` | 0.007 | 0.021 (3.0×) | 0.041 (5.9×) |
| `spot_contrast` | 0.004 | 0.24 (60.6×) | 0.12 (30.3×) |
| `spot_count` | 0.206 | — | — |

`spot_burden` is the one to track: even under varied head angle it stays 5×
better than `spot_count` measured under *ideal* conditions. `spot_contrast` is
the more precise of the two when the capture is controlled and the least robust
when it is not — a percentile of a spatial residual depends on how many pixels
each mark spans, so apparent face size moves it directly. Treat it as a
fixed-distance measurement, or expect it to drift.

### Roughness is measured at a face-relative scale

The band-pass σ is a fraction of **face width**, not a fixed pixel count. A
fixed σ band-passes a different *physical* scale at every capture distance, so
it reports how close the subject stood as much as how their skin looks.
`QualityConfig.face_area_frac_band` admits a 9.2× area range — about 3× linear —
and across it, on an unchanged skin patch:

```
fixed sigma=3.0            face-relative sigma
  scale 1.00 -> 0.04732      scale 1.00 -> 0.04732
  scale 0.75 -> 0.05802      scale 0.75 -> 0.04688
  scale 0.58 -> 0.06981      scale 0.58 -> 0.04652
  scale 0.45 -> 0.08366      scale 0.45 -> 0.04606
  scale 0.33 -> 0.10307      scale 0.33 -> 0.04565
```

A 2.2× swing from distance alone, against a 3.4% residual after the fix — and
the residual is genuine detail lost to downsampling, not a scale error.
`roughness_rel` did not save this: it subtracted a face-wide std computed at the
same wrong scale.

`compute_metrics` therefore takes `face`. Without it σ falls back to the fixed
`roughness_sigma` and the distance sensitivity returns, which is why `analyze`
always passes it. The default `roughness_sigma_face_frac = 0.008` reproduces the
historical σ = 3.0 at a 375px face, the middle of the range the fixtures occupy.

`spots.py` already scaled its median kernel this way (`median_kernel_face_frac`);
roughness simply never got the same treatment.

## Comparability

Every result carries four identifiers:

```python
result.comparable_key  # (version, config_hash, weights_hash, landmarker_hash)
```

Two stored results are comparable only when all four match.

- `version` — `PREPROCESSING_VERSION`, hand-maintained, a human-readable
  changelog.
- `config_hash` — SHA-256 of the fully-resolved config. Stable across
  processes and machines: no `hash()`, no dict ordering, no `PYTHONHASHSEED`
  dependence.
- `weights_hash` — content hash of the BiSeNet checkpoint.
- `landmarker_hash` — content hash of the MediaPipe asset. Separate because it
  is a separate model with its own release cadence, and a landmarker update
  moves every region boundary, hence every per-region metric.

The three hashes are **computed from what actually ran**, so a tuned threshold
or a swapped checkpoint is detectable whether or not anyone remembered to bump
the version. That is the point: the version string is a changelog, not the
integrity mechanism.

File paths are excluded from `config_hash` — moving a checkpoint is not a
pipeline change, and its content hash covers it either way.

### Version bump rule

`PREPROCESSING_VERSION` is `MAJOR.MINOR.PATCH`.

- **MAJOR** — the result shape changed incompatibly: a metric added, removed or
  renamed; the region set changed; a field's meaning changed.
- **MINOR** — output values can shift: any threshold default, the mask model,
  an algorithm change, a new stage, a changed default in any config dataclass.
- **PATCH** — provably value-preserving: docs, type hints, refactors whose
  output is byte-identical on the fixtures.

**When in doubt, bump MINOR.** The cost of an unnecessary bump is a cosmetic
break in a chart. The cost of a missed bump is a user being told their skin
changed when only a threshold did.

Bump in the same commit as the change. `config_hash` will catch a forgotten
bump, but only after the fact, and only for someone who thinks to look.

## Determinism

Same image + same config + same weights ⇒ **byte-identical** metrics. No
randomness anywhere in the library.

`tests/test_determinism.py` asserts this by running the full pipeline twice on
each fixture and comparing metrics, masks, spots, quality and the corrected
image exactly — not approximately, because approximate comparison is exactly
what would let real drift through.

Scope of the guarantee: identical within a fixed environment. It does **not**
extend across BLAS/thread-count changes, OpenCV or torch major versions, or
CPU-vs-CUDA. Those are environment changes, not config changes, and the hashes
do not capture them. Pin your runtime alongside your weights.

## Known limitations

**Mirrored captures swap the side labels.** A mirrored photo is
indistinguishable from an unmirrored photo of a mirror-image person, so the
landmarker labels the apparent left eye as the left eye and every side label
follows. If a capture app mirrors its front camera in one session and not the
next, `left_cheek` silently changes cheeks and manufactures a longitudinal
change that never happened. **The library cannot detect this — the caller must
keep mirroring consistent.** Pinned by
`test_mirrored_capture_swaps_the_side_labels`.

**`monk_bin` is a display value, not a measurement.** There is no official
ITA→Monk mapping; this one subdivides the standard ITA classes into ten bins.
Never use `monk_bin` for fairness or bias evaluation — that audits tone
fairness against this approximation rather than against real skin tone, and the
result is circular. Fairness work needs human-assigned Monk labels or raw ITA
bands. `ita` is the source of truth; `monk_bin` is derived from it.

**Thresholds are not calibrated on real data.** The defaults are reasoned, not
fitted — they have been sanity-checked against four fixtures, which is enough
to catch an inverted comparison and nowhere near enough to set an operating
point. Calibrate on real captures from the actual device before trusting any
flag rate. In particular:

**`too_dark` conflated dark skin with underexposure — fixed in 8.0.0.** It used
to threshold mean skin L\* at 32.0, and a correctly exposed deep skin tone has a
genuinely low mean L\*. Inverting the ITA scale against published skin
colorimetry (Chardon / Del Bino) shows what that rejected:

| ITA class | b\*=12 | b\*=16 | b\*=20 |
|---|---|---|---|
| dark (<−30) | 43.1 | 40.8 | 38.5 |
| dark, deep (~−50) | 35.7 | **30.9** | **26.2** |
| dark, deepest (~−65) | **24.3** | **15.7** | **7.1** |

The floor began rejecting at ITA −42 to −56 depending on b\*, while
`monk_ita_edges` reaches −30 — **the library claimed to classify skin its own
gate refused to measure**. Two constants in the same repo disagreeing, provable
from arithmetic with no dataset at all.

`too_dark` now fires on `shadow_clipped_fraction` > 0.02, which is
tone-independent: real underexposure crushes pixels against black and destroys
information, whereas dark skin merely reflects less. The threshold sits ~10×
above the worst observed good capture (0.00199) and does catch a genuinely
underexposed clinical image measured at 0.027. `luminance_band`'s lower bound
drops to 12.0 and survives only as a no-signal backstop.

Pinned by `test_dark_skin_is_not_underexposure`.

**`blurry` had the same defect.** Laplacian variance scales with contrast, and
contrast scales with how much light skin reflects, so an absolute bar rejected
darker skin as out of focus. One unchanged photo, scaled toward deeper tones:

| scale | mean L\* | linear var | log var |
|---|---|---|---|
| 1.00 | 57.7 | 68.8 | 0.000944 |
| 0.45 | 27.9 | 15.6 | 0.000865 |
| 0.25 | 16.3 | 5.9 | 0.000813 |

An 11.7x swing against 1.16x. Focus is now measured on **log luminance**, where
a multiplicative brightness change is an additive constant the Laplacian removes
exactly — the argument `log_luminance` already made for `roughness`, which had
never been applied to the blur check. Genuine blur still separates by 13.7x
(a sigma=1 Gaussian takes it from 0.001125 to 0.000082).

Pinned by `test_darker_skin_is_not_blur` and `test_genuine_blur_is_still_caught`.

Both bugs share a root cause: **an absolute threshold on a quantity that scales
with reflected light**. Together they made deep skin tones unmeasurable however
good the capture was. Neither was found by an empirical audit — both fell out of
asking what the threshold means physically.

**This does not close the fairness question.** It fixes one threshold that was
provably wrong. Whether MediaPipe and BiSeNet detect and parse darker faces as
reliably is upstream of this library and still unmeasured — see
`tools/tone_audit.py`, and note that three runs against Fitzpatrick17k could not
answer it (21 faces from 188 images, none at Fitzpatrick I or VI). Absolute flag
rates on your own users need first-party consented captures; no public
dermatology dataset substitutes for them.

**Facial hair suppression over-removes by design.** Beard shadow reads as low
reflectance across a broad area, which is exactly what `melanin_index`
measures, so leaving it in would bias pigmentation upward in a way that looks
like a real finding. The suppressor is tuned to take some genuine skin with it
and does visibly remove pixels around the mouth on clean-shaven faces. Tune via
`ParseConfig.facial_hair_darkness_sigma` and `facial_hair_texture_min`.

**BiSeNet's `skin` class is anatomical, not facial.** A bald scalp parses as
skin. The mask is therefore clipped to the landmark face oval, and the parsed
component is selected by overlap with the face rather than by area — an orange
flight suit on one fixture parsed as a skin region several times the size of
the face and would otherwise have become the mask outright.

## Deviation from the original spec

**`possibly_filtered` condition (b) was replaced.** The spec called for
detecting texture variance that is "suspiciously uniform across regions that
normally differ — nose vs cheek". Measurement showed this runs backwards:
smoothing *raises* the nose/cheek texture ratio (1.17 → 1.5–2.0 across filters)
because cheek texture falls to the noise floor while edge-preserving filters
explicitly protect the nose's structural edges. An unfiltered fixture measured
1.17 against a 1.18 "uniform" threshold, so the check would have fired on clean
photos and stayed silent on filtered ones.

Condition (b) is now a **fine/coarse texture band ratio**. A beauty filter
removes pore-scale detail while leaving facial structure intact, so the ratio
collapses: measured 0.591 → 0.259, 0.397 → 0.191, 0.348 → 0.149 across three
fixtures under heavy bilateral filtering. Condition (a), low high-frequency
energy, is unchanged, and both must hold.

The nose/cheek ratio is still computed and reported as `texture_ratio` in
`QualityResult.measures` for inspection. Nothing keys off it.

Note that ordinary defocus also collapses the fine/coarse ratio — but a
defocused photo trips `blurry` as well, so it is already caught. The case this
check exists for is the filtered photo that stays *sharp*: bilateral smoothing
leaves Laplacian variance at 66 (above the blur threshold) while pore detail is
gone.

## Findings from real photos (0.2.0)

Three defects surfaced when the library was first run on ordinary phone photos
rather than fixtures. All are fixed; all shift stored values, hence the bump to
0.2.0.

**CIELAB was quantised to integers.** Colour metrics went through OpenCV's
8-bit Lab, which packs a\* and b\* into one byte each. On a real photo the
entire skin mask held just **17 distinct a\* values**, in steps of 1.0. Since
`erythema` is a percentile of a\*, it could only ever return an integer — a
resolution ceiling roughly **8x the measured session-to-session noise** (0.13
a\* units). Conversion now happens in float32: same photo, 799 distinct values.
Mean-based metrics were unaffected, which is why this hid for so long.

**The spot threshold was set by the artifacts it was meant to reject.** A
sigma-based threshold uses the residual's standard deviation, which hairline
shadow and stubble inflate — raising the bar and hiding the genuine
low-contrast spots underneath. Now MAD-based by default.

**The spot filters rejected real features and passed artifacts.** On a 768px
photo the relative area floor computed to 1.9px, so 3-pixel components reached
the shape tests, where eccentricity is meaningless — it rejected **59 of 102
candidates** whose median area was 6px. Meanwhile a 229px blob straddling the
hairline passed, because boundary rejection tested only the centroid, which sat
comfortably inside the mask. Now: an absolute area floor, shape tests gated on
area, and boundary rejection over the whole component.

Net effect on that photo: 2 detections (both false positives — a hairline
shadow and a stubble hair) became 6, of which roughly half are genuine marks.

### Still open: concave shadows

The remaining false positives cluster on the inner eye corner, the nasal crease
and the brow furrow — shadows that survive median-blur background subtraction.

`SpotsConfig.reject_neutral_shadows` is an **experimental, off-by-default**
filter for them. A shadow scales R, G and B by the same factor and so preserves
channel ratios; melanin absorbs more strongly at short wavelengths and does
not. Differencing a shading-invariant chromaticity (log R - log B) against the
same background therefore responds to pigment but not to shading. On one photo
the clearest shadow false positives scored z=2.13 and z=1.68 against genuine
marks at z=3.88-4.73 — and notably the melanin residual alone ranked the
*worst* false positive highest, so this is information the default pipeline
does not use.

**It is off because it is not tone-neutral as implemented.** On a dark-skinned
fixture the chroma-residual spread was 4.5x larger (robust sigma 0.0385 vs
0.0085), compressing every z-score: a z>=2.5 cut removed 2 of 3 false positives
on one face and **all eight detections** on the other. Turning it on without
validating across skin tones would silently make spot detection worse for
darker skin. Validate before enabling.

### What still limits spot detection

Resolution, more than algorithm. At 768px the face is ~330px wide and a real
blemish is a handful of pixels — below the contrast the residual threshold
needs, while hairline shading and stubble are large and high-contrast. Native
camera captures (3000px+) are needed before spot metrics mean anything, which
is what `low_resolution` exists to say.

Note also that this detects **dark** spots via melanin excess. Inflammatory
acne is *red*; a melanin-residual detector is structurally the wrong instrument
for active lesions and will largely miss them. What it can catch is
post-inflammatory hyperpigmentation, the marks left after acne heals.

Since 2.0.0 the haemoglobin map exists (`chromophore.separate_chromophores`),
which is the right instrument for active lesions — papules and pustules are
haemoglobin excess exactly as PIH is melanin excess. Detection proper is still a
later phase, but it is now the same background-subtract → threshold →
`regionprops` machinery in `spots.py` pointed at the other channel, rather than
a new signal that has to be built first.

The haemoglobin map is also the principled version of `reject_neutral_shadows`.
That filter uses log R − log B, which *is* shading-invariant — but it is an
arbitrary axis in the shading-free plane, mixing the two pigments in a ratio
that shifts with baseline tone. That is a plausible explanation for why it
removed all eight detections on the dark-skinned fixture while working on the
light-skinned one. Re-derive it against the melanin coordinate before enabling.

## Quality flags

Flags come in two tiers.

**Hard blocks** — the capture cannot be measured, so metrics are not computed
and `usable` is False: `no_face`, `multiple_faces`, `too_dark`, `too_bright`,
`blurry`, `too_far`, `too_close`, `mask_too_small`.

**Advisory** — metrics still compute, and the affected ones are named in
`quality.unreliable_metrics`: `side_lit`, `high_specular`, `possibly_filtered`,
`low_resolution`.

A single boolean would be too coarse. Specular highlights wreck roughness and
spot detection while barely touching ITA; a beauty filter destroys pores while
leaving colour roughly intact. A tracker should be able to keep a session's
erythema and discard its roughness rather than discarding the whole capture:

```python
for name, value in result.metrics.global_.items():
    if result.quality.trusted(name):
        store(name, value)
```

Both tiers are configurable via `QualityConfig.blocking_flags` and
`unreliable_metrics_by_flag`.

## Colour constancy

Two estimators with precedence, recorded in `result.color.estimator`:

1. **Sclera-referenced** (primary when confident). Eye-white is an actual
   neutral surface in the frame, so it measures the illuminant rather than
   assuming anything about the scene.
2. **Shades-of-grey, Minkowski p=6** (fallback, always computed). Assumes the
   scene averages to grey — an assumption that breaks against a strongly
   coloured wall, which is common where these photos actually get taken.

Sclera confidence combines three independent failure modes: too few pixels
(squint, closed eyes), sclera too dark (shadowed), and the two eyes disagreeing
(one lit, one shaded — the case where a one-eye estimate would be confidently
wrong). Below `sclera_min_confidence` the sclera gains are discarded and the
confidence is still reported, because a low value is diagnostic.

The illuminant is estimated from the **whole frame**, not from skin pixels.
Estimating from skin would drive mean skin colour toward grey and destroy the
erythema and melanin signal being measured. `sog_estimate_from="skin"` exists
but should not be used for that reason.

## Visual inspection

```bash
python tools/visualize.py photo.jpg -o panel.png
```

Renders a contact sheet — original, skin mask, regions with per-region pixel
counts, detected spots — plus flags and the config hash. Look at this before
trusting any number. It is how the periorbital ring was caught annexing the
glabella and the nose bridge, and how the flight-suit mask was caught: both
produced perfectly plausible numbers.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Tests skip cleanly when model assets are absent. `tests/fixtures/SOURCES.md`
documents fixture provenance and why each one is there.

## Configuration

Everything tunable lives in `skinlib/config.py` as frozen dataclasses. No
thresholds are buried in function bodies, and there is no global mutable state.

```python
from dataclasses import replace
from skinlib import Config

config = replace(base, spots=replace(base.spots, threshold_sigma=2.5))
```

Frozen means a `Config` is a value: safe to share between calls and threads,
and impossible to mutate out from under a stored result whose `config_hash`
claims otherwise.
