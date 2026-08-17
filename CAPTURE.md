# Capture protocol

A calibration shoot for `tools/repeatability.py`. Roughly one hour, ~70 photos.

The point is not to take good photos. It is to vary **one thing at a time** so
that when a metric moves, you can say what moved it. The first attempt at this
varied distance, illumination and camera metering together, and the result was
uninterpretable: `melanin_density` correlated 0.78 with face width and there was
no way to tell whether that was a scale bug, an exposure effect, or the camera
re-metering.

Set A is the ruler. Every other set is read as a multiple of it.

## Before you start

- **Settings → Camera → Formats → Most Compatible.** `cv2.imread` cannot decode
  HEIC, so a High Efficiency capture fails at load.
- **Settings → Camera → Mirror Front Camera** — note whether it is on, and do
  not change it for the rest of the shoot, or ever. A mirrored capture is
  indistinguishable from an unmirrored photo of a mirror-image person, so the
  side labels silently swap and `left_cheek` becomes the other cheek. The
  library cannot detect this. See README, "Known limitations".
- **Live Photo off.**
- **One lamp. Curtains closed.** Daylight drifts measurably over an hour, and
  set D is the only place illumination is allowed to vary.
- **Tape two marks on the floor** — one at arm's length, one about 1.5m.
- Clean face. No makeup, no product applied during the shoot. All sets in one
  sitting: the skin must be the constant.

**AE/AF lock**: tap and hold on your face until `AE/AF LOCK` appears. This
freezes exposure *and* white balance. Sets A–D and F use it; E deliberately does
not.

## The sets

Each is 10 shots. Front camera throughout — selfies are the product, so
calibrating on the rear camera would calibrate the wrong instrument.

### A — baseline · `A_baseline/`
AE locked, near mark, one lamp. Take 10 shots without moving at all.

Gives the **instrument noise floor**: with the skin, pose and light all constant,
everything that moves is measurement error. This is the best any metric can ever
do, and the number every other set is compared against.

### B — distance · `B_distance/`
AE locked, pose and lighting fixed. Vary only distance, near mark → far mark,
roughly two shots at each of five distances.

Catches **scale bugs** — a metric whose value depends on how close you stood.
This is the set that caught the fixed-σ roughness bug, back when it was
accidental.

### C — pose · `C_pose/`
AE locked, near mark, lighting fixed. Vary head angle ±20° yaw and pitch, plus a
couple of expressions. Face stays fully in frame.

Gives **geometry sensitivity**: how much the region masks and the shading model
care about not facing straight ahead.

### D — lighting · `D_lighting/`
Fixed distance and pose. Move the lamp — front, 45°, side, overhead — and try a
second room. About three shots per condition.

**Re-lock AE after each lamp move**, so every shot is correctly exposed under its
own illuminant. The variable under test is the illuminant's colour and
direction, not exposure.

Measures **colour constancy quality**: how well the sclera and shades-of-grey
estimators actually hold pigment metrics steady.

### E — natural · `E_natural/`
**AE unlocked.** No marks, no lamp discipline. Shoot the way you would if you
were just using the app — any room, any time, however you naturally hold it.

This is the **real-world total error**, and the only number that describes what a
user actually gets. Everything above exists to explain why this one is as large
as it is.

### F — flash · `F_flash/`
AE locked, fixed distance, flash ON.

Settles empirically whether flash helps. The reasoning against it is that the
iPhone 11 front camera has no LED — "flash" is the screen going white, which is
weak, broad, and coloured by whatever is on it — and that flash falls off as
1/d², reintroducing a distance-brightness coupling. If F's noise floor beats A's,
that reasoning is wrong and flash-dominant capture is the protocol.

### G — flash pairs · `G_pairs/` *(optional)*
10 back-to-back pairs at fixed distance: one flash, one without, same framing.

Only needed if you want **oiliness**. Differencing a flash and no-flash frame
isolates the specular component, which is the principled way to measure sebum —
as opposed to the current specular fraction, which measures the room's lighting
geometry as much as the skin and is therefore only a quality flag.

## Transfer

USB, and nothing else. Messaging apps, email and browser uploads recompress:
the first attempt arrived as 640px WebP at 0.07 bytes/px, against 3088×2316 at
0.35 bytes/px from the same shots over a cable.

```bash
idevicepair pair                      # tap Trust on the phone
mkdir -p ~/iphone && ifuse ~/iphone
cp ~/iphone/DCIM/*/IMG_*.JPG ~/Pictures/skin/<set>/
fusermount -u ~/iphone
```

Verify before analysing anything — you want ~3088×2316 and ≥0.3 bytes/px:

```bash
python - <<'EOF'
import cv2, glob, os
for f in sorted(glob.glob('*.JPG')):
    im = cv2.imread(f); h, w = im.shape[:2]
    print(f'{f}  {w}x{h}  {os.path.getsize(f)/(w*h):.2f} bytes/px')
EOF
```

## Then

```bash
python tools/repeatability.py ~/Pictures/skin/A_baseline
```

per set, and compare each against A. A metric whose noise in E is close to its
noise in A is genuinely robust. One that blows up in D has a colour constancy
problem; one that blows up in B has a scale problem; one that blows up in C has
a geometry problem. That attribution is the entire purpose of shooting five sets
instead of one.
