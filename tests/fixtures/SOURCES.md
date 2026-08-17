# Fixture provenance

All fixtures are **public domain** NASA photographs, retrieved from Wikimedia
Commons at 1920px on the long edge.

| File | Subject | Source |
|---|---|---|
| `portrait_a.jpg` | Victor Glover, official portrait 2020 (cropped) | [Commons](https://commons.wikimedia.org/wiki/File:Victor_Glover_official_portrait_2020_(cropped).jpg) |
| `portrait_b.jpg` | Jessica Watkins, official NASA portrait 2021 (cropped) | [Commons](https://commons.wikimedia.org/wiki/File:Jessica_Watkins_Official_NASA_Portrait_in_2021_(cropped).jpg) |
| `portrait_c.jpg` | Kjell Lindgren, official NASA portrait 2021 (cropped) | [Commons](https://commons.wikimedia.org/wiki/File:Kjell_Lindgren_Official_NASA_Portrait_in_2021_(cropped).jpg) |
| `portrait_d.jpg` | Kalpana Chawla, NASA portrait in orange suit | [Commons](https://commons.wikimedia.org/wiki/File:Kalpana_Chawla,_NASA_photo_portrait_in_orange_suit.jpg) |

## Why these

**Skin tone range.** A measurement library exercised only on one skin tone has
failures on other tones that no test can see. These four span roughly ITA +46
down to ITA −60 as measured by the library itself, which is what makes it
possible to notice when a threshold is calibrated for light skin only.

**Known properties, used deliberately by the tests:**

- `portrait_a` — evenly lit, frontal, visible facial hair. The primary "good
  capture" fixture, and the one the facial-hair suppression is exercised on.
- `portrait_b` — three-quarter head turn. The far cheek is genuinely occluded,
  which is what makes it a good test that regions degrade honestly rather than
  collapsing.
- `portrait_c` — dramatic side lighting. Mean skin L\* ≈ 29 despite light skin;
  the gate flags it `too_dark`, correctly.
- `portrait_d` — framed too far away (face ≈ 4.4% of frame). Exercises the
  `too_far` branch.

**Derived fixtures are not committed.** The multi-face and face-free images are
synthesised in `conftest.py` from these files and from `skimage.data`, so they
are reproducible and add no binary blobs.

## A note on using real faces as fixtures

These are public-domain photographs of public figures in their official
capacity, which is why they are appropriate to commit. Do not add photographs of
private individuals to this directory — a face is biometric data, and a test
fixture is the easiest place in a repository for it to end up permanently and
unnoticed. If you need more variety than these four provide, prefer additional
public-domain official portraits, or synthetic faces.
