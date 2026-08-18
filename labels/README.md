# Hand labels

Ground truth for the spot and lesion detectors, produced with `tools/label.py`
and scored with `tools/evaluate.py`.

| file | face | px/mm | marks | lesions |
|---|---|---|---|---|
| `personA.labels.json` | subject A | 5.1 | 6 | — |
| `personB.labels.json` | subject B | 5.0 | 1 | — |
| `dark_spots_2.labels.json` | public dataset | 8.0 | 23 | — |
| `dark_spots_hi2.labels.json` | public dataset | 10.7 | 13 | — |
| `dark_spots_hi3.labels.json` | public dataset | 9.1 | 6 | — |
| `inflammatory_acne_hi3.labels.json` | public dataset | 8.2 | 1 | 37 |

**49 marks across five faces; 37 lesions on one.** Coordinates are in the source
image's own pixels, so `evaluate.py` scales rather than registers them.

These set `min_mark_mm` (confirmed at 1.2) and `min_lesion_mm` (2.0, and still
resting on a single face). Before changing either, re-run:

```bash
python tools/evaluate.py <photo> labels/<name>.labels.json --sweep
python tools/evaluate.py <photo> labels/<name>.labels.json --kind lesion --sweep
```

The images themselves are not committed: four are public-dataset photographs
under their own terms, and two are photographs of people. The labels are
coordinates, which carry no likeness.
