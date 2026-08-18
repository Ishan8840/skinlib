"""Preprocessing version stamp.

Every :class:`skinlib.types.AnalysisResult` carries ``PREPROCESSING_VERSION``.
Longitudinal comparisons are only valid between results sharing a version.

See README "Version bump rule" for when to bump.
"""

from __future__ import annotations

# MAJOR.MINOR.PATCH
#   MAJOR  incompatible result shape (metric added/removed/renamed, region set changed)
#   MINOR  output values can shift (threshold default, mask model, algorithm change)
#   PATCH  provably value-preserving (docs, typing, refactor with byte-identical output)
# 13.0.0 Regional detection density, aggregated across the burst, as the
#        "which part of the face" answer: SessionResult.mark_density /
#        .lesion_density and .worst_regions().
#        Per-region `spot_burden` was the obvious candidate and is measured
#        WRONG for this. Against hand labels on five faces it correlates -0.16
#        with where marks actually are and picks the wrong top region on 5 of
#        5, favouring periorbital and nose — the shading and mask-boundary
#        regions — because it counts pixels over a threshold and boundaries
#        clear that threshold. Eroding the mask moves it to only +0.10 and
#        still 0 of 5, so it is not recoverable by filtering.
#        Detection density instead correlates +0.56, matches the exact region
#        2 of 5 and the coarse zone 3 of 5, on a detector scoring F1 ~0.42 per
#        spot — it works at all only because per-spot errors are near
#        independent and average out over a region, which is also why it is
#        medianed across the burst rather than taken from one frame.
#        NOT YET MEASURED: the burst aggregation itself. Every labelled face
#        available is a single image, not a burst, so the gain over single
#        frame is expected but unverified.
#        MINOR: new fields, no existing metric changes.
# 12.0.0 An absolute FLOOR under the MAD threshold, because the relative bar
#        scaled with each face's own noise: a quiet face got a low bar and the
#        detector filled it with whatever was locally unusual. Measured on two
#        labelled faces, the milder had HALF the robust sigma of the heavy one
#        and so half the threshold, admitting features at contrast 0.081 that
#        the heavy face would never have accepted. The lesion detector
#        therefore only worked when there was a lot to find.
#          floor   pooled F1     heavy P/R     mild P/R
#          0.00       0.37       0.46/0.43    0.11/0.08
#          0.07       0.42       0.46/0.43    0.36/0.33
#          0.09       0.42       0.46/0.43    0.50/0.25   <- chosen
#          0.11       0.32       0.34/0.27    0.75/0.25
#        0.09 is the MIDDLE of a flat 0.07-0.10 plateau rather than the best
#        single score, because a narrow optimum on two faces would not survive
#        a third.
#        The same floor is measured NOT to help marks and is left at 0.0 there:
#        across 53 marks on five faces it buys precision (0.44 -> 0.70) and
#        pays far more in recall (0.43 -> 0.13). Physically consistent — a
#        pigmented mark is low-contrast by nature, an inflammatory lesion is
#        high-contrast red, so an absolute bar removes real marks and only
#        noise-level lesions.
#        min_lesion_mm = 2.0 also re-confirmed on the second labelled face.
#        MAJOR: lesion counts change.
# 11.0.0 Ground truth, at last: 49 hand-labelled marks across FIVE faces and
#        37 labelled lesions, against the 6 marks on one face that set 10.0.0.
#        Two findings.
#        (1) `min_mark_mm` = 1.2 is CONFIRMED. Pooled over all five faces the
#        F1 optimum lands exactly there — it could as easily have come out at
#        0.8 or 2.0 and shown the value was fitted to one person:
#          mm    prec  rec    F1
#          0.4   0.13  0.58  0.21
#          1.0   0.25  0.48  0.33
#          1.2   0.37  0.44  0.40  <- default, confirmed
#          1.5   0.43  0.36  0.39
#          2.0   0.62  0.20  0.30
#        The F1 ceiling of 0.40 is also confirmed across five faces and four
#        people, so it is a property of the method rather than of one face.
#        (2) Lesions need their OWN floor, added as `min_lesion_mm` = 2.0. An
#        inflammatory papule is 2-5mm where a pigmented mark is 1-3mm, and the
#        first lesion ground truth shows a clean optimum well above the mark
#        threshold (F1 0.32 -> 0.44 going 1.2mm -> 2.0mm, peaking there and
#        falling to 0.36 by 3.2mm). It also separates faces better: an
#        acne-affected face against a clear one went from 4.4x the lesion count
#        to 17.5x.
#        CAVEAT: the mark threshold is pooled over five faces; the lesion
#        threshold is 37 labels on ONE. Re-check it on a second face.
#        New tools/benchmark.py scores every metric against a folder-per-class
#        labelled corpus by one-vs-rest AUC. On 4,235 public images only
#        `roughness` passes (0.756); spot_burden is chance (0.537) and
#        melanin_density, hemoglobin_density and erythema_index rank their own
#        target class LAST of five. Those are web images where class correlates
#        with camera and skin tone, so it does not prove the metrics wrong
#        under controlled capture — it proves nobody has shown them right.
#        MAJOR: lesion counts change substantially.
# 10.0.0 Spot detection: minimum mark size is stated PHYSICALLY (1.2mm,
#        converted through apparent face width) instead of as 8 pixels, and
#        the mask-boundary margin likewise. A pixel floor means ~1.6mm on a
#        close capture and ~0.4mm on a distant one, so the same setting admits
#        sub-pore noise at one distance and rejects real marks at another.
#        Scored against the first hand-labelled face (6 marks, via the new
#        tools/label.py, with tools/evaluate.py taught to read its JSON):
#          min mark    det   TP   FP   precision  recall    F1
#          0.4mm (old)  31    4   27     0.13      0.67     0.22
#          1.2mm (new)   4    2    2     0.50      0.33     0.40
#        Chosen for PRECISION over F1: a wrong circle on someone's face costs
#        more than a missed one, because it teaches the user to distrust
#        everything else on screen. That is not hypothetical — it is what the
#        old default did to the first person who looked at its output.
#        !! F1 DOES NOT EXCEED 0.40 AT ANY SETTING, swept across threshold_mad
#        !! 1.4-3.0, size 0.4-2.2mm, edge margin 0.6-3% of face width, and
#        !! working resolution 1536-3088. Every configuration lands on one
#        !! fixed trade curve; parameters slide along it and nothing moves it.
#        !! Real marks and boundary artifacts overlap in size AND contrast, so
#        !! this detector cannot separate them. `spot_count` and the Spot
#        !! records are INDICATIVE. Track `spot_burden`, which never decides
#        !! whether a single pixel is a mark and measures CV 0.007 against the
#        !! count's 0.206 from the identical residual map.
#        Also fixed in tools/evaluate.py: it found ground truth by detecting
#        saturated red, so red content in the PHOTOGRAPH became phantom labels
#        — three of them on the first real face, understating recall by a
#        third. It now reads exact coordinates, which also removes the
#        registration step entirely.
#        MAJOR: spot_count and spot_area_fraction change substantially.
# 9.0.0  Completes the sweep for absolute thresholds on brightness-dependent
#        quantities — the defect pattern behind `too_dark` (8.0.0) and
#        `blurry`. Every remaining QualityConfig threshold was measured against
#        one unchanged photo scaled toward deeper tones:
#          specular_fraction  0.00628 -> 0.00000   BROKEN, fixed below
#          hf_energy                     1.10x     acceptable
#          side_lit_ratio                1.08x     tone-independent
#          scale_ratio / texture_ratio   1.01x     clean
#        So `side_lit` firing on 12/12 of the baseline captures is NOT a
#        fairness defect; that lighting is genuinely directional.
#        Specular detection was the worst of the three: `V >= 0.92` did not
#        shift on darker skin, it stopped working entirely, because the
#        brightest channel never reaches 0.92 once median V is 0.59. Shine was
#        undetectable on anything but light skin. Now `V > 1.25 * median(V over
#        skin)`, since a highlight is bright RELATIVE to the diffuse level
#        around it; 1.25 reproduces the old measure on the unchanged image
#        (0.00688 vs 0.00628) and holds within 1.46x across the range.
#        `specular_v_min` drops to 0.15, a backstop against quantisation noise.
#        Verified: 0 of 56 real captures flagged high_specular.
#        `too_bright` gets the fix `too_dark` got in 8.0.0. The argument was
#        never one-sided: mean L* rises with skin LIGHTNESS as much as with
#        exposure, so the L* > 82 ceiling was wrong in both directions at once.
#        It would reject correctly exposed very light skin (ITA > 55 reaches
#        L* 84.6 at b* = 20) while staying silent on a capture measured with
#        17% of the skin already blown out. Now keys on
#        `highlight_clipped_fraction` > 0.02, mirroring `shadow_clipped_max`
#        and ~5x above the worst observed good capture (0.00406, `flash` set).
#        `luminance_band` becomes (12.0, 95.0), both bounds backstops only.
#        Verified: 0 of 56 real captures newly flagged.
#        `spot_area_fraction` divided two different pixel populations — the
#        numerator summed components found on the RAW skin mask, the
#        denominator excluded clipped pixels, biasing the fraction upward
#        exactly where the capture is worst. Both are now the raw mask.
#        `pixel_counts` still reports what was measured (unclipped).
#        MAJOR: flag semantics changed, spot_area_fraction values shift.
# 8.0.0  `too_dark` now fires on INFORMATION LOSS rather than on darkness.
#        The old gate thresholded mean skin L* at 32.0, which rejected
#        correctly exposed deep skin by construction. Inverting the ITA scale
#        against published skin colorimetry (Chardon/Del Bino), mean L* of
#        correctly exposed skin is:
#          ITA class        b*=12  b*=16  b*=20
#          dark (<-30)       43.1   40.8   38.5
#          dark deep (~-50)  35.7   30.9   26.2   <- partly rejected
#          dark deepest      24.3   15.7    7.1   <- always rejected
#        so the floor began rejecting at ITA -42 to -56 depending on b*, while
#        `monk_ita_edges` reaches -30: the library claimed to CLASSIFY skin its
#        own gate refused to MEASURE. Two constants in one repo disagreeing —
#        provable without any dataset, which three separate empirical audits
#        against Fitzpatrick17k could not manage (21 faces from 188 images, and
#        zero at both Fitzpatrick I and VI).
#        `too_dark` now keys on `shadow_clipped_fraction` > 0.02, which is
#        tone-independent: real underexposure crushes pixels against black and
#        destroys information, dark skin merely reflects less. 0.02 is ~10x the
#        worst observed good capture (0.00199) and does catch the genuinely
#        dark clinical image measured at 0.027. `luminance_band` drops to
#        (12.0, 82.0) and its lower bound is now a no-signal backstop only.
#        `blurry` had the SAME defect and is fixed the same way. Laplacian
#        variance scales with contrast and contrast scales with reflected
#        light, so an absolute bar on it rejected darker skin as out of focus.
#        On one unchanged photo scaled toward deeper tones the linear measure
#        swung 11.7x (68.8 -> 5.9) while a log-domain measure moved 1.16x
#        (0.000944 -> 0.000813). Focus is now the variance of the Laplacian of
#        LOG luminance — the same argument `log_luminance` already made for
#        `roughness`, never applied here — with a threshold of 0.0004: 2.2x
#        below the worst good capture across all five sets (0.00087) and 4.9x
#        above a sigma=1 blur (0.000082). Genuine blur still separates by 13.7x.
#        The linear measure stays in QualityResult.measures for inspection.
#        Also: `analyze_session` computed lesions and discarded them; the
#        records are now returned on `SessionResult.spots`/`.lesions` from the
#        reference frame, for display rather than tracking.
#        MAJOR: flag semantics changed; SessionResult gained fields.
# 7.0.0  Regions are cut in the face's own 3D frame (new canonical.py) instead
#        of the 2D image-plane projection, so a threshold stops sliding across
#        anatomy when the head turns. Every per-region metric shifts.
#        Measured on the `angle` set against `constant`, median per-region
#        penalty: spot_burden 4.63x -> 3.87x, inflammation_burden 4.58x ->
#        3.79x, roughness 4.33x -> 3.61x, uniformity 5.46x -> 5.04x,
#        melanin_density 6.19x -> 6.40x (no gain).
#        Region drift is therefore real but explains only ~20% of the angle
#        penalty. The residual 3.87x is most likely irreducible: a turned head
#        shows different skin at a different sampling density, which no frame
#        recovers. Costs ~50ms; total region area moves 1%.
#        MAJOR: per-region values change.
# 6.0.0  Surface geometry and derived quantities. `Face.landmarks_z` retains
#        the depth MediaPipe always returned and the library used to discard;
#        new `geometry.py` builds a depth surface, per-pixel normals, and a
#        least-squares light direction. New `derived.py` adds left/right
#        asymmetry and a periorbital decomposition splitting under-eye darkness
#        into pigment, vascular and structural parts.
#        Validated: normals are unit vectors, forehead reads cos 0.95 against
#        the chin's 0.63 (a plane against the jawline), and the light solver
#        recovers a synthetic Lambertian illuminant to r^2 > 0.98.
#        NEGATIVE RESULT, recorded in geometry.py: incidence is a real covariate
#        of the per-region metrics (mean |r| 0.51 for spot_burden across the
#        angle set) but NOT a correctable one — its sign is inconsistent between
#        regions (forehead +0.50, nose -0.85, perioral +0.61, periorbital_left
#        -0.80), so no single divisor can help. The likeliest cause is that the
#        angle penalty is region-mask drift under out-of-plane rotation rather
#        than shading, which would make 3D canonical region definition the real
#        fix. Incidence is therefore offered as a weight, never a divisor.
#        Also a pure-performance pass, provably value-preserving: `analyze`
#        builds each expensive map once instead of letting the spot detector,
#        the lesion detector and the burden metrics each rebuild it. The
#        chromophore separation ran 3x per frame and the large-kernel median 4x;
#        `_valid_pixels` rescanned the full frame on all ten region calls.
#        analyze() 6303ms -> 3102ms, suite 155s -> 90s, output bit-identical.
#        MAJOR: Face gained a field and new public modules landed; no metric
#        column changed.
# 5.0.0  Burst analysis: `analyze_session` plus `SessionResult`, `FrameReport`
#        and `SessionConfig`. Metrics are unchanged; this adds a second entry
#        point beside `analyze`, so stored single-frame results stay valid.
#        Ten frames over two seconds feel like one photo and buy averaging,
#        frame selection, and a per-session error bar measured on THIS capture
#        rather than assumed. Measured on the 12-frame `constant` burst:
#          detectable change   1 frame -> 10-frame session
#          spot_burden         0.00302 -> 0.00102
#          spot_count            28.7  ->   11.4
#        Frame selection is relative to the burst, never absolute: Laplacian
#        variance depends on how much detail a face has. One illuminant is
#        estimated for the whole burst from the frame with the most confident
#        sclera, since re-estimating per frame injects the estimator's own
#        noise into every colour metric.
#        Specular recovery from between-frame variation is present but OFF and
#        measured NOT to work — it tracks edge gradient (r = +0.551) rather
#        than luminance (r = -0.221, wrong sign), i.e. registration residual at
#        the lash line rather than glare. Gating out edges removes the artifact
#        and leaves no signal (T-zone/cheek 1.27x -> 1.08x). See SessionConfig.
#        MAJOR: new public result type; no metric column changed.
# 4.0.0  ACTIVE inflammation, from local haemoglobin excess. Added
#        `inflammation_burden` and `inflammation_contrast`, plus
#        `detect_lesions` returning the new `Lesion` type and
#        `AnalysisResult.lesions`. The melanin family sees a healed mark; this
#        one sees the lesion itself, which acne being red and melanin brown
#        made structurally impossible before.
#        Measured on the AE-locked sets, the inflammation family is markedly
#        more ROBUST than the melanin one even though it is noisier at rest,
#        because haemoglobin density has shading and exposure projected out
#        analytically before the local residual is taken:
#          baseline noise / distance / angle
#          spot_contrast          0.00044 | 60.6x | 30.3x
#          inflammation_contrast  0.00234 |  3.6x |  2.0x
#          spot_burden            0.00109 |  3.0x |  5.9x
#          inflammation_burden    0.00196 |  2.1x |  2.5x
#        `detect_spots` was refactored onto the shared `_components` extractor
#        it now uses with `detect_lesions`; spot output is unchanged and pinned
#        by test_refactor_preserved_spot_detection.
#        MAJOR: columns added, AnalysisResult gained a field.
# 3.0.0  Added `spot_burden` and `spot_contrast`, continuous replacements for
#        `spot_count`. Measured across 12 identical AE-locked captures, the
#        count's noise IS counting statistics: observed CV tracked 1/sqrt(N) at
#        every filter stage (ratios 1.02, 0.85, 1.11, 1.46), so at N=50 it
#        cannot beat +-7 spots however good the detector. Tuning confirmed it —
#        hysteresis seeding gave 0.204 against 0.206, and raising the area floor
#        made it worse as N fell (0.21 -> 0.30 -> 0.74 at N = 7 -> 5 -> 2).
#        Dropping the discretisation removes the floor entirely:
#          spot_count 0.206 | spot_area_fraction 0.130
#          spot_burden 0.007 | spot_contrast 0.004
#        Sensitivity holds — burden spans 76x across regions of one face and
#        2.4x across four faces. Both read the melanin residual directly, so
#        unlike spot_count they do not require the detector to have run.
#        MAJOR: columns added.
# 2.0.0  Roughness band-pass sigma scales with face width instead of being a
#        fixed 3.0px. A fixed sigma band-passes a different physical scale at
#        every capture distance: measured across the allowed distance band on
#        an unchanged skin patch it swung roughness 0.0473 -> 0.1031 (2.2x),
#        which a tracker reported as rougher skin when the subject had only
#        stood closer. Face-relative holds the same patch to 3.4%. Every
#        stored `roughness` and `roughness_rel` changes.
#        Added `melanin_density` and `hemoglobin_density`: melanin/haemoglobin
#        coordinates in optical-density space with the (1,1,1) shading axis
#        projected out analytically (Tsumura). Against a shading gradient and
#        a +1/3 stop exposure change they move -0.0089 / +0.0027 where
#        `melanin_index` moves +0.0875 / -0.0754. Unlike the _rel family they
#        can see a face-wide pigment shift, which self-normalisation cancels
#        by construction. MAJOR: columns added, roughness changed value.
# 1.0.0  Log-domain erythema (log10 R/G) and roughness (band-pass of log
#        luminance), so exposure cancels rather than approximately cancelling.
#        Added self-normalised <metric>_rel columns and the D(x,y) map.
#        Absolute tone metrics marked internal-only. Sclera confidence
#        recalibrated against observed pixel counts. MAJOR: columns added,
#        roughness changed meaning.
# 0.3.0  Spot threshold default 3.0 -> 2.2, calibrated against a hand-labelled
#        face (recall 0.33 -> 0.78, F1 0.40 -> 0.78). See tools/evaluate.py.
# 0.2.0  Colour metrics moved to float32 CIELAB (8-bit Lab quantised a* to
#        integer steps, capping `erythema` resolution at ~1 unit against a
#        measured noise floor of 0.13). Spot detection: MAD-based threshold,
#        absolute minimum area, shape tests gated on area, whole-component
#        boundary rejection. All of these shift stored values.
PREPROCESSING_VERSION: str = "13.0.0"
