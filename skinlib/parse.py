"""Face parsing -> binary skin mask.

BiSeNet trained on CelebAMask-HQ supplies the class map; everything after that
is deterministic morphology. No model is trained here — see README for where to
get the checkpoint.

The mask defines where every metric may read. Two things the parser cannot do
are handled explicitly afterwards:

* **Nostrils** fall inside the ``nose`` class, so they are carved out from
  landmark geometry intersected with a darkness test.
* **Facial hair** has no class at all. It is suppressed in the lower face by a
  darkness-and-texture test, tuned to over-suppress: losing some real chin skin
  costs less than measuring beard shadow as pigmentation.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from . import regions as regions_module
from .config import Config, ParseConfig
from .detect import AssetNotFoundError, file_hash
from .types import Face

__all__ = ["load_parser", "parse_skin", "resolve_weights", "weights_hash"]

_ENV_WEIGHTS = "SKINLIB_BISENET_WEIGHTS"
N_CLASSES = 19

# ImageNet statistics, as used when the checkpoint was trained. Changing these
# changes the class map, so they are a property of the checkpoint rather than a
# tunable.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_weights(config: ParseConfig) -> Path:
    """Locate the BiSeNet checkpoint.

    Order: explicit config, then ``$SKINLIB_BISENET_WEIGHTS``. Never downloads:
    fetching a checkpoint at analysis time could silently swap the mask model
    between two sessions of a longitudinal series.
    """
    candidate = config.weights_path
    if candidate is None:
        env = os.environ.get(_ENV_WEIGHTS)
        candidate = Path(env) if env else None
    if candidate is None:
        raise AssetNotFoundError(
            "BiSeNet weights not configured. Set ParseConfig.weights_path or "
            f"${_ENV_WEIGHTS}. See README 'Model assets' for the download."
        )
    candidate = Path(candidate)
    if not candidate.is_file():
        raise AssetNotFoundError(f"BiSeNet weights not found at {candidate}")
    return candidate


def weights_hash(config: ParseConfig) -> str:
    """Content hash of the checkpoint, for the result's comparability key."""
    return file_hash(resolve_weights(config))


def load_parser(config: Config | None = None):
    """Load BiSeNet in eval mode.

    Returned explicitly rather than cached in a module global: a cached model
    would be shared mutable state. Callers analysing many images should hoist
    this call and pass the result into ``parse_skin``.
    """
    import torch

    from ._bisenet import BiSeNet

    config = config or Config()
    path = resolve_weights(config.parse)

    model = BiSeNet(n_classes=N_CLASSES)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    # Surfaced loudly: a checkpoint that only half-loads produces a plausible
    # but wrong mask, which is far worse than a crash.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"BiSeNet checkpoint does not match the architecture: "
            f"{len(missing)} missing and {len(unexpected)} unexpected keys. "
            f"First missing: {list(missing)[:3]}; first unexpected: {list(unexpected)[:3]}"
        )
    model.eval()
    model.to(config.parse.device)
    return model


def _class_map(image: np.ndarray, model, config: ParseConfig) -> np.ndarray:
    """Run BiSeNet and return a per-pixel class id map at the image's size."""
    import torch

    height, width = image.shape[:2]
    size = config.input_size

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # INTER_AREA when shrinking, INTER_LINEAR when growing: pinned so the class
    # map does not depend on how the caller happened to size the input.
    shrinking = size < max(height, width)
    resized = cv2.resize(
        rgb,
        (size, size),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )

    normalised = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    tensor = torch.from_numpy(normalised.transpose(2, 0, 1)[None]).to(config.device)

    with torch.no_grad():
        logits = model(tensor)
    labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Nearest neighbour: class ids are categorical, and interpolating them
    # would invent classes that lie between two real ones.
    return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)


def _face_component(mask: np.ndarray, face: Face) -> np.ndarray:
    """Keep the connected region belonging to the detected face.

    Selected by overlap with the landmark hull, NOT by area. Area alone picks
    the wrong blob whenever something larger than the face gets labelled skin —
    bare arms, a hand, or (observed on a fixture) an orange flight suit, which
    parsed as one skin region several times the size of the face and would
    otherwise have become the mask outright.
    """
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 2:  # background plus at most one region
        return mask

    hull = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(
        hull, cv2.convexHull(np.round(face.landmarks).astype(np.int32)), 1
    )
    inside = hull.astype(bool)

    # Row 0 of the label set is background; count each component's pixels that
    # fall inside the face hull.
    overlaps = np.bincount(labels[inside].ravel(), minlength=count)
    overlaps[0] = 0
    best = int(np.argmax(overlaps))
    if overlaps[best] == 0:
        # Nothing intersects the face at all. Returning an empty mask lets the
        # quality gate raise `mask_too_small` rather than silently measuring
        # whatever else happened to be labelled skin.
        return np.zeros_like(mask)
    return labels == best


def _carve_nostrils(
    mask: np.ndarray, image: np.ndarray, face: Face, config: Config
) -> np.ndarray:
    """Remove the nostril apertures.

    Landmark geometry alone would take a fixed blob of ala skin with it, so the
    bracket polygon is intersected with a darkness test and only the genuinely
    dark aperture is removed.
    """
    bracket = regions_module.nostril_polygons(face, config) & mask
    if not bracket.any():
        return mask

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = grey[bracket]
    # Otsu on the bracket alone: it contains exactly the two populations we are
    # separating (lit ala skin, unlit aperture), which is the situation Otsu is
    # actually valid for.
    threshold, _ = cv2.threshold(
        inside.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    aperture = bracket & (grey <= threshold)
    aperture = regions_module.dilate(aperture, config.parse.nostril_dilate_px)
    return mask & ~aperture


def _suppress_facial_hair(
    mask: np.ndarray, image: np.ndarray, face: Face, config: Config
) -> np.ndarray:
    """Remove beard and stubble from the lower face.

    Deliberately aggressive. Beard shadow reads as low reflectance across a
    broad area, which is exactly what ``melanin_index`` measures, so leaving it
    in would not merely add noise — it would bias pigmentation upward in a way
    that looks like a real finding.
    """
    parse_config = config.parse
    polygons = regions_module.build_region_polygons(face, config)
    area = np.zeros(mask.shape, dtype=bool)
    for name in parse_config.facial_hair_regions:
        area |= polygons[name]
    area &= mask
    if area.sum() < 100:
        return mask

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)

    values = lightness[area]
    median = float(np.median(values))
    # Median absolute deviation, not std: beard pixels are exactly the outliers
    # being detected, and they would inflate a standard deviation enough to
    # hide themselves.
    mad = float(np.median(np.abs(values - median)))
    spread = max(mad * 1.4826, 1e-3)
    dark = lightness < (median - parse_config.facial_hair_darkness_sigma * spread)

    # Local relative contrast: stubble is high-frequency, a cast shadow is not.
    blurred = cv2.GaussianBlur(lightness, (0, 0), 2.0)
    local_var = cv2.GaussianBlur((lightness - blurred) ** 2, (0, 0), 2.0)
    texture = np.sqrt(np.maximum(local_var, 0.0)) / np.maximum(blurred, 1.0)
    textured = texture > parse_config.facial_hair_texture_min

    hair_like = area & dark & textured
    if not hair_like.any():
        return mask
    # Grown to catch the soft shadow halo around each stubble cluster, which is
    # darkened skin rather than hair and would otherwise survive the test.
    hair_like = regions_module.dilate(hair_like, parse_config.facial_hair_dilate_px)
    return mask & ~(hair_like & area)


def parse_skin(
    image: np.ndarray,
    face: Face,
    config: Config | None = None,
    parser=None,
) -> np.ndarray:
    """Binary mask of facial skin.

    Excludes eyes, brows, lips, hair, nostrils, glasses, facial hair and
    background. Every metric is computed inside this mask and nowhere else.

    Pass ``parser`` (from ``load_parser``) to avoid reloading the checkpoint per
    image.
    """
    config = config or Config()
    parse_config = config.parse

    if image.shape[:2] != face.image_size:
        raise ValueError(
            f"image shape {image.shape[:2]} does not match the detected face's "
            f"image size {face.image_size}; landmarks would not align"
        )

    model = parser if parser is not None else load_parser(config)
    labels = _class_map(image, model, parse_config)

    skin = np.isin(labels, np.asarray(parse_config.skin_classes))
    excluded = np.isin(labels, np.asarray(parse_config.exclude_classes))
    # The exclusions are grown before subtraction: class boundaries are soft,
    # and a lash or lip pixel misfiled as skin sits right at the edge.
    excluded = regions_module.dilate(excluded, parse_config.exclude_dilate_px)

    mask = skin & ~excluded

    if parse_config.carve_landmark_features:
        # Landmark geometry backs up the class map at the eye, brow and lip
        # boundaries, where BiSeNet is soft and a stray sclera or lip pixel
        # would be a large outlier in both a* and melanin index.
        features = regions_module.feature_polygons(face)
        mask &= ~regions_module.dilate(features, parse_config.feature_dilate_px)

    if parse_config.limit_to_face_oval:
        # `skin` is anatomical, not facial: a bald scalp parses as skin and sat
        # well above the face on a fixture, so global metrics would have been
        # measuring head, not face.
        oval = regions_module.face_oval_polygon(face)
        mask &= regions_module.dilate(oval, parse_config.face_oval_dilate_px)

    # Pull off the silhouette, where background bleeds into the skin class.
    mask = regions_module.erode(mask, parse_config.skin_erode_px)
    mask = _face_component(mask, face)

    if parse_config.suppress_nostrils:
        mask = _carve_nostrils(mask, image, face, config)
    if parse_config.suppress_facial_hair:
        mask = _suppress_facial_hair(mask, image, face, config)

    return np.ascontiguousarray(mask)
