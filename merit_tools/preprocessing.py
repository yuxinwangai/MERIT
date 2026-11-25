from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .image_ops import window_hu_to_uint8
from .volume_io import Volume, canonicalize, resample_to_grid


@dataclass(frozen=True)
class PreprocessingConfig:
    retained_fraction: float = 0.30
    bone_threshold_hu: float = 150.0
    soft_tissue_min_hu: float = -20.0
    soft_tissue_max_hu: float = 120.0
    bone_closing_radius: int = 5
    final_closing_radius: int = 3
    minimum_component_pixels: int = 1000
    cta_window_center: float = 200.0
    cta_window_width: float = 600.0
    ncct_window_center: float = 40.0
    ncct_window_width: float = 80.0
    tile_size: int = 512
    inplane_spacing_mm: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.retained_fraction <= 1:
            raise ValueError("retained_fraction must be in (0, 1]")
        if self.minimum_component_pixels < 1 or self.tile_size < 1:
            raise ValueError("Component size and tile size must be positive")
        if self.bone_closing_radius < 0 or self.final_closing_radius < 0:
            raise ValueError("Closing radii must be nonnegative")
        if self.soft_tissue_min_hu >= self.soft_tissue_max_hu:
            raise ValueError("Invalid soft-tissue HU interval")
        if self.cta_window_width <= 0 or self.ncct_window_width <= 0:
            raise ValueError("Window widths must be positive")
        if self.inplane_spacing_mm <= 0:
            raise ValueError("In-plane spacing must be positive")
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Preprocessing parameters must be finite")

    @classmethod
    def load(cls, path: str | Path) -> PreprocessingConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def align_inputs(cta: Volume, ncct: Volume, ncct_to_cta: np.ndarray | None = None) -> tuple[Volume, Volume]:
    if ncct_to_cta is not None:
        transform = np.asarray(ncct_to_cta, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Expected a finite four-by-four NCCT-to-CTA transform")
        rotation = transform[:3, :3]
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Invalid rigid-transform last row")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Expected a rigid NCCT-to-CTA transform")
        ncct = Volume(ncct.data, transform @ ncct.affine)
    aligned_cta = canonicalize(cta)
    aligned_ncct = resample_to_grid(ncct, aligned_cta.data.shape, aligned_cta.affine)
    return aligned_cta, aligned_ncct


def disk(radius: int) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    return x*x + y*y <= radius*radius


def standardize_inplane(cta: Volume, ncct: Volume, config: PreprocessingConfig) -> tuple[Volume, Volume]:
    if cta.data.shape != ncct.data.shape or not np.allclose(cta.affine, ncct.affine):
        raise ValueError("CTA and NCCT must share a canonical grid")
    if not np.allclose(cta.affine[:3, :3], np.diag(np.diag(cta.affine[:3, :3])), atol=1e-5):
        raise ValueError("Expected an axis-aligned canonical grid")
    support = np.zeros(cta.data.shape[:2], dtype=bool)
    for index in range(ncct.data.shape[2]):
        support |= intracranial_mask(ncct.data[:, :, index], config)
    points = np.argwhere(support)
    if not len(points):
        raise ValueError("No intracranial support for physical centering")
    lower, upper = points.min(axis=0), points.max(axis=0)
    spacing = np.diag(cta.affine)[:2]
    extent = (upper - lower + 1) * spacing
    field = config.tile_size * config.inplane_spacing_mm
    if np.any(extent > field - 2 * config.inplane_spacing_mm):
        raise ValueError("Intracranial support exceeds the configured physical field of view")
    center = cta.affine @ [*(0.5 * (lower + upper)), 0, 1]
    affine = cta.affine.copy()
    affine[0, 0] = affine[1, 1] = config.inplane_spacing_mm
    affine[:2, 3] = center[:2] - (config.tile_size - 1) * config.inplane_spacing_mm / 2
    shape = (config.tile_size, config.tile_size, cta.data.shape[2])
    return resample_to_grid(cta, shape, affine), resample_to_grid(ncct, shape, affine)


def display_geometry(volume: Volume, tile_size: int) -> dict:
    nx, ny = volume.data.shape[:2]
    side = max(nx, ny)
    scale = side / tile_size
    mapping = np.eye(4)
    mapping[0, 0] = scale
    mapping[1, 1] = -scale
    mapping[0, 3] = (scale - 1) / 2 - (side - nx) // 2
    mapping[1, 3] = ny - 1 + (side - ny) // 2 - (scale - 1) / 2
    affine = volume.affine @ mapping
    return {
        "coordinate_order": ["column", "row", "original_axial_index", "homogeneous"],
        "display_to_world_mm": affine.tolist(),
        "display_pixel_spacing_mm": np.linalg.norm(affine[:3, :2], axis=0).tolist(),
        "field_of_view_mm": (np.linalg.norm(affine[:3, :2], axis=0) * tile_size).tolist(),
        "display_size": [tile_size, tile_size],
    }


def intracranial_mask(image_hu: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    from scipy import ndimage

    finite = np.isfinite(image_hu)
    bone = finite & (image_hu > config.bone_threshold_hu)
    bone = ndimage.binary_closing(bone, structure=disk(config.bone_closing_radius))
    envelope = ndimage.binary_fill_holes(bone)
    soft = (
        envelope
        & finite
        & (image_hu >= config.soft_tissue_min_hu)
        & (image_hu <= config.soft_tissue_max_hu)
    )
    labels, count = ndimage.label(soft, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros(image_hu.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    eligible = np.flatnonzero(sizes >= config.minimum_component_pixels)
    eligible = eligible[eligible != 0]
    if not len(eligible):
        return np.zeros(image_hu.shape, dtype=bool)
    centers = np.asarray(ndimage.center_of_mass(soft, labels, eligible))
    center = (np.asarray(image_hu.shape) - 1) / 2
    distance = np.linalg.norm(centers - center, axis=1)
    central = eligible[distance <= 0.35 * min(image_hu.shape)]
    if not len(central):
        return np.zeros(image_hu.shape, dtype=bool)
    selected = int(central[np.argmax(sizes[central])])
    mask = ndimage.binary_closing(labels == selected, structure=disk(config.final_closing_radius))
    return ndimage.binary_fill_holes(mask)


def select_levels(ncct: Volume, config: PreprocessingConfig) -> tuple[np.ndarray, np.ndarray, list[int]]:
    masks = np.zeros(ncct.data.shape, dtype=bool)
    ratios = np.zeros(ncct.data.shape[2], dtype=float)
    for index in range(ncct.data.shape[2]):
        mask = intracranial_mask(ncct.data[:, :, index], config)
        masks[:, :, index] = mask
        ratios[index] = mask.mean()
    centres = select_targets(ratios, config.retained_fraction)
    return masks, ratios, centres


def select_targets(ratios: np.ndarray, fraction: float = 0.30) -> list[int]:
    ratios = np.asarray(ratios, dtype=float)
    if ratios.ndim != 1 or not np.isfinite(ratios).all() or np.any((ratios < 0) | (ratios > 1)):
        raise ValueError("Expected finite brain-content ratios in [0, 1]")
    if not 0 < fraction <= 1:
        raise ValueError("Retained fraction must be in (0, 1]")
    count = max(1, math.ceil(len(ratios) * fraction))
    ranking = np.lexsort((np.arange(len(ratios)), -ratios))
    targets = sorted(int(i) for i in ranking[:count] if ratios[i] > 0)
    if not targets:
        raise ValueError("No positive brain-content target levels")
    return targets


def target_context_blocks(targets: list[int], total_levels: int) -> list[dict]:
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Expected unique target levels")
    if any(i < 4 or i + 4 >= total_levels for i in targets):
        raise ValueError("Every retained target requires four original levels on either side")
    return [{"centre_axial_index": i, "target_axial_index": i,
             "axial_indices": list(range(i - 4, i + 5)),
             "retained_tile_positions": [4]} for i in sorted(targets)]

def select_centres(ratios: np.ndarray, fraction: float = 0.30) -> list[int]:
    ratios = np.asarray(ratios, dtype=float)
    if ratios.ndim != 1 or not np.isfinite(ratios).all() or np.any((ratios < 0) | (ratios > 1)):
        raise ValueError("Expected finite brain-content ratios in [0, 1]")
    if not 0 < fraction <= 1:
        raise ValueError("Candidate fraction must be in (0, 1]")
    count = max(1, math.ceil(len(ratios) * fraction))
    ranking = np.lexsort((np.arange(len(ratios)), -ratios))
    candidates = [int(i) for i in ranking[:count] if ratios[i] > 0]
    if len(candidates) < 2:
        raise ValueError("Two distinct candidate centres are required")
    highest = candidates[0]
    lowest = min((i for i in candidates if i != highest), key=lambda i: (ratios[i], i))
    centres = [highest, lowest]
    if any(i < 4 or i + 4 >= len(ratios) for i in centres):
        raise ValueError("Each selected centre requires four original levels on either side")
    return centres


def nine_slice_blocks(centres: list[int], total_levels: int) -> list[dict]:
    if len(centres) != 2 or len(set(centres)) != 2:
        raise ValueError("Expected two distinct block centres")
    if any(i < 4 or i + 4 >= total_levels for i in centres):
        raise ValueError("Each centre requires four original levels on either side")
    blocks = []
    seen = set()
    for centre in sorted(centres):
        indices = list(range(centre - 4, centre + 5))
        blocks.append({
            "centre_axial_index": centre,
            "axial_indices": indices,
            "retained_tile_positions": [i for i, z in enumerate(indices) if z not in seen],
        })
        seen.update(indices)
    return blocks


def render_slice(image_hu: np.ndarray, mask: np.ndarray, center: float, width: float, tile_size: int) -> Image.Image:
    values = np.nan_to_num(image_hu, nan=-1024.0, posinf=-1024.0, neginf=-1024.0)
    gray = window_hu_to_uint8(values, center=center, width=width)
    gray = np.where(mask, gray, 0).astype(np.uint8)
    image = Image.fromarray(gray.T[::-1]).convert("RGB")
    side = max(image.size)
    padded = Image.new("RGB", (side, side), (0, 0, 0))
    padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return padded.resize((tile_size, tile_size), Image.Resampling.BILINEAR)



