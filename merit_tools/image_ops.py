from __future__ import annotations

import io

import numpy as np
from PIL import Image

MOSAIC_GRID = 3
MOSAIC_CELL_SIZE = 512
MODEL_IMAGE_SIZE = 1024
CTA_WINDOW_CENTER = 200.0
CTA_WINDOW_WIDTH = 600.0


class PipelineError(RuntimeError):
    pass


def window_hu_to_uint8(
    image_hu: np.ndarray,
    *,
    center: float = CTA_WINDOW_CENTER,
    width: float = CTA_WINDOW_WIDTH,
) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    return np.rint(np.clip((image_hu - low) / max(high - low, 1.0) * 255.0, 0, 255)).astype(np.uint8)


def _fit_cell(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BILINEAR)
    cell = Image.new("RGB", (size, size), (0, 0, 0))
    cell.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return cell


def compose_3x3_mosaic(pngs: list[bytes], *, cell_size: int = MOSAIC_CELL_SIZE, final_size: int = MODEL_IMAGE_SIZE) -> bytes:
    if len(pngs) != MOSAIC_GRID * MOSAIC_GRID:
        raise PipelineError(f"Expected 9 PNG slices per mosaic, got {len(pngs)}")
    canvas = Image.new("RGB", (MOSAIC_GRID * cell_size, MOSAIC_GRID * cell_size), (0, 0, 0))
    for idx, png in enumerate(pngs):
        image = Image.open(io.BytesIO(png))
        cell = _fit_cell(image, cell_size)
        row, col = divmod(idx, MOSAIC_GRID)
        canvas.paste(cell, (col * cell_size, row * cell_size))
    if canvas.size != (final_size, final_size):
        canvas = canvas.resize((final_size, final_size), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def split_mosaic(image_bytes: bytes, grid_size: int = MOSAIC_GRID) -> list[np.ndarray]:
    if grid_size not in (1, 2, 3):
        raise PipelineError("The output grid size must be one, two or three")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    x_edges = np.rint(np.linspace(0, width, grid_size + 1)).astype(int).tolist()
    y_edges = np.rint(np.linspace(0, height, grid_size + 1)).astype(int).tolist()
    tiles: list[np.ndarray] = []
    for row in range(grid_size):
        for col in range(grid_size):
            tile = image.crop((x_edges[col], y_edges[row], x_edges[col + 1], y_edges[row + 1]))
            tiles.append(np.asarray(tile, dtype=np.uint8))
    return tiles


def split_3x3_mosaic(image_bytes: bytes) -> list[np.ndarray]:
    return split_mosaic(image_bytes, MOSAIC_GRID)

