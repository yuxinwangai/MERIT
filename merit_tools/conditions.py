from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .preprocessing import PreprocessingConfig, align_inputs, display_geometry, nine_slice_blocks, render_slice, select_levels, select_centres, target_context_blocks, standardize_inplane
from .volume_io import load_volume


IMAGING_FIELDS = (
    ("arterial_occlusion_or_severe_stenosis", "Arterial occlusion or severe stenosis"),
    ("distal_arterial_filling", "Distal arterial filling"),
    ("collateral_appearance", "Collateral appearance"),
    ("early_ischemic_change_on_ncct", "Early ischemic change on NCCT"),
    ("other_admission_imaging_findings", "Other admission imaging findings"),
)
CLINICAL_FIELDS = (
    ("nihss", "Admission NIHSS"),
    ("onset_to_door_minutes", "Onset-to-door time in minutes"),
)
STATUSES = {"positive", "negative", "uncertain", "unassessable", "unavailable"}


def format_condition(fields: dict) -> str:
    if not isinstance(fields, dict):
        raise ValueError("Expected a condition-field object")
    permitted = {key for key, label in IMAGING_FIELDS + CLINICAL_FIELDS}
    unexpected = set(fields) - permitted
    if unexpected:
        raise ValueError("Unexpected condition fields " + ", ".join(sorted(unexpected)))
    lines = ["Image orientation = neurological, patient right is image right"]
    for key, label in IMAGING_FIELDS:
        value = fields.get(key)
        if value is None:
            text = "unavailable"
        elif isinstance(value, str):
            text = value.strip() or "unavailable"
        elif isinstance(value, dict):
            if set(value) - {"status", "description"}:
                raise ValueError(f"Unexpected fields in {key}")
            status = value.get("status", "unavailable")
            description = value.get("description", "")
            if status not in STATUSES or not isinstance(description, str):
                raise ValueError(f"Invalid imaging condition {key}")
            text = status
            if description.strip():
                text += " | " + description.strip()
        else:
            raise ValueError(f"Invalid imaging condition {key}")
        lines.append(label + " = " + text)
    for key, label in CLINICAL_FIELDS:
        value = fields.get(key)
        if value is None or value == "":
            text = "unavailable"
        elif isinstance(value, str):
            text = value.strip() or "unavailable"
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0:
            text = f"{value:g}"
        else:
            raise ValueError(f"Invalid clinical field {key}")
        lines.append(label + " = " + text)
    return "\n".join(lines) + "\n"


@contextmanager
def output_directory(destination: str | Path, retain_failed: bool = False):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / ("." + destination.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix="." + destination.name + "-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "result"
            staging.mkdir()
            try:
                yield staging
                if destination.exists():
                    raise FileExistsError(destination)
                staging.rename(destination)
            except BaseException:
                if retain_failed and staging.exists():
                    staging.rename(destination.parent / (destination.name + ".failed." + uuid.uuid4().hex))
                raise
    finally:
        lock.unlink()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_case(
    cta_path: str | Path,
    ncct_path: str | Path,
    fields: dict,
    output_dir: str | Path,
    case_id: str,
    config: PreprocessingConfig | None = None,
    ncct_to_cta: np.ndarray | None = None,
    workflow: str = "primary",
) -> dict:
    if not case_id.strip() or any(character in case_id for character in "/\\"):
        raise ValueError("Expected a nonempty case identifier without path separators")
    config = config or PreprocessingConfig()
    text = format_condition(fields)
    original_cta = load_volume(cta_path)
    original_ncct = load_volume(ncct_path)
    cta, ncct = align_inputs(original_cta, original_ncct, ncct_to_cta)
    cta, ncct = standardize_inplane(cta, ncct, config)
    masks, ratios, centres = select_levels(ncct, config)
    if workflow == "primary":
        blocks = target_context_blocks(centres, ncct.data.shape[2])
    elif workflow == "two_centre_ablation":
        centres = select_centres(ratios, config.retained_fraction)
        blocks = nine_slice_blocks(centres, ncct.data.shape[2])
    else:
        raise ValueError("Unknown workflow")
    levels = sorted({block["axial_indices"][i] for block in blocks for i in block["retained_tile_positions"]})
    with output_directory(output_dir) as output:
        np.savez_compressed(output / "intracranial_masks.npz", masks=masks[:, :, levels], axial_indices=levels, affine=cta.affine)
        covered = {level for block in blocks for level in block["axial_indices"]}
        neighbours = {target + offset for target in levels for offset in (-2, -1, 0, 1)}
        context_images = {}
        for axial_index in sorted(neighbours - covered):
            if not 0 <= axial_index < cta.data.shape[2]:
                continue
            image = render_slice(cta.data[:, :, axial_index], masks[:, :, axial_index], config.cta_window_center, config.cta_window_width, config.tile_size)
            ImageDraw.Draw(image).text((8, 8), f"z={axial_index}", fill=(255, 255, 255))
            relative = f"context/cta_{axial_index:04d}.png"
            (output / "context").mkdir(exist_ok=True)
            image.save(output / relative)
            context_images[str(axial_index)] = relative
        records = []
        for index, block in enumerate(blocks, 1):
            name = f"block_{index:03d}"
            directory = output / "blocks" / name
            directory.mkdir(parents=True)
            indices = block["axial_indices"]
            ncct_index = max(indices, key=lambda level: ratios[level])
            panel = Image.new("RGB", (config.tile_size * 3, config.tile_size * 3))
            for tile, axial_index in enumerate(indices):
                image = render_slice(cta.data[:, :, axial_index], masks[:, :, axial_index], config.cta_window_center, config.cta_window_width, config.tile_size)
                ImageDraw.Draw(image).text((8, 8), f"z={axial_index}", fill=(255, 255, 255))
                row, column = divmod(tile, 3)
                panel.paste(image, (column * config.tile_size, row * config.tile_size))
            selected_ncct = render_slice(ncct.data[:, :, ncct_index], masks[:, :, ncct_index], config.ncct_window_center, config.ncct_window_width, config.tile_size)
            ImageDraw.Draw(selected_ncct).text((8, 8), f"NCCT z={ncct_index}", fill=(255, 255, 255))
            composite = Image.new("RGB", (config.tile_size * 4, config.tile_size * 3))
            composite.paste(panel, (0, 0))
            composite.paste(selected_ncct, (config.tile_size * 3, config.tile_size))
            panel.save(directory / "cta_panel.png")
            selected_ncct.save(directory / "ncct.png")
            composite.save(directory / "condition.png")
            (directory / "condition.txt").write_text(text, encoding="utf-8")
            positions = [float((cta.affine @ [0, 0, level, 1])[2]) for level in indices]
            record = {
                "block_id": name,
                **block,
                "ncct_axial_index": ncct_index,
                "physical_z_mm": positions,
                "span_mm": positions[-1] - positions[0],
                "adjacent_spacing_mm": np.diff(positions).tolist(),
                "cta_panel_size": list(panel.size),
                "condition_image_size": list(composite.size),
                "condition_image": f"blocks/{name}/condition.png",
                "condition_text": f"blocks/{name}/condition.txt",
            }
            write_json(directory / "block.json", record)
            records.append(record)
        result = {
            "case_id": case_id,
            "orientation": "RAS, neurological display, anterior at image top",
            "shape": list(cta.data.shape),
            "affine": cta.affine.tolist(),
            "display_geometry": display_geometry(cta, config.tile_size),
            "original_cta_affine": original_cta.affine.tolist(),
            "original_ncct_affine": original_ncct.affine.tolist(),
            "ncct_to_cta_transform": None if ncct_to_cta is None else np.asarray(ncct_to_cta).tolist(),
            "config": asdict(config),
            "valid_axial_indices": levels,
            "brain_content_ratios": ratios.tolist(),
            "candidate_centre_fraction": config.retained_fraction,
            "workflow": workflow,
            "generation_centre_indices": centres,
            "condition_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "blocks": records,
            "context_images": context_images,
        }
        write_json(output / "case.json", result)
    return result


