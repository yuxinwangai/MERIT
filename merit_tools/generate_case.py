from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .conditions import output_directory, write_json
from .generation import create_generator, TechnicalGenerationError
from .settings import load_settings, resolve_model
from .image_ops import split_mosaic


def input_file(case_dir: Path, relative: str) -> Path:
    path = (case_dir / relative).resolve()
    if not path.is_relative_to(case_dir):
        raise ValueError("A condition path is outside the prepared case directory")
    return path


def generate_case(
    case_dir: str | Path,
    output_dir: str | Path,
    model: str,
    generator=None,
    prompt_path: str | Path | None = None,
    technical_retries: int = 1,
    settings_path: str | Path | None = None,
    device: str = "cuda",
) -> dict:
    if technical_retries not in (0, 1):
        raise ValueError("technical_retries must be zero or one")
    case_dir = Path(case_dir).resolve()
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    prompt_path = Path(prompt_path) if prompt_path else input_file(case_dir, case["generation_prompt"]) if case.get("generation_prompt") else Path(__file__).resolve().parents[1] / "prompts" / "merit_submission_prompt.txt"
    template = prompt_path.read_text(encoding="utf-8")
    if "{clinical_situation}" not in template:
        raise ValueError("The prompt must contain {clinical_situation}")
    if not case.get("blocks"):
        raise ValueError("The prepared case contains no blocks")
    expected_levels = case["valid_axial_indices"]
    retained_levels = []
    block_ids = set()
    if int(case["config"]["tile_size"]) < 1:
        raise ValueError("The prepared tile size must be positive")
    for block in case["blocks"]:
        indices = block["axial_indices"]
        positions = block["retained_tile_positions"]
        grid_size = int(block.get("output_grid_size", 3))
        if grid_size not in (1, 2, 3) or len(indices) != grid_size * grid_size or len(set(positions)) != len(positions) or any(position not in range(grid_size * grid_size) for position in positions):
            raise ValueError("Invalid output-grid block metadata")
        block_id = block["block_id"]
        if not block_id or Path(block_id).name != block_id or any(character in block_id for character in "/\\"):
            raise ValueError("Invalid block identifier")
        if block_id in block_ids:
            raise ValueError("Duplicate block identifier")
        block_ids.add(block_id)
        retained_levels.extend(indices[position] for position in positions)
        for key in ("condition_image", "condition_text"):
            if not input_file(case_dir, block[key]).is_file():
                raise FileNotFoundError(block[key])
    if len(retained_levels) != len(set(retained_levels)) or set(retained_levels) != set(expected_levels):
        raise ValueError("The prepared blocks do not cover the retained axial levels exactly once")
    settings = load_settings(settings_path)
    provider, values = resolve_model(model, settings)
    model = values["model"]
    generator = generator or create_generator(model, settings_path, device)
    rows = []
    events = []
    written = set()
    with output_directory(output_dir, retain_failed=True) as output:
        (output / "panels").mkdir()
        (output / "tiles").mkdir()
        (output / "failed_attempts").mkdir()
        def persist_audit():
            write_json(output / "generation.json", {
                "case_id": case["case_id"], "model": model, "provider": provider,
                "status": "incomplete", "blocks": events,
            })
        for block in case["blocks"]:
            indices = block["axial_indices"]
            positions = block["retained_tile_positions"]
            image = input_file(case_dir, block["condition_image"]).read_bytes()
            clinical_text = input_file(case_dir, block["condition_text"]).read_text(encoding="utf-8")
            prompt = template.replace("{clinical_situation}", clinical_text)
            attempts = []
            event = {
                "block_id": block["block_id"], "attempts": attempts,
                "condition_image_sha256": hashlib.sha256(image).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            events.append(event)
            for attempt in range(technical_retries + 1):
                started = datetime.now(timezone.utc).isoformat()
                attempt_record = {"attempt": attempt + 1, "started_at_utc": started, "status": "started"}
                attempts.append(attempt_record)
                persist_audit()
                raw = None
                for attribute in ("last_response_id", "last_response_model"):
                    if hasattr(generator, attribute):
                        setattr(generator, attribute, None)
                try:
                    raw = generator.edit_image(image, prompt)
                    with Image.open(io.BytesIO(raw)) as panel:
                        panel_size = list(panel.size)
                        panel_format = panel.format
                        if panel.size != (1024, 1024):
                            raise TechnicalGenerationError("Expected a 1024-by-1024 output panel")
                        panel.verify()
                    tiles = split_mosaic(raw, int(block.get("output_grid_size", 3)))
                except (TechnicalGenerationError, UnidentifiedImageError, OSError) as error:
                    attempt_record.update({
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "technical_failure",
                        "error_type": type(error).__name__,
                        "response_id": getattr(generator, "last_response_id", None),
                        "returned_model": getattr(generator, "last_response_model", None),
                    })
                    if raw is not None:
                        failed_path = f"failed_attempts/{block['block_id']}_{attempt+1}.bin"
                        (output / failed_path).write_bytes(raw)
                        attempt_record.update({"raw_output_path": failed_path, "output_sha256": hashlib.sha256(raw).hexdigest()})
                    persist_audit()
                    if attempt == technical_retries:
                        write_json(output / "failure.json", {
                            "case_id": case["case_id"],
                            "model": model,
                            "block_id": block["block_id"],
                            "completed_blocks": len(events) - 1,
                        })
                        raise
                else:
                    attempt_record.update({
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "valid_image",
                        "response_id": getattr(generator, "last_response_id", None),
                        "returned_model": getattr(generator, "last_response_model", None),
                        "output_sha256": hashlib.sha256(raw).hexdigest(),
                    })
                    persist_audit()
                    break
            block_id = block["block_id"]
            extension = ".jpg" if panel_format == "JPEG" else "." + panel_format.lower()
            panel_path = f"panels/{block_id}{extension}"
            (output / panel_path).write_bytes(raw)
            for position in positions:
                axial_index = indices[position]
                if axial_index in written:
                    raise ValueError("Duplicate retained axial level")
                image_tile = Image.fromarray(tiles[position])
                native_size = list(image_tile.size)
                tile_size = int(case["config"]["tile_size"])
                image_tile = image_tile.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
                name = f"tiles/slice_{axial_index:04d}.png"
                image_tile.save(output / name)
                written.add(axial_index)
                rows.append({
                    "case_id": case["case_id"],
                    "slice_index": axial_index,
                    "tile_path": name,
                    "block_id": block_id,
                    "source_tile_index": position,
                    "native_width": native_size[0],
                    "native_height": native_size[1],
                })
            event.update({
                "panel_size": panel_size,
                "panel_format": panel_format,
                "panel_path": panel_path,
                "condition_image_sha256": hashlib.sha256(image).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "output_sha256": hashlib.sha256(raw).hexdigest(),
                "response_id": getattr(generator, "last_response_id", None),
                "returned_model": getattr(generator, "last_response_model", None),
            })
            persist_audit()
        if written != set(case["valid_axial_indices"]):
            raise ValueError("The generated stack does not cover the retained axial levels")
        rows.sort(key=lambda row: row["slice_index"])
        with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = {
            "status": "complete",
            "case_id": case["case_id"],
            "model": model,
            "provider": provider,
            "sdk_max_retries": 0,
            "technical_retries": technical_retries,
            "affine": case["affine"],
            "orientation": case["orientation"],
            "valid_axial_indices": case["valid_axial_indices"],
            "generated_slice_count": len(rows),
            "blocks": events,
        }
        write_json(output / "generation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MERIT_MODEL"))
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--technical-retries", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    if not args.model:
        parser.error("Supply --model or set MERIT_MODEL")
    result = generate_case(args.case_dir, args.output_dir, args.model, prompt_path=args.prompt, technical_retries=args.technical_retries, settings_path=args.settings, device=args.device)
    print(json.dumps({"case_id": result["case_id"], "generated_slice_count": result["generated_slice_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

