from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .generation import ImageEditGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate one panel from a prepared condition image and clinical text.')
    parser.add_argument("--condition-image", type=Path, required=True)
    parser.add_argument("--condition-text", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=Path(__file__).resolve().parents[1] / "prompts" / "merit_submission_prompt.txt")
    parser.add_argument("--model", default=os.environ.get("MERIT_MODEL"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.model:
        parser.error("Supply --model or set MERIT_MODEL")
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    image = args.condition_image.read_bytes()
    with Image.open(io.BytesIO(image)) as condition:
        condition.verify()
    template = args.prompt.read_text(encoding="utf-8")
    clinical_text = args.condition_text.read_text(encoding="utf-8")
    if "{clinical_situation}" not in template:
        parser.error("The prompt must contain {clinical_situation}")
    prompt = template.replace("{clinical_situation}", clinical_text)
    generator = ImageEditGenerator(args.model, os.environ.get("MERIT_BASE_URL") or None)
    started = datetime.now(timezone.utc).isoformat()
    generated = generator.edit_image(image, prompt)
    with Image.open(io.BytesIO(generated)) as panel:
        panel_format = panel.format
        panel.verify()

    args.output_dir.mkdir(parents=True)
    extension = ".jpg" if panel_format == "JPEG" else "." + panel_format.lower()
    panel_name = "generated_panel" + extension
    (args.output_dir / panel_name).write_bytes(generated)
    receipt = {
        "model": args.model,
        "requested_size": "1024x1024",
        "panel_format": panel_format,
        "panel_path": panel_name,
        "sdk_max_retries": 0,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition_image_sha256": hashlib.sha256(image).hexdigest(),
        "condition_text_sha256": hashlib.sha256(clinical_text.encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(generated).hexdigest(),
    }
    (args.output_dir / "generation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / panel_name)


if __name__ == "__main__":
    main()

