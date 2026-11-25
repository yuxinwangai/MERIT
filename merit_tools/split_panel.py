from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from .image_ops import split_3x3_mosaic


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract a row-major nine-tile panel and record the tile dimensions.')
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize", type=int)
    args = parser.parse_args()
    if args.resize is not None and args.resize < 1:
        parser.error("--resize must be positive")
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    raw = args.input.read_bytes()
    tiles = split_3x3_mosaic(raw)
    args.output_dir.mkdir(parents=True)
    manifest = []
    for index, tile in enumerate(tiles, 1):
        image = Image.fromarray(tile)
        native_size = list(image.size)
        if args.resize:
            image = image.resize((args.resize, args.resize), Image.Resampling.BILINEAR)
        filename = f"tile_{index:02d}.png"
        image.save(args.output_dir / filename)
        row, column = divmod(index - 1, 3)
        manifest.append({
            "file": filename,
            "tile_index": index,
            "row": row,
            "column": column,
            "native_size": native_size,
            "saved_size": list(image.size),
        })
    receipt = {"source_sha256": hashlib.sha256(raw).hexdigest(), "tiles": manifest}
    (args.output_dir / "tiles.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()

