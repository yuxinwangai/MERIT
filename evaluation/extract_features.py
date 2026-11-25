import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .evaluate import extract_inception_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({(row["case_id"], int(row["slice_index"])) for row in rows}) != len(rows):
        raise ValueError("Expected a nonempty manifest with unique patient/index pairs")
    arrays = {}
    for key in ["reference", "prediction"]:
        images = []
        for row in rows:
            image = Image.open(args.manifest.parent/row[key+"_path"]).convert("RGB")
            mask = Image.open(args.manifest.parent/row["mask_path"]).convert("L")
            if image.size != (512,512) or mask.size != image.size:
                raise ValueError("Expected images and masks on the 512-by-512 display grid")
            mask = mask.point(lambda value: 255 if value else 0)
            images.append(Image.composite(image, Image.new("RGB", image.size), mask))
        arrays[key] = extract_inception_features(images, args.device, args.batch_size).astype(np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, patient_ids=np.asarray([row["case_id"] for row in rows]),
                        axial_indices=np.asarray([int(row["slice_index"]) for row in rows]))


if __name__ == "__main__":
    main()
