from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .conditions import prepare_case
from .preprocessing import PreprocessingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cta", type=Path, required=True)
    parser.add_argument("--ncct", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "preprocessing.json")
    parser.add_argument("--ncct-to-cta-transform", type=Path)
    parser.add_argument("--workflow", choices=["primary", "two_centre_ablation"], default="primary")
    args = parser.parse_args()
    fields = json.loads(args.condition.read_text(encoding="utf-8"))
    transform = None
    if args.ncct_to_cta_transform:
        transform = np.asarray(json.loads(args.ncct_to_cta_transform.read_text(encoding="utf-8")), dtype=float)
    result = prepare_case(
        args.cta,
        args.ncct,
        fields,
        args.output_dir,
        args.case_id,
        PreprocessingConfig.load(args.config),
        transform,
        args.workflow,
    )
    print(json.dumps({"case_id": result["case_id"], "retained_levels": len(result["valid_axial_indices"]), "blocks": len(result["blocks"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()


