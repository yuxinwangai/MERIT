import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from merit_tools.conditions import output_directory, write_json
from merit_tools.volume_io import Volume, load_volume


PALETTE = np.asarray([[0,0,0], [0,0,255], [0,255,255], [0,255,0],
                      [255,255,0], [255,128,0], [255,0,0]], dtype=float)


def render_tmax(values, mask, percentile=True):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values) & (values >= 0)
    rendered = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not valid.any():
        return rendered
    lower, upper = np.percentile(values[valid], [2, 98]) if percentile else (0.0, 20.0)
    if upper <= lower:
        normalized = np.zeros(values.shape)
    else:
        normalized = np.clip((np.nan_to_num(values)-lower)/(upper-lower), 0, 1)
    if percentile:
        normalized = normalized ** 0.85
    positions = normalized * (len(PALETTE)-1)
    lo = np.floor(positions).astype(int)
    hi = np.minimum(lo+1, len(PALETTE)-1)
    mix = positions[..., None]-lo[..., None]
    colors = np.rint(PALETTE[lo]*(1-mix)+PALETTE[hi]*mix).astype(np.uint8)
    rendered[valid] = colors[valid]
    return rendered


def prepare_reference(case_dir, tmax_path, output_dir, tmax_to_cta=None):
    case_dir = Path(case_dir)
    case = json.loads((case_dir/"case.json").read_text(encoding="utf-8"))
    volume = load_volume(tmax_path)
    if tmax_to_cta is not None:
        transform = np.asarray(tmax_to_cta, dtype=float)
        if transform.shape != (4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0,0,0,1]):
            raise ValueError("Invalid Tmax-to-CTA transform")
        if not np.allclose(transform[:3,:3].T @ transform[:3,:3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(transform[:3,:3]), 1, atol=1e-5):
            raise ValueError("Expected a rigid Tmax-to-CTA transform")
        volume = Volume(volume.data, transform @ volume.affine)
    affine = np.asarray(case["affine"], dtype=float)
    mapping = np.linalg.solve(volume.affine, affine)
    shape = tuple(case["shape"])
    finite = np.isfinite(volume.data) & (volume.data >= 0)
    def resample(data):
        return ndimage.affine_transform(data.astype(np.float32), mapping[:3,:3], offset=mapping[:3,3],
                                        output_shape=shape, order=1, mode="constant", cval=0.0, prefilter=False)
    weights = resample(finite)
    values = resample(np.where(finite, volume.data, 0))
    values = np.divide(values, weights, out=np.full(shape, np.nan, dtype=np.float32), where=weights >= 1-1e-5)
    with np.load(case_dir/"intracranial_masks.npz", allow_pickle=False) as masks:
        support = masks["masks"].astype(bool)
        indices = masks["axial_indices"].astype(int)
        if not np.array_equal(indices, case["valid_axial_indices"]) or not np.allclose(masks["affine"], affine):
            raise ValueError("Prepared reference and case masks do not share a grid")
    if support.shape != (shape[0], shape[1], len(indices)):
        raise ValueError("Unexpected prepared mask shape")
    quantitative = np.stack([values[:,:,index].T[::-1] for index in indices])
    display_support = np.stack([support[:,:,i].T[::-1] for i in range(len(indices))])
    evaluation_support = display_support & np.isfinite(quantitative) & (quantitative >= 0)
    rows = []
    with output_directory(output_dir) as output:
        for directory in ["images", "fixed_scale", "masks"]:
            (output/directory).mkdir()
        for i, index in enumerate(indices):
            name = f"slice_{index:04d}.png"
            Image.fromarray(render_tmax(quantitative[i], evaluation_support[i])).save(output/"images"/name)
            Image.fromarray(render_tmax(quantitative[i], evaluation_support[i], False)).save(output/"fixed_scale"/name)
            Image.fromarray((evaluation_support[i]*255).astype(np.uint8)).save(output/"masks"/name)
            rows.append({"case_id":case["case_id"], "slice_index":int(index), "tile_path":"images/"+name,
                         "mask_path":"masks/"+name, "fixed_scale_path":"fixed_scale/"+name})
        np.savez_compressed(output/"quantitative.npz", tmax=quantitative, masks=evaluation_support,
                            intracranial_masks=display_support,
                            axial_indices=indices, display_to_world_mm=case["display_geometry"]["display_to_world_mm"])
        with (output/"manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        write_json(output/"reference.json", {"case_id":case["case_id"], "display_geometry":case["display_geometry"],
                    "percentiles":[2,98], "gamma":0.85, "fixed_scale_seconds":[0,20],
                    "tmax_to_cta":None if tmax_to_cta is None else np.asarray(tmax_to_cta).tolist()})
    return {"case_id":case["case_id"], "levels":len(indices)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--tmax", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tmax-to-cta", type=Path)
    args = parser.parse_args()
    transform = np.asarray(json.loads(args.tmax_to_cta.read_text(encoding="utf-8"))) if args.tmax_to_cta else None
    print(json.dumps(prepare_reference(args.case_dir, args.tmax, args.output_dir, transform)))


if __name__ == "__main__":
    main()
