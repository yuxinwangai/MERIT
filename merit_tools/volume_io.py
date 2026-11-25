from __future__ import annotations

import gzip
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
    1280: np.uint64,
}


@dataclass(frozen=True)
class Volume:
    data: np.ndarray
    affine: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim != 3 or any(size < 1 for size in self.data.shape):
            raise ValueError("Expected a nonempty three-dimensional volume")
        if self.affine.shape != (4, 4) or not np.isfinite(self.affine).all():
            raise ValueError("Expected a finite four-by-four affine")
        if not np.allclose(self.affine[3], [0, 0, 0, 1]):
            raise ValueError("Invalid affine last row")
        if abs(np.linalg.det(self.affine[:3, :3])) < 1e-10:
            raise ValueError("The volume affine is singular")


def load_volume(path: str | Path) -> Volume:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as content:
            return Volume(
                np.asarray(content["data"], dtype=np.float32),
                np.asarray(content["affine"], dtype=np.float64),
            )
    if path.suffix.lower() not in {".nii", ".gz"}:
        raise ValueError("Expected a NIfTI-1 or NPZ volume")
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as stream:
        blob = stream.read()
    if len(blob) < 352:
        raise ValueError("Truncated NIfTI header")
    if struct.unpack("<i", blob[:4])[0] == 348:
        endian = "<"
    elif struct.unpack(">i", blob[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError("Expected a NIfTI-1 header")
    if blob[344:348] != b"n+1\x00":
        raise ValueError("Expected a single-file NIfTI-1 image")
    dimensions = struct.unpack(endian + "8h", blob[40:56])
    ndim = dimensions[0]
    if not 3 <= ndim <= 7:
        raise ValueError("Expected three spatial NIfTI dimensions")
    shape = tuple(int(value) for value in dimensions[1:ndim + 1])
    if any(value < 1 for value in shape) or any(value != 1 for value in shape[3:]):
        raise ValueError("Expected one three-dimensional CT volume")
    datatype = struct.unpack(endian + "h", blob[70:72])[0]
    if datatype not in DTYPES:
        raise ValueError(f"Unsupported NIfTI datatype {datatype}")
    dtype = np.dtype(DTYPES[datatype]).newbyteorder(endian)
    offset_value = struct.unpack(endian + "f", blob[108:112])[0]
    if not math.isfinite(offset_value) or offset_value < 352 or not offset_value.is_integer():
        raise ValueError("Invalid NIfTI data offset")
    offset = int(offset_value)
    count = math.prod(shape)
    if offset + count * dtype.itemsize > len(blob):
        raise ValueError("Truncated NIfTI voxel data")
    data = np.frombuffer(blob, dtype=dtype, count=count, offset=offset)
    data = data.reshape(shape, order="F").reshape(shape[:3]).astype(np.float32)
    slope, intercept = struct.unpack(endian + "2f", blob[112:120])
    if math.isfinite(slope) and slope != 0:
        data = data * slope + (intercept if math.isfinite(intercept) else 0.0)
    pixdim = struct.unpack(endian + "8f", blob[76:108])
    qform_code, sform_code = struct.unpack(endian + "2h", blob[252:256])
    affine = np.eye(4, dtype=np.float64)
    if sform_code > 0:
        affine[:3] = np.asarray(struct.unpack(endian + "12f", blob[280:328])).reshape(3, 4)
    elif qform_code > 0:
        b, c, d, x, y, z = struct.unpack(endian + "6f", blob[256:280])
        norm = b * b + c * c + d * d
        if norm > 1:
            scale = math.sqrt(norm)
            b, c, d = b / scale, c / scale, d / scale
            a = 0.0
        else:
            a = math.sqrt(max(0.0, 1.0 - norm))
        rotation = np.asarray([
            [a*a+b*b-c*c-d*d, 2*b*c-2*a*d, 2*b*d+2*a*c],
            [2*b*c+2*a*d, a*a+c*c-b*b-d*d, 2*c*d-2*a*b],
            [2*b*d-2*a*c, 2*c*d+2*a*b, a*a+d*d-c*c-b*b],
        ])
        spacing = np.abs(np.asarray(pixdim[1:4], dtype=float))
        spacing[2] *= -1 if pixdim[0] < 0 else 1
        affine[:3, :3] = rotation @ np.diag(spacing)
        affine[:3, 3] = [x, y, z]
    else:
        raise ValueError("NIfTI input requires qform or sform spatial metadata")
    unit_scale = {1: 1000.0, 2: 1.0, 3: 0.001}.get(blob[123] & 7, 1.0)
    affine[:3] *= unit_scale
    return Volume(data, affine)


def resample_to_grid(volume: Volume, shape: tuple[int, int, int], affine: np.ndarray) -> Volume:
    from scipy import ndimage

    if volume.data.shape == shape and np.allclose(volume.affine, affine, atol=1e-6):
        return Volume(volume.data.copy(), affine.copy())
    mapping = np.linalg.solve(volume.affine, affine)
    data = np.nan_to_num(volume.data, nan=-1024.0, posinf=-1024.0, neginf=-1024.0)
    output = ndimage.affine_transform(
        data,
        mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=shape,
        order=1,
        mode="constant",
        cval=-1024.0,
        prefilter=False,
    )
    return Volume(output.astype(np.float32), affine.copy())


def canonicalize(volume: Volume) -> Volume:
    matrix = volume.affine[:3, :3]
    dominant = np.argmax(np.abs(matrix), axis=0)
    if len(set(dominant.tolist())) == 3:
        permutation = np.argsort(dominant)
        oriented = matrix[:, permutation]
        if np.allclose(oriented - np.diag(np.diag(oriented)), 0, atol=1e-5):
            data = np.transpose(volume.data, permutation)
            reindex = np.zeros((4, 4), dtype=float)
            reindex[3, 3] = 1
            for axis, source_axis in enumerate(permutation):
                sign = 1 if oriented[axis, axis] > 0 else -1
                reindex[source_axis, axis] = sign
                if sign < 0:
                    data = np.flip(data, axis=axis)
                    reindex[source_axis, 3] = volume.data.shape[source_axis] - 1
            return Volume(data.copy(), volume.affine @ reindex)
    corners = np.asarray([
        [x, y, z, 1]
        for x in [0, volume.data.shape[0] - 1]
        for y in [0, volume.data.shape[1] - 1]
        for z in [0, volume.data.shape[2] - 1]
    ])
    physical = corners @ volume.affine.T
    lower = physical[:, :3].min(axis=0)
    upper = physical[:, :3].max(axis=0)
    inverse = np.linalg.inv(matrix)
    spacing = 1.0 / np.linalg.norm(inverse, axis=0)
    shape = tuple((np.ceil((upper - lower) / spacing).astype(int) + 1).tolist())
    affine = np.eye(4)
    affine[:3, :3] = np.diag(spacing)
    affine[:3, 3] = lower
    return resample_to_grid(volume, shape, affine)

