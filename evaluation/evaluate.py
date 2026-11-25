from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

try:
    import numpy as np
except ImportError:  
    np = None

from PIL import Image, ImageChops, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_ROOT = PROJECT_ROOT / "new_single_result"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "metrics"
TRUTH_SUBDIR = "tmax"
INPUT_SUBDIR = "input"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

PICTURE_RESULT_DATA = PROJECT_ROOT / "results" / "picture_result" / "data"
RESAMPLE_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)


@dataclass(frozen=True)
class TileRecord:
    case_id: str
    method: str
    source_group: str
    source_panel: str
    slice_index: int
    pred_tile_path: Path
    truth_tile_path: Path
    pred_image: Image.Image
    truth_image: Image.Image


@dataclass(frozen=True)
class PerTileMetrics:
    case_id: str
    method: str
    source_group: str
    source_panel: str
    slice_index: int
    psnr: float | None
    ssim: float | None
    niqe: float | None
    brisque: float | None

PER_TILE_FIELDNAMES = [
    "experiment",
    "case_id",
    "model",
    "model_dir",
    "slice_index",
    "psnr",
    "ssim",
    "niqe",
    "brisque",
    "pred_path",
    "truth_path",
]

PER_CASE_FIELDNAMES = [
    "experiment",
    "case_id",
    "model",
    "model_dir",
    "n_tiles",
    "psnr",
    "psnr_std",
    "ssim",
    "ssim_std",
    "fid",
    "kid",
    "niqe",
    "niqe_std",
    "brisque",
    "brisque_std",
]

OVERALL_FIELDNAMES = [
    "experiment",
    "method",
    "model_dir",
    "result_root",
    "truth_root",
    "n_cases",
    "n_tiles",
    "NIQE ↓",
    "NIQE_q1",
    "NIQE_q3",
    "BRISQUE ↓",
    "BRISQUE_q1",
    "BRISQUE_q3",
    "FID ↓",
    "KID ↓",
    "PSNR, dB ↑",
    "PSNR_q1",
    "PSNR_q3",
    "SSIM ↑",
    "SSIM_q1",
    "SSIM_q3",
]

BY_PATIENT_FIELDNAMES = [
    "experiment",
    "model",
    "model_dir",
    "n_tiles",
    "psnr",
    "psnr_std",
    "ssim",
    "ssim_std",
    "fid",
    "kid",
    "niqe",
    "niqe_std",
    "brisque",
    "brisque_std",
]

SKIPPED_FIELDNAMES = [
    "experiment",
    "model_dir",
    "case_id",
    "slice_index",
    "reason",
    "pred_path",
    "truth_path",
]

PAIRWISE_TILE_FIELDNAMES = [
    "experiment",
    "pair",
    "model_a",
    "model_b",
    "case_id",
    "slice_index",
    "psnr",
    "ssim",
    "image_a_path",
    "image_b_path",
]

PAIRWISE_CASE_FIELDNAMES = [
    "experiment",
    "pair",
    "model_a",
    "model_b",
    "case_id",
    "n_tiles",
    "psnr",
    "psnr_std",
    "ssim",
    "ssim_std",
]

PAIRWISE_SUMMARY_FIELDNAMES = [
    "experiment",
    "pair",
    "model_a",
    "model_b",
    "n_cases",
    "n_tiles",
    "psnr",
    "psnr_std",
    "ssim",
    "ssim_std",
]


@dataclass(frozen=True)
class TruthTile:
    case_id: str
    slice_index: int
    tile_name: str
    truth_path: Path
    truth_image: Image.Image
    support_mask: Image.Image | None = None


@dataclass
class DistributionMetrics:
    global_fid: float | None
    global_kid_mean: float | None
    global_kid_std: float | None
    per_case: dict[str, dict[str, float | None]]


@dataclass
class ModelEvalResult:
    model_dir: str
    model_root: Path
    records: list[TileRecord]
    per_tile: list[PerTileMetrics]
    per_tile_rows: list[dict]
    per_case_rows: list[dict]
    summary_row: dict
    skipped: list[dict]


@dataclass
class PredictionCatalog:
    by_tile: dict[tuple[str, str], Path]
    by_slice: dict[tuple[str, int], Path]


def warn_unavailable(metric: str, detail: str) -> None:
    if not hasattr(warn_unavailable, "_shown"):
        warn_unavailable._shown = set()
    shown = warn_unavailable._shown
    if metric in shown:
        return
    shown.add(metric)
    print(f"warning: {metric} unavailable ({detail}); related CSV fields will be empty")


def load_image_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, RESAMPLE_BICUBIC)
    return image


def align_prediction_to_label(pred_image: Image.Image, truth_image: Image.Image) -> Image.Image:
    if pred_image.size == truth_image.size:
        return pred_image
    return pred_image.resize(truth_image.size, RESAMPLE_BICUBIC)


def image_to_array(image: Image.Image):
    if np is None:
        return None
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def compute_psnr(pred_image: Image.Image, truth_image: Image.Image) -> float | None:
    pred = align_prediction_to_label(pred_image, truth_image)
    diff = ImageChops.difference(pred, truth_image)
    stat = ImageStat.Stat(diff)
    mse = sum(v * v for v in stat.rms) / len(stat.rms)
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim_channel(pred_vals: list[int], truth_vals: list[int]) -> float | None:
    n = len(pred_vals)
    if n < 2:
        return None
    mean_x = sum(pred_vals) / n
    mean_y = sum(truth_vals) / n
    var_x = sum((x - mean_x) ** 2 for x in pred_vals) / (n - 1)
    var_y = sum((y - mean_y) ** 2 for y in truth_vals) / (n - 1)
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(pred_vals, truth_vals)) / (n - 1)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    denom = (mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2)
    if denom == 0:
        return None
    return ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / denom


def compute_ssim(pred_image: Image.Image, truth_image: Image.Image) -> float | None:
    pred = align_prediction_to_label(pred_image, truth_image)
    if np is not None:
        pred_arr = image_to_array(pred)
        truth_arr = image_to_array(truth_image)
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        scores = []
        for channel in range(3):
            x = pred_arr[:, :, channel]
            y = truth_arr[:, :, channel]
            mean_x = float(x.mean())
            mean_y = float(y.mean())
            var_x = float(x.var(ddof=1))
            var_y = float(y.var(ddof=1))
            cov_xy = float(((x - mean_x) * (y - mean_y)).sum() / max(x.size - 1, 1))
            denom = (mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2)
            if denom:
                scores.append(((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / denom)
        return float(sum(scores) / len(scores)) if scores else None

    channels = []
    pred_channels = pred.split()
    truth_channels = truth_image.split()
    for pred_ch, truth_ch in zip(pred_channels, truth_channels):
        score = compute_ssim_channel(list(pred_ch.getdata()), list(truth_ch.getdata()))
        if score is not None:
            channels.append(score)
    return float(sum(channels) / len(channels)) if channels else None


def prepare_no_reference_metrics(metrics: set[str], device: str) -> dict[str, object]:
    models: dict[str, object] = {}
    requested = {"niqe", "brisque"} & metrics
    if not requested:
        return models
    try:
        import pyiqa
    except ImportError:
        warn_unavailable("NIQE/BRISQUE", "install pyiqa to compute no-reference metrics")
        return models

    for metric in sorted(requested):
        try:
            models[metric] = pyiqa.create_metric(metric, device=device)
        except Exception as exc:  
            warn_unavailable(metric.upper(), str(exc))
    return models


def compute_no_reference_metric(metric_model: object, image: Image.Image) -> float | None:
    if np is None:
        warn_unavailable("NIQE/BRISQUE", "numpy is required to feed pyiqa")
        return None
    try:
        import torch
    except ImportError:
        warn_unavailable("NIQE/BRISQUE", "torch is required to feed pyiqa")
        return None

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    device = None
    if hasattr(metric_model, "parameters"):
        try:
            device = next(metric_model.parameters()).device
        except StopIteration:
            device = None
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    try:
        value = metric_model(tensor)
        return float(value.detach().cpu().item())
    except Exception as exc:  
        warn_unavailable("NIQE/BRISQUE", str(exc))
        return None


def compute_tile_metrics(
    records: list[TileRecord],
    metrics: set[str],
    no_ref: dict[str, object],
    device: str,
    batch_size: int,
    metric_backend: str,
) -> list[PerTileMetrics]:
    full_ref_scores = compute_full_reference_metrics(
        records=records,
        metrics=metrics,
        device=device,
        batch_size=batch_size,
        metric_backend=metric_backend,
    )
    out: list[PerTileMetrics] = []
    for idx, rec in enumerate(records):
        pred = align_prediction_to_label(rec.pred_image, rec.truth_image)
        niqe_model = no_ref.get("niqe")
        brisque_model = no_ref.get("brisque")
        out.append(
            PerTileMetrics(
                case_id=rec.case_id,
                method=rec.method,
                source_group=rec.source_group,
                source_panel=rec.source_panel,
                slice_index=rec.slice_index,
                psnr=full_ref_scores["psnr"][idx] if "psnr" in metrics else None,
                ssim=full_ref_scores["ssim"][idx] if "ssim" in metrics else None,
                niqe=compute_no_reference_metric(niqe_model, pred)
                if "niqe" in metrics and niqe_model is not None
                else None,
                brisque=compute_no_reference_metric(brisque_model, pred)
                if "brisque" in metrics and brisque_model is not None
                else None,
            )
        )
    return out


def compute_full_reference_metrics(
    records: list[TileRecord],
    metrics: set[str],
    device: str,
    batch_size: int,
    metric_backend: str,
) -> dict[str, list[float | None]]:
    requested = {"psnr", "ssim"} & metrics
    scores: dict[str, list[float | None]] = {metric: [None] * len(records) for metric in requested}
    if not requested:
        return scores

    if metric_backend in {"auto", "torchmetrics", "torch"}:
        try:
            return compute_full_reference_metrics_torch(
                records=records,
                metrics=requested,
                device=device,
                batch_size=batch_size,
                prefer_torchmetrics=metric_backend != "torch",
            )
        except Exception as exc:
            if metric_backend in {"torchmetrics", "torch"}:
                raise
            warn_unavailable("torchmetrics PSNR/SSIM", str(exc))

    if metric_backend == "skimage":
        return compute_full_reference_metrics_skimage(records=records, metrics=requested)

    for idx, rec in enumerate(records):
        pred = align_prediction_to_label(rec.pred_image, rec.truth_image)
        if "psnr" in requested:
            scores["psnr"][idx] = compute_psnr(pred, rec.truth_image)
        if "ssim" in requested:
            scores["ssim"][idx] = compute_ssim(pred, rec.truth_image)
    return scores


def pil_rgb_to_tensor(image: Image.Image, torch_module):
    image = image.convert("RGB")
    width, height = image.size
    data = torch_module.frombuffer(bytearray(image.tobytes()), dtype=torch_module.uint8)
    return data.reshape(height, width, 3).permute(2, 0, 1).to(dtype=torch_module.float32).div(255.0)


def compute_full_reference_metrics_torch(
    records: list[TileRecord],
    metrics: set[str],
    device: str,
    batch_size: int,
    prefer_torchmetrics: bool,
) -> dict[str, list[float | None]]:
    import torch

    structural_similarity_index_measure = None
    if "ssim" in metrics:
        if prefer_torchmetrics:
            try:
                from torchmetrics.functional.image import structural_similarity_index_measure
            except ImportError:
                structural_similarity_index_measure = None

    resolved_device = torch.device(resolve_device(device))
    scores: dict[str, list[float | None]] = {metric: [None] * len(records) for metric in metrics}

    with torch.inference_mode():
        indices_by_size: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, rec in enumerate(records):
            indices_by_size[rec.truth_image.size].append(idx)

        for indices in indices_by_size.values():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                preds = []
                truths = []
                for record_idx in batch_indices:
                    rec = records[record_idx]
                    pred = align_prediction_to_label(rec.pred_image, rec.truth_image)
                    preds.append(pil_rgb_to_tensor(pred, torch))
                    truths.append(pil_rgb_to_tensor(rec.truth_image, torch))

                pred_batch = torch.stack(preds).to(resolved_device, non_blocking=True)
                truth_batch = torch.stack(truths).to(resolved_device, non_blocking=True)

                if "psnr" in metrics:
                    mse = (pred_batch - truth_batch).square().flatten(1).mean(dim=1)
                    psnr = torch.where(
                        mse <= 0,
                        torch.full_like(mse, float("inf")),
                        20.0 * torch.log10(torch.ones_like(mse) / torch.sqrt(mse)),
                    )
                    for offset, value in enumerate(psnr.detach().cpu().tolist()):
                        scores["psnr"][batch_indices[offset]] = float(value)

                if "ssim" in metrics:
                    if structural_similarity_index_measure is not None:
                        try:
                            ssim = structural_similarity_index_measure(
                                pred_batch,
                                truth_batch,
                                data_range=1.0,
                                reduction="none",
                            )
                        except TypeError:
                            values = [
                                structural_similarity_index_measure(
                                    pred_batch[idx : idx + 1],
                                    truth_batch[idx : idx + 1],
                                    data_range=1.0,
                                )
                                for idx in range(pred_batch.shape[0])
                            ]
                            ssim = torch.stack([value.reshape(()) for value in values])
                    else:
                        ssim = torch_ssim_uniform(pred_batch, truth_batch, torch)
                    if ssim.ndim == 0:
                        ssim = ssim.repeat(pred_batch.shape[0])
                    for offset, value in enumerate(ssim.detach().cpu().flatten().tolist()):
                        scores["ssim"][batch_indices[offset]] = float(value)
    return scores


def torch_ssim_uniform(pred, truth, torch_module, window_size: int = 7, data_range: float = 1.0):
    import torch.nn.functional as F

    channels = pred.shape[1]
    kernel = torch_module.ones(
        (channels, 1, window_size, window_size),
        dtype=pred.dtype,
        device=pred.device,
    ) / float(window_size * window_size)
    pad = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(truth, kernel, padding=pad, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(truth * truth, kernel, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(pred * truth, kernel, padding=pad, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim_map.flatten(1).mean(dim=1)


def compute_full_reference_metrics_skimage(
    records: list[TileRecord],
    metrics: set[str],
) -> dict[str, list[float | None]]:
    if np is None:
        raise RuntimeError("numpy is required for scikit-image metrics")
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as exc:
        raise RuntimeError("install scikit-image to use --metric-backend skimage") from exc

    scores: dict[str, list[float | None]] = {metric: [None] * len(records) for metric in metrics}
    for idx, rec in enumerate(records):
        pred = align_prediction_to_label(rec.pred_image, rec.truth_image)
        pred_arr = np.asarray(pred.convert("RGB"), dtype=np.uint8)
        truth_arr = np.asarray(rec.truth_image.convert("RGB"), dtype=np.uint8)
        if "psnr" in metrics:
            scores["psnr"][idx] = float(peak_signal_noise_ratio(truth_arr, pred_arr, data_range=255))
        if "ssim" in metrics:
            scores["ssim"][idx] = float(
                structural_similarity(
                    truth_arr,
                    pred_arr,
                    channel_axis=2,
                    data_range=255,
                    win_size=7,
                    gaussian_weights=False,
                )
            )
    return scores


def extract_inception_features(
    images: list[Image.Image],
    device: str,
    batch_size: int,
):
    if np is None:
        raise RuntimeError("numpy is required")
    import torch
    from torchvision import models, transforms

    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )

    features = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([preprocess(img.convert("RGB")) for img in images[start : start + batch_size]])
            batch = batch.to(device)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            features.append(out.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(real_features, fake_features) -> float | None:
    if np is None:
        return None
    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        return None
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    mu_real = real_features.mean(axis=0)
    mu_fake = fake_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma_real)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("Reference covariance is not positive semidefinite")
    root_real = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    product = root_real @ sigma_fake @ root_real
    product = (product + product.T) / 2.0
    product_eigenvalues = np.linalg.eigvalsh(product)
    product_tolerance = 1e-10 * max(1.0, float(np.max(np.abs(product_eigenvalues))))
    if float(product_eigenvalues.min()) < -product_tolerance:
        raise ValueError("Covariance product is not positive semidefinite")
    trace_root = float(np.sqrt(np.maximum(product_eigenvalues, 0.0)).sum())
    diff = mu_real - mu_fake
    distance = float(diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2.0 * trace_root)
    return max(distance, 0.0)


def compute_kid(real_features, fake_features) -> tuple[float | None, float | None]:
    if np is None:
        return None, None
    n = min(real_features.shape[0], fake_features.shape[0])
    if n < 2:
        return None, None
    real = real_features[:n]
    fake = fake_features[:n]
    dim = real.shape[1]

    k_xx = ((real @ real.T) / dim + 1.0) ** 3
    k_yy = ((fake @ fake.T) / dim + 1.0) ** 3
    k_xy = ((real @ fake.T) / dim + 1.0) ** 3
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    kid = k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (n * (n - 1)) - 2.0 * k_xy.mean()
    return float(kid), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR, SSIM, FID, KID, NIQE and BRISQUE against Tmax reference images."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Experiment directory or root containing multiple experiments")
    parser.add_argument("--truth-subdir", default=TRUTH_SUBDIR, help="Reference subdirectory name, default tmax")
    parser.add_argument("--input-subdir", default=INPUT_SUBDIR, help="Source-image subdirectory name, default input")
    parser.add_argument("--settings", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "model_settings.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory, default {root}/metrics. Multiple experiments also receive separate metrics directories.",
    )
    parser.add_argument(
        "--recursive-experiments",
        action="store_true",
        help="Treat each immediate subdirectory containing the reference subdirectory as an experiment",
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=None,
        help="Evaluate the specified prediction directories. By default, scan directories other than input, tmax and metrics.",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        default=None,
        help="Additional directory names to exclude during automatic discovery",
    )
    parser.add_argument(
        "--include-input-baseline",
        action="store_true",
        help="Evaluate the input images against the Tmax references as an additional baseline",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for pyiqa and PyTorch, default auto. Select CUDA when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for image metrics and feature extraction, default 64",
    )
    parser.add_argument(
        "--metric-backend",
        choices=["auto", "cpu", "torch", "torchmetrics", "skimage"],
        default=None,
        help=(
            "PSNR/SSIM backend; defaults to the CPU global-moment implementation selected in model_settings.json."
        ),
    )
    parser.add_argument("--patient", default=None, help="Evaluate a single case, for example sub-stroke0001")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["psnr", "ssim", "niqe", "brisque", "fid", "kid"],
        default=["psnr", "ssim", "niqe", "brisque", "fid", "kid"],
    )
    parser.add_argument(
        "--export-to-picture-result",
        action="store_true",
        help="Also write results/picture_result/data/per_case_metrics.csv",
    )
    parser.add_argument(
        "--pairwise-generated",
        action="store_true",
        help="Write pairwise PSNR/SSIM comparisons between generated images matched by case and slice",
    )
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    if args.metric_backend is None:
        args.metric_backend = settings["evaluation"]["metric_backend"]
    if args.metric_backend not in {"auto", "cpu", "torch", "torchmetrics", "skimage"}:
        parser.error("Unsupported metric backend in model settings")
    return args


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.6f}"


def metric_mean_std(rows: list[object], key: str) -> tuple[float | None, float | None]:
    vals = [getattr(row, key) for row in rows if getattr(row, key) is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((value - mean) ** 2 for value in vals) / len(vals)
    return float(mean), float(math.sqrt(var))


def iter_manifest_paths(root: Path) -> list[Path]:
    
    paths = set(root.glob("*/split_unique_tiles/manifest.csv"))
    paths.update(root.glob("*/manifest.csv"))
    if (root / "manifest.csv").is_file():
        paths.add(root / "manifest.csv")
    return sorted(paths)


def iter_truth_manifest_rows(truth_root: Path, patient: str | None):
    for manifest_path in iter_manifest_paths(truth_root):
        with manifest_path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                case_id = row.get("case_id", "").strip()
                if patient and case_id != patient:
                    continue
                yield manifest_path, row


def skip_row(
    model_dir: str,
    case_id: str,
    slice_index: str,
    reason: str,
    pred_path: str,
    truth_path: str,
) -> dict[str, str]:
    return {
        "model_dir": model_dir,
        "case_id": case_id,
        "slice_index": slice_index,
        "reason": reason,
        "pred_path": pred_path,
        "truth_path": truth_path,
    }


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def find_case_id(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("sub-stroke"):
            return part
    return None


def parse_slice_index(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    patterns = (
        r"(?:slice|slice_index|idx|index)[_-]?(\d+)",
        r"(?:^|[_-])(\d{1,4})(?:$|[_-])",
    )
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def build_prediction_catalog(model_root: Path) -> PredictionCatalog:
    by_tile: dict[tuple[str, str], Path] = {}
    by_slice: dict[tuple[str, int], Path] = {}

    for manifest_path in iter_manifest_paths(model_root):
        with manifest_path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                case_id = row.get("case_id", "").strip() or find_case_id(manifest_path)
                tile_name = row.get("tile_path", "").strip()
                slice_index = parse_slice_index(row.get("slice_index", ""))
                if not case_id or not tile_name:
                    continue
                pred_path = manifest_path.parent / tile_name
                if not is_image_file(pred_path):
                    continue
                by_tile.setdefault((case_id, tile_name), pred_path)
                if slice_index is not None:
                    by_slice.setdefault((case_id, slice_index), pred_path)

    for pred_path in sorted(p for p in model_root.rglob("*") if is_image_file(p)):
        case_id = find_case_id(pred_path.relative_to(model_root))
        if not case_id:
            continue
        by_tile.setdefault((case_id, pred_path.name), pred_path)
        try:
            by_tile.setdefault((case_id, pred_path.relative_to(model_root / case_id).as_posix()), pred_path)
        except ValueError:
            pass
        slice_index = parse_slice_index(pred_path.stem)
        if slice_index is not None:
            by_slice.setdefault((case_id, slice_index), pred_path)

    return PredictionCatalog(by_tile=by_tile, by_slice=by_slice)


def resolve_prediction_path(
    model_root: Path,
    catalog: PredictionCatalog,
    truth: TruthTile,
) -> Path:
    standard_paths = (
        model_root / truth.case_id / "split_unique_tiles" / truth.tile_name,
        model_root / truth.case_id / truth.tile_name,
    )
    for standard_path in standard_paths:
        if standard_path.is_file():
            return standard_path

    candidates = (
        catalog.by_tile.get((truth.case_id, truth.tile_name)),
        catalog.by_tile.get((truth.case_id, Path(truth.tile_name).name)),
        catalog.by_slice.get((truth.case_id, truth.slice_index)),
    )
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    return standard_paths[0]


def load_truth_catalog(truth_root: Path, patient: str | None) -> dict[tuple[str, int], TruthTile]:
    
    catalog: dict[tuple[str, int], TruthTile] = {}
    for manifest_path, row in iter_truth_manifest_rows(truth_root, patient):
        case_id = row.get("case_id", "").strip()
        try:
            slice_index = int(row.get("slice_index", ""))
        except ValueError:
            continue
        tile_name = row.get("tile_path", "").strip()
        if not tile_name:
            continue
        truth_path = manifest_path.parent / tile_name
        if not truth_path.is_file():
            continue
        key = (case_id, slice_index)
        if key in catalog:
            continue
        mask_name = row.get("mask_path", "").strip()
        if not mask_name:
            raise ValueError("The reference manifest requires mask_path for matching evaluation support")
        mask = Image.open(manifest_path.parent / mask_name).convert("L")
        truth_image = load_image_rgb(truth_path)
        if mask.size != truth_image.size:
            raise ValueError("Reference image and support mask dimensions differ")
        mask = mask.point(lambda value: 255 if value else 0)
        truth_image = Image.composite(truth_image, Image.new("RGB", truth_image.size), mask)
        catalog[key] = TruthTile(
            case_id=case_id,
            slice_index=slice_index,
            tile_name=tile_name,
            truth_path=truth_path,
            truth_image=truth_image,
            support_mask=mask,
        )
    return catalog


def load_records_against_truth(
    model_root: Path,
    truth_catalog: dict[tuple[str, int], TruthTile],
    model_label: str,
) -> tuple[list[TileRecord], list[dict[str, str]]]:
    
    records: list[TileRecord] = []
    skipped: list[dict[str, str]] = []
    model_dir = model_root.name
    pred_catalog = build_prediction_catalog(model_root)

    for (case_id, slice_index), truth in sorted(truth_catalog.items()):
        pred_path = resolve_prediction_path(model_root, pred_catalog, truth)
        if not pred_path.is_file():
            skipped.append(
                skip_row(
                    model_dir,
                    case_id,
                    str(slice_index),
                    "missing_pred",
                    str(pred_path),
                    str(truth.truth_path),
                )
            )
            continue

        pred_image = load_image_rgb(
            pred_path,
            size=truth.truth_image.size,
        )
        if truth.support_mask is not None:
            pred_image = Image.composite(pred_image, Image.new("RGB", pred_image.size), truth.support_mask)
        records.append(
            TileRecord(
                case_id=case_id,
                method=model_label,
                source_group=model_dir,
                source_panel="",
                slice_index=slice_index,
                pred_tile_path=pred_path,
                truth_tile_path=truth.truth_path,
                pred_image=pred_image,
                truth_image=truth.truth_image,
            )
        )
    return records, skipped


def compute_fid_low_rank(real_features: np.ndarray, fake_features: np.ndarray) -> float | None:
    n = min(real_features.shape[0], fake_features.shape[0])
    if n < 2:
        return None
    real = real_features[:n]
    fake = fake_features[:n]
    mu_real = real.mean(axis=0)
    mu_fake = fake.mean(axis=0)
    real_centered = real - mu_real
    fake_centered = fake - mu_fake
    cov_scale = float(n - 1)
    trace_real = float(np.sum(real_centered * real_centered) / cov_scale)
    trace_fake = float(np.sum(fake_centered * fake_centered) / cov_scale)
    cross = real_centered @ fake_centered.T / cov_scale
    sqrt_trace = float(np.linalg.svd(cross, compute_uv=False).sum())
    diff = mu_real - mu_fake
    return float(diff @ diff + trace_real + trace_fake - 2.0 * sqrt_trace)


def compute_distribution_metrics(
    records: list[TileRecord],
    metrics: set[str],
    device: str,
    batch_size: int,
) -> DistributionMetrics:
    
    empty = DistributionMetrics(None, None, None, {})
    if not ({"fid", "kid"} & metrics) or not records:
        return empty
    if np is None:
        warn_unavailable("FID/KID", "numpy is required")
        return empty
    try:
        import torch  
        import torchvision.models as models  
    except ImportError:
        warn_unavailable("FID/KID", "install torch and torchvision to compute Inception features")
        return empty

    truth_images = [rec.truth_image for rec in records]
    pred_images = [align_prediction_to_label(rec.pred_image, rec.truth_image) for rec in records]
    try:
        real_features = extract_inception_features(truth_images, device=device, batch_size=batch_size)
        fake_features = extract_inception_features(pred_images, device=device, batch_size=batch_size)
    except Exception as exc:
        warn_unavailable("FID/KID", str(exc))
        return empty

    global_fid = compute_fid(real_features, fake_features) if "fid" in metrics else None
    global_kid_mean, global_kid_std = (None, None)
    if "kid" in metrics:
        global_kid_mean, global_kid_std = compute_kid(real_features, fake_features)

    index_by_case: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        index_by_case[rec.case_id].append(idx)

    per_case: dict[str, dict[str, float | None]] = {}
    for case_id, indices in index_by_case.items():
        if len(indices) < 2:
            per_case[case_id] = {"fid": None, "kid_mean": None, "kid_std": None}
            continue
        real = real_features[indices]
        fake = fake_features[indices]
        kid_mean, kid_std = compute_kid(real, fake) if "kid" in metrics else (None, None)
        per_case[case_id] = {
            "fid": compute_fid_low_rank(real, fake) if "fid" in metrics else None,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
        }

    return DistributionMetrics(global_fid, global_kid_mean, global_kid_std, per_case)


def build_per_tile_rows(
    records: list[TileRecord],
    per_tile: list[PerTileMetrics],
    model_dir: str,
) -> list[dict[str, str | int | float | None]]:
    path_by_key = {
        (rec.case_id, rec.slice_index): (str(rec.pred_tile_path), str(rec.truth_tile_path))
        for rec in records
    }
    rows: list[dict[str, str | int | float | None]] = []
    for row in per_tile:
        pred_path, truth_path = path_by_key.get((row.case_id, row.slice_index), ("", ""))
        rows.append(
            {
                "case_id": row.case_id,
                "model": row.method,
                "model_dir": model_dir,
                "slice_index": row.slice_index,
                "psnr": fmt(row.psnr),
                "ssim": fmt(row.ssim),
                "niqe": fmt(row.niqe),
                "brisque": fmt(row.brisque),
                "pred_path": pred_path,
                "truth_path": truth_path,
            }
        )
    return rows


def build_per_case_rows(
    records: list[TileRecord],
    per_tile: list[PerTileMetrics],
    dist: DistributionMetrics,
    model_dir: str,
    model_label: str,
) -> list[dict[str, str | int | float | None]]:
    by_case_records: dict[str, list[TileRecord]] = defaultdict(list)
    by_case_tiles: dict[str, list[PerTileMetrics]] = defaultdict(list)
    for rec in records:
        by_case_records[rec.case_id].append(rec)
    for row in per_tile:
        by_case_tiles[row.case_id].append(row)

    out: list[dict[str, str | int | float | None]] = []
    for case_id in sorted(by_case_records):
        tiles = by_case_tiles.get(case_id, [])
        psnr_mean, psnr_std = metric_mean_std(tiles, "psnr")
        ssim_mean, ssim_std = metric_mean_std(tiles, "ssim")
        niqe_mean, niqe_std = metric_mean_std(tiles, "niqe")
        brisque_mean, brisque_std = metric_mean_std(tiles, "brisque")
        case_dist = dist.per_case.get(case_id, {})
        out.append(
            {
                "case_id": case_id,
                "model": model_label,
                "model_dir": model_dir,
                "n_tiles": len(by_case_records[case_id]),
                "psnr": fmt(psnr_mean),
                "psnr_std": fmt(psnr_std),
                "ssim": fmt(ssim_mean),
                "ssim_std": fmt(ssim_std),
                "fid": fmt(case_dist.get("fid")),
                "kid": fmt(case_dist.get("kid_mean")),
                "niqe": fmt(niqe_mean),
                "niqe_std": fmt(niqe_std),
                "brisque": fmt(brisque_mean),
                "brisque_std": fmt(brisque_std),
            }
        )
    return out


def build_summary_row(
    model_dir: str,
    model_label: str,
    model_root: Path,
    truth_root: Path,
    records: list[TileRecord],
    per_tile: list[PerTileMetrics],
    dist: DistributionMetrics,
) -> dict[str, str | int]:
    by_patient = defaultdict(list)
    for row in per_tile:
        by_patient[row.case_id].append(row)
    summaries = {}
    for metric in ("psnr", "ssim", "niqe", "brisque"):
        means = [metric_mean_std(rows, metric)[0] for rows in by_patient.values()]
        means = [value for value in means if value is not None]
        summaries[metric] = np.quantile(means, [0.5, 0.25, 0.75]).tolist() if means else [None]*3
    psnr, ssim, niqe, brisque = [summaries[key] for key in ("psnr", "ssim", "niqe", "brisque")]

    return {
        "method": model_label,
        "model_dir": model_dir,
        "result_root": str(model_root),
        "truth_root": str(truth_root),
        "n_cases": len({rec.case_id for rec in records}),
        "n_tiles": len(records),
        "NIQE ↓": fmt(niqe[0]),
        "NIQE_q1": fmt(niqe[1]),
        "NIQE_q3": fmt(niqe[2]),
        "BRISQUE ↓": fmt(brisque[0]),
        "BRISQUE_q1": fmt(brisque[1]),
        "BRISQUE_q3": fmt(brisque[2]),
        "FID ↓": fmt(dist.global_fid),
        "KID ↓": fmt(dist.global_kid_mean),
        "PSNR, dB ↑": fmt(psnr[0]),
        "PSNR_q1": fmt(psnr[1]),
        "PSNR_q3": fmt(psnr[2]),
        "SSIM ↑": fmt(ssim[0]),
        "SSIM_q1": fmt(ssim[1]),
        "SSIM_q3": fmt(ssim[2]),
    }


def pair_label(model_a: str, model_b: str) -> str:
    return f"{model_a}__vs__{model_b}"


def build_pairwise_records(
    result_a: ModelEvalResult,
    result_b: ModelEvalResult,
) -> list[TileRecord]:
    records_a = {(rec.case_id, rec.slice_index): rec for rec in result_a.records}
    records_b = {(rec.case_id, rec.slice_index): rec for rec in result_b.records}
    records: list[TileRecord] = []

    for case_id, slice_index in sorted(set(records_a) & set(records_b)):
        rec_a = records_a[(case_id, slice_index)]
        rec_b = records_b[(case_id, slice_index)]
        image_a = align_prediction_to_label(rec_a.pred_image, rec_b.pred_image)
        records.append(
            TileRecord(
                case_id=case_id,
                method=pair_label(result_a.model_dir, result_b.model_dir),
                source_group=result_a.model_dir,
                source_panel=result_b.model_dir,
                slice_index=slice_index,
                pred_tile_path=rec_a.pred_tile_path,
                truth_tile_path=rec_b.pred_tile_path,
                pred_image=image_a,
                truth_image=rec_b.pred_image,
            )
        )
    return records


def build_pairwise_tile_rows(
    experiment: str,
    model_a: str,
    model_b: str,
    records: list[TileRecord],
    per_tile: list[PerTileMetrics],
) -> list[dict[str, str | int]]:
    pair = pair_label(model_a, model_b)
    path_by_key = {
        (rec.case_id, rec.slice_index): (str(rec.pred_tile_path), str(rec.truth_tile_path))
        for rec in records
    }
    rows: list[dict[str, str | int]] = []
    for metric_row in per_tile:
        image_a_path, image_b_path = path_by_key[(metric_row.case_id, metric_row.slice_index)]
        rows.append(
            {
                "experiment": experiment,
                "pair": pair,
                "model_a": model_a,
                "model_b": model_b,
                "case_id": metric_row.case_id,
                "slice_index": metric_row.slice_index,
                "psnr": fmt(metric_row.psnr),
                "ssim": fmt(metric_row.ssim),
                "image_a_path": image_a_path,
                "image_b_path": image_b_path,
            }
        )
    return rows


def build_pairwise_case_rows(
    experiment: str,
    model_a: str,
    model_b: str,
    records: list[TileRecord],
    per_tile: list[PerTileMetrics],
) -> list[dict[str, str | int]]:
    pair = pair_label(model_a, model_b)
    by_case_records: dict[str, list[TileRecord]] = defaultdict(list)
    by_case_tiles: dict[str, list[PerTileMetrics]] = defaultdict(list)
    for rec in records:
        by_case_records[rec.case_id].append(rec)
    for row in per_tile:
        by_case_tiles[row.case_id].append(row)

    rows: list[dict[str, str | int]] = []
    for case_id in sorted(by_case_records):
        tiles = by_case_tiles[case_id]
        psnr_mean, psnr_std = metric_mean_std(tiles, "psnr")
        ssim_mean, ssim_std = metric_mean_std(tiles, "ssim")
        rows.append(
            {
                "experiment": experiment,
                "pair": pair,
                "model_a": model_a,
                "model_b": model_b,
                "case_id": case_id,
                "n_tiles": len(by_case_records[case_id]),
                "psnr": fmt(psnr_mean),
                "psnr_std": fmt(psnr_std),
                "ssim": fmt(ssim_mean),
                "ssim_std": fmt(ssim_std),
            }
        )
    return rows


def build_pairwise_summary_row(
    experiment: str,
    model_a: str,
    model_b: str,
    records: list[TileRecord],
    per_tile: list[PerTileMetrics],
) -> dict[str, str | int]:
    psnr_mean, psnr_std = metric_mean_std(per_tile, "psnr")
    ssim_mean, ssim_std = metric_mean_std(per_tile, "ssim")
    return {
        "experiment": experiment,
        "pair": pair_label(model_a, model_b),
        "model_a": model_a,
        "model_b": model_b,
        "n_cases": len({rec.case_id for rec in records}),
        "n_tiles": len(records),
        "psnr": fmt(psnr_mean),
        "psnr_std": fmt(psnr_std),
        "ssim": fmt(ssim_mean),
        "ssim_std": fmt(ssim_std),
    }


def evaluate_pairwise_generated(
    experiment: str,
    results: list[ModelEvalResult],
    device: str,
    batch_size: int,
    metric_backend: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    per_tile_rows: list[dict] = []
    per_case_rows: list[dict] = []
    summary_rows: list[dict] = []
    metrics = {"psnr", "ssim"}

    for result_a, result_b in combinations(results, 2):
        records = build_pairwise_records(result_a, result_b)
        if not records:
            print(f"  pairwise {result_a.model_dir} vs {result_b.model_dir}: no same-slice matches")
            continue
        per_tile = compute_tile_metrics(
            records=records,
            metrics=metrics,
            no_ref={},
            device=device,
            batch_size=batch_size,
            metric_backend=metric_backend,
        )
        per_tile_rows.extend(
            build_pairwise_tile_rows(experiment, result_a.model_dir, result_b.model_dir, records, per_tile)
        )
        per_case_rows.extend(
            build_pairwise_case_rows(experiment, result_a.model_dir, result_b.model_dir, records, per_tile)
        )
        summary_rows.append(
            build_pairwise_summary_row(experiment, result_a.model_dir, result_b.model_dir, records, per_tile)
        )
        print(
            f"  pairwise {result_a.model_dir} vs {result_b.model_dir}: "
            f"{summary_rows[-1]['n_cases']} cases, {summary_rows[-1]['n_tiles']} tiles"
        )

    return per_tile_rows, per_case_rows, summary_rows


def evaluate_model(
    model_dir: str,
    model_label: str,
    model_root: Path,
    truth_catalog: dict[tuple[str, int], TruthTile],
    truth_root: Path,
    metrics: set[str],
    no_ref: dict[str, object],
    device: str,
    batch_size: int,
    metric_backend: str,
) -> ModelEvalResult | None:
    records, skipped = load_records_against_truth(model_root, truth_catalog, model_label)
    if not records:
        print(f"skip {model_dir}: no matched tiles against {truth_root.name}")
        return None

    per_tile = compute_tile_metrics(records, metrics, no_ref, device, batch_size, metric_backend)
    dist = compute_distribution_metrics(records, metrics, device, batch_size)

    return ModelEvalResult(
        model_dir=model_dir,
        model_root=model_root,
        records=records,
        per_tile=per_tile,
        per_tile_rows=build_per_tile_rows(records, per_tile, model_dir),
        per_case_rows=build_per_case_rows(records, per_tile, dist, model_dir, model_label),
        summary_row=build_summary_row(
            model_dir, model_label, model_root, truth_root, records, per_tile, dist
        ),
        skipped=skipped,
    )


def discover_experiments(root: Path, truth_subdir: str, recursive: bool) -> list[Path]:
    if not recursive:
        return [root]

    experiments = sorted(p for p in root.iterdir() if p.is_dir() and (p / truth_subdir).is_dir())
    if not experiments:
        raise SystemExit(f"no experiment dirs with {truth_subdir}/ found under {root}")
    return experiments


def resolve_eval_models(
    experiment_root: Path,
    truth_subdir: str,
    input_subdir: str,
    model_dirs: list[str] | None,
    exclude_dirs: list[str] | None,
    include_input_baseline: bool,
) -> list[tuple[str, str, Path]]:
    if model_dirs:
        missing = [name for name in model_dirs if not (experiment_root / name).is_dir()]
        if missing:
            raise SystemExit(
                f"model dirs not found under {experiment_root}: {', '.join(missing)}"
            )
        return [(name, name, experiment_root / name) for name in model_dirs]

    excluded = {truth_subdir, "metrics", "__pycache__"}
    if not include_input_baseline:
        excluded.add(input_subdir)
    if exclude_dirs:
        excluded.update(exclude_dirs)

    specs = []
    for model_root in sorted(p for p in experiment_root.iterdir() if p.is_dir()):
        model_dir = model_root.name
        if model_dir in excluded:
            continue
        specs.append((model_dir, model_dir, model_root))

    if not specs:
        raise SystemExit(f"no prediction dirs found under {experiment_root}")
    return specs


def add_experiment(rows: list[dict], experiment: str) -> list[dict]:
    for row in rows:
        row["experiment"] = experiment
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def write_by_patient_csvs(output_dir: Path, per_case_rows: list[dict]) -> None:
    
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in per_case_rows:
        by_patient[str(row["case_id"])].append(row)

    patient_dir = output_dir / "by_patient"
    for case_id in sorted(by_patient):
        rows = []
        for row in by_patient[case_id]:
            rows.append({key: row.get(key, "") for key in BY_PATIENT_FIELDNAMES})
        write_csv(patient_dir / f"{case_id}.csv", BY_PATIENT_FIELDNAMES, rows)
    print(f"wrote {len(by_patient)} patient files under {patient_dir}")


def export_picture_result(per_case_rows: list[dict], summary_rows: list[dict], output_dir: Path) -> None:
    PICTURE_RESULT_DATA.mkdir(parents=True, exist_ok=True)
    picture_case_fields = [
        "case_id",
        "model",
        "model_dir",
        "n_tiles",
        "psnr",
        "ssim",
        "fid",
        "kid",
        "niqe",
        "brisque",
    ]
    picture_rows = [{k: row.get(k, "") for k in picture_case_fields} for row in per_case_rows]
    per_case_path = PICTURE_RESULT_DATA / "per_case_metrics.csv"
    write_csv(per_case_path, picture_case_fields, picture_rows)

    summary_copy = output_dir / "summary_for_picture_result.csv"
    write_csv(summary_copy, OVERALL_FIELDNAMES, summary_rows)

    tables_dir = PICTURE_RESULT_DATA / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_copy, tables_dir / "new_single_result_summary.csv")
    print(f"exported picture_result data → {per_case_path}")


def write_metric_outputs(
    output_dir: Path,
    per_tile_rows: list[dict],
    per_case_rows: list[dict],
    summary_rows: list[dict],
    skipped_rows: list[dict],
) -> None:
    write_csv(output_dir / "per_tile_metrics.csv", PER_TILE_FIELDNAMES, per_tile_rows)
    write_csv(output_dir / "per_case_metrics.csv", PER_CASE_FIELDNAMES, per_case_rows)
    write_csv(output_dir / "overall_summary.csv", OVERALL_FIELDNAMES, summary_rows)
    write_csv(output_dir / "summary.csv", OVERALL_FIELDNAMES, summary_rows)
    write_by_patient_csvs(output_dir, per_case_rows)
    if skipped_rows:
        write_csv(output_dir / "skipped_tiles.csv", SKIPPED_FIELDNAMES, skipped_rows)


def write_pairwise_outputs(
    output_dir: Path,
    per_tile_rows: list[dict],
    per_case_rows: list[dict],
    summary_rows: list[dict],
) -> None:
    if not summary_rows:
        return
    write_csv(output_dir / "pairwise_generated_tile_metrics.csv", PAIRWISE_TILE_FIELDNAMES, per_tile_rows)
    write_csv(output_dir / "pairwise_generated_case_metrics.csv", PAIRWISE_CASE_FIELDNAMES, per_case_rows)
    write_csv(output_dir / "pairwise_generated_summary.csv", PAIRWISE_SUMMARY_FIELDNAMES, summary_rows)


def evaluate_experiment(
    experiment_root: Path,
    args: argparse.Namespace,
    metrics: set[str],
    no_ref: dict[str, object],
    device: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    experiment = experiment_root.name
    truth_root = (experiment_root / args.truth_subdir).resolve()
    if not truth_root.is_dir():
        raise SystemExit(f"truth root not found: {truth_root}")

    model_specs = resolve_eval_models(
        experiment_root,
        args.truth_subdir,
        args.input_subdir,
        args.model_dirs,
        args.exclude_dirs,
        args.include_input_baseline,
    )

    print(f"\nexperiment: {experiment}")
    print(f"GT: {truth_root}")
    print(f"models: {', '.join(label for _, label, _ in model_specs)}")

    t0 = time.perf_counter()
    truth_catalog = load_truth_catalog(truth_root, args.patient)
    if not truth_catalog:
        raise SystemExit(f"no truth tiles found under {truth_root}")
    print(f"truth catalog: {len(truth_catalog)} tiles ({time.perf_counter() - t0:.1f}s)")

    all_per_tile: list[dict] = []
    all_per_case: list[dict] = []
    all_summary: list[dict] = []
    all_skipped: list[dict] = []
    evaluated_results: list[ModelEvalResult] = []

    for model_dir, model_label, model_root in model_specs:
        t_model = time.perf_counter()
        print(f"evaluating {model_label} ({model_dir}) ...")
        result = evaluate_model(
            model_dir,
            model_label,
            model_root,
            truth_catalog,
            truth_root,
            metrics,
            no_ref,
            device,
            args.batch_size,
            args.metric_backend,
        )
        if result is None:
            continue
        evaluated_results.append(result)
        all_per_tile.extend(add_experiment(result.per_tile_rows, experiment))
        all_per_case.extend(add_experiment(result.per_case_rows, experiment))
        all_summary.extend(add_experiment([result.summary_row], experiment))
        all_skipped.extend(add_experiment(result.skipped, experiment))
        print(
            f"  {model_label}: {result.summary_row['n_cases']} cases, "
            f"{result.summary_row['n_tiles']} tiles, "
            f"{len(result.skipped)} skipped, "
            f"{time.perf_counter() - t_model:.1f}s"
        )

    pairwise_per_tile: list[dict] = []
    pairwise_per_case: list[dict] = []
    pairwise_summary: list[dict] = []
    if args.pairwise_generated and len(evaluated_results) >= 2:
        print("evaluating same-slice generated image pairs ...")
        pairwise_per_tile, pairwise_per_case, pairwise_summary = evaluate_pairwise_generated(
            experiment,
            evaluated_results,
            device,
            args.batch_size,
            args.metric_backend,
        )

    return (
        all_per_tile,
        all_per_case,
        all_summary,
        all_skipped,
        pairwise_per_tile,
        pairwise_per_case,
        pairwise_summary,
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    metrics = set(args.metrics)
    device = resolve_device(args.device)

    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    experiments = discover_experiments(root, args.truth_subdir, args.recursive_experiments)
    no_ref = prepare_no_reference_metrics(metrics, device)

    print(f"root: {root}")
    print(f"device: {device}, batch_size: {args.batch_size}")
    print(f"experiments: {', '.join(p.name for p in experiments)}")

    t0 = time.perf_counter()
    aggregate_per_tile: list[dict] = []
    aggregate_per_case: list[dict] = []
    aggregate_summary: list[dict] = []
    aggregate_skipped: list[dict] = []
    aggregate_pairwise_per_tile: list[dict] = []
    aggregate_pairwise_per_case: list[dict] = []
    aggregate_pairwise_summary: list[dict] = []

    for experiment_root in experiments:
        (
            per_tile,
            per_case,
            summary,
            skipped,
            pairwise_per_tile,
            pairwise_per_case,
            pairwise_summary,
        ) = evaluate_experiment(
            experiment_root,
            args,
            metrics,
            no_ref,
            device,
        )
        if summary:
            if args.output_dir and len(experiments) > 1:
                experiment_output_dir = (args.output_dir / experiment_root.name / "metrics").resolve()
            elif args.output_dir:
                experiment_output_dir = args.output_dir.resolve()
            else:
                experiment_output_dir = (experiment_root / "metrics").resolve()
            write_metric_outputs(experiment_output_dir, per_tile, per_case, summary, skipped)
            if args.pairwise_generated:
                write_pairwise_outputs(
                    experiment_output_dir,
                    pairwise_per_tile,
                    pairwise_per_case,
                    pairwise_summary,
                )

        aggregate_per_tile.extend(per_tile)
        aggregate_per_case.extend(per_case)
        aggregate_summary.extend(summary)
        aggregate_skipped.extend(skipped)
        aggregate_pairwise_per_tile.extend(pairwise_per_tile)
        aggregate_pairwise_per_case.extend(pairwise_per_case)
        aggregate_pairwise_summary.extend(pairwise_summary)

    if not aggregate_summary:
        print("no records found, skip writing csv")
        return 1

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (root / "metrics").resolve()
    )
    if len(experiments) > 1:
        write_metric_outputs(
            output_dir,
            aggregate_per_tile,
            aggregate_per_case,
            aggregate_summary,
            aggregate_skipped,
        )
        if args.pairwise_generated:
            write_pairwise_outputs(
                output_dir,
                aggregate_pairwise_per_tile,
                aggregate_pairwise_per_case,
                aggregate_pairwise_summary,
            )

    if args.export_to_picture_result:
        export_picture_result(aggregate_per_case, aggregate_summary, output_dir)

    print(
        f"done in {time.perf_counter() - t0:.1f}s: "
        f"{len(aggregate_summary)} models, "
        f"{len(aggregate_per_case)} per-case rows, "
        f"{len(aggregate_per_tile)} per-tile rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

