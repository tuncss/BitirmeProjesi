"""Evaluate trained segmentation checkpoints on the BraTS test split.

Usage:
    python scripts/evaluate.py --model attention_unet3d \
        --checkpoint data/models/attention_unet3d_best.pth

    python scripts/evaluate.py --compare \
        --unet-checkpoint data/models/unet3d_best.pth \
        --attention-checkpoint data/models/attention_unet3d_best.pth

    # Uses local artifacts from ../model_artifacts when checkpoints are omitted.
    python scripts/evaluate.py --compare --skip-hd95 --limit 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import create_model, load_checkpoint
from src.data.dataset import get_dataloaders
from src.training import ExperimentConfig, compute_brats_metrics


MODEL_CHOICES = ("unet3d", "attention_unet3d")
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT.parent / "model_artifacts"
LOCAL_ARTIFACT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "unet3d": (
        "unet3d_pipeline_validation_best.pth",
        "unet3d_placeholder_best.pth",
        "unet3d_best.pth",
    ),
    "attention_unet3d": (
        "attention_unet3d_pipeline_validation_best.pth",
        "attention_unet3d_best.pth",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate brain tumor segmentation checkpoints")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="Model architecture for single-checkpoint evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for single-checkpoint evaluation",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Evaluate and compare U-Net and Attention U-Net checkpoints",
    )
    parser.add_argument(
        "--unet-checkpoint",
        default=None,
        help="U-Net checkpoint path for --compare",
    )
    parser.add_argument(
        "--attention-checkpoint",
        default=None,
        help="Attention U-Net checkpoint path for --compare",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help=(
            "Directory used to auto-resolve local checkpoints when explicit "
            "checkpoint paths are omitted"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output",
        default="data/models/evaluation_report.json",
        help="Path where the evaluation report JSON will be written",
    )
    parser.add_argument(
        "--case-metrics-dir",
        default=None,
        help="Directory for per-case metric JSON files. Defaults to output directory.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Evaluation device",
    )
    parser.add_argument(
        "--skip-hd95",
        action="store_true",
        help="Skip HD95 metrics for faster smoke evaluations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test cases for quick smoke checks",
    )
    args = parser.parse_args(argv)
    _resolve_missing_checkpoint_args(args)
    _validate_args(args)
    return args


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: str | torch.device,
    include_hd95: bool = True,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run model inference on the test loader and return per-case metrics."""
    target_device = torch.device(device)
    device_type = target_device.type
    model = model.to(target_device)
    model.eval()

    case_metrics: list[dict[str, Any]] = []
    inference_times: list[float] = []

    with torch.no_grad():
        progress = tqdm(test_loader, desc="Evaluating", leave=False)
        for images, masks, metadata in progress:
            images = images.to(target_device, non_blocking=True)

            started = time.perf_counter()
            with torch.amp.autocast(device_type, enabled=device_type == "cuda"):
                logits = model(images)
            pred_labels = logits.argmax(dim=1).cpu().numpy()
            elapsed = time.perf_counter() - started

            masks_np = masks.cpu().numpy()
            case_ids = _case_ids_from_metadata(metadata, batch_size=pred_labels.shape[0])
            per_sample_time = elapsed / max(pred_labels.shape[0], 1)

            for sample_idx, case_id in enumerate(case_ids):
                metrics = compute_brats_metrics(
                    pred_labels[sample_idx],
                    masks_np[sample_idx],
                    include_hd95=include_hd95,
                )
                case_metrics.append(
                    {
                        "case_id": case_id,
                        **{key: float(value) for key, value in metrics.items()},
                        "inference_time_sec": float(per_sample_time),
                    }
                )
                inference_times.append(float(per_sample_time))

                if limit is not None and len(case_metrics) >= limit:
                    return case_metrics, _summarize_inference_times(inference_times)

    return case_metrics, _summarize_inference_times(inference_times)


def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: str | Path,
    config: ExperimentConfig,
    test_loader,
    device: str | torch.device,
    include_hd95: bool = True,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one checkpoint and return summary plus per-case metrics."""
    model = create_model(model_name, config)
    checkpoint = load_checkpoint(model, checkpoint_path, str(device))
    total_params = sum(parameter.numel() for parameter in model.parameters())

    logger.info(f"Evaluating {model_name}: {checkpoint_path}")
    case_metrics, timing = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        include_hd95=include_hd95,
        limit=limit,
    )

    summary = summarize_case_metrics(
        case_metrics,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint.get("epoch"),
        total_params=total_params,
        timing=timing,
        targets=config.evaluation.target_dice,
    )
    return summary, case_metrics


def summarize_case_metrics(
    case_metrics: list[dict[str, Any]],
    model_name: str,
    checkpoint_path: str | Path,
    checkpoint_epoch: int | None,
    total_params: int,
    timing: dict[str, float] | None,
    targets: dict[str, float],
) -> dict[str, Any]:
    """Create aggregate statistics for a model's per-case metrics."""
    metric_keys = sorted(
        key
        for key in case_metrics[0]
        if key not in {"case_id", "inference_time_sec"}
    ) if case_metrics else []

    metrics = {
        key: _summarize_values([float(case_metric[key]) for case_metric in case_metrics])
        for key in metric_keys
    }
    targets_met = {
        region: metrics.get(f"{region}_dice", {}).get("mean", -math.inf) >= target
        for region, target in targets.items()
    }

    return {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "total_params": total_params,
        "total_cases": len(case_metrics),
        "metrics": metrics,
        "mean_dice": metrics.get("mean_dice", {}).get("mean"),
        "targets_met": targets_met,
        "inference_time_sec": timing or {},
    }


def build_comparison_report(
    model_summaries: dict[str, dict[str, Any]],
    targets: dict[str, float],
) -> dict[str, Any]:
    """Build a compact comparison section for U-Net vs Attention U-Net."""
    comparison: dict[str, Any] = {
        "dice_differences_attention_minus_unet": {},
        "targets": targets,
        "better_model_by_mean_dice": None,
    }

    unet = model_summaries.get("unet3d")
    attention = model_summaries.get("attention_unet3d")
    if unet is None or attention is None:
        return comparison

    for region in targets:
        metric_name = f"{region}_dice"
        unet_mean = _metric_mean(unet, metric_name)
        attention_mean = _metric_mean(attention, metric_name)
        comparison["dice_differences_attention_minus_unet"][region] = (
            attention_mean - unet_mean
            if unet_mean is not None and attention_mean is not None
            else None
        )

    unet_mean_dice = unet.get("mean_dice")
    attention_mean_dice = attention.get("mean_dice")
    if unet_mean_dice is not None and attention_mean_dice is not None:
        comparison["mean_dice_difference_attention_minus_unet"] = attention_mean_dice - unet_mean_dice
        comparison["better_model_by_mean_dice"] = (
            "attention_unet3d" if attention_mean_dice >= unet_mean_dice else "unet3d"
        )

    return comparison


def build_report(
    model_summaries: dict[str, dict[str, Any]],
    targets: dict[str, float],
    include_hd95: bool,
) -> dict[str, Any]:
    """Build the final evaluation report payload."""
    return {
        "evaluation": {
            "split": "test",
            "include_hd95": include_hd95,
            "models_evaluated": list(model_summaries),
        },
        "models": model_summaries,
        "comparison": build_comparison_report(model_summaries, targets),
    }


def save_json(payload: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=True)
        file.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    include_hd95 = not args.skip_hd95
    device = _resolve_device(args.device)

    loaders = get_dataloaders(config.to_dict(), seed=config.seed)
    test_loader = loaders["test"]

    output_path = Path(args.output)
    case_metrics_dir = Path(args.case_metrics_dir) if args.case_metrics_dir else output_path.parent

    runs: list[tuple[str, str | Path]]
    if args.compare:
        runs = [
            ("unet3d", args.unet_checkpoint),
            ("attention_unet3d", args.attention_checkpoint),
        ]
    else:
        runs = [(args.model, args.checkpoint)]

    summaries: dict[str, dict[str, Any]] = {}
    for model_name, checkpoint_path in runs:
        config.model_name = model_name
        summary, case_metrics = evaluate_checkpoint(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config=config,
            test_loader=test_loader,
            device=device,
            include_hd95=include_hd95,
            limit=args.limit,
        )
        summaries[model_name] = summary
        save_json(case_metrics, case_metrics_dir / f"{model_name}_case_metrics.json")

    report = build_report(
        model_summaries=summaries,
        targets=config.evaluation.target_dice,
        include_hd95=include_hd95,
    )
    save_json(report, output_path)
    logger.info(f"Evaluation report saved: {output_path}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

    if args.compare:
        missing = [
            name
            for name, value in {
                "--unet-checkpoint": args.unet_checkpoint,
                "--attention-checkpoint": args.attention_checkpoint,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"--compare requires: {', '.join(missing)}")
        return

    if args.model is None or args.checkpoint is None:
        raise ValueError("single-model evaluation requires --model and --checkpoint")


def _resolve_missing_checkpoint_args(args: argparse.Namespace) -> None:
    artifact_dir = Path(args.artifact_dir)

    if args.compare:
        if args.unet_checkpoint is None:
            args.unet_checkpoint = _find_local_checkpoint("unet3d", artifact_dir)
        if args.attention_checkpoint is None:
            args.attention_checkpoint = _find_local_checkpoint("attention_unet3d", artifact_dir)
        return

    if args.model is not None and args.checkpoint is None:
        args.checkpoint = _find_local_checkpoint(args.model, artifact_dir)


def _find_local_checkpoint(model_name: str, artifact_dir: Path) -> str | None:
    for filename in LOCAL_ARTIFACT_CANDIDATES[model_name]:
        candidate = artifact_dir / filename
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _case_ids_from_metadata(metadata: Any, batch_size: int) -> list[str]:
    if isinstance(metadata, dict) and "case_id" in metadata:
        raw_case_ids = metadata["case_id"]
        if isinstance(raw_case_ids, str):
            case_ids = [raw_case_ids]
        elif isinstance(raw_case_ids, (list, tuple)):
            case_ids = [str(case_id) for case_id in raw_case_ids]
        else:
            case_ids = [str(raw_case_ids)]
    else:
        case_ids = []

    if len(case_ids) == batch_size:
        return case_ids
    return [f"case_{index:05d}" for index in range(batch_size)]


def _summarize_values(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]

    summary: dict[str, Any] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "inf_count": int(np.isinf(array).sum()),
        "nan_count": int(np.isnan(array).sum()),
    }
    if finite.size == 0:
        summary.update({"mean": None, "std": None, "median": None, "min": None, "max": None})
        return summary

    summary.update(
        {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    )
    return summary


def _summarize_inference_times(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "total": 0.0}
    return {
        "mean": float(np.mean(values)),
        "total": float(np.sum(values)),
    }


def _metric_mean(summary: dict[str, Any], metric_name: str) -> float | None:
    metric = summary.get("metrics", {}).get(metric_name, {})
    return metric.get("mean")


if __name__ == "__main__":
    raise SystemExit(main())
