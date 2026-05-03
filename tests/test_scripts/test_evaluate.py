"""Tests for the evaluation CLI helpers."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
import torch

from scripts.evaluate import (
    build_comparison_report,
    build_report,
    evaluate_model,
    parse_args,
    summarize_case_metrics,
)


def test_parse_args_supports_single_model_evaluation() -> None:
    args = parse_args(
        [
            "--model",
            "unet3d",
            "--checkpoint",
            "data/models/unet3d_best.pth",
            "--config",
            "configs/custom.yaml",
            "--output",
            "tmp/report.json",
            "--case-metrics-dir",
            "tmp/cases",
            "--device",
            "cpu",
            "--skip-hd95",
            "--limit",
            "3",
        ]
    )

    assert args.model == "unet3d"
    assert args.checkpoint == "data/models/unet3d_best.pth"
    assert args.config == "configs/custom.yaml"
    assert args.output == "tmp/report.json"
    assert args.case_metrics_dir == "tmp/cases"
    assert args.device == "cpu"
    assert args.skip_hd95 is True
    assert args.limit == 3


def test_parse_args_supports_compare_mode() -> None:
    args = parse_args(
        [
            "--compare",
            "--unet-checkpoint",
            "unet.pth",
            "--attention-checkpoint",
            "attention.pth",
        ]
    )

    assert args.compare is True
    assert args.unet_checkpoint == "unet.pth"
    assert args.attention_checkpoint == "attention.pth"


def test_parse_args_resolves_local_artifacts_for_compare() -> None:
    root = _fresh_tmp_dir()
    try:
        unet_path = root / "unet3d_placeholder_best.pth"
        attention_path = root / "attention_unet3d_pipeline_validation_best.pth"
        unet_path.touch()
        attention_path.touch()

        args = parse_args(["--compare", "--artifact-dir", str(root)])

        assert args.unet_checkpoint == str(unet_path)
        assert args.attention_checkpoint == str(attention_path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_parse_args_resolves_local_artifact_for_single_model() -> None:
    root = _fresh_tmp_dir()
    try:
        checkpoint_path = root / "attention_unet3d_pipeline_validation_best.pth"
        checkpoint_path.touch()

        args = parse_args(
            [
                "--model",
                "attention_unet3d",
                "--artifact-dir",
                str(root),
            ]
        )

        assert args.checkpoint == str(checkpoint_path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_parse_args_rejects_missing_single_checkpoint() -> None:
    with pytest.raises(ValueError, match="--model and --checkpoint"):
        parse_args(["--model", "unet3d", "--artifact-dir", "tmp/does_not_exist"])


def test_evaluate_model_uses_predictions_not_ground_truth() -> None:
    model = _AllBackgroundModel()
    images = torch.zeros(1, 4, 2, 2, 2)
    masks = torch.tensor([[[[1, 2], [3, 0]], [[1, 2], [3, 0]]]], dtype=torch.long)
    loader = [(images, masks, {"case_id": ["BraTS_test"]})]

    case_metrics, timing = evaluate_model(
        model=model,
        test_loader=loader,
        device="cpu",
        include_hd95=False,
    )

    assert case_metrics[0]["case_id"] == "BraTS_test"
    assert case_metrics[0]["WT_dice"] == 0.0
    assert case_metrics[0]["TC_dice"] == 0.0
    assert case_metrics[0]["ET_dice"] == 0.0
    assert case_metrics[0]["mean_dice"] == 0.0
    assert timing["total"] >= 0.0


def test_summarize_case_metrics_computes_stats_and_targets() -> None:
    summary = summarize_case_metrics(
        [
            {
                "case_id": "case_1",
                "WT_dice": 0.90,
                "TC_dice": 0.80,
                "ET_dice": 0.70,
                "mean_dice": 0.80,
                "inference_time_sec": 0.10,
            },
            {
                "case_id": "case_2",
                "WT_dice": 1.00,
                "TC_dice": 0.90,
                "ET_dice": 0.80,
                "mean_dice": 0.90,
                "inference_time_sec": 0.20,
            },
        ],
        model_name="unet3d",
        checkpoint_path="model.pth",
        checkpoint_epoch=5,
        total_params=123,
        timing={"mean": 0.15, "total": 0.30},
        targets={"WT": 0.88, "TC": 0.82, "ET": 0.78},
    )

    assert summary["model_name"] == "unet3d"
    assert summary["checkpoint_epoch"] == 5
    assert summary["total_cases"] == 2
    assert summary["mean_dice"] == pytest.approx(0.85)
    assert summary["metrics"]["WT_dice"]["mean"] == pytest.approx(0.95)
    assert summary["targets_met"] == {"WT": True, "TC": True, "ET": False}


def test_build_comparison_report_prefers_higher_mean_dice() -> None:
    summaries = {
        "unet3d": {
            "mean_dice": 0.80,
            "metrics": {"WT_dice": {"mean": 0.90}},
        },
        "attention_unet3d": {
            "mean_dice": 0.85,
            "metrics": {"WT_dice": {"mean": 0.95}},
        },
    }

    comparison = build_comparison_report(summaries, targets={"WT": 0.88})

    assert comparison["better_model_by_mean_dice"] == "attention_unet3d"
    assert comparison["mean_dice_difference_attention_minus_unet"] == pytest.approx(0.05)
    assert comparison["dice_differences_attention_minus_unet"]["WT"] == pytest.approx(0.05)


def test_build_report_records_models_and_hd95_setting() -> None:
    report = build_report(
        model_summaries={"unet3d": {"mean_dice": 0.8}},
        targets={"WT": 0.88},
        include_hd95=False,
    )

    assert report["evaluation"]["split"] == "test"
    assert report["evaluation"]["include_hd95"] is False
    assert report["evaluation"]["models_evaluated"] == ["unet3d"]
    assert "unet3d" in report["models"]


class _AllBackgroundModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, height, width = images.shape
        logits = torch.zeros(batch_size, 4, depth, height, width)
        logits[:, 0] = 1.0
        return logits


def _fresh_tmp_dir() -> Path:
    root = Path("tmp") / f"test_evaluate_script_{uuid.uuid4().hex}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root
