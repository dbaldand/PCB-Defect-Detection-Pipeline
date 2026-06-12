"""Evaluate a mounted-PCB YOLO11 model on the test split (threshold sweep @ 0.30/0.40/0.50)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import greedy_match  # noqa: E402
from utils.paths import repo_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_yolov8_4class import (  # noqa: E402
    DEFECT_FR,
    THRESHOLDS,
    load_gt_for_split,
    per_class_metrics,
    run_predict,
)

REPO = repo_root(__file__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/mounted_pcb/yolo11_4class_best/weights/best.pt")
    p.add_argument("--data", default="mounted_data.yaml.example")
    p.add_argument("--output-dir", default="runs/mounted_pcb/yolo11_4class_best")
    p.add_argument("--iou-thres", type=float, default=0.50)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = REPO / weights

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO / data_path
    with open(data_path, encoding="utf-8") as f:
        yolo_root = Path(yaml.safe_load(f)["path"])

    model = YOLO(str(weights))
    samples = load_gt_for_split(yolo_root, "test")
    predictions = run_predict(model, samples)

    val_results = model.val(data=str(data_path), split="test", conf=0.4, verbose=False)
    map50 = float(val_results.box.map50) if val_results.box.map50 is not None else 0.0
    map5095 = float(val_results.box.map) if val_results.box.map is not None else 0.0

    thresh_rows, per_class_all = [], []
    for th in THRESHOLDS:
        _, pc = per_class_metrics(predictions, samples, 5, th, args.iou_thres)
        defects = [r for r in pc if r["class_idx"] in DEFECT_FR]
        sc = next(r for r in pc if r["class_name"] == "short_circuit")
        thresh_rows.append({
            "confidence_threshold": th,
            "macro_precision_defects": float(np.mean([r["precision"] for r in defects])),
            "macro_recall_defects": float(np.mean([r["recall"] for r in defects])),
            "macro_f1_defects": float(np.mean([r["f1"] for r in defects])),
            "short_circuit_precision": sc["precision"],
            "short_circuit_recall": sc["recall"],
            "short_circuit_f1": sc["f1"],
            "short_circuit_tp": sc["tp"],
            "short_circuit_fp": sc["fp"],
            "short_circuit_fn": sc["fn"],
        })
        for r in pc:
            per_class_all.append({**r, "confidence_threshold": th})

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(thresh_rows).to_csv(out_dir / "threshold_comparison.csv", index=False)
    pd.DataFrame(per_class_all).to_csv(out_dir / "per_class_metrics.csv", index=False)

    row040 = next(r for r in thresh_rows if r["confidence_threshold"] == 0.40)
    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_map_50": map50,
            "test_map_50_95": map5095,
            "macro_precision_defects_conf_040": row040["macro_precision_defects"],
            "macro_recall_defects_conf_040": row040["macro_recall_defects"],
            "macro_f1_defects_conf_040": row040["macro_f1_defects"],
            "conf_thres": 0.40,
            "iou_thres": args.iou_thres,
            "weights": str(weights),
            "eval_split": "test",
        }, f, indent=2)

    print(f"Test mAP@.50={map50:.4f}  mAP@[.50:.95]={map5095:.4f}")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
