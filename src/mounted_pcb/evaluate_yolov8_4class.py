"""Evaluate a mounted-PCB YOLOv8 model on the test split (threshold sweep @ 0.30/0.40/0.50)."""

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

REPO = repo_root(__file__)
THRESHOLDS = [0.30, 0.40, 0.50]
YOLO_TO_FR = {0: 1, 1: 2, 2: 3, 3: 4}
FR_NAMES = {0: "__background__", 1: "well_soldered", 2: "item_missing",
            3: "mis_soldered", 4: "short_circuit"}
DEFECT_FR = (2, 3, 4)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="runs/mounted_pcb/yolov8best/weights/best.pt")
    p.add_argument("--data", type=str, default="mounted_data.yaml.example")
    p.add_argument("--output-dir", type=str, default="runs/mounted_pcb/yolov8best")
    p.add_argument("--iou-thres", type=float, default=0.50)
    return p.parse_args()


def load_gt_for_split(yolo_root: Path, split: str):
    from PIL import Image

    img_dir = yolo_root / "images" / split
    lbl_dir = yolo_root / "labels" / split
    samples = []
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        iw, ih = Image.open(img_path).size
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes, labels = [], []
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c, xc, yc, w, h = int(parts[0]), *map(float, parts[1:5])
                boxes.append([(xc - w / 2) * iw, (yc - h / 2) * ih,
                              (xc + w / 2) * iw, (yc + h / 2) * ih])
                labels.append(YOLO_TO_FR[c])
        samples.append({
            "path": str(img_path),
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4)),
            "labels": np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
        })
    return samples


def per_class_metrics(predictions, gt_samples, num_classes, conf_thres, iou_thres):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(predictions, gt_samples):
        keep = pred["scores"] >= conf_thres
        p_boxes, p_scores, p_labels = pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]
        g_boxes, g_labels = gt["boxes"], gt["labels"]
        matches = greedy_match(p_boxes, p_scores, p_labels, g_boxes, g_labels, iou_thres)
        matched_p, matched_g = {p for p, _ in matches}, {g for _, g in matches}
        for p, g in matches:
            cm[int(g_labels[g]), int(p_labels[p])] += 1
        for pi in range(len(p_labels)):
            if pi not in matched_p:
                cm[0, int(p_labels[pi])] += 1
        for gi in range(len(g_labels)):
            if gi not in matched_g:
                cm[int(g_labels[gi]), 0] += 1
    rows = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({"class_idx": c, "class_name": FR_NAMES[c],
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1})
    return cm, rows


def run_predict(model, samples, conf_infer=0.001):
    preds = []
    for s in samples:
        res = model.predict(s["path"], conf=conf_infer, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            preds.append({"boxes": np.zeros((0, 4)), "scores": np.zeros(0),
                          "labels": np.zeros(0, dtype=np.int64)})
            continue
        fr_labels = np.array([YOLO_TO_FR[int(c)] for c in boxes.cls.cpu().numpy().astype(int)],
                             dtype=np.int64)
        preds.append({"boxes": boxes.xyxy.cpu().numpy(), "scores": boxes.conf.cpu().numpy(),
                      "labels": fr_labels})
    return preds


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
    final = {
        "test_map_50": map50,
        "test_map_50_95": map5095,
        "macro_precision_defects_conf_040": row040["macro_precision_defects"],
        "macro_recall_defects_conf_040": row040["macro_recall_defects"],
        "conf_thres": 0.40,
        "iou_thres": args.iou_thres,
        "weights": str(weights),
    }
    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print(f"Test mAP@.50={map50:.4f}  mAP@[.50:.95]={map5095:.4f}")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
