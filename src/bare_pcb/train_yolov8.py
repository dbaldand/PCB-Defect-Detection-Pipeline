"""Train YOLOv8s on a Roboflow-exported bare PCB defect dataset (YOLO format)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import repo_root  # noqa: E402

REPO = repo_root(__file__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8s on bare PCB defects.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=REPO / "data" / "bare_pcb",
        help="Folder containing data.yaml and train/valid/test splits.",
    )
    p.add_argument("--model", type=str, default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--output-dir", type=Path, default=REPO / "runs" / "bare_pcb")
    p.add_argument("--name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"Missing dataset config: {data_yaml}")

    run_name = args.name or f"pcb_defect_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project = args.output_dir.resolve()
    project.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "data": str(data_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "project": str(project),
        "name": run_name,
    }
    run_dir = project / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(project),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        plots=True,
        val=True,
        seed=42,
    )

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        eval_model = YOLO(str(best))
        bench_data = root / "_bench_data.yaml"
        bench_data.write_text(
            f"path: {root.as_posix()}\n"
            "train: train/images\nval: valid/images\ntest: test/images\n"
            "nc: 6\n"
            "names: ['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']\n",
            encoding="utf-8",
        )
        test_metrics = eval_model.val(
            data=str(bench_data),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            verbose=False,
        )
        summary = {
            "run": run_name,
            "best_weights": str(best),
            "test_map50": float(test_metrics.box.map50),
            "test_map50_95": float(test_metrics.box.map),
            "test_precision": float(test_metrics.box.mp),
            "test_recall": float(test_metrics.box.mr),
            "speed_ms": test_metrics.speed,
        }
        with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        print(json.dumps(summary, indent=2))

    print(f"Done. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
