"""Train YOLOv8s on the mounted-PCB 4-class YOLO dataset."""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import repo_root  # noqa: E402

REPO = repo_root(__file__)


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on mounted PCB defects.")
    p.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    p.add_argument("--model", type=str, default="yolov8s.pt")
    p.add_argument("--output-dir", type=str, default="runs/mounted_pcb/yolov8best")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training in this script.")
    print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    device = args.device or 0

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        pretrained=True,
        device=device,
        verbose=True,
    )
    print(f"[done] training finished -> {out_dir}")


if __name__ == "__main__":
    main()
