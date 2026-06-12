"""Run YOLOv8 inference on bare PCB images (folder or single file)."""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import repo_root  # noqa: E402

REPO = repo_root(__file__)


def parse_args():
    p = argparse.ArgumentParser(description="Bare PCB defect inference.")
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    p.add_argument("--source", type=str, required=True, help="Image, folder, or glob")
    p.add_argument("--output-dir", type=str, default="runs/inference/bare_pcb")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--iou", type=float, default=0.50)
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = REPO / weights
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=str(out_dir),
        name="predict",
        exist_ok=True,
    )
    print(f"[done] predictions saved under {out_dir / 'predict'}")


if __name__ == "__main__":
    main()
