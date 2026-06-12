"""Run YOLO11 inference on mounted PCB images (optional defect-only filter)."""

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import repo_root  # noqa: E402

REPO = repo_root(__file__)
DEFECT_IDS = {1, 2, 3}


def parse_args():
    p = argparse.ArgumentParser(description="Mounted PCB YOLO11 inference.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--output-dir", type=str, default="runs/inference/mounted_yolo11")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--defect-only", action="store_true")
    p.add_argument("--max-images", type=int, default=24)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO / data_path
    with open(data_path, encoding="utf-8") as f:
        root = Path(yaml.safe_load(f)["path"])
    images = sorted((root / "images" / args.split).glob("*.*"))[: args.max_images]

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = REPO / weights
    model = YOLO(str(weights))
    classes = list(DEFECT_IDS) if args.defect_only else None
    for img_path in images:
        model.predict(
            source=str(img_path),
            conf=args.conf,
            save=True,
            project=str(out_dir),
            name=args.split,
            exist_ok=True,
            classes=classes,
        )
    print(f"[done] -> {out_dir / args.split}")


if __name__ == "__main__":
    main()
