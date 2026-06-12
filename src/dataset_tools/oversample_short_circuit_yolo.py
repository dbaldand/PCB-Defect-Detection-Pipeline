"""
Oversample short_circuit (YOLO class id 3) in training split only.

Duplicates each train image whose label contains class 3 by N extra copies.
Val and test are copied unchanged.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import yaml

ROOT = REPO
SC_CLASS_ID = 3
EXTRA_COPIES = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str,
                   default=str(REPO / "data" / "mounted_pcb_yolo"))
    p.add_argument("--dst", type=str,
                   default=r"data/pcb-said")
    p.add_argument("--extra-copies", type=int, default=EXTRA_COPIES)
    p.add_argument("--report-dir", type=str,
                   default=str(ROOT / "outputs/yolov8_4class_sc_oversample"))
    return p.parse_args()


def has_short_circuit(label_path):
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            cls = int(line.split()[0])
            if cls == SC_CLASS_ID:
                return True
    return False


def copy_split(src_root, dst_root, split, extra_copies=0, oversample_train=False):
    src_img = src_root / "images" / split
    src_lbl = src_root / "labels" / split
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "instances": Counter(), "sc_images": 0, "sc_dupes": 0}

    for lbl_path in sorted(src_lbl.glob("*.txt")):
        stem = lbl_path.stem
        # find image (any common ext)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = src_img / (stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        lbl_text = lbl_path.read_text(encoding="utf-8")
        dst_lbl_file = dst_lbl / (stem + ".txt")
        shutil.copy2(img_path, dst_img / img_path.name)
        dst_lbl_file.write_text(lbl_text, encoding="utf-8")
        stats["images"] += 1
        for line in lbl_text.strip().splitlines():
            if line.strip():
                stats["instances"][int(line.split()[0])] += 1

        if oversample_train and has_short_circuit(lbl_path):
            stats["sc_images"] += 1
            for k in range(1, extra_copies + 1):
                dup_stem = f"{stem}_scdup{k}"
                dup_img_name = f"{dup_stem}{img_path.suffix}"
                shutil.copy2(img_path, dst_img / dup_img_name)
                (dst_lbl / f"{dup_stem}.txt").write_text(lbl_text, encoding="utf-8")
                stats["sc_dupes"] += 1
                stats["images"] += 1
                for line in lbl_text.strip().splitlines():
                    if line.strip():
                        stats["instances"][int(line.split()[0])] += 1

    return stats


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    train_stats = copy_split(src, dst, "train", args.extra_copies, oversample_train=True)
    val_stats = copy_split(src, dst, "val")
    test_stats = copy_split(src, dst, "test")

    yaml_path = dst / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({
            "path": str(dst).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 4,
            "names": {0: "well_soldered", 1: "item_missing",
                      2: "mis_soldered", 3: "short_circuit"},
        }, f, default_flow_style=False, sort_keys=False)

    report = report_dir / "dataset_oversampling_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Short-circuit train oversampling report\n\n")
        f.write(f"- Source: `{src}`\n")
        f.write(f"- Destination: `{dst}`\n")
        f.write(f"- Extra copies per SC train image: **{args.extra_copies}**\n")
        f.write("- Val/test: **unchanged**\n\n")
        f.write("## Train split\n\n")
        f.write(f"| Metric | Original | Oversampled |\n")
        f.write(f"|--------|----------|-------------|\n")
        orig_train_imgs = len(list((src / "images/train").glob("*.*")))
        f.write(f"| Images | {orig_train_imgs} | {train_stats['images']} |\n")
        f.write(f"| SC images duplicated | — | {train_stats['sc_images']} |\n")
        f.write(f"| SC duplicate files added | — | {train_stats['sc_dupes']} |\n")
        f.write(f"| short_circuit instances (train) | — | {train_stats['instances'][3]} |\n")
        f.write("\n## Val / test\n\n")
        f.write(f"- Val images: {val_stats['images']} (same as source)\n")
        f.write(f"- Test images: {test_stats['images']} (same as source)\n")

    print(f"Train images: {train_stats['images']}  (SC dupes: {train_stats['sc_dupes']})")
    print(f"Saved dataset -> {dst}")
    print(f"Saved report -> {report}")


if __name__ == "__main__":
    main()
