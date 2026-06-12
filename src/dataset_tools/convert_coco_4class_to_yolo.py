"""
Convert merged 4-class COCO annotations to Ultralytics YOLO format.
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import pandas as pd
import yaml

ROOT = REPO

COCO_TO_YOLO = {1: 0, 2: 1, 3: 2, 4: 3}
YOLO_NAMES = {
    0: "well_soldered",
    1: "item_missing",
    2: "mis_soldered",
    3: "short_circuit",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str,
                   default=str(REPO / "data" / "pcb-said"))
    p.add_argument("--ann", type=str, default="annotations_merged_4class_coco.json")
    p.add_argument("--split-dir", type=str,
                   default=str(REPO / "runs" / "mounted_pcb" / "splits"))
    p.add_argument("--yolo-root", type=str,
                   default=str(REPO / "data" / "mounted_pcb_yolo"))
    p.add_argument("--report-dir", type=str,
                   default=str(REPO / "runs" / "mounted_pcb" / "reports"))
    return p.parse_args()


def load_split(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                img_id, fname = line.split("\t", 1)
                mapping[int(img_id)] = fname
    return mapping


def coco_to_yolo_line(bbox, img_w, img_h, yolo_cls):
    x, y, w, h = bbox
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{yolo_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    yolo_root = Path(args.yolo_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(data_root / args.ann, encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
            continue
        anns_by_image[ann["image_id"]].append(ann)

    split_dir = Path(args.split_dir)
    splits = {
        "train": load_split(split_dir / "train_images.txt"),
        "val": load_split(split_dir / "val_images.txt"),
        "test": load_split(split_dir / "test_images.txt"),
    }

    errors = []
    class_counts = Counter()
    split_counts = []

    for split_name, id_to_file in splits.items():
        img_out = yolo_root / "images" / split_name
        lbl_out = yolo_root / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id, fname in id_to_file.items():
            info = images_by_id[img_id]
            src = data_root / "images" / fname
            dst_img = img_out / fname
            if not src.exists():
                errors.append(f"missing image: {src}")
                continue
            shutil.copy2(src, dst_img)

            iw, ih = info["width"], info["height"]
            lines = []
            for ann in anns_by_image.get(img_id, []):
                cid = ann["category_id"]
                if cid not in COCO_TO_YOLO:
                    errors.append(f"bad category {cid} on image {img_id}")
                    continue
                yc = COCO_TO_YOLO[cid]
                line = coco_to_yolo_line(ann["bbox"], iw, ih, yc)
                parts = [float(x) for x in line.split()]
                if not (0 <= parts[1] <= 1 and 0 <= parts[2] <= 1
                        and 0 < parts[3] <= 1 and 0 < parts[4] <= 1):
                    errors.append(f"coords out of range: {fname} {line}")
                if yc not in range(4):
                    errors.append(f"bad yolo class: {yc}")
                lines.append(line)
                class_counts[(split_name, yc)] += 1

            lbl_path = lbl_out / (Path(fname).stem + ".txt")
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            if not lbl_path.exists():
                errors.append(f"missing label file for {fname}")
            split_counts.append({
                "split": split_name,
                "images": len(id_to_file),
                "labels_written": 1,
                "instances": len(lines),
            })

    # Per-split image counts (aggregate)
    split_rows = []
    for split_name in ("train", "val", "test"):
        n_img = len(splits[split_name])
        n_inst = sum(v for (s, _), v in class_counts.items() if s == split_name)
        split_rows.append({"split": split_name, "images": n_img, "instances": n_inst})
    pd.DataFrame(split_rows).to_csv(report_dir / "split_counts_yolo.csv", index=False)

    class_rows = []
    for split_name in ("train", "val", "test"):
        for yid, name in YOLO_NAMES.items():
            class_rows.append({
                "split": split_name,
                "yolo_class_id": yid,
                "class_name": name,
                "count": class_counts[(split_name, yid)],
            })
    pd.DataFrame(class_rows).to_csv(report_dir / "class_counts_yolo.csv", index=False)

    yaml_path = yolo_root / "dataset.yaml"
    yolo_path_str = str(yolo_root).replace("\\", "/")
    ds_yaml = {
        "path": yolo_path_str,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: YOLO_NAMES[i] for i in range(4)},
        "nc": 4,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(ds_yaml, f, default_flow_style=False, sort_keys=False)

    # Validate all images have labels
    for split_name in ("train", "val", "test"):
        img_dir = yolo_root / "images" / split_name
        lbl_dir = yolo_root / "labels" / split_name
        for img_path in img_dir.iterdir():
            lbl = lbl_dir / (img_path.stem + ".txt")
            if not lbl.exists():
                errors.append(f"no label for {img_path.name}")

    report = report_dir / "dataset_conversion_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# YOLO 4-class dataset conversion report\n\n")
        f.write(f"- COCO: `{data_root / args.ann}`\n")
        f.write(f"- YOLO root: `{yolo_root}`\n")
        f.write(f"- Split: `{split_dir}`\n\n")
        f.write("## Class mapping (YOLO 0-based)\n\n")
        for yid, name in YOLO_NAMES.items():
            f.write(f"- {yid} = {name} (COCO id {yid + 1})\n")
        f.write("\n## Split counts\n\n")
        f.write(pd.DataFrame(split_rows).to_string(index=False))
        f.write(f"\n\n## Validation errors: {len(errors)}\n\n")
        for e in errors[:50]:
            f.write(f"- {e}\n")
        if len(errors) > 50:
            f.write(f"- ... and {len(errors) - 50} more\n")
        f.write(f"\n**Status:** {'FAILED' if errors else 'OK'}\n")

    if errors:
        print(f"WARNING: {len(errors)} validation issues (see report)")
    print(f"Saved YOLO dataset -> {yolo_root}")
    print(f"Saved report -> {report}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
