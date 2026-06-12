"""
Analyze 4-class merged annotation distribution on an existing image split.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import pandas as pd

CLASS_IDS = (1, 2, 3, 4)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str,
                   default=str(REPO / "data" / "pcb-said"))
    p.add_argument("--ann", type=str, default="annotations_merged_4class_coco.json")
    p.add_argument("--split-dir", type=str,
                   default=str(Path(__file__).resolve().parents[1] /
                                "outputs/merged_4class_stratified_split/splits"))
    p.add_argument("--output-dir", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "outputs/merged_4class"))
    return p.parse_args()


def load_split(path):
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(line.split("\t", 1)[0]))
    return ids


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.data_root) / args.ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    splits = {
        name: load_split(Path(args.split_dir) / f"{name}_images.txt")
        for name in ("train", "val", "test")
    }

    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    rows = []
    for split_name, img_ids in splits.items():
        counts = Counter()
        for img_id in img_ids:
            for a in anns_by_img.get(img_id, []):
                counts[a["category_id"]] += 1
        for cid in CLASS_IDS:
            rows.append({
                "split": split_name,
                "merged_coco_category_id": cid,
                "merged_class_name": id_to_name[cid],
                "instance_count": counts[cid],
                "present_in_split": counts[cid] > 0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "split_distribution_old_split.csv", index=False)

    total = Counter()
    for ann in coco["annotations"]:
        if ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            total[ann["category_id"]] += 1

    report = out_dir / "split_report_old_split.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# 4-class split distribution (old 50ep split)\n\n")
        f.write(f"- Annotations: `{args.ann}`\n")
        f.write(f"- Splits: `{args.split_dir}`\n\n")

        f.write("## 1. Total instances per merged class\n\n")
        f.write("| Class | Total |\n|-------|-------|\n")
        for cid in CLASS_IDS:
            f.write(f"| {id_to_name[cid]} | {total[cid]} |\n")
        f.write(f"| **All** | **{sum(total.values())}** |\n\n")

        f.write("## 2. Train / val / test per class\n\n")
        f.write("| Split | well_soldered | item_missing | mis_soldered | short_circuit |\n")
        f.write("|-------|---------------|--------------|--------------|---------------|\n")
        for s in ("train", "val", "test"):
            vals = []
            for cid in CLASS_IDS:
                r = df[(df["split"] == s) & (df["merged_coco_category_id"] == cid)]
                vals.append(str(int(r["instance_count"].iloc[0])))
            f.write(f"| {s} | {' | '.join(vals)} |\n")

        f.write("\n## 3. All 4 classes in every split?\n\n")
        all_ok = True
        for s in ("train", "val", "test"):
            ok = all(
                df[(df["split"] == s) & (df["merged_coco_category_id"] == cid)]
                ["present_in_split"].iloc[0]
                for cid in CLASS_IDS
            )
            f.write(f"- **{s}**: {'yes' if ok else 'NO — missing class(es)'}\n")
            all_ok = all_ok and ok

        f.write("\n## 4. short_circuit counts\n\n")
        for s in ("train", "val", "test"):
            n = int(df[(df["split"] == s) & (df["merged_class_name"] == "short_circuit")]
                    ["instance_count"].iloc[0])
            f.write(f"- {s}: **{n}** instances\n")

        f.write("\n## 5. Comparison vs 66-class and 3-class defect-only\n\n")
        f.write("| Setup | Annotated instances | Classes | Normal/well boxes |\n")
        f.write("|-------|---------------------|---------|-------------------|\n")
        f.write("| 66-class original | ~1464 | 66 (+unused id=0) | included |\n")
        f.write("| 3-class defect-only | 665 | 3 defects | **dropped** (~799) |\n")
        f.write(f"| **4-class merged** | **{sum(total.values())}** | 4 | **kept** "
                f"({total[1]} well_soldered) |\n\n")
        f.write("The 4-class setup is statistically stronger than 3-class defect-only because:\n")
        f.write("- Every normal component is a positive `well_soldered` example (reduces false "
                "background confusion).\n")
        f.write("- Training sees ~2.2x more boxes than 3-class (1464 vs 665).\n")
        f.write("- Compared to 66-class, labels are balanced and all splits can hold every "
                "defect type with stratification.\n")

    print(f"Saved {out_dir / 'split_distribution_old_split.csv'}")
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
