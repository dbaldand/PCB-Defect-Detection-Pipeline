"""
Create merged 4-class PCB-SAID COCO annotations (includes well_soldered).

  1 = well_soldered
  2 = item_missing
  3 = mis_soldered
  4 = short_circuit
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

MERGED_CATEGORIES = [
    {
        "id": 1,
        "name": "well_soldered",
        "display_name": "Well Soldered",
        "supercategory": "pcb_assembly",
    },
    {
        "id": 2,
        "name": "item_missing",
        "display_name": "Item Missing",
        "supercategory": "pcb_assembly_defect",
    },
    {
        "id": 3,
        "name": "mis_soldered",
        "display_name": "Mis-soldered / Poorly Soldered",
        "supercategory": "pcb_assembly_defect",
    },
    {
        "id": 4,
        "name": "short_circuit",
        "display_name": "Short Circuit",
        "supercategory": "pcb_assembly_defect",
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Merge PCB-SAID to 4 classes.")
    p.add_argument("--data-root", type=str,
                   default=str(REPO / "data" / "pcb-said"))
    p.add_argument("--ann-in", type=str, default="annotations_coco.json")
    p.add_argument("--ann-out", type=str, default="annotations_merged_4class_coco.json")
    p.add_argument("--output-dir", type=str, default=str(REPO / "runs" / "mounted_pcb" / "merged_4class"))
    return p.parse_args()


def classify_original(cat_id, cat_name):
    """Return (merged_id, merged_name, action)."""
    if cat_id == 0 or cat_name == "SMD Components":
        return None, None, "drop_supercategory"
    if cat_name == "Short Circuit":
        return 4, "short_circuit", "keep"
    if cat_name == "Power Inductor Correct":
        return 1, "well_soldered", "keep"
    if "Well Soldered" in cat_name:
        return 1, "well_soldered", "keep"
    if "Missing" in cat_name:
        return 2, "item_missing", "keep"
    if "Poorly Soldered" in cat_name:
        return 3, "mis_soldered", "keep"
    return None, None, "drop_unmapped"


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    ann_in = data_root / args.ann_in
    ann_out = data_root / args.ann_out
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_in, "r", encoding="utf-8") as f:
        src = json.load(f)

    id_to_name = {c["id"]: c["name"] for c in src["categories"]}
    valid_image_ids = {im["id"] for im in src["images"]}

    mapping_rows = []
    kept_anns = []
    dropped_super = 0
    dropped_unmapped = 0
    dropped_invalid_bbox = 0

    for ann in src["annotations"]:
        old_cid = ann["category_id"]
        old_name = id_to_name.get(old_cid, f"unknown_{old_cid}")
        merged_id, merged_name, action = classify_original(old_cid, old_name)

        mapping_rows.append({
            "original_coco_category_id": old_cid,
            "original_class_name": old_name,
            "action": action,
            "merged_coco_category_id": merged_id if merged_id else "",
            "merged_class_name": merged_name if merged_name else "",
        })

        if action == "drop_supercategory":
            dropped_super += 1
            continue
        if action == "drop_unmapped":
            dropped_unmapped += 1
            continue

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            dropped_invalid_bbox += 1
            continue
        if ann["image_id"] not in valid_image_ids:
            continue

        new_ann = dict(ann)
        new_ann["category_id"] = merged_id
        if "area" not in new_ann or new_ann["area"] is None:
            new_ann["area"] = float(w * h)
        kept_anns.append(new_ann)

    for ann in kept_anns:
        assert ann["category_id"] in (1, 2, 3, 4)
        assert ann["image_id"] in valid_image_ids
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0

    class_counts = Counter(a["category_id"] for a in kept_anns)
    old_ids_in_kept = {a["category_id"] for a in kept_anns}
    assert old_ids_in_kept <= {1, 2, 3, 4}

    merged = {
        "info": src.get("info", {}),
        "licenses": src.get("licenses", []),
        "images": src["images"],
        "categories": MERGED_CATEGORIES,
        "annotations": kept_anns,
    }
    desc = merged.get("info", {}).get("description", "")
    merged["info"]["description"] = (
        f"{desc} | Merged 4-class: well_soldered, item_missing, mis_soldered, short_circuit"
    ).strip(" |")

    with open(ann_out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    map_df = pd.DataFrame(mapping_rows).drop_duplicates(
        subset=["original_coco_category_id", "original_class_name", "action"]
    )
    map_df.to_csv(out_dir / "merge_mapping.csv", index=False)

    counts_df = pd.DataFrame([
        {"merged_coco_category_id": c["id"], "merged_class_name": c["name"],
         "display_name": c["display_name"], "instance_count": class_counts[c["id"]]}
        for c in MERGED_CATEGORIES
    ])
    counts_df.to_csv(out_dir / "merged_class_counts.csv", index=False)

    report_path = out_dir / "merge_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Merged 4-class PCB-SAID annotation report\n\n")
        f.write(f"- **Input:** `{ann_in}`\n")
        f.write(f"- **Output:** `{ann_out}`\n\n")
        f.write("## Mapping rules\n\n")
        f.write("- `Well Soldered` in name, or `Power Inductor Correct` -> well_soldered\n")
        f.write("- `Missing` in name -> item_missing\n")
        f.write("- `Poorly Soldered` in name -> mis_soldered\n")
        f.write("- Exact `Short Circuit` -> short_circuit\n")
        f.write("- Unused supercategory id=0 / SMD Components -> dropped\n\n")
        f.write("## Counts\n\n")
        f.write(f"| Metric | Count |\n|--------|-------|\n")
        f.write(f"| Original annotations | {len(src['annotations'])} |\n")
        f.write(f"| Kept annotations | {len(kept_anns)} |\n")
        f.write(f"| Dropped supercategory | {dropped_super} |\n")
        f.write(f"| Dropped unmapped | {dropped_unmapped} |\n")
        f.write(f"| Dropped invalid bbox | {dropped_invalid_bbox} |\n")
        f.write(f"| Images | {len(valid_image_ids)} |\n\n")
        f.write("## Merged class instances\n\n")
        f.write("| id | name | instances |\n|----|------|----------|\n")
        for c in MERGED_CATEGORIES:
            f.write(f"| {c['id']} | {c['name']} | {class_counts[c['id']]} |\n")

    print("=" * 60)
    print("Merged 4-class PCB-SAID annotations")
    print("=" * 60)
    for c in MERGED_CATEGORIES:
        print(f"  {c['name']}: {class_counts[c['id']]}")
    print(f"  total kept: {len(kept_anns)}")
    print(f"\nSaved: {ann_out}")


if __name__ == "__main__":
    main()
