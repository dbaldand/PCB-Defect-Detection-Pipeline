"""
Image-level stratified 70/15/15 split for 4-class PCB-SAID.

Optimizes class balance across train/val/test with random search (5000 seeds).
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd

CLASS_IDS = (1, 2, 3, 4)
CLASS_NAMES = {1: "well_soldered", 2: "item_missing", 3: "mis_soldered", 4: "short_circuit"}
TARGET_RATIO = (0.70, 0.15, 0.15)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str,
                   default=str(REPO / "data" / "pcb-said"))
    p.add_argument("--ann", type=str, default="annotations_merged_4class_coco.json")
    p.add_argument("--old-split-dir", type=str,
                   default=str(REPO / "runs" / "mounted_pcb" / "splits"))
    p.add_argument("--output-dir", type=str,
                   default=str(REPO / "runs" / "mounted_pcb" / "stratified_split"))
    p.add_argument("--n-candidates", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split_counts(split_dir, anns_by_img):
    rows = []
    for split in ("train", "val", "test"):
        path = Path(split_dir) / f"{split}_images.txt"
        ids = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(int(line.split("\t", 1)[0]))
        c = Counter()
        for iid in ids:
            for a in anns_by_img.get(iid, []):
                c[a["category_id"]] += 1
        rows.append({"split": split, **{CLASS_NAMES[k]: c[k] for k in CLASS_IDS}})
    return pd.DataFrame(rows)


def image_class_flags(anns_by_img, image_ids):
    """Per image: which classes appear (instance-level presence)."""
    flags = {}
    for iid in image_ids:
        present = set()
        for a in anns_by_img.get(iid, []):
            present.add(a["category_id"])
        flags[iid] = present
    return flags


def split_sizes(n, ratio=TARGET_RATIO):
    n_train = int(round(n * ratio[0]))
    n_val = int(round(n * ratio[1]))
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def assign_stratified(image_ids, flags, rng):
    """Greedy assignment: sort images by rarity, assign to split minimizing imbalance."""
    n = len(image_ids)
    n_train, n_val, n_test = split_sizes(n)
    caps = {"train": n_train, "val": n_val, "test": n_test}
    counts = {s: Counter() for s in caps}
    assigned = {}

    def rarity(iid):
        return len(flags[iid])

    ordered = sorted(image_ids, key=lambda i: (rarity(i), -i))
    for iid in ordered:
        present = flags[iid]
        best_split = None
        best_score = float("inf")
        for s in caps:
            if len([x for x in assigned.values() if x == s]) >= caps[s]:
                continue
            trial = counts[s].copy()
            for c in present:
                trial[c] += 1
            # penalize missing required classes in val/test
            score = 0.0
            for c in CLASS_IDS:
                score += abs(trial[c] - counts["train"][c] * caps[s] / max(caps["train"], 1))
            if s != "train" and 4 in present:
                score -= 5.0
            if len([x for x in assigned.values() if x == s]) >= caps[s] - 1:
                score += 1000
            if score < best_score:
                best_score = score
                best_split = s
        if best_split is None:
            for s in caps:
                if len([x for x in assigned.values() if x == s]) < caps[s]:
                    best_split = s
                    break
        assigned[iid] = best_split
        for c in present:
            counts[best_split][c] += 1

    return assigned


def random_split(image_ids, flags, rng):
    ids = list(image_ids)
    rng.shuffle(ids)
    n_train, n_val, n_test = split_sizes(len(ids))
    assigned = {}
    for iid in ids[:n_train]:
        assigned[iid] = "train"
    for iid in ids[n_train:n_train + n_val]:
        assigned[iid] = "val"
    for iid in ids[n_train + n_val:]:
        assigned[iid] = "test"
    return assigned


def split_instance_counts(assigned, anns_by_img):
    counts = {s: Counter() for s in ("train", "val", "test")}
    for iid, sp in assigned.items():
        for a in anns_by_img.get(iid, []):
            counts[sp][a["category_id"]] += 1
    return counts


def score_split(counts, assigned, n_images):
    """Lower is better."""
    total = Counter()
    for sp in counts:
        total.update(counts[sp])

    global_frac = {c: total[c] / max(sum(total.values()), 1) for c in CLASS_IDS}
    penalty = 0.0

    for sp in ("train", "val", "test"):
        sp_total = sum(counts[sp].values())
        if sp_total == 0:
            return 1e9
        for c in CLASS_IDS:
            frac = counts[sp][c] / sp_total
            penalty += abs(frac - global_frac[c]) * 100
            if counts[sp][c] == 0:
                penalty += 500.0

    if counts["val"][4] < 2:
        penalty += 200
    if counts["test"][4] < 2:
        penalty += 200
    if counts["train"][4] < 5:
        penalty += 100

    n_tr = sum(1 for v in assigned.values() if v == "train")
    n_va = sum(1 for v in assigned.values() if v == "val")
    n_te = sum(1 for v in assigned.values() if v == "test")
    penalty += abs(n_tr / n_images - 0.70) * 50
    penalty += abs(n_va / n_images - 0.15) * 50
    penalty += abs(n_te / n_images - 0.15) * 50
    return penalty


def evaluate_assignment(assignment, anns_by_img, n_images):
    counts = split_instance_counts(assignment, anns_by_img)
    return score_split(counts, assignment, n_images), counts


def write_splits(assigned, file_name_map, out_dir):
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    for iid, sp in assigned.items():
        buckets[sp].append(iid)
    for name, ids in buckets.items():
        with open(splits_dir / f"{name}_images.txt", "w", encoding="utf-8") as f:
            for iid in sorted(ids):
                f.write(f"{iid}\t{file_name_map[iid]}\n")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.data_root) / args.ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            anns_by_img[ann["image_id"]].append(ann)

    image_ids = [im["id"] for im in coco["images"]]
    file_name_map = {im["id"]: im["file_name"] for im in coco["images"]}
    flags = image_class_flags(anns_by_img, image_ids)
    n_images = len(image_ids)

    best_score = float("inf")
    best_assigned = None
    best_counts = None

    rng = random.Random(args.seed)
    for trial in range(args.n_candidates):
        seed_i = rng.randint(0, 2**31 - 1)
        tr = random.Random(seed_i)
        if trial == 0:
            cand = assign_stratified(image_ids, flags, tr)
        else:
            cand = random_split(image_ids, flags, tr)
        sc, cnt = evaluate_assignment(cand, anns_by_img, n_images)
        if sc < best_score:
            best_score = sc
            best_assigned = cand
            best_counts = cnt

    assigned = best_assigned
    write_splits(assigned, file_name_map, out_dir)

    rows = []
    for sp in ("train", "val", "test"):
        for cid in CLASS_IDS:
            rows.append({
                "split": sp,
                "class_name": CLASS_NAMES[cid],
                "instance_count": best_counts[sp][cid],
                "present": best_counts[sp][cid] > 0,
            })
    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(out_dir / "class_distribution_by_split.csv", index=False)

    old_df = load_split_counts(args.old_split_dir, anns_by_img)

    quality = out_dir / "split_quality_report.md"
    with open(quality, "w", encoding="utf-8") as f:
        f.write("# 4-class stratified split quality report\n\n")
        f.write("**This is the new best-practical split.** Results trained on "
                "`outputs/merged_4class_stratified_split/splits` are **not directly "
                "comparable** to older runs (`faster_rcnn_50ep`, `faster_rcnn_3defect`).\n\n")
        f.write(f"- Candidates evaluated: {args.n_candidates}\n")
        f.write(f"- Best score: {best_score:.2f}\n\n")
        f.write("## Instance counts\n\n")
        f.write("| Split | well_soldered | item_missing | mis_soldered | short_circuit |\n")
        f.write("|-------|---------------|--------------|--------------|---------------|\n")
        for sp in ("train", "val", "test"):
            f.write(f"| {sp} | {best_counts[sp][1]} | {best_counts[sp][2]} | "
                    f"{best_counts[sp][3]} | {best_counts[sp][4]} |\n")
        n_tr = sum(1 for v in assigned.values() if v == "train")
        n_va = sum(1 for v in assigned.values() if v == "val")
        n_te = sum(1 for v in assigned.values() if v == "test")
        f.write(f"\n## Image counts\n\n")
        f.write(f"- train: {n_tr} ({100*n_tr/n_images:.1f}%)\n")
        f.write(f"- val: {n_va} ({100*n_va/n_images:.1f}%)\n")
        f.write(f"- test: {n_te} ({100*n_te/n_images:.1f}%)\n")
        f.write("\n## All classes in all splits\n\n")
        for sp in ("train", "val", "test"):
            ok = all(best_counts[sp][c] > 0 for c in CLASS_IDS)
            f.write(f"- {sp}: {'OK' if ok else 'MISSING CLASS'}\n")
        f.write(f"\n## short_circuit\n\n")
        f.write(f"- train: {best_counts['train'][4]}\n")
        f.write(f"- val: {best_counts['val'][4]}\n")
        f.write(f"- test: {best_counts['test'][4]}\n")

    cmp_path = out_dir / "old_vs_new_split_comparison.md"
    with open(cmp_path, "w", encoding="utf-8") as f:
        f.write("# Old vs new 4-class split comparison\n\n")
        f.write("| | Old (50ep random) | New (stratified) |\n")
        f.write("|---|-------------------|------------------|\n")
        f.write("| Split dir | `outputs/faster_rcnn_50ep/splits` | "
                "`outputs/merged_4class_stratified_split/splits` |\n")
        f.write("| Comparable to prior runs | yes (same images) | **no** |\n\n")
        f.write("### Old split (instances)\n\n")
        f.write(old_df.to_string(index=False))
        f.write("\n\n### New split (instances)\n\n")
        f.write(dist_df.pivot(index="split", columns="class_name", values="instance_count")
                .to_string())

    print(f"Best score={best_score:.2f}")
    print(f"Saved splits -> {out_dir / 'splits'}")


if __name__ == "__main__":
    main()
