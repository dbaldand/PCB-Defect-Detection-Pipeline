"""Bounding-box matching utilities shared by mounted-PCB evaluation scripts."""

import numpy as np


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two xyxy box arrays. Shapes [N,4] and [M,4] -> [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_thres: float,
) -> list[tuple[int, int]]:
    """Greedy IoU matching ordered by descending prediction score."""
    order = np.argsort(-pred_scores) if len(pred_scores) else np.array([], dtype=int)
    ious = box_iou_np(pred_boxes, gt_boxes)
    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    matches: list[tuple[int, int]] = []
    for p in order:
        if len(gt_boxes) == 0:
            break
        candidates = np.where(~gt_used)[0]
        if len(candidates) == 0:
            break
        ious_p = ious[p, candidates]
        best = int(np.argmax(ious_p))
        if ious_p[best] >= iou_thres:
            gt_idx = int(candidates[best])
            gt_used[gt_idx] = True
            matches.append((int(p), gt_idx))
    return matches
