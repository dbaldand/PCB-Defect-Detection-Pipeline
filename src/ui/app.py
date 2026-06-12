"""Real-time PCB inspection demo — USB camera + Ultralytics YOLO."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "inspection"
CONFIG_PATH = Path(os.environ.get(
    "INSPECTION_CONFIG",
    REPO_ROOT / "src" / "configs" / "inspection_config.yaml",
))

WELL_SOLDERED = "well_soldered"


def load_config() -> dict:
    """Load inspection UI settings from YAML."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    """Resolve model paths relative to the repository root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


@st.cache_resource(show_spinner="Loading model…")
def load_model(path_str: str) -> YOLO:
    return YOLO(path_str)


def _open_usb_capture(cfg: dict) -> cv2.VideoCapture:
    cam = cfg["camera"]
    index = int(cam.get("device_index", 0))
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    width = cam.get("width")
    height = cam.get("height")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return cap


def _read_camera_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    for _ in range(2):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok or frame is None or frame.size == 0:
        ok, frame = cap.read()
    if ok and frame is not None and frame.size > 0:
        return frame
    return None


def get_frame_from_camera(cfg: dict) -> tuple[np.ndarray | None, str | None]:
    """Return (BGR frame, error_message) from USB webcam."""
    cam = cfg["camera"]
    index = int(cam.get("device_index", 0))
    cap = st.session_state.get("video_cap")
    opened_index = st.session_state.get("camera_index")

    if cap is None or opened_index != index or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = _open_usb_capture(cfg)
        st.session_state.video_cap = cap
        st.session_state.camera_index = index

    if not cap.isOpened():
        return None, (
            f"Cannot open USB camera (device index {index}).\n"
            "Check the cable, close other apps using the camera, "
            "or try device_index: 1 in the inspection config."
        )

    frame = _read_camera_frame(cap)
    if frame is not None:
        return frame, None

    cap.release()
    st.session_state.video_cap = None
    return None, f"USB camera {index} opened but returned no frame. Click Reconnect camera."


def release_camera():
    cap = st.session_state.get("video_cap")
    if cap is not None:
        cap.release()
    st.session_state.video_cap = None
    st.session_state.camera_index = None


def run_detection(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    iou: float,
    mode: str,
    show_well_soldered: bool,
) -> list[dict]:
    results = model.predict(frame, conf=conf, iou=iou, verbose=False)
    if not results:
        return []
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    names = r.names
    detections = []
    for box in r.boxes:
        cls_id = int(box.cls.item())
        class_name = names[cls_id] if isinstance(names, dict) else names[cls_id]
        if mode == "Mounted PCB" and not show_well_soldered and class_name == WELL_SOLDERED:
            continue
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        detections.append(
            {
                "class_name": class_name,
                "confidence": float(box.conf.item()),
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
            }
        )
    return detections


def _color_for_class(class_name: str, cfg: dict) -> tuple[int, int, int]:
    colors = cfg.get("class_colors", {})
    bgr = colors.get(class_name, colors.get("default", [240, 240, 240]))
    return tuple(int(c) for c in bgr)


def draw_detections(frame: np.ndarray, detections: list[dict], cfg: dict) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = _color_for_class(det["class_name"], cfg)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 8, th + 8)
        cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 8, ty + baseline), color, -1)
        cv2.putText(out, label, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    return out


def save_snapshot(
    raw_frame: np.ndarray,
    annotated_frame: np.ndarray,
    detections: list[dict],
    pcb_mode: str,
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = RUNS_DIR / ts
    snap_dir.mkdir(parents=True, exist_ok=True)

    raw_name = f"{ts}_raw.jpg"
    ann_name = f"{ts}_annotated.jpg"
    cv2.imwrite(str(snap_dir / raw_name), raw_frame)
    cv2.imwrite(str(snap_dir / ann_name), annotated_frame)

    csv_path = RUNS_DIR / "inspection_log.csv"
    rows = []
    stamp = datetime.now().isoformat(timespec="seconds")
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        rows.append(
            {
                "timestamp": stamp,
                "pcb_mode": pcb_mode,
                "image_name": ann_name,
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    if not rows:
        rows.append(
            {
                "timestamp": stamp,
                "pcb_mode": pcb_mode,
                "image_name": ann_name,
                "class_name": "",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            }
        )
    df = pd.DataFrame(rows)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)
    return snap_dir


def inject_styles():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 100%; }
        [data-testid="stSidebar"] { min-width: 18rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session(cfg: dict):
    defaults = cfg.get("inference", {})
    st.session_state.setdefault("detecting", False)
    st.session_state.setdefault("pcb_mode", "Bare PCB")
    st.session_state.setdefault("conf", defaults.get("default_conf", 0.40))
    st.session_state.setdefault("iou", defaults.get("default_iou", 0.50))
    st.session_state.setdefault("show_well_soldered", False)
    st.session_state.setdefault("live_preview", True)
    st.session_state.setdefault("cached_detections", [])


def sidebar_controls(cfg: dict):
    st.sidebar.header("Inspection")
    mode = st.sidebar.selectbox("PCB mode", ["Bare PCB", "Mounted PCB"], key="pcb_mode_select")
    st.session_state.pcb_mode = mode

    if mode == "Mounted PCB":
        st.session_state.show_well_soldered = st.sidebar.checkbox(
            "Show well_soldered / normal components",
            value=st.session_state.show_well_soldered,
        )

    st.session_state.conf = st.sidebar.slider(
        "Confidence", 0.05, 0.95, float(st.session_state.conf), 0.05
    )
    st.session_state.iou = st.sidebar.slider(
        "IoU threshold", 0.10, 0.90, float(st.session_state.iou), 0.05
    )

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Start", use_container_width=True, type="primary"):
            st.session_state.detecting = True
    with c2:
        if st.button("Stop", use_container_width=True):
            st.session_state.detecting = False

    if st.sidebar.button("Snapshot", use_container_width=True):
        st.session_state.do_snapshot = True

    st.session_state.live_preview = st.sidebar.checkbox(
        "Live camera refresh", value=st.session_state.get("live_preview", True)
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("USB camera")
    idx = int(cfg.get("camera", {}).get("device_index", 0))
    st.sidebar.write(f"Device index: **{idx}**")
    if st.sidebar.button("Reconnect camera"):
        release_camera()
        st.rerun()


def load_model_safe(mode: str, cfg: dict) -> YOLO | None:
    path_key = "bare_model_path" if mode == "Bare PCB" else "mounted_model_path"
    model_path = resolve_path(cfg[path_key])
    if not model_path.is_file():
        st.session_state.model_error = f"Model file missing:\n`{model_path}`"
        return None
    try:
        st.session_state.model_error = None
        return load_model(str(model_path))
    except Exception as exc:
        st.session_state.model_error = str(exc)
        return None


def main():
    st.set_page_config(page_title="PCB Inspection", layout="wide")
    inject_styles()
    cfg = load_config()
    init_session(cfg)

    sidebar_controls(cfg)
    mode = st.session_state.pcb_mode

    model = load_model_safe(mode, cfg)
    if st.session_state.get("model_error"):
        st.error(st.session_state.model_error)

    frame, cam_err = get_frame_from_camera(cfg)
    if cam_err:
        st.error(cam_err)
        return

    detections: list[dict] = []
    annotated = frame

    stride = int(cfg.get("camera", {}).get("inference_stride", 1))
    frame_idx = st.session_state.get("frame_idx", 0) + 1
    st.session_state.frame_idx = frame_idx
    run_infer = st.session_state.detecting and model is not None and (
        frame_idx % max(stride, 1) == 0
    )

    if run_infer:
        detections = run_detection(
            model,
            frame,
            st.session_state.conf,
            st.session_state.iou,
            mode,
            st.session_state.show_well_soldered,
        )
        st.session_state.cached_detections = detections
        annotated = draw_detections(frame, detections, cfg)
    elif st.session_state.detecting:
        detections = st.session_state.get("cached_detections", [])
        annotated = draw_detections(frame, detections, cfg)

    if st.session_state.pop("do_snapshot", False):
        snap_det = detections
        snap_ann = annotated
        if model is not None and not snap_det:
            snap_det = run_detection(
                model, frame, st.session_state.conf, st.session_state.iou,
                mode, st.session_state.show_well_soldered,
            )
            snap_ann = draw_detections(frame, snap_det, cfg)
        out_dir = save_snapshot(frame, snap_ann, snap_det, mode)
        st.sidebar.success(f"Saved to `{out_dir.name}/`")

    display = annotated if st.session_state.detecting else frame
    st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    preview_fps = float(cfg.get("camera", {}).get("preview_fps", 15))
    if st.session_state.get("live_preview", True) or st.session_state.detecting:
        time.sleep(1.0 / max(preview_fps, 1))
        st.rerun()


if __name__ == "__main__":
    main()
