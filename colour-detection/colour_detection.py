"""
IPCV colour detection: HSV thresholding on webcam (or image) with live trackbars.

- Drag sliders to tune hue / saturation / value range.
- Left-click the camera view to sample a colour at that pixel (sets sliders around it).
- Press 'q' to quit. 'm' toggles mask-only vs overlay. 'r' resets sliders to a green-ish default.

On macOS, allow Camera access for Terminal / Cursor / your Python host app.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

WINDOW_MAIN = "Colour detection"
WINDOW_CTRL = "HSV controls"


def open_camera(preferred_index: int = 0):
    if sys.platform == "darwin":
        api = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    else:
        api = cv2.CAP_ANY

    indices = [preferred_index]
    for i in (0, 1, 2):
        if i not in indices:
            indices.append(i)

    for idx in indices:
        cap = cv2.VideoCapture(idx, api)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def default_hsv_bounds() -> Tuple[np.ndarray, np.ndarray]:
    """Approximate 'green' object under typical indoor light (tune per scene)."""
    lower = np.array([35, 60, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    return lower, upper


# Global refs for trackbar + mouse callback (OpenCV callback signature limitation)
_tbar_lower = np.array([35, 60, 40], dtype=np.int32)
_tbar_upper = np.array([90, 255, 255], dtype=np.int32)


def _sync_trackbars_from_arrays():
    for i, name in enumerate(["H_min", "S_min", "V_min"]):
        cv2.setTrackbarPos(name, WINDOW_CTRL, int(_tbar_lower[i]))
    for i, name in enumerate(["H_max", "S_max", "V_max"]):
        cv2.setTrackbarPos(name, WINDOW_CTRL, int(_tbar_upper[i]))


def _on_trackbar(_) -> None:
    global _tbar_lower, _tbar_upper
    _tbar_lower = np.array(
        [
            cv2.getTrackbarPos("H_min", WINDOW_CTRL),
            cv2.getTrackbarPos("S_min", WINDOW_CTRL),
            cv2.getTrackbarPos("V_min", WINDOW_CTRL),
        ],
        dtype=np.uint8,
    )
    _tbar_upper = np.array(
        [
            cv2.getTrackbarPos("H_max", WINDOW_CTRL),
            cv2.getTrackbarPos("S_max", WINDOW_CTRL),
            cv2.getTrackbarPos("V_max", WINDOW_CTRL),
        ],
        dtype=np.uint8,
    )


def _ensure_h_order():
    """Keep min <= max per channel; sync sliders if we auto-correct."""
    changed = False
    for i in range(3):
        if int(_tbar_lower[i]) > int(_tbar_upper[i]):
            _tbar_lower[i], _tbar_upper[i] = _tbar_upper[i], _tbar_lower[i]
            changed = True
    if changed:
        _sync_trackbars_from_arrays()


def mask_colour(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Single-range HSV mask. For hues that wrap (red), use two ranges + cv2.bitwise_or.
    """
    return cv2.inRange(hsv, lower, upper)


def refine_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def largest_contour_roi(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 200:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h


def sample_hsv_at(
    hsv_frame: np.ndarray, x: int, y: int, margin: Tuple[int, int, int] = (15, 60, 60)
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = hsv_frame.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    pixel = hsv_frame[y, x]
    hh, hs, hv = int(pixel[0]), int(pixel[1]), int(pixel[2])
    mh, ms, mv = margin
    lower = np.array(
        [
            max(0, hh - mh),
            max(0, hs - ms),
            max(0, hv - mv),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            min(179, hh + mh),
            min(255, hs + ms),
            min(255, hv + mv),
        ],
        dtype=np.uint8,
    )
    if lower[0] > upper[0]:
        lower[0], upper[0] = upper[0], lower[0]
    return lower, upper


def make_control_window():
    cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CTRL, 420, 120)
    lo, hi = default_hsv_bounds()
    global _tbar_lower, _tbar_upper
    _tbar_lower = lo.astype(np.int32)
    _tbar_upper = hi.astype(np.int32)

    def add_tbar(name: str, max_v: int, val: int):
        cv2.createTrackbar(name, WINDOW_CTRL, val, max_v, _on_trackbar)

    add_tbar("H_min", 179, int(lo[0]))
    add_tbar("H_max", 179, int(hi[0]))
    add_tbar("S_min", 255, int(lo[1]))
    add_tbar("S_max", 255, int(hi[1]))
    add_tbar("V_min", 255, int(lo[2]))
    add_tbar("V_max", 255, int(hi[2]))
    _on_trackbar(0)


def run_image(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}")
        return
    make_control_window()
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    show_mask_only = False

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lo, hi = sample_hsv_at(hsv, x, y)
        global _tbar_lower, _tbar_upper
        _tbar_lower = lo.astype(np.int32)
        _tbar_upper = hi.astype(np.int32)
        _sync_trackbars_from_arrays()

    cv2.setMouseCallback(WINDOW_MAIN, on_mouse)

    while True:
        _on_trackbar(0)
        _ensure_h_order()
        lower = _tbar_lower.astype(np.uint8)
        upper = _tbar_upper.astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        raw = mask_colour(hsv, lower, upper)
        m = refine_mask(raw)
        overlay = img.copy()
        roi = largest_contour_roi(m)
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(overlay, (cx, cy), 6, (0, 255, 255), -1)

        if show_mask_only:
            vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        else:
            vis = cv2.bitwise_and(overlay, overlay, mask=m)
            # Dim outside mask slightly for context
            inv = cv2.bitwise_not(m)
            bg = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
            vis = cv2.addWeighted(overlay, 0.95, bg, 0.15, 0)

        info = f"HSV lower={lower.tolist()} upper={upper.tolist()}"
        cv2.putText(
            vis,
            info,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_MAIN, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mask_only = not show_mask_only
        if key == ord("r"):
            lo, hi = default_hsv_bounds()
            _tbar_lower[:] = lo.astype(np.int32)
            _tbar_upper[:] = hi.astype(np.int32)
            _sync_trackbars_from_arrays()

    cv2.destroyAllWindows()


def run_camera(cam_index: int) -> None:
    cap = open_camera(cam_index)
    if cap is None:
        print("Could not open any camera.")
        if sys.platform == "darwin":
            print(
                "macOS: System Settings → Privacy & Security → Camera — enable "
                "the app running this script, then restart it."
            )
        return

    make_control_window()
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    show_mask_only = False
    frame_holder: dict[str, object] = {"hsv": None, "w": 0, "h": 0}

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        hsv = frame_holder.get("hsv")
        if hsv is None:
            return
        lo, hi = sample_hsv_at(hsv, x, y)
        global _tbar_lower, _tbar_upper
        _tbar_lower = lo.astype(np.int32)
        _tbar_upper = hi.astype(np.int32)
        _sync_trackbars_from_arrays()

    cv2.setMouseCallback(WINDOW_MAIN, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_holder["hsv"] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_holder["w"] = frame.shape[1]
            frame_holder["h"] = frame.shape[0]

            _on_trackbar(0)
            _ensure_h_order()
            lower = _tbar_lower.astype(np.uint8)
            upper = _tbar_upper.astype(np.uint8)
            hsv = frame_holder["hsv"]
            raw = mask_colour(hsv, lower, upper)
            m = refine_mask(raw)
            overlay = frame.copy()
            roi = largest_contour_roi(m)
            if roi is not None:
                x, y, w, h = roi
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cx, cy = x + w // 2, y + h // 2
                cv2.circle(overlay, (cx, cy), 6, (0, 255, 255), -1)

            if show_mask_only:
                vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            else:
                vis = cv2.bitwise_and(overlay, overlay, mask=m)
                inv = cv2.bitwise_not(m)
                bg = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
                vis = cv2.addWeighted(overlay, 0.95, bg, 0.15, 0)

            info = f"HSV lower={lower.tolist()} upper={upper.tolist()}  [m] mask  [r] reset  [q] quit"
            cv2.putText(
                vis,
                info,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                info,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_MAIN, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                show_mask_only = not show_mask_only
            if key == ord("r"):
                lo, hi = default_hsv_bounds()
                _tbar_lower[:] = lo.astype(np.int32)
                _tbar_upper[:] = hi.astype(np.int32)
                _sync_trackbars_from_arrays()
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="HSV colour detection (webcam or image)")
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to image file instead of webcam",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=0,
        help="Camera index (default 0)",
    )
    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    else:
        run_camera(args.camera)


if __name__ == "__main__":
    main()
