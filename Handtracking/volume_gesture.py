"""
Hand gesture volume control: thumb–index pinch distance maps to system volume (macOS).

Press 'q' to quit. Press 'c' to reset smoothing. Edit PINCH_CLOSE / PINCH_OPEN if the
mapping feels too sensitive for your camera distance.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Optional

import cv2

from hand_detector import HandDetector

# MediaPipe landmark indices: thumb tip = 4, index tip = 8
THUMB_TIP = 4
INDEX_TIP = 8


def get_macos_volume() -> Optional[int]:
    try:
        r = subprocess.run(
            [
                "osascript",
                "-e",
                "output volume of (get volume settings)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(r.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def set_macos_volume(percent: int) -> bool:
    percent = max(0, min(100, int(percent)))
    try:
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {percent}"],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def map_distance_to_volume(
    dist: float, d_min: float, d_max: float
) -> int:
    if d_max <= d_min:
        return 50
    t = (dist - d_min) / (d_max - d_min)
    t = max(0.0, min(1.0, t))
    return int(round(t * 100))


def open_camera(preferred_index: int = 0):
    """
    Open a capture device. On macOS, prefer AVFoundation (CAP_AVFOUNDATION) and try
    a few indices — index 0 is often the built-in camera on laptops.
    """
    if sys.platform == "darwin":
        # Native macOS backend; avoids some FFmpeg/FFMPEG device-list issues.
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


def print_macos_camera_help():
    print(
        "\nCamera access was denied or no camera opened.\n"
        "On macOS: System Settings → Privacy & Security → Camera\n"
        "  Turn ON the app that runs this script (Terminal, iTerm, Cursor, "
        "VS Code, or Python).\n"
        "Then quit that app completely and reopen it, or log out/in, and run again.\n"
    )


def main():
    if sys.platform != "darwin":
        print(
            "This script uses macOS AppleScript for volume. "
            "On Linux try `amixer`/`pactl`; on Windows use `pycaw`."
        )

    cap = open_camera(0)
    if cap is None:
        print("Could not open any camera.")
        if sys.platform == "darwin":
            print_macos_camera_help()
        else:
            print("Try a different camera index or check drivers.")
        return

    detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.7)

    # Pixel distance thumb tip ↔ index tip (tune for your camera / arm length)
    pinch_close = 35.0
    pinch_open = 260.0
    smooth = 0.0
    smooth_alpha = 0.25

    p_time = 0.0
    print(
        "Thumb + index: spread = louder, pinch = quieter. "
        "'c' = reset smoothing. 'q' = quit."
    )

    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break

            img = detector.find_hands(img)
            lm_list, _bbox = detector.find_position(img, draw=True)

            h, w, _ = img.shape
            vol_text = "--"

            if len(lm_list) >= 21:
                length, img, _ = detector.find_distance(
                    THUMB_TIP, INDEX_TIP, img, draw=True
                )
                target_vol = map_distance_to_volume(
                    length, pinch_close, pinch_open
                )
                if smooth == 0.0:
                    smooth = float(target_vol)
                else:
                    smooth = (
                        smooth_alpha * target_vol
                        + (1 - smooth_alpha) * smooth
                    )
                vol_int = int(round(smooth))
                if set_macos_volume(vol_int):
                    vol_text = str(vol_int)
                else:
                    vol_text = str(vol_int) + " (set failed)"

                cv2.putText(
                    img,
                    f"Vol: {vol_text}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            c_time = time.time()
            fps = 1.0 / max(c_time - p_time, 1e-6)
            p_time = c_time
            cv2.putText(
                img,
                f"FPS: {int(fps)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 255),
                2,
            )

            cur = get_macos_volume()
            if cur is not None:
                cv2.putText(
                    img,
                    f"System: {cur}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

            cv2.imshow("Hand volume", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                smooth = 0.0
                print("Volume smoothing reset.")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
