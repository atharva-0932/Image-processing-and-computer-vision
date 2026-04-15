"""
Hand landmarks using MediaPipe Tasks (HandLandmarker). The legacy `mp.solutions`
API is not shipped in current mediapipe wheels for Python 3.12+.
"""

from __future__ import annotations

import math
from pathlib import Path
import ssl
from typing import Optional
import urllib.request

import certifi
import cv2

from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode

BaseOptions = base_options_module.BaseOptions

# Official task model (float16 bundle)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_hand_model() -> Path:
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "hand_landmarker.task"
    if not target.is_file():
        print(f"Downloading hand landmarker model to {target} ...")
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(_MODEL_URL, context=ctx) as resp:
            target.write_bytes(resp.read())
    return target


class HandDetector:
    def __init__(
        self,
        mode=False,
        max_hands=2,
        detection_con=0.5,
        track_con=0.5,
        model_complexity=1,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self._model_complexity = model_complexity  # unused in Tasks API; kept for API compat

        model_path = _ensure_hand_model()
        options = hand_landmarker.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=detection_con,
            min_tracking_confidence=track_con,
        )
        self._landmarker = hand_landmarker.HandLandmarker.create_from_options(
            options
        )
        self._ts_ms = 0
        self._result: Optional[hand_landmarker.HandLandmarkerResult] = None
        self.tip_ids = [4, 8, 12, 16, 20]
        self.lm_list: list[list[int]] = []

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def find_hands(self, img, draw=True):
        self._ts_ms += 1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(
            image_format=mp_image.ImageFormat.SRGB, data=rgb
        )
        self._result = self._landmarker.detect_for_video(mp_img, self._ts_ms)

        if draw and self._result.hand_landmarks:
            spec = drawing_utils.DrawingSpec(
                color=(255, 0, 255), thickness=2, circle_radius=2
            )
            conn_spec = drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2)
            for hand_lm in self._result.hand_landmarks:
                drawing_utils.draw_landmarks(
                    img,
                    hand_lm,
                    hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=spec,
                    connection_drawing_spec=conn_spec,
                )
        return img

    def find_position(self, img, hand_no=0, draw=True):
        self.lm_list = []
        if (
            not self._result
            or not self._result.hand_landmarks
            or len(self._result.hand_landmarks) <= hand_no
        ):
            return self.lm_list, ()

        my_hand = self._result.hand_landmarks[hand_no]
        h, w, _ = img.shape
        x_list = []
        y_list = []
        for idx, lm in enumerate(my_hand):
            cx, cy = int(lm.x * w), int(lm.y * h)
            x_list.append(cx)
            y_list.append(cy)
            self.lm_list.append([idx, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)
        bbox = (xmin, ymin, xmax, ymax)
        if draw:
            cv2.rectangle(
                img,
                (bbox[0] - 20, bbox[1] - 20),
                (bbox[2] + 20, bbox[3] + 20),
                (0, 255, 0),
                2,
            )
        return self.lm_list, bbox

    def fingers_up(self):
        fingers = []
        if len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for i in range(1, 5):
            if self.lm_list[self.tip_ids[i]][2] < self.lm_list[self.tip_ids[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True):
        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
