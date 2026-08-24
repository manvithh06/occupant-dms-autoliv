"""
driver/driver_monitor.py — Real-time driver state analysis.

Detects:
* Drowsiness  — Eye Aspect Ratio (EAR) + PERCLOS
* Head pose   — Yaw / pitch distraction
* Phone use   — YOLOv8 COCO cell-phone class

Updated for MediaPipe 1.0 (mp.tasks API).
All thresholds are read from ``config.py`` / ``config.yaml``.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from ultralytics import YOLO

from config import cfg

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions as mp_base

_RunningMode    = mp_vision.RunningMode
_FaceLandmarker = mp_vision.FaceLandmarker
_FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions

# Default bundled face-landmarker model (downloaded alongside the package).
_DEFAULT_FACE_MODEL = Path(__file__).resolve().parent.parent.parent / "face_landmarker.task"

# Eye landmark indices (MediaPipe 478-landmark model, refine=True)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Key face landmarks used for head-pose PnP solve
_HEAD_POSE_INDICES = [1, 9, 57, 130, 287, 359]


# ── EAR ──────────────────────────────────────────────────────────────────────
def calculate_ear(
    landmarks: list,
    eye_indices: list[int],
    image_w: int,
    image_h: int,
) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (‖p1−p5‖ + ‖p2−p4‖) / (2 · ‖p0−p3‖)

    Args:
        landmarks:   List of NormalizedLandmark (x, y, z in [0,1]).
        eye_indices: Six landmark indices defining the eye contour.
        image_w:     Frame width in pixels.
        image_h:     Frame height in pixels.

    Returns:
        EAR value as a float.
    """
    pts = [
        (landmarks[idx].x * image_w, landmarks[idx].y * image_h)
        for idx in eye_indices
    ]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)


# ── Head pose ─────────────────────────────────────────────────────────────────
def get_head_pose(
    landmarks: list,
    image_w: int,
    image_h: int,
) -> tuple[float, float, float]:
    """
    Estimate head pitch, yaw, and roll (degrees) via solvePnP.

    Args:
        landmarks: List of NormalizedLandmark.
        image_w:   Frame width in pixels.
        image_h:   Frame height in pixels.

    Returns:
        Tuple of (pitch, yaw, roll) in degrees.
    """
    face_2d, face_3d = [], []
    for idx in _HEAD_POSE_INDICES:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    cam_matrix = np.array([
        [image_w, 0,        image_w / 2],
        [0,        image_w, image_h / 2],
        [0,        0,        1],
    ], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _       = cv2.Rodrigues(rot_vec)
    angles, *_    = cv2.RQDecomp3x3(rmat)

    return angles[0] * 360, angles[1] * 360, angles[2] * 360


# ── DriverMonitor ─────────────────────────────────────────────────────────────
class DriverMonitor:
    """
    Analyses a BGR camera frame and returns the driver's current state.

    State values: ``"alert"``, ``"drowsy"``, ``"sleeping"``,
    ``"distracted"``, ``"phone_use"``, ``"no_face"``.

    Compatible with MediaPipe ≥ 1.0 (mp.tasks API).
    """

    def __init__(self, face_model_path: str | None = None) -> None:
        model_path = face_model_path or str(_DEFAULT_FACE_MODEL)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe FaceLandmarker model not found at: {model_path}\n"
                "Download it with:\n"
                "  python -c \"import urllib.request; urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')\""
            )

        options = _FaceLandmarkerOptions(
            base_options           = mp_base(model_asset_path=model_path),
            running_mode           = _RunningMode.VIDEO,
            num_faces              = 1,
            min_face_detection_confidence = 0.5,
            min_face_presence_confidence  = 0.5,
            min_tracking_confidence       = 0.5,
        )
        self._landmarker: _FaceLandmarker = _FaceLandmarker.create_from_options(options)
        self._timestamp_ms: int = 0   # monotonically increasing for VIDEO mode

        self._yolo: YOLO | None = None

        self._perclos_buffer: deque[int] = deque(maxlen=cfg.perclos_window)
        self._ear_counter:    int        = 0
        self._driver_state:   str        = "alert"
        self._frame_count:    int        = 0
        self._phone_detected: bool       = False
        self._last_phone_box: list[int]  = []

        print(f"[DriverMonitor] Initialized — model: {model_path}")

    def _get_yolo(self) -> YOLO:
        if self._yolo is None:
            self._yolo = YOLO(str(cfg.model_path))
        return self._yolo

    def set_yolo(self, yolo: YOLO) -> None:
        """Inject a shared YOLO instance to avoid duplicate weight loads."""
        self._yolo = yolo

    def analyze(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Run face landmarking + drowsiness + head-pose + phone detection.

        Args:
            frame: BGR image.

        Returns:
            ``(annotated_frame, state_info_dict)``
        """
        h, w = frame.shape[:2]

        # MediaPipe 1.0 VIDEO mode requires a monotonically-increasing timestamp
        self._timestamp_ms += 33   # approximate 30 fps

        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        state_info: dict = {
            "state":   "no_face",
            "ear":     None,
            "perclos": None,
            "yaw":     None,
            "pitch":   None,
            "alert":   None,
        }

        if not result.face_landmarks:
            return frame, state_info

        landmarks = result.face_landmarks[0]   # list of NormalizedLandmark

        # ── EAR ──────────────────────────────────────────────────────────────
        left_ear  = calculate_ear(landmarks, LEFT_EYE,  w, h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
        ear       = (left_ear + right_ear) / 2.0

        # ── PERCLOS ───────────────────────────────────────────────────────────
        self._perclos_buffer.append(1 if ear < cfg.ear_threshold else 0)
        perclos = sum(self._perclos_buffer) / len(self._perclos_buffer)

        # ── Consecutive closed-eye counter ────────────────────────────────────
        if ear < cfg.ear_threshold:
            self._ear_counter += 1
        else:
            self._ear_counter = 0

        # ── Head pose ─────────────────────────────────────────────────────────
        pitch, yaw, roll = get_head_pose(landmarks, w, h)

        # ── Determine state ───────────────────────────────────────────────────
        alert_msg: str | None = None

        if self._ear_counter >= cfg.ear_consec_frames:
            self._driver_state = "sleeping"
            alert_msg = "ALERT: Driver Sleeping!"
        elif perclos >= cfg.perclos_threshold:
            self._driver_state = "drowsy"
            alert_msg = "WARNING: Driver Drowsy!"
        elif abs(yaw) > cfg.yaw_threshold or abs(pitch) > cfg.pitch_threshold:
            self._driver_state = "distracted"
            alert_msg = "WARNING: Driver Distracted!"
        else:
            self._driver_state = "alert"

        state_info = {
            "state":   self._driver_state,
            "ear":     round(ear, 3),
            "perclos": round(perclos, 3),
            "yaw":     round(yaw, 2),
            "pitch":   round(pitch, 2),
            "alert":   alert_msg,
        }

        self._frame_count += 1
        if self._frame_count % cfg.yolo_frame_skip == 0:
            self._phone_detected = False
            yolo_results = self._get_yolo()(frame, conf=cfg.phone_conf, verbose=False, imgsz=cfg.yolo_imgsz)
            for res in yolo_results:
                for box in res.boxes:
                    if int(box.cls[0]) == 67:   # COCO class 67 = cell phone
                        self._phone_detected = True
                        self._last_phone_box = [int(v) for v in box.xyxy[0].tolist()]

        if self._phone_detected:
            x1, y1, x2, y2 = self._last_phone_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, "PHONE DETECTED",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            self._driver_state  = "phone_use"
            state_info["state"] = "phone_use"
            state_info["alert"] = "ALERT: Phone Use Detected!"

        frame = self._draw(frame, state_info)
        return frame, state_info

    def _draw(self, frame: np.ndarray, info: dict) -> np.ndarray:
        state   = info["state"]
        ear     = info["ear"]
        perclos = info["perclos"]
        yaw     = info["yaw"]
        alert   = info["alert"]

        color_map = {
            "alert":      (0, 255, 0),
            "drowsy":     (0, 165, 255),
            "sleeping":   (0, 0, 255),
            "distracted": (0, 255, 255),
            "phone_use":  (0, 0, 255),
            "no_face":    (128, 128, 128),
        }
        color = color_map.get(state, (255, 255, 255))

        cv2.putText(frame, f"State   : {state.upper()}", (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"EAR     : {ear}",            (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"PERCLOS : {perclos}",         (10, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Yaw     : {yaw} deg",         (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if alert:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, alert, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    with DriverMonitor() as monitor:
        cap = cv2.VideoCapture(cfg.camera_index)
        if not cap.isOpened():
            raise SystemExit("[ERROR] Cannot open webcam")

        print("[INFO] Webcam opened — press Q to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, info = monitor.analyze(frame)
            print(f"\r{info}", end="")
            cv2.imshow("Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Done")