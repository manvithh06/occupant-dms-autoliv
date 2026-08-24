"""
vitals/rppg.py — Remote Photoplethysmography (rPPG) vital-signs monitor.

Estimates heart rate (BPM), breathing rate, and stress level from subtle
colour changes in the driver's forehead region captured by the cabin camera.

Key fix over the original: FPS is no longer a hard-coded constant.
``RPPGMonitor`` accepts the actual measured camera FPS so that FFT-based
frequency calculations remain accurate regardless of the capture rate.
Updated for MediaPipe 1.0 (mp.tasks API).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt

from config import cfg
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions as mp_base

# ── MediaPipe 1.0 tasks API ───────────────────────────────────────────────────
_FaceLandmarker = mp_vision.FaceLandmarker
_FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
_RunningMode = mp_vision.RunningMode

_DEFAULT_FACE_MODEL = Path(__file__).resolve().parent.parent.parent / "face_landmarker.task"

# ── Forehead ROI landmarks (MediaPipe 478-landmark model) ─────────────────────
FOREHEAD_POINTS = [10, 338, 297, 332, 284]


# ── Butterworth bandpass filter ───────────────────────────────────────────────
def bandpass_filter(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """
    Apply a 4th-order Butterworth bandpass filter.

    Args:
        signal: 1-D signal array.
        low:    Lower cutoff frequency (Hz).
        high:   Upper cutoff frequency (Hz).
        fs:     Sampling frequency (Hz).

    Returns:
        Filtered signal array.
    """
    nyq      = fs / 2.0
    b, a     = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


# ── RPPGMonitor ───────────────────────────────────────────────────────────────
class RPPGMonitor:
    """
    Estimates driver vital signs from forehead skin-colour fluctuations.

    Args:
        fps: Actual camera frames per second.  Pass ``None`` to use the value
             from ``config.py`` (``cfg.rppg_fps``).  Should be updated once
             the camera reports its real FPS via
             ``cap.get(cv2.CAP_PROP_FPS)``.
    """

    def __init__(self, fps: float | None = None, face_model_path: str | None = None) -> None:
        self._fps = float(fps) if fps else float(cfg.rppg_fps)
        self._update_derived()

        model_path = face_model_path or str(_DEFAULT_FACE_MODEL)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"MediaPipe FaceLandmarker model not found at: {model_path}")

        options = _FaceLandmarkerOptions(
            base_options           = mp_base(model_asset_path=model_path),
            running_mode           = _RunningMode.VIDEO,
            num_faces              = 1,
            min_face_detection_confidence = 0.5,
            min_face_presence_confidence  = 0.5,
            min_tracking_confidence       = 0.5,
        )
        self._landmarker: _FaceLandmarker = _FaceLandmarker.create_from_options(options)
        self._timestamp_ms: int = 0

        self._green_buffer: deque[float] = deque(maxlen=self._window_size)
        self._frame_count:  int          = 0

        # Last computed values
        self.bpm            = 0.0
        self.breathing_rate = 0.0
        self.stress_level   = "unknown"

        print(f"[RPPGMonitor] Initialized — FPS={self._fps:.1f}, window={self._window_size} frames")

    def set_fps(self, fps: float) -> None:
        """
        Update the FPS used for signal analysis.

        Call this once the pipeline has measured the actual camera FPS,
        e.g.::

            rppg.set_fps(cap.get(cv2.CAP_PROP_FPS))
        """
        self._fps = float(fps)
        self._update_derived()
        # Resize buffer to new window; existing samples are preserved up to new limit.
        old_data = list(self._green_buffer)
        self._green_buffer = deque(old_data[-self._window_size:], maxlen=self._window_size)

    def _update_derived(self) -> None:
        """Recompute derived constants whenever FPS changes."""
        self._window_size = int(self._fps * cfg.rppg_window_seconds)
        self._low_freq    = cfg.low_bpm  / 60.0
        self._high_freq   = cfg.high_bpm / 60.0

    # ── ROI extraction ────────────────────────────────────────────────────────
    def _get_forehead_roi(self, frame: np.ndarray, landmarks) -> tuple[np.ndarray, tuple]:
        h, w = frame.shape[:2]
        pts  = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in FOREHEAD_POINTS]

        x1 = max(0, min(p[0] for p in pts) - 10)
        y1 = max(0, min(p[1] for p in pts) - 10)
        x2 = min(w, max(p[0] for p in pts) + 10)
        y2 = min(h, max(p[1] for p in pts) + 10)

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    # ── BPM ───────────────────────────────────────────────────────────────────
    def _calculate_bpm(self) -> float:
        half = self._window_size // 2
        if len(self._green_buffer) < half:
            return 0.0

        signal = np.array(self._green_buffer)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

        try:
            filtered = bandpass_filter(signal, self._low_freq, self._high_freq, self._fps)
        except Exception:
            return 0.0

        fft_vals = np.abs(np.fft.rfft(filtered))
        fft_freq = np.fft.rfftfreq(len(filtered), d=1.0 / self._fps)

        valid = (fft_freq >= self._low_freq) & (fft_freq <= self._high_freq)
        if not np.any(valid):
            return 0.0

        peak_freq = fft_freq[valid][np.argmax(fft_vals[valid])]
        return round(peak_freq * 60.0, 1)

    # ── Breathing rate ────────────────────────────────────────────────────────
    def _calculate_breathing_rate(self) -> float:
        half = self._window_size // 2
        if len(self._green_buffer) < half:
            return 0.0

        signal = np.array(self._green_buffer)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

        # Breathing: 0.1–0.5 Hz (6–30 breaths/min)
        try:
            filtered = bandpass_filter(signal, 0.1, 0.5, self._fps)
        except Exception:
            return 0.0

        fft_vals = np.abs(np.fft.rfft(filtered))
        fft_freq = np.fft.rfftfreq(len(filtered), d=1.0 / self._fps)

        valid = (fft_freq >= 0.1) & (fft_freq <= 0.5)
        if not np.any(valid):
            return 0.0

        peak_freq = fft_freq[valid][np.argmax(fft_vals[valid])]
        return round(peak_freq * 60.0, 1)

    # ── Stress ────────────────────────────────────────────────────────────────
    def _calculate_stress(self, bpm: float) -> str:
        if bpm == 0:
            return "unknown"
        if bpm < 60:
            return "low"
        if bpm < 80:
            return "normal"
        if bpm < 100:
            return "elevated"
        return "high"

    # ── Main analysis ─────────────────────────────────────────────────────────
    def analyze(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Analyse one frame for vital signs.

        Args:
            frame: BGR image.

        Returns:
            ``(annotated_frame, vitals_dict)``

            vitals_dict keys:
                ``bpm``, ``breathing_rate``, ``stress``, ``signal_quality``
        """
        h, w = frame.shape[:2]
        self._timestamp_ms += 33

        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        vitals: dict = {
            "bpm":            self.bpm,
            "breathing_rate": self.breathing_rate,
            "stress":         self.stress_level,
            "signal_quality": "weak",
        }

        if not result.face_landmarks:
            frame = self._draw(frame, vitals, roi_box=None)
            return frame, vitals

        landmarks = result.face_landmarks[0]
        roi, roi_box = self._get_forehead_roi(frame, landmarks)

        if roi.size == 0:
            frame = self._draw(frame, vitals, roi_box=None)
            return frame, vitals

        green_mean = float(np.mean(roi[:, :, 1]))   # green channel in BGR
        self._green_buffer.append(green_mean)
        self._frame_count += 1

        half = self._window_size // 2
        if self._frame_count % 30 == 0 and len(self._green_buffer) >= half:
            self.bpm            = self._calculate_bpm()
            self.breathing_rate = self._calculate_breathing_rate()
            self.stress_level   = self._calculate_stress(self.bpm)

        vitals = {
            "bpm":            self.bpm,
            "breathing_rate": self.breathing_rate,
            "stress":         self.stress_level,
            "signal_quality": "good" if len(self._green_buffer) >= half else "collecting",
        }

        frame = self._draw(frame, vitals, roi_box)
        return frame, vitals

    def _draw(self, frame: np.ndarray, vitals: dict, roi_box) -> np.ndarray:
        """Overlay vitals HUD on *frame*."""
        bpm     = vitals["bpm"]
        breath  = vitals["breathing_rate"]
        stress  = vitals["stress"]
        quality = vitals["signal_quality"]

        if roi_box:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(frame, "ROI", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        stress_color = {
            "low":      (0, 255, 0),
            "normal":   (0, 255, 0),
            "elevated": (0, 165, 255),
            "high":     (0, 0, 255),
            "unknown":  (128, 128, 128),
        }.get(stress, (255, 255, 255))

        y0 = frame.shape[0] - 120
        cv2.putText(frame, f"HR       : {bpm} BPM",    (10, y0),      cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Breathing: {breath} /min", (10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Stress   : {stress.upper()}", (10, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, stress_color, 2)
        cv2.putText(frame, f"Signal   : {quality}",     (10, y0 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

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
    cap = cv2.VideoCapture(cfg.camera_index)
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    with RPPGMonitor(fps=cam_fps if cam_fps > 1 else None) as monitor:
        print("[INFO] Collecting signal — stay still and face camera")
        print(f"[INFO] FPS = {monitor._fps:.1f} — BPM appears after ~5 s")
        print("[INFO] Press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, vitals = monitor.analyze(frame)
            cv2.imshow("rPPG Vital Signs", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()