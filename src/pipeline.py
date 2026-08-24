"""
pipeline.py — Core DMS processing pipeline.

``DMSPipeline`` owns the camera, all detection modules, and the alert engine.
Both ``main.py`` (standalone service) and ``backend/api.py`` (FastAPI server)
delegate to this class instead of duplicating the camera loop logic.

Usage (standalone)::

    pipeline = DMSPipeline()
    pipeline.run()          # blocks until Ctrl-C / signal

Usage (embedded in FastAPI)::

    pipeline = DMSPipeline()
    pipeline.start()        # starts background thread
    ...
    state = pipeline.get_state()
    ...
    pipeline.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from config import cfg
from utils.image import normalize_frame
from occupant.occupant_detector import OccupantDetector
from driver.driver_monitor import DriverMonitor
from vitals.rppg import RPPGMonitor
from alerts.alert_engine import AlertEngine

log = logging.getLogger("DMS.pipeline")

# ── Default empty state ────────────────────────────────────────────────────────
_EMPTY_STATE: dict[str, Any] = {
    "driver":    {"state": "unknown"},
    "occupant":  {"total_occupied": 0, "detections": []},
    "vitals":    {"bpm": 0, "breathing_rate": 0, "stress": "unknown", "signal_quality": "weak"},
    "alerts":    [],
    "timestamp": "",
}


class DMSPipeline:
    """
    Encapsulates the full Occupant DMS processing loop.

    Attributes:
        running:  True while the camera loop is active.
        state:    Most-recently computed system state dict (thread-safe read).
    """

    def __init__(self, camera_index: int | None = None) -> None:
        self._camera_index = camera_index if camera_index is not None else cfg.camera_index

        log.info("Initializing DMS modules…")
        self.occupant_detector = OccupantDetector(
            model_path=str(cfg.model_path),
            confidence=cfg.occupant_conf,
        )
        self.driver_monitor = DriverMonitor()
        self.rppg_monitor   = RPPGMonitor()
        self.alert_engine   = AlertEngine(log_path=str(cfg.alert_log_path))

        # Share the single YOLO model loaded by OccupantDetector with DriverMonitor
        # so phone-detection does NOT load a second copy of the weights.
        self.driver_monitor.set_yolo(self.occupant_detector.model)

        self._lock:    threading.Lock   = threading.Lock()
        self._state:   dict[str, Any]   = dict(_EMPTY_STATE)
        self._thread:  threading.Thread | None = None
        self.running:  bool = False

        # FPS tracking for adaptive throttling
        self._frame_count: int = 0
        self._fps_measured: float = float(cfg.target_fps)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Return the latest system state (thread-safe copy)."""
        with self._lock:
            return dict(self._state)

    def get_alert_engine(self) -> AlertEngine:
        """Expose alert engine so API endpoints can query history."""
        return self.alert_engine

    def start(self) -> None:
        """Start the processing loop in a background daemon thread."""
        if self.running:
            log.warning("Pipeline already running.")
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="dms-pipeline",
            daemon=True,
        )
        self._thread.start()
        log.info("Pipeline background thread started.")

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("Pipeline stopped.")

    def run(self) -> None:
        """
        Run the pipeline on the calling thread (blocking).
        Intended for ``main.py`` standalone mode.
        """
        self._loop()

    # ── Internal loop ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            log.error("Cannot open camera (index=%d). Stopping.", self._camera_index)
            self.running = False
            return

        # Use camera-reported FPS if available; fall back to config value.
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        if cam_fps and cam_fps > 1:
            self._fps_measured = cam_fps
            self.rppg_monitor.set_fps(cam_fps)     # ← accurate BPM calculation
            log.info("Camera FPS detected: %.1f", cam_fps)
        else:
            self._fps_measured = float(cfg.target_fps)

        frame_interval = 1.0 / cfg.target_fps
        log.info("Camera opened. Processing at target %.0f fps.", cfg.target_fps)

        try:
            while self.running:
                t0 = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame read failed — retrying…")
                    time.sleep(0.05)
                    continue

                new_state = self._process_frame(frame)

                with self._lock:
                    self._state = new_state

                # ── Periodic state persistence ──────────────────────────────
                self._frame_count += 1
                if self._frame_count % cfg.state_save_interval == 0:
                    self._save_state(new_state)

                # ── FPS throttle ─────────────────────────────────────────────
                elapsed = time.monotonic() - t0
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except Exception as exc:
            log.exception("Unexpected error in pipeline loop: %s", exc)
        finally:
            cap.release()
            log.info("Camera released.")

    def _process_frame(self, frame: "cv2.Mat") -> dict[str, Any]:
        """Run all detection modules on a single frame and return state dict."""
        try:
            frame = normalize_frame(frame)

            detections     = self.occupant_detector.detect(frame)
            seat_map       = self.occupant_detector.get_seat_map(detections)
            _, driver_info = self.driver_monitor.analyze(frame)
            _, vitals_info = self.rppg_monitor.analyze(frame)

            alerts: list[dict] = []
            alerts += self.alert_engine.process_driver_state(driver_info)
            alerts += self.alert_engine.process_vitals(vitals_info)
            alerts += self.alert_engine.process_occupants(seat_map)

            return {
                "driver":    driver_info,
                "occupant":  seat_map,
                "vitals":    vitals_info,
                "alerts":    alerts,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as exc:
            log.error("Frame processing error: %s", exc)
            # Return last known state with updated timestamp rather than crashing.
            with self._lock:
                stale = dict(self._state)
            stale["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return stale

    def _save_state(self, state: dict[str, Any]) -> None:
        """Persist latest state to disk as JSON (non-blocking best-effort)."""
        try:
            path = cfg.state_log_path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as exc:
            log.error("State save error: %s", exc)
