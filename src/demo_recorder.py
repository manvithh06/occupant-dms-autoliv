"""
demo_recorder.py — Autoliv DMS live demo with video recording.

Run with::

    python src/demo_recorder.py

Press Q to stop.  A timestamped MP4 is saved to ``demo/``.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2

# ── Local imports ─────────────────────────────────────────────────────────────
from config import cfg
from utils.image import normalize_frame          # ← shared utility, no longer duplicated
from occupant.occupant_detector import OccupantDetector
from driver.driver_monitor import DriverMonitor
from vitals.rppg import RPPGMonitor
from alerts.alert_engine import AlertEngine

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_NAME = "Autoliv DMS — Live Demo"
FPS         = 20
RECORD      = True   # set False to view without saving video

# ── Colours ───────────────────────────────────────────────────────────────────
C_GREEN  = (0, 255, 140)
C_ORANGE = (0, 165, 255)
C_RED    = (0, 0, 255)
C_BLUE   = (255, 200, 0)
C_WHITE  = (255, 255, 255)
C_GRAY   = (150, 150, 150)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "alert":      C_GREEN,
    "drowsy":     C_ORANGE,
    "sleeping":   C_RED,
    "distracted": (0, 220, 255),
    "phone_use":  C_RED,
    "no_face":    C_GRAY,
    "unknown":    C_GRAY,
}


def draw_hud(
    frame,
    driver_info: dict,
    vitals_info: dict,
    seat_map: dict,
    alerts: list[dict],
    elapsed: float,
):
    """Overlay a three-column HUD + optional alert banner on *frame*."""
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ──────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        "AUTOLIV — Occupant Detection & Driver Monitoring System",
        (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLUE, 1,
    )
    cv2.putText(
        frame,
        f"NMAM Institute of Technology  |  {datetime.now().strftime('%H:%M:%S')}  |  REC {int(elapsed)}s",
        (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1,
    )

    # Recording indicator
    cv2.circle(frame, (w - 24, 20), 7, C_RED, -1)
    cv2.putText(frame, "REC", (w - 50, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_RED, 1)

    # ── Semi-transparent bottom panel ─────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 140), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay2, 0.78, frame, 0.22, 0, frame)

    # Left column — driver state ───────────────────────────────────────────────
    state       = driver_info.get("state", "unknown")
    state_color = STATE_COLORS.get(state, C_GRAY)
    ear         = driver_info.get("ear") or 0
    perclos     = driver_info.get("perclos") or 0
    yaw         = driver_info.get("yaw") or 0

    cv2.putText(frame, "DRIVER STATE",     (12, h - 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
    cv2.putText(frame, state.upper(),      (12, h - 96),  cv2.FONT_HERSHEY_SIMPLEX, 0.90, state_color, 2)
    cv2.putText(frame, f"EAR     : {ear}", (12, h - 72),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)
    cv2.putText(frame, f"PERCLOS : {perclos}", (12, h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)
    cv2.putText(frame, f"Yaw     : {yaw} deg", (12, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)

    # Middle column — vitals ──────────────────────────────────────────────────
    bpm     = vitals_info.get("bpm") or 0
    breath  = vitals_info.get("breathing_rate") or 0
    stress  = vitals_info.get("stress", "unknown")
    quality = vitals_info.get("signal_quality", "collecting")

    stress_color = (
        C_GREEN if stress in ("normal", "low")
        else C_ORANGE if stress == "elevated"
        else C_RED
    )
    mid = w // 3
    cv2.putText(frame, "VITAL SIGNS", (mid, h - 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
    cv2.putText(
        frame,
        f"{bpm} BPM" if bpm > 0 else "Collecting…",
        (mid, h - 96), cv2.FONT_HERSHEY_SIMPLEX, 0.90,
        C_BLUE if bpm > 0 else C_GRAY, 2,
    )
    cv2.putText(frame, f"Breathing : {breath} /min",    (mid, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)
    cv2.putText(frame, f"Stress    : {stress.upper()}", (mid, h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, stress_color, 1)
    cv2.putText(frame, f"Signal    : {quality}",        (mid, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)

    # Right column — occupants ────────────────────────────────────────────────
    occupied   = seat_map.get("total_occupied", 0)
    detections = seat_map.get("detections", [])
    right = (2 * w) // 3

    cv2.putText(frame, "OCCUPANTS",           (right, h - 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
    cv2.putText(frame, f"{occupied} IN VEHICLE", (right, h - 96),  cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_ORANGE, 2)
    for i, det in enumerate(detections[:3]):
        cv2.putText(
            frame,
            f"Seat {i+1}: {det['label'].upper()} ({int(det['confidence'] * 100)}%)",
            (right, h - 72 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_WHITE, 1,
        )

    # Alert banner ─────────────────────────────────────────────────────────────
    if alerts:
        latest = alerts[0]
        level  = latest.get("level", "WARNING")
        msg    = latest.get("message", "")
        color  = (0, 0, 200) if level == "CRITICAL" else (0, 130, 200)
        cv2.rectangle(frame, (0, 52), (w, 90), color, -1)
        cv2.putText(frame, f"  [{level}]  {msg}", (8, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_WHITE, 2)

    # PERCLOS bar ──────────────────────────────────────────────────────────────
    bar_w  = int((w // 4) * min(float(perclos), 1.0))
    bar_x  = w - (w // 4) - 12
    bar_col = C_RED if perclos > 0.8 else C_ORANGE if perclos > 0.5 else C_GREEN
    cv2.putText(frame, "PERCLOS", (bar_x, h - 118), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)
    cv2.rectangle(frame, (bar_x, h - 108), (bar_x + w // 4, h - 96), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (bar_x, h - 108), (bar_x + bar_w, h - 96), bar_col, -1)
    cv2.putText(frame, f"{int(perclos * 100)}%",
                (bar_x + w // 4 + 4, h - 96), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bar_col, 1)

    return frame


def run_demo() -> None:
    cfg.demo_output_dir.mkdir(parents=True, exist_ok=True)

    print("[Demo] Initializing modules…")
    occupant_detector = OccupantDetector(model_path=str(cfg.model_path))
    driver_monitor    = DriverMonitor()
    rppg_monitor      = RPPGMonitor()
    alert_engine      = AlertEngine()

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print("[Error] Cannot open camera")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if RECORD:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(cfg.demo_output_dir / f"autoliv_demo_{ts}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, FPS, (fw, fh))
        print(f"[Demo] Recording to {out_path}")

    print("[Demo] Running — press Q to stop")
    start  = time.time()
    alerts: list[dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = normalize_frame(frame)   # ← imported from utils.image

        detections         = occupant_detector.detect(frame)
        seat_map           = occupant_detector.get_seat_map(detections)
        frame, driver_info = driver_monitor.analyze(frame)
        frame, vitals_info = rppg_monitor.analyze(frame)
        frame              = occupant_detector.draw(frame, detections)

        new_alerts  = []
        new_alerts += alert_engine.process_driver_state(driver_info)
        new_alerts += alert_engine.process_vitals(vitals_info)
        new_alerts += alert_engine.process_occupants(seat_map)
        if new_alerts:
            alerts = new_alerts

        elapsed = time.time() - start
        frame   = draw_hud(frame, driver_info, vitals_info, seat_map, alerts, elapsed)

        if writer:
            writer.write(frame)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"[Demo] Saved to {out_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()