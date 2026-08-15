import cv2
import time
import json
import logging
import signal
import sys
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Logging Setup ─────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/dms.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("DMS")

# ── Imports ───────────────────────────────────────────────
from occupant.occupant_detector import OccupantDetector
from driver.driver_monitor      import DriverMonitor
from vitals.rppg                import RPPGMonitor
from alerts.alert_engine        import AlertEngine

# ── Windows Notification ──────────────────────────────────
def send_notification(title, message, level="WARNING"):
    try:
        from plyer import notification
        notification.notify(
            title    = title,
            message  = message,
            app_name = "Autoliv DMS",
            timeout  = 5 if level == "WARNING" else 8
        )
        log.info(f"[NOTIFY] {title}: {message}")
    except Exception as e:
        log.warning(f"Notification failed: {e}")

        
# ── Audio Alert ───────────────────────────────────────────
def play_beep(level):
    try:
        import winsound
        if level == "CRITICAL":
            for _ in range(3):
                winsound.Beep(1000, 300)
                time.sleep(0.1)
        else:
            winsound.Beep(600, 400)
    except Exception:
        pass

# ── Lighting Normalization ────────────────────────────────
def normalize_frame(frame):
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    lab     = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ── State Save ────────────────────────────────────────────
def save_state(data):
    try:
        with open("logs/latest_state.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        log.error(f"State save error: {e}")

# ── Notification Cooldown ─────────────────────────────────
notif_cooldown = {}
NOTIF_COOLDOWN_SEC = 15

def should_notify(alert_type):
    now     = time.time()
    last    = notif_cooldown.get(alert_type, 0)
    if now - last > NOTIF_COOLDOWN_SEC:
        notif_cooldown[alert_type] = now
        return True
    return False

# ── Alert Handler ─────────────────────────────────────────
NOTIF_CONFIG = {
    "sleeping": {
        "title":   "CRITICAL — Driver Sleeping",
        "message": "Driver eyes closed for too long. Immediate attention required.",
    },
    "phone_use": {
        "title":   "CRITICAL — Phone Use Detected",
        "message": "Driver is using phone while driving.",
    },
    "infant_alone": {
        "title":   "CRITICAL — Infant Left Alone",
        "message": "Infant detected in vehicle without adult present.",
    },
    "drowsy": {
        "title":   "WARNING — Driver Drowsy",
        "message": "Driver showing signs of drowsiness. Consider a break.",
    },
    "distracted": {
        "title":   "WARNING — Driver Distracted",
        "message": "Driver looking away from road.",
    },
    "high_hr": {
        "title":   "WARNING — High Heart Rate",
        "message": "Driver heart rate elevated above 100 BPM.",
    },
    "low_hr": {
        "title":   "CRITICAL — Low Heart Rate",
        "message": "Driver heart rate critically low. Check on driver.",
    },
    "high_stress": {
        "title":   "WARNING — High Stress",
        "message": "Driver stress level detected as high.",
    },
    "abnormal_breathing": {
        "title":   "WARNING — Abnormal Breathing",
        "message": "Driver breathing rate outside normal range.",
    },
}

def handle_alerts(alerts):
    for alert in alerts:
        alert_type = alert.get("type")
        level      = alert.get("level", "WARNING")

        if should_notify(alert_type):
            config = NOTIF_CONFIG.get(alert_type)
            if config:
                # Send notification in background thread
                threading.Thread(
                    target=send_notification,
                    args=(config["title"], config["message"], level),
                    daemon=True
                ).start()

                # Play audio beep
                threading.Thread(
                    target=play_beep,
                    args=(level,),
                    daemon=True
                ).start()

# ── Shutdown Handler ──────────────────────────────────────
running = True

def signal_handler(sig, frame):
    global running
    log.info("Shutdown signal received. Stopping DMS...")
    running = False

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ── Main DMS Loop ─────────────────────────────────────────
def run_dms():
    global running

    log.info("=" * 50)
    log.info("  Occupant DMS — Autoliv")
    log.info("  Starting background service...")
    log.info("=" * 50)

    log.info("Initializing modules...")
    occupant_detector = OccupantDetector()
    driver_monitor    = DriverMonitor()
    rppg_monitor      = RPPGMonitor()
    alert_engine      = AlertEngine(log_path="logs/alert_log.json")

    log.info("Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        log.error("Cannot open camera. Exiting.")
        send_notification(
            "DMS Error",
            "Cannot open camera. System stopped.",
            "CRITICAL"
        )
        return

    log.info("DMS running. Press Ctrl+C to stop.")
    send_notification(
        "Autoliv DMS Started",
        "Occupant detection and driver monitoring is now active.",
        "INFO"
    )

    frame_count = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame read failed. Retrying...")
            time.sleep(0.1)
            continue

        try:
            # Normalize lighting
            frame = normalize_frame(frame)

            # Run all modules
            detections     = occupant_detector.detect(frame)
            seat_map       = occupant_detector.get_seat_map(detections)
            _, driver_info = driver_monitor.analyze(frame)
            _, vitals_info = rppg_monitor.analyze(frame)

            # Process alerts
            alerts = []
            alerts += alert_engine.process_driver_state(driver_info)
            alerts += alert_engine.process_vitals(vitals_info)
            alerts += alert_engine.process_occupants(seat_map)

            # Handle notifications
            if alerts:
                handle_alerts(alerts)

            # Save state every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                save_state({
                    "driver":    driver_info,
                    "occupant":  seat_map,
                    "vitals":    vitals_info,
                    "alerts":    alerts,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        except Exception as e:
            log.error(f"Frame processing error: {e}")
            continue

    cap.release()
    log.info("Camera released. DMS stopped.")
    send_notification("Autoliv DMS Stopped", "Background service has been stopped.", "INFO")


if __name__ == "__main__":
    run_dms()