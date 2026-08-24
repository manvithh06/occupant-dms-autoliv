"""
main.py — Autoliv Occupant DMS — Standalone background service entry point.

Run with::

    python src/main.py

Press Ctrl+C to stop cleanly.
"""

import logging
import signal
import sys
import threading
import time
from pathlib import Path

# ── Logging Setup ─────────────────────────────────────────────────────────────
# Must happen before importing local modules so every logger inherits the setup.
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/dms.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("DMS")

# ── Local imports ─────────────────────────────────────────────────────────────
from config import cfg
from pipeline import DMSPipeline

# ── OS Notification ───────────────────────────────────────────────────────────
def send_notification(title: str, message: str, level: str = "WARNING") -> None:
    """Send a desktop push notification (best-effort; silently skipped if unavailable)."""
    try:
        from plyer import notification
        notification.notify(
            title    = title,
            message  = message,
            app_name = "Autoliv DMS",
            timeout  = 5 if level == "WARNING" else 8,
        )
        log.info("[NOTIFY] %s: %s", title, message)
    except Exception as exc:
        log.debug("Notification unavailable: %s", exc)


# ── Audio Alert ───────────────────────────────────────────────────────────────
_audio_queue: list[str] = []
_audio_lock  = threading.Lock()


def _audio_worker() -> None:
    """Single-worker thread that drains the audio queue one beep at a time."""
    while True:
        level = None
        with _audio_lock:
            if _audio_queue:
                level = _audio_queue.pop(0)
        if level:
            _play_beep(level)
        else:
            time.sleep(0.05)


def _play_beep(level: str) -> None:
    """Play a synchronous system beep (Windows only; silently skipped elsewhere)."""
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


def queue_beep(level: str) -> None:
    """Enqueue an audio alert to be played by the single audio worker thread."""
    with _audio_lock:
        _audio_queue.append(level)


# ── Notification Cooldown ─────────────────────────────────────────────────────
_notif_cooldown: dict[str, float] = {}


def _should_notify(alert_type: str) -> bool:
    now  = time.time()
    last = _notif_cooldown.get(alert_type, 0.0)
    if now - last > cfg.notif_cooldown_sec:
        _notif_cooldown[alert_type] = now
        return True
    return False


# ── Notification Config ───────────────────────────────────────────────────────
_NOTIF_CONFIG: dict[str, dict[str, str]] = {
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


def handle_alerts(alerts: list[dict]) -> None:
    """Dispatch OS notifications and audio beeps for incoming alerts."""
    for alert in alerts:
        alert_type = alert.get("type")
        level      = alert.get("level", "WARNING")
        if not _should_notify(alert_type):
            continue
        config = _NOTIF_CONFIG.get(alert_type)
        if config:
            threading.Thread(
                target=send_notification,
                args=(config["title"], config["message"], level),
                daemon=True,
            ).start()
            queue_beep(level)


# ── Shutdown Handler ───────────────────────────────────────────────────────────
_pipeline: DMSPipeline | None = None


def _signal_handler(sig, frame) -> None:
    log.info("Shutdown signal received — stopping DMS…")
    if _pipeline:
        _pipeline.stop()


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Main ───────────────────────────────────────────────────────────────────────
def run_dms() -> None:
    global _pipeline

    log.info("=" * 52)
    log.info("  Occupant DMS — Autoliv")
    log.info("  Starting background service…")
    log.info("=" * 52)

    # Start the single audio worker thread.
    audio_thread = threading.Thread(target=_audio_worker, name="dms-audio", daemon=True)
    audio_thread.start()

    send_notification(
        "Autoliv DMS Started",
        "Occupant detection and driver monitoring is now active.",
        "INFO",
    )

    _pipeline = DMSPipeline()

    # Wrap the pipeline's get_state so alerts fire notifications.
    _original_process = _pipeline._process_frame

    def _process_with_notify(frame):
        state = _original_process(frame)
        if state.get("alerts"):
            handle_alerts(state["alerts"])
        return state

    _pipeline._process_frame = _process_with_notify  # type: ignore[method-assign]

    log.info("DMS running — press Ctrl+C to stop.")
    try:
        _pipeline.run()   # blocks until stopped
    finally:
        send_notification("Autoliv DMS Stopped", "Background service has been stopped.", "INFO")
        log.info("DMS exited.")


if __name__ == "__main__":
    run_dms()