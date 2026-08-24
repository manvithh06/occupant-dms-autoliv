"""
alerts/alert_engine.py — Rule-based alert generation and logging.

Key improvements over the original:
* Uses ``config.py`` for cooldown timing (no hardcoded magic numbers).
* Alert log is written in **JSON-Lines** (one JSON object per line, append
  mode) instead of overwriting the entire history on every alert.  This
  is O(1) per write and is safe against mid-write corruption.
* ``get_active_alerts()`` respects the configurable active-alert window.
"""

from __future__ import annotations

import time
from datetime import datetime
from enum import Enum
from pathlib import Path

from config import cfg


# ── Alert levels ──────────────────────────────────────────────────────────────
class AlertLevel(Enum):
    NONE     = 0
    INFO     = 1   # log only
    WARNING  = 2   # visual warning on dashboard
    CRITICAL = 3   # audio + visual + log


# ── Alert rule registry ───────────────────────────────────────────────────────
ALERT_RULES: dict[str, dict] = {
    # Driver states
    "drowsy": {
        "level":   AlertLevel.WARNING,
        "message": "Driver appears drowsy",
        "action":  "visual_warning",
    },
    "sleeping": {
        "level":   AlertLevel.CRITICAL,
        "message": "Driver is sleeping!",
        "action":  "audio_visual_emergency",
    },
    "distracted": {
        "level":   AlertLevel.WARNING,
        "message": "Driver is distracted",
        "action":  "visual_warning",
    },
    "phone_use": {
        "level":   AlertLevel.CRITICAL,
        "message": "Driver using phone!",
        "action":  "audio_visual_emergency",
    },
    # Occupant states
    "infant_alone": {
        "level":   AlertLevel.CRITICAL,
        "message": "Infant left alone in vehicle!",
        "action":  "audio_visual_emergency",
    },
    "seatbelt_off": {
        "level":   AlertLevel.WARNING,
        "message": "Occupant seatbelt not fastened",
        "action":  "visual_warning",
    },
    # Vital signs
    "high_hr": {
        "level":   AlertLevel.WARNING,
        "message": "Driver heart rate elevated",
        "action":  "visual_warning",
    },
    "low_hr": {
        "level":   AlertLevel.CRITICAL,
        "message": "Driver heart rate critically low!",
        "action":  "audio_visual_emergency",
    },
    "high_stress": {
        "level":   AlertLevel.WARNING,
        "message": "Driver stress level high",
        "action":  "visual_warning",
    },
    "abnormal_breathing": {
        "level":   AlertLevel.WARNING,
        "message": "Abnormal breathing rate detected",
        "action":  "visual_warning",
    },
}


# ── AlertEngine ───────────────────────────────────────────────────────────────
class AlertEngine:
    """
    Processes detection outputs and emits structured alert events.

    Alert log is written as **JSON-Lines** (one entry per line, append mode).
    This avoids the O(n) full-file rewrite of the original implementation and
    is safe against interrupt-during-write corruption.

    Args:
        log_path: Path to the ``.jsonl`` alert log file.
    """

    def __init__(self, log_path: str | None = None) -> None:
        self._log_path     = Path(log_path or str(cfg.alert_log_path))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._alert_history: list[dict] = []
        self._last_alerts:   dict[str, float] = {}   # alert_type → last-fired timestamp

        print("[AlertEngine] Initialized — log:", self._log_path)

    # ── Cooldown ──────────────────────────────────────────────────────────────
    def _is_on_cooldown(self, alert_type: str) -> bool:
        """Return True if *alert_type* was fired within the cooldown window."""
        last = self._last_alerts.get(alert_type, 0.0)
        return time.time() - last < cfg.alert_cooldown_sec

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log_alert(
        self,
        alert_type: str,
        level: AlertLevel,
        message: str,
    ) -> dict:
        """
        Append one alert entry to the JSON-Lines log file and return it.

        JSON-Lines format: one valid JSON object per line, file opened in
        append mode — O(1) write, no full-file rewrite.
        """
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type":      alert_type,
            "level":     level.name,
            "message":   message,
        }
        self._alert_history.append(entry)
        self._last_alerts[alert_type] = time.time()

        # Append a single JSON line — safe and O(1)
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            print(f"[AlertEngine] Log write error: {exc}")

        print(f"[ALERT] [{level.name}] {message}")
        return entry

    # ── Processors ───────────────────────────────────────────────────────────
    def process_driver_state(self, driver_info: dict) -> list[dict]:
        """
        Generate alerts from :class:`~driver.driver_monitor.DriverMonitor` output.

        Args:
            driver_info: Dict with keys ``state``, ``ear``, ``perclos``,
                         ``yaw``, ``pitch``, ``alert``.

        Returns:
            List of newly emitted alert dicts (may be empty).
        """
        state  = driver_info.get("state", "alert")
        alerts: list[dict] = []

        if state in ALERT_RULES and not self._is_on_cooldown(state):
            rule  = ALERT_RULES[state]
            entry = self._log_alert(state, rule["level"], rule["message"])
            alerts.append(entry)

        return alerts

    def process_vitals(self, vitals_info: dict) -> list[dict]:
        """
        Generate alerts from :class:`~vitals.rppg.RPPGMonitor` output.

        Args:
            vitals_info: Dict with keys ``bpm``, ``breathing_rate``,
                         ``stress``, ``signal_quality``.

        Returns:
            List of newly emitted alert dicts (may be empty).
        """
        bpm     = vitals_info.get("bpm", 0)
        breath  = vitals_info.get("breathing_rate", 0)
        stress  = vitals_info.get("stress", "normal")
        quality = vitals_info.get("signal_quality", "weak")
        alerts: list[dict] = []

        # Only act on reliable signal
        if quality != "good":
            return alerts

        if bpm > 100 and not self._is_on_cooldown("high_hr"):
            rule  = ALERT_RULES["high_hr"]
            entry = self._log_alert("high_hr", rule["level"], rule["message"])
            alerts.append(entry)

        elif 0 < bpm < 50 and not self._is_on_cooldown("low_hr"):
            rule  = ALERT_RULES["low_hr"]
            entry = self._log_alert("low_hr", rule["level"], rule["message"])
            alerts.append(entry)

        if breath > 0 and not cfg.low_breathing <= breath <= cfg.high_breathing:
            if not self._is_on_cooldown("abnormal_breathing"):
                rule  = ALERT_RULES["abnormal_breathing"]
                entry = self._log_alert("abnormal_breathing", rule["level"], rule["message"])
                alerts.append(entry)

        if stress == "high" and not self._is_on_cooldown("high_stress"):
            rule  = ALERT_RULES["high_stress"]
            entry = self._log_alert("high_stress", rule["level"], rule["message"])
            alerts.append(entry)

        return alerts

    def process_occupants(self, occupant_info: dict) -> list[dict]:
        """
        Generate alerts from :class:`~occupant.occupant_detector.OccupantDetector` output.

        Args:
            occupant_info: Dict with keys ``total_occupied``, ``detections``.

        Returns:
            List of newly emitted alert dicts (may be empty).
        """
        detections = occupant_info.get("detections", [])
        labels     = [d["label"] for d in detections]
        alerts: list[dict] = []

        has_infant = "infant" in labels
        has_adult  = "adult"  in labels

        if has_infant and not has_adult and not self._is_on_cooldown("infant_alone"):
            rule  = ALERT_RULES["infant_alone"]
            entry = self._log_alert("infant_alone", rule["level"], rule["message"])
            alerts.append(entry)

        return alerts

    # ── Queries ───────────────────────────────────────────────────────────────
    def get_history(self) -> list[dict]:
        """Return the full in-memory alert history for this session."""
        return list(self._alert_history)

    def get_active_alerts(self) -> list[dict]:
        """
        Return alerts fired within the active-alert window
        (``cfg.active_alert_window_sec``, default 10 s).
        """
        now    = time.time()
        active = []
        for alert_type, timestamp in self._last_alerts.items():
            if now - timestamp < cfg.active_alert_window_sec:
                rule = ALERT_RULES.get(alert_type, {})
                active.append({
                    "type":    alert_type,
                    "message": rule.get("message", ""),
                    "level":   rule.get("level", AlertLevel.INFO).name,
                })
        return active


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = AlertEngine()

    print("\n--- Testing Driver States ---")
    engine.process_driver_state({"state": "sleeping"})
    engine.process_driver_state({"state": "phone_use"})
    engine.process_driver_state({"state": "distracted"})

    print("\n--- Testing Vitals ---")
    engine.process_vitals({
        "bpm": 110, "breathing_rate": 28,
        "stress": "high", "signal_quality": "good",
    })

    print("\n--- Testing Occupants ---")
    engine.process_occupants({
        "total_occupied": 1,
        "detections":     [{"label": "infant"}],
    })

    print("\n--- Alert History ---")
    for alert in engine.get_history():
        print(alert)