import time
import json
from datetime import datetime
from enum import Enum

# ─── Alert Levels ────────────────────────────────────────
class AlertLevel(Enum):
    NONE     = 0
    INFO     = 1   # just log it
    WARNING  = 2   # visual warning on dashboard
    CRITICAL = 3   # audio + visual + log

# ─── Alert Types ─────────────────────────────────────────
ALERT_RULES = {
    # Driver states
    "drowsy": {
        "level":   AlertLevel.WARNING,
        "message": "Driver appears drowsy",
        "action":  "visual_warning"
    },
    "sleeping": {
        "level":   AlertLevel.CRITICAL,
        "message": "Driver is sleeping!",
        "action":  "audio_visual_emergency"
    },
    "distracted": {
        "level":   AlertLevel.WARNING,
        "message": "Driver is distracted",
        "action":  "visual_warning"
    },
    "phone_use": {
        "level":   AlertLevel.CRITICAL,
        "message": "Driver using phone!",
        "action":  "audio_visual_emergency"
    },

    # Occupant states
    "infant_alone": {
        "level":   AlertLevel.CRITICAL,
        "message": "Infant left alone in vehicle!",
        "action":  "audio_visual_emergency"
    },
    "seatbelt_off": {
        "level":   AlertLevel.WARNING,
        "message": "Occupant seatbelt not fastened",
        "action":  "visual_warning"
    },

    # Vital signs
    "high_hr": {
        "level":   AlertLevel.WARNING,
        "message": "Driver heart rate elevated",
        "action":  "visual_warning"
    },
    "low_hr": {
        "level":   AlertLevel.CRITICAL,
        "message": "Driver heart rate critically low!",
        "action":  "audio_visual_emergency"
    },
    "high_stress": {
        "level":   AlertLevel.WARNING,
        "message": "Driver stress level high",
        "action":  "visual_warning"
    },
    "abnormal_breathing": {
        "level":   AlertLevel.WARNING,
        "message": "Abnormal breathing rate detected",
        "action":  "visual_warning"
    },
}

# ─── Alert Engine ────────────────────────────────────────
class AlertEngine:
    def __init__(self, log_path="demo/alert_log.json"):
        self.log_path      = log_path
        self.alert_history = []
        self.last_alerts   = {}   # cooldown tracker
        self.COOLDOWN_SEC  = 5    # don't repeat same alert within 5s

        print("[AlertEngine] Initialized")

    def _is_on_cooldown(self, alert_type):
        """Prevent spamming same alert repeatedly."""
        if alert_type not in self.last_alerts:
            return False
        elapsed = time.time() - self.last_alerts[alert_type]
        return elapsed < self.COOLDOWN_SEC

    def _log_alert(self, alert_type, level, message):
        """Save alert to history and JSON log."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type":      alert_type,
            "level":     level.name,
            "message":   message
        }
        self.alert_history.append(entry)
        self.last_alerts[alert_type] = time.time()

        # Write to JSON log
        try:
            with open(self.log_path, "w") as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            print(f"[AlertEngine] Log error: {e}")

        print(f"[ALERT] [{level.name}] {message}")
        return entry

    def process_driver_state(self, driver_info):
        """
        Process output from DriverMonitor.
        driver_info = {state, ear, perclos, yaw, pitch, alert}
        """
        state  = driver_info.get("state", "alert")
        alerts = []

        if state in ALERT_RULES and not self._is_on_cooldown(state):
            rule  = ALERT_RULES[state]
            entry = self._log_alert(state, rule["level"], rule["message"])
            alerts.append(entry)

        return alerts

    def process_vitals(self, vitals_info):
        """
        Process output from RPPGMonitor.
        vitals_info = {bpm, breathing_rate, stress, signal_quality}
        """
        bpm      = vitals_info.get("bpm", 0)
        breath   = vitals_info.get("breathing_rate", 0)
        stress   = vitals_info.get("stress", "normal")
        quality  = vitals_info.get("signal_quality", "weak")
        alerts   = []

        # Only process if signal is good
        if quality != "good":
            return alerts

        # Heart rate checks
        if bpm > 100 and not self._is_on_cooldown("high_hr"):
            rule  = ALERT_RULES["high_hr"]
            entry = self._log_alert("high_hr", rule["level"], rule["message"])
            alerts.append(entry)

        elif bpm < 50 and bpm > 0 and not self._is_on_cooldown("low_hr"):
            rule  = ALERT_RULES["low_hr"]
            entry = self._log_alert("low_hr", rule["level"], rule["message"])
            alerts.append(entry)

        # Breathing rate check (normal: 12-20 breaths/min)
        if breath > 0 and (breath < 8 or breath > 25):
            if not self._is_on_cooldown("abnormal_breathing"):
                rule  = ALERT_RULES["abnormal_breathing"]
                entry = self._log_alert(
                    "abnormal_breathing", rule["level"], rule["message"]
                )
                alerts.append(entry)

        # Stress check
        if stress == "high" and not self._is_on_cooldown("high_stress"):
            rule  = ALERT_RULES["high_stress"]
            entry = self._log_alert("high_stress", rule["level"], rule["message"])
            alerts.append(entry)

        return alerts

    def process_occupants(self, occupant_info):
        """
        Process output from OccupantDetector.
        occupant_info = {total_occupied, detections}
        """
        alerts     = []
        detections = occupant_info.get("detections", [])

        # Check for infant alone (no adult present)
        labels     = [d["label"] for d in detections]
        has_infant = "infant" in labels
        has_adult  = "adult"  in labels

        if has_infant and not has_adult:
            if not self._is_on_cooldown("infant_alone"):
                rule  = ALERT_RULES["infant_alone"]
                entry = self._log_alert(
                    "infant_alone", rule["level"], rule["message"]
                )
                alerts.append(entry)

        return alerts

    def get_history(self):
        """Return full alert history."""
        return self.alert_history

    def get_active_alerts(self):
        """Return alerts from last 10 seconds."""
        now     = time.time()
        active  = []
        for alert_type, timestamp in self.last_alerts.items():
            if now - timestamp < 10:
                rule = ALERT_RULES.get(alert_type, {})
                active.append({
                    "type":    alert_type,
                    "message": rule.get("message", ""),
                    "level":   rule.get("level", AlertLevel.INFO).name
                })
        return active


# ─── Quick Test ──────────────────────────────────────────
if __name__ == "__main__":
    engine = AlertEngine()

    # Simulate driver sleeping
    print("\n--- Testing Driver States ---")
    engine.process_driver_state({"state": "sleeping"})
    engine.process_driver_state({"state": "phone_use"})
    engine.process_driver_state({"state": "distracted"})

    # Simulate vitals
    print("\n--- Testing Vitals ---")
    engine.process_vitals({
        "bpm": 110,
        "breathing_rate": 28,
        "stress": "high",
        "signal_quality": "good"
    })

    # Simulate occupants
    print("\n--- Testing Occupants ---")
    engine.process_occupants({
        "total_occupied": 1,
        "detections": [{"label": "infant"}]
    })

    print("\n--- Alert History ---")
    for alert in engine.get_history():
        print(alert)