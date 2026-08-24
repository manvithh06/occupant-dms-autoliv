"""
tests/test_alert_engine.py — Unit tests for AlertEngine.

These tests exercise:
* Cooldown logic (same alert not re-fired within cooldown window)
* process_driver_state — known alert vs. safe state
* process_vitals — high HR, low HR, abnormal breathing, high stress
* process_occupants — infant alone, infant with adult (no alert)
* get_active_alerts — only recent alerts returned
* JSON-Lines log file created and parseable
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from alerts.alert_engine import AlertEngine, AlertLevel


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def engine(tmp_path: Path) -> AlertEngine:
    """Fresh AlertEngine writing to a temp log file."""
    log_file = tmp_path / "alerts.jsonl"
    return AlertEngine(log_path=str(log_file))


def _good_vitals(**kwargs) -> dict:
    """Base vitals dict with signal_quality='good' and all values in safe range."""
    base = {
        "bpm":            72,
        "breathing_rate": 16,
        "stress":         "normal",
        "signal_quality": "good",
    }
    base.update(kwargs)
    return base


def _driver(**kwargs) -> dict:
    base = {"state": "alert", "ear": 0.30, "perclos": 0.1, "yaw": 5.0, "pitch": 2.0, "alert": None}
    base.update(kwargs)
    return base


# ── Cooldown ──────────────────────────────────────────────────────────────────
class TestCooldown:
    def test_first_alert_fires(self, engine: AlertEngine) -> None:
        alerts = engine.process_driver_state(_driver(state="drowsy"))
        assert len(alerts) == 1

    def test_second_alert_suppressed_during_cooldown(self, engine: AlertEngine) -> None:
        engine.process_driver_state(_driver(state="drowsy"))
        alerts = engine.process_driver_state(_driver(state="drowsy"))
        assert len(alerts) == 0, "Alert should be on cooldown"

    def test_alert_refires_after_cooldown(self, engine: AlertEngine, monkeypatch) -> None:
        engine.process_driver_state(_driver(state="drowsy"))
        # Advance the stored timestamp by (cooldown + 1) seconds
        from config import cfg
        engine._last_alerts["drowsy"] = time.time() - cfg.alert_cooldown_sec - 1
        alerts = engine.process_driver_state(_driver(state="drowsy"))
        assert len(alerts) == 1


# ── Driver state ──────────────────────────────────────────────────────────────
class TestProcessDriverState:
    @pytest.mark.parametrize("state", ["drowsy", "sleeping", "distracted", "phone_use"])
    def test_known_states_produce_alerts(self, engine: AlertEngine, state: str) -> None:
        alerts = engine.process_driver_state(_driver(state=state))
        assert len(alerts) == 1
        assert alerts[0]["type"] == state

    def test_alert_state_produces_no_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_driver_state(_driver(state="alert"))
        assert len(alerts) == 0

    def test_unknown_state_produces_no_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_driver_state(_driver(state="no_face"))
        assert len(alerts) == 0

    def test_alert_has_required_keys(self, engine: AlertEngine) -> None:
        alerts = engine.process_driver_state(_driver(state="sleeping"))
        entry  = alerts[0]
        assert {"timestamp", "type", "level", "message"} <= entry.keys()

    def test_sleeping_is_critical(self, engine: AlertEngine) -> None:
        alerts = engine.process_driver_state(_driver(state="sleeping"))
        assert alerts[0]["level"] == "CRITICAL"

    def test_drowsy_is_warning(self, engine: AlertEngine) -> None:
        alerts = engine.process_driver_state(_driver(state="drowsy"))
        assert alerts[0]["level"] == "WARNING"


# ── Vitals ─────────────────────────────────────────────────────────────────────
class TestProcessVitals:
    def test_weak_signal_suppresses_all_alerts(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(
            bpm=110, signal_quality="weak"
        ))
        assert len(alerts) == 0

    def test_high_bpm_produces_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(bpm=110))
        types  = [a["type"] for a in alerts]
        assert "high_hr" in types

    def test_low_bpm_produces_critical_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(bpm=40))
        types  = [a["type"] for a in alerts]
        assert "low_hr" in types
        assert any(a["level"] == "CRITICAL" for a in alerts if a["type"] == "low_hr")

    def test_normal_bpm_no_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(bpm=72))
        assert not any(a["type"] in ("high_hr", "low_hr") for a in alerts)

    def test_zero_bpm_no_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(bpm=0))
        assert len(alerts) == 0

    def test_high_breathing_rate_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(breathing_rate=30))
        types  = [a["type"] for a in alerts]
        assert "abnormal_breathing" in types

    def test_low_breathing_rate_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(breathing_rate=4))
        types  = [a["type"] for a in alerts]
        assert "abnormal_breathing" in types

    def test_normal_breathing_no_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(breathing_rate=16))
        assert not any(a["type"] == "abnormal_breathing" for a in alerts)

    def test_high_stress_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(stress="high"))
        assert any(a["type"] == "high_stress" for a in alerts)

    def test_normal_stress_no_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_vitals(_good_vitals(stress="normal"))
        assert not any(a["type"] == "high_stress" for a in alerts)


# ── Occupants ─────────────────────────────────────────────────────────────────
class TestProcessOccupants:
    def _seat_map(self, *labels) -> dict:
        return {
            "total_occupied": len(labels),
            "detections":     [{"label": l, "confidence": 0.9, "bbox": [0, 0, 50, 50]} for l in labels],
        }

    def test_infant_alone_fires_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_occupants(self._seat_map("infant"))
        assert len(alerts) == 1
        assert alerts[0]["type"] == "infant_alone"
        assert alerts[0]["level"] == "CRITICAL"

    def test_infant_with_adult_no_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_occupants(self._seat_map("infant", "adult"))
        assert len(alerts) == 0

    def test_adult_only_no_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_occupants(self._seat_map("adult"))
        assert len(alerts) == 0

    def test_empty_vehicle_no_alert(self, engine: AlertEngine) -> None:
        alerts = engine.process_occupants({"total_occupied": 0, "detections": []})
        assert len(alerts) == 0

    def test_child_alone_no_infant_alert(self, engine: AlertEngine) -> None:
        """A child alone does NOT trigger infant_alone (infant class is distinct)."""
        alerts = engine.process_occupants(self._seat_map("child"))
        assert not any(a["type"] == "infant_alone" for a in alerts)


# ── History and active alerts ─────────────────────────────────────────────────
class TestHistoryAndActive:
    def test_get_history_returns_all_fired_alerts(self, engine: AlertEngine) -> None:
        engine.process_driver_state(_driver(state="drowsy"))
        engine.process_driver_state(_driver(state="sleeping"))
        history = engine.get_history()
        assert len(history) == 2

    def test_get_active_alerts_returns_recent(self, engine: AlertEngine) -> None:
        engine.process_driver_state(_driver(state="drowsy"))
        active = engine.get_active_alerts()
        assert any(a["type"] == "drowsy" for a in active)

    def test_get_active_alerts_excludes_old(self, engine: AlertEngine) -> None:
        engine.process_driver_state(_driver(state="drowsy"))
        # Push timestamp into the past
        from config import cfg
        engine._last_alerts["drowsy"] = time.time() - cfg.active_alert_window_sec - 1
        active = engine.get_active_alerts()
        assert not any(a["type"] == "drowsy" for a in active)


# ── JSON-Lines log ────────────────────────────────────────────────────────────
class TestJsonLinesLog:
    def test_log_file_created_on_first_alert(self, engine: AlertEngine) -> None:
        engine.process_driver_state(_driver(state="drowsy"))
        assert engine._log_path.exists()

    def test_log_file_is_valid_jsonlines(self, engine: AlertEngine) -> None:
        engine.process_driver_state(_driver(state="drowsy"))
        engine.process_driver_state(_driver(state="sleeping"))
        # Force second firing (skip cooldown)
        engine._last_alerts.pop("sleeping", None)
        engine.process_driver_state(_driver(state="phone_use"))

        lines = engine._log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        for line in lines:
            obj = json.loads(line)   # must not raise
            assert "timestamp" in obj
            assert "type"      in obj
            assert "level"     in obj

    def test_multiple_alerts_appended_not_overwritten(self, engine: AlertEngine) -> None:
        engine.process_driver_state(_driver(state="drowsy"))
        engine._last_alerts.pop("sleeping", None)
        engine._last_alerts.pop("drowsy",   None)
        engine.process_driver_state(_driver(state="sleeping"))

        lines = engine._log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2, "Each alert should add one line (append mode)"
