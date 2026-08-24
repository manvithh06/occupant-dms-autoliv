"""
config.py — Centralized configuration for Occupant DMS.

All tunable constants live here. Override any value by creating a
``config.yaml`` at the project root:

    ear_threshold: 0.22
    alert_cooldown_sec: 10

Values in config.yaml take precedence over the defaults below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml  # PyYAML is already in requirements.txt

log = logging.getLogger("DMS.config")

# ── Project root (two levels up from this file: src/config.py → root) ────────
_ROOT = Path(__file__).resolve().parent.parent
_SRC  = Path(__file__).resolve().parent


@dataclass
class DMSConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    # Absolute path to the YOLOv8 weights file.
    # Falls back to the bundled yolov8n.pt inside src/ if a fine-tuned model
    # is not yet present in models/.
    model_path: Path = field(
        default_factory=lambda: (
            _ROOT / "models" / "occupant_dms.pt"
            if (_ROOT / "models" / "occupant_dms.pt").exists()
            else _SRC / "yolov8n.pt"
        )
    )

    # ── Driver Monitor ────────────────────────────────────────────────────────
    # Eye Aspect Ratio — below this value the eye is considered closed.
    ear_threshold: float = 0.25
    # Consecutive frames of closed eyes required to trigger "sleeping".
    ear_consec_frames: int = 20
    # Sliding window length (frames) for PERCLOS calculation (~3 s @ 30 fps).
    perclos_window: int = 90
    # Fraction of closed-eye frames in the window that triggers "drowsy".
    perclos_threshold: float = 0.80
    # Head yaw (degrees) beyond which driver is considered "distracted".
    yaw_threshold: float = 20.0
    # Head pitch (degrees) beyond which driver is considered "distracted".
    pitch_threshold: float = 15.0
    # Confidence threshold for phone-use YOLO detection.
    phone_conf: float = 0.40

    # ── rPPG / Vitals ─────────────────────────────────────────────────────────
    # Target camera frames per second used for signal analysis.
    # The pipeline will attempt to measure actual FPS and override this.
    rppg_fps: int = 30
    # Seconds of signal history to analyse per BPM calculation.
    rppg_window_seconds: int = 10
    # Valid heart-rate range (BPM) — values outside are discarded.
    low_bpm: int = 45
    high_bpm: int = 180
    # Valid breathing-rate range (breaths / min).
    low_breathing: float = 8.0
    high_breathing: float = 25.0

    # ── Occupant Detector ─────────────────────────────────────────────────────
    # Minimum YOLO confidence to accept a detection.
    occupant_conf: float = 0.50
    # Bounding box image size passed to YOLO (defaults to 320 for CPU speed).
    yolo_imgsz: int = 320
    # Process YOLO only every N frames (defaults to 2 for 2x CPU speedup).
    yolo_frame_skip: int = 2
    # Bounding-box height / frame height thresholds for person size heuristic.
    # Larger ratio → adult; intermediate → child; small → infant.
    adult_ratio_threshold: float = 0.45
    child_ratio_threshold: float = 0.25

    # ── Alert Engine ──────────────────────────────────────────────────────────
    # Seconds before the same alert type can be fired again.
    alert_cooldown_sec: float = 5.0
    # "Active" window — alerts within this many seconds are shown on dashboard.
    active_alert_window_sec: float = 10.0

    # ── Notification (main.py) ───────────────────────────────────────────────
    # Seconds before the same OS notification is re-sent.
    notif_cooldown_sec: float = 15.0

    # ── Paths ─────────────────────────────────────────────────────────────────
    log_dir: Path = field(default_factory=lambda: _ROOT / "logs")
    alert_log_path: Path = field(default_factory=lambda: _ROOT / "logs" / "alert_log.jsonl")
    state_log_path: Path = field(default_factory=lambda: _ROOT / "logs" / "latest_state.json")
    demo_output_dir: Path = field(default_factory=lambda: _ROOT / "demo")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # Target frames per second for the main processing loop.
    target_fps: int = 30
    # Save system state to disk every N frames.
    state_save_interval: int = 30

    # ── Camera ────────────────────────────────────────────────────────────────
    camera_index: int = 0


def _load_yaml_overrides(cfg: DMSConfig, yaml_path: Path) -> None:
    """Apply overrides from ``config.yaml`` onto an existing DMSConfig object."""
    if not yaml_path.exists():
        return
    try:
        with yaml_path.open() as f:
            overrides: dict = yaml.safe_load(f) or {}
        for key, value in overrides.items():
            if hasattr(cfg, key):
                # Convert string paths back to Path objects when needed.
                if isinstance(getattr(cfg, key), Path):
                    value = Path(value)
                setattr(cfg, key, value)
                log.debug("config override: %s = %s", key, value)
            else:
                log.warning("config.yaml: unknown key '%s' (ignored)", key)
    except Exception as exc:
        log.error("Failed to load config.yaml: %s", exc)


def _build() -> DMSConfig:
    cfg = DMSConfig()
    _load_yaml_overrides(cfg, _ROOT / "config.yaml")
    # Ensure log directory exists.
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# ── Module-level singleton ────────────────────────────────────────────────────
cfg: DMSConfig = _build()


# ── Convenience re-exports (match old import style in existing files) ─────────
MODEL_PATH          = cfg.model_path
EAR_THRESHOLD       = cfg.ear_threshold
EAR_CONSEC_FRAMES   = cfg.ear_consec_frames
PERCLOS_WINDOW      = cfg.perclos_window
PERCLOS_THRESHOLD   = cfg.perclos_threshold
YAW_THRESHOLD       = cfg.yaw_threshold
PITCH_THRESHOLD     = cfg.pitch_threshold
RPPG_FPS            = cfg.rppg_fps
LOW_BPM             = cfg.low_bpm
HIGH_BPM            = cfg.high_bpm
ALERT_COOLDOWN_SEC  = cfg.alert_cooldown_sec
NOTIF_COOLDOWN_SEC  = cfg.notif_cooldown_sec
LOG_DIR             = cfg.log_dir
ALERT_LOG_PATH      = cfg.alert_log_path
STATE_LOG_PATH      = cfg.state_log_path
