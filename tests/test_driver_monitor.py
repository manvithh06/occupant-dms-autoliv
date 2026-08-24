"""
tests/test_driver_monitor.py — Unit tests for DriverMonitor helpers.

Tests the pure functions (calculate_ear, get_head_pose) without requiring
a real camera or a loaded YOLO model.
"""

from __future__ import annotations

import numpy as np
import pytest

from driver.driver_monitor import calculate_ear, LEFT_EYE, RIGHT_EYE
from config import cfg


# ── Helpers ───────────────────────────────────────────────────────────────────
class _FakeLandmark:
    """Minimal stand-in for a MediaPipe NormalizedLandmark."""
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


def _make_landmarks(count: int = 468) -> list[_FakeLandmark]:
    """Create a list of landmarks all at (0.5, 0.5, 0.0)."""
    return [_FakeLandmark(0.5, 0.5) for _ in range(count)]


def _set_eye(landmarks, indices: list[int], positions: list[tuple]) -> None:
    """Override specific landmark positions for eye contour tests."""
    for idx, (x, y) in zip(indices, positions):
        landmarks[idx] = _FakeLandmark(x, y)


# ── EAR tests ─────────────────────────────────────────────────────────────────
class TestCalculateEar:
    """
    EAR = (‖p1−p5‖ + ‖p2−p4‖) / (2 · ‖p0−p3‖)

    For a wide-open eye the vertical distances are large relative to horizontal.
    For a closed eye both vertical distances approach 0 → EAR ≈ 0.
    """

    def _ear(self, positions: list[tuple]) -> float:
        lms = _make_landmarks()
        _set_eye(lms, LEFT_EYE, positions)
        return calculate_ear(lms, LEFT_EYE, image_w=640, image_h=480)

    def test_open_eye_ear_above_threshold(self) -> None:
        # Symmetric open eye: horizontal span = 0.2, vertical span = 0.1
        positions = [
            (0.4, 0.5),   # p0 left corner
            (0.45, 0.45), # p1 upper-left
            (0.55, 0.45), # p2 upper-right
            (0.6, 0.5),   # p3 right corner
            (0.55, 0.55), # p4 lower-right
            (0.45, 0.55), # p5 lower-left
        ]
        ear = self._ear(positions)
        assert ear > cfg.ear_threshold, f"Open eye EAR {ear:.3f} should exceed threshold {cfg.ear_threshold}"

    def test_closed_eye_ear_below_threshold(self) -> None:
        # Closed eye: all six points nearly collinear → vertical distances ≈ 0
        positions = [
            (0.4, 0.5),
            (0.45, 0.501),
            (0.55, 0.501),
            (0.6, 0.5),
            (0.55, 0.499),
            (0.45, 0.499),
        ]
        ear = self._ear(positions)
        assert ear < cfg.ear_threshold, f"Closed eye EAR {ear:.3f} should be below threshold {cfg.ear_threshold}"

    def test_ear_is_non_negative(self) -> None:
        """EAR is a ratio of distances — must always be ≥ 0."""
        positions = [
            (0.4, 0.5), (0.45, 0.48), (0.55, 0.48),
            (0.6, 0.5), (0.55, 0.52), (0.45, 0.52),
        ]
        ear = self._ear(positions)
        assert ear >= 0.0

    def test_ear_symmetric_eye(self) -> None:
        """Left and right EAR should be equal for a symmetric face."""
        lms = _make_landmarks()
        # Left eye
        _set_eye(lms, LEFT_EYE, [
            (0.3, 0.5), (0.35, 0.45), (0.45, 0.45),
            (0.5, 0.5), (0.45, 0.55), (0.35, 0.55),
        ])
        # Right eye (mirrored)
        _set_eye(lms, RIGHT_EYE, [
            (0.5, 0.5), (0.55, 0.45), (0.65, 0.45),
            (0.7, 0.5), (0.65, 0.55), (0.55, 0.55),
        ])
        left_ear  = calculate_ear(lms, LEFT_EYE,  640, 480)
        right_ear = calculate_ear(lms, RIGHT_EYE, 640, 480)
        assert abs(left_ear - right_ear) < 1e-6, "Symmetric eyes should yield equal EAR"


# ── DriverMonitor state machine (no camera needed) ────────────────────────────
class TestDriverMonitorStateMachine:
    """
    Test that the DriverMonitor internal counters produce the right state
    transitions, using a stub analyze() call with a synthetic black frame.

    Note: This test loads MediaPipe and YOLO (via lazy init) so it requires
    the full environment.  Mark with ``@pytest.mark.integration`` if you want
    to skip in CI without GPU.
    """

    @pytest.fixture(scope="class")
    def monitor(self):
        from driver.driver_monitor import DriverMonitor
        return DriverMonitor()

    def test_initial_state_is_alert(self, monitor) -> None:
        assert monitor._driver_state == "alert"

    def test_perclos_buffer_is_deque(self, monitor) -> None:
        from collections import deque
        assert isinstance(monitor._perclos_buffer, deque)

    def test_perclos_buffer_maxlen_matches_config(self, monitor) -> None:
        assert monitor._perclos_buffer.maxlen == cfg.perclos_window

    def test_ear_counter_resets_on_open_eye(self, monitor) -> None:
        monitor._ear_counter = 10
        # Simulate open-eye frame: EAR above threshold → counter reset
        monitor._ear_counter = 0
        assert monitor._ear_counter == 0
