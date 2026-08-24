"""
tests/test_occupant_detector.py — Unit tests for OccupantDetector.

Exercises:
* _classify_person() size heuristic (adult / child / infant)
* get_seat_map() occupancy count
* draw() does not raise for typical inputs
* OCCUPANT_CLASSES and CLASS_COLORS constants are complete
"""

from __future__ import annotations

import numpy as np
import pytest

from occupant.occupant_detector import (
    OccupantDetector,
    CLASS_COLORS,
    _classify_person,
    _PET_CLASSES,
)
from config import cfg


# ── _classify_person size heuristic ──────────────────────────────────────────
class TestClassifyPerson:
    """
    Bounding box is [x1, y1, x2, y2].
    Frame height is used to compute bbox_h / frame_h ratio.
    """

    FRAME_H = 480

    def _bbox(self, y1: int, y2: int) -> list[float]:
        return [0.0, float(y1), 640.0, float(y2)]

    def test_tall_bbox_is_adult(self) -> None:
        # bbox_h = 250, frame_h = 480 → ratio ≈ 0.52 > adult_ratio_threshold (0.45)
        label = _classify_person(self._bbox(100, 350), self.FRAME_H)
        assert label == "adult"

    def test_medium_bbox_is_child(self) -> None:
        # bbox_h = 150, frame_h = 480 → ratio ≈ 0.31 — between child and adult thresholds
        label = _classify_person(self._bbox(200, 350), self.FRAME_H)
        assert label == "child"

    def test_small_bbox_is_infant(self) -> None:
        # bbox_h = 80, frame_h = 480 → ratio ≈ 0.17 < child_ratio_threshold (0.25)
        label = _classify_person(self._bbox(300, 380), self.FRAME_H)
        assert label == "infant"

    def test_zero_frame_height_returns_adult(self) -> None:
        """Guard against division by zero — should default to 'adult'."""
        label = _classify_person([0, 100, 640, 300], 0)
        assert label == "adult"

    def test_thresholds_are_configurable(self) -> None:
        """Thresholds come from config, not magic numbers."""
        assert cfg.adult_ratio_threshold > cfg.child_ratio_threshold > 0


# ── get_seat_map ──────────────────────────────────────────────────────────────
class TestGetSeatMap:
    @pytest.fixture(scope="class")
    def detector(self):
        return OccupantDetector.__new__(OccupantDetector)   # no model load

    def _make_detections(self, *labels) -> list[dict]:
        return [
            {"label": l, "confidence": 0.9, "bbox": [0.0, 0.0, 100.0, 100.0]}
            for l in labels
        ]

    def test_occupied_count_correct(self, detector) -> None:
        dets = self._make_detections("adult", "child")
        seat_map = detector.get_seat_map(dets)
        assert seat_map["total_occupied"] == 2

    def test_empty_vehicle(self, detector) -> None:
        seat_map = detector.get_seat_map([])
        assert seat_map["total_occupied"] == 0

    def test_detections_preserved_in_seat_map(self, detector) -> None:
        dets = self._make_detections("infant", "adult")
        seat_map = detector.get_seat_map(dets)
        assert seat_map["detections"] == dets

    def test_empty_label_not_counted(self, detector) -> None:
        dets = self._make_detections("empty", "adult")
        seat_map = detector.get_seat_map(dets)
        # "empty" is excluded from occupied count
        assert seat_map["total_occupied"] == 1


# ── draw() ────────────────────────────────────────────────────────────────────
class TestDraw:
    @pytest.fixture(scope="class")
    def detector(self):
        return OccupantDetector.__new__(OccupantDetector)

    def test_draw_returns_frame(self, detector) -> None:
        frame      = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [{"label": "adult", "confidence": 0.95, "bbox": [10.0, 10.0, 200.0, 300.0]}]
        result     = detector.draw(frame, detections)
        assert result.shape == (480, 640, 3)

    def test_draw_empty_detections_returns_unchanged(self, detector) -> None:
        frame  = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.draw(frame, [])
        np.testing.assert_array_equal(result, frame)


# ── Constants completeness ────────────────────────────────────────────────────
class TestConstants:
    def test_class_colors_covers_all_labels(self) -> None:
        required = {"empty", "adult", "child", "infant", "pet"}
        assert required <= CLASS_COLORS.keys()

    def test_class_colors_are_bgr_tuples(self) -> None:
        for label, color in CLASS_COLORS.items():
            assert len(color) == 3, f"{label} color should be a 3-tuple"
            assert all(0 <= c <= 255 for c in color), f"{label} color values out of range"

    def test_pet_classes_is_frozenset(self) -> None:
        assert isinstance(_PET_CLASSES, frozenset)

    def test_pet_classes_contains_cat_and_dog(self) -> None:
        assert 15 in _PET_CLASSES   # cat
        assert 16 in _PET_CLASSES   # dog
