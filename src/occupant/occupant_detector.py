"""
occupant/occupant_detector.py — YOLOv8-based vehicle occupant detector.

Uses a size-heuristic to distinguish adult / child / infant when a generic
COCO-pretrained model is in use (until a fine-tuned model is available).
The heuristic compares the detected bounding-box height to the frame height:

    bbox_h / frame_h  > adult_ratio_threshold  → "adult"
    bbox_h / frame_h  > child_ratio_threshold  → "child"
    otherwise                                  → "infant"

Thresholds are configurable via config.py / config.yaml.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from config import cfg

# ── COCO classes that count as pets ──────────────────────────────────────────
# 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow
_PET_CLASSES: frozenset[int] = frozenset({15, 16, 17, 18, 19})

# ── Display colours per label ─────────────────────────────────────────────────
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "empty":  (128, 128, 128),  # grey
    "adult":  (0,   255, 0),    # green
    "child":  (0,   255, 255),  # yellow
    "infant": (0,   0,   255),  # red
    "pet":    (255, 165, 0),    # orange
}


def _classify_person(bbox: list[float], frame_h: int) -> str:
    """
    Classify a detected ``person`` bounding box as adult / child / infant
    using a height-ratio heuristic.

    This is an **interim approach** until a fine-tuned model that explicitly
    detects occupant age categories is available.  Replace this function (or
    override with a trained model) for production accuracy.

    Args:
        bbox:    [x1, y1, x2, y2] bounding box in pixels.
        frame_h: Height of the source frame in pixels.

    Returns:
        One of ``"adult"``, ``"child"``, or ``"infant"``.
    """
    if frame_h <= 0:
        return "adult"
    bbox_h = bbox[3] - bbox[1]
    ratio  = bbox_h / frame_h
    if ratio >= cfg.adult_ratio_threshold:
        return "adult"
    if ratio >= cfg.child_ratio_threshold:
        return "child"
    return "infant"


class OccupantDetector:
    """
    Detects and classifies vehicle occupants using YOLOv8.

    Args:
        model_path:  Path to the YOLOv8 weights file.
        confidence:  Minimum detection confidence threshold (0–1).
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
    ) -> None:
        _path = model_path or str(cfg.model_path)
        self.model      = YOLO(_path)
        self.confidence = confidence if confidence is not None else cfg.occupant_conf
        self._frame_count = 0
        self._last_detections: list[dict] = []
        print(f"[OccupantDetector] Model loaded: {_path}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLO detection on a single frame.

        Args:
            frame: BGR image as a NumPy array.

        Returns:
            List of detection dicts:
                ``{"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}``
        """
        self._frame_count += 1
        if self._frame_count % cfg.yolo_frame_skip == 0 or not self._last_detections:
            frame_h = frame.shape[0]
            results  = self.model(frame, conf=self.confidence, verbose=False, imgsz=cfg.yolo_imgsz)
            detections: list[dict] = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    bbox   = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    if cls_id == 0:
                        # person — use size heuristic to sub-classify
                        label = _classify_person(bbox, frame_h)
                    elif cls_id in _PET_CLASSES:
                        label = "pet"
                    else:
                        # skip irrelevant COCO classes (cars, furniture, etc.)
                        continue

                    detections.append({
                        "label":      label,
                        "confidence": round(conf, 2),
                        "bbox":       bbox,
                    })
            self._last_detections = detections

        return self._last_detections

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on *frame* in-place.

        Args:
            frame:      BGR image.
            detections: Output of :meth:`detect`.

        Returns:
            Annotated frame.
        """
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label  = det["label"]
            conf   = det["confidence"]
            color  = CLASS_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.0%}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        return frame

    def get_seat_map(self, detections: list[dict]) -> dict:
        """
        Return a summary of vehicle occupancy.

        Args:
            detections: Output of :meth:`detect`.

        Returns:
            ``{"total_occupied": int, "detections": list}``
        """
        occupied = len([d for d in detections if d["label"] != "empty"])
        return {
            "total_occupied": occupied,
            "detections":     detections,
        }


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = OccupantDetector()
    cap      = cv2.VideoCapture(cfg.camera_index)
    print("[INFO] Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        frame      = detector.draw(frame, detections)
        seat_map   = detector.get_seat_map(detections)

        cv2.putText(
            frame,
            f"Occupied: {seat_map['total_occupied']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.imshow("Occupant Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()