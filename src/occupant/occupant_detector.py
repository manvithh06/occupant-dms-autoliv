import cv2
from ultralytics import YOLO
import numpy as np

# Class names for occupant detection
OCCUPANT_CLASSES = {
    0: "empty",
    1: "adult",
    2: "child",
    3: "infant",
    4:  "pet"
}

# Alert colors for each class
CLASS_COLORS = {
    "empty":  (128, 128, 128),  # grey
    "adult":  (0, 255, 0),      # green
    "child":  (0, 255, 255),    # yellow
    "infant": (0, 0, 255),      # red
    "pet":    (255, 165, 0),    # orange
}

class OccupantDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        """
        Initialize occupant detector.
        We use pretrained YOLOv8 for now (detects 'person' class).
        Fine-tuning on occupant-specific data comes later.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"[OccupantDetector] Model loaded: {model_path}")

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns: list of detections with bbox, class, confidence
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                # For now map COCO 'person' class (0) to 'adult'
                # COCO class mappings
                # 0: person, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow
                PET_CLASSES = [15, 16, 17, 18, 19]

                if cls_id == 0:
                    label = "adult"
                elif cls_id in PET_CLASSES:
                    label = "pet"
                else:
                    label = "empty"

                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": bbox
                })

        return detections

    def draw(self, frame, detections):
        """
        Draw bounding boxes and labels on frame.
        """
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = det["label"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(label, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        return frame

    def get_seat_map(self, detections):
        """
        Returns a simple seat occupancy summary.
        """
        occupied = len([d for d in detections if d["label"] != "empty"])
        return {
            "total_occupied": occupied,
            "detections": detections
        }


# Quick test
if __name__ == "__main__":
    detector = OccupantDetector()
    cap = cv2.VideoCapture(0)  # webcam

    print("[INFO] Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        frame = detector.draw(frame, detections)
        seat_map = detector.get_seat_map(detections)

        cv2.putText(
            frame,
            f"Occupied seats: {seat_map['total_occupied']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

        cv2.imshow("Occupant Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()