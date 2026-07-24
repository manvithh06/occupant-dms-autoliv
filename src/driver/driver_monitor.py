from unittest import result

import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from scipy.spatial import distance

# ─── Constants ───────────────────────────────────────────
EAR_THRESHOLD    = 0.25   # below this = eyes closing
EAR_CONSEC_FRAMES = 20    # frames eyes must be closed to trigger
PERCLOS_WINDOW   = 90     # frames to calculate PERCLOS over (~3s at 30fps)
PERCLOS_THRESHOLD = 0.8   # 80% eye closure = drowsy
YAW_THRESHOLD    = 20     # degrees head turn = distracted

# ─── MediaPipe Setup ─────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Eye landmark indices (MediaPipe 468 landmarks)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Nose and face boundary for head pose
NOSE_TIP     = 4
CHIN         = 152
LEFT_EAR_PT  = 234
RIGHT_EAR_PT = 454
LEFT_MOUTH   = 287
RIGHT_MOUTH  = 57

# ─── EAR Calculation ─────────────────────────────────────
def calculate_ear(landmarks, eye_indices, image_w, image_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * image_w, lm.y * image_h))

    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ─── Head Pose Estimation ────────────────────────────────
def get_head_pose(landmarks, image_w, image_h):
    face_3d = []
    face_2d = []

    key_indices = [1, 9, 57, 130, 287, 359]

    for idx in key_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = image_w
    cam_matrix = np.array([
        [focal_length, 0, image_w / 2],
        [0, focal_length, image_h / 2],
        [0, 0, 1]
    ])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0] * 360
    yaw   = angles[1] * 360
    roll  = angles[2] * 360

    return pitch, yaw, roll


# ─── Main Driver Monitor Class ───────────────────────────
class DriverMonitor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False

        )
        self.ear_counter     = 0
        self.yolo = YOLO("yolov8n.pt")
        self.perclos_buffer  = []
        self.driver_state    = "alert"

        print("[DriverMonitor] Initialized with MediaPipe Face Mesh")

    def analyze(self, frame):
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        state_info = {
            "state":      "no_face",
            "ear":        None,
            "perclos":    None,
            "yaw":        None,
            "pitch":      None,
            "alert":      None
        }

        if not result.multi_face_landmarks:
            return frame, state_info

        landmarks = result.multi_face_landmarks[0].landmark

        # ── EAR ──────────────────────────────────────────
        left_ear  = calculate_ear(landmarks, LEFT_EYE,  w, h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
        ear       = (left_ear + right_ear) / 2.0

        # ── PERCLOS ──────────────────────────────────────
        eye_closed = 1 if ear < EAR_THRESHOLD else 0
        self.perclos_buffer.append(eye_closed)
        if len(self.perclos_buffer) > PERCLOS_WINDOW:
            self.perclos_buffer.pop(0)
        perclos = sum(self.perclos_buffer) / len(self.perclos_buffer)

        # ── Head Pose ────────────────────────────────────
        pitch, yaw, roll = get_head_pose(landmarks, w, h)

        # ── EAR consecutive frames ───────────────────────
        if ear < EAR_THRESHOLD:
            self.ear_counter += 1
        else:
            self.ear_counter = 0

        # ── Determine Driver State ───────────────────────
        alert_msg = None

        if self.ear_counter >= EAR_CONSEC_FRAMES:
            self.driver_state = "sleeping"
            alert_msg = "ALERT: Driver Sleeping!"
        elif perclos >= PERCLOS_THRESHOLD:
            self.driver_state = "drowsy"
            alert_msg = "WARNING: Driver Drowsy!"
        elif abs(yaw) > YAW_THRESHOLD:
            self.driver_state = "distracted"
            alert_msg = "WARNING: Driver Distracted!"
        else:
            self.driver_state = "alert"

        state_info = {
            "state":   self.driver_state,
            "ear":     round(ear, 3),
            "perclos": round(perclos, 3),
            "yaw":     round(yaw, 2),
            "pitch":   round(pitch, 2),
            "alert":   alert_msg
        }
        # Phone detection using YOLOv8
        yolo_results = self.yolo(frame, conf=0.4, verbose=False)
        for result in yolo_results:
            for box in result.boxes:
                if int(box.cls[0]) == 67:  # COCO class 67 = cell phone
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "PHONE DETECTED",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
                    self.driver_state = "phone_use"
                    state_info["state"] = "phone_use"
                    state_info["alert"] = "ALERT: Phone Use Detected!"

        # ── Draw on Frame ────────────────────────────────
        frame = self._draw(frame, state_info)
        return frame, state_info

    def _draw(self, frame, info):
        state   = info["state"]
        ear     = info["ear"]
        perclos = info["perclos"]
        yaw     = info["yaw"]
        alert   = info["alert"]

        # State color
        color_map = {
            "alert":      (0, 255, 0),
            "drowsy":     (0, 165, 255),
            "sleeping":   (0, 0, 255),
            "distracted": (0, 255, 255),
            "no_face":    (128, 128, 128)
        }
        color = color_map.get(state, (255, 255, 255))

        # HUD
        cv2.putText(frame, f"State   : {state.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"EAR     : {ear}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"PERCLOS : {perclos}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"Yaw     : {yaw} deg",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Alert banner
        if alert:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, alert,
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        return frame


# ─── Quick Test ──────────────────────────────────────────
if __name__ == "__main__":
    monitor = DriverMonitor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        exit()

    print("[INFO] Webcam opened successfully")
    print("[INFO] Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        try:
            frame, info = monitor.analyze(frame)
            print(f"\r{info}", end="")
        except Exception as e:
            print(f"[ERROR] {e}")
            break

        cv2.imshow("Driver Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Done")