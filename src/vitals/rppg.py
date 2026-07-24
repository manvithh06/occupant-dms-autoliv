import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks

# ─── Constants ───────────────────────────────────────────
FPS             = 30
WINDOW_SECONDS  = 10        # analyze last 10 seconds of signal
WINDOW_SIZE     = FPS * WINDOW_SECONDS
LOW_BPM         = 45        # min valid heart rate
HIGH_BPM        = 180       # max valid heart rate
LOW_FREQ        = LOW_BPM  / 60.0
HIGH_FREQ       = HIGH_BPM / 60.0

# Forehead ROI landmarks (MediaPipe)
FOREHEAD_POINTS = [10, 338, 297, 332, 284]

# ─── Butterworth Bandpass Filter ─────────────────────────
def bandpass_filter(signal, low, high, fs):
    nyq = fs / 2.0
    low_cut  = low  / nyq
    high_cut = high / nyq
    b, a = butter(4, [low_cut, high_cut], btype='band')
    return filtfilt(b, a, signal)

# ─── rPPG Class ──────────────────────────────────────────
class RPPGMonitor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Signal buffers
        self.green_buffer = deque(maxlen=WINDOW_SIZE)
        self.time_buffer  = deque(maxlen=WINDOW_SIZE)
        self.frame_count  = 0

        # Results
        self.bpm             = 0
        self.breathing_rate  = 0
        self.stress_level    = "normal"

        print("[RPPGMonitor] Initialized")

    def _get_forehead_roi(self, frame, landmarks):
        """Extract forehead region for signal analysis."""
        h, w = frame.shape[:2]
        pts = []
        for idx in FOREHEAD_POINTS:
            lm = landmarks[idx]
            pts.append((int(lm.x * w), int(lm.y * h)))

        # Bounding box of forehead points
        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        x1 = min(x_coords)
        x2 = max(x_coords)
        y1 = min(y_coords)
        y2 = max(y_coords)

        # Add small padding
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)

    def _calculate_bpm(self):
        """Calculate BPM from green channel signal."""
        if len(self.green_buffer) < WINDOW_SIZE // 2:
            return 0

        signal = np.array(self.green_buffer)

        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

        # Bandpass filter
        try:
            filtered = bandpass_filter(signal, LOW_FREQ, HIGH_FREQ, FPS)
        except Exception:
            return 0

        # FFT to find dominant frequency
        fft_vals = np.abs(np.fft.rfft(filtered))
        fft_freq = np.fft.rfftfreq(len(filtered), d=1.0 / FPS)

        # Only keep frequencies in valid BPM range
        valid = (fft_freq >= LOW_FREQ) & (fft_freq <= HIGH_FREQ)
        if not np.any(valid):
            return 0

        peak_freq = fft_freq[valid][np.argmax(fft_vals[valid])]
        bpm = peak_freq * 60.0
        return round(bpm, 1)

    def _calculate_breathing_rate(self):
        """Estimate breathing rate from slow signal modulation."""
        if len(self.green_buffer) < WINDOW_SIZE // 2:
            return 0

        signal = np.array(self.green_buffer)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

        # Breathing is 0.1-0.5 Hz (6-30 breaths/min)
        try:
            filtered = bandpass_filter(signal, 0.1, 0.5, FPS)
        except Exception:
            return 0

        fft_vals = np.abs(np.fft.rfft(filtered))
        fft_freq = np.fft.rfftfreq(len(filtered), d=1.0 / FPS)

        valid = (fft_freq >= 0.1) & (fft_freq <= 0.5)
        if not np.any(valid):
            return 0

        peak_freq = fft_freq[valid][np.argmax(fft_vals[valid])]
        breaths_per_min = peak_freq * 60.0
        return round(breaths_per_min, 1)

    def _calculate_stress(self, bpm):
        """Simple stress estimation based on BPM."""
        if bpm == 0:
            return "unknown"
        elif bpm < 60:
            return "low"
        elif bpm < 80:
            return "normal"
        elif bpm < 100:
            return "elevated"
        else:
            return "high"

    def analyze(self, frame):
        """
        Main analysis function.
        Returns frame with overlay and vitals dict.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        vitals = {
            "bpm":            self.bpm,
            "breathing_rate": self.breathing_rate,
            "stress":         self.stress_level,
            "signal_quality": "weak"
        }

        if not result.multi_face_landmarks:
            frame = self._draw(frame, vitals, roi_box=None)
            return frame, vitals

        landmarks = result.multi_face_landmarks[0].landmark

        # Get forehead ROI
        roi, roi_box = self._get_forehead_roi(frame, landmarks)

        if roi.size == 0:
            frame = self._draw(frame, vitals, roi_box=None)
            return frame, vitals

        # Extract mean green channel from ROI
        green_mean = np.mean(roi[:, :, 1])  # index 1 = green in BGR
        self.green_buffer.append(green_mean)
        self.frame_count += 1

        # Calculate vitals every 30 frames (every 1 second)
        if self.frame_count % 30 == 0 and len(self.green_buffer) >= WINDOW_SIZE // 2:
            self.bpm            = self._calculate_bpm()
            self.breathing_rate = self._calculate_breathing_rate()
            self.stress_level   = self._calculate_stress(self.bpm)

        vitals = {
            "bpm":            self.bpm,
            "breathing_rate": self.breathing_rate,
            "stress":         self.stress_level,
            "signal_quality": "good" if len(self.green_buffer) >= WINDOW_SIZE // 2 else "collecting"
        }

        frame = self._draw(frame, vitals, roi_box)
        return frame, vitals

    def _draw(self, frame, vitals, roi_box):
        """Draw vitals HUD on frame."""
        bpm     = vitals["bpm"]
        breath  = vitals["breathing_rate"]
        stress  = vitals["stress"]
        quality = vitals["signal_quality"]

        # Draw forehead ROI box
        if roi_box:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(frame, "ROI", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Stress color
        stress_color = {
            "low":      (0, 255, 0),
            "normal":   (0, 255, 0),
            "elevated": (0, 165, 255),
            "high":     (0, 0, 255),
            "unknown":  (128, 128, 128)
        }.get(stress, (255, 255, 255))

        # HUD bottom left
        y_start = frame.shape[0] - 120
        cv2.putText(frame, f"HR      : {bpm} BPM",
                    (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Breathing: {breath} /min",
                    (10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Stress  : {stress.upper()}",
                    (10, y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, stress_color, 2)
        cv2.putText(frame, f"Signal  : {quality}",
                    (10, y_start + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return frame


# ─── Quick Test ──────────────────────────────────────────
if __name__ == "__main__":
    monitor = RPPGMonitor()
    cap = cv2.VideoCapture(0)

    print("[INFO] Collecting signal — stay still and face the camera")
    print("[INFO] BPM will appear after ~5 seconds")
    print("[INFO] Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, vitals = monitor.analyze(frame)
        cv2.imshow("rPPG Vital Signs", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()