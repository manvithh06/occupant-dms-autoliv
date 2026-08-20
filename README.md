<div align="center">

# 🚗 Occupant Detection & Driver Monitoring System
### Autoliv India — Industry Project | NMAM Institute of Technology, Mangaluru

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)](https://ultralytics.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?style=flat-square)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

An AI-powered automotive safety system for real-time occupant classification, driver behaviour monitoring, and contactless vital signs detection.

</div>

---

## 📌 Project Overview

This system addresses three core automotive safety challenges:

- **Occupant Detection** — Classify all vehicle occupants in real time (adult, child, infant, pet, empty)
- **Driver Monitoring** — Detect drowsiness, distraction, phone use, and microsleep
- **Vital Signs** — Measure heart rate, breathing rate, and stress level contactlessly using rPPG

Built as a **software-first prototype** that runs on a standard laptop webcam, with full hardware integration planned for Phase 2.

---

## 👥 Team

| Name | Role | USN |
|------|------|-----|
| **Manvith Devadiga** | AI/ML Lead | NNM24AM029 |
| **Abhinav D Shetty** | Backend | NNM24AM001 |
| **Adish Shetty** | Data & Frontend | NNM24AM003 |

**Guide:** Mr. Mahesh B L  
**Industry Mentor:** Abhijeet Atwadkar, Autoliv India

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SENSOR LAYER                           │
│         RGB Camera | IR Camera | mmWave Radar (Phase 2)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   AI PROCESSING LAYER                       │
│  Occupant Detection  │  Driver Monitoring  │  Vital Signs   │
│     (YOLOv8n)        │  (MediaPipe + EAR)  │    (rPPG)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     ALERT ENGINE                            │
│       Multi-level alerts | Cooldown | JSON event log        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    BACKEND API                              │
│         FastAPI + WebSocket | REST endpoints                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    DASHBOARD                                │
│       React + Vite | Live driver state | BPM chart          │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Features

### Occupant Detection
- Detects and classifies occupants as **adult, child, infant, pet, or empty**
- Powered by **YOLOv8n** pretrained on COCO dataset
- Infant-alone detection triggers CRITICAL alert
- Real-time bounding boxes with confidence scores

### Driver Monitoring
- **EAR (Eye Aspect Ratio)** — detects eye closure
- **PERCLOS** — percentage eye closure over 90-frame window
- **Head Pose Estimation** — yaw, pitch, roll via solvePnP
- **Phone Detection** — YOLOv8 COCO class 67
- States: `alert | drowsy | sleeping | distracted | phone_use | no_face`

### Vital Signs (rPPG)
- **Heart rate (BPM)** from forehead skin color changes
- **Breathing rate** from slow signal modulation
- **Stress level** from HRV analysis
- Completely contactless — standard webcam only
- Signal quality indicator with 10-second collection window

### Alert Engine
- 9 alert types across driver, occupant, and vitals modules
- **CRITICAL** — sleeping, phone use, infant alone, low heart rate
- **WARNING** — drowsy, distracted, high HR, high stress, abnormal breathing
- 5-second cooldown prevents alert spam
- Full JSON event log with timestamps

### Backend API
- FastAPI + Uvicorn
- WebSocket stream pushing live data every 500ms
- REST endpoints: `/start` `/stop` `/status` `/alerts`
- All modules integrated in a single camera loop

### Live Dashboard
- React + Vite + Recharts
- Real-time driver state with color-coded indicator
- Live BPM trend chart
- Occupant seat map
- Alert log with severity badges
- System status panel

### Background Service
- Runs silently with no terminal window
- Windows toast notifications for all alerts
- Audio beep alerts (CRITICAL = 3 beeps, WARNING = 1 beep)
- Lighting normalization via CLAHE
- State saved to `logs/latest_state.json` every second

### Demo Recorder
- Combined camera view with full HUD overlay
- Records MP4 video with all detections visible
- PERCLOS progress bar, vital signs, alert banners

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Vision / Detection | YOLOv8n (Ultralytics) |
| Face Analysis | MediaPipe Face Mesh (468 landmarks) |
| Vital Signs | rPPG (custom implementation) |
| Signal Processing | SciPy + NumPy (Butterworth filter, FFT) |
| Computer Vision | OpenCV |
| Backend API | FastAPI + Uvicorn |
| WebSocket | FastAPI WebSocket |
| Frontend | React + Vite + Recharts |
| HTTP Client | Axios |
| Notifications | Plyer (Windows toast) |
| Version Control | GitHub (Private repo) |
| Runtime | Python 3.11 + Node.js |

---

## 📁 Project Structure

```
occupant-dms-autoliv/
│
├── src/
│   ├── occupant/
│   │   └── occupant_detector.py     # YOLOv8 occupant detection
│   ├── driver/
│   │   └── driver_monitor.py        # MediaPipe driver monitoring
│   ├── vitals/
│   │   └── rppg.py                  # rPPG vital signs
│   ├── alerts/
│   │   └── alert_engine.py          # Multi-level alert system
│   ├── backend/
│   │   ├── api.py                   # FastAPI + WebSocket
│   │   └── main.py                  # FastAPI entry
│   ├── dashboard/                   # React + Vite frontend
│   ├── main.py                      # Background service runner
│   └── demo_recorder.py             # Demo video recorder
│
├── data/
│   ├── raw/                         # Raw datasets (gitignored)
│   ├── processed/                   # Processed data
│   └── labels/                      # Annotation files
│
├── models/                          # Trained model weights (gitignored)
├── notebooks/                       # EDA and research notebooks
├── docs/                            # Documentation and reports
├── demo/                            # Demo videos and screenshots
├── logs/                            # Runtime logs (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- Webcam

### 1. Clone the repository

```bash
git clone https://github.com/manvithh06/occupant-dms-autoliv.git
cd occupant-dms-autoliv
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```bash
cd src/dashboard
npm install
cd ../..
```

---

## 🚀 Running the Project

### Option 1 — Full System (Backend + Dashboard)

**Terminal 1 — Backend:**
```bash
git checkout backend/abhinav
uvicorn src.backend.api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Dashboard:**
```bash
git checkout frontend/adish
cd src/dashboard
npm run dev
```

Open browser → `http://localhost:5173` → Click **START**

Also open → `http://localhost:8000/start` to start the camera loop.

---

### Option 2 — Background Service (Silent, with notifications)

```bash
git checkout ml/manvith
cd src
python main.py
```

---

### Option 3 — Demo Recorder (Camera view + video recording)

```bash
git checkout ml/manvith
cd src
python demo_recorder.py
```

---

### Option 4 — Run Individual Modules

```bash
# Occupant detection only
python src/occupant/occupant_detector.py

# Driver monitoring only
python src/driver/driver_monitor.py

# rPPG vital signs only
python src/vitals/rppg.py

# Alert engine test
python src/alerts/alert_engine.py
```

---

## 📊 Live Results

| Module | Metric | Observed Result |
|--------|--------|----------------|
| Occupant Detection | Confidence | 87% average on webcam |
| Driver Monitoring | Sleeping detection | Within 20 frames of eye closure |
| Driver Monitoring | Distraction | Detected at >20° head yaw |
| Driver Monitoring | Phone detection | YOLOv8 class 67 working |
| Vital Signs (rPPG) | Heart rate | 78 BPM measured live |
| Vital Signs (rPPG) | Breathing rate | 12 /min measured live |
| Alert Engine | Alert types | 9 alert types firing correctly |
| Backend API | WebSocket | ~500ms live update interval |

---

## 🔔 Alert Types

| Alert | Level | Trigger |
|-------|-------|---------|
| Driver Sleeping | CRITICAL | EAR < 0.25 for ≥ 20 frames |
| Phone Use | CRITICAL | YOLOv8 detects phone |
| Infant Alone | CRITICAL | Infant detected, no adult |
| Low Heart Rate | CRITICAL | BPM < 50 |
| Driver Drowsy | WARNING | PERCLOS > 80% |
| Driver Distracted | WARNING | Head yaw > 20° |
| High Heart Rate | WARNING | BPM > 100 |
| High Stress | WARNING | Stress level = high |
| Abnormal Breathing | WARNING | < 8 or > 25 breaths/min |

---

## 🛣️ Roadmap

- [x] Phase 1 — Research, datasets, project setup
- [x] Phase 2 — Occupant detection (YOLOv8)
- [x] Phase 3 — Driver monitoring (MediaPipe + EAR + PERCLOS)
- [x] Phase 4 — Vital signs (rPPG)
- [x] Phase 5 — Alert engine
- [x] Phase 6 — FastAPI backend + WebSocket
- [x] Phase 7 — React dashboard
- [x] Phase 8 — Background service + Windows notifications
- [x] Phase 9 — Demo recorder
- [ ] Phase 10 — rPPG accuracy validation (UBFC-rPPG dataset)
- [ ] Phase 11 — Hardware integration (RealSense D435i + Jetson)
- [ ] Phase 12 — mmWave radar vital signs (TI IWR6843AOP)
- [ ] Phase 13 — In-vehicle deployment

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ by Team Autoliv — NMAM Institute of Technology, Mangaluru
</div>
