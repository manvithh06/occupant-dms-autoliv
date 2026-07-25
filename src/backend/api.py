import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import asyncio
import json
import threading
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from occupant.occupant_detector import OccupantDetector
from driver.driver_monitor import DriverMonitor
from vitals.rppg import RPPGMonitor
from alerts.alert_engine import AlertEngine

# ─── App Setup ───────────────────────────────────────────
app = FastAPI(title="Occupant DMS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ────────────────────────────────────────
latest_data = {
    "driver":   {"state": "unknown"},
    "occupant": {"total_occupied": 0, "detections": []},
    "vitals":   {"bpm": 0, "breathing_rate": 0, "stress": "unknown"},
    "alerts":   [],
    "timestamp": ""
}

camera_running = False
camera_thread  = None

# ─── Module Instances ────────────────────────────────────
occupant_detector = OccupantDetector()
driver_monitor    = DriverMonitor()
rppg_monitor      = RPPGMonitor()
alert_engine      = AlertEngine()

# ─── Camera Loop (runs in background thread) ─────────────
def camera_loop():
    global latest_data, camera_running
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[Camera] Started")

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run all modules
        detections        = occupant_detector.detect(frame)
        seat_map          = occupant_detector.get_seat_map(detections)
        _, driver_info    = driver_monitor.analyze(frame)
        _, vitals_info    = rppg_monitor.analyze(frame)

        # Process alerts
        alerts = []
        alerts += alert_engine.process_driver_state(driver_info)
        alerts += alert_engine.process_vitals(vitals_info)
        alerts += alert_engine.process_occupants(seat_map)

        # Update global state
        latest_data = {
            "driver":    driver_info,
            "occupant":  seat_map,
            "vitals":    vitals_info,
            "alerts":    alerts,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    cap.release()
    print("[Camera] Stopped")

# ─── REST Endpoints ──────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Occupant DMS API running"}

@app.get("/health")
def health():
    return {"status": "ok", "camera": camera_running}

@app.get("/start")
def start_camera():
    global camera_running, camera_thread
    if not camera_running:
        camera_running = True
        camera_thread  = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
        return {"status": "Camera started"}
    return {"status": "Already running"}

@app.get("/stop")
def stop_camera():
    global camera_running
    camera_running = False
    return {"status": "Camera stopped"}

@app.get("/status")
def get_status():
    return latest_data

@app.get("/alerts")
def get_alerts():
    return {
        "active":  alert_engine.get_active_alerts(),
        "history": alert_engine.get_history()
    }

# ─── WebSocket for Live Dashboard ────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    try:
        while True:
            await websocket.send_json(latest_data)
            await asyncio.sleep(0.5)  # send update every 500ms
    except WebSocketDisconnect:
        print("[WS] Client disconnected")