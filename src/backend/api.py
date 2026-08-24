"""
backend/api.py — FastAPI REST + WebSocket server for Occupant DMS.

Start with::

    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /          — health check
    GET  /health    — camera + pipeline status
    GET  /start     — start the camera pipeline
    GET  /stop      — stop the camera pipeline
    GET  /status    — latest detection state
    GET  /alerts    — active + full alert history
    WS   /ws        — live push every 500 ms
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── sys.path hacks removed — install package with:  pip install -e .  ─────────
from config import cfg
from pipeline import DMSPipeline

log = logging.getLogger("DMS.api")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Occupant DMS API",
    description = "Autoliv Driver & Occupant Monitoring System REST + WebSocket API",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Pipeline singleton ────────────────────────────────────────────────────────
_pipeline: DMSPipeline = DMSPipeline()

# ── Thread-safe state access is handled inside DMSPipeline.get_state() ────────


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _on_startup() -> None:
    log.info("API starting up — pipeline ready (not yet running).")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    log.info("API shutting down — stopping pipeline.")
    _pipeline.stop()


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/", summary="Root health check")
def root() -> dict[str, str]:
    return {"status": "Occupant DMS API running"}


@app.get("/health", summary="Pipeline + camera status")
def health() -> dict[str, Any]:
    return {
        "status":  "ok",
        "camera":  _pipeline.running,
        "version": "1.0.0",
    }


@app.get("/start", summary="Start the camera pipeline")
def start_camera() -> dict[str, str]:
    if _pipeline.running:
        return {"status": "already_running"}
    _pipeline.start()
    return {"status": "started"}


@app.get("/stop", summary="Stop the camera pipeline")
def stop_camera() -> dict[str, str]:
    _pipeline.stop()
    return {"status": "stopped"}


@app.get("/status", summary="Latest detection state")
def get_status() -> dict[str, Any]:
    return _pipeline.get_state()


@app.get("/alerts", summary="Active and historical alerts")
def get_alerts() -> dict[str, Any]:
    engine = _pipeline.get_alert_engine()
    return {
        "active":  engine.get_active_alerts(),
        "history": engine.get_history(),
    }


# ── WebSocket — live dashboard feed ──────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info("[WS] Client connected: %s", websocket.client)
    try:
        while True:
            state = _pipeline.get_state()   # thread-safe
            await websocket.send_json(state)
            await asyncio.sleep(0.5)        # push update every 500 ms
    except WebSocketDisconnect:
        log.info("[WS] Client disconnected.")
    except Exception as exc:
        log.warning("[WS] Connection error: %s", exc)