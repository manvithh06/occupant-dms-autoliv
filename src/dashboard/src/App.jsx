import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";
import axios from "axios";

const API = "http://localhost:8000";
const WS  = "ws://localhost:8000/ws";

const STATE_META = {
  alert:      { color: "#4fffb0", bg: "#4fffb010", label: "ALERT" },
  drowsy:     { color: "#ffb347", bg: "#ffb34710", label: "DROWSY" },
  sleeping:   { color: "#ff4560", bg: "#ff456010", label: "SLEEPING" },
  distracted: { color: "#f9e900", bg: "#f9e90010", label: "DISTRACTED" },
  phone_use:  { color: "#ff4560", bg: "#ff456010", label: "PHONE USE" },
  no_face:    { color: "#555",    bg: "#55555510", label: "NO FACE" },
  unknown:    { color: "#555",    bg: "#55555510", label: "STANDBY" },
};

const STRESS_COLOR = {
  low:      "#4fffb0",
  normal:   "#4fffb0",
  elevated: "#ffb347",
  high:     "#ff4560",
  unknown:  "#555",
};

function Ring({ value, max, color, size = 80, stroke = 7 }) {
  const r   = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const pct  = Math.min(value / max, 1);
  const dash = pct * circ;
  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#1a1a2e" strokeWidth={stroke} />
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color}
        strokeWidth={stroke} strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round" style={{ transition: "stroke-dasharray 0.5s ease" }} />
    </svg>
  );
}

function Metric({ label, value, unit, color = "#aaa" }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 10, color: "#444", letterSpacing: 2, textTransform: "uppercase", marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>
        {value ?? "—"}<span style={{ fontSize: 12, color: "#555", marginLeft: 4 }}>{unit}</span>
      </div>
    </div>
  );
}

function AlertBadge({ alert }) {
  const isC = alert.level === "CRITICAL";
  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: 10,
      padding: "10px 12px", marginBottom: 8, borderRadius: 6,
      background: isC ? "#ff456008" : "#ffb34708",
      borderLeft: `3px solid ${isC ? "#ff4560" : "#ffb347"}`,
    }}>
      <div style={{ paddingTop: 1 }}>
        <div style={{ width: 7, height: 7, borderRadius: "50%", background: isC ? "#ff4560" : "#ffb347", marginTop: 2 }} />
      </div>
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, color: isC ? "#ff4560" : "#ffb347", letterSpacing: 1 }}>{alert.level}</div>
        <div style={{ fontSize: 12, color: "#ccc", marginTop: 2 }}>{alert.message}</div>
        <div style={{ fontSize: 10, color: "#333", marginTop: 3 }}>{alert.timestamp}</div>
      </div>
    </div>
  );
}

export default function App() {
  const [data,       setData]       = useState(null);
  const [alerts,     setAlerts]     = useState([]);
  const [bpmHistory, setBpmHistory] = useState([]);
  const [camRunning, setCamRunning] = useState(false);
  const [connected,  setConnected]  = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS);
      wsRef.current = ws;
      ws.onopen    = () => setConnected(true);
      ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        setData(d);
        if (d.vitals?.bpm > 0) {
          setBpmHistory(prev =>
            [...prev, { t: new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" }), bpm: d.vitals.bpm }].slice(-30)
          );
        }
        if (d.alerts?.length > 0) {
          setAlerts(prev => [...d.alerts, ...prev].slice(0, 15));
        }
      };
      ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  const startCamera = async () => { await axios.get(`${API}/start`); setCamRunning(true); };
  const stopCamera  = async () => { await axios.get(`${API}/stop`);  setCamRunning(false); };

  const state    = data?.driver?.state || "unknown";
  const meta     = STATE_META[state] || STATE_META.unknown;
  const perclos  = data?.driver?.perclos ?? 0;
  const bpm      = data?.vitals?.bpm ?? 0;
  const stress   = data?.vitals?.stress ?? "unknown";
  const occupied = data?.occupant?.total_occupied ?? 0;

  return (
    <div style={{
      background: "#07070f",
      minHeight: "100vh",
      color: "#fff",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
      padding: "24px 28px",
      boxSizing: "border-box",
    }}>

      {/* ── Top Bar ── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 3, height: 22, background: "#4fffb0", borderRadius: 2 }} />
            <span style={{ fontSize: 16, fontWeight: 700, letterSpacing: 2, color: "#fff" }}>OCCUPANT DMS</span>
            <span style={{ fontSize: 10, color: "#555", letterSpacing: 3, marginLeft: 4 }}>AUTOLIV</span>
          </div>
          <div style={{ fontSize: 10, color: "#333", letterSpacing: 2, marginTop: 4, marginLeft: 13 }}>
            DRIVER MONITORING & OCCUPANT DETECTION SYSTEM
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: connected ? "#4fffb0" : "#ff4560",
              boxShadow: connected ? "0 0 6px #4fffb0" : "0 0 6px #ff4560",
            }} />
            <span style={{ fontSize: 10, color: "#444", letterSpacing: 2 }}>
              {connected ? "LIVE" : "OFFLINE"}
            </span>
          </div>
          <button onClick={camRunning ? stopCamera : startCamera} style={{
            background: "transparent",
            border: `1px solid ${camRunning ? "#ff4560" : "#4fffb0"}`,
            borderRadius: 4,
            padding: "7px 18px",
            color: camRunning ? "#ff4560" : "#4fffb0",
            cursor: "pointer",
            fontFamily: "inherit",
            fontSize: 11,
            letterSpacing: 2,
            transition: "all 0.2s",
          }}>
            {camRunning ? "STOP" : "START"}
          </button>
        </div>
      </div>

      {/* ── Driver State Banner ── */}
      <div style={{
        background: meta.bg,
        border: `1px solid ${meta.color}30`,
        borderLeft: `4px solid ${meta.color}`,
        borderRadius: 8,
        padding: "16px 24px",
        marginBottom: 20,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div>
            <div style={{ fontSize: 10, color: "#444", letterSpacing: 3, marginBottom: 4 }}>DRIVER STATE</div>
            <div style={{ fontSize: 32, fontWeight: 800, color: meta.color, letterSpacing: 4 }}>{meta.label}</div>
          </div>
          <div style={{ width: 1, height: 48, background: "#1a1a2e" }} />
          <div style={{ display: "flex", gap: 32 }}>
            {[
              { l: "EAR",     v: data?.driver?.ear     },
              { l: "PERCLOS", v: data?.driver?.perclos },
              { l: "YAW",     v: data?.driver?.yaw,  u: "°" },
              { l: "PITCH",   v: data?.driver?.pitch, u: "°" },
            ].map(({ l, v, u }) => (
              <div key={l}>
                <div style={{ fontSize: 9, color: "#444", letterSpacing: 2 }}>{l}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: "#ccc", marginTop: 2 }}>
                  {v ?? "—"}{u ?? ""}
                </div>
              </div>
            ))}
          </div>
        </div>
        {/* PERCLOS Ring */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
          <div style={{ position: "relative", width: 72, height: 72 }}>
            <Ring value={perclos} max={1} color={perclos > 0.8 ? "#ff4560" : perclos > 0.5 ? "#ffb347" : "#4fffb0"} size={72} stroke={6} />
            <div style={{
              position: "absolute", inset: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 14, fontWeight: 700,
              color: perclos > 0.8 ? "#ff4560" : "#ccc",
            }}>
              {Math.round(perclos * 100)}%
            </div>
          </div>
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 2 }}>PERCLOS</div>
        </div>
      </div>

      {/* ── Middle Row ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 16 }}>

        {/* Vital Signs */}
        <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: 8, padding: 20 }}>
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 3, marginBottom: 16 }}>VITAL SIGNS</div>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16 }}>
            <div style={{ position: "relative", width: 80, height: 80 }}>
              <Ring value={bpm} max={160} color={bpm > 100 ? "#ff4560" : "#00c8ff"} size={80} stroke={7} />
              <div style={{
                position: "absolute", inset: 0,
                display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
              }}>
                <div style={{ fontSize: 16, fontWeight: 800, color: "#00c8ff" }}>{bpm || "—"}</div>
                <div style={{ fontSize: 8, color: "#444" }}>BPM</div>
              </div>
            </div>
            <div>
              <Metric label="Breathing" value={data?.vitals?.breathing_rate} unit="/min" color="#aaa" />
              <Metric label="Stress" value={stress?.toUpperCase()} color={STRESS_COLOR[stress]} />
            </div>
          </div>
          <div style={{
            fontSize: 10, color: "#444", letterSpacing: 2,
            borderTop: "1px solid #1a1a2e", paddingTop: 10,
          }}>
            SIGNAL &nbsp;
            <span style={{ color: data?.vitals?.signal_quality === "good" ? "#4fffb0" : "#555" }}>
              {data?.vitals?.signal_quality?.toUpperCase() ?? "WAITING"}
            </span>
          </div>
        </div>

        {/* Occupants */}
        <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: 8, padding: 20 }}>
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 3, marginBottom: 16 }}>OCCUPANTS</div>
          <div style={{ fontSize: 42, fontWeight: 800, color: "#ffb347", marginBottom: 8 }}>
            {occupied}
            <span style={{ fontSize: 12, color: "#444", marginLeft: 8, fontWeight: 400 }}>IN VEHICLE</span>
          </div>
          <div style={{ borderTop: "1px solid #1a1a2e", paddingTop: 12 }}>
            {data?.occupant?.detections?.length > 0
              ? data.occupant.detections.map((d, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: "#555", letterSpacing: 2 }}>SEAT {i + 1}</span>
                  <span style={{ fontSize: 11, color: "#ccc" }}>
                    {d.label.toUpperCase()}
                    <span style={{ color: "#444", marginLeft: 8 }}>{Math.round(d.confidence * 100)}%</span>
                  </span>
                </div>
              ))
              : <div style={{ fontSize: 11, color: "#333", letterSpacing: 2 }}>NO OCCUPANTS DETECTED</div>
            }
          </div>
        </div>

        {/* System Status */}
        <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: 8, padding: 20 }}>
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 3, marginBottom: 16 }}>SYSTEM STATUS</div>
          {[
            { label: "OCCUPANT DETECTION", ok: !!data },
            { label: "DRIVER MONITORING",  ok: !!data?.driver },
            { label: "VITAL SIGNS (rPPG)", ok: data?.vitals?.signal_quality === "good" },
            { label: "ALERT ENGINE",       ok: !!data },
            { label: "WEBSOCKET STREAM",   ok: connected },
            { label: "CAMERA FEED",        ok: camRunning },
          ].map(({ label, ok }) => (
            <div key={label} style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
              <span style={{ fontSize: 10, color: "#444", letterSpacing: 1 }}>{label}</span>
              <span style={{ fontSize: 10, color: ok ? "#4fffb0" : "#333" }}>{ok ? "ONLINE" : "OFFLINE"}</span>
            </div>
          ))}
          <div style={{ borderTop: "1px solid #1a1a2e", paddingTop: 10, marginTop: 4 }}>
            <div style={{ fontSize: 9, color: "#333", letterSpacing: 1 }}>
              {data?.timestamp ?? "AWAITING DATA"}
            </div>
          </div>
        </div>
      </div>

      {/* ── Bottom Row ── */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>

        {/* BPM Chart */}
        <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: 8, padding: 20 }}>
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 3, marginBottom: 16 }}>HEART RATE TREND</div>
          {bpmHistory.length > 2 ? (
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={bpmHistory} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#111" />
                <XAxis dataKey="t" tick={{ fontSize: 9, fill: "#333" }} interval="preserveStartEnd" />
                <YAxis domain={[40, 160]} tick={{ fontSize: 9, fill: "#333" }} />
                <Tooltip
                  contentStyle={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: 4, fontSize: 11 }}
                  labelStyle={{ color: "#555" }}
                  itemStyle={{ color: "#00c8ff" }}
                />
                <ReferenceLine y={100} stroke="#ff456040" strokeDasharray="4 4" />
                <ReferenceLine y={60}  stroke="#ffb34740" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="bpm" stroke="#00c8ff" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 11, color: "#333", letterSpacing: 2 }}>COLLECTING SIGNAL</div>
                <div style={{ fontSize: 10, color: "#222", marginTop: 6 }}>face the camera to begin</div>
              </div>
            </div>
          )}
        </div>

        {/* Alert Log */}
        <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: 8, padding: 20 }}>
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 3, marginBottom: 16 }}>ALERT LOG</div>
          <div style={{ maxHeight: 220, overflowY: "auto" }}>
            {alerts.length === 0
              ? <div style={{ fontSize: 11, color: "#222", letterSpacing: 2 }}>NO ACTIVE ALERTS</div>
              : alerts.map((a, i) => <AlertBadge key={i} alert={a} />)
            }
          </div>
        </div>
      </div>
    </div>
  );
}