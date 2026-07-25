import { useState, useEffect, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import axios from "axios";

const API = "http://localhost:8000";
const WS  = "ws://localhost:8000/ws";

const STATE_COLORS = {
  alert:      "#00ff88",
  drowsy:     "#ffa500",
  sleeping:   "#ff4444",
  distracted: "#ffff00",
  phone_use:  "#ff4444",
  no_face:    "#888888",
  unknown:    "#888888"
};

export default function App() {
  const [data, setData]         = useState(null);
  const [alerts, setAlerts]     = useState([]);
  const [bpmHistory, setBpmHistory] = useState([]);
  const [camRunning, setCamRunning] = useState(false);
  const [connected, setConnected]   = useState(false);
  const wsRef = useRef(null);

  // WebSocket connection
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        console.log("WS connected");
      };

      ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        setData(d);

        // Update BPM history
        if (d.vitals?.bpm > 0) {
          setBpmHistory(prev => {
            const updated = [...prev, {
              time: new Date().toLocaleTimeString(),
              bpm:  d.vitals.bpm
            }];
            return updated.slice(-20); // keep last 20 points
          });
        }

        // Update alerts
        if (d.alerts?.length > 0) {
          setAlerts(prev => [...d.alerts, ...prev].slice(0, 10));
        }
      };

      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 2000); // reconnect
      };
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  const startCamera = async () => {
    await axios.get(`${API}/start`);
    setCamRunning(true);
  };

  const stopCamera = async () => {
    await axios.get(`${API}/stop`);
    setCamRunning(false);
  };

  const driverState = data?.driver?.state || "unknown";
  const stateColor  = STATE_COLORS[driverState] || "#888";

  return (
    <div style={{
      background: "#0a0a0f",
      minHeight:  "100vh",
      color:      "#fff",
      fontFamily: "monospace",
      padding:    "20px"
    }}>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "22px", color: "#00ccff" }}>
            🚗 Occupant DMS — Autoliv
          </h1>
          <p style={{ margin: 0, fontSize: "12px", color: "#555" }}>
            Driver Monitoring & Occupant Detection System
          </p>
        </div>
        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          <span style={{ fontSize: "12px", color: connected ? "#00ff88" : "#ff4444" }}>
            ● {connected ? "WS Connected" : "Disconnected"}
          </span>
          <button onClick={camRunning ? stopCamera : startCamera} style={{
            background:   camRunning ? "#ff4444" : "#00cc66",
            border:       "none",
            borderRadius: "6px",
            padding:      "8px 16px",
            color:        "#fff",
            cursor:       "pointer",
            fontFamily:   "monospace",
            fontSize:     "13px"
          }}>
            {camRunning ? "⏹ Stop Camera" : "▶ Start Camera"}
          </button>
        </div>
      </div>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px", marginBottom: "16px" }}>

        {/* Driver State */}
        <div style={{ background: "#111", borderRadius: "10px", padding: "20px", border: `1px solid ${stateColor}40` }}>
          <p style={{ margin: "0 0 8px", fontSize: "11px", color: "#555", textTransform: "uppercase" }}>Driver State</p>
          <p style={{ margin: "0 0 16px", fontSize: "28px", fontWeight: "bold", color: stateColor }}>
            {driverState.toUpperCase()}
          </p>
          <div style={{ fontSize: "13px", color: "#aaa", lineHeight: "2" }}>
            <div>EAR &nbsp;&nbsp;&nbsp;: <span style={{ color: "#fff" }}>{data?.driver?.ear ?? "—"}</span></div>
            <div>PERCLOS: <span style={{ color: "#fff" }}>{data?.driver?.perclos ?? "—"}</span></div>
            <div>Yaw &nbsp;&nbsp;&nbsp;: <span style={{ color: "#fff" }}>{data?.driver?.yaw ?? "—"}°</span></div>
            <div>Pitch &nbsp;&nbsp;: <span style={{ color: "#fff" }}>{data?.driver?.pitch ?? "—"}°</span></div>
          </div>
        </div>

        {/* Vital Signs */}
        <div style={{ background: "#111", borderRadius: "10px", padding: "20px", border: "1px solid #00ccff40" }}>
          <p style={{ margin: "0 0 8px", fontSize: "11px", color: "#555", textTransform: "uppercase" }}>Vital Signs</p>
          <p style={{ margin: "0 0 16px", fontSize: "36px", fontWeight: "bold", color: "#00ccff" }}>
            {data?.vitals?.bpm > 0 ? `${data.vitals.bpm}` : "—"}
            <span style={{ fontSize: "14px", color: "#555", marginLeft: "6px" }}>BPM</span>
          </p>
          <div style={{ fontSize: "13px", color: "#aaa", lineHeight: "2" }}>
            <div>Breathing : <span style={{ color: "#fff" }}>{data?.vitals?.breathing_rate ?? "—"} /min</span></div>
            <div>Stress &nbsp;&nbsp;&nbsp;: <span style={{
              color: data?.vitals?.stress === "high" ? "#ff4444" :
                     data?.vitals?.stress === "elevated" ? "#ffa500" : "#00ff88"
            }}>{data?.vitals?.stress?.toUpperCase() ?? "—"}</span></div>
            <div>Signal &nbsp;&nbsp;&nbsp;: <span style={{ color: "#fff" }}>{data?.vitals?.signal_quality ?? "—"}</span></div>
          </div>
        </div>

        {/* Occupant Info */}
        <div style={{ background: "#111", borderRadius: "10px", padding: "20px", border: "1px solid #ffa50040" }}>
          <p style={{ margin: "0 0 8px", fontSize: "11px", color: "#555", textTransform: "uppercase" }}>Occupants</p>
          <p style={{ margin: "0 0 16px", fontSize: "36px", fontWeight: "bold", color: "#ffa500" }}>
            {data?.occupant?.total_occupied ?? "—"}
            <span style={{ fontSize: "14px", color: "#555", marginLeft: "6px" }}>detected</span>
          </p>
          <div style={{ fontSize: "13px", color: "#aaa", lineHeight: "2" }}>
            {data?.occupant?.detections?.map((d, i) => (
              <div key={i}>
                Seat {i + 1}: <span style={{ color: "#fff" }}>{d.label}</span>
                <span style={{ color: "#555" }}> ({Math.round(d.confidence * 100)}%)</span>
              </div>
            )) ?? <div>No detections</div>}
          </div>
        </div>
      </div>

      {/* BPM Chart + Alerts */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "16px" }}>

        {/* BPM Chart */}
        <div style={{ background: "#111", borderRadius: "10px", padding: "20px", border: "1px solid #00ccff20" }}>
          <p style={{ margin: "0 0 16px", fontSize: "11px", color: "#555", textTransform: "uppercase" }}>Heart Rate History</p>
          {bpmHistory.length > 1 ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={bpmHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#555" }} />
                <YAxis domain={[40, 160]} tick={{ fontSize: 10, fill: "#555" }} />
                <Tooltip contentStyle={{ background: "#111", border: "1px solid #333" }} />
                <Line type="monotone" dataKey="bpm" stroke="#00ccff" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center", color: "#333" }}>
              Collecting signal...
            </div>
          )}
        </div>

        {/* Alert Log */}
        <div style={{ background: "#111", borderRadius: "10px", padding: "20px", border: "1px solid #ff444420" }}>
          <p style={{ margin: "0 0 16px", fontSize: "11px", color: "#555", textTransform: "uppercase" }}>Alert Log</p>
          <div style={{ maxHeight: 200, overflowY: "auto" }}>
            {alerts.length === 0 ? (
              <p style={{ color: "#333", fontSize: "13px" }}>No alerts</p>
            ) : alerts.map((a, i) => (
              <div key={i} style={{
                padding:      "8px",
                marginBottom: "6px",
                borderRadius: "6px",
                background:   a.level === "CRITICAL" ? "#ff444420" : "#ffa50020",
                borderLeft:   `3px solid ${a.level === "CRITICAL" ? "#ff4444" : "#ffa500"}`,
                fontSize:     "12px"
              }}>
                <div style={{ color: a.level === "CRITICAL" ? "#ff4444" : "#ffa500" }}>{a.level}</div>
                <div style={{ color: "#ccc" }}>{a.message}</div>
                <div style={{ color: "#444", fontSize: "11px" }}>{a.timestamp}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Timestamp */}
      <p style={{ textAlign: "right", color: "#333", fontSize: "11px", marginTop: "16px" }}>
        Last update: {data?.timestamp ?? "—"}
      </p>
    </div>
  );
}