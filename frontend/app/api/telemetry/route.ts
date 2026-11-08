import { NextRequest, NextResponse } from "next/server";

interface SessionState {
  sessionId: string;
  deviceOnline: boolean;
  streamActive: boolean;
  lastHeartbeat: number;
  lastTelemetry: number;
  material?: string;
  scaleArea?: number;
  device: {
    name?: string;
    userAgent?: string;
  };
  diagnostics?: {
    ambientLight: string;
    orientation: string;
    compass: string;
    gps: string;
  };
  sensors?: {
    illuminance_lux: number;
    pitch_deg: number;
    roll_deg: number;
    compass_heading_deg: number;
    acceleration?: { x: number; y: number; z: number };
    gps?: { latitude: number; longitude: number };
  };
  stats?: {
    packetsSent: number;
  };
  calculatedPhysics?: any;
}

// Global in-memory storage for active sessions
const globalSessions = globalThis as unknown as {
  __telemetrySessions: Map<string, SessionState>;
};
if (!globalSessions.__telemetrySessions) {
  globalSessions.__telemetrySessions = new Map();
}
const sessions = globalSessions.__telemetrySessions;

// POST: Mobile node transmits heartbeat or telemetry
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const sessionId = body.sessionId;
    if (!sessionId) {
      return NextResponse.json({ error: "Missing sessionId" }, { status: 400 });
    }

    const action = body.action || (body.stats?.streamActive ? "telemetry" : "heartbeat");
    const existing: SessionState = sessions.get(sessionId) || {
      sessionId,
      deviceOnline: false,
      streamActive: false,
      lastHeartbeat: 0,
      lastTelemetry: 0,
      device: {},
      diagnostics: undefined,
      sensors: undefined,
      stats: undefined,
    };

    let calculatedPhysics = existing.calculatedPhysics;

    // Attach active material and scale configuration if not present
    const payloadToPython = {
      ...body,
      material: body.material || existing.material || "CsPbI3",
      scaleArea: body.scaleArea !== undefined ? body.scaleArea : (existing.scaleArea !== undefined ? existing.scaleArea : 1.0),
    };

    // Forward packet to Python Causal MCRSA backend on port 5000 for server-side evaluation & logging
    try {
      const pyRes = await fetch("http://127.0.0.1:5000/api/telemetry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payloadToPython),
        cache: "no-store",
      });
      if (pyRes.ok) {
        const pyData = await pyRes.json();
        if (pyData.calculatedPhysics) {
          calculatedPhysics = pyData.calculatedPhysics;
        }
      }
    } catch {
      // Python backend may be offline or starting up
    }

    if (action === "set_material") {
      sessions.set(sessionId, {
        ...existing,
        material: body.material || existing.material,
        scaleArea: body.scaleArea !== undefined ? body.scaleArea : existing.scaleArea,
        calculatedPhysics,
      });
      return NextResponse.json({
        ok: true,
        status: "material_configured",
        material: body.material,
        scaleArea: body.scaleArea,
      });
    }

    if (action === "heartbeat") {
      // Device connected handshake
      sessions.set(sessionId, {
        ...existing,
        deviceOnline: true,
        streamActive: false,
        lastHeartbeat: Date.now(),
        device: body.device || existing.device || { name: "Mobile Sensor Probe" },
        diagnostics: body.diagnostics || existing.diagnostics,
        calculatedPhysics,
      });
      return NextResponse.json({ ok: true, status: "handshake_accepted", timestamp: Date.now() });
    }

    if (action === "telemetry") {
      // Live 10 Hz sensor stream
      sessions.set(sessionId, {
        ...existing,
        deviceOnline: true,
        streamActive: true,
        lastHeartbeat: Date.now(),
        lastTelemetry: Date.now(),
        device: body.device || existing.device,
        sensors: body.sensors || existing.sensors,
        stats: body.stats || existing.stats,
        calculatedPhysics,
      });
      return NextResponse.json({
        ok: true,
        status: "telemetry_accepted",
        timestamp: Date.now(),
        calculatedPhysics,
      });
    }

    if (action === "pause_stream") {
      // User tapped Stop Stream, but phone remains connected
      sessions.set(sessionId, {
        ...existing,
        deviceOnline: true,
        streamActive: false,
        lastHeartbeat: Date.now(),
      });
      return NextResponse.json({ ok: true, status: "stream_paused" });
    }

    return NextResponse.json({ ok: true });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Internal error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// GET: Desktop lab queries the latest state for its session
export async function GET(req: NextRequest) {
  const sessionId = req.nextUrl.searchParams.get("sessionId");
  if (!sessionId) {
    return NextResponse.json({ error: "Missing sessionId" }, { status: 400 });
  }

  const s = sessions.get(sessionId);
  if (!s) {
    return NextResponse.json({
      deviceOnline: false,
      streamActive: false,
      message: "No device paired yet",
    });
  }

  // Device is online if heartbeat arrived in the last 6 seconds
  const now = Date.now();
  const isOnline = (now - s.lastHeartbeat) < 6000;
  // Stream is actively transmitting if telemetry arrived in the last 2.5 seconds
  const isStreaming = isOnline && s.streamActive && (now - s.lastTelemetry) < 2500;

  return NextResponse.json({
    deviceOnline: isOnline,
    streamActive: isStreaming,
    device: s.device,
    diagnostics: s.diagnostics,
    sensors: isStreaming ? s.sensors : null,
    stats: s.stats,
    calculatedPhysics: isStreaming ? s.calculatedPhysics : null,
    latencyMs: Math.max(2, Math.min(500, now - (isStreaming ? s.lastTelemetry : s.lastHeartbeat))),
  });
}
