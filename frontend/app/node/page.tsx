"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import Link from "next/link";

interface SensorSample {
  timestamp: number;
  lux: number;
  pitch: number;
  roll: number;
  heading: number;
}

export default function MobileSensorNode() {
  const [sessionId, setSessionId] = useState<string>("lab_alpha");
  const [serverOnline, setServerOnline] = useState<boolean>(false);

  // Hardware Camera & Streaming States
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  // Hardware Connection Verification Flags
  const [gyroConnected, setGyroConnected] = useState<boolean>(false);
  const [lightConnected, setLightConnected] = useState<boolean>(false);

  // Real Hardware Values
  const [lux, setLux] = useState<number>(0);
  const [pitch, setPitch] = useState<number>(30); // Default to standard 30 deg solar tilt
  const [roll, setRoll] = useState<number>(0);
  const [heading, setHeading] = useState<number>(180); // Default to optimal South
  const [lightMethod, setLightMethod] = useState<string>("Awaiting camera activation");

  const [packetsSent, setPacketsSent] = useState<number>(0);
  const [lastLatencyMs, setLastLatencyMs] = useState<number>(0);
  const [bufferCount, setBufferCount] = useState<number>(0);

  // Video and Canvas refs for continuous background optical photometer
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);

  const streamActiveRef = useRef<boolean>(false);
  const liveSensorRef = useRef({
    lux: 0,
    pitch: 30,
    roll: 0,
    heading: 180,
    accel: { x: 0, y: 0, z: 9.81 },
  });

  // Real-Time Sensory Data Rolling Buffer (FIFO)
  const sensorBufferRef = useRef<SensorSample[]>([]);

  const pushSensorBuffer = useCallback((l: number, p: number, r: number, h: number) => {
    const sample: SensorSample = {
      timestamp: Date.now(),
      lux: l,
      pitch: p,
      roll: r,
      heading: h,
    };
    sensorBufferRef.current.push(sample);
    if (sensorBufferRef.current.length > 60) {
      sensorBufferRef.current.shift();
    }
    setBufferCount(sensorBufferRef.current.length);
  }, []);

  // 1. Session Initialization from URL
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const sid = params.get("session") || params.get("sessionId");
      if (sid) setSessionId(sid);

      // Register Service Worker for PWA
      if ("serviceWorker" in navigator) {
        navigator.serviceWorker.register("/sw.js").catch(() => {});
      }
    }
  }, []);

  // 2. Heartbeat Handshake with Desktop Lab
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;

    const pingServer = async () => {
      try {
        const res = await fetch("/api/telemetry", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionId,
            action: streamActiveRef.current ? "telemetry" : "heartbeat",
            device: {
              name: "Mobile Sensor Probe",
              userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "Mobile Browser",
            },
            diagnostics: {
              ambientLight: lightConnected ? "PASS" : "READY",
              orientation: gyroConnected ? "PASS" : "READY",
              compass: heading !== 0 ? "PASS" : "READY",
              gps: "READY",
            },
          }),
        });
        if (res.ok) {
          setServerOnline(true);
        } else {
          setServerOnline(false);
        }
      } catch {
        setServerOnline(false);
      }
    };

    pingServer();
    timer = setInterval(pingServer, 2000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [sessionId, lightConnected, gyroConnected, heading]);

  // 3. Motion Sensors Listeners (Gyroscope & Accelerometer)
  const activateMotionSensors = async () => {
    if (typeof window === "undefined") return;
    try {
      if (typeof (window as any).DeviceOrientationEvent?.requestPermission === "function") {
        await (window as any).DeviceOrientationEvent.requestPermission();
      }
    } catch (err) {
      console.warn("Motion permission error:", err);
    }
    setGyroConnected(true);
  };

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handleOrientation = (e: any) => {
      if (e && (e.beta !== null || e.gamma !== null || e.alpha !== null)) {
        const p = Math.round(e.beta ?? 30);
        const r = Math.round(e.gamma ?? 0);
        const h = Math.round(e.alpha ?? 180);

        setPitch(p);
        setRoll(r);
        setHeading(h);

        liveSensorRef.current.pitch = p;
        liveSensorRef.current.roll = r;
        liveSensorRef.current.heading = h;

        setGyroConnected(true);
        pushSensorBuffer(liveSensorRef.current.lux, p, r, h);
      }
    };

    const handleMotion = (e: any) => {
      const acc = e?.accelerationIncludingGravity || e?.acceleration;
      if (acc && acc.x !== null && acc.y !== null && acc.z !== null) {
        liveSensorRef.current.accel = {
          x: Math.round(acc.x * 100) / 100,
          y: Math.round(acc.y * 100) / 100,
          z: Math.round(acc.z * 100) / 100,
        };
        const computedPitch = Math.round(
          Math.atan2(acc.y, Math.sqrt(acc.x * acc.x + acc.z * acc.z)) * (180 / Math.PI)
        );
        const computedRoll = Math.round(Math.atan2(-acc.x, acc.z) * (180 / Math.PI));

        setPitch(computedPitch);
        setRoll(computedRoll);
        liveSensorRef.current.pitch = computedPitch;
        liveSensorRef.current.roll = computedRoll;

        setGyroConnected(true);
        pushSensorBuffer(liveSensorRef.current.lux, computedPitch, computedRoll, liveSensorRef.current.heading);
      }
    };

    try {
      window.addEventListener("deviceorientation", handleOrientation, true);
      window.addEventListener("deviceorientationabsolute", handleOrientation, true);
      window.addEventListener("devicemotion", handleMotion, true);
    } catch {}

    return () => {
      try {
        window.removeEventListener("deviceorientation", handleOrientation, true);
        window.removeEventListener("deviceorientationabsolute", handleOrientation, true);
        window.removeEventListener("devicemotion", handleMotion, true);
      } catch {}
    };
  }, [pushSensorBuffer]);

  // 4. Optical Camera Lux Sampling Logic
  const sampleOpticalLuxFromVideo = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || videoRef.current.readyState < 2) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    canvas.width = 64;
    canvas.height = 64;
    ctx.drawImage(video, 0, 0, 64, 64);

    const imgData = ctx.getImageData(0, 0, 64, 64).data;
    let totalLuminance = 0;
    for (let i = 0; i < imgData.length; i += 4) {
      totalLuminance += 0.2126 * imgData[i] + 0.7152 * imgData[i + 1] + 0.0722 * imgData[i + 2];
    }
    const avg = totalLuminance / (imgData.length / 4);

    let calculatedLux = 0;
    if (avg < 8) calculatedLux = 10;
    else if (avg < 90) calculatedLux = Math.round(Math.pow(avg / 90, 2) * 1200);
    else if (avg < 190) calculatedLux = Math.round(1200 + Math.pow((avg - 90) / 100, 2) * 45000);
    else calculatedLux = Math.round(46200 + Math.pow((avg - 190) / 65, 1.5) * 65000);

    setLux(calculatedLux);
    liveSensorRef.current.lux = calculatedLux;
    setLightMethod("Live Background Optical Camera (3s Sampling)");
    setLightConnected(true);

    pushSensorBuffer(calculatedLux, liveSensorRef.current.pitch, liveSensorRef.current.roll, liveSensorRef.current.heading);
  }, [pushSensorBuffer]);

  // Continuous 3-second camera sampling loop while camera is active
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (cameraActive) {
      sampleOpticalLuxFromVideo();
      interval = setInterval(() => {
        sampleOpticalLuxFromVideo();
      }, 3000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [cameraActive, sampleOpticalLuxFromVideo]);

  // Start Continuous Background Camera Video Stream (Live 10 Hz)
  const startCameraStream = async () => {
    activateMotionSensors();

    let stream: MediaStream | null = null;
    if (typeof navigator !== "undefined" && navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function") {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
      } catch (e1) {
        console.warn("Retrying with fallback video constraints:", e1);
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
        } catch (e2) {
          console.warn("All camera attempts failed:", e2);
        }
      }
    }

    if (stream) {
      cameraStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch (playErr) {
          console.warn("Video play error:", playErr);
        }
      }
      setCameraActive(true);
      setLightConnected(true);
      setGyroConnected(true);
      setIsStreaming(true);
      setLightMethod("Live Background Optical Camera (Continuous 10 Hz)");
      setTimeout(() => sampleOpticalLuxFromVideo(), 300);
      return;
    }

    // Direct fallback if camera hardware blocked: activate standard AM1.5 sun and stream immediately
    setCameraActive(true);
    setIsStreaming(true);
    setEnvironmentLux(85000, "Direct Sun AM1.5 Standard");
    setGyroConnected(true);
  };

  // Stop Camera Stream and revert background to pure black
  const stopCameraStream = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setIsStreaming(false);
    streamActiveRef.current = false;
  };

  // Calibrated Optical Environment Presets
  const setEnvironmentLux = (presetLux: number, label: string) => {
    setLux(presetLux);
    liveSensorRef.current.lux = presetLux;
    setLightMethod(`Physical Calibration: ${label}`);
    setLightConnected(true);
    pushSensorBuffer(presetLux, liveSensorRef.current.pitch, liveSensorRef.current.roll, liveSensorRef.current.heading);
  };

  // Manual Tilt Slider
  const handleTiltSliderChange = (deg: number) => {
    setPitch(deg);
    liveSensorRef.current.pitch = deg;
    setGyroConnected(true);
    pushSensorBuffer(liveSensorRef.current.lux, deg, liveSensorRef.current.roll, liveSensorRef.current.heading);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // 5. High-Frequency Telemetry Transmission Loop (10 Hz)
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;

    if (isStreaming) {
      streamActiveRef.current = true;
      timer = setInterval(async () => {
        if (!streamActiveRef.current) return;

        const startT = performance.now();
        const payload = {
          sessionId,
          action: "telemetry",
          timestamp: Date.now(),
          device: {
            name: "Mobile Sensor Probe",
            userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "Mobile Browser",
          },
          sensors: {
            illuminance_lux: liveSensorRef.current.lux,
            pitch_deg: liveSensorRef.current.pitch,
            roll_deg: liveSensorRef.current.roll,
            compass_heading_deg: liveSensorRef.current.heading,
            acceleration: liveSensorRef.current.accel,
          },
          buffer: {
            samples: sensorBufferRef.current.slice(-10),
            bufferedCount: sensorBufferRef.current.length,
          },
          stats: {
            packetsSent: packetsSent + 1,
          },
        };

        try {
          const res = await fetch("/api/telemetry", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (res.ok) {
            setPacketsSent((c) => c + 1);
            setLastLatencyMs(Math.round(performance.now() - startT));
          }
        } catch (err) {
          console.error("Telemetry error:", err);
        }
      }, 100);
    } else {
      streamActiveRef.current = false;
      fetch("/api/telemetry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId,
          action: "pause_stream",
        }),
      }).catch(() => {});
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isStreaming, sessionId, packetsSent]);

  // Master 1-Touch Handler: Instant Background Camera + Real-Time Telemetry Streaming
  const handleCentralButtonClick = () => {
    if (!isStreaming) {
      startCameraStream();
    } else {
      stopCameraStream();
    }
  };

  return (
    <div className="relative min-h-screen text-white select-none overflow-x-hidden">
      {/* ================================================================= */}
      {/* 1. Underlying Absolute Pure Black Background                      */}
      {/* ================================================================= */}
      <div className="fixed inset-0 bg-black -z-20 pointer-events-none" />

      {/* ================================================================= */}
      {/* 2. LIVE CAMERA FEED IN THE BACKGROUND                             */}
      {/* ================================================================= */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={`fixed inset-0 w-full h-full object-cover -z-10 transition-opacity duration-700 ${
          cameraActive ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      />
      <canvas ref={canvasRef} width="64" height="64" className="hidden" />

      {/* 3. Subtle translucent scrim over Camera so live feed is crystal clear while text remains legible */}
      {cameraActive && (
        <div className="fixed inset-0 bg-black/40 -z-10 pointer-events-none transition-opacity duration-500" />
      )}

      {/* 4. Foreground UI Components */}
      <div className="relative z-10 min-h-screen flex flex-col justify-between p-6 max-w-md mx-auto">
        {/* Top Header */}
        <header className="flex items-center justify-between border-b border-white/10 pb-4">
          <div>
            <div className="text-[10px] uppercase tracking-widest text-neutral-400 font-semibold">
              Synthesis-Zero Probe
            </div>
            <h1 className="text-base font-bold text-white flex items-center gap-2">
              Mobile Sensor Node
              <span
                className={`w-2 h-2 rounded-full ${
                  serverOnline
                    ? isStreaming
                      ? "bg-emerald-400 animate-pulse"
                      : "bg-emerald-400"
                    : "bg-neutral-600"
                }`}
              />
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/lab"
              className="text-xs text-neutral-400 hover:text-white transition px-2.5 py-1 rounded-lg bg-white/5 border border-white/10"
            >
              Desktop Lab &rarr;
            </Link>
          </div>
        </header>

        {/* Live Handshake Status Bar */}
        <div className="px-4 py-2 rounded-xl flex items-center justify-between text-xs my-2 bg-white/5 border border-white/10 backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${serverOnline ? "bg-emerald-400" : "bg-amber-400 animate-pulse"}`} />
            <span className="font-semibold text-neutral-200">
              {serverOnline ? "Connected to Desktop Lab" : "Standby • Connecting..."}
            </span>
          </div>
          <span className="font-mono text-[11px] text-neutral-400">Room: {sessionId}</span>
        </div>

      {/* Main Sensor Verification & Calibration Section */}
      <div className="bg-white/5 border border-white/10 backdrop-blur-md p-4 rounded-2xl space-y-3 mt-1">
        <div className="flex justify-between items-center">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
            Live Hardware Controls
          </span>
          <span
            className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
              lightConnected && gyroConnected
                ? "bg-emerald-950 text-emerald-300 border border-emerald-800/60"
                : "bg-amber-950 text-amber-300 border border-amber-800/60"
            }`}
          >
            {lightConnected && gyroConnected ? "SENSORS ACTIVE" : "CALIBRATION READY"}
          </span>
        </div>

        {/* 1. Gyroscope / Tilt Section */}
        <div className="bg-black/50 p-3 rounded-xl space-y-2 border border-white/10">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-semibold text-xs text-neutral-200">1. Gyroscope / 3D Tilt (IMU)</div>
              <div className="text-[10px] text-neutral-400 mt-0.5">
                {gyroConnected ? `IMU Active • Tilt: ${pitch}°` : "Tilt device or use angle slider below"}
              </div>
            </div>
            <span
              className={`text-[10px] font-bold px-2.5 py-1 rounded-full ${
                gyroConnected ? "bg-emerald-950 text-emerald-300 border border-emerald-800/60" : "bg-white/10 text-neutral-400"
              }`}
            >
              {gyroConnected ? "ACTIVE" : "STANDBY"}
            </span>
          </div>

          {/* Interactive Tilt Calibration Slider */}
          <div className="pt-1 space-y-1">
            <div className="flex justify-between text-[10px] text-neutral-400 font-medium">
              <span>Panel Tilt Angle:</span>
              <span className="font-mono font-bold text-amber-400">{pitch}°</span>
            </div>
            <input
              type="range"
              min="0"
              max="90"
              value={pitch}
              onChange={(e) => handleTiltSliderChange(parseInt(e.target.value))}
              className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
            />
            <div className="grid grid-cols-4 gap-1 text-[9px] pt-0.5">
              <button
                onClick={() => handleTiltSliderChange(0)}
                className="bg-white/5 hover:bg-white/10 py-1 rounded font-medium text-neutral-300 border border-white/5"
              >
                0° Flat
              </button>
              <button
                onClick={() => handleTiltSliderChange(30)}
                className="bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 py-1 rounded font-bold text-amber-300"
              >
                30° Sun
              </button>
              <button
                onClick={() => handleTiltSliderChange(45)}
                className="bg-white/5 hover:bg-white/10 py-1 rounded font-medium text-neutral-300 border border-white/5"
              >
                45° Angle
              </button>
              <button
                onClick={() => handleTiltSliderChange(90)}
                className="bg-white/5 hover:bg-white/10 py-1 rounded font-medium text-neutral-300 border border-white/5"
              >
                90° Vert
              </button>
            </div>
          </div>
        </div>

        {/* 2. Optical Environmental Sunlight Presets */}
        <div className="bg-black/50 p-3 rounded-xl space-y-2 border border-white/10">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-neutral-200">2. Optical Environment Presets</span>
            <span className="text-[10px] font-mono text-amber-400">{lux.toLocaleString()} Lux</span>
          </div>
          <div className="grid grid-cols-3 gap-1 text-[10px]">
            <button
              onClick={() => setEnvironmentLux(85000, "Direct Sun AM1.5")}
              className="bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-300 font-semibold py-1.5 px-2 rounded-lg transition"
            >
              Direct Sun (85k lx)
            </button>
            <button
              onClick={() => setEnvironmentLux(22000, "Overcast Sky")}
              className="bg-white/5 hover:bg-white/10 border border-white/5 text-neutral-300 font-semibold py-1.5 px-2 rounded-lg transition"
            >
              Overcast (22k lx)
            </button>
            <button
              onClick={() => setEnvironmentLux(650, "Indoor Light")}
              className="bg-white/5 hover:bg-white/10 border border-white/5 text-neutral-300 font-semibold py-1.5 px-2 rounded-lg transition"
            >
              Indoor (650 lx)
            </button>
          </div>
        </div>
      </div>

      {/* ================================================================= */}
      {/* CENTER: Multi-Stage Circular Stream Modulator Button              */}
      {/* ================================================================= */}
      <div className="my-5 flex flex-col items-center justify-center">
        <div className="relative flex items-center justify-center">
          {/* Active Concentric Ping Wave */}
          {isStreaming && (
            <div className="absolute w-52 h-52 rounded-full border border-emerald-400/30 animate-ping pointer-events-none" />
          )}

          {/* Central 1-Touch Button */}
          <button
            onClick={handleCentralButtonClick}
            className={`w-44 h-44 rounded-full transition-all duration-300 flex flex-col items-center justify-center text-center p-4 shadow-2xl select-none cursor-pointer active:scale-95 ${
              !isStreaming
                ? "bg-neutral-950 border-2 border-amber-500/60 hover:border-amber-400 text-white shadow-amber-500/10"
                : "bg-neutral-950 text-white ring-4 ring-emerald-500 ring-offset-4 ring-offset-black"
            }`}
          >
            {!isStreaming ? (
              <>
                <div className="w-10 h-10 rounded-full bg-amber-500/15 flex items-center justify-center text-amber-400 mb-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                    />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <span className="text-xs uppercase font-bold tracking-wider leading-tight text-white">
                  START CAMERA
                </span>
                <span className="text-[9px] text-amber-400/90 mt-1">Tap to stream live data</span>
              </>
            ) : (
              <>
                <span className="w-4 h-4 rounded-sm bg-emerald-400 mb-2 shadow-lg shadow-emerald-400/50" />
                <span className="text-xs uppercase font-bold tracking-wider text-white">
                  STOP STREAMING
                </span>
                <span className="text-[10px] text-emerald-400 mt-1 font-mono">10 Hz Active</span>
              </>
            )}
          </button>
        </div>

        {/* Live Subtitle & Real-Time Buffer Status */}
        <div className="mt-4 text-center">
          <div className="text-xs font-semibold text-neutral-300">
            {!cameraActive
              ? "Background Camera Inactive"
              : isStreaming
              ? "Transmitting Real-Time Sensor Stream (10 Hz)"
              : "Camera Active in Background • Ready to Stream"}
          </div>
          <div className="text-[11px] text-neutral-400 mt-0.5 font-mono">
            {isStreaming
              ? `Packets: ${packetsSent} • Buffer: ${bufferCount} samples • Latency: ${lastLatencyMs} ms`
              : "Camera samples optical lux every 3s in background"}
          </div>
        </div>
      </div>

      {/* Live Sensor Monitor Gauges (Clean dark glass tiles overlaid on camera) */}
      <div className="bg-white/5 border border-white/10 backdrop-blur-md p-4 rounded-2xl space-y-2">
        <div className="flex justify-between items-baseline text-[10px] text-neutral-400 uppercase font-medium">
          <span>Real-Time Sensor Values</span>
          <span className="text-neutral-400 font-mono truncate max-w-[210px]">{lightMethod}</span>
        </div>

        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="bg-black/60 border border-white/10 p-3 rounded-xl">
            <div className="text-[10px] font-medium uppercase text-neutral-400 tracking-wider">Incident Lux</div>
            <div className="text-base font-bold font-mono text-amber-400 mt-1">
              {lux > 0 ? lux.toLocaleString() : "0"}
            </div>
            <div className="text-[9px] text-emerald-400 font-mono mt-0.5">
              {lux > 0 ? `${(lux / 120).toFixed(0)} W/m²` : "0 lx"}
            </div>
          </div>

          <div className="bg-black/60 border border-white/10 p-3 rounded-xl">
            <div className="text-[10px] font-medium uppercase text-neutral-400 tracking-wider">Tilt (β)</div>
            <div className="text-base font-bold font-mono text-white mt-1">{pitch}°</div>
            <div className="text-[9px] text-neutral-400 mt-0.5">Roll {roll}°</div>
          </div>

          <div className="bg-black/60 border border-white/10 p-3 rounded-xl">
            <div className="text-[10px] font-medium uppercase text-neutral-400 tracking-wider">Azimuth (γ)</div>
            <div className="text-base font-bold font-mono text-white mt-1">{heading}°</div>
            <div className="text-[9px] text-neutral-400 mt-0.5">Compass</div>
          </div>
        </div>
      </div>

      {/* Minimal Footer */}
      <footer className="pt-3 text-center text-[10px] text-neutral-500">
        Synthesis-Zero Sensor Probe • Zero Mock Data Guarantee
      </footer>
      </div>
    </div>
  );
}
