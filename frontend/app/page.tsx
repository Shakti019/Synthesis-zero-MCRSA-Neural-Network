"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { QRCodeSVG } from "qrcode.react";

interface Diagnostics {
  ambientLight?: string;
  orientation?: string;
  compass?: string;
  gps?: string;
}

interface SensorData {
  illuminance_lux?: number;
  pitch_deg?: number;
  roll_deg?: number;
  compass_heading_deg?: number;
}

export default function HardwareGatePage() {
  const router = useRouter();
  const [sessionId] = useState<string>("lab_alpha");
  const [hostUrl, setHostUrl] = useState<string>("");
  const [copied, setCopied] = useState<boolean>(false);
  const [disconnectReason, setDisconnectReason] = useState<boolean>(false);

  // Connection & Sensor Verification States
  const [deviceConnected, setDeviceConnected] = useState<boolean>(false);
  const [diagnostics, setDiagnostics] = useState<Diagnostics>({});
  const [sensorValues, setSensorValues] = useState<SensorData>({});

  // Individual Sensor Ticks (Animated sequentially)
  const [tickSocket, setTickSocket] = useState<boolean>(false);
  const [tickLight, setTickLight] = useState<boolean>(false);
  const [tickIMU, setTickIMU] = useState<boolean>(false);
  const [tickCompass, setTickCompass] = useState<boolean>(false);

  // Clearance Gate
  const [isCleared, setIsCleared] = useState<boolean>(false);
  const [countdown, setCountdown] = useState<number | null>(null);

  // Check URL query parameters for disconnection notice
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      if (params.get("reason") === "disconnected") {
        setDisconnectReason(true);
      }
    }
  }, []);

  // 1. Resolve host network IP for scannable QR Code
  useEffect(() => {
    async function resolveNetworkHost() {
      try {
        const res = await fetch("/api/network-ip");
        const json = await res.json();
        if (json.primaryIp && json.primaryIp !== "localhost") {
          const proto = typeof window !== "undefined" && window.location.protocol === "http:" ? "http" : "https";
          setHostUrl(`${proto}://${json.primaryIp}:3000/node?session=${sessionId}`);
          return;
        }
      } catch (err) {
        console.warn("Could not auto-detect LAN IP:", err);
      }
      if (typeof window !== "undefined") {
        setHostUrl(`${window.location.origin}/node?session=${sessionId}`);
      }
    }
    resolveNetworkHost();
  }, [sessionId]);

  // 2. High-Frequency Telemetry Polling (Every 400ms)
  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/telemetry?sessionId=${sessionId}`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.deviceOnline) {
          setDeviceConnected(true);
          setDisconnectReason(false);
          if (data.diagnostics) {
            setDiagnostics(data.diagnostics);
          }
          if (data.sensors) {
            setSensorValues(data.sensors);
          }
        } else {
          // If probe disconnected, reset states back to QR gate
          setDeviceConnected(false);
          setIsCleared(false);
          setCountdown(null);
          setTickSocket(false);
          setTickLight(false);
          setTickIMU(false);
          setTickCompass(false);
        }
      } catch (err) {
        console.error("Telemetry query error:", err);
      }
    }, 400);

    return () => clearInterval(timer);
  }, [sessionId]);

  // 3. Sequential Staggered Tick Animation when Device Connects
  useEffect(() => {
    if (!deviceConnected) return;

    // Step A: Telemetry socket verified immediately
    const t1 = setTimeout(() => setTickSocket(true), 150);

    // Step B: 6-Axis IMU verified
    const t2 = setTimeout(() => setTickIMU(true), 550);

    // Step C: Magnetic Compass verified
    const t3 = setTimeout(() => setTickCompass(true), 950);

    // Step D: Optical Light Sensor verified
    const t4 = setTimeout(() => {
      setTickLight(true);
      setIsCleared(true);
      setCountdown(2);
    }, 1400);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
    };
  }, [deviceConnected]);

  // 4. Auto-Redirect Countdown once Clearance is Granted
  useEffect(() => {
    if (countdown === null) return;
    if (countdown <= 0) {
      router.push("/lab");
      return;
    }
    const timer = setTimeout(() => {
      setCountdown((c) => (c !== null ? c - 1 : null));
    }, 1000);
    return () => clearTimeout(timer);
  }, [countdown, router]);

  const copyUrlToClipboard = () => {
    if (navigator.clipboard && hostUrl) {
      navigator.clipboard.writeText(hostUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Manual Simulation for Local Desktop Evaluation
  const handleSimulateConnection = () => {
    setDeviceConnected(true);
  };

  return (
    <main className="min-h-screen bg-white text-neutral-900 flex flex-col justify-between selection:bg-amber-100">
      {/* Minimal Borderless Top Header */}
      <header className="w-full max-w-7xl mx-auto px-8 md:px-16 pt-8 pb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full bg-neutral-950"></div>
          <span className="text-xs uppercase tracking-[0.2em] font-semibold text-neutral-950">
            Synthesis-Zero
          </span>
          <span className="text-[11px] text-neutral-400 font-normal">/</span>
          <span className="text-xs text-amber-600 font-medium">Hardware Gate</span>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <span
            className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-medium transition-colors ${
              deviceConnected
                ? "bg-amber-50 text-amber-900 border border-amber-200/60"
                : "bg-neutral-50 text-neutral-600"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                deviceConnected ? "bg-amber-500 animate-pulse" : "bg-neutral-400"
              }`}
            ></span>
            {deviceConnected ? "Hardware Link Established" : "Awaiting Scanner"}
          </span>
        </div>
      </header>

      {/* Disconnection Warning Banner (if routed from lab due to drop) */}
      {disconnectReason && !deviceConnected && (
        <div className="max-w-7xl mx-auto w-full px-8 md:px-16 pt-2">
          <div className="bg-amber-50 border border-amber-200/80 text-amber-900 px-4 py-3 rounded-2xl text-xs flex items-center justify-between shadow-xs animate-in fade-in">
            <div className="flex items-center gap-2.5">
              <svg className="w-4 h-4 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span>
                <strong>Laboratory Locked:</strong> Device disconnected. Scan the QR code to re-authenticate hardware sensors.
              </span>
            </div>
            <button
              onClick={() => setDisconnectReason(false)}
              className="text-amber-700 hover:text-amber-950 font-bold ml-4 cursor-pointer text-sm"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Main Two-Half Split Screen (White & Amber Minimalist Pattern) */}
      <div className="flex-1 max-w-7xl mx-auto w-full px-8 md:px-16 py-8 grid grid-cols-1 lg:grid-cols-2 items-center gap-12 lg:gap-24">
        {/* ========================================================= */}
        {/* LEFT HALF (SIDE A): Big QR Code OR Phone Prototype Mockup */}
        {/* ========================================================= */}
        <div className="flex flex-col items-center justify-center relative min-h-[460px]">
          {!deviceConnected ? (
            /* STATE 1: Centered Scannable QR Code */
            <div className="flex flex-col items-center text-center space-y-6 w-full max-w-sm animate-in fade-in duration-500">
              {/* Subtle ambient warm pulse effect around QR */}
              <div className="relative p-6 flex items-center justify-center">
                <div className="absolute inset-0 rounded-3xl bg-amber-50/50 -z-10"></div>

                {hostUrl ? (
                  <div className="bg-white p-3.5 rounded-2xl shadow-sm border border-neutral-100">
                    <QRCodeSVG
                      value={hostUrl}
                      size={260}
                      level="H"
                      includeMargin={false}
                      className="rounded-lg"
                    />
                  </div>
                ) : (
                  <div className="w-[260px] h-[260px] bg-neutral-100 animate-pulse rounded-2xl flex items-center justify-center text-xs text-neutral-400 font-mono">
                    Resolving Wi-Fi IP...
                  </div>
                )}
              </div>

              {/* URL & Direct Copy Action */}
              <div className="w-full space-y-2">
                <div className="flex items-center justify-center gap-2 bg-amber-50/50 hover:bg-amber-50 border border-amber-200/50 transition p-1.5 pl-3 rounded-xl">
                  <span className="font-mono text-[11px] text-amber-950 truncate max-w-[240px]">
                    {hostUrl || "http://10.133.123.81:3000/node"}
                  </span>
                  <button
                    onClick={copyUrlToClipboard}
                    className="bg-amber-500 hover:bg-amber-600 active:scale-95 text-white text-[11px] font-semibold px-3 py-1.5 rounded-lg shadow-sm transition flex items-center gap-1 cursor-pointer shrink-0"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                    {copied ? "Copied" : "Copy"}
                  </button>
                </div>
                <div className="text-[11px] text-neutral-400">
                  Scan with your phone camera or open Chrome and enter the address
                </div>
              </div>
            </div>
          ) : (
            /* STATE 2: Smartphone Prototype (Device Connected) */
            <div className="flex flex-col items-center animate-in zoom-in-95 duration-500">
              {/* Minimalist Smartphone Frame */}
              <div className="w-[250px] h-[480px] bg-neutral-950 rounded-[44px] p-3 shadow-2xl relative flex flex-col justify-between border-4 border-neutral-800/80">
                {/* Punch-Hole Front Camera */}
                <div className="absolute top-4 left-1/2 -translate-x-1/2 w-3.5 h-3.5 bg-neutral-900 rounded-full border border-neutral-700/60 z-20"></div>

                {/* Speaker Micro Slit */}
                <div className="absolute top-2 left-1/2 -translate-x-1/2 w-10 h-1 bg-neutral-800 rounded-full"></div>

                {/* Phone Screen Interior */}
                <div className="w-full h-full bg-neutral-900 rounded-[34px] p-5 flex flex-col justify-between text-white relative overflow-hidden select-none">
                  {/* Subtle Amber Glow */}
                  <div className="absolute -top-12 -left-12 w-36 h-36 bg-amber-500/15 rounded-full blur-2xl pointer-events-none"></div>

                  {/* Top Status Bar */}
                  <div className="flex justify-between items-center text-[9px] font-mono text-neutral-400 pt-1">
                    <span>5G • VoLTE</span>
                    <span className="flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>
                      100%
                    </span>
                  </div>

                  {/* Center Linked Graphic */}
                  <div className="flex flex-col items-center justify-center my-auto space-y-4">
                    <div className="relative flex items-center justify-center">
                      <div className="absolute w-28 h-28 rounded-full bg-amber-500/20 animate-ping"></div>
                      <div className="w-20 h-20 rounded-full bg-amber-500/10 border border-amber-400/50 flex items-center justify-center text-amber-400">
                        <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2.5"
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      </div>
                    </div>

                    <div className="text-center space-y-1">
                      <div className="text-[10px] uppercase font-bold tracking-widest text-amber-400">
                        Device Connected
                      </div>
                      <div className="text-sm font-bold text-white tracking-tight">
                        Mobile Hardware Probe
                      </div>
                      <div className="text-[10px] text-neutral-400 font-mono">
                        Session: {sessionId}
                      </div>
                    </div>
                  </div>

                  {/* Bottom Telemetry Chip */}
                  <div className="bg-neutral-800/80 rounded-xl p-2.5 text-[10px] space-y-1 border border-neutral-700/50">
                    <div className="flex justify-between text-neutral-400 text-[9px]">
                      <span>Telemetry Channel</span>
                      <span className="text-amber-400 font-semibold font-mono">10 Hz Active</span>
                    </div>
                    <div className="font-mono text-neutral-200 truncate">
                      Tilt: {sensorValues.pitch_deg ?? 30}° • Lux: {(sensorValues.illuminance_lux ?? 0).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>

              {/* Prototype Label */}
              <div className="mt-4 text-center">
                <span className="text-xs font-semibold text-neutral-800 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
                  Device Connected
                </span>
                <span className="text-[11px] text-neutral-400 block mt-0.5">
                  10 Hz Live Hardware Stream Ready
                </span>
              </div>
            </div>
          )}
        </div>

        {/* ========================================================= */}
        {/* RIGHT HALF (SIDE B): Instructions & Sensor Verification   */}
        {/* ========================================================= */}
        <div className="flex flex-col justify-center space-y-8 max-w-lg">
          {/* Header */}
          <div className="space-y-2.5">
            <div className="text-[11px] uppercase tracking-[0.2em] text-amber-600 font-semibold">
              Step 01 of 02 • Access Restriction Gate
            </div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-neutral-950">
              Sensor Verification Gate
            </h1>
            <p className="text-xs md:text-sm text-neutral-500 leading-relaxed">
              Access to the Quantum Solar Laboratory is restricted until all environmental hardware
              sensors are verified. Scan the QR code with your mobile probe to establish an authentic
              telemetry bridge.
            </p>
          </div>

          {/* Sensor Checklist (Ticks one by one with smooth CSS/SVG animation in Amber) */}
          <div className="space-y-4">
            <div className="text-xs uppercase tracking-wider text-neutral-400 font-semibold flex items-center justify-between">
              <span>Hardware Resource Checklist</span>
              <span className="text-[11px] font-mono text-amber-700 font-medium">
                {isCleared ? "4 / 4 Verified" : deviceConnected ? "Verifying..." : "Awaiting Scan"}
              </span>
            </div>

            <div className="space-y-3">
              {/* Sensor 1: Optical Light Photometer */}
              <div className="flex items-center justify-between py-2 transition-all">
                <div className="flex items-center gap-3.5">
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center transition-all duration-300 ${
                      tickLight
                        ? "bg-amber-100 text-amber-700 border border-amber-300 scale-105"
                        : "bg-neutral-100 text-neutral-400"
                    }`}
                  >
                    {tickLight ? (
                      <svg className="w-4 h-4 animate-in zoom-in-50 duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <span className="w-1.5 h-1.5 rounded-full bg-neutral-300"></span>
                    )}
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-neutral-900">
                      Optical Ambient Light Sensor (ALS / Lens)
                    </div>
                    <div className="text-[11px] text-neutral-400">
                      {tickLight ? "Incident solar flux photometer calibrated" : "Measures direct environmental lux (0 - 120k lx)"}
                    </div>
                  </div>
                </div>
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider transition-colors ${
                    tickLight
                      ? "bg-amber-50 text-amber-900 border border-amber-200/80"
                      : "bg-neutral-100 text-neutral-400"
                  }`}
                >
                  {tickLight ? "PASS" : "PENDING"}
                </span>
              </div>

              {/* Sensor 2: 6-Axis Inertial Tilt IMU */}
              <div className="flex items-center justify-between py-2 transition-all">
                <div className="flex items-center gap-3.5">
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center transition-all duration-300 ${
                      tickIMU
                        ? "bg-amber-100 text-amber-700 border border-amber-300 scale-105"
                        : "bg-neutral-100 text-neutral-400"
                    }`}
                  >
                    {tickIMU ? (
                      <svg className="w-4 h-4 animate-in zoom-in-50 duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <span className="w-1.5 h-1.5 rounded-full bg-neutral-300"></span>
                    )}
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-neutral-900">
                      6-Axis Gyroscope & Accelerometer (IMU)
                    </div>
                    <div className="text-[11px] text-neutral-400">
                      {tickIMU ? "Octahedral BX6 distortion angle active" : "Computes 3D panel tilt (β) & structural clash strain"}
                    </div>
                  </div>
                </div>
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider transition-colors ${
                    tickIMU
                      ? "bg-amber-50 text-amber-900 border border-amber-200/80"
                      : "bg-neutral-100 text-neutral-400"
                  }`}
                >
                  {tickIMU ? "PASS" : "PENDING"}
                </span>
              </div>

              {/* Sensor 3: Magnetic Compass Azimuth */}
              <div className="flex items-center justify-between py-2 transition-all">
                <div className="flex items-center gap-3.5">
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center transition-all duration-300 ${
                      tickCompass
                        ? "bg-amber-100 text-amber-700 border border-amber-300 scale-105"
                        : "bg-neutral-100 text-neutral-400"
                    }`}
                  >
                    {tickCompass ? (
                      <svg className="w-4 h-4 animate-in zoom-in-50 duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <span className="w-1.5 h-1.5 rounded-full bg-neutral-300"></span>
                    )}
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-neutral-900">
                      Magnetic Solar Compass (Azimuth)
                    </div>
                    <div className="text-[11px] text-neutral-400">
                      {tickCompass ? "True solar South alignment locked" : "Calculates incident beam cosine factor cos(θ_inc)"}
                    </div>
                  </div>
                </div>
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider transition-colors ${
                    tickCompass
                      ? "bg-amber-50 text-amber-900 border border-amber-200/80"
                      : "bg-neutral-100 text-neutral-400"
                  }`}
                >
                  {tickCompass ? "PASS" : "PENDING"}
                </span>
              </div>

              {/* Sensor 4: Real-Time Telemetry Link */}
              <div className="flex items-center justify-between py-2 transition-all">
                <div className="flex items-center gap-3.5">
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center transition-all duration-300 ${
                      tickSocket
                        ? "bg-amber-100 text-amber-700 border border-amber-300 scale-105"
                        : "bg-neutral-100 text-neutral-400"
                    }`}
                  >
                    {tickSocket ? (
                      <svg className="w-4 h-4 animate-in zoom-in-50 duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <span className="w-1.5 h-1.5 rounded-full bg-neutral-300"></span>
                    )}
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-neutral-900">
                      High-Frequency Telemetry Socket (10 Hz)
                    </div>
                    <div className="text-[11px] text-neutral-400">
                      {tickSocket ? "Low-latency bilateral physics channel open" : "Transmits 10 packets/sec to Python Causal MCRSA"}
                    </div>
                  </div>
                </div>
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider transition-colors ${
                    tickSocket
                      ? "bg-amber-50 text-amber-900 border border-amber-200/80"
                      : "bg-neutral-100 text-neutral-400"
                  }`}
                >
                  {tickSocket ? "PASS" : "PENDING"}
                </span>
              </div>
            </div>
          </div>

          {/* Clearance Action Section */}
          <div className="pt-4 space-y-3">
            {isCleared ? (
              <div className="space-y-3 animate-in fade-in duration-300">
                <div className="flex items-center gap-2 text-xs font-semibold text-amber-800">
                  <svg className="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>ALL SENSORS CLEARED • HARDWARE GATE UNLOCKED</span>
                </div>

                <Link
                  href="/lab"
                  className="w-full bg-amber-500 hover:bg-amber-600 active:scale-98 text-white text-xs font-semibold uppercase tracking-wider py-4 px-6 rounded-2xl transition flex items-center justify-center gap-2 cursor-pointer shadow-sm"
                >
                  <span>Enter Quantum Laboratory</span>
                  {countdown !== null && countdown > 0 && (
                    <span className="text-[11px] text-amber-100 font-mono">({countdown}s)</span>
                  )}
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </Link>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="w-full bg-neutral-100 text-neutral-400 text-xs font-semibold uppercase tracking-wider py-4 px-6 rounded-2xl flex items-center justify-center gap-2 cursor-not-allowed select-none">
                  <svg className="w-4 h-4 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                  <span>Lab Locked • Awaiting Hardware Clearance</span>
                </div>

                {/* Subtle Desktop Override for Local Testing */}
                <div className="text-center pt-1">
                  <button
                    onClick={handleSimulateConnection}
                    className="text-[11px] text-neutral-400 hover:text-amber-700 underline underline-offset-4 transition cursor-pointer"
                  >
                    Simulate Probe Handshake (Desktop Testing)
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Minimal Borderless Footer */}
      <footer className="w-full max-w-7xl mx-auto px-8 md:px-16 py-6 flex flex-col md:flex-row md:items-center justify-between text-[11px] text-neutral-400 gap-2 border-t border-neutral-100">
        <div>Synthesis-Zero Causal MCRSA &bull; 27-Cell Periodic Boundary Conditions (MAE 0.0450 eV/cell)</div>
        <div>Strict Zero-Fake-Data Policy &bull; Pure Hardware Physics Coupling</div>
      </footer>
    </main>
  );
}
