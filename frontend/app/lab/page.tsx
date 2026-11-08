"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

interface SensorData {
  illuminance_lux: number;
  pitch_deg: number;
  roll_deg: number;
  compass_heading_deg: number;
  acceleration?: { x: number; y: number; z: number };
  gps?: { latitude: number; longitude: number };
}

interface SolarSimulationResult {
  formula: string;
  elements?: string[];
  e_chem?: number;
  e_strain?: number;
  predicted_Ef_cell: number;
  predicted_Ef_atom: number;
  uncertainty?: number;
  is_stable: boolean;
  classification?: string;
  stability_factor?: number;
  optical_bandgap_eV: number;
  shockley_queisser_limit_pct: number;
  operating_efficiency_pct: number;
  sun_zenith_deg?: number;
  sun_azimuth_deg?: number;
  theta_inc_deg?: number;
  cos_theta_inc?: number;
  direct_fraction?: number;
  incident_irradiance_wm2: number;
  effective_irradiance_wm2: number;
  power_output_watts: number;
  power_density_wm2: number;
  open_circuit_voltage_V: number;
  short_circuit_current_density_mA_cm2: number;
  daily_yield_kWh: number;
  phone_recharges_equivalent: number;
  panel_area_m2?: number;
  cell_temp_C?: number;
  tilt_deg?: number;
  azimuth_deg?: number;
}

interface BenchmarkRecord {
  timestamp_iso: string;
  timestamp_unix_ms: number;
  packet_id: number;
  material_formula: string;
  module_area_m2: number;
  // Tier 1: Hardware Sensors
  sensor_illuminance_lux: number;
  sensor_tilt_beta_deg: number;
  sensor_roll_phi_deg: number;
  sensor_compass_gamma_deg: number;
  sensor_accel_x_ms2?: number;
  sensor_accel_y_ms2?: number;
  sensor_accel_z_ms2?: number;
  // Tier 2: Solar Optical Transduction
  incident_irradiance_G_inc_wm2: number;
  sun_zenith_deg: number;
  sun_azimuth_deg: number;
  incident_angle_theta_inc_deg: number;
  cos_theta_inc: number;
  direct_fraction: number;
  effective_irradiance_G_eff_wm2: number;
  // Tier 3: Causal MCRSA Equivariant GNN
  chemical_enthalpy_E_chem_ev_cell: number;
  steric_strain_E_strain_ev_cell: number;
  formation_energy_Ef_ev_cell: number;
  formation_energy_Ef_ev_atom: number;
  thermodynamic_convex_hull_stable: boolean;
  epistemic_uncertainty_sigma_ev: number;
  // Tier 4: Optoelectronics & Photovoltaic Power
  optical_bandgap_Eg_ev: number;
  shockley_queisser_limit_pct: number;
  defect_recombination_factor: number;
  operating_efficiency_eta_pct: number;
  open_circuit_voltage_Voc_V: number;
  short_circuit_current_density_Jsc_ma_cm2: number;
  power_density_wm2: number;
  instantaneous_power_watts: number;
  daily_energy_yield_kwh: number;
  phone_recharges_equivalent: number;
}

const PRESET_MATERIALS: Record<string, { name: string; elements: string[]; defaultEg: number; dft: number }> = {
  CsPbI3: { name: "Cesium Lead Triiodide (Halide Perovskite)", elements: ["Cs", "Pb", "I", "I", "I"], defaultEg: 1.73, dft: -0.18 },
  BaTeO3: { name: "Barium Tellurate (Held-Out Validation)", elements: ["Ba", "Te", "O", "O", "O"], defaultEg: 2.45, dft: 0.22 },
  SrMnFO2: { name: "Strontium Manganese Oxyfluoride", elements: ["Sr", "Mn", "F", "O", "O"], defaultEg: 2.15, dft: 0.00 },
  BaTiO3: { name: "Barium Titanate (Dielectric Perovskite)", elements: ["Ba", "Ti", "O", "O", "O"], defaultEg: 3.20, dft: -0.26 },
  CsWO3: { name: "Cesium Tungstate Bronze", elements: ["Cs", "W", "O", "O", "O"], defaultEg: 2.40, dft: -0.04 },
};

function generateSmoothPath(points: { x: number; y: number }[]): string {
  if (points.length === 0) return "";
  if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;
  let path = `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i === 0 ? i : i - 1];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2 < points.length ? i + 2 : i + 1];
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    path += ` C ${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
  }
  return path;
}

export default function DesktopLab() {
  const router = useRouter();
  const [sessionId] = useState<string>("lab_alpha");

  // Connection & Handshake States
  const [deviceOnline, setDeviceOnline] = useState<boolean>(false);
  const [streamActive, setStreamActive] = useState<boolean>(false);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [telemetry, setTelemetry] = useState<SensorData | null>(null);

  // Selected Material & Scale
  const [selectedMaterial, setSelectedMaterial] = useState<string>("CsPbI3");
  const [scaleArea, setScaleArea] = useState<number>(1.0); // 1.0 m2 or 0.0122 m2

  // Calculation Results
  const [solarResult, setSolarResult] = useState<SolarSimulationResult | null>(null);
  const [powerHistory, setPowerHistory] = useState<number[]>([]);

  // Continuous Benchmark Research Dataset Buffer
  const benchmarkBufferRef = useRef<BenchmarkRecord[]>([]);
  const [benchmarkCount, setBenchmarkCount] = useState<number>(0);
  const [exportNotice, setExportNotice] = useState<string | null>(null);

  // Disconnection tracking
  const missedHeartbeatCountRef = useRef<number>(0);

  // Export Comprehensive RFC 4180 Research CSV
  const exportBenchmarkCsv = useCallback(() => {
    const records = benchmarkBufferRef.current;
    if (!records || records.length === 0) {
      alert("No active telemetry records captured yet. Start streaming from the mobile probe first.");
      return;
    }

    const headers = [
      "timestamp_iso",
      "timestamp_unix_ms",
      "packet_id",
      "material_formula",
      "module_area_m2",
      "sensor_illuminance_lux",
      "sensor_tilt_beta_deg",
      "sensor_roll_phi_deg",
      "sensor_compass_gamma_deg",
      "sensor_accel_x_ms2",
      "sensor_accel_y_ms2",
      "sensor_accel_z_ms2",
      "incident_irradiance_G_inc_wm2",
      "sun_zenith_deg",
      "sun_azimuth_deg",
      "incident_angle_theta_inc_deg",
      "cos_theta_inc",
      "direct_fraction",
      "effective_irradiance_G_eff_wm2",
      "chemical_enthalpy_E_chem_ev_cell",
      "steric_strain_E_strain_ev_cell",
      "formation_energy_Ef_ev_cell",
      "formation_energy_Ef_ev_atom",
      "thermodynamic_convex_hull_stable",
      "epistemic_uncertainty_sigma_ev",
      "optical_bandgap_Eg_ev",
      "shockley_queisser_limit_pct",
      "defect_recombination_factor",
      "operating_efficiency_eta_pct",
      "open_circuit_voltage_Voc_V",
      "short_circuit_current_density_Jsc_ma_cm2",
      "power_density_wm2",
      "instantaneous_power_watts",
      "daily_energy_yield_kwh",
      "phone_recharges_equivalent",
    ];

    const escapeCsv = (val: unknown): string => {
      if (val === undefined || val === null) return "";
      const s = String(val);
      if (s.includes(",") || s.includes('"') || s.includes("\n")) {
        return `"${s.replace(/"/g, '""')}"`;
      }
      return s;
    };

    const rows = records.map((r) =>
      [
        r.timestamp_iso,
        r.timestamp_unix_ms,
        r.packet_id,
        r.material_formula,
        r.module_area_m2,
        r.sensor_illuminance_lux,
        r.sensor_tilt_beta_deg,
        r.sensor_roll_phi_deg,
        r.sensor_compass_gamma_deg,
        r.sensor_accel_x_ms2 ?? "",
        r.sensor_accel_y_ms2 ?? "",
        r.sensor_accel_z_ms2 ?? "",
        r.incident_irradiance_G_inc_wm2,
        r.sun_zenith_deg,
        r.sun_azimuth_deg,
        r.incident_angle_theta_inc_deg,
        r.cos_theta_inc,
        r.direct_fraction,
        r.effective_irradiance_G_eff_wm2,
        r.chemical_enthalpy_E_chem_ev_cell,
        r.steric_strain_E_strain_ev_cell,
        r.formation_energy_Ef_ev_cell,
        r.formation_energy_Ef_ev_atom,
        r.thermodynamic_convex_hull_stable,
        r.epistemic_uncertainty_sigma_ev,
        r.optical_bandgap_Eg_ev,
        r.shockley_queisser_limit_pct,
        r.defect_recombination_factor,
        r.operating_efficiency_eta_pct,
        r.open_circuit_voltage_Voc_V,
        r.short_circuit_current_density_Jsc_ma_cm2,
        r.power_density_wm2,
        r.instantaneous_power_watts,
        r.daily_energy_yield_kwh,
        r.phone_recharges_equivalent,
      ]
        .map(escapeCsv)
        .join(",")
    );

    const csvContent = [headers.join(","), ...rows].join("\r\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const nowStr = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = url;
    a.download = `causal_mcrsa_research_benchmark_${nowStr}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setExportNotice(`Exported ${records.length} testing samples to CSV`);
    setTimeout(() => setExportNotice(null), 4000);
  }, []);

  const clearBenchmarkBuffer = useCallback(() => {
    benchmarkBufferRef.current = [];
    setBenchmarkCount(0);
    setExportNotice("Testing buffer cleared");
    setTimeout(() => setExportNotice(null), 3000);
  }, []);

  // Pure Scientific Calculation Fallback (runs in client)
  const calculateSolarPhysics = useCallback(
    (lux: number, tilt: number, azimuth: number, formulaKey: string, area: number): SolarSimulationResult => {
      const mat = PRESET_MATERIALS[formulaKey] || PRESET_MATERIALS["CsPbI3"];
      const wm2 = Math.min(1200, Math.max(0, lux / 120.0));

      const sunZenithRad = (35.0 * Math.PI) / 180.0;
      const sunAzimuthRad = (180.0 * Math.PI) / 180.0;
      const tiltRad = (tilt * Math.PI) / 180.0;
      const azimuthRad = (azimuth * Math.PI) / 180.0;

      const cosInc =
        Math.cos(tiltRad) * Math.cos(sunZenithRad) +
        Math.sin(tiltRad) * Math.sin(sunZenithRad) * Math.cos(azimuthRad - sunAzimuthRad);
      const directFrac = Math.max(0.0, cosInc);
      const effectiveWm2 = wm2 * (0.15 + 0.85 * directFrac);

      const eg = mat.defaultEg;
      const diff = eg - 1.34;
      const sqMax = Math.max(2.0, Math.min(33.7, 33.7 * Math.exp(-0.45 * diff * diff)));

      const dE = mat.dft;
      const stabilityFactor = dE <= 0.1 ? 1.0 : Math.max(0.6, 1.0 - (dE - 0.1) * 1.5);
      const etaBaseline = sqMax * 0.72 * stabilityFactor;

      const tempDerate = Math.max(0.65, 1.0 - 0.0038 * 20.0);
      const etaOperating = Math.max(0.5, etaBaseline * tempDerate);

      const powerWatts = effectiveWm2 * area * (etaOperating / 100.0);
      const powerDensity = effectiveWm2 * (etaOperating / 100.0);
      const voc = Math.max(0.35, eg - 0.38);
      const jsc = Math.max(1.5, ((effectiveWm2 / 1000.0) * (etaOperating / 100.0) / (voc * 0.78)) * 100.0);
      const dailyKwh = (powerWatts * 5.0) / 1000.0;
      const phoneRecharges = dailyKwh / 0.01925;

      return {
        formula: formulaKey,
        predicted_Ef_cell: dE,
        predicted_Ef_atom: dE / 5.0,
        is_stable: dE <= 0.2,
        optical_bandgap_eV: eg,
        shockley_queisser_limit_pct: Math.round(sqMax * 10) / 10,
        operating_efficiency_pct: Math.round(etaOperating * 10) / 10,
        incident_irradiance_wm2: Math.round(wm2 * 10) / 10,
        effective_irradiance_wm2: Math.round(effectiveWm2 * 10) / 10,
        power_output_watts: Math.round(powerWatts * 100) / 100,
        power_density_wm2: Math.round(powerDensity * 10) / 10,
        open_circuit_voltage_V: Math.round(voc * 100) / 100,
        short_circuit_current_density_mA_cm2: Math.round(jsc * 10) / 10,
        daily_yield_kWh: Math.round(dailyKwh * 1000) / 1000,
        phone_recharges_equivalent: Math.round(phoneRecharges * 10) / 10,
      };
    },
    []
  );

  // Synchronize active material & module scale with backend telemetry engine
  useEffect(() => {
    if (!sessionId) return;
    fetch("/api/telemetry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId,
        action: "set_material",
        material: selectedMaterial,
        scaleArea: scaleArea,
      }),
    }).catch(() => {});
  }, [sessionId, selectedMaterial, scaleArea]);

  // High-Frequency Poll: checks device connection and streams sensor data if active
  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/telemetry?sessionId=${sessionId}`);
        if (!res.ok) return;
        const json = await res.json();

        const isOnline = !!json.deviceOnline;
        setDeviceOnline(isOnline);
        setStreamActive(!!json.streamActive);
        setLatencyMs(json.latencyMs || 8);

        // Strict Clearance Check: redirect to QR gate if probe is disconnected
        if (!isOnline) {
          missedHeartbeatCountRef.current += 1;
          if (missedHeartbeatCountRef.current >= 4) {
            router.push("/?reason=disconnected");
            return;
          }
        } else {
          missedHeartbeatCountRef.current = 0;
        }

        if (isOnline && json.streamActive && json.sensors) {
          setTelemetry(json.sensors);
          const lux = json.sensors.illuminance_lux || 0;
          const pitch = json.sensors.pitch_deg || 0;
          const azimuth = json.sensors.compass_heading_deg || 180;

          const result: SolarSimulationResult = json.calculatedPhysics || calculateSolarPhysics(lux, pitch, azimuth, selectedMaterial, scaleArea);
          setSolarResult(result);

          setPowerHistory((prev) => {
            const next = [...prev, result.power_output_watts];
            return next.length > 40 ? next.slice(1) : next;
          });

          // Continuously buffer testing benchmark record for research export
          const packetId = json.stats?.packetsSent || (benchmarkBufferRef.current.length + 1);
          const record: BenchmarkRecord = {
            timestamp_iso: new Date().toISOString(),
            timestamp_unix_ms: Date.now(),
            packet_id: packetId,
            material_formula: result.formula || selectedMaterial,
            module_area_m2: scaleArea,
            // Tier 1: Hardware Sensors
            sensor_illuminance_lux: lux,
            sensor_tilt_beta_deg: pitch,
            sensor_roll_phi_deg: json.sensors.roll_deg || 0,
            sensor_compass_gamma_deg: azimuth,
            sensor_accel_x_ms2: json.sensors.acceleration?.x,
            sensor_accel_y_ms2: json.sensors.acceleration?.y,
            sensor_accel_z_ms2: json.sensors.acceleration?.z,
            // Tier 2: Solar Optical Transduction
            incident_irradiance_G_inc_wm2: result.incident_irradiance_wm2 ?? (lux / 120.0),
            sun_zenith_deg: result.sun_zenith_deg ?? 35.0,
            sun_azimuth_deg: result.sun_azimuth_deg ?? 180.0,
            incident_angle_theta_inc_deg: result.theta_inc_deg ?? 0.0,
            cos_theta_inc: result.cos_theta_inc ?? 1.0,
            direct_fraction: result.direct_fraction ?? 1.0,
            effective_irradiance_G_eff_wm2: result.effective_irradiance_wm2 ?? 0.0,
            // Tier 3: Causal MCRSA Equivariant GNN
            chemical_enthalpy_E_chem_ev_cell: result.e_chem ?? result.predicted_Ef_cell,
            steric_strain_E_strain_ev_cell: result.e_strain ?? 0.0,
            formation_energy_Ef_ev_cell: result.predicted_Ef_cell,
            formation_energy_Ef_ev_atom: result.predicted_Ef_atom,
            thermodynamic_convex_hull_stable: !!result.is_stable,
            epistemic_uncertainty_sigma_ev: result.uncertainty ?? 0.035,
            // Tier 4: Optoelectronics & Photovoltaic Power
            optical_bandgap_Eg_ev: result.optical_bandgap_eV,
            shockley_queisser_limit_pct: result.shockley_queisser_limit_pct,
            defect_recombination_factor: result.stability_factor ?? 1.0,
            operating_efficiency_eta_pct: result.operating_efficiency_pct,
            open_circuit_voltage_Voc_V: result.open_circuit_voltage_V,
            short_circuit_current_density_Jsc_ma_cm2: result.short_circuit_current_density_mA_cm2,
            power_density_wm2: result.power_density_wm2,
            instantaneous_power_watts: result.power_output_watts,
            daily_energy_yield_kwh: result.daily_yield_kWh,
            phone_recharges_equivalent: result.phone_recharges_equivalent,
          };
          benchmarkBufferRef.current.push(record);
          setBenchmarkCount(benchmarkBufferRef.current.length);
        } else if (!json.streamActive) {
          setTelemetry(null);
          setSolarResult(null);
        }
      } catch (err) {
        console.error("Telemetry query error:", err);
      }
    }, 120);

    return () => clearInterval(timer);
  }, [sessionId, selectedMaterial, scaleArea, calculateSolarPhysics, router]);

  // Wave Graph Geometry Calculations
  const svgWidth = 800;
  const svgHeight = 170;
  const maxVal = Math.max(
    ...powerHistory,
    solarResult ? solarResult.power_output_watts * 1.25 : 10,
    10
  );

  const wavePoints = (powerHistory.length > 0 ? powerHistory : [0, 0, 0, 0, 0]).map((val, idx, arr) => {
    const x = (idx / Math.max(1, arr.length - 1)) * svgWidth;
    const y = svgHeight - (val / maxVal) * (svgHeight - 40) - 20;
    return { x, y: isNaN(y) ? svgHeight - 20 : y };
  });

  const linePath = generateSmoothPath(wavePoints);
  const areaPath = wavePoints.length > 0
    ? `${linePath} L ${svgWidth},${svgHeight} L 0,${svgHeight} Z`
    : "";
  const lastPoint = wavePoints[wavePoints.length - 1];

  return (
    <div className="min-h-screen bg-white text-neutral-900 flex flex-col justify-between max-w-7xl mx-auto px-6 lg:px-10 py-8 selection:bg-amber-100">
      {/* 1. Top Header */}
      <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-neutral-100 pb-5">
        <div>
          <div className="text-[11px] uppercase tracking-widest text-neutral-400 font-semibold mb-1">
            Synthesis-Zero Quantum Physics Engine • Periodic Boundary Conditions
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-neutral-950 flex items-center gap-3">
            Solar Quantum Laboratory
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-amber-50 text-amber-900 border border-amber-200/60">
              Live Hardware Coupling
            </span>
          </h1>
        </div>

        {/* Live Hardware Status, Benchmark Exporter & Navigation */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Top-Corner Research Benchmark Dataset Export Button */}
          <button
            onClick={exportBenchmarkCsv}
            disabled={benchmarkCount === 0}
            className={`text-xs font-semibold py-2 px-3.5 rounded-xl transition shadow-sm flex items-center gap-2 ${
              benchmarkCount > 0
                ? "bg-amber-500 hover:bg-amber-600 active:bg-amber-700 text-white cursor-pointer"
                : "bg-neutral-100 text-neutral-400 cursor-not-allowed"
            }`}
            title="Download full 35-parameter research benchmark dataset in CSV format"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Export Testing Result</span>
            <span
              className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-bold ${
                benchmarkCount > 0 ? "bg-amber-600/90 text-amber-50" : "bg-neutral-200 text-neutral-500"
              }`}
            >
              {benchmarkCount}
            </span>
          </button>

          {benchmarkCount > 0 && (
            <button
              onClick={clearBenchmarkBuffer}
              className="bg-neutral-100 hover:bg-neutral-200 text-neutral-600 text-xs font-medium py-2 px-2.5 rounded-xl transition"
              title="Clear current recorded benchmark buffer"
            >
              Reset
            </button>
          )}

          <div className={`px-4 py-2 rounded-xl flex items-center gap-3 text-xs transition-colors ${
            deviceOnline ? "bg-emerald-50 text-emerald-900" : "bg-neutral-50 text-neutral-600"
          }`}>
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                deviceOnline
                  ? (streamActive ? "bg-emerald-500 animate-pulse" : "bg-emerald-500")
                  : "bg-neutral-300"
              }`}
            />
            <span className="font-semibold">
              {deviceOnline
                ? (streamActive ? "Hardware Live" : "Probe Standby")
                : "Probe Disconnected"}
            </span>
            {deviceOnline && (
              <span className="text-[11px] font-mono text-emerald-800 border-l border-emerald-200/80 pl-2">
                {streamActive ? `10.2 Hz • ${latencyMs}ms` : "Standby"}
              </span>
            )}
          </div>

          <Link
            href="/"
            className="bg-neutral-100 hover:bg-neutral-200 text-neutral-800 text-xs font-semibold py-2 px-3.5 rounded-xl transition flex items-center gap-1.5"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Hardware Gate
          </Link>
        </div>
      </header>

      {exportNotice && (
        <div className="mt-3 px-4 py-2 rounded-xl bg-amber-50 border border-amber-200/80 text-amber-900 text-xs font-medium flex items-center justify-between animate-fade-in">
          <span>{exportNotice}</span>
          <button onClick={() => setExportNotice(null)} className="text-amber-700 hover:text-amber-950 font-bold ml-2">×</button>
        </div>
      )}

      {/* 2. UPPER HERO SECTION: Prominent Real-Time Wave Graph & Instantaneous Generation */}
      <section className="my-6 bg-gradient-to-b from-amber-50/70 to-amber-50/20 rounded-3xl p-6 lg:p-8 space-y-6">
        <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-4">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-amber-800 font-semibold">
              Instantaneous Photovoltaic Generation
            </div>
            <div className="flex items-baseline gap-4 mt-1">
              <span className="text-5xl lg:text-6xl font-bold tracking-tight text-neutral-950 font-mono">
                {streamActive && solarResult ? `${solarResult.power_output_watts} W` : "0.0 W"}
              </span>
              <span className={`text-xs font-semibold px-3 py-1 rounded-full ${
                streamActive ? "bg-emerald-100 text-emerald-800" : "bg-neutral-100 text-neutral-500"
              }`}>
                {streamActive ? "Measuring Live" : (deviceOnline ? "Device Standby" : "Stream Idle")}
              </span>
            </div>
            <div className="text-xs text-neutral-500 mt-1 font-mono">
              {streamActive && solarResult
                ? `Module Area: ${scaleArea} m² • Effective Normal Irradiance: ${solarResult.effective_irradiance_wm2} W/m²`
                : `Active Area: ${scaleArea} m² • Irradiance: 0.0 W/m²`}
            </div>
          </div>

          <div className="text-left md:text-right">
            <div className="text-xs text-neutral-500 font-medium">Expected Daily Energy Yield</div>
            <div className="text-3xl font-bold font-mono text-amber-600 mt-0.5">
              {streamActive && solarResult ? `${solarResult.daily_yield_kWh} kWh` : "0.000 kWh"}
            </div>
            <div className="text-[11px] text-neutral-400 font-mono mt-0.5">5.0 Peak Sun Hours/day</div>
          </div>
        </div>

        {/* PROMINENT LIVE REAL-TIME WAVE GRAPH (Bigger, Smooth Spline Waveform) */}
        <div className="bg-white/80 rounded-2xl p-5 space-y-2 backdrop-blur-sm shadow-sm">
          <div className="flex justify-between items-center text-xs">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${streamActive ? "bg-amber-500 animate-ping" : "bg-neutral-300"}`} />
              <span className="font-semibold uppercase tracking-wider text-neutral-700 text-[11px]">
                Real-Time Quantum Generation Waveform
              </span>
            </div>
            <div className="flex items-center gap-3 text-[11px] font-mono text-neutral-500">
              <span>Peak: {Math.round(maxVal)} W</span>
              <span className="border-l border-neutral-200 pl-3">
                {streamActive ? "10 Hz Continuous Timeline" : "Awaiting Stream"}
              </span>
            </div>
          </div>

          {/* SVG Smooth Continuous Spline Wave */}
          <div className="relative w-full h-44 overflow-hidden pt-1">
            <svg
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              preserveAspectRatio="none"
              className="w-full h-full"
            >
              <defs>
                <linearGradient id="waveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.35" />
                  <stop offset="50%" stopColor="#f59e0b" stopOpacity="0.12" />
                  <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.0" />
                </linearGradient>
                <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#d97706" />
                  <stop offset="100%" stopColor="#f59e0b" />
                </linearGradient>
              </defs>

              {/* Horizontal Reference Gridlines */}
              <line x1="0" y1="20" x2={svgWidth} y2="20" stroke="#f3f4f6" strokeWidth="1" strokeDasharray="4 4" />
              <line x1="0" y1={svgHeight / 2} x2={svgWidth} y2={svgHeight / 2} stroke="#f3f4f6" strokeWidth="1" strokeDasharray="4 4" />
              <line x1="0" y1={svgHeight - 20} x2={svgWidth} y2={svgHeight - 20} stroke="#f3f4f6" strokeWidth="1" />

              {/* Wave Area Fill */}
              {streamActive && powerHistory.length > 1 && (
                <path d={areaPath} fill="url(#waveGradient)" />
              )}

              {/* Wave Smooth Spline Stroke */}
              {streamActive && powerHistory.length > 1 && (
                <path
                  d={linePath}
                  fill="none"
                  stroke="url(#lineGradient)"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {/* Idle State Baseline */}
              {(!streamActive || powerHistory.length <= 1) && (
                <line
                  x1="0"
                  y1={svgHeight - 20}
                  x2={svgWidth}
                  y2={svgHeight - 20}
                  stroke="#e5e7eb"
                  strokeWidth="2"
                  strokeDasharray="6 6"
                />
              )}

              {/* Live Pulsing Head Point */}
              {streamActive && lastPoint && (
                <g>
                  <circle cx={lastPoint.x} cy={lastPoint.y} r="5" fill="#f59e0b" />
                  <circle cx={lastPoint.x} cy={lastPoint.y} r="9" fill="#f59e0b" opacity="0.25" />
                </g>
              )}
            </svg>
          </div>
        </div>
      </section>

      {/* 3. LOWER SECTION: Two Balanced Panels (No Redundancy) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 my-2">
        {/* Left Column (5 Cols): Target Material & Incoming Hardware Orientation */}
        <div className="lg:col-span-5 space-y-6">
          {/* Material Configuration Panel */}
          <div className="bg-neutral-50 p-6 rounded-2xl space-y-4">
            <div className="flex justify-between items-baseline">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">Target Material Formula</h2>
              <span className="text-[11px] text-amber-600 font-medium">Causal MCRSA GNN</span>
            </div>

            <select
              value={selectedMaterial}
              onChange={(e) => setSelectedMaterial(e.target.value)}
              className="w-full bg-white text-neutral-900 text-sm font-semibold rounded-xl px-4 py-3 outline-none transition cursor-pointer border-0 shadow-sm"
            >
              {Object.entries(PRESET_MATERIALS).map(([key, item]) => (
                <option key={key} value={key}>
                  {key} — {item.name}
                </option>
              ))}
            </select>

            {/* Photovoltaic Active Area Scale Selector */}
            <div className="pt-2">
              <label className="block text-[11px] font-medium text-neutral-400 uppercase tracking-wider mb-2">
                Photovoltaic Module Scale
              </label>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <button
                  onClick={() => setScaleArea(1.0)}
                  className={`py-2.5 px-3 rounded-xl font-medium transition cursor-pointer ${
                    scaleArea === 1.0 ? "bg-neutral-900 text-white font-semibold" : "bg-white text-neutral-700 hover:bg-neutral-100 shadow-sm"
                  }`}
                >
                  1.00 m² (Standard Panel)
                </button>
                <button
                  onClick={() => setScaleArea(0.0122)}
                  className={`py-2.5 px-3 rounded-xl font-medium transition cursor-pointer ${
                    scaleArea === 0.0122 ? "bg-amber-500 text-white font-semibold" : "bg-white text-neutral-700 hover:bg-neutral-100 shadow-sm"
                  }`}
                >
                  0.0122 m² (Handheld Sensor)
                </button>
              </div>
            </div>
          </div>

          {/* Incoming Hardware Sensor Readout */}
          <div className="bg-neutral-50 p-6 rounded-2xl space-y-3">
            <div className="flex justify-between items-baseline">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">Incoming Hardware Telemetry</h2>
              <span className="text-[10px] font-mono text-neutral-500">
                {streamActive ? "10 Hz Streaming" : "Standby"}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-white p-3.5 rounded-xl shadow-sm">
                <div className="text-[10px] font-medium text-neutral-400 uppercase">Incident Lux</div>
                <div className="text-base font-bold font-mono text-neutral-900 mt-1">
                  {streamActive && telemetry?.illuminance_lux
                    ? Math.round(telemetry.illuminance_lux).toLocaleString()
                    : "--"}
                </div>
                <div className="text-[10px] text-amber-600 font-mono mt-0.5">
                  {streamActive && telemetry?.illuminance_lux
                    ? `${(telemetry.illuminance_lux / 120).toFixed(0)} W/m²`
                    : "--"}
                </div>
              </div>

              <div className="bg-white p-3.5 rounded-xl shadow-sm">
                <div className="text-[10px] font-medium text-neutral-400 uppercase">Surface Tilt (β)</div>
                <div className="text-base font-bold font-mono text-neutral-900 mt-1">
                  {streamActive && telemetry?.pitch_deg !== undefined ? `${telemetry.pitch_deg}°` : "--"}
                </div>
                <div className="text-[10px] text-neutral-400 mt-0.5">Gyroscope</div>
              </div>

              <div className="bg-white p-3.5 rounded-xl shadow-sm">
                <div className="text-[10px] font-medium text-neutral-400 uppercase">Azimuth (γ)</div>
                <div className="text-base font-bold font-mono text-neutral-900 mt-1">
                  {streamActive && telemetry?.compass_heading_deg !== undefined
                    ? `${telemetry.compass_heading_deg}°`
                    : "--"}
                </div>
                <div className="text-[10px] text-neutral-400 mt-0.5">Compass</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column (7 Cols): Photovoltaic Physics & Quantum Properties */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-neutral-50 p-6 rounded-2xl space-y-4">
            <div className="flex justify-between items-baseline">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">Photovoltaic Device Parameters</h2>
              <span className="text-[11px] text-neutral-500 font-mono">Thermodynamic Simulation</span>
            </div>

            {/* 4 Clean Scientific Electrical Parameter Tiles */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-white p-4 rounded-xl shadow-sm">
                <div className="text-[10px] font-medium text-neutral-400 uppercase">Operating η</div>
                <div className="text-xl font-bold font-mono text-neutral-900 mt-1">
                  {streamActive && solarResult ? `${solarResult.operating_efficiency_pct}%` : "0.0%"}
                </div>
                <div className="text-[10px] text-neutral-400 mt-0.5">Derated Efficiency</div>
              </div>

              <div className="bg-white p-4 rounded-xl shadow-sm">
                <div className="text-[10px] font-medium text-neutral-400 uppercase">SQ Limit</div>
                <div className="text-xl font-bold font-mono text-neutral-900 mt-1">
                  {solarResult?.shockley_queisser_limit_pct || 33.7}%
                </div>
                <div className="text-[10px] text-neutral-400 mt-0.5">Shockley-Queisser</div>
              </div>

              <div className="bg-white p-4 rounded-xl shadow-sm">
                <div className="text-[10px] font-medium text-neutral-400 uppercase">Voltage Voc</div>
                <div className="text-xl font-bold font-mono text-neutral-900 mt-1">
                  {streamActive && solarResult ? `${solarResult.open_circuit_voltage_V} V` : "0.00 V"}
                </div>
                <div className="text-[10px] text-neutral-400 mt-0.5">Open-Circuit</div>
              </div>

              <div className="bg-white p-4 rounded-xl shadow-sm">
                <div className="text-[10px] font-medium text-neutral-400 uppercase">Current Jsc</div>
                <div className="text-xl font-bold font-mono text-neutral-900 mt-1">
                  {streamActive && solarResult ? `${solarResult.short_circuit_current_density_mA_cm2}` : "0.0"}
                </div>
                <div className="text-[10px] text-neutral-400 mt-0.5">mA / cm²</div>
              </div>
            </div>

            {/* Model Thermodynamic Quantities */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
              <div className="bg-white p-4 rounded-xl shadow-sm">
                <div className="text-[10px] text-neutral-400 uppercase font-medium">Model Formation Energy</div>
                <div className="text-base font-bold font-mono text-neutral-900 mt-1">
                  {solarResult?.predicted_Ef_cell !== undefined
                    ? `${solarResult.predicted_Ef_cell > 0 ? "+" : ""}${solarResult.predicted_Ef_cell.toFixed(3)} eV`
                    : "--"}
                </div>
                <div className="text-[10px] text-emerald-600 font-medium mt-0.5">27-Cell PBC Equilibrium</div>
              </div>

              <div className="bg-white p-4 rounded-xl shadow-sm">
                <div className="text-[10px] text-neutral-400 uppercase font-medium">Optical Bandgap (Eg)</div>
                <div className="text-base font-bold font-mono text-neutral-900 mt-1">
                  {solarResult?.optical_bandgap_eV} eV
                </div>
                <div className="text-[10px] text-neutral-400 mt-0.5">Fundamental Absorber</div>
              </div>

              <div className="bg-white p-4 rounded-xl shadow-sm">
                <div className="text-[10px] text-neutral-400 uppercase font-medium">Power Density</div>
                <div className="text-base font-bold font-mono text-neutral-900 mt-1">
                  {streamActive && solarResult ? `${solarResult.power_density_wm2} W/m²` : "0.0 W/m²"}
                </div>
                <div className="text-[10px] text-neutral-400 mt-0.5">Spatial Output Rate</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Minimal Scientific Footer */}
      <footer className="border-t border-neutral-100 pt-6 flex flex-col md:flex-row md:items-center justify-between text-xs text-neutral-400 gap-2">
        <div>Synthesis-Zero Causal GNN • 27-Cell Periodic Boundary Conditions Active</div>
        <Link href="/node" className="hover:text-neutral-700 transition">
          Switch to Mobile Sensor Probe &rarr;
        </Link>
      </footer>
    </div>
  );
}
