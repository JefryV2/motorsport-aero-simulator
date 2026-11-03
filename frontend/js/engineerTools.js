/** Engineering utilities — exports, comparisons, unit formatting. */

import { reynoldsNumber } from './flowEngine.js';
import { REFERENCE } from './carSpec.js';

const RHO = REFERENCE.rho; // kg/m³ sea level, the same constant the geometry uses

export function dynamicPressurePa(flow) {
  const v = flow.windSpeedKmh / 3.6;
  return 0.5 * RHO * v * v;
}

export function engineeringContext(flow, sim = null, speedRange = null) {
  const vMs = flow.windSpeedKmh / 3.6;
  const q = dynamicPressurePa(flow);
  const re = reynoldsNumber(flow);
  const ctx = {
    windSpeedKmh: flow.windSpeedKmh,
    windSpeedMs: vMs,
    yawDeg: flow.yawDeg ?? 0,
    pitchDeg: flow.pitchDeg ?? 0,
    reynolds: re,
    dynamicPressurePa: q,
    dynamicPressureKPa: q / 1000,
    airDensity: RHO,
    turbulencePct: (flow.turbulence ?? 0) * 100,
  };
  if (speedRange) {
    ctx.flowSpeedMinMs = speedRange.min;
    ctx.flowSpeedMaxMs = speedRange.max;
    ctx.flowSpeedMinKmh = speedRange.min * 3.6;
    ctx.flowSpeedMaxKmh = speedRange.max * 3.6;
  }
  if (sim?.forces) {
    const f = sim.forces;
    ctx.cl = f.cl_total;
    ctx.cd = f.cd_total;
    ctx.downforceN = f.downforce_total_N;
    ctx.downforceFrontN = f.downforce_front_N;
    ctx.downforceRearN = f.downforce_rear_N;
    ctx.dragN = f.drag_N;
    ctx.efficiency = f.efficiency;
    ctx.balanceFrontPct = f.balance_front_pct;
    ctx.copXm = f.cop_x_m;
    ctx.lapDeltaS = sim.lap_time_delta_s;
    ctx.method = sim.method;
  }
  return ctx;
}

export function compareToBaseline(current, baseline) {
  if (!current?.forces || !baseline?.forces) return null;
  const c = current.forces;
  const b = baseline.forces;
  return {
    downforceN: c.downforce_total_N - b.downforce_total_N,
    dragN: c.drag_N - b.drag_N,
    efficiency: c.efficiency - b.efficiency,
    balanceFrontPct: c.balance_front_pct - b.balance_front_pct,
    cl: c.cl_total - b.cl_total,
    cd: c.cd_total - b.cd_total,
    lapDeltaS: current.lap_time_delta_s - baseline.lap_time_delta_s,
  };
}

export function formatDelta(val, digits = 1, unit = '') {
  const sign = val >= 0 ? '+' : '';
  return `${sign}${val.toFixed(digits)}${unit}`;
}

export function exportSetupJson(config, flow, meta = {}) {
  const payload = {
    exported_at: new Date().toISOString(),
    config,
    flow,
    ...meta,
  };
  downloadBlob(JSON.stringify(payload, null, 2), 'aero-setup.json', 'application/json');
}

export function exportResultsCsv(sim, engCtx) {
  const rows = [
    ['Parameter', 'Value', 'Unit'],
    ['Downforce total', sim?.forces?.downforce_total_N ?? '', 'N'],
    ['Downforce front', sim?.forces?.downforce_front_N ?? '', 'N'],
    ['Downforce rear', sim?.forces?.downforce_rear_N ?? '', 'N'],
    ['Drag', sim?.forces?.drag_N ?? '', 'N'],
    ['L/D efficiency', sim?.forces?.efficiency ?? '', ''],
    ['Cl', sim?.forces?.cl_total ?? '', ''],
    ['Cd', sim?.forces?.cd_total ?? '', ''],
    ['Balance front', sim?.forces?.balance_front_pct ?? '', '%'],
    ['CoP x', sim?.forces?.cop_x_m ?? '', 'm'],
    ['Lap delta', sim?.lap_time_delta_s ?? '', 's'],
    ['Wind speed', engCtx?.windSpeedKmh ?? '', 'km/h'],
    ['Dynamic pressure', engCtx?.dynamicPressureKPa ?? '', 'kPa'],
    ['Reynolds number', engCtx?.reynolds ?? '', ''],
    [
      'Reynolds basis',
      `car length ${REFERENCE.carLengthM} m nose to tail; the analytical and VLM tiers apply no Re scaling, ` +
        'the CFD tier reads a chord Reynolds number in its own section case',
      '',
    ],
    ['Flow speed min', engCtx?.flowSpeedMinKmh ?? '', 'km/h'],
    ['Flow speed max', engCtx?.flowSpeedMaxKmh ?? '', 'km/h'],
    ['Method', sim?.method ?? '', ''],
    ['Fidelity', sim?.fidelity ?? 'unknown', ''],
    ['Provenance verified', sim?.provenance?.is_synthetic === true ? 'synthetic fallback' : 'solver/analytical', ''],
  ];
  const csv = rows.map((r) => r.map((c) => `"${c}"`).join(',')).join('\n');
  downloadBlob(csv, 'aero-results.csv', 'text/csv');
}

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function renderEngineerPanel(ctx, delta = null) {
  const lines = [];
  lines.push(`V∞ ${ctx.windSpeedKmh} km/h (${ctx.windSpeedMs.toFixed(1)} m/s) · yaw ${ctx.yawDeg}° · pitch ${ctx.pitchDeg}°`);
  lines.push(
    `Re ${(ctx.reynolds / 1e6).toFixed(2)}×10⁶ on L ${REFERENCE.carLengthM} m (readout - the analytical and VLM tiers apply no Re scaling) · ` +
      `q ${ctx.dynamicPressureKPa.toFixed(2)} kPa · ρ ${ctx.airDensity} kg/m³`,
  );
  if (ctx.flowSpeedMinMs != null) {
    lines.push(`Flow field ${ctx.flowSpeedMinKmh.toFixed(0)}–${ctx.flowSpeedMaxKmh.toFixed(0)} km/h (color scale)`);
  }
  if (ctx.cl != null) {
    lines.push(`Cl ${ctx.cl.toFixed(3)} · Cd ${ctx.cd.toFixed(3)} · CoP ${ctx.copXm.toFixed(2)} m from front axle`);
    lines.push(`DF ${ctx.downforceN.toLocaleString()} N (F ${ctx.downforceFrontN.toLocaleString()} / R ${ctx.downforceRearN.toLocaleString()}) · Drag ${ctx.dragN.toLocaleString()} N`);
  }
  if (delta) {
    lines.push(`Δ vs baseline: DF ${formatDelta(delta.downforceN, 0, ' N')} · Drag ${formatDelta(delta.dragN, 0, ' N')} · L/D ${formatDelta(delta.efficiency, 2)} · Bal ${formatDelta(delta.balanceFrontPct, 1, '%')}`);
  }
  return lines.join('\n');
}
