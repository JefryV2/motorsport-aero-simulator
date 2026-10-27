/** Surface pressure field — normal-based Cp with fixed CFD color scale. */

import { windDirection, DEFAULT_FLOW, speedToColor } from './flowEngine.js';

/** Fixed Cp legend: blue = suction (-2), white = neutral, red = stagnation (+1) */
export const CP_MIN = -2.0;
export const CP_MAX = 1.0;

export function cpToColor(cp) {
  const t = Math.max(0, Math.min(1, (cp - CP_MIN) / (CP_MAX - CP_MIN)));
  // Blue → cyan → white → yellow → red (standard CFD)
  if (t < 0.2) {
    const s = t / 0.2;
    return { r: 0, g: s * 0.35, b: 0.55 + s * 0.45 };
  }
  if (t < 0.45) {
    const s = (t - 0.2) / 0.25;
    return { r: s * 0.3, g: 0.35 + s * 0.55, b: 1 - s * 0.35 };
  }
  if (t < 0.65) {
    const s = (t - 0.45) / 0.2;
    return { r: 0.3 + s * 0.65, g: 0.9 - s * 0.15, b: 0.65 - s * 0.55 };
  }
  if (t < 0.85) {
    const s = (t - 0.65) / 0.2;
    return { r: 0.95, g: 0.75 - s * 0.35, b: 0.1 - s * 0.05 };
  }
  const s = (t - 0.85) / 0.15;
  return { r: 0.95, g: 0.4 - s * 0.35, b: 0.05 };
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(v) {
  const l = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

export function computeCpField(vertices, normals, config, bounds, simForces = null, flow = DEFAULT_FLOW) {
  const wind = windDirection(flow);
  const windInto = [-wind[0], -wind[1], -wind[2]];

  const fw = ((config?.front_wing_main?.angle_deg || 0) + (config?.front_wing_flap?.angle_deg || 0) * 0.5) / 20;
  const rw = ((config?.rear_wing_main?.angle_deg || 0) + (config?.rear_wing_flap?.angle_deg || 0) * 0.5) / 25;
  const ge = Math.exp(-((config?.ride_height_front_mm || 25) + (config?.ride_height_rear_mm || 45)) / 2000 / 0.015);

  const cpValues = [];

  for (let i = 0; i < vertices.length; i++) {
    const n = normals?.[i] ? normalize(normals[i]) : [0, 1, 0];
    const nDot = dot(n, windInto);

    // Potential-flow Cp: 1 at stagnation, negative on accelerated surfaces
    let cp = 1.0 - nDot * nDot;

    // Suction on leeward / upper surfaces (downforce)
    if (n[1] > 0.2) cp -= 0.35 * n[1] * (1 + fw + rw);
    // Underbody suction (ground effect)
    if (n[1] < -0.15) cp -= 0.5 * Math.abs(n[1]) * ge;
    // Leading edge suction peak
    if (nDot > 0.5) cp -= 0.25 * nDot;
    // Wake/base pressure on rear-facing
    const nWake = dot(n, wind);
    if (nWake > 0.3) cp -= 0.4 * nWake;

    cp = Math.max(CP_MIN, Math.min(CP_MAX, cp));
    cpValues.push(cp);
  }

  const colors = cpValues.map((cp) => cpToColor(cp));
  let actualMin = CP_MIN;
  let actualMax = CP_MAX;
  for (let i = 0; i < cpValues.length; i++) {
    const cp = cpValues[i];
    if (cp < actualMin) actualMin = cp;
    if (cp > actualMax) actualMax = cp;
  }

  return {
    cp: cpValues,
    colors,
    range: { min: actualMin, max: actualMax, scaleMin: CP_MIN, scaleMax: CP_MAX },
  };
}

/** Color mesh by local air speed (m/s) at each vertex. */
export function computeSpeedField(vertices, normals, config, bounds, flow, velocityFn) {
  const span = bounds.max.map((v, i) => Math.max(v - bounds.min[i], 0.01));
  const eps = Math.max(span[0], span[1], span[2]) * 0.02;
  const speeds = [];
  let minS = Infinity;
  let maxS = 0;

  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i];
    const n = normals?.[i] || [0, 1, 0];
    const [vx, vy, vz] = velocityFn(
      v[0] + n[0] * eps, v[1] + n[1] * eps, v[2] + n[2] * eps,
      config, bounds, flow,
    );
    const spd = Math.sqrt(vx * vx + vy * vy + vz * vz);
    speeds.push(spd);
    if (spd < minS) minS = spd;
    if (spd > maxS) maxS = spd;
  }

  const ref = (flow.windSpeedKmh || 200) / 3.6;
  const colors = speeds.map((s) => speedToColor(s, ref, minS, maxS));
  return { speeds, colors, range: { min: minS, max: maxS } };
}

export { velocityAt, generateStreamlines, spawnParticles, speedToColor, reynoldsNumber } from './flowEngine.js';
