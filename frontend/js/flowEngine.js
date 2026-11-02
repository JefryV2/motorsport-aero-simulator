/** Wind tunnel flow physics — air/car interaction for any model. */

export const DEFAULT_FLOW = {
  windSpeedKmh: 200,
  yawDeg: 0,
  pitchDeg: 0,
  turbulence: 0.12,
  particleCount: 400,
  streamlineCount: 28,
  flowSource: 'front_grid',
  timeScale: 1.0,
  /** Evolved by the viewer; drives the slow phase change of the eddy field. */
  time: 0,
  showWake: false,
  showTunnel: false,
  showSourcePlane: false,
  showGround: false,
  showInteraction: true,
  particleSize: 0.18,
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function len(v) {
  return Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
}

export function normalize(v) {
  const l = len(v) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

// ---------------------------------------------------------------------------
// Coherent turbulence: curl noise
// ---------------------------------------------------------------------------
// A real tunnel does not jitter frame to frame. Smoke follows eddies that are
// coherent in space and only evolve slowly, which is why the previous
// per-frame Math.random() jitter read as TV static instead of smoke. Sampling
// the curl of a vector potential gives a divergence-free field, so tracers
// swirl without being artificially compressed into sinks.

function hash01(ix, iy, iz) {
  let h = (ix * 374761393 + iy * 668265263 + iz * 1442695041) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function fade(t) {
  return t * t * (3 - 2 * t);
}

/** Trilinearly interpolated value noise on an integer lattice, in [0, 1]. */
function valueNoise3(x, y, z) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const fx = fade(x - x0);
  const fy = fade(y - y0);
  const fz = fade(z - z0);

  const c000 = hash01(x0, y0, z0);
  const c100 = hash01(x0 + 1, y0, z0);
  const c010 = hash01(x0, y0 + 1, z0);
  const c110 = hash01(x0 + 1, y0 + 1, z0);
  const c001 = hash01(x0, y0, z0 + 1);
  const c101 = hash01(x0 + 1, y0, z0 + 1);
  const c011 = hash01(x0, y0 + 1, z0 + 1);
  const c111 = hash01(x0 + 1, y0 + 1, z0 + 1);

  const x00 = c000 + (c100 - c000) * fx;
  const x10 = c010 + (c110 - c010) * fx;
  const x01 = c001 + (c101 - c001) * fx;
  const x11 = c011 + (c111 - c011) * fx;
  const y0v = x00 + (x10 - x00) * fy;
  const y1v = x01 + (x11 - x01) * fy;
  return y0v + (y1v - y0v) * fz;
}

/** Two-octave fractal value noise in [0, 1]. */
function fbm2(x, y, z) {
  return (
    valueNoise3(x, y, z) * 0.62 +
    valueNoise3(x * 2.17 + 13.7, y * 2.17 - 7.3, z * 2.17 + 3.1) * 0.38
  );
}

/** Vector potential built from three decorrelated noise fields. */
function potential3(x, y, z) {
  return [
    fbm2(x, y, z),
    fbm2(x + 71.3, y + 19.1, z + 43.7),
    fbm2(x - 53.9, y - 27.3, z + 11.9),
  ];
}

// Empirically measured RMS magnitude (see scripts/check_flow_field.mjs) so the
// returned field is normalised: one unit of gain is one unit of RMS velocity.
const CURL_NOISE_RMS = 0.76;
const CURL_EPS = 0.15;

/**
 * Divergence-free turbulence sampled at (x, y, z), slowly evolving in time.
 * Coordinates are in eddy lengths; the caller scales by the eddy size.
 */
export function curlNoise(x, y, z, time = 0) {
  const bx = x - time * 1.7; // eddies convect and slowly boil
  const bz = z + time * 0.6;
  const eps = CURL_EPS;
  // Central differences: the forward-difference version had O(eps) divergence
  // error large enough to compress tracers into visible clumps.
  const px1 = potential3(bx + eps, y, bz);
  const px0 = potential3(bx - eps, y, bz);
  const py1 = potential3(bx, y + eps, bz);
  const py0 = potential3(bx, y - eps, bz);
  const pz1 = potential3(bx, y, bz + eps);
  const pz0 = potential3(bx, y, bz - eps);
  const inv = 1 / (2 * eps * CURL_NOISE_RMS);
  return [
    ((py1[2] - py0[2]) - (pz1[1] - pz0[1])) * inv,
    ((pz1[0] - pz0[0]) - (px1[2] - px0[2])) * inv,
    ((px1[1] - px0[1]) - (py1[0] - py0[0])) * inv,
  ];
}

/** Scratch storage so the vortex loop allocates nothing per particle per frame. */
const _vortexOut = [0, 0, 0];

/**
 * Velocity induced by a trailing line vortex along +x with a Lamb-Oseen core.
 * The core keeps the induced velocity finite on the axis; the exponential
 * decay along x fades the vortex out downstream.
 */
function addLambOseen(x, y, z, x0, y0, z0, gamma, core, decayLen) {
  const dx = x - x0;
  if (dx <= 0) return false;
  const ry = y - y0;
  const rz = z - z0;
  const r2 = ry * ry + rz * rz + core * core;
  const decay = Math.exp(-dx / decayLen);
  const t = (gamma / (2 * Math.PI * r2)) * (1 - Math.exp(-r2 / (core * core))) * decay;
  _vortexOut[0] = 0;
  _vortexOut[1] = -rz * t;
  _vortexOut[2] = ry * t;
  return true;
}

export function windVector(flow) {
  const yaw = (flow.yawDeg * Math.PI) / 180;
  const pitch = (flow.pitchDeg * Math.PI) / 180;
  const speed = (flow.windSpeedKmh || 200) / 3.6;
  return [
    speed * Math.cos(pitch) * Math.cos(yaw),
    speed * Math.sin(pitch),
    speed * Math.cos(pitch) * Math.sin(yaw),
  ];
}

export function windDirection(flow) {
  return normalize(windVector(flow));
}

export function distToBBox(x, y, z, bounds) {
  const cx = clamp(x, bounds.min[0], bounds.max[0]);
  const cy = clamp(y, bounds.min[1], bounds.max[1]);
  const cz = clamp(z, bounds.min[2], bounds.max[2]);
  const dx = x - cx;
  const dy = y - cy;
  const dz = z - cz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function isInsideBBox(x, y, z, bounds, pad = 0) {
  return (
    x >= bounds.min[0] - pad && x <= bounds.max[0] + pad &&
    y >= bounds.min[1] - pad && y <= bounds.max[1] + pad &&
    z >= bounds.min[2] - pad && z <= bounds.max[2] + pad
  );
}

/** Velocity field with potential flow doublet + line vortex + source/sink singularity models. */
export function velocityAt(x, y, z, config, bounds, flow) {
  const [wx, wy, wz] = windVector(flow);
  const min = bounds.min;
  const max = bounds.max;
  const span = max.map((v, i) => Math.max(v - min[i], 0.01));
  const cx = (min[0] + max[0]) / 2;
  const cy = (min[1] + max[1]) / 2;
  const cz = (min[2] + max[2]) / 2;

  let vx = wx;
  let vy = wy;
  let vz = wz;

  if (flow.showInteraction !== false) {
    // 1. Regularized potential flow doublet perturbation function (for volumetric blockage)
    const getDoubletPert = (dx, dy, dz, R, ux, uy, uz) => {
      const d2 = dx*dx + dy*dy + dz*dz;
      const eps2 = 0.0025; // core regularization
      const d = Math.sqrt(d2 + eps2);
      const d3 = d*d*d;
      const d5 = d3*d*d;
      
      const uDotD = ux*dx + uy*dy + uz*dz;
      const R3 = R*R*R;
      // Standard doublet for flow past a body: the perturbation decays and,
      // ahead of the body, decelerates the flow (stagnation). Sign matters:
      // the inverted form accelerates flow into the nose, which made the inlet
      // run faster than freestream.
      const t1 = R3 / (2 * d3);
      const t2 = -(3 * R3 * uDotD) / (2 * d5);
      
      return [
        t1 * ux + t2 * dx,
        t1 * uy + t2 * dy,
        t1 * uz + t2 * dz
      ];
    };

    // 2. Line vortex perturbation function (using Biot-Savart for wings upwash/downwash)
    const getVortexPert = (dx, dy, dz, spanHalf, gamma) => {
      const r2_2d = dx*dx + dy*dy;
      const eps2 = 0.0025;
      const factor = gamma / (2 * Math.PI * (r2_2d + eps2));
      
      const zDist = Math.abs(dz);
      const spanFalloff = Math.exp(-Math.max(0, zDist - spanHalf) * 4.0);
      
      return [
        factor * dy * spanFalloff,
        -factor * dx * spanFalloff,
        0
      ];
    };

    // 3. Source/Sink perturbation function (for underbody Venturi ground effect)
    const getSourcePert = (dx, dy, dz, Q) => {
      const d2 = dx*dx + dy*dy + dz*dz;
      const eps2 = 0.0025;
      const d = Math.sqrt(d2 + eps2);
      const d3 = d*d*d;
      const factor = Q / (4 * Math.PI * d3);
      return [
        factor * dx,
        factor * dy,
        factor * dz
      ];
    };

    // --- Place Singularity Distributions ---
    const doublets = [];
    const zCenter = cz;

    // A. Chassis doublets (representing nose, sidepods, and engine block blockage)
    doublets.push({ cx: min[0] + span[0] * 0.15, cy: min[1] + span[1] * 0.22, cz: zCenter, R: span[2] * 0.26 });
    doublets.push({ cx: min[0] + span[0] * 0.40, cy: min[1] + span[1] * 0.35, cz: zCenter, R: span[2] * 0.38 });
    doublets.push({ cx: min[0] + span[0] * 0.65, cy: min[1] + span[1] * 0.36, cz: zCenter, R: span[2] * 0.44 });
    doublets.push({ cx: min[0] + span[0] * 0.85, cy: min[1] + span[1] * 0.32, cz: zCenter, R: span[2] * 0.28 });

    // B. Wheel doublets (representing the 4 tires)
    const xFrontW = min[0] + span[0] * 0.22;
    const yFrontW = min[1] + span[1] * 0.25;
    const zFrontWL = min[2] + span[2] * 0.12;
    const zFrontWR = max[2] - span[2] * 0.12;
    const rFrontW = span[1] * 0.33;

    const xRearW = min[0] + span[0] * 0.80;
    const yRearW = min[1] + span[1] * 0.28;
    const zRearWL = min[2] + span[2] * 0.08;
    const zRearWR = max[2] - span[2] * 0.08;
    const rRearW = span[1] * 0.36;

    doublets.push({ cx: xFrontW, cy: yFrontW, cz: zFrontWL, R: rFrontW });
    doublets.push({ cx: xFrontW, cy: yFrontW, cz: zFrontWR, R: rFrontW });
    doublets.push({ cx: xRearW, cy: yRearW, cz: zRearWL, R: rRearW });
    doublets.push({ cx: xRearW, cy: yRearW, cz: zRearWR, R: rRearW });

    // Sum doublet perturbations
    doublets.forEach((d) => {
      const [px, py, pz] = getDoubletPert(x - d.cx, y - d.cy, z - d.cz, d.R, wx, wy, wz);
      vx += px;
      vy += py;
      vz += pz;
    });

    // C. Wing Vortices (front and rear wing downforce)
    const fwAngle = ((config?.front_wing_main?.angle_deg || 2) + (config?.front_wing_flap?.angle_deg || 8)) / 2;
    const gammaFW = -fwAngle * 0.15 * Math.sqrt(wx*wx + wy*wy + wz*wz);
    const xFW = min[0] + span[0] * 0.04;
    const yFW = min[1] + span[1] * 0.14;
    const spanFW = span[2] * 0.95;

    const rwAngle = ((config?.rear_wing_main?.angle_deg || 5) + (config?.rear_wing_flap?.angle_deg || 15)) / 2;
    const gammaRW = -rwAngle * 0.18 * Math.sqrt(wx*wx + wy*wy + wz*wz);
    const xRW = min[0] + span[0] * 0.88;
    const yRW = min[1] + span[1] * 0.74;
    const spanRW = span[2] * 0.55;

    // Apply Front Wing Vortex
    const [fwVx, fwVy] = getVortexPert(x - xFW, y - yFW, z - zCenter, spanFW / 2, gammaFW);
    vx += fwVx;
    vy += fwVy;

    // Apply Rear Wing Vortex
    const [rwVx, rwVy] = getVortexPert(x - xRW, y - yRW, z - zCenter, spanRW / 2, gammaRW);
    vx += rwVx;
    vy += rwVy;

    // D. Floor Sink & Diffuser Source (underbody suction / Venturi acceleration)
    const frh = config?.ride_height_front_mm || 25;
    const geStrength = Math.exp(-frh / 1000 / 0.015) * 0.45 * Math.sqrt(wx*wx + wy*wy + wz*wz);

    const xSink = min[0] + span[0] * 0.45;
    const qSink = -geStrength * span[0] * span[2] * 0.85;

    const xSource = min[0] + span[0] * 0.82;
    const ySource = min[1] + span[1] * 0.15;
    const qSource = geStrength * span[0] * span[2] * 0.85;

    // Apply floor sink
    const [sinkVx, sinkVy, sinkVz] = getSourcePert(x - xSink, y - min[1], z - zCenter, qSink);
    vx += sinkVx;
    vy += sinkVy;
    vz += sinkVz;

    // Apply diffuser source
    const [srcVx, srcVy, srcVz] = getSourcePert(x - xSource, y - ySource, z - zCenter, qSource);
    vx += srcVx;
    vy += srcVy;
    vz += srcVz;

    // Singularities grow as 1/d^2 / 1/d^3, so inside a core the perturbation
    // reaches hundreds of m/s and tracers skip clean through the car. Cap it at
    // a plausible attached-flow overshoot to keep |v| near 1.5 U.
    const uRef = Math.sqrt(wx * wx + wy * wy + wz * wz) || 1;
    const px = vx - wx;
    const py = vy - wy;
    const pz = vz - wz;
    const pMag = Math.sqrt(px * px + py * py + pz * pz);
    // Cap at a plausible attached-flow overshoot. 1.15 U let singularities hand
    // tracers 2.3 U near the body cores; 0.85 keeps local peaks realistic.
    const pCap = 0.85 * uRef;
    if (pMag > pCap) {
      const k = pCap / pMag;
      vx = wx + px * k;
      vy = wy + py * k;
      vz = wz + pz * k;
    }
  }

  const uRef = Math.sqrt(wx * wx + wy * wy + wz * wz) || 1;
  const nx = (x - min[0]) / span[0];

  if (flow.showInteraction !== false) {
    // ---- Mean-flow structure behind the car -----------------------------
    // A car leaves a momentum deficit: smoke there drifts slower than
    // freestream, and the rear wing pushes it down. Without this the wake
    // reads as freestream plus noise, which is the main "not real" tell.
    const wakeX0 = max[0] - span[0] * 0.05; // just before the tail
    if (x > wakeX0) {
      const dxn = (x - wakeX0) / span[0];
      const decay = Math.exp(-dxn / 1.1);
      const yWake = min[1] + span[1] * 0.42;
      const sigma = span[1] * (0.24 + 0.42 * dxn);
      const g = Math.exp(-((y - yWake) ** 2) / (2 * sigma * sigma));
      vx *= 1 - 0.4 * decay * g;
      vy -= 0.1 * uRef * decay * g;
    }

    // ---- Trailing wingtip vortices --------------------------------------
    // Counter-rotating pairs roll the wake into the two filaments a real
    // tunnel shows. Sign convention: negative on the +z side, positive on
    // -z, which makes the flow between the tips go down (downwash).
    const rwMain = config?.rear_wing_main?.angle_deg ?? 5;
    const rwFlap = config?.rear_wing_flap?.angle_deg ?? 15;
    const fwMain = config?.front_wing_main?.angle_deg ?? 2;
    const fwFlap = config?.front_wing_flap?.angle_deg ?? 8;
    const rearLoad = Math.min(1.6, 0.4 + (rwMain + rwFlap) / 30);
    const frontLoad = Math.min(1.4, 0.35 + (fwMain + fwFlap) / 12);
    const tipZ = cz;

    const vortexSites = [
      // x0, y0, z offset, span half, core, decay, gamma magnitude
      [min[0] + span[0] * 0.06, min[1] + span[1] * 0.14, span[2] * 0.47,
        span[1] * 0.045, span[0] * 1.7, 0.018 * uRef * span[2] * frontLoad],
      [min[0] + span[0] * 0.9, min[1] + span[1] * 0.74, span[2] * 0.27,
        span[1] * 0.06, span[0] * 1.2, 0.034 * uRef * span[2] * rearLoad],
    ];
    for (const [x0, y0, dz, core, decayLen, gammaMag] of vortexSites) {
      for (const side of [1, -1]) {
        const gamma = side > 0 ? -gammaMag : gammaMag;
        if (addLambOseen(x, y, z, x0, y0, tipZ + side * dz, gamma, core, decayLen)) {
          vx += _vortexOut[0];
          vy += _vortexOut[1];
          vz += _vortexOut[2];
        }
      }
    }
  }

  // ---- Coherent turbulence ------------------------------------------------
  // Divergence-free curl noise, amplified where the tunnel is actually
  // turbulent (the wake) and scaled by the turbulence slider. This replaces
  // per-frame white jitter: eddies persist in space and evolve slowly.
  const turbLevel = flow.turbulence ?? 0.12;
  if (turbLevel > 1e-3) {
    const yWake = min[1] + span[1] * 0.42;
    const sigma = span[1] * (0.3 + 0.5 * Math.max(0, nx - 0.6));
    const behind = Math.max(0, Math.min(1, (nx - 0.45) / 0.4));
    const inWake = behind * Math.exp(-((y - yWake) ** 2) / (2 * sigma * sigma));
    const eddy = Math.max(span[0] * 0.22, 0.05);
    const c = curlNoise(
      (x - min[0]) / eddy,
      (y - min[1]) / eddy,
      (z - min[2]) / eddy + 17,
      (flow.time || 0) * 1.6,
    );
    const gain = turbLevel * uRef * (0.55 + 2.4 * inWake);
    let tx = c[0] * gain;
    let ty = c[1] * gain;
    let tz = c[2] * gain;
    const tm = Math.sqrt(tx * tx + ty * ty + tz * tz);
    const cap = gain * 1.6;
    if (tm > cap) {
      const s = cap / tm;
      tx *= s;
      ty *= s;
      tz *= s;
    }
    vx += tx;
    vy += ty;
    vz += tz;
  }

  // ---- Floor boundary layer ----------------------------------------------
  // The tunnel floor is a no-slip wall: entrained smoke crawls within the
  // boundary layer and grades back to freestream above it. This is the cue
  // that makes smoke rakes near the ground look right.
  const yRel = y - min[1];
  const blH = span[1] * 0.18;
  if (yRel < blH && yRel > -0.05) {
    const eta = Math.max(0, Math.min(1, yRel / blH));
    const profile = Math.max(0.18, Math.pow(eta, 1 / 7));
    vx *= profile;
    vy *= profile;
    vz *= profile;
  }

  // Ground plane boundary condition — no penetration
  if (y <= min[1] + 0.01 && vy < 0) vy = Math.abs(vy) * 0.15;

  return [vx, vy, vz];
}

export function speedAt(x, y, z, config, bounds, flow) {
  return len(velocityAt(x, y, z, config, bounds, flow));
}

/** Spawn particle at flow source. */

// Monotonic serial used by the seeding sequences below. Respawning a tracer
// must not put it back on exactly the same line (that leaves fixed streaks);
// rotating the low-discrepancy offset keeps coverage even over time.
let spawnSerial = 0;

/**
 * R2 low-discrepancy pair in [0, 1)^2 (Roberts' sequence). Compared with
 * independent random draws it fills the inlet plane far more evenly, which is
 * what makes the smoke curtain look continuous rather than blotchy.
 */
function stratified2(i) {
  const t = i + spawnSerial * 0.6180339887;
  return [(t * 0.7548776662) % 1, (t * 0.5698402910) % 1];
}

export function spawnOneParticle(bounds, flow, index = 0) {
  const min = bounds.min;
  const max = bounds.max;
  const span = max.map((v, i) => v - min[i]);
  const src = flow.flowSource || 'front_grid';
  let x, y, z;
  spawnSerial += 1;

  switch (src) {
    case 'smoke_rake':
      {
        // Classic tunnel rake: five heights x nine lateral tubes, entering
        // just upstream of the model. Thin filaments read best with many
        // closely spaced tubes and very little jitter.
        const rows = 5;
        const cols = 9;
        // Advance tube by tube on every spawn. Using index + serial here made
        // both counters step together, so only alternate tubes were ever fed.
        const s = spawnSerial % (rows * cols);
        const row = Math.floor(s / cols);
        const col = s % cols;
        const fy = [0.05, 0.16, 0.3, 0.48, 0.72][row];
        x = min[0] - span[0] * (0.04 + Math.random() * 0.02);
        y = min[1] + span[1] * fy + (Math.random() - 0.5) * 0.006 * span[1];
        z = min[2] + span[2] * ((col + 0.5) / cols) + (Math.random() - 0.5) * 0.006 * span[2];
      }
      break;
    case 'chamber':
      {
        // Surround source: front curtain plus a top sheet and both side
        // sheets, so the working section fills volumetrically instead of
        // showing one sparse stream.
        const slot = spawnSerial % 10;
        if (slot < 5) {
          const [u, v] = stratified2(index);
          x = min[0] - span[0] * (0.02 + Math.random() * 0.06);
          y = min[1] + span[1] * (0.04 + 0.92 * u);
          z = min[2] + span[2] * (0.04 + 0.92 * v);
        } else if (slot < 7) {
          const [u, v] = stratified2(index + 137);
          x = min[0] + span[0] * u;
          y = max[1] + span[1] * (0.1 + Math.random() * 0.18);
          z = min[2] + span[2] * (0.04 + 0.92 * v);
        } else if (slot < 9) {
          const [u, v] = stratified2(index + 271);
          x = min[0] + span[0] * u;
          y = min[1] + span[1] * (0.05 + 0.9 * v);
          z = min[2] - span[2] * (0.12 + Math.random() * 0.2);
        } else {
          const [u, v] = stratified2(index + 409);
          x = min[0] + span[0] * u;
          y = min[1] + span[1] * (0.05 + 0.9 * v);
          z = max[2] + span[2] * (0.12 + Math.random() * 0.2);
        }
      }
      break;
    case 'smoke_wand':
      {
        x = min[0] - span[0] * 0.05;
        const wy = flow.wandY !== undefined ? flow.wandY : (min[1] + max[1]) / 2;
        const wz = flow.wandZ !== undefined ? flow.wandZ : (min[2] + max[2]) / 2;
        x += (Math.random() - 0.5) * 0.015;
        y = wy + (Math.random() - 0.5) * 0.015;
        z = wz + (Math.random() - 0.5) * 0.015;
      }
      break;
    case 'top':
      {
        const [u, v] = stratified2(index);
        x = min[0] + span[0] * u;
        y = max[1] + span[1] * 0.35;
        z = min[2] + span[2] * (0.04 + 0.92 * v);
      }
      break;
    case 'side_left':
      {
        const [u, v] = stratified2(index);
        x = min[0] + span[0] * u;
        y = min[1] + span[1] * (0.04 + 0.92 * v);
        z = min[2] - span[2] * 0.4;
      }
      break;
    case 'side_right':
      {
        const [u, v] = stratified2(index + 61);
        x = min[0] + span[0] * u;
        y = min[1] + span[1] * (0.04 + 0.92 * v);
        z = max[2] + span[2] * 0.4;
      }
      break;
    case 'ring':
      {
        const ang = (index / (flow.particleCount || 300)) * Math.PI * 2;
        const r = Math.max(span[1], span[2]) * 0.9;
        x = min[0] - span[0] * 0.5;
        y = (min[1] + max[1]) / 2 + Math.cos(ang) * r * 0.5;
        z = (min[2] + max[2]) / 2 + Math.sin(ang) * r * 0.5;
      }
      break;
    case 'front_wide':
      {
        const [u, v] = stratified2(index + 53);
        x = min[0] - span[0] * (0.02 + Math.random() * 0.06);
        y = min[1] + span[1] * (0.04 + 0.92 * u);
        z = min[2] + span[2] * (0.04 + 0.92 * v);
      }
      break;
    case 'front_grid':
    default:
      {
        // Stratified curtain: the R2 sequence tiles the inlet plane evenly, so
        // the smoke enters as a continuous sheet instead of random clumps.
        const [u, v] = stratified2(index);
        x = min[0] - span[0] * (0.01 + Math.random() * 0.08);
        y = min[1] + span[1] * (0.06 + 0.88 * u);
        z = min[2] + span[2] * (0.05 + 0.9 * v);
      }
  }

  return { x, y, z, life: Math.random() * 2, hit: 0 };
}

export function spawnParticles(bounds, flow) {
  const count = flow.particleCount || 300;
  return Array.from({ length: count }, (_, i) => spawnOneParticle(bounds, flow, i));
}

export function generateStreamlines(config, bounds, flow, velocityFn = velocityAt) {
  const min = bounds.min;
  const max = bounds.max;
  const span = max.map((v, i) => v - min[i]);
  const nLines = flow.streamlineCount || 24;
  const lines = [];
  const layers = 4;

  for (let i = 0; i < nLines; i++) {
    for (let j = 0; j < layers; j++) {
      const t = (i + 0.5) / nLines;
      const zLayer = 0.06 + (j / (layers - 1)) * 0.88;
      let x, y, z;

      if (flow.flowSource === 'top') {
        x = min[0] + span[0] * t;
        y = max[1] + span[1] * 0.2;
        z = min[2] + span[2] * zLayer;
      } else if (flow.flowSource === 'side_left') {
        x = min[0] + span[0] * zLayer;
        y = min[1] + span[1] * t;
        z = min[2] - span[2] * 0.3;
      } else {
        x = min[0] - span[0] * 0.08;
        y = min[1] + span[1] * t;
        z = min[2] + span[2] * zLayer;
      }

      const points = [[x, y, z]];
      for (let step = 0; step < 120; step++) {
        const [vx, vy, vz] = velocityFn(x, y, z, config, bounds, flow);
        const vmag = len([vx, vy, vz]) || 1;
        const dt = (span[0] * 0.028) / vmag;
        x += vx * dt;
        y += vy * dt;
        z += vz * dt;
        if (x > max[0] + span[0] * 1.2 && flow.flowSource !== 'side_left') break;
        if (y > max[1] + span[1] && flow.flowSource === 'top') break;
        if (z > max[2] + span[2] * 1.5 && flow.flowSource === 'side_left') break;
        points.push([x, y, z]);
      }
      if (points.length > 4) lines.push(points);
    }
  }
  return lines;
}

/** Map local speed to color — warm = slow (entrance/stagnation), cool = fast (exit/accelerated). */
export function speedToColor(speed, refSpeed, minSpeed = null, maxSpeed = null) {
  const ref = Math.max(refSpeed, 1);
  let t;
  if (minSpeed != null && maxSpeed != null && maxSpeed > minSpeed + 0.5) {
    t = clamp((speed - minSpeed) / (maxSpeed - minSpeed), 0, 1);
  } else {
    t = clamp(speed / ref, 0, 1.25);
    t = clamp(t / 1.25, 0, 1);
  }

  // Orange/red (slow) → amber → yellow → cyan → blue → white (fast)
  if (t < 0.18) {
    const s = t / 0.18;
    return { r: 0.85 + s * 0.1, g: 0.22 + s * 0.28, b: 0.08 + s * 0.08 };
  }
  if (t < 0.38) {
    const s = (t - 0.18) / 0.2;
    return { r: 0.95 - s * 0.15, g: 0.5 + s * 0.35, b: 0.16 + s * 0.1 };
  }
  if (t < 0.58) {
    const s = (t - 0.38) / 0.2;
    return { r: 0.8 - s * 0.55, g: 0.85 + s * 0.05, b: 0.26 + s * 0.45 };
  }
  if (t < 0.78) {
    const s = (t - 0.58) / 0.2;
    return { r: 0.25 - s * 0.15, g: 0.9 - s * 0.15, b: 0.71 + s * 0.22 };
  }
  const s = (t - 0.78) / 0.22;
  return { r: 0.1 + s * 0.55, g: 0.75 + s * 0.2, b: 0.93 + s * 0.07 };
}

/** Sample speed along a polyline for streamline coloring. */
export function speedsAlongPath(points, config, bounds, flow, velocityFn = velocityAt) {
  return points.map(([x, y, z]) => len(velocityFn(x, y, z, config, bounds, flow)));
}

export function reynoldsNumber(flow, charLengthM = 3.5) {
  const v = flow.windSpeedKmh / 3.6;
  const rho = 1.225;
  const mu = 1.81e-5;
  return Math.round((rho * v * charLengthM) / mu);
}
