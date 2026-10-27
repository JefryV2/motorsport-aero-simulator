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
      const t1 = -R3 / (2 * d3);
      const t2 = (3 * R3 * uDotD) / (2 * d5);
      
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
  }

  // Ground plane boundary condition — no penetration
  if (y <= min[1] + 0.01 && vy < 0) vy = Math.abs(vy) * 0.15;

  // Add random turbulence scale based on wake position (downstream behind car)
  let turbScale = flow.turbulence || 0.12;
  const dist = distToBBox(x, y, z, bounds);
  const nx = (x - min[0]) / span[0];
  if (nx > 0.6 && dist < span[0] * 0.5) {
    turbScale *= 4.0 * nx;
  }

  const t = turbScale * 1.5;
  vx += (Math.random() - 0.5) * t;
  vy += (Math.random() - 0.5) * t;
  vz += (Math.random() - 0.5) * t;

  return [vx, vy, vz];
}

export function speedAt(x, y, z, config, bounds, flow) {
  return len(velocityAt(x, y, z, config, bounds, flow));
}

/** Spawn particle at flow source. */
export function spawnOneParticle(bounds, flow, index = 0) {
  const min = bounds.min;
  const max = bounds.max;
  const span = max.map((v, i) => v - min[i]);
  const src = flow.flowSource || 'front_grid';
  let x, y, z;

  switch (src) {
    case 'smoke_rake':
      {
        const nStreams = 7;
        const streamIdx = index % nStreams;
        x = min[0] - span[0] * 0.05;
        const yCenter = min[1] + span[1] * 0.4;
        const zCenter = min[2] + span[2] * 0.5;
        
        let sy, sz;
        if (streamIdx === 0) { sy = min[1] + span[1] * 0.12; sz = zCenter; }
        else if (streamIdx === 1) { sy = min[1] + span[1] * 0.35; sz = zCenter; }
        else if (streamIdx === 2) { sy = min[1] + span[1] * 0.65; sz = zCenter; }
        else if (streamIdx === 3) { sy = min[1] + span[1] * 0.12; sz = min[2] + span[2] * 0.22; }
        else if (streamIdx === 4) { sy = min[1] + span[1] * 0.12; sz = min[2] + span[2] * 0.78; }
        else if (streamIdx === 5) { sy = min[1] + span[1] * 0.32; sz = min[2] + span[2] * 0.25; }
        else { sy = min[1] + span[1] * 0.32; sz = min[2] + span[2] * 0.75; }
        
        x += (Math.random() - 0.5) * 0.02;
        y = sy + (Math.random() - 0.5) * 0.025 * span[1];
        z = sz + (Math.random() - 0.5) * 0.025 * span[2];
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
      x = min[0] + span[0] * Math.random();
      y = max[1] + span[1] * 0.35;
      z = min[2] + span[2] * Math.random();
      break;
    case 'side_left':
      x = min[0] + span[0] * Math.random();
      y = min[1] + span[1] * Math.random();
      z = min[2] - span[2] * 0.4;
      break;
    case 'side_right':
      x = min[0] + span[0] * Math.random();
      y = min[1] + span[1] * Math.random();
      z = max[2] + span[2] * 0.4;
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
      x = min[0] - span[0] * (0.02 + Math.random() * 0.06);
      y = min[1] + span[1] * (0.05 + (Math.floor(index / 10) % 8) * 0.12);
      z = min[2] + span[2] * (0.05 + (index % 10) * 0.095);
      break;
    case 'front_grid':
    default:
      x = min[0] - span[0] * (0.01 + Math.random() * 0.08);
      y = min[1] + span[1] * (0.08 + Math.random() * 0.82);
      z = min[2] + span[2] * (0.05 + Math.random() * 0.88);
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
