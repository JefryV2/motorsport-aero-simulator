/** Nearest-point queries on car mesh for shape-aware airflow. */

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function len(v) {
  return Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
}

function normalize(v) {
  const l = len(v) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function tangentAlongSurface(normal, wind) {
  const w = normalize(wind);
  const n = normalize(normal);
  const t = [w[0] - dot(w, n) * n[0], w[1] - dot(w, n) * n[1], w[2] - dot(w, n) * n[2]];
  const tl = len(t);
  if (tl < 1e-4) return w;
  return [t[0] / tl, t[1] / tl, t[2] / tl];
}

export class MeshFlowSampler {
  constructor(vertices, normals, bounds, cellCount = 24) {
    this.vertices = vertices;
    this.normals = normals;
    this.bounds = bounds;
    this.cellCount = cellCount;
    this._buildGrid();
  }

  _buildGrid() {
    const min = this.bounds.min;
    const max = this.bounds.max;
    this.span = max.map((v, i) => Math.max(v - min[i], 1e-6));
    this.grid = new Map();

    for (let i = 0; i < this.vertices.length; i++) {
      const v = this.vertices[i];
      const key = this._cellKey(v[0], v[1], v[2]);
      if (!this.grid.has(key)) this.grid.set(key, []);
      this.grid.get(key).push(i);
    }
  }

  _cellKey(x, y, z) {
    const min = this.bounds.min;
    const ix = Math.min(this.cellCount - 1, Math.max(0, Math.floor(((x - min[0]) / this.span[0]) * this.cellCount)));
    const iy = Math.min(this.cellCount - 1, Math.max(0, Math.floor(((y - min[1]) / this.span[1]) * this.cellCount)));
    const iz = Math.min(this.cellCount - 1, Math.max(0, Math.floor(((z - min[2]) / this.span[2]) * this.cellCount)));
    return `${ix},${iy},${iz}`;
  }

  /** Find nearest mesh vertex + normal within search radius. */
  nearest(x, y, z, maxDist = Infinity, cellRadius = 2) {
    const candidates = new Set();
    const [ix, iy, iz] = this._cellKey(x, y, z).split(',').map(Number);

    for (let dx = -cellRadius; dx <= cellRadius; dx++) {
      for (let dy = -cellRadius; dy <= cellRadius; dy++) {
        for (let dz = -cellRadius; dz <= cellRadius; dz++) {
          const cx = ix + dx;
          const cy = iy + dy;
          const cz = iz + dz;
          if (cx < 0 || cy < 0 || cz < 0 || cx >= this.cellCount || cy >= this.cellCount || cz >= this.cellCount) continue;
          const bucket = this.grid.get(`${cx},${cy},${cz}`);
          if (bucket) bucket.forEach((idx) => candidates.add(idx));
        }
      }
    }

    let bestDist = maxDist;
    let best = null;

    for (const idx of candidates) {
      const v = this.vertices[idx];
      const d = len(sub([x, y, z], v));
      if (d < bestDist) {
        bestDist = d;
        best = {
          point: v,
          normal: this.normals[idx] || [0, 1, 0],
          dist: d,
          index: idx,
        };
      }
    }

    return best;
  }

  /** Pick a vertex near a given X (length) position — for nose-to-tail seeding. */
  pickVertexNearX(targetX, seed = 0) {
    let best = seed % this.vertices.length;
    let bestD = Infinity;
    const step = Math.max(1, Math.floor(this.vertices.length / 800));
    for (let i = 0; i < this.vertices.length; i += step) {
      const d = Math.abs(this.vertices[i][0] - targetX);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }
}

export function surfaceOffset(bounds) {
  const span = bounds.max.map((v, i) => Math.max(v - bounds.min[i], 0.01));
  return Math.max(span[0], span[1], span[2]) * 0.018;
}

/** Snap point to just outside nearest surface. Returns [x,y,z,normal] or null. */
export function snapToSurface(x, y, z, sampler, bounds, eps = null) {
  if (!sampler) return null;
  const span = bounds.max.map((v, i) => Math.max(v - bounds.min[i], 0.01));
  const influence = Math.max(span[0], span[1], span[2]) * 0.55;
  const hit = sampler.nearest(x, y, z, influence);
  if (!hit) return null;

  const gap = eps ?? surfaceOffset(bounds);
  const n = normalize(hit.normal);
  return {
    x: hit.point[0] + n[0] * gap,
    y: hit.point[1] + n[1] * gap,
    z: hit.point[2] + n[2] * gap,
    normal: n,
    dist: hit.dist,
    index: hit.index,
  };
}

/**
 * Shape-aware velocity: deflect around body, slide along surface to the rear.
 */
export function velocityAtMesh(x, y, z, sampler, baseVelocityFn, config, bounds, flow) {
  const [wx, wy, wz] = baseVelocityFn(x, y, z, config, bounds, flow);
  if (!sampler || flow.showInteraction === false) return [wx, wy, wz];

  const span = bounds.max.map((v, i) => Math.max(v - bounds.min[i], 0.01));
  const influence = Math.max(span[0], span[1], span[2]) * 0.55;
  const hit = sampler.nearest(x, y, z, influence);
  if (!hit) return [wx, wy, wz];

  const n = normalize(hit.normal);
  const wind = normalize([wx, wy, wz]);
  const speed = len([wx, wy, wz]) || 1;
  const offset = sub([x, y, z], hit.point);
  const signedDist = dot(offset, n);
  const proximity = Math.max(0, 1 - hit.dist / influence);
  const slide = tangentAlongSurface(n, wind);

  // Block penetration through body
  let vx = wx;
  let vy = wy;
  let vz = wz;
  const vn = dot([vx, vy, vz], n);
  if (signedDist < influence * 0.12 && vn < 0) {
    vx -= vn * n[0];
    vy -= vn * n[1];
    vz -= vn * n[2];
  }

  // Near surface: flow slides along skin toward the tail (strong Coanda-style slip)
  const skin = proximity ** 1.2;
  const nearSkin = hit.dist < influence * 0.22 || signedDist < influence * 0.18;
  if (nearSkin) {
    const blend = 0.35 + skin * 0.55;
    vx = vx * (1 - blend) + slide[0] * speed * blend;
    vy = vy * (1 - blend) + slide[1] * speed * blend;
    vz = vz * (1 - blend) + slide[2] * speed * blend;

    // Keep minimum forward progress along car length (wind +X)
    const minFwd = speed * 0.42 * skin;
    if (vx < minFwd) vx = minFwd + slide[0] * speed * 0.25;

    // Lift slightly over wings / bonnet
    if (n[1] > 0.15 && x < bounds.min[0] + span[0] * 0.55) {
      vy += skin * speed * 0.08;
    }
    // Underbody acceleration
    if (n[1] < -0.1 && y < bounds.min[1] + span[1] * 0.35) {
      vx += skin * speed * 0.15;
    }
  } else if (proximity > 0) {
    // Outer field: gentle deflection away from body
    const push = proximity * 0.35;
    vx += slide[0] * speed * push;
    vy += slide[1] * speed * push * 0.5;
    vz += slide[2] * speed * push;
  }

  return [vx, vy, vz];
}

/** Streamlines seeded on the car skin, integrated to the rear. */
export function generateSurfaceStreamlines(sampler, config, bounds, flow, baseVelocityFn, count = 24) {
  if (!sampler) return [];
  const lines = [];
  const min = bounds.min;
  const max = bounds.max;
  const span = max.map((v, i) => Math.max(v - bounds.min[i], 0.01));
  const eps = surfaceOffset(bounds);
  const wind = normalize(baseVelocityFn(min[0] - 1, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2, config, bounds, flow));

  const seeds = [];
  const step = Math.max(1, Math.floor(sampler.vertices.length / (count * 3)));
  for (let i = 0; i < sampler.vertices.length; i += step) {
    const n = normalize(sampler.normals[i] || [0, 1, 0]);
    if (dot(n, wind) < -0.05) seeds.push(i);
    if (seeds.length >= count * 2) break;
  }
  if (seeds.length < count) {
    for (let t = 0; t < count; t++) {
      const tx = min[0] + span[0] * (0.02 + (t / count) * 0.96);
      seeds.push(sampler.pickVertexNearX(tx, t * 131));
    }
  }

  const used = new Set();
  for (let si = 0; si < seeds.length && lines.length < count; si++) {
    const idx = seeds[si % seeds.length];
    if (used.has(idx)) continue;
    used.add(idx);

    const v = sampler.vertices[idx];
    const n = normalize(sampler.normals[idx] || [0, 1, 0]);
    let x = v[0] + n[0] * eps;
    let y = v[1] + n[1] * eps;
    let z = v[2] + n[2] * eps;
    const pts = [[x, y, z]];

    for (let s = 0; s < 140; s++) {
      const [vx, vy, vz] = velocityAtMesh(x, y, z, sampler, baseVelocityFn, config, bounds, flow);
      const vmag = len([vx, vy, vz]) || 1;
      const dt = (span[0] * 0.018) / vmag;
      x += vx * dt;
      y += vy * dt;
      z += vz * dt;

      const snapped = snapToSurface(x, y, z, sampler, bounds, eps);
      if (snapped && snapped.dist < eps * 3) {
        x = snapped.x;
        y = snapped.y;
        z = snapped.z;
      }

      if (x > max[0] + span[0] * 0.35) break;
      if (y < min[1] - eps * 2) break;
      pts.push([x, y, z]);
    }
    if (pts.length > 8 && pts[pts.length - 1][0] > min[0] + span[0] * 0.25) {
      lines.push(pts);
    }
  }
  return lines;
}

export function spawnSurfaceParticle(sampler, bounds, flow, index) {
  const min = bounds.min;
  const max = bounds.max;
  const span = max.map((v, i) => Math.max(v - bounds.min[i], 0.01));
  const eps = surfaceOffset(bounds) * (1 + (index % 4) * 0.15);

  let i;
  if (index % 3 !== 2) {
    const t = (index % 40) / 40;
    const targetX = min[0] + span[0] * (0.02 + t * 0.85);
    i = sampler.pickVertexNearX(targetX, index);
  } else {
    i = (index * 9973) % sampler.vertices.length;
  }

  const v = sampler.vertices[i];
  const n = normalize(sampler.normals[i] || [0, 1, 0]);
  return {
    x: v[0] + n[0] * eps,
    y: v[1] + n[1] * eps,
    z: v[2] + n[2] * eps,
    life: 0,
    hit: 0,
    surface: true,
    seedX: v[0],
  };
}
