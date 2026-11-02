/**
 * Verifies the analytic flow field behind the live smoke visualization.
 *
 * The UI calls this field every frame, so a regression here is instantly
 * visible (static jitter, exploding tracers, flat smoke near the floor). This
 * script is the cheap, dependency-free gate:
 *
 *   1. finite field, bounded perturbation magnitude;
 *   2. divergence-free turbulence (the old white-noise jitter was not);
 *   3. spatial coherence - neighbouring samples move together;
 *   4. streamline integration stays finite and progresses downstream;
 *   5. floor boundary layer slower than freestream, wake behind the car slower.
 *
 * Run: node scripts/check_flow_field.mjs
 *
 * The field lives in a browser ES module but the repo has no package.json, so
 * Node would treat frontend/js/flowEngine.js as CommonJS. We copy it to a
 * temporary .mjs and import that copy instead of changing the source layout.
 */

import { copyFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const here = new URL('../frontend/js/flowEngine.js', import.meta.url);
const tmpDir = mkdtempSync(join(tmpdir(), 'flowcheck-'));
const tmpFile = join(tmpDir, 'flowEngine.mjs');
copyFileSync(here, tmpFile);

let mod;
try {
  mod = await import(pathToFileURL(tmpFile).href);
} finally {
  rmSync(tmpDir, { recursive: true, force: true });
}

const { velocityAt, curlNoise, generateStreamlines, spawnOneParticle } = mod;

const bounds = { min: [-1.75, 0, -0.5], max: [1.75, 0.6, 0.5] };
const baseFlow = {
  windSpeedKmh: 200,
  yawDeg: 0,
  pitchDeg: 0,
  turbulence: 0.12,
  flowSource: 'smoke_rake',
  showInteraction: true,
  time: 0,
};
const config = {
  front_wing_main: { angle_deg: 2 },
  front_wing_flap: { angle_deg: 6 },
  rear_wing_main: { angle_deg: 8 },
  rear_wing_flap: { angle_deg: 12 },
  ride_height_front_mm: 100,
};
const U = 200 / 3.6;
const failures = [];
const check = (ok, label, detail) => {
  if (!ok) failures.push(`${label} (${detail})`);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}  ${detail}`);
};

// ---------------------------------------------------------------------------
// 1. Finite field and bounded perturbation
// ---------------------------------------------------------------------------
const speeds = [];
let nonFinite = 0;
for (let i = 0; i <= 24; i++) {
  for (let j = 0; j <= 12; j++) {
    for (let k = 0; k <= 12; k++) {
      const x = bounds.min[0] - 0.2 + ((bounds.max[0] - bounds.min[0] + 0.9) * i) / 24;
      const y = bounds.min[1] + ((bounds.max[1] - bounds.min[1]) * j) / 12;
      const z = bounds.min[2] + ((bounds.max[2] - bounds.min[2]) * k) / 12;
      const v = velocityAt(x, y, z, config, bounds, baseFlow);
      if (!v.every(Number.isFinite)) nonFinite++;
      speeds.push(Math.hypot(v[0], v[1], v[2]));
    }
  }
}
const meanSpeed = speeds.reduce((a, b) => a + b, 0) / speeds.length;
const maxSpeed = Math.max(...speeds);
check(nonFinite === 0, 'field finite', `${nonFinite} non-finite samples of ${speeds.length}`);
check(maxSpeed < 2.2 * U, 'bounded perturbation', `max |v| = ${maxSpeed.toFixed(1)} m/s (${(maxSpeed / U).toFixed(2)} U)`);
check(meanSpeed > 0.85 * U && meanSpeed < 1.35 * U, 'plausible mean speed', `mean = ${meanSpeed.toFixed(1)} m/s (${(meanSpeed / U).toFixed(2)} U)`);

// ---------------------------------------------------------------------------
// 2. Divergence of the turbulence component
// ---------------------------------------------------------------------------
let divSum = 0;
let magSum = 0;
let n = 0;
const eddyM = 0.4; // metres per eddy length, matches the check below
const hLat = 0.01 / eddyM; // finite-difference step in lattice units
for (let i = 0; i < 40; i++) {
  const x = -1.2 + Math.random() * 3.2;
  const y = 0.02 + Math.random() * 0.55;
  const z = -0.45 + Math.random() * 0.9;
  const t = 0.5;
  const c = curlNoise(x / eddyM, y / eddyM, z / eddyM + 17, t);
  const cx1 = curlNoise((x + hLat * eddyM) / eddyM, y / eddyM, z / eddyM + 17, t);
  const cy1 = curlNoise(x / eddyM, (y + hLat * eddyM) / eddyM, z / eddyM + 17, t);
  const cz1 = curlNoise(x / eddyM, y / eddyM, (z + hLat * eddyM) / eddyM + 17, t);
  const div =
    (cx1[0] - c[0]) / hLat + (cy1[1] - c[1]) / hLat + (cz1[2] - c[2]) / hLat;
  divSum += Math.abs(div);
  magSum += Math.hypot(c[0], c[1], c[2]) / eddyM;
  n++;
}
const meanDiv = divSum / n;
const meanGrad = magSum / n;
check(
  meanDiv < 0.2 * meanGrad,
  'turbulence divergence-free',
  `|div v| = ${meanDiv.toFixed(3)} vs |grad| = ${meanGrad.toFixed(3)} (ratio ${(meanDiv / meanGrad).toFixed(3)})`,
);

// ---------------------------------------------------------------------------
// 3. Spatial coherence: nearby samples must move together
// ---------------------------------------------------------------------------
let coherenceSum = 0;
let coherenceMag = 0;
let m = 0;
for (let i = 0; i < 60; i++) {
  const x = -1.2 + Math.random() * 3.2;
  const y = 0.05 + Math.random() * 0.5;
  const z = -0.45 + Math.random() * 0.9;
  const a = velocityAt(x, y, z, config, bounds, baseFlow);
  const b = velocityAt(x + 0.02, y, z, config, bounds, baseFlow);
  coherenceSum += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  coherenceMag += Math.hypot(a[0], a[1], a[2]);
  m++;
}
const relJump = coherenceSum / coherenceMag;
check(relJump < 0.25, 'spatially coherent', `2 cm offset changes |v| by ${(relJump * 100).toFixed(1)}%`);

// ---------------------------------------------------------------------------
// 4. Streamlines integrate cleanly
// ---------------------------------------------------------------------------
const lines = generateStreamlines(config, bounds, { ...baseFlow, streamlineCount: 12 });
check(lines.length >= 4, 'streamlines generated', `${lines.length} lines`);
let badPoints = 0;
let progressed = 0;
for (const line of lines) {
  if (line[line.length - 1][0] > line[0][0] + 0.5) progressed++;
  for (const [x, y, z] of line) if (![x, y, z].every(Number.isFinite)) badPoints++;
}
check(badPoints === 0, 'streamlines finite', `${badPoints} non-finite points`);
check(progressed >= Math.ceil(lines.length * 0.5), 'streamlines progress downstream', `${progressed}/${lines.length}`);

// ---------------------------------------------------------------------------
// 5. Floor boundary layer and wake deficit
// ---------------------------------------------------------------------------
const floorMid = (bounds.min[2] + bounds.max[2]) / 2;
// Compare two heights *inside* the boundary layer: upstream blockage decelerates
// the freestream locally, so a single far-field reference is not meaningful.
const nearFloor = velocityAt(-1.74, 0.001, 0.45, config, bounds, baseFlow)[0];
const inLayer = velocityAt(-1.74, 0.09, 0.45, config, bounds, baseFlow)[0];
check(
  Math.abs(nearFloor) < 0.8 * Math.abs(inLayer),
  'floor boundary layer profile',
  `u(1 mm) = ${nearFloor.toFixed(1)} vs u(9 cm) = ${inLayer.toFixed(1)} m/s`,
);
check(
  Math.abs(nearFloor) < 0.9 * U,
  'floor velocity below freestream',
  `u(1 mm) = ${(Math.abs(nearFloor) / U).toFixed(2)} U`,
);

const wakeMid = velocityAt(2.3, 0.25, floorMid, config, bounds, baseFlow)[0];
const wakeAbove = velocityAt(2.3, 0.55, floorMid, config, bounds, baseFlow)[0];
check(
  wakeMid < 0.92 * U && wakeMid < 0.95 * Math.abs(wakeAbove),
  'wake momentum deficit',
  `u(wake) = ${wakeMid.toFixed(1)} m/s (${(wakeMid / U).toFixed(2)} U) vs u(above) = ${wakeAbove.toFixed(1)} m/s`,
);

// ---------------------------------------------------------------------------
// 6. Source seeding: tracers must tile the inflow, not clump
// ---------------------------------------------------------------------------
const spanY = bounds.max[1] - bounds.min[1];
const spanZ = bounds.max[2] - bounds.min[2];
const bins = 6;

function inletCoverage(source, samples = 400) {
  const flow = { ...baseFlow, flowSource: source };
  const seen = new Set();
  for (let i = 0; i < samples; i++) {
    const p = spawnOneParticle(bounds, flow, i);
    const by = Math.max(0, Math.min(bins - 1, Math.floor(((p.y - bounds.min[1]) / spanY) * bins)));
    const bz = Math.max(0, Math.min(bins - 1, Math.floor(((p.z - bounds.min[2]) / spanZ) * bins)));
    seen.add(`${by}:${bz}`);
  }
  return seen.size / (bins * bins);
}

for (const source of ['front_grid', 'front_wide', 'chamber']) {
  const cov = inletCoverage(source);
  check(cov >= 0.85, `inlet coverage: ${source}`, `${(cov * 100).toFixed(0)}% of ${bins}x${bins} bins`);
}

// The rake rotates through 4 heights x 6 lateral tubes; check the tubes exist.
const rakeRows = [0.03, 0.18, 0.44, 0.76];
const rakeTubes = new Set();
for (let i = 0; i < 480; i++) {
  const p = spawnOneParticle(bounds, { ...baseFlow, flowSource: 'smoke_rake' }, i);
  const fy = (p.y - bounds.min[1]) / spanY;
  const row = rakeRows.reduce((best, r, idx) => (Math.abs(r - fy) < Math.abs(rakeRows[best] - fy) ? idx : best), 0);
  const col = Math.max(0, Math.min(5, Math.floor(((p.z - bounds.min[2]) / spanZ) * 6)));
  rakeTubes.add(`${row}:${col}`);
}
check(rakeTubes.size >= 20, 'rake tube coverage', `${rakeTubes.size}/24 tubes seeded`);

// Surround source must actually feed all four faces.
const faces = new Set();
for (let i = 0; i < 480; i++) {
  const p = spawnOneParticle(bounds, { ...baseFlow, flowSource: 'chamber' }, i);
  if (p.x < bounds.min[0]) faces.add('front');
  if (p.y > bounds.max[1]) faces.add('top');
  if (p.z < bounds.min[2]) faces.add('left');
  if (p.z > bounds.max[2]) faces.add('right');
}
check(faces.size === 4, 'surround source feeds all faces', [...faces].join(', ') || 'none');

console.log('');
if (failures.length) {
  console.error(`FLOW FIELD CHECK FAILED (${failures.length}):`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log('Flow field check passed.');
