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
 *   5. floor boundary layer slower than freestream, wake behind the car slower;
 *   6. source seeding tiles the inflow, not clumps;
 *   7. the parametric built-in car: solved stations drive the singularities,
 *      the low-order lift terms still match the backend, and every setup knob
 *      that moves the body must move the smoke.
 *   8. skin pressure: each drawn element carries the pressure jump its own Cl
 *      says, faces the right way, and follows the car when it is yawed.
 *   9. the Reynolds readout: priced on the length the car is drawn at, and
 *      labelled with it.
 *  10. the inflow tilt: yaw and pitch move the tracers, leave |U| and the
 *      singularity strengths alone, and pitch up means air arriving from below.
 *
 * Run: node scripts/check_flow_field.mjs
 *
 * The field lives in browser ES modules but the repo has no package.json, so
 * Node would treat frontend/js/*.js as CommonJS. We copy the modules the
 * gate needs to a temporary .mjs and import those copies instead of changing
 * the source layout. None of them imports Three.js.
 */

import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const tmpDir = mkdtempSync(join(tmpdir(), 'flowcheck-'));
const copies = [
  ['../frontend/js/flowEngine.js', 'flowEngine.mjs'],
  ['../frontend/js/carSpec.js', 'carSpec.mjs'],
  ['../frontend/js/aeroField.js', 'aeroField.mjs'],
];
const urls = copies.map(([src, name]) => {
  const dest = join(tmpDir, name);
  copyFileSync(new URL(src, import.meta.url), dest);
  // The gate imports .mjs copies, so relative specifiers have to follow them.
  writeFileSync(dest, readFileSync(dest, 'utf8').replace(/(from '\.\/[\w-]+)\.js'/g, "$1.mjs'"));
  return pathToFileURL(dest).href;
});

let mod;
let spec;
let press;
try {
  mod = await import(urls[0]);
  spec = await import(urls[1]);
  press = await import(urls[2]);
} finally {
  rmSync(tmpDir, { recursive: true, force: true });
}

const {
  velocityAt, curlNoise, generateStreamlines, spawnOneParticle, flowFrame, wingCl,
  underbodyTerms, atStation, reynoldsNumber, reynoldsLabel, windVector,
} = mod;
const { deriveSurfaces, toFlowGeo, summarize, rearFlapAngleDeg, lapTimeDeltaS, REFERENCE } = spec;
const { computeCpField } = press;

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
// 300 probes, not 40: the ratio is a mean over randomly placed points, and at
// 40 it swung 0.115-0.206 between runs, crossing the bound below at random.
for (let i = 0; i < 300; i++) {
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

// The rake rotates through 5 heights x 9 lateral tubes; check the tubes exist.
const rakeRows = [0.05, 0.16, 0.3, 0.48, 0.72];
const rakeTubes = new Set();
for (let i = 0; i < 480; i++) {
  const p = spawnOneParticle(bounds, { ...baseFlow, flowSource: 'smoke_rake' }, i);
  const fy = (p.y - bounds.min[1]) / spanY;
  const row = rakeRows.reduce((best, r, idx) => (Math.abs(r - fy) < Math.abs(rakeRows[best] - fy) ? idx : best), 0);
  const col = Math.max(0, Math.min(8, Math.floor(((p.z - bounds.min[2]) / spanZ) * 9)));
  rakeTubes.add(`${row}:${col}`);
}
check(rakeTubes.size >= 40, 'rake tube coverage', `${rakeTubes.size}/45 tubes seeded`);

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

// ---------------------------------------------------------------------------
// 7. The parametric built-in car: solved stations, not box fractions
// ---------------------------------------------------------------------------
// The viewer publishes flow.geo (see viewer3d.js::_syncFlowGeo) built by
// carSpec.toFlowGeo. A metric model is centred on x/z at its bounding-box
// middle, unscaled, so world x = car x - centreX. That mapping is the whole
// contract between the drawn body and the smoke, so it is exercised here.

const CAR_CONFIG = {
  front_wing_main: { angle_deg: 2, chord_m: 0.35, span_m: 1.8, gurney_mm: 0 },
  front_wing_flap: { angle_deg: 8, chord_m: 0.12, span_m: 1.75, gurney_mm: 0 },
  rear_wing_main: { angle_deg: 5, chord_m: 0.4, span_m: 0.95, gurney_mm: 0 },
  rear_wing_flap: { angle_deg: 15, chord_m: 0.15, span_m: 0.9, gurney_mm: 0 },
  ride_height_front_mm: 25,
  ride_height_rear_mm: 45,
  rake_deg: 1.5,
  diffuser_angle_deg: 12,
  diffuser_ride_height_mm: 30,
};

function geoFor(config, scale = 1, yawDeg = 0) {
  const s = deriveSurfaces(config);
  const centreX = (s.extents.minX + s.extents.maxX) / 2;
  const th = (yawDeg * Math.PI) / 180;
  // Three.js Y-rotation: local +x -> (cos, 0, -sin), local +z -> (sin, 0, cos).
  return toFlowGeo(s, {
    origin: [-centreX * scale, 0, 0],
    axis: [Math.cos(th), 0, -Math.sin(th)],
    lateral: [Math.sin(th), 0, Math.cos(th)],
    groundY: 0,
    scale,
  });
}

const carGeo = geoFor(CAR_CONFIG);
const carFlow = { ...baseFlow, geo: carGeo };
const carFr = flowFrame(carFlow, bounds);
// With flow.geo the box is irrelevant, so a deliberately wrong one proves it.
const offBounds = { min: [-10, -10, -10], max: [10, 10, 10] };

check(
  Math.abs(carFr.length - 4.23) < 1e-9 && Math.abs(carFr.halfZ - 0.9) < 1e-9,
  'car frame from solved extents',
  `length = ${carFr.length.toFixed(3)} m, halfWidth = ${carFr.halfZ.toFixed(3)} m`,
);
// extents.maxY used to be a flap leading edge plus a whole chord: 1.1 m of a
// car whose drawn top is the rear-wing endplate at 0.91 m, which inflated the
// wake centre, sigma and seeding heights by a fifth.
check(
  Math.abs(carGeo.height - 0.91) < 1e-9,
  'declared height is the drawn endplate crown',
  `${carGeo.height.toFixed(3)} m (a single-seater stays under 1.0 m)`,
);
const noseWorld = atStation(carFr, carFr.nose, 0, 0);
check(
  Math.abs(noseWorld[0] + carFr.length / 2) < 1e-9
    && Math.abs(noseWorld[1]) < 1e-12
    && Math.abs(noseWorld[2]) < 1e-12,
  'metric car centred, floor at y = 0',
  `nose at x = ${noseWorld[0].toFixed(3)} m (-${(carFr.length / 2).toFixed(3)} expected), y = ${noseWorld[1].toFixed(3)}`,
);

// The mirror of backend/simulation/analytical.py has to keep returning the
// numbers that model returns, or the smoke argues with the readout. These
// literals are _wing_cl() of that file at its own defaults, priced at the
// trailing-edge clearance WingStation.min_clearance_m reports for that element —
// null means the model leaves it in the freestream (verified 2026-11-03 by
// running it).
const golden = [
  ['front_wing_main', 0.029999999999999995, 0.24410348659790082],
  ['front_wing_flap', 0.07721482384587533, 0.9380378135449086],
  ['rear_wing_main', null, 0.5046788703966693],
  ['rear_wing_flap', null, 1.4917784990863654],
];
let clErr = 0;
const clearanceMismatch = [];
for (const [id, wantC, wantCl] of golden) {
  const w = carGeo.aero.wings.find((x) => x.id === id);
  clErr = Math.max(clErr, Math.abs(wingCl(w) - wantCl));
  const ok = wantC === null
    ? w.clearanceM === null
    : w.clearanceM !== null && Math.abs(w.clearanceM - wantC) < 1e-12;
  if (!ok) clearanceMismatch.push(`${id} = ${w.clearanceM}, want ${wantC}`);
}
const ub = underbodyTerms(carGeo.aero);
check(clErr < 1e-12, 'wing Cl mirrors analytical.py', `max |diff| = ${clErr.toExponential(1)}`);
check(clearanceMismatch.length === 0, 'wing clearances mirror wing_stations', clearanceMismatch.join('; ') || 'two trailing-edge gaps, two nulls');

// Ground effect is priced from that gap, so it has to bleed off as the nose
// rises — the front ride height moves the wings, not only the floor. These
// literals are _wing_cl(cfg.front_wing_main, clearance) at those ride heights.
const riseCl = [60, 100].map((mm) => {
  const geo = geoFor({ ...CAR_CONFIG, ride_height_front_mm: mm });
  return wingCl(geo.aero.wings.find((x) => x.id === 'front_wing_main'));
});
check(
  Math.abs(riseCl[0] - 0.2378597912862731) < 1e-12
    && Math.abs(riseCl[1] - 0.23239654761549694) < 1e-12
    && riseCl[0] < wingCl(carGeo.aero.wings.find((x) => x.id === 'front_wing_main')) - 1e-3,
  'front wing lift falls as the nose rises',
  `main plane Cl = ${riseCl[0].toFixed(5)} at 60 mm, ${riseCl[1].toFixed(5)} at 100 mm (25 mm gives 0.24410)`,
);
check(
  Math.abs(ub.clUnder - 0.05885059942994359) < 1e-12
    && Math.abs(ub.cdUnder - 0.04089201550876271) < 1e-12,
  'underbody terms mirror analytical.py',
  `Cl = ${ub.clUnder.toFixed(9)}, Cd = ${ub.cdUnder.toFixed(9)}`,
);

// The lap-time proxy is one arithmetic on both sides of the wire: the literals
// are backend analytical.lap_time_delta_s at the same force pair and the same
// track weights the UI offers (verified 2026-11-03 by running the Python).
const lapForces = { drag_N: 1500.0, downforce_total_N: 4000.0 };
const lapBase = {
  drag_N: REFERENCE.lapBaselineDragN,
  downforce_total_N: REFERENCE.lapBaselineDfN,
};
const lapWant = [
  [{ dragMultiplier: 1.0, dfMultiplier: 1.0 }, -0.009999999999999998],
  [{ dragMultiplier: 2.2, dfMultiplier: 0.35 }, 0.05200000000000002],
  [{ dragMultiplier: 0.15, dfMultiplier: 2.4 }, -0.0915],
];
const lapErr = Math.max(
  ...lapWant.map(([track, want]) => Math.abs(lapTimeDeltaS(lapForces, lapBase, track) - want)),
);
check(
  lapErr === 0,
  'lap proxy mirrors analytical.py',
  `max |diff| = ${lapErr.toExponential(1)} s over ${lapWant.length} track weights`,
);
check(
  lapTimeDeltaS(lapBase, lapBase, null) === 0.0,
  'lap proxy is a delta, not an absolute',
  `baseline against itself = ${lapTimeDeltaS(lapBase, lapBase, null)} s`,
);

// Probe cloud: the same physical points for every config, clear of the floor's
// no-penetration epsilon so the scale test below stays clean.
const PROBES = [];
for (let i = 0; i <= 14; i++) {
  for (let j = 0; j <= 7; j++) {
    for (let k = 0; k <= 5; k++) {
      PROBES.push([-2.3 + (5.3 * i) / 14, 0.03 + (0.85 * j) / 7, -0.95 + (1.9 * k) / 5]);
    }
  }
}

const carSpeeds = [];
let carNonFinite = 0;
for (const [x, y, z] of PROBES) {
  const v = velocityAt(x, y, z, CAR_CONFIG, offBounds, carFlow);
  if (!v.every(Number.isFinite)) carNonFinite++;
  carSpeeds.push(Math.hypot(v[0], v[1], v[2]));
}
const carMean = carSpeeds.reduce((a, b) => a + b, 0) / carSpeeds.length;
const carMax = Math.max(...carSpeeds);
check(carNonFinite === 0, 'geo field finite', `${carNonFinite} non-finite of ${PROBES.length}`);
check(carMax < 2.2 * U, 'geo field bounded', `max |v| = ${carMax.toFixed(1)} m/s (${(carMax / U).toFixed(2)} U)`);
check(
  carMean > 0.8 * U && carMean < 1.4 * U,
  'geo field plausible',
  `mean = ${carMean.toFixed(1)} m/s (${(carMean / U).toFixed(2)} U)`,
);

function rmsDelta(a, b) {
  let sum = 0;
  const flowA = { ...baseFlow, geo: geoFor(a) };
  const flowB = { ...baseFlow, geo: geoFor(b) };
  for (const [x, y, z] of PROBES) {
    const p = velocityAt(x, y, z, a, offBounds, flowA);
    const q = velocityAt(x, y, z, b, offBounds, flowB);
    sum += (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
  }
  return Math.sqrt(sum / PROBES.length) / U;
}

const knob = (patch) => ({ ...CAR_CONFIG, ...patch });
for (const [label, patch] of [
  ['rear ride height 45 -> 85 mm', { ride_height_rear_mm: 85 }],
  ['diffuser angle 12 -> 18 deg', { diffuser_angle_deg: 18 }],
  ['rake 1.5 -> 3.5 deg', { rake_deg: 3.5 }],
  ['rear gurney 0 -> 10 mm', { rear_wing_main: { ...CAR_CONFIG.rear_wing_main, gurney_mm: 10 } }],
  ['front wing chord 0.35 -> 0.5 m', { front_wing_main: { ...CAR_CONFIG.front_wing_main, chord_m: 0.5 } }],
  ['front ride height 25 -> 60 mm', { ride_height_front_mm: 60 }],
  ['DRS open (rear flap 15 -> 2 deg)', { drs_active: true }],
]) {
  const d = rmsDelta(CAR_CONFIG, knob(patch));
  // A small threshold is still a real signal: eddy size, lattice origin and
  // wake height all come from the car frame, which none of these knobs move,
  // so the turbulence term cancels exactly in the difference. What is left is
  // the coherent field the knob actually changed.
  check(d > 0.003, `field responds: ${label}`, `RMS change = ${(d * 100).toFixed(2)}% U`);
}

// Ground effect: a shallower rear gap has to run faster, which is the whole
// reason the slider exists. Probed relative to each car's own floor line.
function gapSpeed(rearMm) {
  const cfg = knob({ ride_height_rear_mm: rearMm });
  const flow = { ...baseFlow, turbulence: 0, geo: geoFor(cfg) };
  const fr = flowFrame(flow, offBounds);
  const s = (fr.geo.floor.start + fr.geo.diffuser.start) / 2;
  // A fixed offset above the car's own floor line, so both cars are sampled in
  // the same part of the gap and the tunnel floor's boundary-layer profile
  // multiplies both by the same factor.
  const h = fr.geo.floor.heightStart
    + (fr.geo.floor.heightEnd - fr.geo.floor.heightStart) * (s - fr.geo.floor.start) / (fr.geo.floor.end - fr.geo.floor.start)
    + 0.025;
  const p = atStation(fr, s, h, 0);
  return velocityAt(p[0], p[1], p[2], cfg, offBounds, flow)[0];
}
const gapLow = gapSpeed(20);
const gapHigh = gapSpeed(80);
check(
  gapLow > gapHigh + 0.5,
  'underbody gap flow responds to ground effect',
  `u = ${gapLow.toFixed(1)} m/s at 20 mm vs ${gapHigh.toFixed(1)} m/s at 80 mm (U = ${U.toFixed(1)})`,
);

// Downforce means the air left the car moving upwards: momentum gives the wake
// an UPWASH, and drag gives it a deficit. Both used to be sign-wrong.
let wakeU = 0;
let wakeV = 0;
let wakeN = 0;
for (let i = 0; i <= 6; i++) {
  for (let j = 0; j <= 5; j++) {
    const p = atStation(carFr, carFr.tail + 0.35 + 0.5 * (i / 6), 0.1 + 0.5 * (j / 5), 0);
    const v = velocityAt(p[0], p[1], p[2], CAR_CONFIG, offBounds, carFlow);
    wakeU += v[0];
    wakeV += v[1];
    wakeN++;
  }
}
wakeU /= wakeN;
wakeV /= wakeN;
check(wakeU < 0.92 * U, 'geo wake deficit', `mean u = ${wakeU.toFixed(1)} m/s (${(wakeU / U).toFixed(2)} U)`);
check(wakeV > 0.02 * U, 'geo wake upwash (downforce deflects air up)', `mean v = +${wakeV.toFixed(1)} m/s`);

// Rake tubes are true metre heights on a solved car, not fractions of a box.
const tubeH = carGeo.rakeHeightsM;
const seeded = [];
for (let i = 0; i < 500; i++) {
  seeded.push(spawnOneParticle(offBounds, carFlow, i).y);
}
const lowest = Math.min(...seeded);
const rows = new Set(seeded.map((y) => tubeH.indexOf(tubeH.reduce((b, h) => (Math.abs(h - y) < Math.abs(b - y) ? h : b)))));
check(
  Math.abs(lowest - tubeH[0]) < 0.005 && tubeH[0] === 0.03,
  'rake seeds at true heights',
  `lowest tube ${(lowest * 1000).toFixed(0)} mm (spec ${(tubeH[0] * 1000).toFixed(0)} mm)`,
);
check(rows.size === tubeH.length, 'all rake rows reach the field', `${rows.size}/${tubeH.length} heights`);

// "Rake" is two unrelated quantities here: the suction term the lift laws read,
// and the angle the two ride heights actually put the body at. The UI and the
// advisor report both, so pin that they stay distinct.
const carSurfaces = deriveSurfaces(CAR_CONFIG);
const wheelbaseMm = (REFERENCE.rearAxleM - REFERENCE.frontAxleM) * 1000;
const bodyRake = Math.atan(
  (CAR_CONFIG.ride_height_rear_mm - CAR_CONFIG.ride_height_front_mm) / wheelbaseMm,
) / (Math.PI / 180);
check(
  Math.abs(carSurfaces.floor.angleDeg - bodyRake) < 1e-9
    && carSurfaces.aero.rakeDeg === CAR_CONFIG.rake_deg
    && Math.abs(carSurfaces.floor.angleDeg - CAR_CONFIG.rake_deg) > 1e-3
    && Math.abs(summarize(carSurfaces).rakeDeg - bodyRake) < 1e-9,
  'rake term and body rake are separate quantities',
  `term ${CAR_CONFIG.rake_deg} deg, body ${bodyRake.toFixed(3)} deg over ${wheelbaseMm.toFixed(0)} mm wheelbase`,
);

// The drawn diffuser cannot seal below the floor line ahead of it, even though
// the force model reads diffuser_ride_height_mm as an unclamped suction height.
const diff = carSurfaces.diffuser;
check(
  CAR_CONFIG.diffuser_ride_height_mm / 1000 < diff.throatHeight
    && Math.abs(diff.throatHeight - carSurfaces.floor.heightAt(diff.xStart)) < 1e-12
    && diff.exitHeight > diff.throatHeight,
  'diffuser throat clamped to the floor line',
  `requested ${CAR_CONFIG.diffuser_ride_height_mm} mm, drawn ${(diff.throatHeight * 1000).toFixed(1)} mm`,
);

// DRS is a solver mode, so the fold has to land in the geometry the field and
// the drawn car both read from - not in a UI-only override of the slider value.
const drsConfig = { ...CAR_CONFIG, drs_active: true };
const foldedFlap = rearFlapAngleDeg(drsConfig);
check(
  foldedFlap === REFERENCE.drsFlapAngleDeg
    && rearFlapAngleDeg(CAR_CONFIG) === CAR_CONFIG.rear_wing_flap.angle_deg
    && deriveSurfaces(drsConfig).wings.find((w) => w.id === 'rear_wing_flap').angle === foldedFlap,
  'DRS folds the rear flap in the shared geometry',
  `requested ${CAR_CONFIG.rear_wing_flap.angle_deg} deg, DRS ${foldedFlap} deg`,
);

// Display scale must not change the physics: every length in the field is
// published in world metres and every coefficient in true metres, so a 2x
// model is the same flow at twice the size.
const flowBig = { ...baseFlow, geo: geoFor(CAR_CONFIG, 2) };
let scaleErr = 0;
for (const [x, y, z] of PROBES) {
  const a = velocityAt(x, y, z, CAR_CONFIG, offBounds, carFlow);
  const b = velocityAt(2 * x, 2 * y, 2 * z, CAR_CONFIG, offBounds, flowBig);
  scaleErr = Math.max(scaleErr, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) / U);
}
check(scaleErr < 0.01, 'field invariant under display scale', `max deviation ${(scaleErr * 100).toFixed(2)}% U`);

// Yawing the model has to carry its wake with it: the deficit follows the axis.
const yawed = geoFor(CAR_CONFIG, 1, 25);
const flowYaw = { ...baseFlow, geo: yawed };
const frYaw = flowFrame(flowYaw, offBounds);
const behind = atStation(frYaw, frYaw.tail + 0.6, 0.3, 0);
const aside = atStation(frYaw, frYaw.tail + 0.6, 0.3, frYaw.spanZ * 1.4);
const uBehind = velocityAt(behind[0], behind[1], behind[2], CAR_CONFIG, offBounds, flowYaw)[0];
const uAside = velocityAt(aside[0], aside[1], aside[2], CAR_CONFIG, offBounds, flowYaw)[0];
check(
  uBehind < uAside - 0.05 * U,
  'wake follows the yawed car axis',
  `u on axis = ${uBehind.toFixed(1)} vs off axis = ${uAside.toFixed(1)} m/s`,
);

// ---------------------------------------------------------------------------
// 8. Skin pressure: the map has to carry the model's own element loads
// ---------------------------------------------------------------------------
// Bernoulli on the field cannot split a lifting surface into its suction side
// and its pressure side - the element's own sheet induces the same speed on both
// faces - so the skin map adds the jump condition. Measured here: the split
// across each drawn element has to equal the Cl verified against analytical.py
// above, with the ground-facing face in suction.

const D2R = Math.PI / 180;

function skinCloud(fr, geo) {
  const vertices = [];
  const normals = [];
  const tags = [];
  const push = (p, n, tag) => {
    vertices.push(p);
    normals.push(n);
    tags.push(tag);
  };
  for (const l of geo.lines) {
    const a = (l.angleDeg || 0) * D2R;
    for (const side of [1, -1]) {
      // Panel normal (sin a, cos a) in station/height space, in world axes.
      const n = [side * Math.sin(a) * fr.ax[0], side * Math.cos(a), side * Math.sin(a) * fr.ax[2]];
      for (let ti = 0; ti <= 4; ti++) {
        const t = (ti / 4 - 0.25) * l.chord; // the bound line sits at quarter chord
        for (let zi = 0; zi <= 3; zi++) {
          push(
            atStation(fr, l.station + t * Math.cos(a), l.height - t * Math.sin(a), (zi / 3 - 0.5) * l.halfSpan * 1.8),
            n,
            `${l.id}.${side > 0 ? 'up' : 'dn'}`,
          );
        }
      }
    }
    // Spanwise-facing skin on the same sheet: the drawn tip, the endplate inner
    // face, the nose cap. Edge on to the load, so no jump belongs there.
    for (let ti = 0; ti <= 4; ti++) {
      const t = (ti / 4 - 0.25) * l.chord;
      push(
        atStation(fr, l.station + t * Math.cos(a), l.height - t * Math.sin(a), l.halfSpan * 0.98),
        [0, 0, 1],
        `${l.id}.tip`,
      );
    }
  }
  const f = geo.floor;
  const d = geo.diffuser;
  for (const [s0, s1, h0, h1, tag] of [
    [f.start, d.start, f.heightStart, f.heightEnd, 'floor.dn'],
    [d.start, d.end, d.throat, d.exit, 'diffuser.dn'],
  ]) {
    for (let ti = 0; ti <= 6; ti++) {
      const t = ti / 6;
      for (let zi = 0; zi <= 3; zi++) {
        push(
          atStation(fr, s0 + t * (s1 - s0), h0 + t * (h1 - h0), (zi / 3 - 0.5) * f.halfWidth * 1.8),
          [0, -1, 0],
          tag,
        );
      }
    }
  }
  return { vertices, normals, tags };
}

function cpMeans(result, tags) {
  const acc = {};
  result.cp.forEach((cp, i) => {
    const a = acc[tags[i]] ??= { sum: 0, n: 0 };
    a.sum += cp;
    a.n += 1;
  });
  return Object.fromEntries(Object.entries(acc).map(([k, a]) => [k, a.sum / a.n]));
}

const cloud = skinCloud(carFr, carGeo);
const cpResult = computeCpField(cloud.vertices, cloud.normals, CAR_CONFIG, offBounds, carFlow);
const cp = cpMeans(cpResult, cloud.tags);
const split = (tag) => cp[`${tag}.up`] - cp[`${tag}.dn`];

check(
  cpResult.loadsApplied && cpResult.cp.every(Number.isFinite),
  'skin Cp is finite and reads the element loads',
  `${cloud.vertices.length} samples on the drawn surfaces`,
);

let splitErr = 0;
for (const [id, , want] of golden) {
  splitErr = Math.max(splitErr, Math.abs(split(id) - want));
}
check(
  splitErr < 0.02,
  'Cp split across each element equals its Cl',
  `max |error| = ${splitErr.toFixed(4)} (${golden.map(([id, , w]) => `${id} ${split(id).toFixed(2)}/${w.toFixed(2)}`).join(', ')})`,
);

check(
  cp['rear_wing_main.dn'] < 0 && cp['rear_wing_flap.dn'] < 0 && cp['front_wing_flap.dn'] < 0,
  'suction faces the ground, not the sky',
  `undersides: rear main ${cp['rear_wing_main.dn'].toFixed(2)}, rear flap ${cp['rear_wing_flap.dn'].toFixed(2)}, front flap ${cp['front_wing_flap.dn'].toFixed(2)}`,
);
check(
  cp['floor.dn'] < 0 && cp['diffuser.dn'] < 0,
  'underbody and diffuser read suction',
  `floor ${cp['floor.dn'].toFixed(2)}, diffuser ${cp['diffuser.dn'].toFixed(2)}`,
);

// An edge-on face belongs to the element but sees neither side of its jump, so
// it keeps the field level. Reading the face by its vertical normal alone put
// these skin points at Cp = 0 on the real mesh instead.
const edgeOnPlateau = golden
  .map(([id, , cl]) => [id, cp[`${id}.tip`], cl / 2])
  .filter(([, v, half]) => Math.abs(Math.abs(v) - half) < 0.02);
check(
  edgeOnPlateau.length === 0,
  'edge-on skin keeps the field level, not the jump',
  golden.map(([id, , cl]) => `${id} tip ${cp[`${id}.tip`].toFixed(2)} (jump plateau +-${(cl / 2).toFixed(2)})`).join(', '),
);

// A knob that changes a load has to change the map.
const highRwConfig = {
  ...CAR_CONFIG,
  rear_wing_main: { ...CAR_CONFIG.rear_wing_main, angle_deg: 12 },
};
const highRw = cpMeans(
  computeCpField(cloud.vertices, cloud.normals, highRwConfig, offBounds, { ...baseFlow, geo: geoFor(highRwConfig) }),
  cloud.tags,
);
check(
  highRw['rear_wing_main.dn'] < cp['rear_wing_main.dn'] - 0.05,
  'more rear incidence deepens its suction',
  `rear main underside ${cp['rear_wing_main.dn'].toFixed(2)} -> ${highRw['rear_wing_main.dn'].toFixed(2)} for 5 -> 12 deg`,
);
const drsCp = cpMeans(
  computeCpField(cloud.vertices, cloud.normals, drsConfig, offBounds, { ...baseFlow, geo: geoFor(drsConfig) }),
  cloud.tags,
);
const drsSplit = drsCp['rear_wing_flap.up'] - drsCp['rear_wing_flap.dn'];
check(
  drsSplit <= split('rear_wing_flap') - 0.05,
  'DRS unloads the flap on the pressure map too',
  `split ${split('rear_wing_flap').toFixed(2)} -> ${drsSplit.toFixed(2)}`,
);

// Ownership resolves in the car frame, so a yawed and 2x-drawn model has to put
// the same load on the same skin. This catches a panel stored as a world point
// rather than a station/height pair: those coincide only at zero yaw, scale 1.
const yawGeo = geoFor(CAR_CONFIG, 2, 25);
const yawFr = flowFrame({ ...baseFlow, geo: yawGeo }, offBounds);
const yawCloud = skinCloud(yawFr, yawGeo);
const yawCp = cpMeans(
  computeCpField(yawCloud.vertices, yawCloud.normals, CAR_CONFIG, offBounds, { ...baseFlow, geo: yawGeo }),
  yawCloud.tags,
);
let yawErr = 0;
for (const [id, , want] of golden) {
  yawErr = Math.max(yawErr, Math.abs(yawCp[`${id}.up`] - yawCp[`${id}.dn`] - want));
}
check(
  yawErr < 0.02,
  'loads land on the yawed, scaled model',
  `max |error| = ${yawErr.toFixed(4)} at 25 deg yaw, 2x scale`,
);

// An uploaded model has no solved elements, so it must not invent a load split.
const cadCp = computeCpField(cloud.vertices, cloud.normals, CAR_CONFIG, bounds, baseFlow);
check(
  !cadCp.loadsApplied && cadCp.cp.every(Number.isFinite),
  'no geometry, no invented loads',
  `field-only Cp ${cadCp.range.min.toFixed(2)} to ${cadCp.range.max.toFixed(2)}`,
);

// The tooltips on the two inflow sliders claim the tilt moves the smoke and not
// the interaction. These three checks are what makes that claim true.
const windAt = (over) => windVector({ ...carFlow, ...over });
const speed0 = Math.hypot(...windAt({ yawDeg: 0, pitchDeg: 0 }));
const speedTilted = Math.hypot(...windAt({ yawDeg: 35, pitchDeg: 15 }));
check(
  Math.abs(speed0 - speedTilted) < 1e-9,
  'inflow tilt leaves |U| alone',
  `${speed0.toFixed(4)} vs ${speedTilted.toFixed(4)} m/s at yaw 35 / pitch 15 - ` +
    'so the singularity strengths built on it are identical',
);
const wyUp = windAt({ pitchDeg: 15 })[1];
const wyDown = windAt({ pitchDeg: -15 })[1];
check(
  wyUp > 14 && Math.abs(wyUp + wyDown) < 1e-9,
  'airflow pitch up + means air arriving from below',
  `+15 deg gives wy = ${wyUp.toFixed(2)} m/s, -15 deg gives ${wyDown.toFixed(2)}`,
);
const probe = [0.35, 0.3, 0.2];
const pertNorm = (over) => {
  const flow = { ...carFlow, turbulence: 0, ...over };
  const v = velocityAt(probe[0], probe[1], probe[2], CAR_CONFIG, offBounds, flow);
  const w = windVector(flow);
  return Math.hypot(v[0] - w[0], v[1] - w[1], v[2] - w[2]);
};
const pert0 = pertNorm({ pitchDeg: 0 });
const drift = Math.max(
  Math.abs(pertNorm({ pitchDeg: 15 }) / pert0 - 1),
  Math.abs(pertNorm({ pitchDeg: -15 }) / pert0 - 1),
);
check(
  drift < 0.2,
  'the tilt is not an angle of attack',
  `near-body perturbation moves ${drift.toFixed(3)} at +/-15 deg pitch, ` +
    `${pert0.toFixed(1)} m/s at 0 - if this grows, the slider tooltips need rewriting`,
);

// The Re the HUD prints has to be the Re of the car on screen, not of a length
// the module invented beside it.
const reDefault = reynoldsNumber({ windSpeedKmh: 200 });
const reDeclared = (REFERENCE.rho * U * REFERENCE.carLengthM) / REFERENCE.muAirPaS;
check(
  Math.abs(reDefault - reDeclared) < 1,
  'Re is priced on the drawn car length',
  `Re ${reDefault.toExponential(3)} at 200 km/h on L=${REFERENCE.carLengthM} m, ` +
    `rho=${REFERENCE.rho}, mu=${REFERENCE.muAirPaS}`,
);
check(
  reynoldsLabel({ windSpeedKmh: 200 }).includes(`L=${REFERENCE.carLengthM} m`),
  'the Re readout names its length',
  reynoldsLabel({ windSpeedKmh: 200 }),
);

console.log('');
if (failures.length) {
  console.error(`FLOW FIELD CHECK FAILED (${failures.length}):`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log('Flow field check passed.');
