/** Wind tunnel flow physics — air/car interaction for any model. */

import { deriveSurfaces, rearFlapAngleDeg, underbodyInputs, REFERENCE } from './carSpec.js';

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
  /**
   * Solved car stations in world metres, published by the viewer for the
   * built-in parametric body. Null for uploaded CAD, which has no geometry the
   * solver knows about, so the field falls back to bounding-box fractions.
   */
  geo: null,
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

// ---------------------------------------------------------------------------
// Low-order aero terms mirrored from the backend
// ---------------------------------------------------------------------------
// backend/simulation/analytical.py is the model behind the reported numbers, and
// the browser cannot call it per tracer. These transcriptions let the tunnel
// respond to the same setup fields the numbers do — incidence, chord, span,
// gurney, both ride heights, rake and diffuser angle — rather than to a
// bounding box. All of it is in TRUE metres: the ground-effect decay constants
// are calibrated lengths, so the user's display scale must not stretch them.

const DEG = Math.PI / 180;
const OSWALD_E = 0.85; // _wing_cl e

/** analytical.ground_effect_lift_factor. Null clearance = not priced in ground effect. */
export function groundEffectLiftFactor(clearanceM) {
  if (clearanceM === null || clearanceM === undefined) return 1.0;
  return 1 + REFERENCE.geLiftGainMax * Math.exp(-Math.max(0, clearanceM) / REFERENCE.geLiftDecayM);
}

/** analytical.STALL_ONSET_DEG: where the thin-airfoil line reaches sectionClMax. */
const STALL_ONSET_DEG = (REFERENCE.sectionClMax / (2 * Math.PI)) / DEG;

/** analytical._stall_blend: 0 while attached, 1 once the section is a plate. */
export function stallBlend(angleDeg) {
  const excess = Math.abs(angleDeg) - STALL_ONSET_DEG;
  if (excess <= 0) return 0;
  return Math.min(1, excess / (REFERENCE.stallBlendMaxDeg - STALL_ONSET_DEG));
}

/** analytical._flat_plate_forces: [Cl, Cd] of the fully separated section. */
function flatPlateForces(angleDeg) {
  const a = Math.abs(angleDeg) * DEG;
  const cn = REFERENCE.flatPlateCn * Math.sin(a);
  return [cn * Math.cos(a), cn * Math.sin(a)];
}

/**
 * Downforce-positive lift coefficient (_wing_cl).
 *
 * Past the stall onset the section is handed to the flat-plate polar instead of
 * being pinned at the top of the thin-airfoil line, which is where the solver
 * keeps its post-stall loss of downforce. Below the onset the blend weight is
 * zero and the element prices exactly as the attached line alone.
 */
export function wingCl(w) {
  const blend = stallBlend(w.angleDeg);
  let cl2d;
  if (blend > 0) {
    const sign = w.angleDeg >= 0 ? 1 : -1;
    cl2d = sign * ((1 - blend) * REFERENCE.sectionClMax + blend * flatPlateForces(w.angleDeg)[0]);
  } else {
    cl2d = Math.max(-0.5, Math.min(2 * Math.PI * (w.angleDeg * DEG), REFERENCE.sectionClMax));
  }
  const ar = Math.max(0.5, w.spanM / w.chordM);
  let cl = cl2d / (1 + cl2d / (Math.PI * ar * OSWALD_E)); // Prandtl tip loss
  cl += w.gurneyM * 8; // _wing_cl — 0.08 Cl per 10 mm
  cl *= groundEffectLiftFactor(w.clearanceM);
  return cl;
}

/** _wing_cl's second return: induced drag plus the separation drag of a stalled element. */
export function wingCdFactor(w) {
  const ar = Math.max(0.5, w.spanM / w.chordM);
  const blend = stallBlend(w.angleDeg);
  const plateCd = blend > 0 ? flatPlateForces(w.angleDeg)[1] : 0;
  const cl = wingCl(w);
  return (cl * cl) / (Math.PI * ar * OSWALD_E) + blend * plateCd;
}

/**
 * Underbody and diffuser terms (_underbody_forces), from the same REFERENCE keys
 * the cross-language test pins to the solver. The diffuser decays on
 * `aero.diffuserThroatM`, which carSpec's `underbodyInputs` supplies - the built-in
 * car hands over the drawn throat, already raised to the floor line exactly as
 * `analytical.diffuser_throat_height_m` does, rather than the raw slider the
 * force model has ignored below that line since.
 */
export function underbodyTerms(aero) {
  const geFront = Math.exp(-aero.frontRideHeightM / REFERENCE.geDecayFrontM);
  const geRear = Math.exp(-aero.rearRideHeightM / REFERENCE.geDecayRearM);
  const rakeBoost = 1 + aero.rakeDeg * REFERENCE.rakeCoefPerDeg;
  const clFloor = REFERENCE.clFloorMax * geRear * rakeBoost;
  const clDiff = REFERENCE.clDiffuserMax * Math.sin(aero.diffuserAngleDeg * DEG)
    * Math.exp(-aero.diffuserThroatM / REFERENCE.geDecayDiffuserM) * rakeBoost;
  return {
    geFront,
    geRear,
    clUnder: clFloor + clDiff,
    cdUnder: REFERENCE.cdFloorBase
      + REFERENCE.cdFloorGapCoef * (1 - geRear)
      + REFERENCE.cdDiffuserBase,
  };
}

/**
 * The gap's through-flow, in m³/s. Δp = ρU·Δu together with Δp = Cl·q gives
 * Δu = Cl·U/2, and a source/sink pair D apart induces ≈ 4Q/(πD²) between them,
 * so Q = Δu·πD²/4. Every tier of buildSources prices it here and pushes the pair
 * inlet-first, so the flow runs along the gap rather than against it.
 */
function underbodyJetQ(u, clUnder, sepM) {
  return (0.5 * u * clUnder) * Math.PI * sepM * sepM / 4;
}

/** Σ Cd·A in m²: body and wheels, every wing element, the underbody. */
function dragAreaM2(aero) {
  let cdArea = aero.refAreaM2 * aero.bodyCd;
  for (const w of aero.wings) {
    cdArea += w.chordM * w.spanM * (aero.profileCd + wingCdFactor(w));
  }
  return cdArea + aero.floorAreaM2 * underbodyTerms(aero).cdUnder;
}

/** Σ Cl·A in m²: the downforce the wake has to deflect upward. */
function downforceAreaM2(aero) {
  let clArea = 0;
  for (const w of aero.wings) {
    clArea += wingCl(w) * w.chordM * w.spanM;
  }
  return clArea + aero.floorAreaM2 * underbodyTerms(aero).clUnder;
}

/**
 * Momentum balance on the control volume: drag leaves a velocity deficit and
 * downforce an upwash, both spread over the blockage rectangle the setup is
 * charged on. The 0.15-0.6 and 0.35 bounds are the picture's budget, not solved
 * limits.
 *
 * The rectangle is `track * rear wing height`, the body's own blockage, and NOT
 * the drawn bounding box: an uploaded mesh's width says nothing about the car
 * the panel is pricing, so a box-sized denominator would let the same setup
 * show a different wake than the numbers report.
 */
function wakeBudget(aero) {
  const aWake = Math.max(0.4, aero.trackM * aero.rwHeightM);
  return {
    deficit: Math.min(0.6, Math.max(0.15, 0.5 * dragAreaM2(aero) / aWake)),
    upwash: Math.min(0.35, 0.5 * downforceAreaM2(aero) / aWake),
  };
}

// ---------------------------------------------------------------------------
// Field frame: car stations to world points
// ---------------------------------------------------------------------------

/**
 * Where the model sits and how big it is, in world metres. With flow.geo these
 * are the built-in car's true stations; without it (uploaded CAD) they fall back
 * to the model AABB, the best available guess. Heights are measured above the
 * tunnel floor, stations along the car axis from its nose.
 */
export function flowFrame(flow, bounds) {
  const geo = flow.geo;
  if (geo) {
    return {
      o: geo.origin,
      ax: geo.axis,
      la: geo.lateral,
      groundY: geo.groundY,
      nose: geo.stations.nose,
      tail: geo.stations.tail,
      top: geo.stations.top,
      length: geo.length,
      height: geo.height,
      halfZ: geo.halfWidth,
      spanZ: geo.halfWidth * 2,
      tubes: geo.rakeHeightsM,
      absoluteTubes: true,
      scale: geo.scale,
      aero: geo.aero,
      geo,
      // Softening radii stretch with the model, or 2x scale is a different car.
      soft2: 0.0025 * geo.scale * geo.scale,
    };
  }
  const min = bounds.min;
  const max = bounds.max;
  const spanX = Math.max(max[0] - min[0], 0.01);
  const spanY = Math.max(max[1] - min[1], 0.01);
  const spanZ = Math.max(max[2] - min[2], 0.01);
  return {
    o: [min[0], min[1], (min[2] + max[2]) / 2],
    ax: [1, 0, 0],
    la: [0, 0, 1],
    groundY: min[1],
    nose: 0,
    tail: spanX,
    top: spanY,
    length: spanX,
    height: spanY,
    halfZ: spanZ / 2,
    spanZ,
    tubes: [0.05, 0.16, 0.3, 0.48, 0.72],
    absoluteTubes: false,
    scale: 1,
    aero: null,
    geo: null,
    soft2: 0.0025,
  };
}

/** World point at car station s, height h above the floor, lateral offset z. */
export function atStation(fr, s, h, z) {
  const lz = z || 0;
  return [
    fr.o[0] + fr.ax[0] * s + fr.la[0] * lz,
    fr.o[1] + h + fr.la[1] * lz,
    fr.o[2] + fr.ax[2] * s + fr.la[2] * lz,
  ];
}

/** Position along the car axis as a fraction of its length: 0 nose, 1 tail. */
function stationFrac(fr, x, y, z) {
  const s = (x - fr.o[0]) * fr.ax[0] + (y - fr.o[1]) * fr.ax[1] + (z - fr.o[2]) * fr.ax[2];
  return (s - fr.nose) / fr.length;
}

/** Signed offset from the car axis: 0 on the centreline, positive to the right. */
function lateralOf(fr, x, y, z) {
  return (x - fr.o[0]) * fr.la[0] + (y - fr.o[1]) * fr.la[1] + (z - fr.o[2]) * fr.la[2];
}

/**
 * Induced velocity of a straight vortex filament with a Lamb-Oseen core, so the
 * field stays finite on the axis. Covers both cases a car needs: a spanwise
 * bound lifting line of finite half-span, and a trailing tip vortex that starts
 * at the surface and fades downstream. Induced v = Γ/2πr² · (d × r).
 */
function filament(x, y, z, p, dir, gamma, core, decayLen, halfSpan) {
  const dx = x - p[0];
  const dy = y - p[1];
  const dz = z - p[2];
  const t = dx * dir[0] + dy * dir[1] + dz * dir[2];
  if (decayLen && t <= 0) return false; // semi-infinite: nothing upstream of p
  const rx = dx - t * dir[0];
  const ry = dy - t * dir[1];
  const rz = dz - t * dir[2];
  const r2 = rx * rx + ry * ry + rz * rz;
  const c2 = core * core;
  let f = (gamma / (2 * Math.PI * (r2 + c2))) * (1 - Math.exp(-r2 / c2));
  if (decayLen) f *= Math.exp(-t / decayLen);
  if (halfSpan) {
    // A bound vortex ends at its tip, so its influence falls off along itself.
    const e = Math.abs(t) / (halfSpan * 1.6);
    f *= 1 / (1 + e * e * e * e);
  }
  _vortexOut[0] = (dir[1] * rz - dir[2] * ry) * f;
  _vortexOut[1] = (dir[2] * rx - dir[0] * rz) * f;
  _vortexOut[2] = (dir[0] * ry - dir[1] * rx) * f;
  return true;
}

/** Regularized potential-flow doublet: perturbation for a body of radius r. */
function doubletPert(x, y, z, p, r, ux, uy, uz, s2) {
  const dx = x - p[0];
  const dy = y - p[1];
  const dz = z - p[2];
  const d = Math.sqrt(dx * dx + dy * dy + dz * dz + s2);
  const d3 = d * d * d;
  const d5 = d3 * d * d;
  const r3 = r * r * r;
  const t1 = r3 / (2 * d3);
  // Sign matters: this standard form decelerates the flow ahead of the body
  // (stagnation). The inverted one accelerated air into the nose instead.
  const t2 = -(3 * r3 * (ux * dx + uy * dy + uz * dz)) / (2 * d5);
  _vortexOut[0] = t1 * ux + t2 * dx;
  _vortexOut[1] = t1 * uy + t2 * dy;
  _vortexOut[2] = t1 * uz + t2 * dz;
}

/** Source/sink of strength Q: mass flux spread over a sphere. */
function sourcePert(x, y, z, p, Q, s2) {
  const dx = x - p[0];
  const dy = y - p[1];
  const dz = z - p[2];
  const d = Math.sqrt(dx * dx + dy * dy + dz * dz + s2);
  const f = Q / (4 * Math.PI * d * d * d);
  _vortexOut[0] = f * dx;
  _vortexOut[1] = f * dy;
  _vortexOut[2] = f * dz;
}

/**
 * Wake cross-section `ds` downstream of its start: [vertical, lateral] Gaussian
 * semi-axes. Both spread linearly, so the ellipse keeps its shape downstream.
 */
function wakeAxes(wake, ds, length) {
  const k = ds / length;
  return [wake.sigma0 + wake.growth * k, wake.z0 + wake.zGrowth * k];
}

/**
 * Share of a surface's bound circulation each tip vortex rolls up. 0.6 is the
 * elliptic-loading value, tuned for looks rather than solved; both filament
 * tiers use it so a wing of a given Cl draws the same roll-up either way.
 */
const TIP_ROLLUP_SHARE = 0.6;

/**
 * The model's singularity distribution: blockage doublets, lifting lines with
 * their tip vortices, the underbody through-flow, and the wake parameters.
 * Built once per geometry or wind change because velocityAt runs per tracer.
 */
function buildSources(fr, config, wind) {
  const [wx, wy, wz] = wind;
  const u = Math.hypot(wx, wy, wz) || 1;
  const src = {
    doublets: [],
    filaments: [],
    jets: [],
    wake: {
      sStart: fr.tail - fr.length * 0.05,
      hCenter: fr.height * 0.42,
      sigma0: fr.height * 0.24,
      growth: fr.height * 0.42,
      // Lateral semi-axis: the wake is as wide as the body that made it, and
      // spreads at the same relative rate as the vertical axis (0.42 per car
      // length), so the cross-section stays self-similar downstream.
      z0: fr.halfZ,
      zGrowth: fr.halfZ * 0.42,
      decayLen: fr.length * 1.1,
    },
  };

  if (fr.geo) {
    const g = fr.geo;
    for (const c of g.chassis) {
      src.doublets.push({ p: atStation(fr, c.station, c.height, 0), r: c.radius });
    }
    for (const w of g.wheels) {
      src.doublets.push({ p: atStation(fr, w.station, w.height, w.lateral), r: w.radius });
    }

    const byId = {};
    for (const w of g.aero.wings) byId[w.id] = w;
    const tipSites = [];
    for (const l of g.lines) {
      const w = byId[l.id];
      const cl = w ? wingCl(w) : 0.2;
      // Γ = ½·U·c·Cl. With v ∝ d × r and d along +lateral, positive Γ lifts the
      // air behind the surface — the reaction a downforce wing requires.
      const gamma = 0.5 * u * l.chord * cl;
      src.filaments.push({
        p: atStation(fr, l.station, l.height, 0),
        dir: fr.la,
        gamma,
        core: Math.max(0.01 * fr.scale, 0.1 * l.chord),
        decayLen: 0,
        halfSpan: l.halfSpan,
      });
      if (l.id.endsWith('_main')) {
        tipSites.push({
          station: l.station + 0.75 * l.chord,
          height: l.height,
          halfSpan: l.halfSpan,
          chord: l.chord,
          gamma,
        });
      }
    }
    // Each tip rolls up a share of its surface's bound circulation.
    for (const t of tipSites) {
      const core = Math.max(0.015 * fr.scale, 0.16 * t.chord);
      for (const side of [1, -1]) {
        src.filaments.push({
          p: atStation(fr, t.station, t.height, side * t.halfSpan),
          dir: fr.ax,
          gamma: side * TIP_ROLLUP_SHARE * t.gamma,
          core,
          decayLen: 1.3 * fr.length,
          halfSpan: 0,
        });
      }
    }

    // Underbody through-flow: source at the floor inlet, sink at the diffuser
    // exit, so the pair drags air along the gap instead of against it.
    const ub = underbodyTerms(g.aero);
    const pIn = atStation(fr, g.floor.start, g.floor.heightStart, 0);
    const pOut = atStation(fr, g.diffuser.end, (g.diffuser.throat + g.diffuser.exit) / 2, 0);
    const sep = Math.hypot(pOut[0] - pIn[0], pOut[1] - pIn[1], pOut[2] - pIn[2]);
    const Q = underbodyJetQ(u, ub.clUnder, sep);
    src.jets.push({ p: pIn, Q }, { p: pOut, Q: -Q });

    // Momentum balance turns drag into the velocity deficit and downforce into
    // the wake upwash.
    Object.assign(src.wake, wakeBudget(g.aero));
    return src;
  }

  // --- Uploaded CAD has no solved geometry, so the distribution is placed ---
  // --- from the model AABB, as the field was before stations existed.      ---
  const body = [[0.15, 0.22, 0.26], [0.4, 0.35, 0.38], [0.65, 0.36, 0.44], [0.85, 0.32, 0.28]];
  for (const [fs, fh, fz] of body) {
    src.doublets.push({ p: atStation(fr, fr.length * fs, fr.height * fh, 0), r: fr.spanZ * fz });
  }
  const wheels = [
    [0.22, 0.25, 0.12, 0.33], [0.22, 0.25, 0.88, 0.33],
    [0.8, 0.28, 0.08, 0.36], [0.8, 0.28, 0.92, 0.36],
  ];
  for (const [fs, fh, fz, fr2] of wheels) {
    src.doublets.push({ p: atStation(fr, fr.length * fs, fr.height * fh, fr.spanZ * (fz - 0.5)), r: fr.height * fr2 });
  }

  // The setup's own wing elements. An uploaded model has no solved geometry to
  // place them on, so they still sit at box fractions below, but their strength
  // is now the same lifting-line law the solved branch uses: Gamma = 1/2 U c Cl
  // with Cl from wingCl, which is where the section's post-stall loss of
  // downforce lives. Raw degrees kept buying circulation past the knee the
  // numbers say the wing is losing it at, and priced none of the other fields
  // that move the reported lift (gurney height, ground-effect clearance, the
  // DRS-clamped flap). Chord is stretched by the length uploaded meshes are
  // normalised to, so a bigger model carries proportionally stronger vortices.
  const lenScale = fr.length / REFERENCE.carLengthM;
  // One payload for the whole branch: the setup's own surfaces, with the
  // underbody inputs this tier takes a null throat on (an upload has no floor
  // line to clamp the diffuser against). `underbodyTerms` reads only those
  // inputs, so the wings, the gap jet and the wake all price off this object.
  const cadAero = { ...deriveSurfaces(config).aero, ...underbodyInputs(config, null) };
  const gammaOf = { front: 0, rear: 0, frontMain: 0, rearMain: 0 };
  for (const w of cadAero.wings) {
    const end = w.id.startsWith('front_wing') ? 'front' : 'rear';
    const g = 0.5 * u * w.chordM * lenScale * wingCl(w);
    gammaOf[end] += g;
    if (w.id.endsWith('_main')) gammaOf[`${end}Main`] = g;
  }
  const bound = [
    [0.04, 0.14, gammaOf.front, 0.475],
    [0.88, 0.74, gammaOf.rear, 0.275],
  ];
  for (const [fs, fh, gamma, halfTipFrac] of bound) {
    src.filaments.push({
      p: atStation(fr, fr.length * fs, fr.height * fh, 0),
      dir: fr.la,
      gamma,
      core: 0.05 * fr.length,
      decayLen: 0,
      halfSpan: fr.spanZ * halfTipFrac,
    });
  }
  // Tips roll up the main element's own bound circulation, as in the solved tier.
  const tips = [
    [0.06, 0.14, 0.475, 0.045, 1.7, gammaOf.frontMain],
    [0.9, 0.74, 0.275, 0.06, 1.2, gammaOf.rearMain],
  ];
  for (const [fs, fh, dz, coreFrac, decayFrac, gammaMain] of tips) {
    for (const side of [1, -1]) {
      src.filaments.push({
        p: atStation(fr, fr.length * fs, fr.height * fh, side * fr.spanZ * dz),
        dir: fr.ax,
        gamma: side * TIP_ROLLUP_SHARE * gammaMain,
        core: fr.height * coreFrac,
        decayLen: fr.length * decayFrac,
        halfSpan: 0,
      });
    }
  }

  // Underbody through-flow, priced by the same law as the solved tier: the
  // sliders that move the charged floor move this gap flow, and it runs front to
  // rear. `cadAero` carries a null throat because an uploaded model has no floor
  // line to clamp it against, so `diffuser_ride_height_mm` stands on its own
  // here; only the built-in car gets carSpec's floor-raised throat.
  const ub = underbodyTerms(cadAero);
  const pIn = atStation(fr, fr.length * 0.45, 0, 0);
  const pOut = atStation(fr, fr.length * 0.82, fr.height * 0.15, 0);
  const sep = Math.hypot(pOut[0] - pIn[0], pOut[1] - pIn[1], pOut[2] - pIn[2]);
  const Q = underbodyJetQ(u, ub.clUnder, sep);
  src.jets.push({ p: pIn, Q }, { p: pOut, Q: -Q });

  // The same momentum balance the built-in car gets. This tier used to leave
  // 0.4 and 0.1 standing for every setup, so an upload's wake was the one part
  // of the field that no knob could move - and 0.1 of upwash drew a deflection
  // a third of the size the identical setup prices on the solved branch.
  Object.assign(src.wake, wakeBudget(cadAero));
  return src;
}

let _srcCache = { geo: null, bounds: null, sig: '', fr: null, src: null };

function sourcesFor(flow, config, bounds, wind) {
  // Over-invalidation only costs a rebuild; under-invalidation leaves the
  // field drawn from a setup that no longer exists.
  const sig = [
    flow.windSpeedKmh, flow.yawDeg, flow.pitchDeg, flow.showInteraction,
    rearFlapAngleDeg(config),
    config ? JSON.stringify(config) : '',
  ].join('|');
  const geo = flow.geo || null;
  if (_srcCache.geo === geo && _srcCache.bounds === bounds && _srcCache.sig === sig) {
    return _srcCache;
  }
  const fr = flowFrame(flow, bounds);
  _srcCache = { geo, bounds, sig, fr, src: buildSources(fr, config, wind) };
  return _srcCache;
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

/**
 * Ceiling on |v - U| as a fraction of U. Singularities grow as 1/d² and 1/d³,
 * so inside a core the perturbation reaches hundreds of m/s and tracers skip
 * clean through the car; this is the attached-flow overshoot the field is
 * allowed to report. meshFlow.js holds its own deflection to the same budget,
 * so nothing downstream can quietly exceed it.
 */
export const PERTURBATION_BUDGET = 0.85;

/**
 * Freestream plus the model's singularity system, with the near-body cap.
 *
 * This is the inviscid half of the field, and the only half a static pressure
 * may be read from: Bernoulli relates Cp to an inviscid streamline, while the
 * wake deficit, curl noise and floor boundary layer added by velocityAt() below
 * are viscous display layers for the tracers. Feeding those into
 * 1 - (|v|/U)^2 would report a total-pressure loss as a pressure rise, which is
 * how the rear wing ended up coloured in compression while the solver still
 * priced it as the largest downforce element.
 */
function potentialFrom(fr, src, wx, wy, wz, uRef, x, y, z) {
  let px = 0;
  let py = 0;
  let pz = 0;

  const s2 = fr.soft2;
  for (const d of src.doublets) {
    doubletPert(x, y, z, d.p, d.r, wx, wy, wz, s2);
    px += _vortexOut[0];
    py += _vortexOut[1];
    pz += _vortexOut[2];
  }

  for (const j of src.jets) {
    sourcePert(x, y, z, j.p, j.Q, s2);
    px += _vortexOut[0];
    py += _vortexOut[1];
    pz += _vortexOut[2];
  }

  for (const f of src.filaments) {
    if (filament(x, y, z, f.p, f.dir, f.gamma, f.core, f.decayLen, f.halfSpan)) {
      px += _vortexOut[0];
      py += _vortexOut[1];
      pz += _vortexOut[2];
    }
  }

  const pMag = Math.sqrt(px * px + py * py + pz * pz);
  const pCap = PERTURBATION_BUDGET * uRef;
  if (pMag > pCap) {
    const k = pCap / pMag;
    px *= k;
    py *= k;
    pz *= k;
  }
  return [wx + px, wy + py, wz + pz];
}

/**
 * Inviscid velocity, for pressure: the same singularities the smoke is advected
 * by, without the wake deficit, turbulence or floor boundary layer.
 */
export function potentialVelocityAt(x, y, z, config, bounds, flow) {
  const [wx, wy, wz] = windVector(flow);
  if (flow.showInteraction === false) return [wx, wy, wz];
  const cache = sourcesFor(flow, config, bounds, [wx, wy, wz]);
  const uRef = Math.sqrt(wx * wx + wy * wy + wz * wz) || 1;
  return potentialFrom(cache.fr, cache.src, wx, wy, wz, uRef, x, y, z);
}

/** Velocity field: the model's singularity distribution over the freestream. */
export function velocityAt(x, y, z, config, bounds, flow) {
  const [wx, wy, wz] = windVector(flow);
  const uRef = Math.sqrt(wx * wx + wy * wy + wz * wz) || 1;
  const cache = sourcesFor(flow, config, bounds, [wx, wy, wz]);
  const fr = cache.fr;
  const src = cache.src;
  const wake = src.wake;
  const wakeH = fr.groundY + wake.hCenter;

  let vx = wx;
  let vy = wy;
  let vz = wz;

  // Station along the car axis, so the field follows a yawed or scaled model.
  const s = (x - fr.o[0]) * fr.ax[0] + (y - fr.o[1]) * fr.ax[1] + (z - fr.o[2]) * fr.ax[2];
  // And the offset across it, for the same reason: the wake has a width.
  const zz = (x - fr.o[0]) * fr.la[0] + (y - fr.o[1]) * fr.la[1] + (z - fr.o[2]) * fr.la[2];

  if (flow.showInteraction !== false) {
    const p = potentialFrom(fr, src, wx, wy, wz, uRef, x, y, z);
    vx = p[0];
    vy = p[1];
    vz = p[2];

    // ---- Mean flow behind the car ----------------------------------------
    // Momentum balance on the control volume: drag leaves a velocity deficit and
    // downforce means the air has been deflected UP, so the wake rises. Both
    // strengths come from the same low-order terms that produced the numbers.
    if (s > wake.sStart) {
      const ds = s - wake.sStart;
      const decay = Math.exp(-ds / wake.decayLen);
      const axes = wakeAxes(wake, ds, fr.length);
      const dy = y - wakeH;
      const g = Math.exp(-((dy * dy) / (2 * axes[0] * axes[0]) + (zz * zz) / (2 * axes[1] * axes[1])));
      const m = decay * g;
      vx -= wx * wake.deficit * m;
      vy += uRef * wake.upwash * m;
    }
  }

  // ---- Coherent turbulence ------------------------------------------------
  // Divergence-free curl noise, amplified where the tunnel is actually
  // turbulent (the wake) and scaled by the turbulence slider. This replaces
  // per-frame white jitter: eddies persist in space and evolve slowly.
  const turbLevel = flow.turbulence ?? 0.12;
  if (turbLevel > 1e-3) {
    const n = (s - fr.nose) / fr.length;
    const behind = clamp((n - 0.45) / 0.4, 0, 1);
    const dy = y - wakeH;
    // Eddies live where the deficit lives: the same ellipse, not a sheet that
    // reaches outboard long after the wake it belongs to has gone.
    const axes = wakeAxes(wake, Math.max(0, s - wake.sStart), fr.length);
    const inWake = behind * Math.exp(-((dy * dy) / (2 * axes[0] * axes[0]) + (zz * zz) / (2 * axes[1] * axes[1])));
    const eddy = Math.max(fr.length * 0.22, 0.05);
    const c = curlNoise(
      (x - fr.o[0]) / eddy,
      (y - fr.groundY) / eddy,
      (z - fr.o[2]) / eddy + 17,
      (flow.time || 0) * 1.6,
    );
    const gain = turbLevel * uRef * (0.55 + 2.4 * inWake);
    let tx = c[0] * gain;
    let ty = c[1] * gain;
    let tz = c[2] * gain;
    const tm = Math.sqrt(tx * tx + ty * ty + tz * tz);
    const cap = gain * 1.6;
    if (tm > cap) {
      const k = cap / tm;
      tx *= k;
      ty *= k;
      tz *= k;
    }
    vx += tx;
    vy += ty;
    vz += tz;
  }

  // ---- Floor boundary layer ----------------------------------------------
  // The tunnel floor is a no-slip wall: entrained smoke crawls within the
  // boundary layer and grades back to freestream above it. This is the cue
  // that makes smoke rakes near the ground look right.
  const yRel = y - fr.groundY;
  const blH = fr.height * 0.18;
  if (yRel < blH && yRel > -0.05 * fr.scale) {
    const eta = clamp(yRel / blH, 0, 1);
    const profile = Math.max(0.18, Math.pow(eta, 1 / 7));
    vx *= profile;
    vy *= profile;
    vz *= profile;
  }

  // Ground plane boundary condition — no penetration
  if (y <= fr.groundY + 0.01 * fr.scale && vy < 0) vy = Math.abs(vy) * 0.15;

  return [vx, vy, vz];
}

export function speedAt(x, y, z, config, bounds, flow) {
  return len(velocityAt(x, y, z, config, bounds, flow));
}

/**
 * The wake as the field defines it, ready to draw.
 *
 * `edge` is the profile boundary: the ellipse on which the local deficit has
 * fallen to `edge` of the centreline deficit at the same station. Because the
 * field's cross-section is a Gaussian whose axes both grow linearly, that
 * ellipse spreads downstream - the self-similar wake edge, not a fixed contour.
 * (Drawing an absolute iso-deficit surface instead would taper to a point,
 * because the streamwise decay eats the contour faster than the spread grows;
 * that taper is what the old bounding-box cone got by accident.)
 *
 * `recover` sets the length: the envelope stops where the centreline deficit
 * has recovered to `recover` of its value at the tail, i.e. `decayLen` *
 * ln(1/`recover`) behind the start. Each ring's downward semi-axis is clipped
 * at the tunnel floor, below which the model has no air.
 *
 * Reads the same cached sources the tracers are advected by, so the picture
 * cannot drift from the computation.
 */
export function wakeShape(flow, config, bounds, edge = 0.5, recover = 0.25) {
  const [wx, wy, wz] = windVector(flow);
  const cache = sourcesFor(flow, config, bounds, [wx, wy, wz]);
  const fr = cache.fr;
  const wake = cache.src.wake;
  const f = Math.sqrt(2 * Math.log(1 / edge));
  const reach = wake.decayLen * Math.log(1 / recover);
  const rings = [];
  const steps = 14;
  for (let i = 0; i <= steps; i++) {
    const ds = (reach * i) / steps;
    const axes = wakeAxes(wake, ds, fr.length);
    const rY = axes[0] * f;
    const s = wake.sStart + ds;
    rings.push({
      c: [fr.o[0] + fr.ax[0] * s, fr.groundY + wake.hCenter, fr.o[2] + fr.ax[2] * s],
      rY,
      rYlow: Math.min(rY, wake.hCenter),
      rZ: axes[1] * f,
    });
  }
  return {
    rings,
    lateral: fr.la,
    deficit: wake.deficit,
    upwash: wake.upwash,
    hCenter: wake.hCenter,
    sigmaY: wake.sigma0,
    sigmaZ: wake.z0,
    startStation: wake.sStart,
    reach,
    edge,
    recover,
    length: fr.length,
  };
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
  const fr = flowFrame(flow, bounds);
  const src = flow.flowSource || 'front_grid';
  let s = 0;
  let h = 0;
  let z = 0;
  spawnSerial += 1;

  // Seeding is expressed in car-frame terms — station along the axis, height
  // above the tunnel floor, offset across the model — so the built-in car gets
  // real metre positions and an uploaded mesh still gets its own box.
  const along = (f) => fr.nose + fr.length * f;
  const across = (f) => fr.spanZ * (f - 0.5);

  switch (src) {
    case 'smoke_rake':
      {
        // Classic tunnel rake: five heights x nine lateral tubes, entering just
        // upstream of the model. Thin filaments read best with many closely
        // spaced tubes and very little jitter.
        const rows = 5;
        const cols = 9;
        // Advance tube by tube on every spawn. Using index + serial here made
        // both counters step together, so only alternate tubes were ever fed.
        const t = spawnSerial % (rows * cols);
        const row = Math.floor(t / cols);
        const col = t % cols;
        const fy = [0.05, 0.16, 0.3, 0.48, 0.72][row];
        // With solved geometry these are the rake's true tube heights in metres,
        // so the lowest filament runs 30 mm over the floor, not 5% of a box.
        h = (fr.absoluteTubes ? fr.tubes[row] : fr.height * fy)
          + (Math.random() - 0.5) * 0.006 * fr.height;
        s = along(-0.04 - Math.random() * 0.02);
        z = across((col + 0.5) / cols) + (Math.random() - 0.5) * 0.006 * fr.spanZ;
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
          s = along(-0.02 - Math.random() * 0.06);
          h = fr.height * (0.04 + 0.92 * u);
          z = across(0.04 + 0.92 * v);
        } else if (slot < 7) {
          const [u, v] = stratified2(index + 137);
          s = along(u);
          h = fr.height * (1.1 + Math.random() * 0.18);
          z = across(0.04 + 0.92 * v);
        } else if (slot < 9) {
          const [u, v] = stratified2(index + 271);
          s = along(u);
          h = fr.height * (0.05 + 0.9 * v);
          z = across(-0.12 - Math.random() * 0.2);
        } else {
          const [u, v] = stratified2(index + 409);
          s = along(u);
          h = fr.height * (0.05 + 0.9 * v);
          z = across(1.12 + Math.random() * 0.2);
        }
      }
      break;
    case 'smoke_wand':
      {
        s = along(-0.05) + (Math.random() - 0.5) * 0.015;
        const wy = flow.wandY !== undefined ? flow.wandY - fr.groundY : fr.height / 2;
        const wz = flow.wandZ !== undefined ? flow.wandZ - fr.o[2] : 0;
        h = wy + (Math.random() - 0.5) * 0.015;
        z = wz + (Math.random() - 0.5) * 0.015;
      }
      break;
    case 'top':
      {
        const [u, v] = stratified2(index);
        s = along(u);
        h = fr.height * 1.35;
        z = across(0.04 + 0.92 * v);
      }
      break;
    case 'side_left':
      {
        const [u, v] = stratified2(index);
        s = along(u);
        h = fr.height * (0.04 + 0.92 * v);
        z = across(-0.4);
      }
      break;
    case 'side_right':
      {
        const [u, v] = stratified2(index + 61);
        s = along(u);
        h = fr.height * (0.04 + 0.92 * v);
        z = across(1.4);
      }
      break;
    case 'ring':
      {
        const ang = (index / (flow.particleCount || 300)) * Math.PI * 2;
        const r = Math.max(fr.height, fr.spanZ) * 0.9;
        s = along(-0.5);
        h = fr.height * 0.5 + Math.cos(ang) * r * 0.5;
        z = Math.sin(ang) * r * 0.5;
      }
      break;
    case 'front_wide':
      {
        const [u, v] = stratified2(index + 53);
        s = along(-0.02 - Math.random() * 0.06);
        h = fr.height * (0.04 + 0.92 * u);
        z = across(0.04 + 0.92 * v);
      }
      break;
    case 'front_grid':
    default:
      {
        // Stratified curtain: the R2 sequence tiles the inlet plane evenly, so
        // the smoke enters as a continuous sheet instead of random clumps.
        const [u, v] = stratified2(index);
        s = along(-0.01 - Math.random() * 0.08);
        h = fr.height * (0.06 + 0.88 * u);
        z = across(0.05 + 0.9 * v);
      }
  }

  const p = atStation(fr, s, h, z);
  return { x: p[0], y: p[1], z: p[2], life: Math.random() * 2, hit: 0 };
}

export function spawnParticles(bounds, flow) {
  const count = flow.particleCount || 300;
  return Array.from({ length: count }, (_, i) => spawnOneParticle(bounds, flow, i));
}

export function generateStreamlines(config, bounds, flow, velocityFn = velocityAt) {
  const fr = flowFrame(flow, bounds);
  const nLines = flow.streamlineCount || 24;
  const lines = [];
  const layers = 4;

  for (let i = 0; i < nLines; i++) {
    for (let j = 0; j < layers; j++) {
      const t = (i + 0.5) / nLines;
      const zLayer = 0.06 + (j / (layers - 1)) * 0.88;
      let s;
      let h;
      let z;

      if (flow.flowSource === 'top') {
        s = fr.nose + fr.length * t;
        h = fr.height * 1.2;
        z = fr.spanZ * (zLayer - 0.5);
      } else if (flow.flowSource === 'side_left') {
        s = fr.nose + fr.length * zLayer;
        h = fr.height * t;
        z = -fr.spanZ * 0.8;
      } else {
        s = fr.nose - fr.length * 0.08;
        h = fr.height * t;
        z = fr.spanZ * (zLayer - 0.5);
      }

      let [x, y, wob] = atStation(fr, s, h, z);
      const points = [[x, y, wob]];
      for (let step = 0; step < 120; step++) {
        const [vx, vy, vz] = velocityFn(x, y, wob, config, bounds, flow);
        const vmag = len([vx, vy, vz]) || 1;
        const dt = (fr.length * 0.028) / vmag;
        x += vx * dt;
        y += vy * dt;
        wob += vz * dt;
        const along = stationFrac(fr, x, y, wob);
        if (along > 2.2 && flow.flowSource !== 'side_left') break;
        if (y - fr.groundY > 2 * fr.height && flow.flowSource === 'top') break;
        if (lateralOf(fr, x, y, wob) > 2 * fr.spanZ && flow.flowSource === 'side_left') break;
        points.push([x, y, wob]);
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

/**
 * Full-car Reynolds number at the speed the tunnel is showing, on the drawn
 * car length. Display only: the analytical and VLM tiers apply no Re scaling,
 * and the CFD tier reads a chord Reynolds number in its own section case, so
 * this number moves the readout and never the forces.
 */
export function reynoldsNumber(flow, charLengthM = REFERENCE.carLengthM) {
  const v = flow.windSpeedKmh / 3.6;
  return Math.round((REFERENCE.rho * v * charLengthM) / REFERENCE.muAirPaS);
}

/** HUD text for `reynoldsNumber`, with the length the number was taken on. */
export function reynoldsLabel(flow) {
  return `Re=${(reynoldsNumber(flow) / 1e6).toFixed(2)}M (L=${REFERENCE.carLengthM} m)`;
}
