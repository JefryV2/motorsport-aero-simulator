/**
 * Surface pressure field.
 *
 * Two terms, both from the model:
 *
 *   1. the level, Bernoulli on the inviscid field (1 - (|v|/U)^2), sampled just
 *      outside the skin;
 *   2. the pressure jump across the element that owns the point, +/- dCp/2 on
 *      either face, with dCp from the same low-order terms that seeded the field
 *      (wingCl per lifting element, underbodyTerms.clUnder for floor and
 *      diffuser).
 *
 * A point owned by a loaded element reports term 2 only, never term 1: a sheet
 * induces the same speed on both faces, so sampling the field there measures the
 * element's own singularity, not a pressure. Measured before the split existed,
 * the bare field put the rear wing main at Cp = +0.29 on both faces and the front
 * wing main's split came out *negative*, while the solver prices both as
 * downforce. Everything the model does not solve as a load - body, wheels,
 * endplates, the tunnel walls - keeps the field level, and so does skin that is
 * edge on to an element's own load: a tip or nose cap sees neither side of its
 * jump.
 *
 * Not a CFD result: no Reynolds solver ran, and the load terms are the same
 * semi-empirical ones the rest of the app is built on.
 */

import {
  DEFAULT_FLOW,
  flowFrame,
  potentialVelocityAt,
  speedToColor,
  underbodyTerms,
  wingCl,
} from './flowEngine.js';

const DEG = Math.PI / 180;

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

function normalize(v) {
  const l = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/** Bernoulli from the sampled speed; the side-averaged pressure level. */
export function cpFromSpeed(speedMs, refMs) {
  const r = speedMs / (refMs || 1);
  return 1.0 - r * r;
}

/**
 * Elements whose load the model knows, in the field's own station frame.
 * Returns null for an uploaded model: it has no solved geometry, so no element
 * may claim a pressure jump.
 */
function pressureElements(fr) {
  const geo = fr.geo;
  if (!geo?.lines?.length) return null;
  const byId = {};
  for (const w of geo.aero?.wings || []) byId[w.id] = w;

  const panels = [];
  for (const l of geo.lines) {
    const angle = (l.angleDeg || 0) * DEG;
    const w = byId[l.id];
    // Stations and heights in the frame's own 1-D coordinates, never world
    // points: a yawed or lifted model has to land in the same rectangles.
    panels.push({
      dCp: w ? wingCl(w) : 0,
      s: l.station,
      h: l.height,
      cos: Math.cos(angle),
      sin: Math.sin(angle),
      // World sheet normal: chord direction rotated 90 deg toward car-local up,
      // which is world up because the model is only ever yawed and scaled.
      nx: Math.sin(angle) * fr.ax[0],
      ny: Math.cos(angle),
      nz: Math.sin(angle) * fr.ax[2],
      chord: l.chord,
      halfSpan: l.halfSpan,
    });
  }

  const f = geo.floor;
  const d = geo.diffuser;
  const span = Math.max(d.end - f.start, 1e-6);
  const underbody = {
    dCp: underbodyTerms(geo.aero).clUnder,
    sStart: f.start,
    sEnd: d.end,
    halfWidth: f.halfWidth,
    span,
    // Height of the floor line the plank is drawn against: the flat ramp up to
    // the diffuser start, then the diffuser throat-to-exit line.
    heightAt(s) {
      if (s < d.start) {
        const t = Math.min(1, Math.max(0, (s - f.start) / Math.max(d.start - f.start, 1e-6)));
        return f.heightStart + t * (f.heightEnd - f.heightStart);
      }
      const u = Math.min(1, Math.max(0, (s - d.start) / Math.max(d.end - d.start, 1e-6)));
      return d.throat + u * (d.exit - d.throat);
    },
  };
  return { panels, underbody };
}

/**
 * Which loaded element owns this skin point, and how strongly the face sees the
 * jump. `cosFace` is the skin normal against the sheet normal: +1 on the face
 * pointing away from the ground (compression for a download element), -1 on the
 * face pointing at it (suction), ~0 on a tip, endplate or nose cap that is edge
 * on to the load and has no pressure jump of its own.
 */
function elementLoadAt(els, s, h, lat, n) {
  let best = null;
  for (const p of els.panels) {
    const ds = s - p.s;
    const dh = h - p.h;
    const along = ds * p.cos - dh * p.sin;
    const perp = ds * p.sin + dh * p.cos;
    // Quarter-chord line, leading edge at -0.25c, trailing edge at +0.75c.
    if (along < -0.3 * p.chord || along > 0.8 * p.chord) continue;
    if (Math.abs(perp) > 0.3 * p.chord) continue;
    if (Math.abs(lat) > 1.02 * p.halfSpan) continue;
    // The flap hinges inside the main's band, so the nearest sheet owns the
    // point - not the first rectangle that happens to contain it.
    if (!best || Math.abs(perp) < best.perp) {
      best = {
        dCp: p.dCp,
        perp: Math.abs(perp),
        cosFace: n[0] * p.nx + n[1] * p.ny + n[2] * p.nz,
      };
    }
  }
  if (best) return best;
  const u = els.underbody;
  const cosFace = n[1];
  if (cosFace < -0.3 && lat <= u.halfWidth && lat >= -u.halfWidth
    && s >= u.sStart - 0.05 * u.span && s <= u.sEnd + 0.05 * u.span
    && h <= u.heightAt(s) + 0.02) {
    return { dCp: u.dCp, cosFace };
  }
  return null;
}

/**
 * Cp of a point the model has solved as a lifting element: the panel-method
 * jump condition, +/- dCp/2 on the face the load points away from. Null when no
 * element claims the point or the face is edge on, which is when the field
 * level describes it instead.
 */
function loadCpAt(els, v, n, fr, o) {
  const dx = v[0] - o[0];
  const dy = v[1] - o[1];
  const dz = v[2] - o[2];
  const s = dx * fr.ax[0] + dy * fr.ax[1] + dz * fr.ax[2];
  const lat = dx * fr.la[0] + dy * fr.la[1] + dz * fr.la[2];
  const load = elementLoadAt(els, s, dy, lat, n);
  // A wing at 0 degrees with no gurney carries nothing, and a face edge on to
  // the load sees neither side's jump: both keep the field level.
  if (!load || !load.dCp || Math.abs(load.cosFace) < 0.25) return null;
  return load.cosFace * load.dCp * 0.5;
}

export function computeCpField(vertices, normals, config, bounds, flow = DEFAULT_FLOW) {
  const ref = (flow.windSpeedKmh || 200) / 3.6;
  const fr = flowFrame(flow, bounds);
  const els = pressureElements(fr);
  // Offset from the model's own size: flow.geo makes the bounding box
  // irrelevant, and a box-sized offset would sit a shallow-chord panel away
  // from the skin it is meant to sample.
  const eps = Math.max(fr.length, fr.height, fr.halfZ * 2) * 0.004;
  const o = fr.o;

  const cpValues = [];
  let actualMin = Infinity;
  let actualMax = -Infinity;

  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i];
    const n = normalize(normals?.[i] || [0, 1, 0]);
    let cp = els ? loadCpAt(els, v, n, fr, o) : null;
    if (cp === null) {
      // Just outside the skin, so the sample is in the tunnel not in the model.
      const [vx, vy, vz] = potentialVelocityAt(
        v[0] + n[0] * eps, v[1] + n[1] * eps, v[2] + n[2] * eps, config, bounds, flow,
      );
      cp = cpFromSpeed(Math.sqrt(vx * vx + vy * vy + vz * vz), ref);
    }

    cpValues.push(cp);
    if (cp < actualMin) actualMin = cp;
    if (cp > actualMax) actualMax = cp;
  }

  const colors = cpValues.map(
    (cp) => cpToColor(Math.max(CP_MIN, Math.min(CP_MAX, cp))),
  );

  return {
    cp: cpValues,
    colors,
    range: {
      min: Number.isFinite(actualMin) ? actualMin : CP_MIN,
      max: Number.isFinite(actualMax) ? actualMax : CP_MAX,
      scaleMin: CP_MIN,
      scaleMax: CP_MAX,
    },
    // Whether the jump term ran: an uploaded model has no solved elements.
    loadsApplied: Boolean(els),
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
