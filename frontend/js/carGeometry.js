/**
 * Builds the built-in formula car as a metric 3D mesh from the aero setup.
 *
 * The four lifting surfaces are drawn from the solver's own station formulas and
 * camber law (see carSpec.js), so wing angle, chord, span, gurney height, ride
 * heights, rake and diffuser angle all move the model. Bodywork, wheels, floor
 * and diffuser are display geometry: the aero model has no chassis, so those are
 * sized to real proportions rather than solved.
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { deriveSurfaces, REFERENCE } from './carSpec.js';

const DEG = Math.PI / 180;

/**
 * Closed wing section in (chordwise, height) space. Camber uses the same
 * parabolic law as backend/simulation/vlm/geometry.py:89 and thickness the
 * standard NACA distribution, so the drawn surface brackets the lifting one.
 */
function wingSection(chord, camber, thickness, points = 13) {
  const upper = [];
  const lower = [];
  for (let i = 0; i <= points; i++) {
    const b = i / points;
    const xi = b * chord;
    // geometry.py: zc = 4 * camber * xc * (1 - xc / c)
    const zc = 4 * camber * xi * (1 - xi / chord);
    const yt =
      5 *
      thickness *
      chord *
      (0.2969 * Math.sqrt(b) - 0.126 * b - 0.3516 * b * b + 0.2843 * b ** 3 - 0.1036 * b ** 4);
    upper.push([xi, zc + yt]);
    lower.push([xi, zc - yt]);
  }
  const shape = new THREE.Shape();
  shape.moveTo(upper[0][0], upper[0][1]);
  for (let i = 1; i < upper.length; i++) shape.lineTo(upper[i][0], upper[i][1]);
  for (let i = lower.length - 1; i >= 0; i--) shape.lineTo(lower[i][0], lower[i][1]);
  shape.closePath();
  return shape;
}

/** Extrude a section along the lateral axis, then place it at its station. */
function extrudedSpan(shape, span, angleDeg, x, y) {
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: span,
    bevelEnabled: false,
    steps: 1,
    curveSegments: 1,
  });
  geo.translate(0, 0, -span / 2);
  // Positive incidence drops the trailing edge (geometry.py:90-91).
  geo.rotateZ(-angleDeg * DEG);
  geo.translate(x, y, 0);
  return geo;
}

function wingGeometry(wing) {
  const parts = [
    extrudedSpan(
      wingSection(wing.chord, wing.camber, wing.thickness),
      wing.span,
      wing.angle,
      wing.x,
      wing.y,
    ),
  ];
  if (wing.gurneyM > 0.0005) {
    const g = new THREE.BoxGeometry(0.004, wing.gurneyM, wing.span);
    // Rotate about the plate's own centre first: translating first would pivot
    // it around the world origin and throw it off the trailing edge.
    g.rotateZ(-wing.angle * DEG);
    const cx = wing.x + wing.chord * Math.cos(wing.angle * DEG);
    const cy = wing.y - wing.chord * Math.sin(wing.angle * DEG) + wing.gurneyM / 2 - 0.002;
    g.translate(cx, cy, 0);
    parts.push(g);
  }
  return parts;
}

/** Rectangular loft: one section per station, corners ordered outward. */
function loft(sections) {
  const pos = [];
  const idx = [];
  for (const s of sections) {
    const hw = s.halfWidth;
    const zc = s.zOff || 0;
    // 0 left-bottom, 1 right-bottom, 2 right-top, 3 left-top
    pos.push(s.x, s.bottom, zc - hw, s.x, s.bottom, zc + hw, s.x, s.top, zc + hw, s.x, s.top, zc - hw);
  }
  const quad = (a, b, c, d) => {
    idx.push(a, b, c, a, c, d);
  };
  for (let i = 0; i < sections.length - 1; i++) {
    const s0 = i * 4;
    const s1 = (i + 1) * 4;
    quad(s0 + 3, s0 + 2, s1 + 2, s1 + 3); // top, +y
    quad(s0 + 0, s1 + 0, s1 + 1, s0 + 1); // bottom, -y
    quad(s0 + 1, s1 + 1, s1 + 2, s0 + 2); // right, +z
    quad(s0 + 0, s0 + 3, s1 + 3, s1 + 0); // left, -z
  }
  quad(0, 1, 2, 3); // nose cap, -x
  const last = (sections.length - 1) * 4;
  quad(last + 0, last + 3, last + 2, last + 1); // tail cap, +x

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setIndex(idx);
  return geo;
}

/** Wheel with its axis along the lateral direction. */
function wheel(x, y, z, r, w) {
  const geo = new THREE.CylinderGeometry(r, r, w, 26, 1, false);
  geo.rotateX(Math.PI / 2);
  geo.translate(x, y, z);
  return geo;
}

function plate(w, h, d, x, y, z) {
  const geo = new THREE.BoxGeometry(w, h, d);
  geo.translate(x, y, z);
  return geo;
}

export function buildCarGeometry(config) {
  const s = deriveSurfaces(config);
  const parts = [];

  // --- Lifting surfaces (solver boundary conditions) -----------------------
  for (const wing of s.wings) parts.push(...wingGeometry(wing));

  // Front wing endplates at the physical tip of the outer element. A plate is
  // centred on its own height, so anchor it by its bottom: hung off the leading
  // edge it crossed the tunnel floor at a 25 mm ride height.
  const fw = s.frontWing.main;
  const flap = s.frontWing.flap;
  const fwTe = flap.x + flap.chord;
  const epHeight = 0.11;
  const epBottom = Math.max(0.002, fw.y - 0.075);
  for (const side of [1, -1]) {
    const z = (side * fw.span) / 2;
    parts.push(
      plate(fwTe - fw.x + 0.06, epHeight, 0.006, (fw.x + fwTe) / 2, epBottom + epHeight / 2, z),
    );
  }

  // --- Wheels -------------------------------------------------------------
  for (const wheelSpec of s.wheels) {
    parts.push(wheel(wheelSpec.x, wheelSpec.y, wheelSpec.z, wheelSpec.r, wheelSpec.w));
  }

  // --- Floor and diffuser -------------------------------------------------
  const f = s.floor;
  const d = s.diffuser;
  const midFloor = f.heightAt((f.xStart + d.xStart) / 2);
  // The plank is 12 mm thick, but a floor line lower than that must not push its
  // underside through the tunnel floor.
  const plank = (x, h) => ({
    x,
    halfWidth: f.halfWidth,
    bottom: Math.max(0.0015, h - 0.012),
    top: h,
  });
  parts.push(
    loft([
      plank(f.xStart, f.front),
      plank((f.xStart + d.xStart) / 2, midFloor),
      plank(d.xStart, f.heightAt(d.xStart)),
    ]),
  );
  parts.push(
    loft([
      { x: d.xStart, halfWidth: f.halfWidth, bottom: d.throatHeight, top: d.throatHeight + 0.014 },
      { x: d.xEnd, halfWidth: f.halfWidth, bottom: d.exitHeight, top: d.exitHeight + 0.018 },
    ]),
  );
  // Strakes: thin fins filling the diffuser channel from the floor ramp up to
  // the underside skin. Lofted station by station, so the opening widens
  // rearwards with the diffuser instead of a rotated box swinging below the
  // tunnel floor.
  for (const z of [-f.halfWidth * 0.55, 0, f.halfWidth * 0.55]) {
    parts.push(
      loft(
        [0, 0.5, 1].map((t) => {
          const x = d.xStart + t * d.length;
          const bottom = Math.max(0.004, f.heightAt(x));
          return {
            x,
            halfWidth: 0.003,
            zOff: z,
            bottom,
            top: Math.max(bottom + 0.012, d.throatHeight + t * (d.exitHeight - d.throatHeight)),
          };
        }),
      ),
    );
  }

  // --- Bodywork (display geometry) ---------------------------------------
  parts.push(
    loft([
      // Tip at carSpec extents.minX: the declared nose, so the drawn body and
      // the flow frame measure the same car.
      { x: 0.02, halfWidth: 0.025, bottom: 0.09, top: 0.15 },
      { x: 0.12, halfWidth: 0.09, bottom: 0.1, top: 0.2 },
      { x: 0.55, halfWidth: 0.12, bottom: 0.13, top: 0.3 },
      { x: 1.1, halfWidth: 0.2, bottom: 0.17, top: 0.42 },
      { x: 1.75, halfWidth: 0.32, bottom: 0.2, top: 0.55 },
      { x: 2.25, halfWidth: 0.34, bottom: 0.22, top: 0.58 },
      { x: 2.8, halfWidth: 0.3, bottom: 0.24, top: 0.56 },
      { x: 3.3, halfWidth: 0.2, bottom: 0.26, top: 0.5 },
      { x: 3.72, halfWidth: 0.08, bottom: 0.28, top: 0.42 },
    ]),
  );
  // Sidepods, one per side.
  for (const side of [1, -1]) {
    parts.push(
      loft([
        { x: 1.5, halfWidth: 0.16, zOff: side * 0.36, bottom: 0.16, top: 0.44 },
        { x: 2.1, halfWidth: 0.19, zOff: side * 0.42, bottom: 0.15, top: 0.46 },
        { x: 2.7, halfWidth: 0.14, zOff: side * 0.38, bottom: 0.18, top: 0.44 },
      ]),
    );
  }
  // Cockpit rim, airbox and driver.
  parts.push(plate(0.62, 0.08, 0.52, 2.05, 0.6, 0));
  parts.push(
    loft([
      { x: 2.28, halfWidth: 0.16, bottom: 0.55, top: 0.62 },
      { x: 2.52, halfWidth: 0.2, bottom: 0.6, top: 0.78 },
      { x: 2.78, halfWidth: 0.12, bottom: 0.5, top: 0.6 },
    ]),
  );
  const helmet = new THREE.SphereGeometry(0.115, 18, 12);
  helmet.translate(1.95, 0.63, 0);
  parts.push(helmet);

  // --- Rear wing supports -------------------------------------------------
  const rw = s.rearWing.main;
  for (const side of [1, -1]) {
    parts.push(plate(0.05, rw.y - 0.5, 0.02, rw.x + 0.1, (rw.y + 0.5) / 2, side * 0.2));
  }
  parts.push(plate(0.06, 0.03, rw.span * 0.72, rw.x + 0.16, rw.y - 0.03, 0));
  const rwFlap = s.rearWing.flap;
  const epCrown = rw.y - REFERENCE.rwEndplateDropM + REFERENCE.rwEndplateHeightM / 2;
  for (const side of [1, -1]) {
    parts.push(
      plate(
        rw.chord + rwFlap.chord + 0.1,
        REFERENCE.rwEndplateHeightM,
        0.005,
        rw.x + 0.05,
        epCrown - REFERENCE.rwEndplateHeightM / 2,
        (side * rw.span) / 2,
      ),
    );
  }

  const merged = mergeGeometries(parts.map(prep), false);
  merged.computeVertexNormals();
  merged.userData.surfaces = s;
  return merged;
}

/**
 * mergeGeometries refuses parts with different attribute sets (the primitives
 * have uv, the lofts do not) or different indexed-ness, and returns null.
 * Everything is flattened here because vertex normals are recomputed after the
 * merge anyway.
 */
function prep(geo) {
  const g = geo.index ? geo.toNonIndexed() : geo;
  for (const name of Object.keys(g.attributes)) {
    if (name.startsWith('uv')) g.deleteAttribute(name);
  }
  if (!g.attributes.normal) g.computeVertexNormals();
  return g;
}
