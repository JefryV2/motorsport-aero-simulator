/**
 * Built-in car geometry, in true metres, derived from the same setup fields the
 * solver uses. This is the single source of truth for both the drawn body
 * (carGeometry.js) and the wind-tunnel singularities (flowEngine.js), so the
 * visualization cannot drift away from the configuration that produced the
 * numbers.
 *
 * Axis convention here matches the frontend: +x forward, +y up, +z lateral.
 * The backend VLM uses [x, y=lateral, z=up]; the station formulas below are
 * transcribed from backend/simulation/analytical.py:wing_stations and
 * backend/simulation/vlm/geometry.py:transform_point, with that swap applied.
 *
 * Only the four lifting surfaces and the two ride heights are aerodynamic
 * boundary conditions of the solver. Everything else here (bodywork, wheels,
 * floor planform) is display geometry sized to real proportions, because the
 * aero model has no chassis representation at all.
 */

// Mirrors the module constants in backend/simulation/analytical.py (RHO,
// REF_AREA, WHEELBASE_M, DRS_FLAP_ANGLE_DEG) and the stations in
// backend/simulation/analytical.py:wing_stations.
export const REFERENCE = {
  wheelbaseM: 3.35,
  refAreaM2: 1.5,
  rho: 1.225,
  // Solver stations: the front wing leading edge sits at x = 0.20 and the rear
  // wing at x = wheelbase + 0.20. Axles are placed so those stations fall where
  // they do on a real car: the wing is ahead of the front axle, the rear wing
  // just ahead of the rear axle.
  fwLeadingEdgeM: 0.2,
  fwFlapStationFrac: 0.75, // wing_stations: fw_flap origin = main chord * 0.75
  fwFlapGapM: 0.035, // wing_stations: z_fw_flap = z_fw_main + 0.035
  rwFlapStationFrac: 0.75, // wing_stations: rw_flap origin = main chord * 0.75
  rwFlapRiseM: 0.05, // wing_stations: rw_flap sits 0.05 above the main plane
  rwMainHeightM: 0.85, // wing_stations: rear wing main origin height
  // The rear-wing endplate is the highest point on the car, so the declared
  // overall height is derived from these same two numbers and carGeometry.js
  // draws the plate from them: they cannot drift apart.
  rwEndplateHeightM: 0.36,
  rwEndplateDropM: 0.12, // plate centre sits this far below the main plane
  frontAxleM: 0.55,
  rearAxleM: 3.9,
  // Camber height ratios used by the VLM surfaces (panel_method._VLM_CAMBER args).
  camber: {
    front_wing_main: 0.03,
    front_wing_flap: 0.05,
    rear_wing_main: 0.04,
    rear_wing_flap: 0.06,
  },
  thickness: { front_wing_main: 0.11, front_wing_flap: 0.09, rear_wing_main: 0.1, rear_wing_flap: 0.08 },
  // Display-only chassis dimensions. The aero model has no track width.
  trackM: 1.6,
  wheelRadiusFrontM: 0.33,
  wheelRadiusRearM: 0.35,
  wheelWidthFrontM: 0.3,
  wheelWidthRearM: 0.4,
  floorHalfWidthM: 0.7,
  floorXStartM: 0.62,
  floorXEndM: 3.62,
  // The aero model has no diffuser length at all (analytical.py prices the
  // underbody over a fixed FLOOR_AREA_M2), so this is a display dimension that
  // turns the throat height and expansion angle into an exit plane.
  diffuserLengthM: 0.6,
  // The drawn exit plane stops here; analytical.DIFFUSER_EXIT_MAX_M is the same
  // limit and the pressure map ends its diffuser region at that height.
  diffuserExitMaxM: 0.32,
  // Constants the analytic tunnel field mirrors so it moves with the same
  // setup fields the reported numbers do.
  floorAreaM2: 2.2, // analytical.FLOOR_AREA_M2: effective underbody area
  bodyCd: 0.75, // analytical.CD_BODY: wheels, suspension, cockpit as one lump
  profileCd: 0.008, // _wing_forces profile drag term
  // The underbody law `flowEngine.underbodyTerms` reproduces, so the drawn wake
  // answers to the setup the forces charge. `analytical.underbody_cl` is the one
  // reader of the first six on the Python side; the last three are the drag terms
  // `_underbody_forces` adds over the same ge_rear.
  geDecayFrontM: 0.015, // analytical.GE_DECAY_FRONT_M
  geDecayRearM: 0.02, // analytical.GE_DECAY_REAR_M
  geDecayDiffuserM: 0.025, // analytical.GE_DECAY_DIFFUSER_M
  rakeCoefPerDeg: 0.08, // analytical.RAKE_COEF_PER_DEG
  clFloorMax: 0.35, // analytical.CL_FLOOR_MAX
  clDiffuserMax: 0.25, // analytical.CL_DIFFUSER_MAX
  cdFloorBase: 0.015, // analytical.CD_FLOOR_BASE
  cdFloorGapCoef: 0.02, // analytical.CD_FLOOR_GAP_COEF
  cdDiffuserBase: 0.008, // analytical.CD_DIFFUSER
  // Wing ground effect: analytical.ground_effect_lift_factor prices the lift
  // gain at each section's trailing-edge clearance, so the front ride height
  // moves the wings as well as the floor. The gain is anchored so the default
  // front wing as a whole keeps the flat 1.12 this model used to apply.
  geLiftGainMax: 0.162, // analytical.GE_LIFT_GAIN_MAX
  geLiftDecayM: 0.14, // analytical.GE_LIFT_DECAY_M
  drsFlapAngleDeg: 2.0, // analytical.DRS_FLAP_ANGLE_DEG: rear flap when DRS is open
  // Reynolds readout. `carLengthM` is the drawn nose-to-tail envelope,
  // rearAxleM + wheelRadiusRearM - the same analytical.CAR_LENGTH_M the backend
  // normalises uploaded meshes with. `muAirPaS` is the dynamic viscosity of air
  // at 15 °C. Re is a readout of the tunnel speed: none of the force tiers reads
  // it, only the OpenFOAM section case, which uses its own chord and
  // nu = 1.5e-5 m²/s (1.5 % from muAirPaS / rho).
  carLengthM: 4.25,
  muAirPaS: 1.81e-5,
  // Lap-time proxy: analytical.lap_time_delta_s is the only implementation and
  // these are its four numbers. The tiers report against lapBaseline*; the UI
  // passes the setup it loaded as the baseline and the track's weighting.
  lapBaselineDragN: 1200.0, // analytical.LAP_BASELINE_DRAG_N
  lapBaselineDfN: 3500.0, // analytical.LAP_BASELINE_DF_N
  lapSecPerDragN: 0.0001, // analytical.LAP_SEC_PER_DRAG_N
  lapSecPerDfN: 0.00008, // analytical.LAP_SEC_PER_DF_N
  // Post-stall branch of analytical._wing_cl. The thin-airfoil line grows with
  // angle forever, so past the angle where it reaches sectionClMax the section
  // is handed over to the flat-plate polar (normal force 2 sin a), and the
  // handover finishes at stallBlendMaxDeg - the plate's own lift peak. Below it
  // the blend weight is zero and the element prices exactly as before.
  sectionClMax: 1.8, // analytical.SECTION_CL_MAX
  stallBlendMaxDeg: 45.0, // analytical.STALL_BLEND_MAX_DEG
  flatPlateCn: 2.0, // analytical.FLAT_PLATE_CN, normal-force coefficient of a plate
};

const DEG = Math.PI / 180;

function num(v, fallback) {
  return Number.isFinite(v) ? v : fallback;
}

/** Element defaults from the FormulaAeroConfig WingElement factories
 * (backend/models/aero_config.py). Exported because `app.js` builds the request
 * payload from these chord/span values - the UI has no chord or span control, so
 * restating them there would be a third copy of a number that cannot be set. */
export const DEFAULT_ELEMENT = {
  front_wing_main: { angle_deg: 2, chord_m: 0.35, span_m: 1.8, gurney_mm: 0 },
  front_wing_flap: { angle_deg: 8, chord_m: 0.12, span_m: 1.75, gurney_mm: 0 },
  rear_wing_main: { angle_deg: 5, chord_m: 0.4, span_m: 0.95, gurney_mm: 0 },
  rear_wing_flap: { angle_deg: 15, chord_m: 0.15, span_m: 0.9, gurney_mm: 0 },
};

/**
 * Rear flap incidence the solver actually uses (analytical.effective_rear_flap).
 * DRS folds the flap toward the main plane; a flap already below that angle
 * cannot be reduced further, so DRS is a no-op.
 */
export function rearFlapAngleDeg(config) {
  const angle = num(config?.rear_wing_flap?.angle_deg, DEFAULT_ELEMENT.rear_wing_flap.angle_deg);
  return config?.drs_active ? Math.min(angle, REFERENCE.drsFlapAngleDeg) : angle;
}

/**
 * Lap-time proxy in seconds (analytical.lap_time_delta_s): a linear sensitivity
 * of one car between two setups, not a track simulation. `baseline` is the force
 * pair the number is measured against - REFERENCE.lapBaseline* for the solver
 * tiers, the setup the session loaded for the UI - and `track` scales the two
 * terms by the circuit's weighting. Negative means the compared setup is faster
 * on this proxy.
 */
export function lapTimeDeltaS(forces, baseline, track = null) {
  if (!forces || !baseline) return 0.0;
  const dragScale = num(track?.dragMultiplier, 1.0);
  const dfScale = num(track?.dfMultiplier, 1.0);
  return (
    (forces.drag_N - baseline.drag_N) * REFERENCE.lapSecPerDragN * dragScale -
    (forces.downforce_total_N - baseline.downforce_total_N) * REFERENCE.lapSecPerDfN * dfScale
  );
}

function element(config, name) {
  const src = config?.[name] || {};
  const d = DEFAULT_ELEMENT[name];
  const angle = name === 'rear_wing_flap' ? rearFlapAngleDeg(config) : num(src.angle_deg, d.angle_deg);
  return {
    angle,
    chord: Math.max(0.02, num(src.chord_m, d.chord_m)),
    span: Math.max(0.05, num(src.span_m, d.span_m)),
    gurneyM: Math.max(0, num(src.gurney_mm, d.gurney_mm)) / 1000,
    camber: REFERENCE.camber[name],
    thickness: REFERENCE.thickness[name],
  };
}

/**
 * Height of a wing's leading edge, exactly as the solver computes it
 * (panel_method.py:38-39). Positive incidence drops the trailing edge.
 */
function wingStations(config) {
  const hf = num(config?.ride_height_front_mm, 25) / 1000;
  const fwMain = element(config, 'front_wing_main');
  const fwFlap = element(config, 'front_wing_flap');
  const rwMain = element(config, 'rear_wing_main');
  const rwFlap = element(config, 'rear_wing_flap');

  const zFwMain = hf + fwMain.chord * Math.max(0, Math.sin(fwMain.angle * DEG)) + 0.005;
  const zFwFlap = zFwMain + REFERENCE.fwFlapGapM + fwFlap.chord * Math.max(0, Math.sin(fwFlap.angle * DEG));
  const xFwMain = REFERENCE.fwLeadingEdgeM;
  const xFwFlap = xFwMain + fwMain.chord * REFERENCE.fwFlapStationFrac;

  const xRwMain = REFERENCE.wheelbaseM + REFERENCE.fwLeadingEdgeM;
  const xRwFlap = xRwMain + rwMain.chord * REFERENCE.rwFlapStationFrac;

  return [
    { ...fwMain, id: 'front_wing_main', x: xFwMain, y: zFwMain },
    { ...fwFlap, id: 'front_wing_flap', x: xFwFlap, y: zFwFlap },
    { ...rwMain, id: 'rear_wing_main', x: xRwMain, y: REFERENCE.rwMainHeightM },
    { ...rwFlap, id: 'rear_wing_flap', x: xRwFlap, y: REFERENCE.rwMainHeightM + REFERENCE.rwFlapRiseM },
  ];
}

/** Floor line: the two ride heights at the two axles define the rake ramp. */
function floorRake(config) {
  const hf = num(config?.ride_height_front_mm, 25) / 1000;
  const hr = num(config?.ride_height_rear_mm, 45) / 1000;
  const dx = REFERENCE.rearAxleM - REFERENCE.frontAxleM;
  const slope = (hr - hf) / Math.max(dx, 0.01);
  const at = (x) => hf + slope * (x - REFERENCE.frontAxleM);
  return {
    slope,
    angleDeg: Math.atan(slope) / DEG,
    heightAt: at,
    front: at(REFERENCE.floorXStartM),
    rear: at(REFERENCE.floorXEndM),
  };
}

/**
 * Diffuser. The solver prices `diffuser_ride_height_mm` as a suction decay
 * height (`analytical._underbody_forces`, exp(-h / GE_DECAY_DIFFUSER_M)) rather
 * than as an exit plane, and `diffuser_angle_deg` only enters through
 * sin(angle). Geometrically the consistent reading is: that height is the
 * throat, and the angle is the ramp slope over a fixed expansion length, so
 * both sliders move the drawn exit plane in the direction the force model
 * rewards. `analytical.diffuser_throat_height_m` now raises a throat set below
 * the floor line here too, so the drawn shape and the priced one are one rule.
 */
function diffuserGeometry(config, floor) {
  const xEnd = REFERENCE.floorXEndM;
  const xStart = xEnd - REFERENCE.diffuserLengthM;
  const throat = Math.max(num(config?.diffuser_ride_height_mm, 30) / 1000, floor.heightAt(xStart));
  const angle = Math.min(20, Math.max(5, num(config?.diffuser_angle_deg, 12))) * DEG;
  const exitHeight = Math.min(throat + REFERENCE.diffuserLengthM * Math.tan(angle), REFERENCE.diffuserExitMaxM);
  return {
    xStart,
    xEnd,
    length: REFERENCE.diffuserLengthM,
    throatHeight: throat,
    exitHeight: Math.max(exitHeight, throat + 0.01),
    angleDeg: angle / DEG,
  };
}

/**
 * The five numbers `flowEngine.underbodyTerms` reads, straight from the setup
 * sliders and already in TRUE metres. One place owns the defaults and the mm ->
 * m conversion, so a tier that has no solved geometry still answers the sliders
 * with the same law as one that has.
 *
 * `throatM` is the diffuser throat: the built-in car passes `diffuserGeometry`'s
 * height, which has already been raised to the floor line exactly as
 * `analytical.diffuser_throat_height_m` does. A caller with no floor line (the
 * uploaded-CAD tier) passes null, and the slider stands on its own.
 */
export function underbodyInputs(config, throatM = null) {
  return {
    frontRideHeightM: num(config?.ride_height_front_mm, 25) / 1000,
    rearRideHeightM: num(config?.ride_height_rear_mm, 45) / 1000,
    diffuserThroatM: throatM === null ? num(config?.diffuser_ride_height_mm, 30) / 1000 : throatM,
    diffuserAngleDeg: num(config?.diffuser_angle_deg, 12),
    rakeDeg: num(config?.rake_deg, 1.5),
  };
}

/**
 * Singularity stations for the analytic tunnel field, in car-local metres.
 * Kept as plain numbers so it can cross into flowEngine without importing
 * Three.js.
 */
export function deriveSurfaces(config) {
  const wings = wingStations(config);
  const floor = floorRake(config);
  const diffuser = diffuserGeometry(config, floor);
  const by = (id) => wings.find((w) => w.id === id);
  const fwMain = by('front_wing_main');
  const fwFlap = by('front_wing_flap');
  const rwMain = by('rear_wing_main');
  const rwFlap = by('rear_wing_flap');

  const rf = REFERENCE.wheelRadiusFrontM;
  const rr = REFERENCE.wheelRadiusRearM;
  const zw = REFERENCE.trackM / 2;

  return {
    wings,
    // Bound-vortex lines at quarter chord, the same station the VLM uses for
    // its horseshoe bound vortices (geometry.py:82).
    liftingLines: wings.map((w) => ({
      id: w.id,
      x: w.x + w.chord * 0.25 * Math.cos(w.angle * DEG),
      y: w.y - w.chord * 0.25 * Math.sin(w.angle * DEG),
      halfSpan: w.span / 2,
      chord: w.chord,
      span: w.span,
      angle: w.angle,
      gurneyM: w.gurneyM,
    })),
    wheels: [
      { x: REFERENCE.frontAxleM, y: rf, z: zw - REFERENCE.wheelWidthFrontM / 2, r: rf, w: REFERENCE.wheelWidthFrontM },
      { x: REFERENCE.frontAxleM, y: rf, z: -(zw - REFERENCE.wheelWidthFrontM / 2), r: rf, w: REFERENCE.wheelWidthFrontM },
      { x: REFERENCE.rearAxleM, y: rr, z: zw - REFERENCE.wheelWidthRearM / 2, r: rr, w: REFERENCE.wheelWidthRearM },
      { x: REFERENCE.rearAxleM, y: rr, z: -(zw - REFERENCE.wheelWidthRearM / 2), r: rr, w: REFERENCE.wheelWidthRearM },
    ],
    chassis: [
      { x: 0.3, y: 0.16, r: 0.13 },
      { x: 1.15, y: 0.28, r: 0.2 },
      { x: 2.1, y: 0.36, r: 0.3 },
      { x: 2.75, y: 0.34, r: 0.28 },
      { x: 3.5, y: 0.3, r: 0.2 },
    ],
    frontWing: { main: fwMain, flap: fwFlap },
    rearWing: { main: rwMain, flap: rwFlap },
    floor: { ...floor, xStart: REFERENCE.floorXStartM, xEnd: REFERENCE.floorXEndM, halfWidth: REFERENCE.floorHalfWidthM },
    diffuser,
    ground: { y: 0 },
    // Everything the analytic tunnel field mirrors from the low-order model, in
    // TRUE metres: its ground-effect decay constants are calibrated lengths, so
    // the viewer's display scale must not stretch them.
    aero: {
      wings: wings.map((w) => ({
        id: w.id,
        angleDeg: w.angle,
        chordM: w.chord,
        spanM: w.span,
        gurneyM: w.gurneyM,
        // Ground-effect gap: the lowest point of the section over the plane,
        // analytical.WingStation.min_clearance_m. Positive incidence drops the
        // trailing edge by chord * sin(angle); the rear wing is null because the
        // model leaves it in the freestream. TRUE metres, never the display scale.
        clearanceM: w.id.startsWith('front_wing')
          ? w.y - w.chord * Math.max(0, Math.sin(w.angle * DEG))
          : null,
      })),
      refAreaM2: REFERENCE.refAreaM2,
      floorAreaM2: REFERENCE.floorAreaM2,
      bodyCd: REFERENCE.bodyCd,
      profileCd: REFERENCE.profileCd,
      trackM: REFERENCE.trackM,
      rwHeightM: REFERENCE.rwMainHeightM,
      ...underbodyInputs(config, diffuser.throatHeight),
    },
    rakeTubeHeightsM: [0.03, 0.08, 0.16, 0.32, 0.62],
    extents: {
      minX: 0.02,
      maxX: REFERENCE.rearAxleM + REFERENCE.wheelRadiusRearM,
      minY: 0,
      // Highest drawn point. A wing section is drawn centred on its own line and
      // pitched down at positive incidence, so its top is its leading edge plus
      // the half-thickness - never a whole chord above it.
      maxY: Math.max(
        REFERENCE.rwMainHeightM - REFERENCE.rwEndplateDropM + REFERENCE.rwEndplateHeightM / 2,
        rwMain.y + 0.5 * rwMain.thickness * rwMain.chord,
        rwFlap.y + 0.5 * rwFlap.thickness * rwFlap.chord,
      ),
      halfZ: Math.max(fwMain.span, REFERENCE.trackM) / 2,
    },
  };
}

/**
 * Map car-local metres into the flow field's world frame.
 *
 * The viewer derives `origin`, `axis`, `lateral` and `scale` from the drawn
 * mesh's world matrix, so the singularities sit on the body it actually
 * rendered. This lives next to the spec rather than in the viewer because
 * scripts/check_flow_field.mjs has to build the same object: the alternative
 * was a copy of the mapping in the test, which could drift silently.
 *
 * Positions and lengths come out in world metres; `aero` stays in TRUE metres,
 * because the ground-effect decay constants it mirrors (exp(-h/0.015)) are
 * calibrated lengths and must not stretch with the user's display scale.
 */
export function toFlowGeo(s, { origin, axis, lateral, groundY, scale }) {
  const f = s.floor;
  const d = s.diffuser;
  const m = (v) => v * scale;
  return {
    origin,
    axis,
    lateral,
    // The model is only ever scaled and yawed, so car-local up is world up.
    groundY,
    scale,
    length: m(s.extents.maxX - s.extents.minX),
    height: m(s.extents.maxY),
    halfWidth: m(s.extents.halfZ),
    lines: s.liftingLines.map((l) => ({
      id: l.id,
      station: m(l.x),
      height: m(l.y),
      halfSpan: m(l.halfSpan),
      chord: m(l.chord),
      angleDeg: l.angle,
      gurneyM: m(l.gurneyM),
    })),
    wheels: s.wheels.map((w) => ({
      station: m(w.x),
      height: m(w.y),
      lateral: m(w.z),
      radius: m(w.r),
      halfWidth: m(w.w / 2),
    })),
    chassis: s.chassis.map((c) => ({
      station: m(c.x),
      height: m(c.y),
      radius: m(c.r),
    })),
    floor: {
      start: m(f.xStart),
      end: m(f.xEnd),
      heightStart: m(f.front),
      heightEnd: m(f.rear),
      halfWidth: m(f.halfWidth),
    },
    diffuser: {
      start: m(d.xStart),
      end: m(d.xEnd),
      throat: m(d.throatHeight),
      exit: m(d.exitHeight),
    },
    stations: {
      nose: m(s.extents.minX),
      tail: m(s.extents.maxX),
      top: m(s.extents.maxY),
    },
    rakeHeightsM: s.rakeTubeHeightsM.map(m),
    aero: s.aero,
  };
}

/** Front-axle-referenced geometry the tunnel field needs, ready to be mapped. */
export function summarize(surfaces) {
  const fw = surfaces.frontWing.main;
  const rw = surfaces.rearWing.main;
  return {
    lengthM: surfaces.extents.maxX - surfaces.extents.minX,
    heightM: surfaces.extents.maxY,
    widthM: surfaces.extents.halfZ * 2,
    frontWingAreaM2: fw.chord * fw.span + surfaces.frontWing.flap.chord * surfaces.frontWing.flap.span,
    rearWingAreaM2: rw.chord * rw.span + surfaces.rearWing.flap.chord * surfaces.rearWing.flap.span,
    rakeDeg: surfaces.floor.angleDeg,
    diffuserLengthM: surfaces.diffuser.length,
  };
}
