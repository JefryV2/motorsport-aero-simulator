/** Main app — vehicle setup + wind tunnel flow controls. */

import { AeroViewer } from './viewer3d.js';
import { DEFAULT_ELEMENT, DEFAULT_SETUP, deriveSurfaces, rearFlapAngleDeg, lapTimeDeltaS } from './carSpec.js';
import { reynoldsLabel } from './flowEngine.js';
import {
  engineeringContext,
  compareToBaseline,
  renderEngineerPanel,
  exportSetupJson,
  exportResultsCsv,
  dynamicPressurePa,
} from './engineerTools.js';

const API = '';
let viewer = null;
let lastSim = null;
let baselineSim = null;
const referenceByFidelity = new Map();
let debounceTimer = null;

const TRACK_DATABASE = {
  balanced: {
    name: "Silverstone",
    dragMultiplier: 1.0,
    dfMultiplier: 1.0,
    description: "Balanced medium-high downforce circuit. High-speed corners."
  },
  monza: {
    name: "Monza",
    dragMultiplier: 2.2,
    dfMultiplier: 0.35,
    description: "Temple of Speed. Ultra-low drag is required."
  },
  monaco: {
    name: "Monaco",
    dragMultiplier: 0.15,
    dfMultiplier: 2.4,
    description: "Tight street circuit. Maximum downforce setup."
  },
  spa: {
    name: "Spa-Francorchamps",
    dragMultiplier: 1.3,
    dfMultiplier: 0.95,
    description: "Speed in Sector 1/3, cornering downforce in Sector 2."
  },
  singapore: {
    name: "Singapore",
    dragMultiplier: 0.6,
    dfMultiplier: 1.8,
    description: "Street circuit. High downforce and stability."
  }
};

// `drs_active` is a model field the backend fills, so the starting state reads
// its default instead of copying it - a copy can move away from the model alone.
let drsActive = DEFAULT_SETUP.drs_active;
let aeroMapGrid = null;
let aeroMapFrhVals = null;
let aeroMapRrhVals = null;

let cfdBackend = 'unknown';
let simulateInFlight = false;
let lastImportedModelId = null;

const FIDELITY_META = {
  empirical: {
    label: 'EMPIRICAL',
    cls: 'empirical',
    short: 'Semi-empirical correlation',
    validation: 'internal trend tests only',
  },
  low_fidelity_vlm: {
    label: 'VLM PHYSICS',
    cls: 'vlm',
    short: '3D vortex lattice + image ground',
    validation: 'V1–V3 internal; V4 misses external reference',
  },
  synthetic_model: {
    label: 'SYNTHETIC MODEL',
    cls: 'synthetic',
    short: 'Calibrated closed-form fallback',
    validation: 'not validated against real CFD',
  },
  rans_hybrid: {
    label: 'RANS + EMPIRICAL',
    cls: 'hybrid',
    short: '2D OpenFOAM front-wing RANS + empirical floor/rear',
    validation: 'solver convergence parsed; total is a hybrid estimate',
  },
  rans: {
    label: 'OPENFOAM RANS',
    cls: 'rans',
    short: 'Mesh solution (simpleFoam)',
    validation: 'solver convergence parsed from logs',
  },
};

function fidelityFor(sim) {
  if (!sim) return null;
  if (sim.fidelity && FIDELITY_META[sim.fidelity]) return sim.fidelity;
  const method = String(sim.method || '').toLowerCase();
  if (method.includes('synthetic')) return 'synthetic_model';
  if (method.includes('openfoam')) return 'rans_hybrid';
  if (method.includes('panel') || method.includes('vlm') || method.includes('enhanced')) return 'low_fidelity_vlm';
  return 'empirical';
}

function isCfdMethod(method = document.getElementById('method')?.value) {
  return method === 'cfd' || method === 'cfd_stub';
}

function syncMethodTabs() {
  const method = document.getElementById('method')?.value;
  const cfd = isCfdMethod(method);
  document.querySelectorAll('.method-tab').forEach((tab) => {
    const value = tab.dataset.method;
    let active = false;
    if (value === 'cfd') active = cfd;
    else if (value === 'analytical') active = method === 'analytical';
    else active = method === 'analytical_enhanced' || method === 'panel_vlm';
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
}

function setMethod(value) {
  const select = document.getElementById('method');
  if (!select) return;
  if (!Array.from(select.options).some((option) => option.value === value)) return;
  select.value = value;
  syncMethodTabs();
  updateMethodNote();

  const status = document.getElementById('status');
  if (isCfdMethod(value)) {
    if (status) {
      status.textContent = 'CFD selected — press Run CFD to start (about one minute).';
      status.className = 'status';
    }
  } else {
    runSimulate();
  }
}

function setFidelityBadge(key) {
  const badge = document.getElementById('fidelityBadge');
  const line = document.getElementById('provenanceLine');
  const meta = key ? FIDELITY_META[key] : null;

  if (badge) {
    badge.textContent = meta ? meta.label : 'NO DATA';
    badge.className = 'fidelity-badge' + (meta ? ' ' + meta.cls : '');
  }

  if (!line) return;
  if (!meta) {
    line.innerHTML = '<strong>Source:</strong> not run yet · <strong>Fidelity:</strong> — · <strong>Validation:</strong> —';
    return;
  }
  line.innerHTML =
    `<strong>Source:</strong> ${meta.short} · ` +
    `<strong>Fidelity:</strong> ${meta.label.toLowerCase()} · ` +
    `<strong>Validation:</strong> ${meta.validation}`;
  if (key === 'synthetic_model') {
    line.innerHTML += ' · <span style="color:#ffb020">no CFD solver ran</span>';
  }
  if (key === 'rans_hybrid') {
    line.innerHTML += ' · <span style="color:#6ee7b7">2D section + empirical floor/rear</span>';
  }
}

function applyHealth(health) {
  cfdBackend = health?.cfd_backend || 'unknown';
  const banner = document.getElementById('provenanceBanner');
  const title = document.getElementById('pbTitle');
  const text = document.getElementById('pbText');
  const cfdOption = document.querySelector('#method option[value="cfd"]');
  const legacyCfdOption = document.querySelector('#method option[value="cfd_stub"]');
  const solverVersion = health?.solver_version || 'OpenFOAM';
  const solverInstalled = cfdBackend === 'native' || cfdBackend === 'wsl';

  if (cfdOption) {
    cfdOption.textContent = solverInstalled
      ? `CFD — OpenFOAM ${solverVersion.replace(/^OpenFOAM\s*/i, '')} RANS + empirical`
      : 'CFD — calibrated fallback (no OpenFOAM)';
  }
  if (legacyCfdOption) legacyCfdOption.textContent = 'CFD — calibrated fallback (legacy)';

  const pill = document.getElementById('solverPill');
  const pillText = document.getElementById('solverPillText');
  const solverCard = document.getElementById('solverCard');
  const solverName = document.getElementById('solverName');
  const solverBackend = document.getElementById('solverBackend');
  const solverNote = document.getElementById('solverNote');
  const cfdTabTag = document.querySelector('.method-tab[data-method="cfd"] .tag');

  if (pill) {
    pill.className = 'status-pill ' + (solverInstalled ? 'ok' : 'warn');
    if (pillText) {
      pillText.textContent = solverInstalled
        ? `${solverVersion} · ${cfdBackend.toUpperCase()}`
        : 'No CFD solver';
    }
  }
  if (solverCard) solverCard.className = 'solver-card ' + (solverInstalled ? 'ok' : 'warn');
  if (solverName) solverName.textContent = solverInstalled ? solverVersion : 'Synthetic fallback only';
  if (solverBackend) solverBackend.textContent = solverInstalled ? cfdBackend : 'none';
  if (solverNote) {
    solverNote.textContent = solverInstalled
      ? 'The CFD action runs a real 2D k-omega SST RANS front-wing section, then adds empirical floor and rear-wing terms. It is a hybrid estimate, not full-car CFD.'
      : 'No OpenFOAM detected. The CFD action returns the labelled calibrated fallback; VLM remains the physics preview.';
  }
  if (cfdTabTag) {
    cfdTabTag.textContent = solverInstalled ? `${solverVersion} · ~1 min` : 'fallback only';
  }

  if (!banner || !title || !text) {
    syncMethodTabs();
    updateMethodNote();
    return;
  }
  banner.hidden = false;

  if (solverInstalled) {
    banner.className = 'provenance-banner ok';
    title.textContent = `${solverVersion} detected — real 2D RANS available`;
    text.innerHTML =
      `CFD solves a 2D front-wing section with <code>simpleFoam</code>; floor and rear-wing terms ` +
      `stay empirical, so the total is <strong>RANS + EMPIRICAL</strong>, not full-car CFD. ` +
      `VLM stays the live preview.`;
  } else {
    banner.className = 'provenance-banner warn';
    title.textContent = 'No CFD solver detected — synthetic fallback is active';
    text.innerHTML =
      `This machine has no OpenFOAM binary. The <strong>CFD</strong> option returns a ` +
      `calibrated closed-form model, not a mesh solution. Results are badge-labelled ` +
      `<strong>SYNTHETIC MODEL</strong>; use <strong>VLM</strong> for the physics path.`;
  }

  if (sessionStorage.getItem('provenanceDismissed') === '1') {
    banner.hidden = true;
  }
  syncMethodTabs();
  updateMethodNote();
}

async function loadHealth() {
  try {
    const res = await fetch(`${API}/api/health`);
    if (!res.ok) throw new Error(`health ${res.status}`);
    applyHealth(await res.json());
  } catch (e) {
    const banner = document.getElementById('provenanceBanner');
    const pill = document.getElementById('solverPill');
    const pillText = document.getElementById('solverPillText');
    if (pill) pill.className = 'status-pill warn';
    if (pillText) pillText.textContent = 'Backend unreachable';
    if (banner) {
      banner.hidden = false;
      banner.className = 'provenance-banner warn';
      document.getElementById('pbTitle').textContent = 'Backend provenance unavailable';
      document.getElementById('pbText').textContent =
        'Could not read /api/health. Treat all forces as unverified until the API is reachable.';
    }
  }
}

function updateMethodNote() {
  const note = document.getElementById('methodNote');
  const method = document.getElementById('method')?.value || 'analytical_enhanced';
  const button = document.getElementById('btnSimulate');
  const hint = document.getElementById('runHint');
  if (!note) return;
  note.classList.remove('warn');

  const solverInstalled = cfdBackend === 'native' || cfdBackend === 'wsl';
  const cfd = isCfdMethod(method);

  if (cfd) {
    note.classList.toggle('warn', !solverInstalled);
    note.textContent = solverInstalled
      ? 'Real 2D OpenFOAM RANS on the current front-wing section, with empirical floor and rear-wing terms. The total is a hybrid estimate, not full-car CFD.'
      : 'No OpenFOAM detected. This returns the labelled calibrated fallback, not a mesh solution.';
    if (button) {
      button.textContent = solverInstalled ? 'Run CFD (OpenFOAM) · ~1 min' : 'Run CFD fallback';
      button.classList.add('cfd-mode');
    }
    if (hint) {
      hint.textContent = solverInstalled
        ? 'Manual run: meshes and solves the 2D section, then combines it with the empirical floor/rear model. Expect about one minute.'
        : 'No solver on this machine; the CFD action will be labelled as a synthetic fallback.';
    }
  } else if (method === 'analytical') {
    note.textContent = 'Empirical correlations. Fast trend indicator; no flow field is solved.';
    if (button) {
      button.textContent = 'Run analytical';
      button.classList.remove('cfd-mode');
    }
    if (hint) hint.textContent = 'Instant correlation baseline. Use VLM for the 3D physics preview.';
  } else {
    note.textContent = 'Low-fidelity VLM: horseshoe vortices and an image ground plane. No mesh, no RANS.';
    if (button) {
      button.textContent = 'Run VLM preview';
      button.classList.remove('cfd-mode');
    }
    if (hint) hint.textContent = 'VLM updates instantly. Choose CFD for a real OpenFOAM solve (about one minute).';
  }

  syncMethodTabs();
}


let chatHistory = [
  { sender: 'assistant', text: 'The VLM preview updates live as you change the setup. Ask the advisor about trade-offs, e.g. "how do I cut drag without losing balance?"' }
];

function getConfig() {
  const windKmh = +document.getElementById('wind_speed').value;
  document.getElementById('speed').value = windKmh;

  // Chord and span come from carSpec's model defaults - the UI has no control
  // for them, so the payload must not carry its own copy of those numbers.
  const wing = (name, angleDeg, gurneyMm = 0) => ({
    angle_deg: angleDeg,
    chord_m: DEFAULT_ELEMENT[name].chord_m,
    span_m: DEFAULT_ELEMENT[name].span_m,
    gurney_mm: gurneyMm,
  });

  return {
    front_wing_main: wing('front_wing_main', +document.getElementById('fw_main').value, +document.getElementById('fw_gurney').value),
    front_wing_flap: wing('front_wing_flap', +document.getElementById('fw_flap').value),
    rear_wing_main: wing('rear_wing_main', +document.getElementById('rw_main').value, +document.getElementById('rw_gurney').value),
    rear_wing_flap: wing('rear_wing_flap', +document.getElementById('rw_flap').value),
    drs_active: drsActive,
    ride_height_front_mm: +document.getElementById('rh_front').value,
    ride_height_rear_mm: +document.getElementById('rh_rear').value,
    rake_deg: +document.getElementById('rake').value,
    diffuser_angle_deg: +document.getElementById('diff_angle').value,
    diffuser_ride_height_mm: +document.getElementById('diff_rh').value,
    speed_kmh: windKmh,
  };
}

// Never add a `geo` key: the viewer publishes the solved car geometry itself,
// and setFlowSettings merges this payload over it.
function getFlowSettings() {
  return {
    windSpeedKmh: +document.getElementById('wind_speed').value,
    yawDeg: +document.getElementById('wind_yaw').value,
    pitchDeg: +document.getElementById('wind_pitch').value,
    turbulence: +document.getElementById('turbulence').value / 100,
    particleCount: +document.getElementById('particle_count').value,
    streamlineCount: +document.getElementById('streamline_count').value,
    flowSource: document.getElementById('flow_source').value,
    timeScale: +document.getElementById('time_scale').value / 100,
    particleSize: +document.getElementById('particle_size').value / 450,
    showWake: document.getElementById('show_wake').checked,
    showTunnel: document.getElementById('show_tunnel').checked,
    showSourcePlane: document.getElementById('show_source').checked,
    showGround: document.getElementById('show_ground').checked,
    showInteraction: document.getElementById('show_interaction').checked,
    showSlice: document.getElementById('show_slice').checked,
    sliceAxis: document.getElementById('slice_axis').value,
    slicePos: +document.getElementById('slice_pos').value,
    sliceMode: document.getElementById('slice_mode').value,
    wandY: +document.getElementById('wand_y').value,
    wandZ: +document.getElementById('wand_z').value,
    streamlineStyle: document.getElementById('streamline_style').value,
    particleStyle: document.getElementById('particle_style').value,
    smokeTint: document.getElementById('smoke_tint')?.checked ? 'speed' : 'neutral',
  };
}

function syncSliderLabels() {
  const map = [
    ['wind_speed', 'wind_speed_val', (v) => v + ' km/h'],
    ['wind_yaw', 'wind_yaw_val', (v) => v + '°'],
    ['wind_pitch', 'wind_pitch_val', (v) => v + '°'],
    ['turbulence', 'turbulence_val', (v) => v + '%'],
    ['particle_count', 'particle_count_val', (v) => v],
    ['streamline_count', 'streamline_count_val', (v) => v],
    ['time_scale', 'time_scale_val', (v) => v + '%'],
    ['particle_size', 'particle_size_val', (v) => v],
    ['model_scale', 'model_scale_val', (v) => v + '%'],
    ['model_rot', 'model_rot_val', (v) => v + '°'],
    ['wand_y', 'wand_y_val', (v) => (+v).toFixed(2) + ' m'],
    ['wand_z', 'wand_z_val', (v) => (+v).toFixed(2) + ' m'],
    ['slice_pos', 'slice_pos_val', (v) => (+v).toFixed(2) + ' m'],
  ];
  map.forEach(([id, labelId, fmt]) => {
    const el = document.getElementById(labelId);
    if (el) el.textContent = fmt(document.getElementById(id).value);
  });
  const geo = document.getElementById('rake_geo_val');
  const throat = document.getElementById('diff_throat_val');
  const flapEff = document.getElementById('rw_flap_eff_val');
  const drsNote = document.getElementById('drsNoteVal');
  if (geo || throat || flapEff) {
    const config = getConfig();
    const surfaces = deriveSurfaces(config);
    if (geo) geo.textContent = `${surfaces.floor.angleDeg.toFixed(2)}°`;
    if (throat) throat.textContent = (surfaces.diffuser.throatHeight * 1000).toFixed(1);
    if (flapEff) {
      const requested = +document.getElementById('rw_flap').value;
      const used = rearFlapAngleDeg(config);
      flapEff.textContent = used.toFixed(1);
      if (drsNote) {
        drsNote.textContent = !drsActive
          ? ''
          : used < requested
            ? ` (DRS open, folded from ${requested.toFixed(1)}°)`
            : ' (DRS open — this flap is already at or below the folded angle, so DRS changes nothing)';
      }
    }
  }
}

function updateEngineerPanel(speedRange = null) {
  const flow = getFlowSettings();
  const sr = speedRange || viewer?.getSpeedRange?.() || null;
  const ctx = engineeringContext(flow, lastSim, sr);
  const delta = baselineSim && lastSim ? compareToBaseline(lastSim, baselineSim) : null;
  const el = document.getElementById('engReadout');
  if (el) el.textContent = renderEngineerPanel(ctx, delta);

  const qEl = document.getElementById('m_q');
  if (qEl) qEl.textContent = (dynamicPressurePa(flow) / 1000).toFixed(2);

  // Update Flow HUD with DRS Active flag
  const drsStatusStr = drsActive ? ' · DRS ACTIVE' : '';
  const hudEl = document.getElementById('flowHud');
  if (hudEl) {
    hudEl.textContent = `${flow.windSpeedKmh} km/h · ${(flow.windSpeedKmh / 3.6).toFixed(1)} m/s · ${reynoldsLabel(flow)}${drsStatusStr}`;
  }
}

function refreshVisualization(simForces = lastSim) {
  if (!viewer) return;
  viewer.beginFlowBatch();
  viewer.setFlowSettings(getFlowSettings());
  viewer.setModelScale(+document.getElementById('model_scale').value);
  viewer.setModelRotation(+document.getElementById('model_rot').value);
  viewer.updateAero(getConfig(), simForces);
  viewer.endFlowBatch();
  updateEngineerPanel();
  document.getElementById('aeroBadge')?.classList.add('active');
}

function getTrackLapDelta(forces, fidelity) {
  const trackId = document.getElementById('track_profile')?.value || 'balanced';
  const track = TRACK_DATABASE[trackId] || TRACK_DATABASE.balanced;
  const baselineFidelity = baselineSim ? fidelityFor(baselineSim) : null;
  const referenceSim = baselineFidelity === fidelity ? baselineSim : referenceByFidelity.get(fidelity);
  if (!referenceSim) return 0.0;

  // Explicitly relative lap-time proxy, and only between runs of the same
  // fidelity family: a CFD hybrid is never presented as a lap-time gain over a
  // VLM preview, because those numbers are not directly comparable. The
  // coefficients themselves are analytical.lap_time_delta_s's, imported.
  return lapTimeDeltaS(forces, referenceSim.forces, track);
}

function populateCfdCard(sim) {
  const card = document.getElementById('cfdCard');
  if (!card) return;

  const provenance = sim?.provenance || {};
  const real = provenance.fidelity === 'rans_hybrid' ||
    (provenance.is_synthetic === false && String(sim?.method || '').includes('openfoam'));
  if (!provenance || (!real && provenance.is_synthetic !== true)) {
    card.hidden = true;
    return;
  }

  const title = document.getElementById('cfdTitle');
  const status = document.getElementById('cfdStatus');
  const note = document.getElementById('cfdNote');
  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };

  card.hidden = false;
  if (!real) {
    if (title) title.textContent = 'Synthetic fallback';
    if (status) {
      status.textContent = 'NO SOLVER';
      status.className = 'cfd-status warn';
    }
    setText('cfdCase', '—');
    setText('cfdCells', '—');
    setText('cfdIters', '—');
    setText('cfdP', '—');
    setText('cfdU', '—');
    setText('cfdTime', '—');
    if (note) {
      note.textContent = provenance.note ||
        'No mesh was generated. These numbers come from the calibrated closed-form fallback.';
    }
    return;
  }

  const converged = provenance.is_converged === true;
  if (title) title.textContent = 'OpenFOAM case details';
  if (status) {
    status.textContent = String(provenance.convergence_status || (converged ? 'converged' : 'unconverged')).toUpperCase();
    status.className = 'cfd-status ' + (converged ? 'ok' : 'warn');
  }
  setText('cfdCase', provenance.case_id || '—');
  setText('cfdCells', provenance.mesh_cells ? Number(provenance.mesh_cells).toLocaleString() : '—');
  setText('cfdIters', provenance.iterations ? String(provenance.iterations) : '—');
  setText('cfdP', Number.isFinite(provenance.residual_p) ? provenance.residual_p.toExponential(1) : '—');
  setText('cfdU', Number.isFinite(provenance.residual_u) ? provenance.residual_u.toExponential(1) : '—');
  setText('cfdTime', Number.isFinite(provenance.execution_time_s) ? `${provenance.execution_time_s.toFixed(1)} s` : '—');
  if (note) {
    note.textContent = provenance.note ||
      '2D front-wing section solved with OpenFOAM; floor and rear-wing terms remain empirical.';
  }
}

function showCfdError(message) {
  const card = document.getElementById('cfdCard');
  if (!card) return;
  card.hidden = false;
  const title = document.getElementById('cfdTitle');
  const status = document.getElementById('cfdStatus');
  const note = document.getElementById('cfdNote');
  if (title) title.textContent = 'OpenFOAM case failed';
  if (status) {
    status.textContent = 'FAILED';
    status.className = 'cfd-status fail';
  }
  ['cfdCase', 'cfdCells', 'cfdIters', 'cfdP', 'cfdU', 'cfdTime'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = '—';
  });
  if (note) {
    note.textContent = `No synthetic fallback was substituted. The force cards still show the last successful result, not this failed run. ${message || 'Check the server log for details.'}`;
  }
}

function showResults(sim) {
  lastSim = sim;

  const fidelity = fidelityFor(sim);
  if (fidelity && !referenceByFidelity.has(fidelity)) {
    referenceByFidelity.set(fidelity, sim);
  }

  // Override the backend proxy with a delta relative to the saved baseline (or
  // the first run) from the same fidelity family and the active track profile.
  sim.lap_time_delta_s = getTrackLapDelta(sim.forces, fidelity);

  setFidelityBadge(fidelity);

  const liveBadge = document.getElementById('aeroBadge');
  if (liveBadge) {
    const meta = fidelity ? FIDELITY_META[fidelity] : null;
    liveBadge.textContent = meta ? `● ${meta.label}` : '● LIVE AERO';
  }
  if (document.getElementById('viewerWrap')) {
    document.getElementById('viewerWrap').dataset.fidelity = fidelity || 'unknown';
  }

  const f = sim.forces;
  document.getElementById('m_df').textContent = f.downforce_total_N.toLocaleString();
  document.getElementById('m_drag').textContent = f.drag_N.toLocaleString();
  document.getElementById('m_ld').textContent = f.efficiency.toFixed(2);
  document.getElementById('m_bal').textContent = f.balance_front_pct.toFixed(1) + '%';
  document.getElementById('m_lap').textContent = (sim.lap_time_delta_s >= 0 ? '+' : '') + sim.lap_time_delta_s.toFixed(4);
  document.getElementById('m_cl').textContent = f.cl_total.toFixed(3);
  document.getElementById('m_cd').textContent = f.cd_total.toFixed(3);
  document.getElementById('m_df_f').textContent = f.downforce_front_N.toLocaleString();
  document.getElementById('m_df_r').textContent = f.downforce_rear_N.toLocaleString();
  document.getElementById('m_cop').textContent = f.cop_x_m.toFixed(2);
  populateCfdCard(sim);
  refreshVisualization(sim);
}

/**
 * Write a setup into the Vehicle Setup controls. The markup carries no defaults
 * of its own, so this is the only path a setup reaches them: boot seeds the panel
 * with the model's default setup and the optimizer writes the config it solved.
 * The readouts below those inputs are derived from the same setup, so they are
 * synced here - an optimizer result used to leave them naming the previous car
 * until the user touched a control.
 */
function setSetupInputs(cfg) {
  const set = (id, value) => {
    document.getElementById(id).value = value;
  };
  set('fw_main', cfg.front_wing_main.angle_deg);
  set('fw_flap', cfg.front_wing_flap.angle_deg);
  set('fw_gurney', cfg.front_wing_main.gurney_mm);
  set('rw_main', cfg.rear_wing_main.angle_deg);
  set('rw_flap', cfg.rear_wing_flap.angle_deg);
  set('rw_gurney', cfg.rear_wing_main.gurney_mm);
  set('rh_front', cfg.ride_height_front_mm);
  set('rh_rear', cfg.ride_height_rear_mm);
  set('rake', cfg.rake_deg);
  set('diff_angle', cfg.diffuser_angle_deg);
  set('diff_rh', cfg.diffuser_ride_height_mm);
  syncSliderLabels();
}

/**
 * The panel before the first response: the air speed and the DRS state travel
 * with the setup, so they are set before the writer syncs the readouts.
 */
function seedSetupPanel() {
  drsActive = DEFAULT_SETUP.drs_active;
  document.getElementById('wind_speed').value = DEFAULT_SETUP.speed_kmh;
  setSetupInputs(DEFAULT_SETUP);
  syncDrsUI();
}

function updateLivePreview() {
  syncSliderLabels();
  clearTimeout(debounceTimer);
  // A batched rebuild re-traces every streamline (~0.5 s), so dragging a slider
  // must not queue one per frame; 90 ms is imperceptible on release.
  debounceTimer = setTimeout(() => refreshVisualization(), 90);
}

function saveBaseline() {
  baselineSim = lastSim;
  const el = document.getElementById('baselineStatus');
  if (!baselineSim) {
    if (el) el.textContent = 'Run Simulate first, then save baseline';
    return;
  }
  if (el) {
    el.textContent = `Baseline saved · DF ${baselineSim.forces.downforce_total_N.toLocaleString()} N · ${new Date().toLocaleTimeString()}`;
  }
  updateEngineerPanel();
}

function showProbe(probe) {
  const el = document.getElementById('probeReadout');
  if (!el) return;
  el.classList.add('active');
  el.innerHTML = [
    '<strong>Probe</strong>',
    `Cp ${probe.cp != null ? probe.cp.toFixed(3) : '—'}`,
    `Speed ${probe.speedKmh.toFixed(1)} km/h (${probe.speedMs.toFixed(1)} m/s)`,
    `V [${probe.velocity.map((v) => v.toFixed(2)).join(', ')}] m/s`,
    `Pos [${probe.position.map((v) => v.toFixed(2)).join(', ')}]`,
  ].join('<br>');

  if (document.getElementById('flow_source').value === 'smoke_wand') {
    document.getElementById('wand_y').value = probe.position[1];
    document.getElementById('wand_z').value = probe.position[2];
    syncSliderLabels();
    updateLivePreview();
  }
}

async function runSimulate() {
  if (simulateInFlight) return;
  simulateInFlight = true;

  const method = document.getElementById('method').value;
  const cfd = isCfdMethod(method);
  const solverInstalled = cfdBackend === 'native' || cfdBackend === 'wsl';
  const status = document.getElementById('status');
  const button = document.getElementById('btnSimulate');
  const progress = document.getElementById('runProgress');
  let timer = null;

  if (button) button.disabled = true;
  if (progress) progress.hidden = false;
  if (status) {
    status.textContent = cfd
      ? (solverInstalled ? 'Meshing + solving with OpenFOAM…' : 'Running CFD fallback…')
      : 'Simulating…';
    status.className = 'status loading';
  }
  const started = Date.now();
  if (cfd) {
    timer = setInterval(() => {
      if (!status) return;
      const seconds = Math.round((Date.now() - started) / 1000);
      status.textContent = `${solverInstalled ? 'OpenFOAM solving' : 'CFD fallback'} · ${seconds}s`;
    }, 500);
  }
  refreshVisualization();

  try {
    const res = await fetch(`${API}/api/simulate?method=${method}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(getConfig()),
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = body?.detail || detail;
      } catch {
        detail = (await res.text()) || detail;
      }
      throw new Error(detail);
    }
    const sim = await res.json();
    showResults(sim);
    if (status) {
      const fidelity = fidelityFor(sim);
      const meta = fidelity ? FIDELITY_META[fidelity] : null;
      status.textContent = cfd
        ? `CFD complete${meta ? ' · ' + meta.label.toLowerCase() : ''}`
        : 'Forces updated';
      status.className = 'status';
    }
  } catch (e) {
    refreshVisualization();
    if (status) {
      const firstLine = String(e.message || 'unknown error').split('\n')[0];
      status.textContent = (cfd ? 'CFD failed · ' : 'Simulation failed · ') + firstLine;
      status.className = 'status error';
    }
    if (cfd) showCfdError(String(e.message || ''));
  } finally {
    if (timer) clearInterval(timer);
    if (progress) progress.hidden = true;
    if (button) button.disabled = false;
    simulateInFlight = false;
    if (!cfd) updateMethodNote();
  }
}

async function runOptimize() {
  const status = document.getElementById('status');
  const requestedMethod = document.getElementById('method').value;
  const cfd = isCfdMethod(requestedMethod);
  // Optimization uses the fast VLM/surrogate path. Never launch real CFD cases
  // behind one button; CFD is for verifying the selected setup. 30 trials keeps
  // the interactive request in the tens of seconds rather than several minutes.
  const optimizeMethod = cfd ? 'analytical_enhanced' : requestedMethod;
  const nTrials = 30;
  const baseStatus = cfd
    ? `Optimizing with VLM (${nTrials} trials; CFD is verification only)`
    : `Optimizing (${nTrials} trials)`;
  const started = Date.now();
  let timer = null;
  status.textContent = `${baseStatus}…`;
  status.className = 'status loading';
  timer = setInterval(() => {
    const seconds = Math.round((Date.now() - started) / 1000);
    status.textContent = `${baseStatus} · ${seconds}s`;
  }, 1000);
  try {
    const res = await fetch(`${API}/api/simulate-and-optimize?method=${optimizeMethod}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: getConfig(),
        objective: document.getElementById('objective').value,
        n_trials: nTrials,
        target_balance_pct: 42.0,
      }),
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        detail = (await res.json())?.detail || detail;
      } catch {
        detail = (await res.text()) || detail;
      }
      throw new Error(detail);
    }
    const data = await res.json();
    if (timer) clearInterval(timer);
    setSetupInputs(data.optimization.best_config);
    showResults(data.optimized_simulation);

    const improvement = Number(data.optimization.improvement_pct);
    const signedImprovement = `${improvement >= 0 ? '+' : ''}${improvement.toFixed(1)}%`;
    const surrogateNote = cfd ? ' · VLM surrogate; run CFD to verify' : '';
    const opt = data.optimization;
    const reachNote = opt.target_reachable === false && opt.achievable_balance_pct?.length === 2
      ? ` · target ${opt.target_balance_pct}% outside sampled range ${opt.achievable_balance_pct[0]}–${opt.achievable_balance_pct[1]}%`
      : '';

    // Add optimizer recommendations to chat
    chatHistory.push({ sender: 'assistant', text: `### Optimization Complete (${signedImprovement} objective score)\n\n${data.advice.advice}${reachNote}` });
    renderChat();

    status.textContent = `${signedImprovement} objective score (${data.advice.source})${surrogateNote}${reachNote}`;
    status.className = 'status';
  } catch (e) {
    if (timer) clearInterval(timer);
    status.textContent = 'Optimization failed: ' + (String(e.message || '').split('\n')[0]);
    status.className = 'status error';
  } finally {
    if (timer) clearInterval(timer);
  }
}

function syncDrsUI() {
  const btn = document.getElementById('btnDRS');
  const txt = document.getElementById('drsText');
  if (btn) btn.classList.toggle('enabled', drsActive);
  if (txt) txt.textContent = drsActive ? 'ACTIVE' : 'OFF';
}

function toggleDRS() {
  drsActive = !drsActive;
  syncDrsUI();
  syncSliderLabels();
  if (isCfdMethod()) {
    refreshVisualization();
    const status = document.getElementById('status');
    if (status) {
      status.textContent = 'DRS changed — re-run CFD to update the solver-backed forces.';
      status.className = 'status';
    }
  } else {
    runSimulate();
  }
}

async function generateAeroMap() {
  const canvas = document.getElementById('aeroMapCanvas');
  const placeholder = document.getElementById('aeroMapPlaceholder');
  const loading = document.getElementById('aeroMapLoading');
  const mapMethod = isCfdMethod() ? 'analytical_enhanced' : document.getElementById('method').value;

  if (placeholder) placeholder.style.display = 'none';
  if (canvas) canvas.style.display = 'none';
  if (loading) {
    loading.style.display = 'block';
    loading.textContent = isCfdMethod()
      ? 'Running 25 VLM map points (CFD is verification only)…'
      : 'Running 25 simulations…';
  }
  
  try {
    const res = await fetch(`${API}/api/aero-map?method=${mapMethod}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: getConfig(),
        min_frh: 10.0,
        max_frh: 50.0,
        min_rrh: 20.0,
        max_rrh: 80.0,
        steps: 5
      })
    });
    if (!res.ok) throw new Error('Aero map API failed');
    const data = await res.json();
    aeroMapGrid = data.grid;
    aeroMapFrhVals = data.frh_vals;
    aeroMapRrhVals = data.rrh_vals;
    
    if (loading) loading.style.display = 'none';
    if (canvas) canvas.style.display = 'block';
    
    drawAeroMap();
  } catch (e) {
    if (loading) loading.style.display = 'none';
    if (placeholder) {
      placeholder.style.display = 'block';
      placeholder.textContent = 'Error: ' + e.message;
    }
  }
}

function drawAeroMap() {
  if (!aeroMapGrid || !aeroMapFrhVals || !aeroMapRrhVals) return;
  const canvas = document.getElementById('aeroMapCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const metric = document.getElementById('aeroMapMetric')?.value || 'downforce';
  
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  
  const padLeft = 20;
  const padBottom = 15;
  const padRight = 5;
  const padTop = 5;
  const plotW = w - padLeft - padRight;
  const plotH = h - padBottom - padTop;
  
  const steps = aeroMapFrhVals.length;
  const cellW = plotW / steps;
  const cellH = plotH / steps;
  
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let r = 0; r < steps; r++) {
    for (let c = 0; c < steps; c++) {
      let v = 0;
      if (metric === 'downforce') v = aeroMapGrid[r][c].downforce;
      else if (metric === 'balance') v = aeroMapGrid[r][c].balance;
      else v = aeroMapGrid[r][c].efficiency;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }
  if (minVal === maxVal) { minVal -= 1; maxVal += 1; }
  
  const currentFrh = +document.getElementById('rh_front').value;
  const currentRrh = +document.getElementById('rh_rear').value;
  let closestI = -1;
  let closestJ = -1;
  let minFrhDist = Infinity;
  let minRrhDist = Infinity;
  
  for (let i = 0; i < steps; i++) {
    const dist = Math.abs(aeroMapFrhVals[i] - currentFrh);
    if (dist < minFrhDist) { minFrhDist = dist; closestI = i; }
  }
  for (let j = 0; j < steps; j++) {
    const dist = Math.abs(aeroMapRrhVals[j] - currentRrh);
    if (dist < minRrhDist) { minRrhDist = dist; closestJ = j; }
  }
  
  for (let j = 0; j < steps; j++) {
    const canvasY = padTop + (steps - 1 - j) * cellH;
    for (let i = 0; i < steps; i++) {
      const canvasX = padLeft + i * cellW;
      
      const cell = aeroMapGrid[i][j];
      let val = 0;
      if (metric === 'downforce') val = cell.downforce;
      else if (metric === 'balance') val = cell.balance;
      else val = cell.efficiency;
      
      const pct = (val - minVal) / (maxVal - minVal);
      
      let fillStyle = '';
      if (metric === 'downforce') {
        const rVal = Math.round(20 * (1 - pct) + 0 * pct);
        const gVal = Math.round(27 * (1 - pct) + 210 * pct);
        const bVal = Math.round(45 * (1 - pct) + 190 * pct);
        fillStyle = `rgb(${rVal}, ${gVal}, ${bVal})`;
      } else if (metric === 'balance') {
        const balPct = (cell.balance - 38) / (46 - 38);
        const rVal = Math.round(225 * (1 - balPct) + 0 * balPct);
        const gVal = Math.round(6 * (1 - balPct) + 210 * balPct);
        const bVal = Math.round(0 * (1 - balPct) + 190 * balPct);
        fillStyle = `rgb(${Math.max(0, Math.min(255, rVal))}, ${Math.max(0, Math.min(255, gVal))}, ${Math.max(0, Math.min(255, bVal))})`;
      } else {
        const rVal = Math.round(58 * (1 - pct) + 0 * pct);
        const gVal = Math.round(16 * (1 - pct) + 210 * pct);
        const bVal = Math.round(16 * (1 - pct) + 120 * pct);
        fillStyle = `rgb(${rVal}, ${gVal}, ${bVal})`;
      }
      
      ctx.fillStyle = fillStyle;
      ctx.fillRect(canvasX, canvasY, cellW - 1, cellH - 1);
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.font = '7px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      let textStr = '';
      if (metric === 'downforce') textStr = Math.round(val).toString();
      else if (metric === 'balance') textStr = val.toFixed(1) + '%';
      else textStr = val.toFixed(2);
      
      ctx.fillText(textStr, canvasX + cellW/2, canvasY + cellH/2);
      
      if (i === closestI && j === closestJ) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(canvasX + 0.5, canvasY + 0.5, cellW - 2, cellH - 2);
        
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(canvasX + cellW - 4, canvasY + 4, 2, 0, 2*Math.PI);
        ctx.fill();
      }
    }
  }
  
  ctx.fillStyle = 'var(--muted)';
  ctx.font = '7px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  
  for (let i = 0; i < steps; i++) {
    const canvasX = padLeft + i * cellW + cellW/2;
    ctx.fillText(Math.round(aeroMapFrhVals[i]), canvasX, h - padBottom + 2);
  }
  
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let j = 0; j < steps; j++) {
    const canvasY = padTop + (steps - 1 - j) * cellH + cellH/2;
    ctx.fillText(Math.round(aeroMapRrhVals[j]), padLeft - 2, canvasY);
  }
}

function handleAeroMapClick(e) {
  if (!aeroMapFrhVals || !aeroMapRrhVals) return;
  const canvas = document.getElementById('aeroMapCanvas');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  
  const padLeft = 20;
  const padRight = 5;
  const padTop = 5;
  const padBottom = 15;
  
  const plotW = canvas.width - padLeft - padRight;
  const plotH = canvas.height - padBottom - padTop;
  
  const steps = aeroMapFrhVals.length;
  const cellW = plotW / steps;
  const cellH = plotH / steps;
  
  if (x >= padLeft && x <= canvas.width - padRight && y >= padTop && y <= canvas.height - padBottom) {
    const i = Math.floor((x - padLeft) / cellW);
    const j = steps - 1 - Math.floor((y - padTop) / cellH);
    
    if (i >= 0 && i < steps && j >= 0 && j < steps) {
      // A programmatic `.value` fires no `input` event, so writing these two
      // controls here used to skip the readouts derived from them. The writer
      // is the only way a setup reaches these inputs.
      setSetupInputs({
        ...getConfig(),
        ride_height_front_mm: aeroMapFrhVals[i],
        ride_height_rear_mm: aeroMapRrhVals[j],
      });
      if (isCfdMethod()) {
        refreshVisualization();
        const status = document.getElementById('status');
        if (status) {
          status.textContent = 'Ride heights applied — press Run CFD to update solver-backed forces.';
          status.className = 'status';
        }
      } else {
        runSimulate();
      }
      drawAeroMap();
    }
  }
}

function renderChat() {
  const container = document.getElementById('chatHistory');
  if (!container) return;

  container.innerHTML = chatHistory.map(msg => {
    const isUser = msg.sender === 'user';
    const label = isUser ? '<strong>Engineer:</strong>' : '<strong>Advisor:</strong>';

    let text = msg.text
      .replace(/## (.*)/g, '<strong>$1</strong>')
      .replace(/### (.*)/g, '<strong>$1</strong>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/- (.*)/g, '&ndash; $1')
      .replace(/\n/g, '<br>');

    return `<div class="chat-msg ${isUser ? 'user' : 'assistant'}">${label}<br>${text}</div>`;
  }).join('');

  container.scrollTop = container.scrollHeight;
}

async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  if (!input) return;
  
  const text = input.value.trim();
  if (!text) return;
  
  chatHistory.push({ sender: 'user', text });
  input.value = '';
  renderChat();
  
  chatHistory.push({ sender: 'assistant', text: 'Thinking...' });
  renderChat();
  
  try {
    const trackId = document.getElementById('track_profile')?.value || 'balanced';
    const track = TRACK_DATABASE[trackId] || TRACK_DATABASE.balanced;
    const config = getConfig();
    const method = document.getElementById('method').value;
    
    // The advisor quotes the forces it is handed, so there is no placeholder
    // result to fall back on: before the first run, a tier has to actually solve
    // the setup the question is about.
    let simRes = lastSim;
    let tierNote = '';
    if (!simRes) {
      const askedForCfd = isCfdMethod();
      const simMethod = askedForCfd ? 'analytical_enhanced' : method;
      const simResponse = await fetch(`${API}/api/simulate?method=${simMethod}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      if (!simResponse.ok) {
        let detail = `HTTP ${simResponse.status}`;
        try {
          detail = (await simResponse.json())?.detail || detail;
        } catch {
          detail = (await simResponse.text()) || detail;
        }
        throw new Error(`no simulation to advise on - ${detail}`);
      }
      simRes = await simResponse.json();
      showResults(simRes);
      const answeredBy = FIDELITY_META[fidelityFor(simRes)]?.short || simMethod;
      const runStatus = document.getElementById('status');
      if (runStatus) {
        runStatus.textContent = `Advisor ran ${answeredBy} for this setup`;
        runStatus.className = 'status';
      }
      if (askedForCfd) {
        tierNote = `Answered from ${answeredBy} (Cl ${simRes.forces.cl_total.toFixed(3)}, ${Math.round(simRes.forces.downforce_total_N)} N); the CFD tier selected here had not been run when the question was asked.`;
      }
    }
    
    // The track goes as data, not prose inside the question: the rule-based
    // advisor keyword-matched whatever arrived in `question`, so a selected
    // circuit's own description decided which answer the user got.
    const payload = {
      config: config,
      simulation: simRes,
      question: text,
      track: {
        name: track.name,
        description: track.description,
        drag_multiplier: track.dragMultiplier,
        downforce_multiplier: track.dfMultiplier,
      },
    };
    
    const res = await fetch(`${API}/api/advise`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    chatHistory.pop();
    if (!res.ok) throw new Error('Advisor API offline');
    
    const data = await res.json();
    chatHistory.push({ sender: 'assistant', text: data.advice + (tierNote ? `\n\n*${tierNote}*` : '') });
  } catch (e) {
    chatHistory.pop();
    chatHistory.push({ sender: 'assistant', text: 'Error contacting advisor: ' + e.message });
  }
  
  renderChat();
}

async function handleModelUpload(file) {
  const status = document.getElementById('modelStatus');
  status.textContent = `Loading ${file.name}...`;
  status.className = 'status loading';
  try {
    const info = await viewer.loadFile(file);
    status.textContent = `${info.filename} — ${info.vertexCount.toLocaleString()} verts`;
    updateSliderRanges();
    refreshVisualization(lastSim);
    let modelId = null;
    try {
      const up = await fetch(`${API}/api/models/upload`, { method: 'POST', body: (() => { const f = new FormData(); f.append('file', file); return f; })() });
      if (up.ok) modelId = (await up.json()).model_id;
    } catch (e) { /* upload stays best-effort for visuals */ }
    lastImportedModelId = modelId;
    const btn = document.getElementById('btnCfd3d');
    if (btn) btn.style.display = modelId ? '' : 'none';
    if (isCfdMethod()) {
      const runStatus = document.getElementById('status');
      if (runStatus) {
        runStatus.textContent = 'Model loaded — press Run CFD to update solver-backed forces.';
        runStatus.className = 'status';
      }
    } else {
      runSimulate();
    }
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  }
}

// 3D CFD on the uploaded mesh: measure first (honest 422 for bad scale), then
// queue the snappyHexMesh job and poll it. Results carry provenance: forces
// here are for the uploaded geometry only, never mixed with the parametric car.
async function runImportedCfd3d() {
  const btn = document.getElementById('btnCfd3d');
  const box = document.getElementById('cfd3dStatus');
  const say = (html) => { box.style.display = ''; box.innerHTML = html; };
  if (!lastImportedModelId) { say('Upload an STL/OBJ first.'); return; }
  btn.disabled = true;
  const speed = Number(document.getElementById('speed')?.value) || 120;
  try {
    const meas = await fetch(`${API}/api/cfd/3d/measure`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: lastImportedModelId, axes: 'three' })
    });
    if (meas.ok) {
      const m = await meas.json();
      say(`Measured: ${m.frontal_area_m2.toFixed(2)} m² frontal · ${(100 * m.watertight_edge_fraction).toFixed(0)}% watertight · ${m.triangle_count.toLocaleString()} triangles` +
        (m.warnings && m.warnings.length ? `<br>⚠ ${m.warnings.join('<br>⚠ ')}` : ''));
    } else if (meas.status === 422) {
      const d = (await meas.json()).detail || '';
      say(`Mesh rejected: ${d}`);
      btn.disabled = false;
      return;
    }
    const res = await fetch(`${API}/api/cfd/3d`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: lastImportedModelId, speed_kmh: speed, axes: 'three', level: 'coarse' })
    });
    if (!res.ok) {
      const detail = (await res.json()).detail || res.statusText;
      say(`3D CFD refused (HTTP ${res.status}): ${detail}`);
      btn.disabled = false;
      return;
    }
    const { job_id } = await res.json();
    say(`3D CFD job ${job_id} queued — meshing + solving can take minutes to hours. Polling…`);
    for (;;) {
      await new Promise(r => setTimeout(r, 5000));
      const poll = await fetch(`${API}/api/cfd/3d/${job_id}`);
      if (!poll.ok) { say(`Job poll failed (HTTP ${poll.status}).`); btn.disabled = false; return; }
      const j = await poll.json();
      if (j.status === 'error') { say(`3D CFD job failed:<br><code>${(j.detail || '').slice(0, 800)}</code>`); btn.disabled = false; return; }
      if (j.status === 'done') {
        const r = j.result, p = r.provenance;
        say(`<b>Your mesh, real OpenFOAM solve</b><br>` +
          `Drag ${r.drag_N.toFixed(1)} N (Cd ${r.cd.toFixed(3)}) · Downforce ${r.downforce_N.toFixed(1)} N (Cl ${r.cl.toFixed(3)})<br>` +
          `At ${Math.round(r.speed_kmh)} km/h · Aref ${r.frontal_area_m2.toFixed(2)} m² · ${p.mesh_cells.toLocaleString()} cells<br>` +
          `${p.solver_version} · ${p.convergence_status} · ${Math.round(p.execution_time_s)} s · case ${p.case_hash}<br>` +
          `<em>Forces are for the uploaded mesh only — not combined with the parametric car.</em>`);
        btn.disabled = false;
        return;
      }
      say(`3D CFD job ${job_id} ${j.status} — polling…`);
    }
  } catch (e) {
    say('3D CFD error: ' + e.message);
    btn.disabled = false;
  }
}

function setupModelImport() {
  const input = document.getElementById('modelInput');
  const dropZone = document.getElementById('viewerWrap');
  const btn3d = document.getElementById('btnCfd3d');
  if (btn3d) btn3d.addEventListener('click', runImportedCfd3d);
  input.addEventListener('change', (e) => { if (e.target.files[0]) handleModelUpload(e.target.files[0]); });
  ['dragover', 'dragenter'].forEach((ev) => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); }));
  dropZone.addEventListener('dragleave', (e) => { if (e.target === dropZone) dropZone.classList.remove('drag-over'); });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleModelUpload(e.dataTransfer.files[0]);
  });
}

function setupEngineeringTools() {
  document.querySelectorAll('[data-view]').forEach((btn) => {
    btn.addEventListener('click', () => viewer.setCameraView(btn.dataset.view));
  });

  document.getElementById('btnSaveBaseline')?.addEventListener('click', saveBaseline);
  document.getElementById('btnExportSetup')?.addEventListener('click', () => {
    exportSetupJson(getConfig(), getFlowSettings(), {
      simulation: lastSim,
      baseline: baselineSim,
    });
  });
  document.getElementById('btnExportCsv')?.addEventListener('click', () => {
    const ctx = engineeringContext(getFlowSettings(), lastSim, viewer?.getSpeedRange?.());
    exportResultsCsv(lastSim, ctx);
  });

  document.getElementById('probeMode')?.addEventListener('change', (e) => {
    viewer.setProbeMode(e.target.checked);
    const el = document.getElementById('probeReadout');
    if (!e.target.checked && el) el.classList.remove('active');
  });
}

/**
 * A range input only ever lands on `min + k * step`, so raw float bounds make
 * the values the readouts print unreachable: with min = -1.103 the closest the
 * slice can get to the centreline is -0.003 m, which reads as "-0.00 m".
 */
function setSliderRange(slider, min, max, step) {
  slider.min = (Math.floor(min / step) * step).toFixed(2);
  slider.max = (Math.ceil(max / step) * step).toFixed(2);
  slider.step = step;
}

function updateSliderRanges() {
  if (!viewer) return;
  const bounds = viewer.bounds;
  const sliceAxis = document.getElementById('slice_axis').value;
  const slicePosSlider = document.getElementById('slice_pos');
  
  let minVal, maxVal, step;
  if (sliceAxis === 'z') {
    minVal = bounds.min[2] - 0.2;
    maxVal = bounds.max[2] + 0.2;
    step = 0.01;
  } else if (sliceAxis === 'y') {
    minVal = bounds.min[1] - 0.05;
    maxVal = bounds.max[1] + 0.3;
    step = 0.01;
  } else { // 'x'
    minVal = bounds.min[0] - 0.2;
    maxVal = bounds.max[0] + 0.2;
    step = 0.02;
  }
  
  setSliderRange(slicePosSlider, minVal, maxVal, step);
  
  const curVal = +slicePosSlider.value;
  if (curVal < minVal || curVal > maxVal) {
    slicePosSlider.value = ((minVal + maxVal) / 2).toFixed(2);
  }
  
  const wandYSlider = document.getElementById('wand_y');
  if (wandYSlider) setSliderRange(wandYSlider, bounds.min[1] - 0.05, bounds.max[1] + 0.3, 0.01);
  const wandZSlider = document.getElementById('wand_z');
  if (wandZSlider) setSliderRange(wandZSlider, bounds.min[2] - 0.1, bounds.max[2] + 0.1, 0.01);
}

function updateUIState() {
  const source = document.getElementById('flow_source').value;
  const wandControls = document.getElementById('wand_controls');
  if (wandControls) {
    wandControls.style.display = source === 'smoke_wand' ? 'block' : 'none';
  }

  const showSlice = document.getElementById('show_slice').checked;
  const sliceControls = document.getElementById('slice_controls');
  if (sliceControls) {
    sliceControls.style.display = showSlice ? 'block' : 'none';
  }
}

function setupControls() {
  if (!viewer) return;

  document.getElementById('btnSimulate').addEventListener('click', runSimulate);
  document.getElementById('btnOptimize').addEventListener('click', runOptimize);
  document.getElementById('btnImport').addEventListener('click', () => document.getElementById('modelInput').click());
  document.getElementById('btnResetView').addEventListener('click', () => viewer.resetCamera());

  document.getElementById('method')?.addEventListener('change', () => {
    syncMethodTabs();
    updateMethodNote();
  });
  document.querySelectorAll('.method-tab').forEach((tab) => {
    tab.addEventListener('click', () => setMethod(tab.dataset.method));
  });
  document.getElementById('pbDismiss')?.addEventListener('click', () => {
    const banner = document.getElementById('provenanceBanner');
    if (banner) banner.hidden = true;
    sessionStorage.setItem('provenanceDismissed', '1');
  });

  // DRS toggle
  document.getElementById('btnDRS')?.addEventListener('click', toggleDRS);
  
  // Track selector change event
  document.getElementById('track_profile')?.addEventListener('change', () => {
    if (lastSim) showResults(lastSim);
    else runSimulate();
  });
  
  // Aero map buttons & canvas events
  document.getElementById('btnGenerateAeroMap')?.addEventListener('click', generateAeroMap);
  document.getElementById('aeroMapMetric')?.addEventListener('change', () => {
    if (aeroMapGrid) drawAeroMap();
  });
  document.getElementById('aeroMapCanvas')?.addEventListener('click', handleAeroMapClick);

  // Chat console events
  document.getElementById('btnSendChat')?.addEventListener('click', sendChatMessage);
  document.getElementById('chatInput')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendChatMessage();
  });

  document.getElementById('model_scale').addEventListener('input', () => {
    syncSliderLabels();
    viewer.setModelScale(+document.getElementById('model_scale').value);
    updateSliderRanges();
  });
  document.getElementById('model_rot').addEventListener('input', () => {
    syncSliderLabels();
    viewer.setModelRotation(+document.getElementById('model_rot').value);
    updateSliderRanges();
    refreshVisualization();
  });

  document.getElementById('viewMode').addEventListener('change', (e) => {
    viewer.setViewMode(e.target.value);
  });
  document.getElementById('showStreamlines').addEventListener('change', (e) => {
    viewer.toggleStreamlines(e.target.checked);
    if (e.target.checked) refreshVisualization();
  });
  document.getElementById('showParticles').addEventListener('change', (e) => viewer.toggleParticles(e.target.checked));
  document.getElementById('showForces').addEventListener('change', (e) => {
    viewer.toggleForces(e.target.checked);
    if (e.target.checked && lastSim) refreshVisualization();
  });

  document.getElementById('slice_axis').addEventListener('change', () => {
    updateSliderRanges();
    syncSliderLabels();
    updateLivePreview();
  });

  document.getElementById('show_slice').addEventListener('change', () => {
    updateUIState();
    updateLivePreview();
  });

  document.getElementById('flow_source').addEventListener('change', () => {
    updateUIState();
    updateLivePreview();
  });

  const flowInputs = [
    'wind_speed', 'wind_yaw', 'wind_pitch', 'turbulence', 'particle_count',
    'streamline_count', 'flow_source', 'time_scale', 'particle_size',
    'show_wake', 'show_tunnel', 'show_source', 'show_ground', 'show_interaction',
    'show_slice', 'slice_axis', 'slice_pos', 'slice_mode', 'wand_y', 'wand_z',
    'streamline_style', 'particle_style', 'smoke_tint',
  ];
  flowInputs.forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', updateLivePreview);
      el.addEventListener('change', updateLivePreview);
    }
  });

  document.querySelectorAll('input[type=number], select').forEach((el) => {
    if (!flowInputs.includes(el.id) && el.id !== 'speed') {
      el.addEventListener('input', updateLivePreview);
    }
  });

  setupEngineeringTools();
}

function applyInitialMethodFromUrl() {
  const select = document.getElementById('method');
  if (!select) return;
  const requested = new URLSearchParams(window.location.search).get('method');
  if (requested && Array.from(select.options).some((o) => o.value === requested)) {
    select.value = requested;
  }
  syncMethodTabs();
}

function boot() {
  try {
    // First, before the viewer: the panel's numbers come from the model, so they
    // have to be there whether or not the 3D canvas ever comes up.
    seedSetupPanel();
    viewer = new AeroViewer(document.getElementById('viewer3d'));
    viewer.onSpeedRange = (range) => updateEngineerPanel(range);
    viewer.onProbe = showProbe;
    applyInitialMethodFromUrl();
    setupModelImport();
    setupControls();
    updateSliderRanges();
    updateUIState();
    syncSliderLabels();
    syncMethodTabs();
    updateMethodNote();
    loadHealth();
    refreshVisualization();
    renderChat();
    if (isCfdMethod()) {
      const status = document.getElementById('status');
      if (status) {
        status.textContent = 'CFD selected — press Run CFD to start (about one minute).';
        status.className = 'status';
      }
    } else {
      runSimulate();
    }
  } catch (err) {
    document.getElementById('bootError').textContent = '3D viewer failed: ' + err.message;
    document.getElementById('bootError').style.display = 'block';
  }
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
