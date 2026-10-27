/** Main app — vehicle setup + wind tunnel flow controls. */

import { AeroViewer } from './viewer3d.js';
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

let drsActive = false;
let aeroMapGrid = null;
let aeroMapFrhVals = null;
let aeroMapRrhVals = null;

let chatHistory = [
  { sender: 'assistant', text: 'Air now follows the car shape. Adjust sliders to update, or ask me setup or track questions below! (e.g. "How to reduce understeer?")' }
];

function getConfig() {
  const windKmh = +document.getElementById('wind_speed').value;
  document.getElementById('speed').value = windKmh;
  
  let rwFlapAngle = +document.getElementById('rw_flap').value;
  if (drsActive) {
    rwFlapAngle = 2.0; // Override rear flap to 2 degrees in DRS
  }
  
  return {
    front_wing_main: { angle_deg: +document.getElementById('fw_main').value, chord_m: 0.35, span_m: 1.80, gurney_mm: 0 },
    front_wing_flap: { angle_deg: +document.getElementById('fw_flap').value, chord_m: 0.12, span_m: 1.75, gurney_mm: 0 },
    rear_wing_main: { angle_deg: +document.getElementById('rw_main').value, chord_m: 0.40, span_m: 0.95, gurney_mm: 0 },
    rear_wing_flap: { angle_deg: rwFlapAngle, chord_m: 0.15, span_m: 0.90, gurney_mm: 0 },
    ride_height_front_mm: +document.getElementById('rh_front').value,
    ride_height_rear_mm: +document.getElementById('rh_rear').value,
    rake_deg: +document.getElementById('rake').value,
    diffuser_angle_deg: +document.getElementById('diff_angle').value,
    diffuser_ride_height_mm: +document.getElementById('diff_rh').value,
    speed_kmh: windKmh,
  };
}

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
    hudEl.textContent = `${flow.windSpeedKmh} km/h · ${(flow.windSpeedKmh / 3.6).toFixed(1)} m/s · Re=${(ctx.reynolds / 1e6).toFixed(2)}M${drsStatusStr}`;
  }
}

function refreshVisualization(simForces = lastSim) {
  if (!viewer) return;
  viewer.setFlowSettings(getFlowSettings());
  viewer.setModelScale(+document.getElementById('model_scale').value);
  viewer.setModelRotation(+document.getElementById('model_rot').value);
  viewer.updateAero(getConfig(), simForces);
  updateEngineerPanel();
  document.getElementById('aeroBadge')?.classList.add('active');
}

function getTrackLapDelta(forces) {
  const trackId = document.getElementById('track_profile')?.value || 'balanced';
  const track = TRACK_DATABASE[trackId] || TRACK_DATABASE.balanced;
  
  const baseline_drag = 1200.0;
  const baseline_df = 3500.0;
  
  const drag_delta = (forces.drag_N - baseline_drag) * 0.0001 * track.dragMultiplier;
  const df_delta = -(forces.downforce_total_N - baseline_df) * 0.00008 * track.dfMultiplier;
  
  return drag_delta + df_delta;
}

function showResults(sim) {
  lastSim = sim;
  
  // Override lap delta client-side based on active track
  sim.lap_time_delta_s = getTrackLapDelta(sim.forces);
  
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
  refreshVisualization(sim);
}

function applyOptimizedConfig(cfg) {
  document.getElementById('fw_main').value = cfg.front_wing_main.angle_deg;
  document.getElementById('fw_flap').value = cfg.front_wing_flap.angle_deg;
  document.getElementById('rw_main').value = cfg.rear_wing_main.angle_deg;
  document.getElementById('rw_flap').value = cfg.rear_wing_flap.angle_deg;
  document.getElementById('rh_front').value = cfg.ride_height_front_mm;
  document.getElementById('rh_rear').value = cfg.ride_height_rear_mm;
  document.getElementById('rake').value = cfg.rake_deg;
  document.getElementById('diff_angle').value = cfg.diffuser_angle_deg;
  document.getElementById('diff_rh').value = cfg.diffuser_ride_height_mm;
}

function updateLivePreview() {
  syncSliderLabels();
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => refreshVisualization(), 16);
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
  const status = document.getElementById('status');
  status.textContent = 'Simulating...';
  status.className = 'status loading';
  refreshVisualization();
  try {
    const res = await fetch(`${API}/api/simulate?method=${document.getElementById('method').value}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(getConfig()),
    });
    if (!res.ok) throw new Error('API offline — local flow active');
    showResults(await res.json());
    status.textContent = 'Forces updated';
    status.className = 'status';
  } catch (e) {
    refreshVisualization();
    status.textContent = 'Local flow · ' + e.message;
    status.className = 'status';
  }
}

async function runOptimize() {
  const status = document.getElementById('status');
  status.textContent = 'Optimizing...';
  status.className = 'status loading';
  try {
    const res = await fetch(`${API}/api/simulate-and-optimize?method=${document.getElementById('method').value}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: getConfig(),
        objective: document.getElementById('objective').value,
        n_trials: 100,
        target_balance_pct: 42.0,
      }),
    });
    const data = await res.json();
    applyOptimizedConfig(data.optimization.best_config);
    showResults(data.optimized_simulation);
    
    // Add optimizer recommendations to chat
    chatHistory.push({ sender: 'assistant', text: `### Optimization Complete (+${data.optimization.improvement_pct}%)\n\n${data.advice.advice}` });
    renderChat();
    
    status.textContent = `+${data.optimization.improvement_pct}% (${data.advice.source})`;
    status.className = 'status';
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  }
}

function toggleDRS() {
  drsActive = !drsActive;
  const btn = document.getElementById('btnDRS');
  const indicator = document.getElementById('drsIndicator');
  const txt = document.getElementById('drsText');
  
  if (drsActive) {
    btn.style.background = 'var(--accent2)';
    btn.style.color = 'var(--bg)';
    btn.style.borderColor = 'var(--accent2)';
    indicator.style.background = '#e10600';
    indicator.style.boxShadow = '0 0 10px #e10600';
    txt.textContent = 'ACTIVE';
  } else {
    btn.style.background = 'var(--bg)';
    btn.style.color = 'var(--muted)';
    btn.style.borderColor = 'var(--border)';
    indicator.style.background = '#444';
    indicator.style.boxShadow = 'none';
    txt.textContent = 'DISABLED';
  }
  
  runSimulate();
}

async function generateAeroMap() {
  const canvas = document.getElementById('aeroMapCanvas');
  const placeholder = document.getElementById('aeroMapPlaceholder');
  const loading = document.getElementById('aeroMapLoading');
  
  placeholder.style.display = 'none';
  canvas.style.display = 'none';
  loading.style.display = 'block';
  
  try {
    const res = await fetch(`${API}/api/aero-map?method=${document.getElementById('method').value}`, {
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
    
    loading.style.display = 'none';
    canvas.style.display = 'block';
    
    drawAeroMap();
  } catch (e) {
    loading.style.display = 'none';
    placeholder.style.display = 'block';
    placeholder.textContent = 'Error: ' + e.message;
  }
}

function drawAeroMap() {
  if (!aeroMapGrid || !aeroMapFrhVals || !aeroMapRrhVals) return;
  const canvas = document.getElementById('aeroMapCanvas');
  const ctx = canvas.getContext('2d');
  const metric = document.getElementById('aeroMapMetric').value;
  
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
      document.getElementById('rh_front').value = aeroMapFrhVals[i];
      document.getElementById('rh_rear').value = aeroMapRrhVals[j];
      runSimulate();
      drawAeroMap();
    }
  }
}

function renderChat() {
  const container = document.getElementById('chatHistory');
  if (!container) return;
  
  container.innerHTML = chatHistory.map(msg => {
    const isUser = msg.sender === 'user';
    const bg = isUser ? 'rgba(0, 210, 190, 0.15)' : 'rgba(255, 255, 255, 0.03)';
    const border = isUser ? '1px solid var(--accent2)' : '1px solid var(--border)';
    const align = isUser ? 'flex-end' : 'flex-start';
    const label = isUser ? '<strong>Engineer (You):</strong>' : '<strong>Assistant:</strong>';
    
    let text = msg.text
      .replace(/## (.*)/g, '<strong>$1</strong>')
      .replace(/### (.*)/g, '<strong>$1</strong>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/- (.*)/g, '• $1')
      .replace(/\n/g, '<br>');
      
    return `<div style="background: ${bg}; border: ${border}; border-radius: 8px; padding: 0.45rem 0.55rem; width: 90%; align-self: ${align}; word-wrap: break-word; line-height: 1.4;">
      ${label}<br>${text}
    </div>`;
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
    
    const simRes = lastSim || {
      config: config,
      forces: {
        drag_N: 1200,
        downforce_total_N: 3500,
        downforce_front_N: 1500,
        downforce_rear_N: 2000,
        balance_front_pct: 42.8,
        cop_x_m: 1.6,
        cl_total: 2.3,
        cd_total: 0.8,
        efficiency: 2.92
      },
      lap_time_delta_s: 0.0,
      method: method
    };
    
    const fullQuestion = `[Active track context: ${track.name} - ${track.description}] User question: ${text}`;
    
    const payload = {
      config: config,
      simulation: simRes,
      question: fullQuestion
    };
    
    const res = await fetch(`${API}/api/advise`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    chatHistory.pop();
    if (!res.ok) throw new Error('Advisor API offline');
    
    const data = await res.json();
    chatHistory.push({ sender: 'assistant', text: data.advice });
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
    fetch(`${API}/api/models/upload`, { method: 'POST', body: (() => { const f = new FormData(); f.append('file', file); return f; })() }).catch(() => {});
    runSimulate();
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  }
}

function setupModelImport() {
  const input = document.getElementById('modelInput');
  const dropZone = document.getElementById('viewerWrap');
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
  
  slicePosSlider.min = minVal;
  slicePosSlider.max = maxVal;
  slicePosSlider.step = step;
  
  const curVal = +slicePosSlider.value;
  if (curVal < minVal || curVal > maxVal) {
    slicePosSlider.value = ((minVal + maxVal) / 2).toFixed(2);
  }
  
  const wandYSlider = document.getElementById('wand_y');
  if (wandYSlider) {
    wandYSlider.min = bounds.min[1] - 0.05;
    wandYSlider.max = bounds.max[1] + 0.3;
    wandYSlider.step = 0.01;
  }
  const wandZSlider = document.getElementById('wand_z');
  if (wandZSlider) {
    wandZSlider.min = bounds.min[2] - 0.1;
    wandZSlider.max = bounds.max[2] + 0.1;
    wandZSlider.step = 0.01;
  }
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
    'streamline_style', 'particle_style',
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

function boot() {
  try {
    viewer = new AeroViewer(document.getElementById('viewer3d'));
    viewer.onSpeedRange = (range) => updateEngineerPanel(range);
    viewer.onProbe = showProbe;
    setupModelImport();
    setupControls();
    updateSliderRanges();
    updateUIState();
    syncSliderLabels();
    refreshVisualization();
    renderChat();
    runSimulate();
  } catch (err) {
    document.getElementById('bootError').textContent = '3D viewer failed: ' + err.message;
    document.getElementById('bootError').style.display = 'block';
  }
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
