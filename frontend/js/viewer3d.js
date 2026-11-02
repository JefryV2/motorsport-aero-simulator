/** Three.js viewer — wind tunnel, air/car interaction, model controls. */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { computeCpField, computeSpeedField, cpToColor } from './aeroField.js';
import {
  DEFAULT_FLOW,
  velocityAt,
  generateStreamlines,
  spawnOneParticle,
  speedToColor,
  windDirection,
  reynoldsNumber,
  distToBBox,
  isInsideBBox,
  speedsAlongPath,
} from './flowEngine.js';
import { VIEW_THEME } from './theme.js';
import { MeshFlowSampler, velocityAtMesh, generateSurfaceStreamlines, spawnSurfaceParticle, snapToSurface, surfaceOffset } from './meshFlow.js';
import { CP_MIN, CP_MAX } from './aeroField.js';

const FALLBACK_CONFIG = {
  front_wing_main: { angle_deg: 2, chord_m: 0.35, span_m: 1.8, gurney_mm: 0 },
  front_wing_flap: { angle_deg: 8, chord_m: 0.12, span_m: 1.75, gurney_mm: 0 },
  rear_wing_main: { angle_deg: 5, chord_m: 0.4, span_m: 0.95, gurney_mm: 0 },
  rear_wing_flap: { angle_deg: 15, chord_m: 0.15, span_m: 0.9, gurney_mm: 0 },
  ride_height_front_mm: 25,
  ride_height_rear_mm: 45,
  rake_deg: 1.5,
  diffuser_angle_deg: 12,
  diffuser_ride_height_mm: 30,
  speed_kmh: 200,
};

function particleTexture() {
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 64;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, VIEW_THEME.particleCore);
  g.addColorStop(0.45, 'rgba(120,160,220,0.7)');
  g.addColorStop(1, VIEW_THEME.particleEdge);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

const SMOKE_CELL_PX = 128;
const SMOKE_GRID = 3;
/** Particle styles rendered as streak tubes rather than screen-space dots. */
const SMOKE_STYLES = new Set(['smoke', 'filament']);
/** History points per filament, and the segments drawn between them. */
const SMOKE_TRAIL = 56;
const SMOKE_SEG = SMOKE_TRAIL - 1;
/** Record the next history point every this fraction of model length. */
const SMOKE_ARC_STEP = 0.02;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Tiled value noise on a cells x cells lattice, smoothstep-interpolated. */
function noiseOctave(size, cells, rnd) {
  const lattice = new Float32Array(cells * cells);
  for (let i = 0; i < lattice.length; i++) lattice[i] = rnd();
  const out = new Float32Array(size * size);
  const k = cells / size;
  for (let y = 0; y < size; y++) {
    const fy = y * k;
    const y0 = Math.floor(fy);
    const ty = fy - y0;
    const sy = ty * ty * (3 - 2 * ty);
    const r0 = y0 % cells;
    const r1 = (r0 + 1) % cells;
    for (let x = 0; x < size; x++) {
      const fx = x * k;
      const x0 = Math.floor(fx);
      const tx = fx - x0;
      const sx = tx * tx * (3 - 2 * tx);
      const c0 = x0 % cells;
      const c1 = (c0 + 1) % cells;
      const top = lattice[r0 * cells + c0] + (lattice[r0 * cells + c1] - lattice[r0 * cells + c0]) * sx;
      const bot = lattice[r1 * cells + c0] + (lattice[r1 * cells + c1] - lattice[r1 * cells + c0]) * sx;
      out[y * size + x] = top + (bot - top) * sy;
    }
  }
  return out;
}

/**
 * Streak cell: alpha falls off across the flow axis but is nearly constant
 * along it, with soft ends and turbulent modulation. Trail quads sampling this
 * butt together into a tube; the round puff cell left gaps the size of the
 * quad spacing, which is why the smoke read as a string of particles.
 */
function drawStreakCell(data, ox, oy, rnd) {
  const size = SMOKE_CELL_PX;
  const atlasW = size * SMOKE_GRID;
  const octs = [
    noiseOctave(size, 4, rnd),
    noiseOctave(size, 11, rnd),
    noiseOctave(size, 23, rnd),
  ];
  const c = size / 2;
  for (let y = 0; y < size; y++) {
    const v = (y - c) / c;
    const across = Math.max(0, 1 - v * v);
    const softAcross = across * across * (3 - 2 * across);
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const u = x / (size - 1);
      // Soft ends: a tube should fade out, not be cut off at the quad edge.
      const endFade = Math.min(1, u / 0.12) * Math.min(1, (1 - u) / 0.12);
      const n = 0.5 * octs[0][i] + 0.32 * octs[1][i] + 0.18 * octs[2][i];
      const alpha = softAcross * endFade * (0.55 + 0.7 * n);
      const p = ((oy + y) * atlasW + (ox + x)) * 4;
      data[p] = 255;
      data[p + 1] = 255;
      data[p + 2] = 255;
      data[p + 3] = Math.max(0, Math.min(255, Math.round(alpha * 255)));
    }
  }
}

function streakAtlasTexture() {
  const size = SMOKE_CELL_PX;
  const atlasW = size * SMOKE_GRID;
  const c = document.createElement('canvas');
  c.width = atlasW;
  c.height = atlasW;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(atlasW, atlasW);
  for (let q = 0; q < SMOKE_GRID * SMOKE_GRID; q++) {
    drawStreakCell(
      img.data,
      (q % SMOKE_GRID) * size,
      Math.floor(q / SMOKE_GRID) * size,
      mulberry32(4241 + q * 6151),
    );
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(c);
  // Thin threads minify the 128 px cell to a few pixels; mipmaps keep them from
  // shimmering. Cell edges are transparent, so gentle cross-cell averaging is
  // invisible.
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.anisotropy = 4;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  return tex;
}

export class AeroViewer {
  constructor(container) {
    this.container = container;
    this.carMesh = null;
    this.carGroup = new THREE.Group();
    this.flowGroup = new THREE.Group();
    this.surfaceFlowGroup = new THREE.Group();
    this.sliceGroup = new THREE.Group();
    this.forceGroup = new THREE.Group();
    this.particleGroup = new THREE.Group();
    this.tunnelGroup = new THREE.Group();
    this.windArrowGroup = new THREE.Group();
    this.bounds = { min: [-1.75, 0, -0.5], max: [1.75, 0.6, 0.5] };
    this.vertexData = null;
    this.simForces = null;
    this.config = { ...FALLBACK_CONFIG };
    this.flow = { ...DEFAULT_FLOW };
    this.viewMode = 'pressure';
    this.showStreamlines = true;
    this.showParticles = true;
    this.showForces = true;
    this.showSlice = false;
    this.sliceAxis = 'z';
    this.slicePos = 0;
    this.sliceMode = 'pressure';
    this.probeMode = false;
    this.speedRange = { min: 0, max: 55 };
    this.probeMarker = null;
    this.wandMarker = null;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.particles = [];
    this.clock = new THREE.Clock();
    this.refSpeed = 55;
    this.meshSampler = null;
    this.userModelScale = 1;
    this.userModelRotY = 0;
    this._defaultCameraPos = new THREE.Vector3(5, 2.8, 5);
    this._defaultTarget = new THREE.Vector3(0, 0.35, 0);
    this._particleTex = particleTexture();

    this._init();
    this.setDefaultCar();
    this._startFlow();
  }

  _init() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(VIEW_THEME.background);

    const w = Math.max(this.container.clientWidth, 400);
    const h = Math.max(this.container.clientHeight, 480);
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 200);
    this.camera.position.copy(this._defaultCameraPos);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = true;
    this.controls.enableZoom = true;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 40;
    this.controls.target.copy(this._defaultTarget);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(4, 6, 3);
    this.scene.add(key);
    this.scene.add(new THREE.GridHelper(14, 28, VIEW_THEME.gridPrimary, VIEW_THEME.gridSecondary));
    this.scene.add(this.tunnelGroup);
    this.scene.add(this.carGroup);
    this.scene.add(this.flowGroup);
    this.scene.add(this.surfaceFlowGroup);
    this.scene.add(this.sliceGroup);
    this.scene.add(this.particleGroup);
    this.scene.add(this.forceGroup);
    this.scene.add(this.windArrowGroup);

    this.probeMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.04, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xffff00, transparent: true, opacity: 0.9 }),
    );
    this.probeMarker.visible = false;
    this.scene.add(this.probeMarker);

    this.wandMarker = new THREE.Mesh(
      new THREE.RingGeometry(0.04, 0.05, 16),
      new THREE.MeshBasicMaterial({ color: 0x00d2be, side: THREE.DoubleSide })
    );
    this.wandMarker.rotation.y = Math.PI / 2;
    this.wandMarker.visible = false;
    this.scene.add(this.wandMarker);

    this.renderer.domElement.addEventListener('click', (e) => this._onCanvasClick(e));

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.container);
    setTimeout(() => this._onResize(), 100);
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w < 10 || h < 10) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _animate() {
    requestAnimationFrame(this._animate);
    const dt = Math.min(this.clock.getDelta(), 0.05) * (this.flow.timeScale || 1);
    // The eddy field evolves slowly in time; tracers advect through it.
    this.flow.time = (this.flow.time || 0) + dt;
    this.controls.update();
    if (this.showParticles) this._tickParticles(dt);
    this._pulseStreamlines();
    this.renderer.render(this.scene, this.camera);
  }

  _buildMeshSampler() {
    if (!this.vertexData?.vertices?.length || !this.carMesh) {
      this.meshSampler = null;
      return;
    }
    this.carGroup.updateMatrixWorld(true);
    this.carMesh.updateMatrixWorld(true);
    const m = this.carMesh.matrixWorld;
    const e = m.elements;
    const worldVerts = [];
    const worldNorms = [];
    for (let i = 0; i < this.vertexData.vertices.length; i++) {
      const [x, y, z] = this.vertexData.vertices[i];
      worldVerts.push([
        e[0] * x + e[4] * y + e[8] * z + e[12],
        e[1] * x + e[5] * y + e[9] * z + e[13],
        e[2] * x + e[6] * y + e[10] * z + e[14],
      ]);
      const [nx, ny, nz] = this.vertexData.normals[i] || [0, 1, 0];
      worldNorms.push([
        e[0] * nx + e[4] * ny + e[8] * nz,
        e[1] * nx + e[5] * ny + e[9] * nz,
        e[2] * nx + e[6] * ny + e[10] * nz,
      ]);
    }
    this.meshSampler = new MeshFlowSampler(worldVerts, worldNorms, this.bounds);
  }

  _seedPuff(p) {
    p.sizeJitter = 0.75 + Math.random() * 0.6;
    const q = Math.floor(Math.random() * SMOKE_GRID * SMOKE_GRID);
    p.uvOff = [(q % SMOKE_GRID) / SMOKE_GRID, Math.floor(q / SMOKE_GRID) / SMOKE_GRID];
    if (!p.hist) p.hist = new Float32Array(SMOKE_TRAIL * 3);
    // Collapse the filament onto the spawn point, otherwise a respawn drags a
    // stale tail across the tunnel for one frame.
    for (let k = 0; k < SMOKE_TRAIL; k++) {
      p.hist[k * 3] = p.x;
      p.hist[k * 3 + 1] = p.y;
      p.hist[k * 3 + 2] = p.z;
    }
    return p;
  }

  _spawnParticles() {
    const count = this.flow.particleCount || 300;
    const useSurface = this.meshSampler && this.flow.showInteraction !== false;
    // Inlet-seeded sources need more of the budget to read as a continuous
    // curtain; surface tracers still get a useful share for near-body detail.
    const inletHeavy = ['front_grid', 'front_wide', 'smoke_rake', 'chamber'].includes(
      this.flow.flowSource,
    );
    // A rake wants most of the budget in the inflow so the threads stay
    // continuous; other sources still need near-body detail.
    const surfaceFraction = this.flow.flowSource === 'smoke_rake' ? 0.28 : inletHeavy ? 0.45 : 0.65;
    const surfaceCount = useSurface ? Math.floor(count * surfaceFraction) : 0;
    const particles = [];
    for (let i = 0; i < count; i++) {
      if (i < surfaceCount) {
        particles.push(this._seedPuff(spawnSurfaceParticle(this.meshSampler, this.bounds, this.flow, i)));
      } else {
        particles.push(this._seedPuff(spawnOneParticle(this.bounds, this.flow, i)));
      }
    }
    return particles;
  }

  _velocityAt(x, y, z, config) {
    if (this.meshSampler && this.flow.showInteraction !== false) {
      return velocityAtMesh(x, y, z, this.meshSampler, velocityAt, config, this.bounds, this.flow);
    }
    return velocityAt(x, y, z, config, this.bounds, this.flow);
  }

  _respawnParticle(p, index) {
    if (p.surface && this.meshSampler) {
      Object.assign(p, spawnSurfaceParticle(this.meshSampler, this.bounds, this.flow, index));
    } else {
      Object.assign(p, spawnOneParticle(this.bounds, this.flow, index));
    }
    return this._seedPuff(p);
  }

  _startFlow() {
    this.particles = this._spawnParticles();
    this._buildParticleMesh();
    this._rebuildFlowScene();
    this.updateAero(this.config, this.simForces);
    document.getElementById('aeroBadge')?.classList.add('active');
  }

  _worldSamplePoints() {
    if (!this.vertexData?.vertices?.length || !this.carMesh) return this.vertexData;
    this.carGroup.updateMatrixWorld(true);
    this.carMesh.updateMatrixWorld(true);
    const m = this.carMesh.matrixWorld;
    const e = m.elements;
    const vertices = [];
    const normals = [];
    for (let i = 0; i < this.vertexData.vertices.length; i++) {
      const [x, y, z] = this.vertexData.vertices[i];
      vertices.push([
        e[0] * x + e[4] * y + e[8] * z + e[12],
        e[1] * x + e[5] * y + e[9] * z + e[13],
        e[2] * x + e[6] * y + e[10] * z + e[14],
      ]);
      const [nx, ny, nz] = this.vertexData.normals[i] || [0, 1, 0];
      normals.push([
        e[0] * nx + e[4] * ny + e[8] * nz,
        e[1] * nx + e[5] * ny + e[9] * nz,
        e[2] * nx + e[6] * ny + e[10] * nz,
      ]);
    }
    return { vertices, normals };
  }

  _onCanvasClick(event) {
    if (!this.probeMode || !this.carMesh) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObject(this.carMesh, false);
    if (!hits.length) return;
    const hit = hits[0];
    this.probeMarker.position.copy(hit.point);
    this.probeMarker.visible = true;
    const probe = this.probeAtPoint(hit.point, hit.face?.normal);
    this.onProbe?.(probe);
  }

  probeAtPoint(point, faceNormal = null) {
    const cfg = this._activeConfig();
    const p = [point.x, point.y, point.z];
    const [vx, vy, vz] = this._velocityAt(p[0], p[1], p[2], cfg);
    const speedMs = Math.sqrt(vx * vx + vy * vy + vz * vz);
    let cp = null;
    let idx = -1;
    if (this.meshSampler) {
      const near = this.meshSampler.nearest(p[0], p[1], p[2], Infinity);
      if (near && this.cpField?.length) {
        idx = near.index;
        cp = this.cpField[idx];
      }
    }
    return {
      position: p,
      speedMs,
      speedKmh: speedMs * 3.6,
      velocity: [vx, vy, vz],
      cp,
      normal: faceNormal ? [faceNormal.x, faceNormal.y, faceNormal.z] : null,
    };
  }

  getSpeedRange() {
    return { ...this.speedRange };
  }

  setCameraView(preset) {
    if (!this.carMesh) return;
    const min = this.bounds.min;
    const max = this.bounds.max;
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    const cz = (min[2] + max[2]) / 2;
    const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1);
    const d = span * 2.2;
    const target = new THREE.Vector3(cx, cy, cz);
    const views = {
      iso: new THREE.Vector3(cx + d * 0.85, cy + d * 0.55, cz + d * 0.85),
      front: new THREE.Vector3(min[0] - d * 0.9, cy + span * 0.15, cz),
      rear: new THREE.Vector3(max[0] + d * 0.9, cy + span * 0.15, cz),
      side: new THREE.Vector3(cx, cy + span * 0.1, max[2] + d * 0.95),
      top: new THREE.Vector3(cx, max[1] + d * 0.95, cz),
      plan: new THREE.Vector3(cx, max[1] + d * 1.1, cz + 0.01),
    };
    const pos = views[preset] || views.iso;
    this.camera.position.copy(pos);
    this.controls.target.copy(target);
    this.controls.update();
  }

  setProbeMode(on) {
    this.probeMode = on;
    this.renderer.domElement.style.cursor = on ? 'crosshair' : '';
    if (!on) {
      this.probeMarker.visible = false;
    }
  }

  toggleProbe(show) { this.setProbeMode(show); }

  setFlowSettings(flow) {
    this.flow = { ...this.flow, ...flow };
    this.refSpeed = (this.flow.windSpeedKmh || 200) / 3.6;
    if (flow.showSlice !== undefined) this.showSlice = flow.showSlice;
    if (flow.sliceAxis !== undefined) this.sliceAxis = flow.sliceAxis;
    if (flow.slicePos !== undefined) this.slicePos = flow.slicePos;
    if (flow.sliceMode !== undefined) this.sliceMode = flow.sliceMode;
    this.particles = this._spawnParticles();
    this._buildParticleMesh();
    this._rebuildFlowScene();
    this._updateFlowHud();
    if (this.showStreamlines) this._updateStreamlines(this._activeConfig());
    this._updateSlicePlane();
    this._updateWandMarker();
  }

  setModelScale(pct) {
    this.userModelScale = pct / 100;
    this.carGroup.scale.setScalar(this.userModelScale);
    this._syncBoundsFromModel();
    this._buildMeshSampler();
    this.particles = this._spawnParticles();
    this._buildParticleMesh();
    this._rebuildFlowScene();
    if (this.showStreamlines) this._updateStreamlines(this._activeConfig());
  }

  setModelRotation(deg) {
    this.userModelRotY = deg;
    this.carGroup.rotation.y = (deg * Math.PI) / 180;
    this._syncBoundsFromModel();
    this._buildMeshSampler();
    this.particles = this._spawnParticles();
    this._buildParticleMesh();
    if (this.showStreamlines) this._updateStreamlines(this._activeConfig());
  }

  resetCamera() {
    this.camera.position.copy(this._defaultCameraPos);
    this.controls.target.copy(this._defaultTarget);
    this.controls.update();
  }

  _syncBoundsFromModel() {
    if (!this.carMesh) return;
    const box = new THREE.Box3().setFromObject(this.carGroup);
    this.bounds = {
      min: [box.min.x, box.min.y, box.min.z],
      max: [box.max.x, box.max.y, box.max.z],
    };
    const h = box.max.y - box.min.y;
    this._defaultTarget.set(0, box.min.y + h * 0.45, 0);
  }

  _updateFlowHud() {
    const el = document.getElementById('flowHud');
    if (!el) return;
    el.textContent = `${this.flow.windSpeedKmh} km/h · ${this.refSpeed.toFixed(1)} m/s · Re=${(reynoldsNumber(this.flow) / 1e6).toFixed(2)}M · ${this.particles.length} particles`;
  }

  _rebuildFlowScene() {
    this._clearGroup(this.tunnelGroup);
    this._clearGroup(this.windArrowGroup);
    const min = this.bounds.min;
    const max = this.bounds.max;
    const span = max.map((v, i) => v - min[i]);
    const pad = 0.35;

    if (this.flow.showTunnel) {
      const box = new THREE.BoxGeometry(span[0] + pad * 2, span[1] + pad * 1.5, span[2] + pad * 2);
      const tunnel = new THREE.LineSegments(
        new THREE.EdgesGeometry(box),
        new THREE.LineBasicMaterial({ color: VIEW_THEME.tunnelLine, transparent: true, opacity: 0.55 }),
      );
      tunnel.position.set((min[0] + max[0]) / 2, (min[1] + max[1]) / 2 + pad * 0.2, (min[2] + max[2]) / 2);
      this.tunnelGroup.add(tunnel);
    }

    if (this.flow.showGround) {
      const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(span[0] + pad * 3, span[2] + pad * 3),
        new THREE.MeshBasicMaterial({ color: 0x1a2030, transparent: true, opacity: 0.65, side: THREE.DoubleSide }),
      );
      ground.rotation.x = -Math.PI / 2;
      ground.position.set((min[0] + max[0]) / 2, min[1], (min[2] + max[2]) / 2);
      this.tunnelGroup.add(ground);
    }

    if (this.flow.showSourcePlane) {
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(span[2] * 1.2, span[1] * 1.2, 10, 10),
        new THREE.MeshBasicMaterial({ color: VIEW_THEME.sourcePlane, transparent: true, opacity: 0.15, wireframe: true, side: THREE.DoubleSide }),
      );
      plane.position.set(min[0] - pad * 0.9, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
      plane.rotation.y = -Math.PI / 2;
      this.tunnelGroup.add(plane);
    }

    if (this.flow.showWake) {
      const wake = new THREE.Mesh(
        new THREE.ConeGeometry(span[2] * 0.5, span[0] * 0.8, 12, 1, true),
        new THREE.MeshBasicMaterial({ color: VIEW_THEME.wake, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }),
      );
      wake.rotation.z = -Math.PI / 2;
      wake.position.set(max[0] + span[0] * 0.4, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
      this.tunnelGroup.add(wake);
    }

    const dir = windDirection(this.flow);
    this.windArrowGroup.add(new THREE.ArrowHelper(
      new THREE.Vector3(dir[0], dir[1], dir[2]),
      new THREE.Vector3(min[0] - span[0] * 0.12, max[1] + span[1] * 0.35, (min[2] + max[2]) / 2),
      1.2, VIEW_THEME.windArrow, 0.3, 0.18,
    ));
  }

  _clearGroup(g) {
    while (g.children.length) {
      const c = g.children[0];
      c.geometry?.dispose();
      if (c.material) {
        if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose());
        else c.material.dispose();
      }
      g.remove(c);
    }
  }

  _clearCar() {
    this._clearGroup(this.carGroup);
    this.carMesh = null;
    this.vertexData = null;
    this.meshSampler = null;
  }

  _autoOrient(geometry) {
    geometry.computeBoundingBox();
    const size = new THREE.Vector3();
    geometry.boundingBox.getSize(size);
    const axes = [{ rot: null, dim: size.x }, { rot: 'yToX', dim: size.y }, { rot: 'zToX', dim: size.z }].sort((a, b) => b.dim - a.dim);
    if (axes[0].rot === 'yToX') geometry.rotateZ(Math.PI / 2);
    else if (axes[0].rot === 'zToX') geometry.rotateY(-Math.PI / 2);
    geometry.computeBoundingBox();
    const s2 = new THREE.Vector3();
    geometry.boundingBox.getSize(s2);
    if (s2.y > s2.z && s2.y > s2.x * 0.5) geometry.rotateX(-Math.PI / 2);
  }

  _fitAndStore(mesh, name) {
    mesh.geometry = mesh.geometry.index ? mesh.geometry.toNonIndexed() : mesh.geometry;
    this._autoOrient(mesh.geometry);
    mesh.geometry.computeBoundingBox();
    const box = mesh.geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    mesh.geometry.translate(-center.x, -box.min.y, -center.z);
    mesh.geometry.computeBoundingBox();
    const size = new THREE.Vector3();
    mesh.geometry.boundingBox.getSize(size);
    mesh.scale.setScalar(3.8 / Math.max(size.x, size.y, size.z));
    mesh.geometry.computeBoundingBox();
    mesh.geometry.computeVertexNormals();

    const pos = mesh.geometry.attributes.position;
    const norm = mesh.geometry.attributes.normal;
    const vertices = [];
    const normals = [];
    for (let i = 0; i < pos.count; i++) {
      vertices.push([pos.getX(i), pos.getY(i), pos.getZ(i)]);
      normals.push([norm.getX(i), norm.getY(i), norm.getZ(i)]);
    }
    this.vertexData = { vertices, normals };
    this.modelName = name;

    mesh.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3));
    mesh.material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    this.carMesh = mesh;
    this.carGroup.add(mesh);
    this.carGroup.scale.setScalar(this.userModelScale);
    this.carGroup.rotation.y = (this.userModelRotY * Math.PI) / 180;
    this._syncBoundsFromModel();
    this._buildMeshSampler();

    this._defaultCameraPos.set(5, this.bounds.max[1] + 2, 5);
    this.controls.target.copy(this._defaultTarget);
    this.camera.position.copy(this._defaultCameraPos);
  }

  setDefaultCar() {
    this._clearCar();
    const group = new THREE.Group();
    const mk = (w, h, d, x, y, z) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial({ color: 0x555 }));
      m.position.set(x, y, z);
      group.add(m);
    };
    mk(0.35, 0.14, 0.5, -1.45, 0.22, 0);
    mk(0.5, 0.2, 0.55, -0.9, 0.35, 0);
    mk(0.9, 0.16, 0.35, -0.3, 0.2, 0.45);
    mk(0.9, 0.16, 0.35, -0.3, 0.2, -0.45);
    mk(0.7, 0.2, 0.5, 0.5, 0.38, 0);
    mk(0.08, 0.025, 1.6, -1.65, 0.12, 0);
    mk(0.04, 0.035, 1.55, -1.58, 0.1, 0);
    mk(0.06, 0.02, 0.85, 1.35, 0.75, 0);
    mk(0.04, 0.04, 0.8, 1.28, 0.72, 0);
    group.updateMatrixWorld(true);
    const geos = [];
    group.traverse((c) => { if (c.isMesh) { const g = c.geometry.clone(); g.applyMatrix4(c.matrixWorld); geos.push(g); } });
    this._fitAndStore(new THREE.Mesh(mergeGeometries(geos)), 'Default formula car');
  }

  async loadFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const buffer = await file.arrayBuffer();
    this._clearCar();

    const finalize = (geos) => {
      if (!geos?.length) throw new Error('No mesh geometry found in file');
      this._fitAndStore(new THREE.Mesh(mergeGeometries(geos)), file.name);
    };

    if (ext === 'stl') {
      finalize([new STLLoader().parse(buffer)]);
    } else if (ext === 'obj') {
      const obj = new OBJLoader().parse(new TextDecoder().decode(buffer));
      finalize(this._collectMeshGeometries(obj));
    } else if (ext === 'glb' || ext === 'gltf') {
      const gltf = await new GLTFLoader().parseAsync(buffer, '');
      finalize(this._collectMeshGeometries(gltf.scene));
    } else if (ext === 'fbx') {
      const fbx = new FBXLoader().parse(buffer, '');
      finalize(this._collectMeshGeometries(fbx));
    } else if (ext === 'dae') {
      const collada = new ColladaLoader().parse(new TextDecoder().decode(buffer), '');
      finalize(this._collectMeshGeometries(collada.scene));
    } else if (ext === 'ply') {
      finalize([new PLYLoader().parse(buffer)]);
    } else {
      throw new Error(`Unsupported ".${ext}". Use STL, OBJ, GLB, GLTF, FBX, DAE, or PLY.`);
    }

    this._startFlow();
    return { filename: file.name, bounds: this.bounds, vertexCount: this.vertexData.vertices.length };
  }

  _collectMeshGeometries(root) {
    const geos = [];
    root.updateMatrixWorld(true);
    root.traverse((c) => {
      if (c.isMesh && c.geometry) {
        const g = c.geometry.clone();
        g.applyMatrix4(c.matrixWorld);
        if (!g.attributes.normal) g.computeVertexNormals();
        geos.push(g);
      }
    });
    return geos;
  }

  updateAero(config, simForces = null) {
    if (!this.carMesh || !this.vertexData) return;
    this.config = config;
    this.simForces = simForces;

    const forces = simForces?.forces || simForces;
    const world = this._worldSamplePoints();
    const velFn = (x, y, z, cfg, bnds, fl) => this._velocityAt(x, y, z, cfg);

    let colors;
    let range;
    if (this.viewMode === 'velocity') {
      ({ colors, range } = computeSpeedField(
        world.vertices, world.normals, config, this.bounds, this.flow, velFn,
      ));
    } else {
      const result = computeCpField(
        world.vertices, world.normals, config, this.bounds, forces, this.flow,
      );
      colors = result.colors;
      range = result.range;
      this.cpField = result.cp;
    }

    const attr = this.carMesh.geometry.attributes.color;
    for (let i = 0; i < colors.length; i++) attr.setXYZ(i, colors[i].r, colors[i].g, colors[i].b);
    attr.needsUpdate = true;
    this._applyViewMode();

    if (this.showStreamlines) this._updateStreamlines(config);
    if (this.showForces && forces) this._updateForces(forces);
    this._updateSlicePlane();
    this._updateWandMarker();

    const el = document.getElementById('cpRange');
    if (el) {
      if (this.viewMode === 'velocity') {
        el.textContent = `Speed ${(range.min * 3.6).toFixed(0)}–${(range.max * 3.6).toFixed(0)} km/h on surface · warm=slow · cool=fast`;
      } else {
        el.textContent = `Cp ${CP_MIN.toFixed(1)} (blue suction) → ${CP_MAX.toFixed(1)} (red stagnation) · car ${range.min.toFixed(2)} to ${range.max.toFixed(2)}`;
      }
    }
    this._updateFlowHud();
  }

  _activeConfig() {
    return this.config || FALLBACK_CONFIG;
  }

  _applyViewMode() {
    if (!this.carMesh) return;
    this.carMesh.material?.dispose();
    if (this.viewMode === 'solid') {
      this.carMesh.material = new THREE.MeshStandardMaterial({ color: 0x8892a8, metalness: 0.5, roughness: 0.4, side: THREE.DoubleSide });
    } else if (this.viewMode === 'wireframe') {
      this.carMesh.material = new THREE.MeshBasicMaterial({ color: 0x6b7280, wireframe: true });
    } else {
      this.carMesh.material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    }
  }

  _updateStreamlines(config) {
    this._clearGroup(this.flowGroup);
    this._clearGroup(this.surfaceFlowGroup);

    const velFn = (x, y, z, cfg) => this._velocityAt(x, y, z, cfg);
    const freestream = generateStreamlines(config, this.bounds, this.flow, velFn);
    const surface = this.meshSampler && this.flow.showInteraction !== false
      ? generateSurfaceStreamlines(
        this.meshSampler, config, this.bounds, this.flow, velocityAt,
        Math.floor((this.flow.streamlineCount || 28) * 0.75),
      )
      : [];

    const tubeR = Math.max(this.bounds.max[1] - this.bounds.min[1], 0.08) * 0.016;
    const velFnFull = (x, y, z, cfg, bnds, fl) => this._velocityAt(x, y, z, cfg);

    const addSpeedColoredLines = (lines, group, opacity) => {
      const style = this.flow.streamlineStyle || 'dashed';
      const isDashed = style === 'dashed';
      const isTubes = style === 'tubes';

      lines.forEach((pts) => {
        if (pts.length < 2) return;
        const speeds = speedsAlongPath(pts, config, this.bounds, this.flow, velFnFull);
        let minS = Infinity;
        let maxS = 0;
        for (let k = 0; k < speeds.length; k++) {
          if (speeds[k] < minS) minS = speeds[k];
          if (speeds[k] > maxS) maxS = speeds[k];
        }

        if (isTubes) {
          const vectors = pts.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
          const curve = new THREE.CatmullRomCurve3(vectors);
          const tubeRadius = tubeR * 0.42;
          const segments = Math.min(pts.length * 2, 70);
          const tubeGeo = new THREE.TubeGeometry(curve, segments, tubeRadius, 6, false);

          const posAttr = tubeGeo.attributes.position;
          const colors = new Float32Array(posAttr.count * 3);
          for (let k = 0; k < posAttr.count; k++) {
            const px = posAttr.getX(k);
            const py = posAttr.getY(k);
            const pz = posAttr.getZ(k);
            const [vx, vy, vz] = velFnFull(px, py, pz, config);
            const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
            const c = speedToColor(speed, this.refSpeed, minS, maxS);
            colors[k * 3] = c.r;
            colors[k * 3 + 1] = c.g;
            colors[k * 3 + 2] = c.b;
          }
          tubeGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

          const tubeMesh = new THREE.Mesh(
            tubeGeo,
            new THREE.MeshStandardMaterial({
              vertexColors: true,
              roughness: 0.28,
              metalness: 0.12,
              transparent: true,
              opacity: opacity * 0.9,
              side: THREE.DoubleSide,
            })
          );
          group.add(tubeMesh);
        } else {
          const positions = new Float32Array(pts.length * 3);
          const colors = new Float32Array(pts.length * 3);
          for (let j = 0; j < pts.length; j++) {
            positions[j * 3] = pts[j][0];
            positions[j * 3 + 1] = pts[j][1];
            positions[j * 3 + 2] = pts[j][2];
            const c = speedToColor(speeds[j], this.refSpeed, minS, maxS);
            colors[j * 3] = c.r;
            colors[j * 3 + 1] = c.g;
            colors[j * 3 + 2] = c.b;
          }

          const geo = new THREE.BufferGeometry();
          geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
          geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

          let line;
          if (isDashed) {
            line = new THREE.Line(
              geo,
              new THREE.LineDashedMaterial({
                vertexColors: true,
                transparent: true,
                opacity: opacity * 0.9,
                dashSize: 0.12,
                gapSize: 0.08,
                scale: 1,
              })
            );
            line.computeLineDistances();
          } else {
            line = new THREE.Line(
              geo,
              new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity, linewidth: 2 }),
            );
          }
          group.add(line);
        }
      });
    };

    addSpeedColoredLines(freestream, this.flowGroup, 0.85);
    addSpeedColoredLines(surface, this.surfaceFlowGroup, 0.95);
  }

  _pulseStreamlines() {
    const elapsed = Date.now() * 0.001;
    const isDashed = this.flow.streamlineStyle === 'dashed';

    if (isDashed) {
      const speed = this.refSpeed * 0.08;
      this.flowGroup.children.forEach((c) => {
        if (c.material && c.material.type === 'LineDashedMaterial') {
          c.material.dashOffset = -elapsed * speed;
        }
      });
      this.surfaceFlowGroup.children.forEach((c) => {
        if (c.material && c.material.type === 'LineDashedMaterial') {
          c.material.dashOffset = -elapsed * speed;
        }
      });
    } else {
      this.flowGroup.children.forEach((c, i) => {
        if (c.material && c.material.type === 'LineBasicMaterial') {
          c.material.opacity = 0.45 + 0.35 * Math.sin(Date.now() * 0.004 + i * 0.5);
        }
      });
      this.surfaceFlowGroup.children.forEach((c, i) => {
        if (c.material && c.material.type === 'LineBasicMaterial') {
          c.material.opacity = 0.65 + 0.3 * Math.sin(Date.now() * 0.005 + i * 0.7);
        }
      });
    }
  }

  _updateSlicePlane() {
    this.sliceGroup.clear();
    if (!this.showSlice) return;

    const min = this.bounds.min;
    const max = this.bounds.max;
    const span = max.map((v, i) => Math.max(v - min[i], 0.01));

    let uMin, uMax, vMin, vMax;

    if (this.sliceAxis === 'z') {
      uMin = min[0] - span[0] * 0.15; uMax = max[0] + span[0] * 0.25;
      vMin = min[1] - 0.05;       vMax = max[1] + span[1] * 0.4;
    } else if (this.sliceAxis === 'y') {
      uMin = min[0] - span[0] * 0.15; uMax = max[0] + span[0] * 0.25;
      vMin = min[2] - span[2] * 0.15; vMax = max[2] + span[2] * 0.15;
    } else { // 'x'
      uMin = min[1] - 0.05;       uMax = max[1] + span[1] * 0.4;
      vMin = min[2] - span[2] * 0.15; vMax = max[2] + span[2] * 0.15;
    }

    const uSegments = 50;
    const vSegments = 35;
    
    const geo = new THREE.PlaneGeometry(uMax - uMin, vMax - vMin, uSegments, vSegments);
    geo.translate((uMin + uMax) / 2, (vMin + vMax) / 2, 0);

    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.82,
      depthWrite: false
    }));

    if (this.sliceAxis === 'z') {
      mesh.position.z = this.slicePos;
    } else if (this.sliceAxis === 'y') {
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = this.slicePos;
    } else { // 'x'
      mesh.rotation.y = Math.PI / 2;
      mesh.position.x = this.slicePos;
    }

    mesh.updateMatrixWorld(true);

    const posAttr = geo.attributes.position;
    const count = posAttr.count;
    const colors = new Float32Array(count * 3);
    const cfg = this._activeConfig();
    const ref = (this.flow.windSpeedKmh || 200) / 3.6;

    const tempV = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      tempV.set(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i));
      tempV.applyMatrix4(mesh.matrixWorld);

      const [vx, vy, vz] = this._velocityAt(tempV.x, tempV.y, tempV.z, cfg);
      const speedMs = Math.sqrt(vx * vx + vy * vy + vz * vz);

      let color;
      if (this.sliceMode === 'pressure') {
        const ratio = speedMs / (ref || 1);
        let cp = 1.0 - ratio * ratio;
        cp = Math.max(-2.0, Math.min(1.0, cp));
        color = cpToColor(cp);
      } else {
        color = speedToColor(speedMs, ref, 0, ref * 1.4);
      }

      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this.sliceGroup.add(mesh);
  }

  _updateWandMarker() {
    if (!this.wandMarker) return;
    if (this.flow.flowSource === 'smoke_wand') {
      const min = this.bounds.min;
      const max = this.bounds.max;
      const span = max.map((v, i) => v - min[i]);
      const wx = min[0] - span[0] * 0.05;
      const wy = this.flow.wandY !== undefined ? this.flow.wandY : (min[1] + max[1]) / 2;
      const wz = this.flow.wandZ !== undefined ? this.flow.wandZ : (min[2] + max[2]) / 2;
      this.wandMarker.position.set(wx, wy, wz);
      this.wandMarker.visible = true;
    } else {
      this.wandMarker.visible = false;
    }
  }

  _buildParticleMesh() {
    this._disposeParticleMesh();
    const n = this.particles.length;
    const isSmoke = SMOKE_STYLES.has(this.flow.particleStyle);

    if (isSmoke) {
      this._streakAtlas = this._streakAtlas || streakAtlasTexture();
      const cell = new THREE.PlaneGeometry(1, 1);
      const geo = new THREE.InstancedBufferGeometry();
      geo.index = cell.index;
      geo.setAttribute('position', cell.getAttribute('position'));
      geo.setAttribute('uv', cell.getAttribute('uv'));
      const inst = n * SMOKE_SEG;
      geo.instanceCount = inst;
      geo.setAttribute('iPos', new THREE.InstancedBufferAttribute(new Float32Array(inst * 3), 3));
      geo.setAttribute('iDir', new THREE.InstancedBufferAttribute(new Float32Array(inst * 3), 3));
      geo.setAttribute('iScale', new THREE.InstancedBufferAttribute(new Float32Array(inst * 2), 2));
      geo.setAttribute('iRot', new THREE.InstancedBufferAttribute(new Float32Array(inst), 1));
      geo.setAttribute('iColor', new THREE.InstancedBufferAttribute(new Float32Array(inst * 3), 3));
      geo.setAttribute('iOpacity', new THREE.InstancedBufferAttribute(new Float32Array(inst), 1));
      geo.setAttribute('iUvOffset', new THREE.InstancedBufferAttribute(new Float32Array(inst * 2), 2));

      this.particleQuads = new THREE.Mesh(geo, new THREE.ShaderMaterial({
        uniforms: { atlas: { value: this._streakAtlas } },
        vertexShader: `
          attribute vec3 iPos;
          attribute vec3 iDir;
          attribute vec2 iScale;
          attribute float iRot;
          attribute vec3 iColor;
          attribute float iOpacity;
          attribute vec2 iUvOffset;
          varying vec2 vUv;
          varying vec3 vColor;
          varying float vOpacity;
          void main() {
            // Stretch the quad along the travelled segment as it appears on
            // screen, so a puff reads as a smoke filament rather than a blob.
            vec4 mv = modelViewMatrix * vec4(iPos, 1.0);
            vec2 dir = (modelViewMatrix * vec4(iDir, 0.0)).xy;
            float ang = (dot(dir, dir) > 1e-8 ? atan(dir.y, dir.x) : 0.0) + iRot;
            float c = cos(ang);
            float s = sin(ang);
            vec2 q = vec2(position.x * iScale.x, position.y * iScale.y);
            mv.xy += vec2(q.x * c - q.y * s, q.x * s + q.y * c);
            gl_Position = projectionMatrix * mv;
            vUv = uv / ${SMOKE_GRID}.0 + iUvOffset;
            vColor = iColor;
            vOpacity = iOpacity;
          }
        `,
        fragmentShader: `
          uniform sampler2D atlas;
          varying vec2 vUv;
          varying vec3 vColor;
          varying float vOpacity;
          void main() {
            float density = texture2D(atlas, vUv).a;
            if (density < 0.01) discard;
            // Fake volume: sample across the tube toward the key light so the
            // lit side is brighter than the shaded side. Without this every
            // quad reads as a flat decal.
            float lit = texture2D(atlas, vUv + vec2(0.0, 0.05)).a;
            float shade = clamp(1.0 + (density - lit) * 2.4, 0.55, 1.25);
            gl_FragColor = vec4(vColor * shade, density * vOpacity);
          }
        `,
        transparent: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
      }));
      // Instance attributes change every frame, so computed bounds are stale.
      this.particleQuads.frustumCulled = false;
      this.particleGroup.add(this.particleQuads);
      return;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.particlePoints = new THREE.Points(geo, new THREE.PointsMaterial({
      map: this._particleTex,
      size: this.flow.particleSize || 0.18,
      vertexColors: true,
      transparent: true,
      opacity: 1,
      sizeAttenuation: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    }));
    this.particleGroup.add(this.particlePoints);
  }

  _disposeParticleMesh() {
    for (const obj of [this.particlePoints, this.particleQuads]) {
      if (!obj) continue;
      obj.geometry.dispose();
      obj.material.dispose();
    }
    this.particlePoints = null;
    this.particleQuads = null;
    this.particleGroup.clear();
  }

  _tickParticles(dt) {
    const isSmoke = SMOKE_STYLES.has(this.flow.particleStyle);
    const isFilament = this.flow.particleStyle === 'filament';
    const obj = isSmoke ? this.particleQuads : this.particlePoints;
    if (!obj) return;
    const cfg = this._activeConfig();
    const min = this.bounds.min;
    const max = this.bounds.max;
    const span = max.map((v, i) => v - min[i]);
    const attrs = obj.geometry.attributes;
    const pos = attrs.position;
    const col = attrs.color;
    const iPos = attrs.iPos;
    const iDir = attrs.iDir;
    const iScale = attrs.iScale;
    const iRot = attrs.iRot;
    const iCol = attrs.iColor;
    const iOpa = attrs.iOpacity;
    const iUv = attrs.iUvOffset;

    const ts = Math.max(0.5, (this.flow.timeScale || 1)) * 0.65;
    const arcStep = span[0] * SMOKE_ARC_STEP;
    const skinEps = surfaceOffset(this.bounds);
    const useMesh = this.meshSampler && this.flow.showInteraction !== false;
    const n = this.particles.length;

    // Pass 1 — move particles, track speeds for dynamic color scale
    let minSpd = Infinity;
    let maxSpd = 0;
    const frameSpeeds = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const p = this.particles[i];
      const [vx, vy, vz] = this._velocityAt(p.x, p.y, p.z, cfg);
      const spd = Math.sqrt(vx * vx + vy * vy + vz * vz);
      const dist = distToBBox(p.x, p.y, p.z, this.bounds);

      p.smoothSpd = p.smoothSpd == null ? spd : p.smoothSpd * 0.82 + spd * 0.18;

      p.x += vx * dt * ts;
      p.y += vy * dt * ts;
      p.z += vz * dt * ts;
      p.life += dt;

      if (useMesh) {
        const snapped = snapToSurface(p.x, p.y, p.z, this.meshSampler, this.bounds, skinEps);
        if (snapped && (p.surface || snapped.dist < span[0] * 0.2)) {
          p.x = snapped.x;
          p.y = snapped.y;
          p.z = snapped.z;
        }
      }

      // Filaments keep flowing well past the tail so the wake stays populated;
      // volumetric smoke resets sooner to avoid over-filling the view.
      const maxLife = p.surface ? 14 : isFilament ? 12 : 8;
      const passedRear = p.x > max[0] + span[0] * (isFilament ? 1.1 : 0.15);
      const tooOld = p.life > maxLife;
      const insideBody = !p.surface && isInsideBBox(p.x, p.y, p.z, this.bounds, 0.015);
      if (passedRear || tooOld || insideBody) {
        this._respawnParticle(p, i);
        const [rvx, rvy, rvz] = this._velocityAt(p.x, p.y, p.z, cfg);
        p.smoothSpd = Math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz);
      }

      const displaySpd = p.smoothSpd ?? spd;
      frameSpeeds[i] = displaySpd;
      if (displaySpd < minSpd) minSpd = displaySpd;
      if (displaySpd > maxSpd) maxSpd = displaySpd;

      if (isSmoke) {
        const h = p.hist;
        const sx = p.x - h[0];
        const sy = p.y - h[1];
        const sz = p.z - h[2];
        // A tracer can cross several arc steps in one frame; subdivide the swept
        // segment so filament length stays fixed instead of tracking dt.
        const steps = Math.min(
          SMOKE_TRAIL - 1,
          Math.floor(Math.sqrt(sx * sx + sy * sy + sz * sz) / arcStep),
        );
        const bxp = h[0];
        const byp = h[1];
        const bzp = h[2];
        for (let s = 0; s < steps; s++) {
          const f = (s + 1) / steps;
          for (let k = SMOKE_TRAIL - 1; k > 0; k--) {
            const c3 = k * 3;
            const b3 = c3 - 3;
            h[c3] = h[b3];
            h[c3 + 1] = h[b3 + 1];
            h[c3 + 2] = h[b3 + 2];
          }
          h[0] = bxp + sx * f;
          h[1] = byp + sy * f;
          h[2] = bzp + sz * f;
        }

        // Filament style: thin, bright threads like a tunnel rake photograph.
        // Volumetric style: wide, soft puffs for a smoke-curtain look.
        const base = (this.flow.particleSize || 0.18) * (isFilament ? 0.26 : 1.05) * p.sizeJitter;
        const fadeIn = Math.min(1, p.life / 0.12);
        const bodyFade = p.surface ? 0.7 : 1;
        // Real smoke puffs grow and dilute as they age; the wake fade alone
        // made young and old filaments look identical.
        const puffAge = Math.min(1, p.life / (p.surface ? 12 : 6));
        for (let k = 0; k < SMOKE_SEG; k++) {
          const a3 = k * 3;
          const b3 = a3 + 3;
          const dx = h[a3] - h[b3];
          const dy = h[a3 + 1] - h[b3 + 1];
          const dz = h[a3 + 2] - h[b3 + 2];
          const segLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
          const mx = h[a3] - dx * 0.5;
          const my = h[a3 + 1] - dy * 0.5;
          const mz = h[a3 + 2] - dz * 0.5;
          const ii = i * SMOKE_SEG + k;
          const nx = (mx - min[0]) / span[0];
          // Wake diffusion: filaments broaden, dilute and dissolve downstream.
          const spread = isFilament ? 1.1 : 1.8;
          const growthK = isFilament ? 0.05 : 0.14;
          const ageGrowth = isFilament ? 0.35 : 0.9;
          const diffuse = 1 + Math.max(0, nx - 0.55) * spread + k * growthK + puffAge * ageGrowth;
          const thick = base * Math.min(isFilament ? 1.8 : 3.2, diffuse);
          const wakeFade = nx < 0.92 ? 1 : Math.max(0, 1 - (nx - 0.92) / 1.0);
          const tailFade = 1 - 0.65 * (k / SMOKE_SEG);
          const ageFade = 1 - 0.3 * puffAge;
          // Freshly respawned tracers keep their old history collapsed on the
          // spawn point; drawing those segments stamped a bright blob at the
          // inlet. Hide them until the trail has real length.
          const dead = segLen < arcStep * 0.3;
          iPos.setXYZ(ii, mx, my, mz);
          iDir.setXYZ(ii, dx, dy, dz);
          // Cross-section quads: overlap neighbours so the streak texture forms
          // a tube instead of a row of separate stamps.
          iScale.setXY(ii, segLen * 1.08 + thick * 0.22, thick);
          iRot.setX(ii, 0);
          iUv.setXY(ii, p.uvOff[0], p.uvOff[1]);
          const gain = isFilament ? 0.5 : 0.17;
          const dilution = isFilament ? 0.45 : 0.85;
          const alpha = dead
            ? 0
            : (fadeIn * wakeFade * tailFade * bodyFade * ageFade * gain) / Math.pow(diffuse, dilution);
          iOpa.setX(ii, alpha);
        }
      } else {
        pos.setXYZ(i, p.x, p.y, p.z);
      }
    }

    // Pass 2 — colour. Neutral smoke by default; hue only when tinting is on.
    const floor = Math.min(minSpd, this.refSpeed * 0.25);
    const ceil = Math.max(maxSpd, this.refSpeed * 0.85);
    const tintBySpeed = this.flow.smokeTint === 'speed';
    const spdSpan = Math.max(1e-6, ceil - floor);
    for (let i = 0; i < n; i++) {
      if (isSmoke && !tintBySpeed) {
        const ratio = Math.max(0, Math.min(1, (frameSpeeds[i] - floor) / spdSpan));
        // Filaments are lit like a tunnel photograph: near-white, slightly
        // brighter where the flow is fast.
        const b = isFilament ? 0.9 + 0.1 * ratio : 0.70 + 0.30 * ratio;
        for (let k = 0; k < SMOKE_SEG; k++) iCol.setXYZ(i * SMOKE_SEG + k, b, b * 1.01, b * 1.05);
      } else {
        const c = speedToColor(frameSpeeds[i], this.refSpeed, floor, ceil);
        if (isSmoke) {
          for (let k = 0; k < SMOKE_SEG; k++) iCol.setXYZ(i * SMOKE_SEG + k, c.r, c.g, c.b);
        } else {
          col.setXYZ(i, c.r, c.g, c.b);
        }
      }
    }

    if (isSmoke) {
      iPos.needsUpdate = true;
      iDir.needsUpdate = true;
      iScale.needsUpdate = true;
      iRot.needsUpdate = true;
      iCol.needsUpdate = true;
      iOpa.needsUpdate = true;
      iUv.needsUpdate = true;
    } else {
      pos.needsUpdate = true;
      col.needsUpdate = true;
    }

    this.speedRange = { min: minSpd === Infinity ? 0 : minSpd, max: maxSpd };
    this.onSpeedRange?.(this.speedRange);
  }

  _updateForces(f) {
    this.forceGroup.clear();
    const cy = (this.bounds.min[1] + this.bounds.max[1]) / 2;
    const o = new THREE.Vector3(0, cy, this.bounds.max[2] + 0.5);
    const sc = 0.00035;
    this.forceGroup.add(new THREE.ArrowHelper(new THREE.Vector3(-1, 0, 0), o, f.drag_N * sc, VIEW_THEME.forceDrag, 0.25, 0.15));
    this.forceGroup.add(new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), o.clone().add(new THREE.Vector3(0, 0.5, 0)), f.downforce_total_N * sc, VIEW_THEME.forceDown, 0.25, 0.15));
  }

  setViewMode(mode) {
    this.viewMode = mode;
    this._applyViewMode();
    this.updateAero(this.config, this.simForces);
  }
  toggleStreamlines(show) {
    this.showStreamlines = show;
    this.flowGroup.visible = show;
    this.surfaceFlowGroup.visible = show;
    if (show) this._updateStreamlines(this._activeConfig());
  }
  toggleParticles(show) { this.showParticles = show; this.particleGroup.visible = show; }
  toggleForces(show) { this.showForces = show; this.forceGroup.visible = show; }
}
