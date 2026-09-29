// One idea: flash attention never stores the huge N×N attention square; it streams small tiles
// through fast on-chip memory, and only the small result is written back.
// v5 (CH8), bridge part 2, on the shared 3D H100 (scenes/_h100.ts). The camera never stops
// (orbit phase driven by the drums, scenes/_cam.ts); every change lands on the 8th grid (_lock.ts).
//  "Flash attention never writes the square,"
//    Flash      the die's SMs flare: an attention kernel starts
//    attention  the N×N score square materializes above the module as 6×6 tiles in a quick wave,
//               a wall that dwarfs the chip; the camera pulls back to show the scale
//               (N = 8192: 8192 × 8192 × 4 B = 268 MB, CH8)
//    never      the square lies down and falls toward an HBM stack, accelerating
//    writes     impact: refused in one hit. The tiles blow back up off the stack, a shock ring
//               runs across the module, "268 MB" is struck through
//    square     the tiles settle as a dim ghost grid in the air: never stored
//  "Streams it through in tiles from here to there"
//    (beat)     dive into one SM
//    Streams    the SM's L1 / shared memory lights: small, fast, on-chip
//    beats      a conveyor of tiles snaps one slot per beat along the shared memory, west to east;
//               each tile flares at the compute spot (the partition lanes fire) and is dropped;
//               the small result bar grows one notch per tile
//    (after)    only the result leaves: a pulse along the path back to HBM, the camera pulls out with it
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font, glyphX } from '../engine/type';
import { Lyrics, type Line, type Word } from '../engine/lyrics';
import { clamp, ease, hash, lerp } from '../engine/util';
import { q, snap, hit } from './_lock';
import { drive, kickPush } from './_cam';
import { H100, glowMat, type PulseItem } from './_h100';

// ---------------------------------------------------------------- layout
const LYR = { x: 120, y: 150, size: 66 };      // the karaoke line, fixed top-left
const N_T = 6;                                 // the square as 6 × 6 tiles
const SQ = { c: new THREE.Vector3(0, 128, -46), size: 228 }; // the standing square (mm): dwarfs the 28.5 mm die
const TS = SQ.size / N_T;
const HBM_K = 0;                               // the stack the square tries to fall into (west, north)
const CELLS = 16;                              // score cells per tile edge in the texture
const STREAM = 6;                              // tiles streamed in line 2

const byWord = (l: Line, s: string): Word => l.words.find((w) => w.w.toLowerCase().startsWith(s)) ?? l.words[0]!;

/** Attention-like scores: most small, a few hot, a soft diagonal band (tokens attend near themselves). */
function scoreTexture(): THREE.CanvasTexture {
  const n = N_T * CELLS, px = 16, cv = document.createElement('canvas');
  cv.width = cv.height = n * px;
  const c = cv.getContext('2d')!;
  c.fillStyle = '#0d0d0e';
  c.fillRect(0, 0, cv.width, cv.height);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const band = Math.exp(-Math.pow((i - j) / 9, 2));
    const v = Math.min(1, Math.pow(hash(i, j, 31), 5) * 1.1 + band * 0.45 * hash(i, j, 7) + 0.05 * hash(i, j, 3));
    const r = Math.round(20 + 235 * v), g = Math.round(14 + 64 * v * v), b = Math.round(12 + 6 * v);
    c.fillStyle = `rgb(${r},${g},${b})`;
    c.fillRect(j * px + 1, i * px + 1, px - 2, px - 2);
  }
  // tile borders (bone hairlines, so a flying shard still reads as a tile)
  c.strokeStyle = 'rgba(238,233,223,0.85)';
  c.lineWidth = 6;
  const tpx = CELLS * px;
  for (let k = 0; k <= N_T; k++) {
    c.beginPath(); c.moveTo(k * tpx, 0); c.lineTo(k * tpx, cv.height); c.stroke();
    c.beginPath(); c.moveTo(0, k * tpx); c.lineTo(cv.width, k * tpx); c.stroke();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/** Plane geometry mapped to tile (i, j) of the score texture (row i from the top). */
function tileGeo(i: number, j: number, size: number): THREE.PlaneGeometry {
  const g = new THREE.PlaneGeometry(size, size);
  const uv = g.attributes.uv as THREE.BufferAttribute;
  const u0 = j / N_T, v0 = 1 - (i + 1) / N_T;
  for (let k = 0; k < uv.count; k++) uv.setXY(k, u0 + uv.getX(k) / N_T, v0 + uv.getY(k) / N_T);
  return g;
}

type Times = {
  l1: Line; l2: Line;
  flash: number; att: number; never: number; writes: number; square: number;
  dive0: number; dive1: number; streams: number; tiles: number[]; out: number; end: number;
};

export default class Flash extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(34, W / H, 0.05, 5000);
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();
  gpu!: H100;
  sm = 58;
  l1World = new THREE.Vector3();
  smAim = new THREE.Vector3();                     // line 2 camera target: between the conveyor spot and the result
  slot = 0.2;
  drums = 0;
  T!: Times;
  tiles: THREE.Mesh[] = [];
  tileMat: THREE.MeshBasicMaterial[] = [];
  ring!: THREE.Mesh;
  // line 2: tiles on the SM's shared memory, and the result bar
  sTiles: THREE.Mesh[] = [];
  sMat: THREE.MeshBasicMaterial[] = [];
  resBar!: THREE.Mesh;
  resGlow!: THREE.Mesh;
  famL = F.archivo(100, 800);
  famM = F.mono(500);
  famB = F.archivo(100, 900);

  override async init() {
    const ly = this.ctx.lyrics, au = this.ctx.audio;
    const l1 = ly.get('Flash attention'), l2 = ly.get('Streams it through');
    const beatAfter = (t: number, n = 1) => au.timeOfBeat(Math.floor(au.beatAt(t) + 1e-6) + n);
    const square = q(au, byWord(l1, 'square').start);
    const streams = q(au, byWord(l2, 'streams').start);
    const t0 = beatAfter(streams);                   // first tile on the beat after "Streams"
    const tiles = Array.from({ length: STREAM }, (_, k) => au.timeOfBeat(Math.round(au.beatAt(t0)) + k));
    this.T = {
      l1, l2,
      flash: q(au, byWord(l1, 'flash').start),
      att: q(au, byWord(l1, 'attention').start),
      never: q(au, byWord(l1, 'never').start),
      writes: q(au, byWord(l1, 'writes').start),
      square,
      dive0: beatAfter(square),
      dive1: streams,
      streams,
      tiles,
      out: beatAfter(tiles[tiles.length - 1]!),
      end: this.ctx.end,
    };

    // the GPU
    const g = this.gpu = new H100({ renderer: this.ctx.renderer });
    this.scene.add(g.group, H100.rig());
    this.scene.environment = g.env;
    this.scene.background = new THREE.Color().setRGB(LIN.ink[0], LIN.ink[1], LIN.ink[2], THREE.LinearSRGBColorSpace);
    if (!g.smEnabled(this.sm)) for (let i = 50; i < 144; i++) if (g.smEnabled(i)) { this.sm = i; break; }

    // the square: 36 tiles, one texture
    const tex = scoreTexture();
    for (let i = 0; i < N_T; i++) for (let j = 0; j < N_T; j++) {
      const m = new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false });
      const mesh = new THREE.Mesh(tileGeo(i, j, TS * 0.985), m);
      mesh.matrixAutoUpdate = false;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      this.tiles.push(mesh);
      this.tileMat.push(m);
    }
    // the shock ring at the HBM stack
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.92, 1, 96).rotateX(-Math.PI / 2), glowMat());
    this.ring.visible = false;
    this.scene.add(this.ring);

    // line 2 props on the SM (sized in SM-local units once the close-up exists)
    const d = g.smDetail(this.sm);
    g.group.updateMatrixWorld(true);
    this.l1World = d.group.localToWorld(d.anchor('l1'));
    {
      const b = d.boxes['l1']!, m = d.boxes['tma']!;
      this.smAim = d.group.localToWorld(new THREE.Vector3(lerp(b.x0, m.x1, 0.58), d.anchor('l1').y + 0.08, (b.z0 + b.z1) / 2 - 0.12));
    }
    g.hideSMDetail();
    const l1b = d.boxes['l1']!;
    // standing tiles (cards on a conveyor); the group is scaled (sx, 1, sz), so undo sx on height
    const ts = (l1b.z1 - l1b.z0) * 1.7, sx = d.group.scale.x;
    this.slot = ts * 1.18;
    for (let k = 0; k < STREAM; k++) {
      const m = new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false });
      const mesh = new THREE.Mesh(tileGeo(k % N_T, (k * 5 + 2) % N_T, ts).scale(1, sx, 1).translate(0, ts * sx / 2, 0), m);
      mesh.visible = false;
      d.group.add(mesh);
      this.sTiles.push(mesh);
      this.sMat.push(m);
    }
    const tma = d.boxes['tma']!;
    this.resBar = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(LIN.signal[0], LIN.signal[1], LIN.signal[2], THREE.LinearSRGBColorSpace) }));
    this.resGlow = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), glowMat());
    this.resBar.visible = this.resGlow.visible = false;
    this.resBar.userData = { x0: tma.x0, x1: tma.x1, z: (tma.z0 + tma.z1) / 2, y: d.anchor('tma').y + 0.03, dz: (tma.z1 - tma.z0) * 0.5 };
    d.group.add(this.resBar, this.resGlow);
  }

  // ---------------------------------------------------------------- the square
  /** Home (standing wall) position of tile (i, j). */
  home(i: number, j: number) {
    return new THREE.Vector3(SQ.c.x + (j - (N_T - 1) / 2) * TS, SQ.c.y + ((N_T - 1) / 2 - i) * TS, SQ.c.z);
  }

  /** Tile matrices over line 1: wave in, stand, lie down and fall onto the stack, blow back up as a ghost. */
  updateSquare(t: number): { alpha: number } {
    const au = this.ctx.audio, T = this.T, g = this.gpu;
    const hb = g.hbm(HBM_K).pos.clone().add(new THREE.Vector3(0, 0.6, 0));
    const fall = t < T.never ? 0 : ease.inCubic(clamp((t - T.never) / (T.writes - T.never)));
    const back = t < T.writes ? 0 : ease.outExpo(clamp((t - T.writes) / (T.square - T.writes)));
    const ghost = snap(au, t, T.square, 0.09);
    const fadeDive = clamp((t - T.dive0) / (T.dive1 - T.dive0) * 3);
    const m = new THREE.Matrix4(), qn = new THREE.Quaternion(), sc = new THREE.Vector3();
    const lie = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
    for (let i = 0; i < N_T; i++) for (let j = 0; j < N_T; j++) {
      const n = i * N_T + j, mesh = this.tiles[n]!, mat = this.tileMat[n]!;
      // wave in on "attention": 16th-note stagger by anti-diagonal, each tile a 60 ms pop
      const tIn = T.att + ((i + j) / (2 * N_T - 2)) * 0.34;
      const pop = t < tIn ? 0 : ease.outBack(clamp((t - tIn) / 0.06));
      const h = this.home(i, j);
      let pos: THREE.Vector3, scale: number, quat: THREE.Quaternion;
      if (t < T.writes) {
        // lie down and fall into the stack: the whole square scales toward the stack's footprint
        const off = new THREE.Vector3((j - (N_T - 1) / 2) * TS, ((N_T - 1) / 2 - i) * TS, 0);
        const s = lerp(1, 0.042, fall);
        const qq = new THREE.Quaternion().slerp(lie, clamp(fall * 1.4));
        off.multiplyScalar(s).applyQuaternion(qq);
        pos = SQ.c.clone().lerp(hb, fall).add(off);
        scale = s * pop;
        quat = qq;
      } else {
        // refused: blown back up off the stack, spinning, to a spread ghost grid in the air
        const gh = h.clone().sub(SQ.c).multiplyScalar(1.22).add(SQ.c).add(new THREE.Vector3(0, 8, (hash(i, j, 5) - 0.5) * 30));
        const at = hb.clone().add(new THREE.Vector3((j - (N_T - 1) / 2) * TS * 0.042, 0, (i - (N_T - 1) / 2) * TS * 0.042));
        pos = at.lerp(gh, back);
        // after settling, a slow float (continuous life, small)
        pos.y += Math.sin(t * 1.3 + n) * 1.5 * back;
        const spin = (1 - back) * Math.PI * (1.5 + hash(i, j, 9));
        const ax = new THREE.Vector3(hash(i, j, 1) - 0.5, hash(i, j, 2) - 0.5, hash(i, j, 4) - 0.5).normalize();
        quat = new THREE.Quaternion().setFromAxisAngle(ax, spin).multiply(new THREE.Quaternion().slerp(lie, 1 - back));
        scale = lerp(0.042, 0.94, back);
      }
      m.compose(pos, quat, sc.setScalar(Math.max(1e-4, scale)));
      mesh.matrix.copy(m);
      mesh.visible = pop > 0 && fadeDive < 1;
      // brightness: hot while it stands / falls, dim ghost after "square"
      const flare = hit(au, t, T.writes, 0.12);
      const a = (1 - 0.72 * ghost) * (1 - fadeDive);
      mat.opacity = a;
      // a scan runs down the standing square, a row per 16th, brighter on the drums (it is being computed)
      const scan = Math.pow(Math.max(0, Math.cos(((t - T.att) * 4.4 - i / N_T) * Math.PI * 2)), 6) * (1 - fall) * (1 - back);
      const k = 1 + 1.6 * flare + 0.6 * hit(au, t, T.att, 0.2) + (0.35 + 0.9 * this.drums) * scan;
      mat.color.setScalar(k);
      qn.identity();
    }
    // the shock ring
    const rt = t - T.writes;
    this.ring.visible = rt >= 0 && rt < 0.5;
    if (this.ring.visible) {
      const r = 3 + 120 * ease.outCubic(clamp(rt / 0.5));
      this.ring.position.copy(hb);
      this.ring.scale.set(r, 1, r);
      const v = 3.2 * (1 - clamp(rt / 0.5));
      (this.ring.material as THREE.MeshBasicMaterial).color.setRGB(LIN.signal[0] * v, LIN.signal[1] * v, LIN.signal[2] * v);
    }
    return { alpha: 1 - fadeDive };
  }

  // ---------------------------------------------------------------- line 2 on the SM
  updateStream(t: number) {
    const au = this.ctx.audio, T = this.T, g = this.gpu;
    const on = t >= T.dive0 + (T.dive1 - T.dive0) * 0.4;
    if (!on) { g.hideSMDetail(); return; }
    const d = g.smDetail(this.sm);
    const l1b = d.boxes['l1']!;
    const y = d.anchor('l1').y + 0.004;
    const zc = (l1b.z0 + l1b.z1) / 2;
    const slot = this.slot;                          // conveyor spacing
    const spot = lerp(l1b.x0, l1b.x1, 0.62);        // the compute spot on the shared memory
    // conveyor position: how many beats have ticked since the first tile (snapped per beat)
    let step = 0;
    for (const tt of T.tiles) step += snap(au, t, tt, 0.09, 1);
    step += snap(au, t, T.out, 0.09, 1);            // the last tile is dropped as the result leaves
    const cur = Math.round(step) - 1;               // index of the tile at the spot (after its snap)
    for (let k = 0; k < STREAM; k++) {
      const mesh = this.sTiles[k]!, mat = this.sMat[k]!;
      // tile k sits k slots west of the spot at step 0, moves one slot east per beat
      const x = spot - (k + 1) * slot + step * slot;
      const done = x > spot + slot * 0.5;            // passed the spot: dropped
      const fromSpot = (x - spot) / slot;            // 0 at the spot
      const west = l1b.x0 - slot * 2.2;
      mesh.visible = !done && x > west - slot;
      // past the spot it is dropped: it sinks into the bar and shrinks away
      const drop = clamp(fromSpot / 0.5);
      mesh.position.set(Math.min(x, spot + slot * 0.25), y - drop * 0.05, zc);
      const fl = hit(au, t, T.tiles[k]!, 0.14);
      mesh.scale.setScalar(Math.max(0.02, (1 - drop) * (1 + 0.12 * fl)));
      mat.color.setScalar(0.5 + (x > l1b.x0 ? 0.5 : 0) + 2.4 * fl);
      mat.opacity = x < l1b.x0 ? clamp(1 - (l1b.x0 - x) / (slot * 2.2)) : 1;
    }
    // the partition lanes fire with each tile, the shared memory glows while tiles stream
    const lastHit = T.tiles.reduce((m, tt) => Math.max(m, hit(au, t, tt, 0.16)), 0);
    d.setLanes((p, kind, idx) => 0.1 + 0.9 * lastHit * (hash(p, idx, cur + 3) < 0.7 ? 1 : 0.2));
    d.setUnits((u) => (u === 'l1' ? snap(au, t, T.streams) * (0.18 + 0.5 * lastHit) : u === 'regfile' ? 0.1 + 0.35 * lastHit : 0));
    d.setTensor((p, i, j, k) => lastHit * (hash(p, i, j, k + cur) < 0.4 ? 1 : 0.1));
    // result bar: one notch per tile, on the TMA block
    const u = this.resBar.userData as { x0: number; x1: number; z: number; y: number; dz: number };
    const notches = clamp(step, 0, STREAM) / STREAM;
    const leave = t < T.out ? 0 : ease.inCubic(clamp((t - T.out) / 0.3));
    this.resBar.visible = this.resGlow.visible = notches > 0.001 && leave < 1;
    const len = (u.x1 - u.x0) * notches;
    this.resBar.scale.set(Math.max(1e-3, len), 0.012, u.dz * 0.5);
    this.resBar.position.set(u.x0 + len / 2, u.y + leave * 0.4, u.z);
    this.resGlow.scale.set(Math.max(1e-3, len) * 1.06, 0.02, u.dz * 0.7);
    this.resGlow.position.copy(this.resBar.position);
    const gv = 0.6 + 1.8 * lastHit;
    (this.resGlow.material as THREE.MeshBasicMaterial).color.setRGB(LIN.ember[0] * gv, LIN.ember[1] * gv, LIN.ember[2] * gv);
  }

  // ---------------------------------------------------------------- camera
  shot(t: number): { pos: THREE.Vector3; tgt: THREE.Vector3; dist: number } {
    const au = this.ctx.audio, T = this.T, g = this.gpu;
    const orbitAz = (t0: number) => 0.1 * drive(au, t, 1, 2.5, t0);
    const sph = (tgt: THREE.Vector3, dist: number, az: number, el: number) =>
      new THREE.Vector3(tgt.x + Math.sin(az) * Math.cos(el) * dist, tgt.y + Math.sin(el) * dist, tgt.z + Math.cos(az) * Math.cos(el) * dist);
    const hb = g.hbm(HBM_K).pos;
    const smp = g.smPos(this.sm);
    const az0 = -0.55 + orbitAz(this.ctx.start);
    // line 1: wide on the module → pulled back to show the wall → following the fall → the ghost
    const pA = ease.inOutCubic(clamp((t - T.att) / (T.never - T.att)));
    const pB = ease.inOutCubic(clamp((t - T.never) / (T.writes - T.never)));
    const tgtA = new THREE.Vector3(0, 2, 0).lerp(new THREE.Vector3(95, 132, -24), pA).lerp(hb.clone().multiplyScalar(0.45).add(new THREE.Vector3(55, 48, 0)), pB * 0.8);
    let dist = lerp(lerp(105, 600, pA), 380, pB);
    const pC = ease.outCubic(clamp((t - T.writes) / (T.dive0 - T.writes)));
    tgtA.lerp(new THREE.Vector3(80, 96, -20), pC);
    dist = lerp(dist, 520, pC) - 28 * kickPush(au, t) - 60 * hit(au, t, T.writes, 0.18);
    let el = lerp(lerp(0.52, 0.26, pA), 0.4, pB);
    let az = az0;
    // the dive: log-distance zoom onto the SM
    const pD = ease.inOutCubic(clamp((t - T.dive0) / (T.dive1 - T.dive0)));
    const tgtB = this.smAim.clone();
    const dIn = 1.55, d0 = dist;
    dist = Math.exp(lerp(Math.log(d0), Math.log(dIn), pD));
    // the aim converges in proportion to the distance covered, so the SM stays centred on the way in
    let tgt = tgtA.clone().lerp(tgtB, clamp((d0 - dist) / (d0 - dIn)));
    el = lerp(el, 0.62, pD);
    az = az0 + pD * 0.75;
    // line 2: close orbit along the shared memory, a small push on each tile
    if (t >= T.dive1) {
      const tp = T.tiles.reduce((m, tt) => Math.max(m, hit(au, t, tt, 0.16)), 0);
      dist = dIn - 0.1 * tp - 0.04 * kickPush(au, t);
      el = 0.62 - 0.12 * clamp((t - T.dive1) / 3);
    }
    // the exit: the result leaves for HBM, the camera pulls out with it
    const pE = ease.outCubic(clamp((t - T.out) / 0.55));
    if (pE > 0) {
      dist = Math.exp(lerp(Math.log(dist), Math.log(48), pE));
      tgt = tgt.lerp(smp.clone().lerp(hb, 0.35), pE * 0.8);
      el = lerp(el, 0.72, pE);
    }
    return { pos: sph(tgt, dist, az, el), tgt, dist };
  }

  // ---------------------------------------------------------------- frame
  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp, audio: au } = this.ctx;
    const T = this.T, t = f.t, g = this.gpu;
    // SMs: flare on "Flash", then the default music-driven activity; the stack flares on the refusal
    const fl = hit(au, t, T.flash, 0.25);
    const inSM = t >= T.dive0 + (T.dive1 - T.dive0) * 0.4;
    g.setSMActivity(inSM ? (i) => (i === this.sm ? 0 : 0.02)
      : fl > 0.02 ? (i) => (g.smEnabled(i) ? 0.25 + 1.4 * fl * (hash(i, 3) < 0.8 ? 1 : 0.3) : 0) : null);
    const hv = hit(au, t, T.writes, 0.2);
    g.setHBMActivity(hv > 0.02 ? (k) => (k === HBM_K ? 0.5 + 3 * hv : null as unknown as number) : null);
    g.update(t, f.a);
    this.drums = f.a.drums;
    this.updateSquare(t);
    // data pulses on the module during line 1 (continuous life), off once inside the SM
    const items: PulseItem[] = [];
    const pg = 1.6 * (1 - clamp((t - T.dive0) / 0.5));
    if (pg > 0) {
      for (let k = 0; k < 6; k++) {
        if (!g.hbm(k).active) continue;
        const path = g.pulsePath(k, 20 + k * 17);
        for (let n = 0; n < 4; n++) items.push({ curve: path, u: ((t * 0.45 + n / 4 + k * 0.17) % 1), len: 0.04, gain: pg, width: 0.22 });
      }
    }
    // the result leaving: one bright pulse from the SM back toward the stack
    if (t >= T.out) {
      const path = g.pulsePath(HBM_K, this.sm);
      const u = 1 - clamp((t - T.out) / 0.9);
      items.push({ curve: path, u, len: 0.03, gain: 4, width: 0.05, color: LIN.signal });
    }
    g.pulses.set(items);
    this.updateStream(t);

    const s = this.shot(t);
    this.cam.position.copy(s.pos);
    this.cam.lookAt(s.tgt);
    H100.fitClip(this.cam, s.dist);

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, this.cam);
    comp.draw(renderer, this.ms.texture, out);

    this.ui.clear();
    const c = this.ui.ctx;
    const scr = c.createLinearGradient(0, 0, 0, 260);
    scr.addColorStop(0, rgba('ink', 0.7));
    scr.addColorStop(1, rgba('ink', 0));
    c.fillStyle = scr;
    c.fillRect(0, 0, W, 260);
    this.drawLabels(c, t);
    this.drawLyrics(c, t);
    comp.draw(renderer, this.ui.upload(), out);

    const imp = hit(au, t, T.writes, 0.12);
    return {
      bloom: 0.6 + 0.5 * imp, bloomThreshold: 0.82, grain: 0.05, vignette: 0.42,
      ca: 1.1 + 2.5 * imp, exposure: 1 + 0.35 * imp,
    };
  }

  // ---------------------------------------------------------------- overlays
  toScreen(p: THREE.Vector3): { x: number; y: number; ok: boolean } {
    const v = p.clone().project(this.cam);
    return { x: (v.x * 0.5 + 0.5) * W, y: (-v.y * 0.5 + 0.5) * H, ok: v.z < 1 && v.z > -1 };
  }

  drawLabels(c: CanvasRenderingContext2D, t: number) {
    const au = this.ctx.audio, T = this.T, g = this.gpu;
    c.textBaseline = 'alphabetic';
    // line 1: what the square is and what storing it would cost (fixed block, right)
    const kA = snap(au, t, T.att + 0.23) * (1 - clamp((t - T.dive0) / 0.25));
    if (kA > 0) {
      const x = 1250, y = 380;
      c.font = font(this.famM, 34);
      c.fillStyle = rgba('bone', 0.9 * kA);
      c.fillText('N × N attention scores', x, y);
      c.font = font(this.famB, 64);
      c.fillStyle = rgba('bone', 0.95 * kA);
      c.fillText('8192 × 8192 × 4 B', x, y + 84);
      const k268 = snap(au, t, T.writes);
      c.font = font(this.famB, 104);
      c.fillStyle = k268 > 0 ? rgba('ash', 0.9 * kA) : rgba('signal', kA);
      c.fillText('= 268 MB', x, y + 196);
      c.font = font(this.famM, 24);
      c.fillStyle = rgba('ash', 0.85 * kA);
      c.fillText('N = 8192 · CH8', x, y + 240);
      if (k268 > 0) {
        // struck through in one hit, then the verdict
        c.fillStyle = rgba('signal', kA);
        c.fillRect(x - 12, y + 158, 480 * k268, 14);
        c.font = font(this.famB, 64);
        c.fillStyle = rgba('signal', kA * snap(au, t, T.writes + 0.05));
        c.fillText('NEVER STORED', x, y + 330);
      }
    }
    // the stack it tries to fall into
    const kh = snap(au, t, T.never) * (1 - snap(au, t, T.square));
    if (kh > 0) {
      const p = this.toScreen(g.hbm(HBM_K).pos);
      if (p.ok) {
        c.fillStyle = rgba('signal', kh);
        c.fillRect(p.x - 6, p.y - 6, 12, 12);
        c.font = font(this.famM, 32);
        c.fillStyle = rgba('bone', 0.95 * kh);
        c.textAlign = 'right';
        c.fillText('GPU memory (HBM)', p.x - 20, p.y + 10);
        c.textAlign = 'left';
      }
    }
    // line 2: the shared memory and the tile count (fixed, bottom-left)
    const kB = snap(au, t, T.streams);
    if (kB > 0 && t < T.out) {
      const d = g.smDetail(this.sm), b = d.boxes['l1']!;
      const L1 = this.toScreen(d.group.localToWorld(new THREE.Vector3(lerp(b.x0, b.x1, 0.2), d.anchor('l1').y, b.z1)));
      const lx = 120, ly = 800;
      c.font = font(this.famM, 36);
      c.fillStyle = rgba('bone', 0.95 * kB);
      c.fillText('on-chip shared memory', lx, ly);
      c.font = font(this.famM, 28);
      c.fillStyle = rgba('ash', 0.95 * kB);
      c.fillText('small, fast', lx, ly + 40);
      if (L1.ok) {
        c.strokeStyle = rgba('bone', 0.7 * kB);
        c.lineWidth = 2;
        c.beginPath(); c.moveTo(lx + 300, ly - 44); c.lineTo(L1.x, L1.y); c.stroke();
        c.fillStyle = rgba('bone', kB);
        c.fillRect(L1.x - 5, L1.y - 5, 10, 10);
      }
      let n = 0;
      for (const tt of T.tiles) if (t >= tt) n++;
      if (n > 0) {
        c.font = font(this.famB, 72);
        c.fillStyle = rgba('bone', 0.95 * kB);
        c.fillText(`tile ${n} of ${STREAM}`, 120, 950);
        c.font = font(this.famM, 28);
        c.fillStyle = rgba('ash', 0.95 * kB);
        c.fillText('computed on-chip, then dropped', 120, 996);
      }
    }
    if (t >= T.out) {
      const k = snap(au, t, T.out);
      c.font = font(this.famB, 56);
      c.fillStyle = rgba('signal', k);
      c.fillText('only the small result', 120, 950);
      c.font = font(this.famM, 30);
      c.fillStyle = rgba('bone', 0.95 * k);
      c.fillText('goes back to GPU memory', 120, 996);
      const u = this.resBar.userData as { x0: number; x1: number; z: number; y: number };
      const d = g.smDetail(this.sm);
      const r = this.toScreen(d.group.localToWorld(new THREE.Vector3(u.x0, u.y, u.z)));
      if (r.ok && k > 0) {
        c.strokeStyle = rgba('signal', 0.8 * k);
        c.lineWidth = 2;
        c.beginPath(); c.moveTo(640, 930); c.lineTo(r.x, r.y); c.stroke();
      }
    }
  }

  drawLyrics(c: CanvasRenderingContext2D, t: number) {
    const T = this.T;
    const swap = T.l2.words[0]!.start - 0.4;
    const line = t < swap ? T.l1 : T.l2;
    const fam = this.famL, size = LYR.size;
    c.font = font(fam, size);
    c.textBaseline = 'alphabetic';
    let from = 0;
    for (const w of line.words) {
      const i = line.text.indexOf(w.w, from);
      if (i < 0) continue;
      from = i + w.w.length;
      const x = LYR.x + glyphX(line.text, i, fam, size);
      const p = Lyrics.wordProgress(w, t);
      c.fillStyle = p <= 0 ? rgba('bone', 0.3) : p < 1 ? rgba('signal', 1) : rgba('bone', 0.96);
      c.fillText(w.w, x, LYR.y);
    }
  }
}
