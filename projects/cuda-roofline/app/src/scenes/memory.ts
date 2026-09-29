// Idea: read neighbouring data together, and keep the data you reuse close, in shared memory.
// `memory` (v5, CH6), 91.61–108.30 s, on the shared H100 model (scenes/_h100.ts). The camera never
// stops (drum-driven orbit drift, kick dolly), big moves land on downbeats, accents snap on the grid.
//  1. "Neighbors read their neighbors side by side, / One trip to memory, nice and wide,": a warp of 32
//     threads floats over an SM, its 32 addresses float over the nearest HBM stack, thread i directly
//     across from cell i. The 32 parallel reads close up into one 128-byte bar; a request streaks to the
//     stack, the one wide bar rides the real data path back (HBM → PHY → L2 → crossbar → SM). Contrast:
//     scattered addresses fan out to every stack, "up to 32 trips" (CH06).
//  2. "Load a tile to shared and keep it near, / Sync the block and reuse it here": a 4×4 tile lifts off
//     the far (west) stack and the camera chases it across the die into the SM close-up, where it settles
//     in the 256 KB L1/shared memory (~30 cycles vs hundreds, CH06). A barrier sweeps the SM; then the
//     lanes read the same tile again and again (beams from the shared-memory block, a read counter).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import type { Line, Word } from '../engine/lyrics';
import { LIN, rgba } from '../engine/palette';
import { F, font, layout } from '../engine/type';
import { clamp, ease, hash, lerp } from '../engine/util';
import { H100, H100_Y, DIM, glowMat, type PulseItem, type SMDetail } from './_h100';
import { hit, q, snap } from './_lock';
import { beatEase, drive, kickPush } from './_cam';

type C2 = CanvasRenderingContext2D;
type V3 = THREE.Vector3;
const v3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const col = (k: keyof typeof LIN, g = 1): [number, number, number] => [LIN[k][0] * g, LIN[k][1] * g, LIN[k][2] * g];

// lyric (screen space, fixed top-left)
const LX = 110, LY = 150, LSIZE = 66;
const LFAM = F.archivo(100, 800);

const HBM_NEAR = 4;   // east middle stack: the addresses of movement 1
const HBM_FAR = 1;    // west middle stack: the tile's source in movement 2
const N = 32;         // one warp
const PITCH = 0.18;   // mm between neighbours in both rows
const ROW_Y = 2.3;    // warp row height over the die top (mm)
const CELL_Y = 1.3;   // address row height over the stack top (mm)

/** A curve walked backwards (SM → HBM for requests). */
class Rev extends THREE.Curve<THREE.Vector3> {
  constructor(public c: THREE.Curve<THREE.Vector3>) { super(); }
  override getPoint(t: number, o = new THREE.Vector3()) { return o.copy(this.c.getPoint(1 - t)); }
}

/** A curve bowed sideways by `o` in the middle (same ends): one of 32 separate trips. */
class Bow extends THREE.Curve<THREE.Vector3> {
  constructor(public c: THREE.Curve<THREE.Vector3>, public o: THREE.Vector3) { super(); }
  override getPoint(t: number, out = new THREE.Vector3()) { return out.copy(this.c.getPoint(t)).addScaledVector(this.o, Math.sin(Math.PI * t)); }
}

interface Shot { pos: V3; tgt: V3; fov: number; dist: number }
/** A label: group-space anchor, text, alpha; `dy` shifts it on screen (px). */
interface Lab { p: V3; s: string; a: number; hot?: boolean; big?: boolean; dy?: number }

export default class Memory extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(34, W / H, 0.05, 5000);
  gpu!: H100;
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();
  famM = F.mono(500);
  famMB = F.mono(600);
  lines!: Line[];
  w!: Word[][];

  sm = 0;
  smC = v3();
  thr: V3[] = []; cell: V3[] = [];
  beams: THREE.Curve<V3>[] = [];
  dataPath!: THREE.CurvePath<V3>;
  reqPath!: Rev;
  ghostPaths: Bow[] = [];
  tilePath!: THREE.CurvePath<V3>;
  waypoints: { p: V3; s: string }[] = [];
  detail!: SMDetail;

  // meshes (group space)
  thrCore!: THREE.InstancedMesh; thrGlow!: THREE.InstancedMesh;
  cellBody!: THREE.InstancedMesh; cellGlow!: THREE.InstancedMesh;
  bar!: THREE.Mesh;
  tileBody!: THREE.InstancedMesh; tileGlow!: THREE.InstancedMesh;
  barrier!: THREE.Mesh;
  tmp = new THREE.Object3D();
  c3 = new THREE.Color();

  override async init() {
    const ly = this.ctx.lyrics;
    this.lines = [ly.get('Neighbors read'), ly.get('One trip to memory'), ly.get('Load a tile'), ly.get('Sync the block')];
    this.w = this.lines.map((l) => l.words);

    const g = this.gpu = new H100({ renderer: this.ctx.renderer });
    this.scene.add(g.group, H100.rig());
    this.scene.environment = g.env;
    this.scene.background = new THREE.Color().setRGB(LIN.ink[0], LIN.ink[1], LIN.ink[2], THREE.LinearSRGBColorSpace);

    // the SM: enabled, east half, level with the near stack
    const hz = g.hbm(HBM_NEAR).pos.z;
    let best = 1e9;
    for (let i = 0; i < g.smCount; i++) {
      if (!g.smEnabled(i)) continue;
      const p = g.smPos(i);
      if (p.x < 5) continue;
      const s = Math.abs(p.z - hz) + Math.abs(p.x - (DIM.die.w / 2 - 4)) * 0.6;
      if (s < best) { best = s; this.sm = i; }
    }
    this.smC = g.smPos(this.sm);
    const hN = g.hbm(HBM_NEAR).pos;
    for (let i = 0; i < N; i++) {
      const z = this.smC.z + (i - (N - 1) / 2) * PITCH;
      this.thr.push(v3(this.smC.x, H100_Y.dieTop + ROW_Y, z));
      this.cell.push(v3(hN.x, hN.y + CELL_Y, z));
      this.beams.push(new THREE.LineCurve3(this.thr[i]!.clone(), this.cell[i]!.clone()));
    }
    this.dataPath = g.pulsePath(HBM_NEAR, this.sm);
    this.reqPath = new Rev(this.dataPath);
    const stacks: Rev[] = [];
    for (let k = 0; k < 6; k++) if (g.hbm(k).active) stacks.push(new Rev(g.pulsePath(k, this.sm)));
    for (let i = 0; i < N; i++) {
      const o = v3((hash(i, 11) - 0.5) * 3, 0.15 + hash(i, 12) * 1.2, (hash(i, 13) - 0.5) * 3);
      this.ghostPaths.push(new Bow(stacks[Math.floor(hash(i, 7) * stacks.length)]!, o));
    }
    this.tilePath = g.pulsePath(HBM_FAR, this.sm);
    // the tile's route across the die (the west L2 partition serves the west stacks; see pulsePath)
    const l2 = g.l2Boxes[0]!.getCenter(v3()), hF = g.hbm(HBM_FAR).pos;
    this.waypoints = [
      { p: v3(-(DIM.die.w / 2 - 1.2), H100_Y.dieTop, hF.z), s: 'PHY' },
      { p: v3(l2.x, H100_Y.dieTop, l2.z), s: 'L2 · 50 MB' },
      { p: v3(0, H100_Y.dieTop, (l2.z + this.smC.z) / 2), s: 'crossbar' },
    ];

    // threads: bone cores + signal glow halos
    const sph = new THREE.SphereGeometry(1, 16, 12);
    this.thrCore = new THREE.InstancedMesh(sph, new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(...col('bone', 1.2)) }), N);
    this.thrGlow = new THREE.InstancedMesh(sph, glowMat(), N);
    // address cells: dark bodies + glow twins
    const box = new THREE.BoxGeometry(1, 1, 1);
    this.cellBody = new THREE.InstancedMesh(box, new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(...col('graphite', 0.5)), metalness: 0.4, roughness: 0.45, envMapIntensity: 0.3 }), N);
    this.cellGlow = new THREE.InstancedMesh(box, glowMat(), N);
    this.bar = new THREE.Mesh(box, glowMat());
    // the tile: 4×4 cubes
    this.tileBody = new THREE.InstancedMesh(box, new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(...col('ash', 0.5)), metalness: 0.5, roughness: 0.35, envMapIntensity: 0.3 }), 16);
    this.tileGlow = new THREE.InstancedMesh(box, glowMat(), 16);
    // the barrier: a thin glowing wall across the SM
    this.barrier = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), glowMat(0x000000, { side: THREE.DoubleSide }));
    for (const m of [this.thrCore, this.thrGlow, this.cellBody, this.cellGlow, this.tileBody, this.tileGlow]) {
      m.frustumCulled = false;
      m.setColorAt(0, this.c3.setRGB(0, 0, 0));
    }
    this.bar.frustumCulled = this.barrier.frustumCulled = false;
    g.group.add(this.thrCore, this.thrGlow, this.cellBody, this.cellGlow, this.bar, this.tileBody, this.tileGlow, this.barrier);
    this.detail = g.smDetail(this.sm);
    g.hideSMDetail();
  }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp, audio: au } = this.ctx;
    const t = f.t;
    const g = this.gpu;
    const [w20, w21, w22, w23] = this.w as [Word[], Word[], Word[], Word[]];
    const tLoad = q(au, w22[0]!.start);
    const m2 = t >= tLoad;

    g.explode(0);
    const items: PulseItem[] = [];
    const lab: Lab[] = [];

    if (!m2) this.coalesce(t, w20, w21, items, lab);
    else this.tile(t, w22, w23, items, lab);
    g.update(t, f.a); // after the scene set its SM/HBM activity, so this frame uses it
    g.pulses.set(items);

    // camera
    const shot = this.shot(t);
    const cam = this.cam;
    cam.position.copy(shot.pos);
    cam.up.set(0, 1, 0);
    cam.lookAt(shot.tgt);
    if (cam.fov !== shot.fov) { cam.fov = shot.fov; cam.updateProjectionMatrix(); }
    H100.fitClip(cam, shot.dist);
    g.group.updateMatrixWorld(true);

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    comp.draw(renderer, this.ms.texture, out);

    // UI: labels + lyric
    const c = this.ui.ctx;
    this.ui.clear();
    this.shade(c);
    for (const L of lab) this.label(c, L.p, L.s, L.a, L.hot, L.big, L.dy);
    if (m2) this.counter(c, t, w23);
    this.lyric(c, t);
    comp.draw(renderer, this.ui.upload(), out);
    return { bloom: 0.6, bloomThreshold: 0.85 };
  }

  // ─── movement 1: coalescing ─────────────────────────────────────────────────────────────────────
  coalesce(t: number, w20: Word[], w21: Word[], items: PulseItem[], lab: Lab[]) {
    const au = this.ctx.audio;
    const g = this.gpu;
    g.hideSMDetail();
    this.tileBody.count = this.tileGlow.count = 0;
    this.barrier.visible = false;
    const aThr = snap(au, t, w20[0]!.start);          // Neighbors
    const aRead = snap(au, t, w20[1]!.start);         // read
    const aClose = snap(au, t, w20[4]!.start);        // side
    const aBar = snap(au, t, w20[6]!.start);          // side,
    const tTrip = q(au, w21[1]!.start);               // trip
    const tMem = q(au, w21[3]!.start);                // memory,
    const tNice = q(au, w21[4]!.start);               // nice
    const tWide = q(au, w21[6]!.start);               // wide,
    const inFlight = t >= tNice && t < tWide;
    const landed = t >= tWide;
    const land = hit(au, t, w21[6]!.start, 0.14);
    const pulse = 0.5 + 0.5 * Math.sin(t * 9);

    // the warp: 32 threads (they fire together on the grid; lit when the data lands)
    for (let i = 0; i < N; i++) {
      const r = 0.055 * aThr * (1 + 0.25 * land);
      this.tmp.position.copy(this.thr[i]!);
      this.tmp.scale.setScalar(Math.max(1e-4, r));
      this.tmp.updateMatrix();
      this.thrCore.setMatrixAt(i, this.tmp.matrix);
      this.tmp.scale.setScalar(Math.max(1e-4, r * 2.4));
      this.tmp.updateMatrix();
      this.thrGlow.setMatrixAt(i, this.tmp.matrix);
      const k = aThr * (0.5 + 0.4 * aRead + (landed ? 1.2 : 0) + 1.5 * land);
      this.thrGlow.setColorAt(i, this.c3.setRGB(...col('signal', k * 0.9)));
    }
    this.thrCore.count = this.thrGlow.count = N;
    this.thrCore.instanceMatrix.needsUpdate = this.thrGlow.instanceMatrix.needsUpdate = true;
    this.thrGlow.instanceColor!.needsUpdate = true;

    // the 32 addresses: gaps close on "side" into one row; hidden while the bar is out on its trip
    const cw = lerp(PITCH * 0.62, PITCH * 1.0, aClose);
    const showCells = t < tNice;
    for (let i = 0; i < N; i++) {
      this.tmp.position.copy(this.cell[i]!);
      this.tmp.scale.set(0.42, 0.09, cw);
      if (!showCells) this.tmp.scale.setScalar(1e-4);
      this.tmp.updateMatrix();
      this.cellBody.setMatrixAt(i, this.tmp.matrix);
      this.tmp.scale.multiplyScalar(1.06);
      this.tmp.updateMatrix();
      this.cellGlow.setMatrixAt(i, this.tmp.matrix);
      const lit = aRead * (0.55 + 0.45 * aClose);
      this.cellGlow.setColorAt(i, this.c3.setRGB(...col('ember', lit * (0.7 + 0.2 * pulse))));
    }
    this.cellBody.count = this.cellGlow.count = N;
    this.cellBody.instanceMatrix.needsUpdate = this.cellGlow.instanceMatrix.needsUpdate = true;
    this.cellGlow.instanceColor!.needsUpdate = true;

    // "side by side," → one 128-byte bar over the row
    this.bar.visible = aBar > 0 && showCells;
    if (this.bar.visible) {
      const c0 = this.cell[0]!, c1 = this.cell[N - 1]!;
      this.bar.position.copy(c0).add(c1).multiplyScalar(0.5).setY(c0.y + 0.1);
      this.bar.scale.set(0.46, 0.03 + 0.02 * aBar, (c1.z - c0.z) + PITCH * 1.1);
      (this.bar.material as THREE.MeshBasicMaterial).color.setRGB(...col('signal', 1.6 * aBar + 1.2 * hit(au, t, w20[6]!.start, 0.12)));
    }

    // neighbours read neighbours: 32 parallel beams (bright on "read", held)
    if (aRead > 0 && t < tTrip + 0.3) {
      const fade = 1 - clamp((t - tTrip) / 0.3);
      for (let i = 0; i < N; i++) items.push({ curve: this.beams[i]!, u: 0.5, len: 1, width: 0.018, gain: (0.9 + 0.5 * aClose) * aRead * fade, color: LIN.signal });
    }
    // request: SM → HBM on "trip" (one streak), "trip" → "memory,"
    if (t >= tTrip && t < tMem + 0.15) {
      const u = ease.inQuad(clamp((t - tTrip) / Math.max(0.1, tMem - tTrip)));
      items.push({ curve: this.reqPath, u: u * 0.97 + 0.015, len: 0.05, width: 0.22, gain: 2.6, color: LIN.bone });
    }
    // contrast: scattered addresses, one trip per thread, fanned over every stack, "memory," → "nice"
    const gA = t >= tMem && t < tNice + 0.2 ? 1 - clamp((t - tNice) / 0.2) : 0;
    let ghostHit = 0;
    if (gA > 0) {
      for (let i = 0; i < N; i++) {
        const path = this.ghostPaths[i]!;
        const u = clamp((t - tMem) / 0.7 - hash(i, 3) * 0.45);
        if (u > 0 && u < 1) items.push({ curve: path, u, len: 0.025, width: 0.12, gain: 1.5 * gA, color: LIN.bone });
        if (u >= 1) ghostHit++;
      }
    }
    // the one wide bar rides the data path back: HBM → PHY → L2 → crossbar → SM, "nice" → "wide,"
    if (inFlight) {
      const u = ease.inOutQuad(clamp((t - tNice) / (tWide - tNice)));
      const L = this.dataPath.getLength();
      items.push({ curve: this.dataPath, u: clamp(u, 0.02, 0.98), len: (N * PITCH) / L, width: 0.4, gain: 2.8, color: LIN.signal });
      for (let k = 1; k <= 4; k++) items.push({ curve: this.dataPath, u: clamp(u - k * 0.025, 0.01, 0.99), len: 0.02, width: 0.3 - k * 0.05, gain: 1.6 / k, color: LIN.ember });
    }

    // SM glow: the chosen SM lights with its warp; the stacks answer the reads
    const reqIn = hit(au, t, w21[3]!.start, 0.12);
    g.setSMActivity((i) => (i === this.sm ? 0.3 + 0.6 * aThr + (landed ? 0.7 : 0) + 1.4 * land : this.ambient(i, t)));
    g.setHBMActivity((k) => (k === HBM_NEAR ? 0.05 + 0.1 * aRead + 0.7 * reqIn : 0.03 + (gA > 0 ? 0.3 * gA * (ghostHit / N) : 0)));

    const mid = (a: V3, b: V3) => a.clone().add(b).multiplyScalar(0.5);
    lab.push({ p: this.thr[0]!.clone().add(v3(0, 0.35, -0.2)), s: '32 threads · 1 warp', a: aThr * (landed ? 0 : 1) });
    lab.push({ p: this.thr[0]!.clone().add(v3(0, 0.35, -0.2)), s: 'coalesced: 1 trip', a: landed ? 1 : 0, hot: true, big: true });
    lab.push({ p: this.cell[N - 1]!.clone().add(v3(0.4, -0.35, 0.3)), s: 'their 32 addresses, side by side', a: aRead * (1 - aBar) });
    lab.push({ p: this.cell[N - 1]!.clone().add(v3(0.4, -0.35, 0.3)), s: 'one wide read · 32 × 4 B = 128 B', a: aBar * (showCells ? 1 : 0), hot: true });
    lab.push({ p: mid(this.smC, v3(0, H100_Y.dieTop, this.smC.z)).add(v3(0, 1.2, -3)), s: 'scattered: up to 32 separate trips · CH06', a: gA, big: true });
  }

  // ─── movement 2: a tile, far → near, then reused ───────────────────────────────────────────────
  tile(t: number, w22: Word[], w23: Word[], items: PulseItem[], lab: Lab[]) {
    const au = this.ctx.audio;
    const g = this.gpu;
    this.thrCore.count = this.thrGlow.count = this.cellBody.count = this.cellGlow.count = 0;
    this.bar.visible = false;
    const tLoad = q(au, w22[0]!.start), tTile = q(au, w22[2]!.start);
    const tArrive = this.tArrive(), tNear = q(au, w22[8]!.start);
    const tSync = q(au, w23[0]!.start), tBlock = q(au, w23[2]!.start), tReuse = q(au, w23[4]!.start);
    const aLoad = snap(au, t, w22[0]!.start);
    const aNear = snap(au, t, w22[8]!.start);

    // the SM close-up appears as the camera closes in
    const d = this.detail;
    const inSM = t >= tArrive - 0.35;
    if (inSM) g.smDetail(this.sm); else g.hideSMDetail();
    const l1 = this.toGroup(d, d.anchor('l1'));
    const smTop = this.smC.clone().add(v3(0, 0.12, 0));

    // tile position and scale: lift at the far stack, fly the data path, settle into L1/shared
    const hF = g.hbm(HBM_FAR).pos;
    const flying = t >= tTile && t < tArrive;
    let pos: V3, s = 1;
    if (t < tTile) {
      pos = hF.clone().add(v3(0, 0.3 + 0.9 * aLoad, 0));
    } else if (flying) {
      const u = beatEase(au, t, tTile, tArrive, ease.inOutQuad);
      pos = this.tilePath.getPointAt(clamp(u)).add(v3(0, 0.35 + 0.9 * (1 - u), 0));
      s = lerp(1, 0.55, u);
      for (let k = 1; k <= 6; k++) items.push({ curve: this.tilePath, u: clamp(u - k * 0.012, 0.005, 0.995), len: 0.012, width: 0.22 - k * 0.025, gain: 1.8 / k, color: LIN.ember });
    } else {
      const k = ease.outExpo(clamp((t - Math.max(tArrive, tNear - 0.2)) / 0.35));
      pos = smTop.clone().lerp(l1.clone().add(v3(0, 0.03, 0)), k);
      s = lerp(0.55, 0.2, k);
    }
    const reads = this.reads(t, tReuse);
    const readHit = reads.hit;
    const cs = 0.2 * s, pitch = 0.26 * s;
    for (let n = 0; n < 16; n++) {
      const i = n % 4, j = Math.floor(n / 4);
      this.tmp.position.copy(pos).add(v3((i - 1.5) * pitch, 0, (j - 1.5) * pitch));
      this.tmp.scale.setScalar(cs * aLoad);
      this.tmp.updateMatrix();
      this.tileBody.setMatrixAt(n, this.tmp.matrix);
      this.tmp.scale.setScalar(cs * aLoad * 1.1);
      this.tmp.updateMatrix();
      this.tileGlow.setMatrixAt(n, this.tmp.matrix);
      const k = 0.5 + 0.3 * aNear + 1.0 * readHit + 0.2 * hash(n, reads.n);
      this.tileGlow.setColorAt(n, this.c3.setRGB(...col('ember', k)));
    }
    this.tileBody.count = this.tileGlow.count = 16;
    this.tileBody.instanceMatrix.needsUpdate = this.tileGlow.instanceMatrix.needsUpdate = true;
    this.tileGlow.instanceColor!.needsUpdate = true;

    // the barrier: on "Sync" a wall sweeps across the SM and stops at the edge of shared memory, so
    // no lane reads the tile until every thread has arrived; on "block" it drops
    const r = g.smSize(this.sm);
    const dir = Math.sign(l1.z - this.smC.z) || 1;
    const aSync = snap(au, t, w23[0]!.start, 0.3);
    const released = t >= tBlock;
    const bz = lerp(this.smC.z - dir * r.d / 2, l1.z - dir * 0.1 * r.d, aSync);
    this.barrier.visible = inSM && t >= tSync && t < tBlock + 0.2;
    if (this.barrier.visible) {
      const drop = released ? ease.inQuad(clamp((t - tBlock) / 0.2)) : 0;
      this.barrier.position.set(this.smC.x, H100_Y.dieTop + 0.07 * (1 - drop), bz);
      this.barrier.scale.set(r.w * 1.02, 0.14 * (1 - drop) + 1e-3, 1);
      (this.barrier.material as THREE.MeshBasicMaterial).color.setRGB(...col('signal', (0.8 + 1.4 * hit(au, t, w23[0]!.start, 0.12)) * (1 - 0.5 * drop)));
    }

    // lanes: idle flicker, dim while held at the barrier, all fire on the release, then a quarter of
    // them fire on every read (the same ones the beams hit)
    const rel = hit(au, t, w23[2]!.start, 0.18);
    const litSet = (p: number, idx: number) => (idx + p + reads.n) % 4 === 0;
    d.setLanes((p, kind, idx) => {
      if (kind !== 'fp32') return 0.04;
      if (t < tSync) return 0.1 + 0.1 * (hash(p, idx, Math.floor(au.beatAt(t) * 2)) < 0.25 ? 1 : 0);
      if (!released) return 0.04;
      return 0.2 + 1.2 * rel + (reads.n > 0 && litSet(p, idx) ? 1.5 * readHit : 0);
    });
    d.setUnits((u) => (u === 'l1' ? 0.1 + 0.2 * aNear + 0.35 * readHit : u === 'sched' ? 0.1 + (t >= tSync && !released ? 0.45 : 0) : 0));
    d.setTensor(() => 0.02);

    // reuse: beams from the tile in shared memory to the lanes on each read (8th notes from "reuse")
    if (reads.n > 0 && readHit > 0.05) {
      const src = pos.clone().add(v3(0, 0.02, 0));
      for (let p = 0; p < 4; p++) for (let idx = 0; idx < 32; idx++) {
        if (!litSet(p, idx)) continue;
        const lane = this.toGroup(d, d.lanePos(p, 'fp32', idx)).add(v3(0, 0.02, 0));
        items.push({ curve: new THREE.LineCurve3(src, lane), u: 0.5, len: 1, width: 0.006, gain: 1.7 * readHit, color: LIN.signal });
      }
    }

    const loadHit = hit(au, t, w22[0]!.start, 0.2);
    g.setSMActivity((i) => (i === this.sm ? 0.4 + 0.6 * readHit : this.ambient(i, t)));
    g.setHBMActivity((k) => (k === HBM_FAR ? 0.06 + 0.1 * aLoad + 0.35 * loadHit : 0.03));

    // the route, named as the tile passes
    const near = (p: V3) => (flying ? clamp(1 - (pos.distanceTo(p) - 2) / 3) : 0);
    for (const w of this.waypoints) lab.push({ p: w.p.clone().add(v3(0, 0.25, 0)), s: w.s, a: near(w.p) });
    const hbmA = t < tTile + 0.9 ? aLoad : 0;
    lab.push({ p: pos.clone(), s: 'HBM · far · hundreds of cycles · CH06', a: hbmA, big: true, dy: 190 });
    lab.push({ p: pos.clone().add(v3(0.7, 0.4, -0.7)), s: 'a tile · 4 × 4', a: hbmA, hot: true });
    lab.push({ p: l1.clone().add(v3(dir * r.w * 0.28, 0.05, 0)), s: 'shared memory · near · ~30 cycles · CH06', a: t < tSync ? aNear : 0, hot: true, big: true });
    lab.push({ p: v3(this.smC.x, H100_Y.dieTop + 0.16, bz), s: 'sync: all wait here', a: t >= tSync && t < tReuse ? aSync : 0, big: true });
  }

  /** The rest of the die idles: a dim flicker on the 8ths, a little brighter with the drums. */
  ambient(i: number, t: number) {
    const b = Math.floor(this.ctx.audio.beatAt(t) * 2);
    return this.gpu.smEnabled(i) ? 0.012 + (hash(i, b) < 0.12 ? 0.04 : 0) : 0;
  }

  /** Arrival of the tile at the SM: the downbeat after "shared" (the flight lands on it). */
  tArrive() {
    const au = this.ctx.audio;
    const w = this.w[2]!;
    const t0 = w[4]!.start; // shared
    return au.downbeats.find((d) => d >= t0 - 0.05) ?? t0 + 1;
  }

  /** Reads of the tile: one per 8th from "reuse" to the end; `hit` decays after each. */
  reads(t: number, tReuse: number) {
    const au = this.ctx.audio;
    if (t < tReuse) return { n: 0, hit: 0 };
    const b0 = Math.round(au.beatAt(tReuse) * 2), b = au.beatAt(t) * 2;
    const n = Math.floor(b - b0) + 1;
    const tk = au.timeOfBeat((b0 + n - 1) / 2);
    return { n, hit: Math.pow(0.5, (t - tk) / 0.09) };
  }

  toGroup(d: SMDetail, v: V3): V3 {
    this.gpu.group.updateMatrixWorld(true);
    const w = d.group.localToWorld(v.clone());
    return this.gpu.group.worldToLocal(w);
  }

  // ─── camera: always moving, big moves land on downbeats ───────────────────────────────────────
  shot(t: number): Shot {
    const au = this.ctx.audio;
    const g = this.gpu;
    const [w20, w21, w22, w23] = this.w as [Word[], Word[], Word[], Word[]];
    const D = au.downbeats;
    const db = (x: number) => D.find((d) => d >= x - 0.02) ?? x;
    const dv = drive(au, t, 1, 1.6, this.ctx.start);
    const push = 1 - 0.035 * kickPush(au, t);
    const orbitAt = (c: V3, dist: number, az: number, el: number, fov = 34): Shot => ({
      pos: c.clone().add(v3(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)).multiplyScalar(dist * push)),
      tgt: c.clone(), fov, dist,
    });
    const mix = (a: Shot, b: Shot, k: number): Shot => ({
      pos: a.pos.clone().lerp(b.pos, k), tgt: a.tgt.clone().lerp(b.tgt, k),
      fov: lerp(a.fov, b.fov, k), dist: lerp(a.dist, b.dist, k),
    });
    const hN = g.hbm(HBM_NEAR).pos, hF = g.hbm(HBM_FAR).pos;
    const rowC = this.smC.clone().lerp(hN, 0.5).setY(H100_Y.dieTop + 1.4);
    const warpC = this.thr[N >> 1]!.clone();

    // S1: over the warp and its addresses (the dive lands on the first downbeat, then a slow push)
    const tN = db(w20[0]!.start);
    const s1 = (tt: number) => {
      const d0 = lerp(46, 27, ease.outCubic(clamp((tt - this.ctx.start) / Math.max(0.3, tN - this.ctx.start))));
      return orbitAt(rowC, lerp(d0, 22, clamp((tt - tN) / 5.4)), 1.42 + 0.025 * dv, 0.86);
    };
    // S2: high over the east half, the whole route in view (HBM → PHY → L2 → crossbar → SM); after the
    // bar lands on "wide," the camera drops onto the warp
    const tTrip = db(w21[0]!.start), tWide = q(au, w21[6]!.start);
    const routeC = v3(DIM.die.w / 2 - 4.5, H100_Y.dieTop, (hN.z + g.l2Boxes[1]!.getCenter(v3()).z) / 2 - 1);
    const s2 = (tt: number) => {
      const k = ease.outCubic(clamp((tt - tWide) / 0.5));
      const a = orbitAt(routeC, 31 - 0.4 * (tt - tTrip), 0.95 + 0.03 * dv, 1.02);
      const b = orbitAt(warpC, 12 - 0.8 * Math.max(0, tt - tWide), 0.75 + 0.03 * dv, 0.72);
      return mix(a, b, k);
    };
    // S3: high over the far stack, looking down the route the tile is about to take
    const s3 = () => orbitAt(hF.clone().add(v3(1.2, 0.8, 0)), 10, -1.0 + 0.04 * dv, 0.95);
    // S4: chase along the data path, a little above so the die reads as it streams past
    const tTile = q(au, w22[2]!.start), tArr = this.tArrive();
    const s4 = (tt: number): Shot => {
      const u = beatEase(au, tt, tTile, tArr, ease.inOutQuad);
      const P = (x: number) => this.tilePath.getPointAt(clamp(x));
      const tgt = P(u + 0.04).add(v3(0, 0.5, 0));
      const pos = P(u - 0.09).add(v3(0, 4.4 - 2.2 * u, 0));
      pos.x += Math.sin(dv * 0.4) * 0.8;
      return { pos, tgt, fov: 42, dist: 5 };
    };
    // S5/S6: dive into the SM, ending on the shared-memory side so the tile and the lanes share the frame
    const d = this.detail;
    const l1 = this.toGroup(d, d.anchor('l1'));
    const dir = Math.sign(l1.z - this.smC.z) || 1;
    const azL1 = dir > 0 ? 0 : Math.PI;
    const smT = this.smC.clone().lerp(l1, 0.3).add(v3(0, 0.04, 0));
    const tSyncDb = db(w23[0]!.start);
    const s5 = (tt: number) => {
      const k = ease.inOutCubic(clamp((tt - tArr) / Math.max(0.3, tSyncDb - tArr)));
      return orbitAt(smT, lerp(4.4, 2.1, k), lerp(-1.57, azL1 - 0.6, k) + 0.04 * dv, lerp(1.15, 0.8, k), 38);
    };
    const tLow = db(w23[4]!.start - 0.5);
    const s6 = (tt: number) => {
      const k = ease.inOutCubic(clamp((tt - (tLow - 0.5)) / 0.5));
      return orbitAt(smT, lerp(2.1, 1.6, k), azL1 - 0.6 + lerp(0, 0.35, k) + 0.04 * dv, lerp(0.8, 0.56, k), 38);
    };

    const tLoad = q(au, w22[0]!.start);
    if (t < tTrip - 0.55) return s1(t);
    if (t < tTrip) return mix(s1(t), s2(t), ease.inOutCubic(clamp((t - (tTrip - 0.55)) / 0.55)));
    if (t < tLoad) return s2(t);
    if (t < tTile) return mix(s2(t), s3(), ease.inOutExpo(clamp((t - tLoad) / 0.2)));
    if (t < tTile + 0.35) return mix(s3(), s4(t), ease.inOutCubic(clamp((t - tTile) / 0.35)));
    if (t < tArr - 0.3) return s4(t);
    if (t < tArr + 0.15) return mix(s4(t), s5(t), ease.inOutCubic(clamp((t - (tArr - 0.3)) / 0.45)));
    if (t < tSyncDb) return s5(t);
    return s6(t);
  }

  // ─── screen-space UI ────────────────────────────────────────────────────────────────────────────
  /** A soft dark corner behind the lyric so it reads over any 3D. */
  shade(c: C2) {
    const gr = c.createLinearGradient(0, 0, 0, 320);
    gr.addColorStop(0, rgba('ink', 0.8));
    gr.addColorStop(0.55, rgba('ink', 0.45));
    gr.addColorStop(1, rgba('ink', 0));
    c.fillStyle = gr;
    c.fillRect(0, 0, W, 320);
  }

  label(c: C2, p: V3, s: string, a: number, hot = false, big = false, dy = 0) {
    if (a <= 0.01) return;
    const v = this.gpu.group.localToWorld(p.clone()).project(this.cam);
    if (v.z > 1 || v.z < -1) return;
    const x = (v.x * 0.5 + 0.5) * W, y = (-v.y * 0.5 + 0.5) * H + dy;
    if (x < 40 || x > W - 40 || y < 250 || y > H - 40) return;
    const size = big ? 30 : 25;
    c.save();
    c.font = font(big ? this.famMB : this.famM, size);
    c.textBaseline = 'alphabetic';
    const tw = c.measureText(s).width;
    const tx = Math.min(x + 18, W - 60 - tw);
    c.fillStyle = rgba('ink', 0.55 * a);
    c.fillRect(tx - 8, y - size - 4, tw + 16, size + 14);
    c.fillStyle = hot ? rgba('signal', a) : rgba('bone', 0.95 * a);
    c.fillText(s, tx, y);
    c.fillStyle = rgba('signal', a);
    c.fillRect(x - 4, y - 4, 8, 8);
    c.restore();
  }

  /** The read counter (bottom right) once the tile is being reused. */
  counter(c: C2, t: number, w23: Word[]) {
    const au = this.ctx.audio;
    const r = this.reads(t, q(au, w23[4]!.start));
    if (r.n <= 0) return;
    c.save();
    c.fillStyle = rgba('ink', 0.72);
    c.fillRect(W - 640, H - 240, 560, 222);
    c.fillStyle = rgba('signal', 0.9);
    c.fillRect(W - 640, H - 240, 6, 222);
    c.textAlign = 'right';
    c.font = font(this.famM, 26);
    c.fillStyle = rgba('bone', 0.9);
    c.fillText('reads from the tile', W - 120, H - 190);
    c.font = font(LFAM, 120);
    c.fillStyle = rgba('signal', 0.85 + 0.15 * r.hit);
    c.fillText(String(r.n), W - 120, H - 80);
    c.font = font(this.famM, 24);
    c.fillStyle = rgba('ash', 0.9);
    c.fillText('load it once · reuse it many times', W - 120, H - 40);
    c.restore();
  }

  /** One line at a time, fixed top-left; each word pops to signal when sung, then bone. */
  lyric(c: C2, t: number) {
    const L = this.lines;
    let k = 0;
    for (let i = 1; i < L.length; i++) if (t >= L[i]!.words[0]!.start - 0.3) k = i;
    const line = L[k]!, text = line.text;
    const lay = layout(text, LFAM, LSIZE);
    c.save();
    c.font = font(LFAM, LSIZE);
    c.textBaseline = 'alphabetic';
    let pos = 0;
    for (let i = 0; i < line.words.length; i++) {
      const w = line.words[i]!;
      const at = text.indexOf(w.w, pos);
      const a = at < 0 ? pos : at;
      pos = a + w.w.length;
      const x = LX + (lay.glyphs[a]?.x ?? 0);
      const next = line.words[i + 1];
      const sung = t >= w.start;
      const cur = sung && (!next || t < next.start) && t < w.end + 0.25;
      c.fillStyle = cur ? rgba('signal', 1) : sung ? rgba('bone', 1) : rgba('bone', 0.3);
      c.fillText(w.w, x, LY);
    }
    c.restore();
  }
}
