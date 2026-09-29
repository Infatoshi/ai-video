// Shared 3D model of an NVIDIA H100 SXM5 module for v5 (SPEC.md "v5"): the board, the package
// substrate, the CoWoS interposer, the GH100 die and six HBM3 stack sites, plus a close-up SM that can
// be dropped onto any SM site. three.js, procedural, 1 unit = 1 mm, y up, die centred on the origin.
//
// Facts (NVIDIA H100 Tensor Core GPU Architecture whitepaper, V1.01–1.04):
//   GH100 full: 8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 144 SMs; 6 HBM3 stacks, 12 × 512-bit memory
//   controllers; 60 MB L2. H100 SXM5: 132 SMs (66 TPCs), 5 HBM3 stacks (80 GB), 10 memory controllers,
//   50 MB L2. TSMC 4N, 80 B transistors, 814 mm² die. Per SM: 128 FP32, 64 INT32, 64 FP64 cores,
//   4 fourth-gen tensor cores, 256 KB register file, 256 KB L1/shared memory, TMA. NVLink 4: 18 links,
//   900 GB/s total. The die surface follows the block diagram's arrangement (Figure 6); board,
//   substrate and component placement are generic (no dimensions are claimed).
//
// ─── API ─────────────────────────────────────────────────────────────────────────────────────────
//   const gpu = new H100({ renderer });       // renderer is needed once for the PMREM environment
//   scene.add(gpu.group); scene.add(H100.rig()); scene.environment = gpu.env;
//   gpu.update(t, f.a)                        // default music-driven activity (pure function of t)
//   gpu.setSMActivity(i => 0..1 | null)        // override SM glow (null → back to the default)
//   gpu.setHBMActivity(k => 0..1 | null)
//   gpu.explode(k)                            // 0..1: die / HBM, interposer, substrate lift off the board
//   gpu.smCount (144 sites), gpu.smEnabled(i), gpu.smPos(i) (die-top centre, in group space),
//     gpu.smSize(i) ({w, d} mm), gpu.gpcBox(g), gpu.l2Box (two partitions: l2Boxes[0|1]),
//     gpu.hbm(k) → { pos, active, box } for k 0..5 (0–2 west north→south, 3–5 east), gpu.nvlinkPorts
//   gpu.smDetail(i)                           // the SMDetail close-up, placed on SM i (visible); see _h100-sm.ts
//   gpu.hideSMDetail()
//   gpu.pulsePath(hbm, sm)                    // Curve3 in group space: HBM stack → PHY → L2 → crossbar → SM
//   gpu.pulses.set([{ curve, u, len, gain }]) // draw data pulses (additive, HDR) along any curves
//   gpu.toWorld(v) / worldToScreen helpers: use gpu.group.localToWorld(v.clone())
//   H100.fitClip(cam, distance)               // near/far for a camera at `distance` mm from its subject
//   glowMat(color?)                           // additive glow that keeps alpha ≤ 1; use it instead of AdditiveBlending
// Group-space positions account for the current explode() offsets.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { LIN } from '../engine/palette';
import { clamp, frameIdx, hash, mulberry32 } from '../engine/util';
import type { AudioSample } from '../engine/audio';
import { DIE_LAYOUT, gpcRect, makeBoardTexture, makeDieTexture, smRect } from './_h100-tex';
import { SMDetail, glowMat } from './_h100-sm';

export { SMDetail, glowMat } from './_h100-sm';
export type { LaneKind, SMUnit } from './_h100-sm';

const lin = (k: keyof typeof LIN, s = 1) => new THREE.Color().setRGB(LIN[k][0] * s, LIN[k][1] * s, LIN[k][2] * s, THREE.LinearSRGBColorSpace);

// ── dimensions (mm) ────────────────────────────────────────────────────────────────────────────────
export const DIM = {
  board: { w: 142, d: 78, t: 2.0 },
  sub: { w: 64, d: 58, t: 1.4, y0: 0.35 },
  int: { w: 50, d: 31, t: 0.35, gap: 0.14 },
  die: { w: 28.5, d: 28.56, t: 0.78, gap: 0.07 },
  hbm: { w: 8.0, d: 9.2, base: 0.14, layer: 0.075, layers: 8, x: 19.45, pitch: 9.6 },
};
const Y = (() => {
  const subTop = DIM.sub.y0 + DIM.sub.t;
  const intBot = subTop + DIM.int.gap, intTop = intBot + DIM.int.t;
  const dieBot = intTop + DIM.die.gap, dieTop = dieBot + DIM.die.t;
  return { subBot: DIM.sub.y0, subTop, intBot, intTop, dieBot, dieTop };
})();
export const H100_Y = Y;

/** Which TPCs are fused off on this (fictional but representative) SXM5 part: 6 TPCs → 132 SMs. */
const FUSED: [number, number][] = [[0, 8], [1, 4], [3, 0], [4, 6], [6, 2], [7, 8]];
const SPARE_HBM = 5;

export interface PulseItem { curve: THREE.Curve<THREE.Vector3>; u: number; len?: number; gain?: number; color?: [number, number, number]; width?: number }

/** Additive glowing capsules along curves (group space of whatever it's added to). */
export class Pulses {
  mesh: THREE.InstancedMesh;
  private tmp = new THREE.Object3D();
  private col = new THREE.Color();
  constructor(capacity = 512) {
    const geo = new THREE.CylinderGeometry(0.5, 0.5, 1, 8, 1).rotateZ(Math.PI / 2); // along +x
    this.mesh = new THREE.InstancedMesh(geo, glowMat(), capacity);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    this.mesh.setColorAt(0, this.col.setRGB(0, 0, 0));
  }
  set(items: PulseItem[]) {
    const m = this.mesh;
    let n = 0;
    const len0 = new Map<THREE.Curve<THREE.Vector3>, number>();
    for (const it of items) {
      if (n >= m.instanceMatrix.count) break;
      if (it.u < 0 || it.u > 1) continue;
      let L = len0.get(it.curve);
      if (L === undefined) { L = it.curve.getLength(); len0.set(it.curve, L); }
      const p = it.curve.getPointAt(clamp(it.u)), tg = it.curve.getTangentAt(clamp(it.u)).normalize();
      this.tmp.position.copy(p);
      this.tmp.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), tg);
      const w = it.width ?? 0.12;
      this.tmp.scale.set(Math.max(w, (it.len ?? 0.04) * L), w, w);
      this.tmp.updateMatrix();
      m.setMatrixAt(n, this.tmp.matrix);
      const c = it.color ?? LIN.ember, g = it.gain ?? 2;
      m.setColorAt(n, this.col.setRGB(c[0] * g, c[1] * g, c[2] * g));
      n++;
    }
    m.count = n;
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }
}

let ENV_CACHE: THREE.Texture | null = null;
let DIE_TEX_CACHE: ReturnType<typeof makeDieTexture> | null = null;
let BOARD_TEX_CACHE: THREE.CanvasTexture | null = null;
let SUB_TEX_CACHE: THREE.CanvasTexture | null = null;

export class H100 {
  group = new THREE.Group();
  env: THREE.Texture | null = null;
  readonly smCount = 144;
  pulses = new Pulses(512);

  // layers (moved by explode)
  private layerSub = new THREE.Group();
  private layerInt = new THREE.Group();
  private layerDie = new THREE.Group();
  private hbmGroups: THREE.Group[] = [];
  private hbmLayers: THREE.Object3D[][] = [];
  private bga!: THREE.InstancedMesh;
  private c4!: THREE.InstancedMesh;
  private ubump!: THREE.InstancedMesh;
  private explodeK = 0;

  private smGlow!: THREE.InstancedMesh;
  private hbmGlow: THREE.Mesh[] = [];
  private phyGlow!: THREE.InstancedMesh;
  private smFn: ((i: number) => number) | null = null;
  private hbmFn: ((k: number) => number) | null = null;
  private detail: SMDetail | null = null;
  private detailAt = -1;
  private tmp = new THREE.Object3D();
  private col = new THREE.Color();
  private smRects: { g: number; s: number; x: number; z: number; w: number; d: number; en: boolean }[] = [];

  constructor(opts: { renderer?: THREE.WebGLRenderer; texSize?: number } = {}) {
    if (opts.renderer && !ENV_CACHE) {
      const pm = new THREE.PMREMGenerator(opts.renderer);
      ENV_CACHE = pm.fromScene(new RoomEnvironment(), 0.04).texture;
      pm.dispose();
    }
    this.env = ENV_CACHE;
    // SM site table
    for (let g = 0; g < 8; g++) for (let s = 0; s < 18; s++) {
      const r = smRect(g, s);
      const en = !FUSED.some(([fg, ft]) => fg === g && ft === s >> 1);
      this.smRects.push({ g, s, x: (r.u0 + r.u1) / 2 * DIM.die.w - DIM.die.w / 2, z: (r.v0 + r.v1) / 2 * DIM.die.d - DIM.die.d / 2, w: (r.u1 - r.u0) * DIM.die.w, d: (r.v1 - r.v0) * DIM.die.d, en });
    }
    DIE_TEX_CACHE ??= makeDieTexture(opts.texSize ?? 4096, (g, s) => this.smRects[g * 18 + s]!.en);
    BOARD_TEX_CACHE ??= makeBoardTexture(2048, 1152, 7, 1);
    SUB_TEX_CACHE ??= makeBoardTexture(1024, 1024, 11, 1.6);
    this.buildBoard();
    this.buildSubstrate();
    this.buildInterposer();
    this.buildDie();
    this.buildHBM();
    this.group.add(this.layerSub, this.layerInt, this.layerDie, this.pulses.mesh);
    this.explode(0);
  }

  // ── materials ────────────────────────────────────────────────────────────────────────────────
  private static edgeMat(op: number) { return new THREE.LineBasicMaterial({ color: lin('bone'), transparent: true, opacity: op, depthWrite: false }); }
  private static std(color: THREE.Color, metal: number, rough: number, extra: THREE.MeshStandardMaterialParameters = {}) {
    return new THREE.MeshStandardMaterial({ color, metalness: metal, roughness: rough, envMapIntensity: 0.22, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, ...extra });
  }

  /** Box geometries (templates × transforms) merged into one mesh + one edge LineSegments. */
  private boxes(list: { x: number; y: number; z: number; w: number; h: number; d: number; ry?: number }[], mat: THREE.Material, edgeOp: number, parent: THREE.Object3D, cyl = false) {
    const geos: THREE.BufferGeometry[] = [];
    const ev: number[] = [];
    const m = new THREE.Matrix4(), q = new THREE.Quaternion();
    for (const b of list) {
      const g = cyl ? new THREE.CylinderGeometry(b.w / 2, b.w / 2, b.h, 20) : new THREE.BoxGeometry(b.w, b.h, b.d);
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), b.ry ?? 0);
      m.compose(new THREE.Vector3(b.x, b.y + b.h / 2, b.z), q, new THREE.Vector3(1, 1, 1));
      g.applyMatrix4(m);
      geos.push(g);
      if (edgeOp > 0) {
        const e = new THREE.EdgesGeometry(cyl ? new THREE.CylinderGeometry(b.w / 2, b.w / 2, b.h, 20) : new THREE.BoxGeometry(b.w, b.h, b.d), 30);
        e.applyMatrix4(m);
        ev.push(...(e.getAttribute('position').array as Float32Array));
      }
    }
    const mesh = new THREE.Mesh(mergeGeometries(geos), mat);
    parent.add(mesh);
    if (ev.length) {
      const eg = new THREE.BufferGeometry();
      eg.setAttribute('position', new THREE.Float32BufferAttribute(ev, 3));
      parent.add(new THREE.LineSegments(eg, H100.edgeMat(edgeOp)));
    }
    return mesh;
  }

  // ── board ───────────────────────────────────────────────────────────────────────────────────
  private buildBoard() {
    const { w, d, t } = DIM.board;
    const top = H100.std(lin('bone', 0.42), 0.15, 0.62, { map: BOARD_TEX_CACHE! });
    const side = H100.std(lin('ink2'), 0.1, 0.8);
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, t, d), [side, side, top, side, side, side]);
    b.position.y = -t / 2;
    this.group.add(b);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, t, d)), H100.edgeMat(0.45));
    e.position.copy(b.position);
    this.group.add(e);
    const rnd = mulberry32(0x5b);
    // VRM: inductor columns flanking the package, power stages beside them
    const ind: Parameters<H100['boxes']>[0] = [], ps: typeof ind = [], caps: typeof ind = [], mlcc: typeof ind = [];
    for (const sx of [-1, 1]) {
      for (const cx of [40, 50]) for (let k = 0; k < 8; k++) ind.push({ x: sx * cx, y: 0, z: -29.75 + k * 8.5, w: 6.2, h: 4.4, d: 6.2 });
      for (const cx of [35.4, 45.2]) for (let k = 0; k < 8; k++) ps.push({ x: sx * cx, y: 0, z: -29.75 + k * 8.5, w: 3.2, h: 0.9, d: 4.2 });
      for (let k = 0; k < 7; k++) caps.push({ x: sx * 61, y: 0, z: -27 + k * 9, w: 5, h: 4, d: 5 });
      for (let k = 0; k < 7; k++) caps.push({ x: sx * 67, y: 0, z: -27 + k * 9, w: 5, h: 4, d: 5 });
    }
    for (let k = 0; k < 260; k++) {
      const x = (rnd() - 0.5) * 132, z = (rnd() - 0.5) * 70;
      if (Math.abs(x) < 34 && Math.abs(z) < 31) continue;
      if (Math.abs(Math.abs(x) - 45) < 9.5) continue;
      mlcc.push({ x, y: 0, z, w: 1.0, h: 0.5, d: 0.5, ry: rnd() < 0.5 ? 0 : Math.PI / 2 });
    }
    this.boxes(ind, H100.std(lin('graphite', 0.32), 0.85, 0.32, { envMapIntensity: 0.35 }), 0.3, this.group);
    this.boxes(ps, H100.std(lin('ink2'), 0.2, 0.6), 0.25, this.group);
    this.boxes(caps, H100.std(lin('ash', 0.22), 0.75, 0.3, { envMapIntensity: 0.35 }), 0.3, this.group, true);
    this.boxes(mlcc, H100.std(lin('ash', 0.2), 0.4, 0.5), 0, this.group);
    // mounting holes: metal rings
    for (const [x, z] of [[-66, -34], [66, -34], [-66, 34], [66, 34]] as const) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(1.7, 3.2, 40).rotateX(-Math.PI / 2), H100.std(lin('ash', 0.6), 0.9, 0.25));
      ring.position.set(x, 0.01, z);
      const hole = new THREE.Mesh(new THREE.CircleGeometry(1.7, 32).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: lin('ink') }));
      hole.position.set(x, 0.012, z);
      this.group.add(ring, hole);
    }
    // underside: the two mezzanine connectors (NVLink / power / PCIe to the baseboard)
    this.boxes([{ x: 0, y: -t - 5.5, z: -24, w: 70, h: 5.5, d: 6 }, { x: 0, y: -t - 5.5, z: 24, w: 70, h: 5.5, d: 6 }], H100.std(lin('ink2'), 0.2, 0.55), 0.35, this.group);
    // BGA balls between board and substrate (shown when exploded)
    const ballGeo = new THREE.SphereGeometry(0.28, 8, 6);
    const nb = 34 * 30;
    this.bga = new THREE.InstancedMesh(ballGeo, H100.std(lin('ash', 0.7), 0.95, 0.2), nb);
    let n = 0;
    for (let i = 0; i < 34; i++) for (let j = 0; j < 30; j++) {
      this.tmp.position.set(-DIM.sub.w / 2 + 2 + i * ((DIM.sub.w - 4) / 33), 0.17, -DIM.sub.d / 2 + 2 + j * ((DIM.sub.d - 4) / 29));
      this.tmp.scale.setScalar(1); this.tmp.updateMatrix();
      this.bga.setMatrixAt(n++, this.tmp.matrix);
    }
    this.group.add(this.bga);
  }

  // ── package substrate ─────────────────────────────────────────────────────────────────────────
  private buildSubstrate() {
    const { w, d, t, y0 } = DIM.sub;
    const top = H100.std(lin('bone', 0.4), 0.2, 0.55, { map: SUB_TEX_CACHE! });
    const side = H100.std(lin('ink2'), 0.15, 0.7);
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, t, d), [side, side, top, side, side, side]);
    m.position.y = y0 + t / 2;
    this.layerSub.add(m);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, t, d)), H100.edgeMat(0.5));
    e.position.copy(m.position);
    this.layerSub.add(e);
    // decoupling capacitors in a ring around the interposer
    const caps: Parameters<H100['boxes']>[0] = [];
    const rnd = mulberry32(0x7c);
    for (let k = 0; k < 150; k++) {
      const side = k % 4, a = rnd();
      const x = side < 2 ? (a - 0.5) * (w - 6) : (side === 2 ? -1 : 1) * (DIM.int.w / 2 + 2.5 + rnd() * 3);
      const z = side < 2 ? (side === 0 ? -1 : 1) * (DIM.int.d / 2 + 2.5 + rnd() * 6) : (a - 0.5) * (d - 6);
      caps.push({ x, y: Y.subTop, z, w: 1.0, h: 0.55, d: 0.5, ry: rnd() < 0.5 ? 0 : Math.PI / 2 });
    }
    this.boxes(caps, H100.std(lin('ash', 0.4), 0.5, 0.45), 0, this.layerSub);
    // C4 bumps (substrate ↔ interposer), visible when exploded
    const nx = 40, nz = 25;
    this.c4 = new THREE.InstancedMesh(new THREE.SphereGeometry(0.12, 6, 4), H100.std(lin('ash', 0.7), 0.95, 0.2), nx * nz);
    let n = 0;
    for (let i = 0; i < nx; i++) for (let j = 0; j < nz; j++) {
      this.tmp.position.set(-DIM.int.w / 2 + 1 + i * ((DIM.int.w - 2) / (nx - 1)), Y.subTop + 0.07, -DIM.int.d / 2 + 1 + j * ((DIM.int.d - 2) / (nz - 1)));
      this.tmp.updateMatrix();
      this.c4.setMatrixAt(n++, this.tmp.matrix);
    }
    this.layerSub.add(this.c4);
  }

  // ── CoWoS interposer ──────────────────────────────────────────────────────────────────────────
  private buildInterposer() {
    const { w, d, t } = DIM.int;
    const mat = H100.std(lin('graphite', 0.32), 0.7, 0.26);
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, t, d), mat);
    m.position.y = Y.intBot + t / 2;
    this.layerInt.add(m);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, t, d)), H100.edgeMat(0.55));
    e.position.copy(m.position);
    this.layerInt.add(e);
    // redistribution wiring between the die edge and each HBM stack: fine parallel traces on the surface
    const lv: number[] = [];
    const y = Y.intTop + 0.004;
    for (const sx of [-1, 1]) for (let k = 0; k < 3; k++) {
      const zc = (k - 1) * DIM.hbm.pitch;
      for (let i = 0; i < 24; i++) {
        const z = zc - DIM.hbm.d * 0.4 + i * (DIM.hbm.d * 0.8 / 23);
        lv.push(sx * (DIM.die.w / 2), y, z, sx * (DIM.hbm.x - DIM.hbm.w / 2), y, z);
      }
    }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(lv, 3));
    this.layerInt.add(new THREE.LineSegments(lg, H100.edgeMat(0.22)));
    // microbumps under the die and the HBM stacks (visible when exploded)
    const pts: [number, number][] = [];
    for (let i = 0; i < 48; i++) for (let j = 0; j < 48; j++) pts.push([-DIM.die.w / 2 + 0.6 + i * ((DIM.die.w - 1.2) / 47), -DIM.die.d / 2 + 0.6 + j * ((DIM.die.d - 1.2) / 47)]);
    for (let k = 0; k < 6; k++) {
      const c = this.hbmCentre(k);
      for (let i = 0; i < 10; i++) for (let j = 0; j < 12; j++) pts.push([c.x - DIM.hbm.w * 0.4 + i * (DIM.hbm.w * 0.8 / 9), c.z - DIM.hbm.d * 0.4 + j * (DIM.hbm.d * 0.8 / 11)]);
    }
    this.ubump = new THREE.InstancedMesh(new THREE.SphereGeometry(0.07, 6, 4), H100.std(lin('bone', 0.6), 0.95, 0.2), pts.length);
    pts.forEach(([x, z], n) => { this.tmp.position.set(x, Y.intTop + 0.035, z); this.tmp.updateMatrix(); this.ubump.setMatrixAt(n, this.tmp.matrix); });
    this.layerInt.add(this.ubump);
  }

  // ── GH100 die ─────────────────────────────────────────────────────────────────────────────────
  private buildDie() {
    const { w, d, t } = DIM.die;
    const tex = DIE_TEX_CACHE!;
    const top = H100.std(lin('bone', 0.62), 0.45, 0.52, { map: tex.map, emissiveMap: tex.emissive, emissive: lin('bone', 0.06), envMapIntensity: 0.32 });
    const side = H100.std(lin('graphite', 0.25), 0.6, 0.3);
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, t, d), [side, side, top, side, side, side]);
    m.position.y = Y.dieBot + t / 2;
    this.layerDie.add(m);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, t, d)), H100.edgeMat(0.7));
    e.position.copy(m.position);
    this.layerDie.add(e);
    // SM activity: one additive quad per SM site, just above the silicon
    const q = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
    this.smGlow = new THREE.InstancedMesh(q, glowMat(), 144);
    this.smRects.forEach((r, n) => {
      this.tmp.position.set(r.x, Y.dieTop + 0.004, r.z);
      this.tmp.scale.set(r.w * 0.98, 1, r.d * 0.98);
      this.tmp.updateMatrix();
      this.smGlow.setMatrixAt(n, this.tmp.matrix);
      this.smGlow.setColorAt(n, this.col.setRGB(0, 0, 0));
    });
    this.layerDie.add(this.smGlow);
    // memory-controller / PHY strips light up with HBM traffic: 6 per side, 2 per stack
    this.phyGlow = new THREE.InstancedMesh(q, glowMat(), 12);
    for (let k = 0; k < 12; k++) {
      const side = k < 6 ? DIE_LAYOUT.mcW : DIE_LAYOUT.mcE, j = k % 6;
      const u = (side.u0 + side.u1) / 2, v = 0.02 + (j + 0.5) * (0.96 / 6);
      this.tmp.position.set(u * w - w / 2, Y.dieTop + 0.005, v * d - d / 2);
      this.tmp.scale.set((side.u1 - side.u0) * w * 0.8, 1, (0.96 / 6) * d * 0.85);
      this.tmp.updateMatrix();
      this.phyGlow.setMatrixAt(k, this.tmp.matrix);
      this.phyGlow.setColorAt(k, this.col.setRGB(0, 0, 0));
    }
    this.layerDie.add(this.phyGlow);
  }

  // ── HBM3 stacks ───────────────────────────────────────────────────────────────────────────────
  private hbmCentre(k: number) {
    const sx = k < 3 ? -1 : 1, j = k % 3;
    return new THREE.Vector3(sx * DIM.hbm.x, 0, (j - 1) * DIM.hbm.pitch);
  }
  private buildHBM() {
    const { w, d, base, layer, layers } = DIM.hbm;
    for (let k = 0; k < 6; k++) {
      const g = new THREE.Group();
      const c = this.hbmCentre(k);
      g.position.set(c.x, Y.dieBot, c.z);
      const active = k !== SPARE_HBM;
      const baseMat = H100.std(lin('graphite', active ? 0.3 : 0.18), 0.6, 0.3);
      const dram = H100.std(lin('graphite', active ? 0.24 : 0.14), 0.55, 0.32);
      const objs: THREE.Object3D[] = [];
      // base (logic) die + DRAM dies with thin gaps between them
      const mk = (y0: number, h: number, mat: THREE.Material, inset: number) => {
        const o = new THREE.Group();
        const m = new THREE.Mesh(new THREE.BoxGeometry(w - inset, h, d - inset), mat);
        m.position.y = h / 2;
        const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w - inset, h, d - inset)), H100.edgeMat(active ? 0.55 : 0.3));
        e.position.y = h / 2;
        o.add(m, e);
        o.position.y = y0;
        o.userData.y0 = y0;
        g.add(o);
        objs.push(o);
      };
      mk(0, base, baseMat, 0);
      for (let i = 0; i < layers; i++) mk(base + i * layer + 0.006, layer - 0.012, dram, 0.25);
      // activity glow on the die-facing side of the stack
      const glow = new THREE.Mesh(new THREE.PlaneGeometry(d * 0.9, base + layers * layer), glowMat(0x000000, { side: THREE.DoubleSide }));
      glow.rotation.y = c.x < 0 ? Math.PI / 2 : -Math.PI / 2;
      glow.position.set((c.x < 0 ? 1 : -1) * (w / 2 + 0.01), (base + layers * layer) / 2, 0);
      g.add(glow);
      this.hbmGlow.push(glow);
      g.userData.active = active;
      this.hbmGroups.push(g);
      this.hbmLayers.push(objs);
      this.layerDie.add(g);
    }
  }

  // ── API ───────────────────────────────────────────────────────────────────────────────────────
  static rig(): THREE.Group {
    const g = new THREE.Group();
    const key = new THREE.DirectionalLight(new THREE.Color().setRGB(0.92, 0.95, 1.0), 1.5);
    key.position.set(-60, 120, 80);
    const rim = new THREE.DirectionalLight(new THREE.Color().setRGB(LIN.ember[0], LIN.ember[1], LIN.ember[2]), 2.2);
    rim.position.set(90, 40, -140);
    const fill = new THREE.HemisphereLight(new THREE.Color().setRGB(0.12, 0.12, 0.13), new THREE.Color().setRGB(0.02, 0.018, 0.016), 0.35);
    g.add(key, rim, fill);
    return g;
  }

  static fitClip(cam: THREE.PerspectiveCamera, distance: number) {
    cam.near = Math.max(0.002, distance * 0.02);
    cam.far = Math.max(50, distance * 60);
    cam.updateProjectionMatrix();
  }

  smEnabled(i: number) { return this.smRects[i]!.en; }
  smSize(i: number) { const r = this.smRects[i]!; return { w: r.w, d: r.d }; }
  /** Die-top centre of SM site i, in group space (includes the explode offset). */
  smPos(i: number) { const r = this.smRects[i]!; return new THREE.Vector3(r.x, Y.dieTop + this.layerDie.position.y, r.z); }
  gpcBox(g: number) {
    const r = gpcRect(g), { w, d } = DIM.die, y = Y.dieTop + this.layerDie.position.y;
    return new THREE.Box3(new THREE.Vector3(r.u0 * w - w / 2, y, r.v0 * d - d / 2), new THREE.Vector3(r.u1 * w - w / 2, y + 0.01, r.v1 * d - d / 2));
  }
  get l2Boxes(): THREE.Box3[] {
    const { core, l2 } = DIE_LAYOUT, { w, d } = DIM.die, y = Y.dieTop + this.layerDie.position.y;
    const mid = (core.u0 + core.u1) / 2;
    return [[core.u0, mid - l2.gap / 2], [mid + l2.gap / 2, core.u1]].map(([u0, u1]) =>
      new THREE.Box3(new THREE.Vector3(u0! * w - w / 2, y, l2.v0 * d - d / 2), new THREE.Vector3(u1! * w - w / 2, y + 0.01, l2.v1 * d - d / 2)));
  }
  get l2Box(): THREE.Box3 { const [a, b] = this.l2Boxes; return a!.clone().union(b!); }
  hbm(k: number) {
    const g = this.hbmGroups[k]!;
    const top = DIM.hbm.base + DIM.hbm.layers * DIM.hbm.layer;
    const pos = new THREE.Vector3(g.position.x, g.position.y + this.layerDie.position.y + top, g.position.z);
    const box = new THREE.Box3().setFromCenterAndSize(pos.clone().setY(pos.y - top / 2), new THREE.Vector3(DIM.hbm.w, top, DIM.hbm.d));
    return { pos, active: g.userData.active as boolean, box };
  }
  /** The 18 NVLink PHY positions on the die's south edge (group space). */
  get nvlinkPorts(): THREE.Vector3[] {
    const r = DIE_LAYOUT.nvlink, { w, d } = DIM.die, y = Y.dieTop + this.layerDie.position.y;
    return Array.from({ length: 18 }, (_, k) => new THREE.Vector3((r.u0 + (k + 0.5) * (r.u1 - r.u0) / 18) * w - w / 2, y, ((r.v0 + r.v1) / 2) * d - d / 2));
  }

  setSMActivity(fn: ((i: number) => number) | null) { this.smFn = fn; }
  setHBMActivity(fn: ((k: number) => number) | null) { this.hbmFn = fn; }

  explode(k: number) {
    this.explodeK = k = clamp(k);
    const e = k;
    this.layerSub.position.y = e * 14;
    this.layerInt.position.y = e * 26;
    this.layerDie.position.y = e * 38;
    this.hbmLayers.forEach((objs) => objs.forEach((o, i) => { o.position.y = (o.userData.y0 as number) + i * e * 1.2; }));
    this.bga.visible = this.c4.visible = this.ubump.visible = e > 0.02;
    // the balls/bumps rise with the layer above them only slightly, so they read as a gap filler
    this.bga.position.y = e * 5;
  }

  /** Default activity: music-driven, deterministic in t. */
  update(t: number, a?: AudioSample) {
    const fi = frameIdx(t) >> 2; // hold each flicker state for 4 frames
    const lvl = a ? 0.3 + 0.9 * clamp(a.rms * 1.2 + a.drums * 0.4) : 0.8;
    const kick = a ? a.kick : 0;
    for (let i = 0; i < 144; i++) {
      const r = this.smRects[i]!;
      let v: number;
      if (this.smFn) v = this.smFn(i);
      else if (!r.en) v = 0;
      else {
        const ph = 0.5 + 0.5 * Math.sin(t * 2.1 + r.g * 0.9 + r.s * 0.35);
        const fl = hash(i, fi) < 0.55 ? 1 : 0.25;
        v = lvl * (0.06 + 0.34 * ph * fl) + kick * 0.7 * (hash(i, fi, 7) < 0.3 ? 1 : 0);
      }
      const g = Math.max(0, v);
      this.smGlow.setColorAt(i, this.col.setRGB(LIN.signal[0] * g, LIN.signal[1] * g, LIN.signal[2] * g));
    }
    this.smGlow.instanceColor!.needsUpdate = true;
    for (let k = 0; k < 6; k++) {
      const active = this.hbmGroups[k]!.userData.active as boolean;
      let v = this.hbmFn ? this.hbmFn(k) : active ? lvl * (0.35 + 0.65 * (hash(k, fi, 3) < 0.6 ? 1 : 0.3)) : 0;
      if (!active) v = 0;
      const gm = this.hbmGlow[k]!.material as THREE.MeshBasicMaterial;
      gm.color.setRGB(LIN.ember[0] * v * 0.8, LIN.ember[1] * v * 0.8, LIN.ember[2] * v * 0.8);
      for (let j = 0; j < 2; j++) {
        const pk = (k < 3 ? 0 : 6) + (k % 3) * 2 + j;
        const pv = v * (0.6 + 0.4 * (hash(pk, fi, 5) < 0.5 ? 1 : 0.4)) * 0.8;
        this.phyGlow.setColorAt(pk, this.col.setRGB(LIN.ember[0] * pv, LIN.ember[1] * pv, LIN.ember[2] * pv));
      }
    }
    this.phyGlow.instanceColor!.needsUpdate = true;
  }

  /** The SM close-up placed on SM site i (created on first use). */
  smDetail(i: number): SMDetail {
    const r = this.smRects[i]!;
    if (!this.detail) this.detail = new SMDetail(r.w, r.d, 1);
    if (this.detailAt !== i) {
      this.detail.group.position.set(r.x, Y.dieTop, r.z);
      this.detail.group.scale.set(r.w / this.detail.w, 1, r.d / this.detail.d);
      if (!this.detail.group.parent) this.layerDie.add(this.detail.group);
      this.detailAt = i;
    }
    this.detail.group.visible = true;
    return this.detail;
  }
  hideSMDetail() { if (this.detail) this.detail.group.visible = false; }

  /**
   * A data path in group space: from the top of HBM stack k, down its die-facing side, across the
   * interposer to the die's memory PHY, along the die top to the L2 partition on that side, through the
   * crossbar gap, and out to SM site i. Manhattan routing, 0.03 mm above surfaces; follows explode().
   */
  pulsePath(k: number, i: number): THREE.CurvePath<THREE.Vector3> {
    const ly = this.layerDie.position.y, li = this.layerInt.position.y;
    const hb = this.hbmGroups[k]!, top = DIM.hbm.base + DIM.hbm.layers * DIM.hbm.layer;
    const sx = hb.position.x < 0 ? -1 : 1;
    const zc = hb.position.z;
    const yI = Y.intTop + 0.03 + li, yD = Y.dieTop + 0.03 + ly, yH = Y.dieBot + ly;
    const r = this.smRects[i]!;
    const [l2a, l2b] = this.l2Boxes;
    const l2 = sx < 0 ? l2a! : l2b!;
    const l2z = (l2.min.z + l2.max.z) / 2;
    const pts = [
      new THREE.Vector3(hb.position.x, yH + top + 0.05, zc),
      new THREE.Vector3(sx * (DIM.hbm.x - DIM.hbm.w / 2 - 0.05), yH + top * 0.5, zc),
      new THREE.Vector3(sx * (DIM.hbm.x - DIM.hbm.w / 2 - 0.3), yI, zc),
      new THREE.Vector3(sx * (DIM.die.w / 2 + 0.1), yI, zc),
      new THREE.Vector3(sx * (DIM.die.w / 2 - 0.9), yD, zc),
      new THREE.Vector3(sx * (DIM.die.w / 2 - 2.6), yD, zc),
      new THREE.Vector3(sx * (DIM.die.w / 2 - 2.6), yD, l2z),
      new THREE.Vector3(sx * 0.2, yD, l2z),
      new THREE.Vector3(0, yD, l2z),
      new THREE.Vector3(0, yD, r.z),
      new THREE.Vector3(r.x, yD, r.z),
    ];
    const cp = new THREE.CurvePath<THREE.Vector3>();
    for (let n = 0; n + 1 < pts.length; n++) if (pts[n]!.distanceTo(pts[n + 1]!) > 1e-4) cp.add(new THREE.LineCurve3(pts[n]!, pts[n + 1]!));
    return cp;
  }

  get explodeAmount() { return this.explodeK; }
}
