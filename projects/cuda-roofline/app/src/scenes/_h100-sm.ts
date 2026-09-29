// The SM close-up for the H100 model (_h100.ts): one GH100 Streaming Multiprocessor as a 3D block
// diagram raised off the silicon, laid out like NVIDIA's SM figure (H100 whitepaper, Figure 7):
// an L1 instruction cache strip, four partitions (processing blocks), each with an L0 i-cache, a warp
// scheduler, a dispatch unit, a register file, 16 INT32 + 32 FP32 + 16 FP64 lanes, one 4th-gen tensor
// core, and LD/ST + SFU units; below them the SM's 256 KB L1 data cache / shared memory and the TMA.
// Per-SM counts from the whitepaper tables: 128 FP32, 64 INT32, 64 FP64 cores, 4 tensor cores, 256 KB
// register file, 256 KB L1/shared. The LD/ST and SFU block counts are drawn generic (unlabelled).
//
// SM-local frame: x across the SM site (width w), z north→south (depth d), y up from the die top.
// Heights are exaggerated relief (a block diagram in 3D, not a physical cross-section).
import * as THREE from 'three';
import { LIN } from '../engine/palette';
import { makeSramTexture } from './_h100-tex';

export type LaneKind = 'int32' | 'fp32' | 'fp64';
export type SMUnit = 'l1i' | 'l0' | 'sched' | 'dispatch' | 'regfile' | 'tensor' | 'ldst' | 'sfu' | 'l1' | 'tma';

/** Additive glow that leaves destination alpha untouched. Plain AdditiveBlending also adds alpha, pushing it
 *  above 1, and the engine's premultiplied composite (ONE, ONE_MINUS_SRC_ALPHA) then subtracts whatever is
 *  already in `out` (the previous frame): an inverted teal ghost. */
export function glowMat(color = 0xffffff, extra: THREE.MeshBasicMaterialParameters = {}) {
  return new THREE.MeshBasicMaterial({
    color, transparent: true, depthWrite: false, ...extra,
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
  });
}

const lin = (k: keyof typeof LIN, s = 1) => new THREE.Color().setRGB(LIN[k][0] * s, LIN[k][1] * s, LIN[k][2] * s, THREE.LinearSRGBColorSpace);

interface Box { x0: number; x1: number; z0: number; z1: number; h: number }
interface UnitMesh { unit: SMUnit; p: number; glow: THREE.Mesh; box: Box }

let SRAM_TEX: THREE.CanvasTexture | null = null;

export class SMDetail {
  group = new THREE.Group();
  /** Everything that lights up (additive, HDR) sits in here; scenes can scale its opacity. */
  glowGroup = new THREE.Group();
  private units: UnitMesh[] = [];
  private lanes: THREE.InstancedMesh;
  private laneGlow: THREE.InstancedMesh;
  private laneMeta: { p: number; kind: LaneKind; idx: number; x: number; z: number }[] = [];
  private tcCubes: THREE.InstancedMesh;
  private tcGlow: THREE.InstancedMesh;
  private tcMeta: { p: number; i: number; j: number; k: number }[] = [];
  private tmp = new THREE.Object3D();
  private col = new THREE.Color();
  readonly boxes: Record<string, Box> = {};

  constructor(public w = 1.67, public d = 1.48, public relief = 1) {
    SRAM_TEX ??= makeSramTexture();
    const matBody = new THREE.MeshStandardMaterial({ color: lin('ink2'), metalness: 0.35, roughness: 0.5, envMapIntensity: 0.22, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    const matSram = new THREE.MeshStandardMaterial({ color: lin('ash'), map: SRAM_TEX, metalness: 0.3, roughness: 0.45, envMapIntensity: 0.22, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    matSram.color.multiplyScalar(0.4);
    const matEdge = new THREE.LineBasicMaterial({ color: lin('bone'), transparent: true, opacity: 0.4, depthWrite: false });
    const edgeVerts: number[] = [];
    const R = relief;

    // base slab
    const slab = new THREE.Mesh(new THREE.BoxGeometry(w, 0.006 * R, d), new THREE.MeshStandardMaterial({ color: lin('ink'), metalness: 0.2, roughness: 0.6, envMapIntensity: 0.2 }));
    slab.position.y = 0.003 * R;
    this.group.add(slab);
    this.addEdges(edgeVerts, { x0: -w / 2, x1: w / 2, z0: -d / 2, z1: d / 2, h: 0.006 * R }, 0);

    const X = (u: number) => -w / 2 + u * w, Z = (v: number) => -d / 2 + v * d;
    const block = (name: string, unit: SMUnit | null, p: number, u0: number, u1: number, v0: number, v1: number, h: number, sram = false) => {
      const b: Box = { x0: X(u0), x1: X(u1), z0: Z(v0), z1: Z(v1), h: h * R };
      const geo = new THREE.BoxGeometry(b.x1 - b.x0, b.h, b.z1 - b.z0);
      const mesh = new THREE.Mesh(geo, sram ? [matBody, matBody, matSram, matBody, matBody, matBody] : matBody);
      mesh.position.set((b.x0 + b.x1) / 2, 0.006 * R + b.h / 2, (b.z0 + b.z1) / 2);
      this.group.add(mesh);
      this.addEdges(edgeVerts, b, 0.006 * R);
      this.boxes[name] = b;
      if (unit) {
        const gm = new THREE.Mesh(new THREE.BoxGeometry(b.x1 - b.x0, b.h, b.z1 - b.z0).scale(1.01, 1.02, 1.01),
          glowMat(0x000000));
        gm.position.copy(mesh.position);
        gm.visible = false;
        this.glowGroup.add(gm);
        this.units.push({ unit, p, glow: gm, box: b });
      }
      return b;
    };

    // SM-level: L1 instruction cache (north strip), L1 data cache / shared memory + TMA (south band)
    block('l1i', 'l1i', -1, 0.06, 0.94, 0.015, 0.06, 0.028);
    block('l1', 'l1', -1, 0.05, 0.72, 0.82, 0.97, 0.07, true);
    block('tma', 'tma', -1, 0.76, 0.95, 0.82, 0.97, 0.05);

    // 4 partitions, 2 × 2
    const laneGeo = new THREE.BoxGeometry(1, 1, 1);
    const laneList: { p: number; kind: LaneKind; idx: number; x: number; z: number; sx: number; sz: number; h: number }[] = [];
    const tcList: { p: number; i: number; j: number; k: number; x: number; y: number; z: number; s: number }[] = [];
    for (let p = 0; p < 4; p++) {
      const pu0 = p & 1 ? 0.51 : 0.04, pu1 = pu0 + 0.45;
      const pv0 = p >> 1 ? 0.075 + 0.735 * 0.51 : 0.075 + 0.735 * 0.02, pv1 = pv0 + 0.735 * 0.47;
      const U = (a: number) => pu0 + a * (pu1 - pu0), V = (a: number) => pv0 + a * (pv1 - pv0);
      // partition floor (so each partition reads as one unit)
      block(`p${p}`, null, p, U(0), U(1), V(0), V(1), 0.004);
      block(`p${p}.l0`, 'l0', p, U(0.05), U(0.95), V(0.03), V(0.09), 0.02);
      block(`p${p}.sched`, 'sched', p, U(0.05), U(0.95), V(0.11), V(0.17), 0.04);
      block(`p${p}.dispatch`, 'dispatch', p, U(0.05), U(0.95), V(0.185), V(0.225), 0.03);
      block(`p${p}.regfile`, 'regfile', p, U(0.05), U(0.95), V(0.245), V(0.43), 0.075, true);
      // lanes: row 0 INT32 ×16, rows 1–2 FP32 ×32, row 3 FP64 ×16
      const rows: LaneKind[] = ['int32', 'fp32', 'fp32', 'fp64'];
      const lz0 = Z(V(0.46)), lz1 = Z(V(0.72)), rowH = (lz1 - lz0) / 4;
      const lx0 = X(U(0.05)), lx1 = X(U(0.95)), colW = (lx1 - lx0) / 16;
      let fpIdx = 0;
      rows.forEach((kind, r) => {
        for (let c = 0; c < 16; c++) {
          const idx = kind === 'fp32' ? fpIdx++ : c;
          laneList.push({ p, kind, idx, x: lx0 + (c + 0.5) * colW, z: lz0 + (r + 0.5) * rowH, sx: colW * 0.72, sz: rowH * 0.7, h: (kind === 'fp32' ? 0.05 : 0.04) * R });
        }
      });
      // tensor core: a shell with a 4 × 4 × 4 lattice of MAC cells inside
      const tb = block(`p${p}.tensor`, 'tensor', p, U(0.05), U(0.55), V(0.75), V(0.97), 0.02);
      const tcx0 = tb.x0 + 0.015, tcx1 = tb.x1 - 0.015, tcz0 = tb.z0 + 0.01, tcz1 = tb.z1 - 0.01;
      const cs = Math.min((tcx1 - tcx0) / 4, (tcz1 - tcz0) / 4);
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) {
        tcList.push({ p, i, j, k, x: (tcx0 + tcx1) / 2 + (i - 1.5) * cs, z: (tcz0 + tcz1) / 2 + (j - 1.5) * cs, y: 0.006 * R + tb.h + (k + 0.5) * cs * 0.9, s: cs * 0.62 });
      }
      block(`p${p}.ldst`, 'ldst', p, U(0.6), U(0.95), V(0.75), V(0.85), 0.035);
      block(`p${p}.sfu`, 'sfu', p, U(0.6), U(0.95), V(0.87), V(0.97), 0.03);
    }

    // lanes (instanced) + their additive glow twins
    const matLane = new THREE.MeshStandardMaterial({ color: lin('graphite', 0.4), metalness: 0.4, roughness: 0.5, envMapIntensity: 0.2 });
    matLane.color.multiplyScalar(0.6);
    this.lanes = new THREE.InstancedMesh(laneGeo, matLane, laneList.length);
    this.laneGlow = new THREE.InstancedMesh(laneGeo, glowMat(), laneList.length);
    laneList.forEach((l, n) => {
      this.tmp.position.set(l.x, 0.006 * R + l.h / 2, l.z);
      this.tmp.scale.set(l.sx, l.h, l.sz);
      this.tmp.updateMatrix();
      this.lanes.setMatrixAt(n, this.tmp.matrix);
      this.tmp.scale.set(l.sx * 1.06, l.h * 1.03, l.sz * 1.06);
      this.tmp.updateMatrix();
      this.laneGlow.setMatrixAt(n, this.tmp.matrix);
      this.laneGlow.setColorAt(n, this.col.setRGB(0, 0, 0));
      this.laneMeta.push({ p: l.p, kind: l.kind, idx: l.idx, x: l.x, z: l.z });
      this.addEdges(edgeVerts, { x0: l.x - l.sx / 2, x1: l.x + l.sx / 2, z0: l.z - l.sz / 2, z1: l.z + l.sz / 2, h: l.h }, 0.006 * R, true);
    });
    this.group.add(this.lanes);
    this.glowGroup.add(this.laneGlow);

    const cubeGeo = new THREE.BoxGeometry(1, 1, 1);
    this.tcCubes = new THREE.InstancedMesh(cubeGeo, new THREE.MeshStandardMaterial({ color: lin('ash'), metalness: 0.5, roughness: 0.35, envMapIntensity: 0.25 }), tcList.length);
    this.tcGlow = new THREE.InstancedMesh(cubeGeo, glowMat(), tcList.length);
    (this.tcCubes.material as THREE.MeshStandardMaterial).color.multiplyScalar(0.35);
    tcList.forEach((c, n) => {
      this.tmp.position.set(c.x, c.y, c.z);
      this.tmp.scale.setScalar(c.s);
      this.tmp.updateMatrix();
      this.tcCubes.setMatrixAt(n, this.tmp.matrix);
      this.tmp.scale.setScalar(c.s * 1.12);
      this.tmp.updateMatrix();
      this.tcGlow.setMatrixAt(n, this.tmp.matrix);
      this.tcGlow.setColorAt(n, this.col.setRGB(0, 0, 0));
      this.tcMeta.push({ p: c.p, i: c.i, j: c.j, k: c.k });
    });
    this.group.add(this.tcCubes);
    this.glowGroup.add(this.tcGlow);

    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.Float32BufferAttribute(edgeVerts, 3));
    this.group.add(new THREE.LineSegments(eg, matEdge));
    this.group.add(this.glowGroup);
  }

  /** 12 box edges (or only the top rectangle for tiny lanes) into a merged LineSegments buffer. */
  private addEdges(out: number[], b: Box, y0: number, topOnly = false) {
    const y1 = y0 + b.h;
    const c = [[b.x0, b.z0], [b.x1, b.z0], [b.x1, b.z1], [b.x0, b.z1]] as const;
    for (let i = 0; i < 4; i++) {
      const [ax, az] = c[i]!, [bx, bz] = c[(i + 1) % 4]!;
      out.push(ax, y1, az, bx, y1, bz);
      if (!topOnly) { out.push(ax, y0, az, bx, y0, bz); out.push(ax, y0, az, ax, y1, az); }
    }
  }

  /** Light the 64 lanes of each partition: fn(partition 0–3, kind, index within kind) → 0..1 (HDR ok). */
  setLanes(fn: (p: number, kind: LaneKind, idx: number) => number, color: [number, number, number] = LIN.signal, gain = 1.3) {
    this.laneMeta.forEach((l, n) => {
      const a = Math.max(0, fn(l.p, l.kind, l.idx)) * gain;
      this.laneGlow.setColorAt(n, this.col.setRGB(color[0] * a, color[1] * a, color[2] * a));
    });
    this.laneGlow.instanceColor!.needsUpdate = true;
  }

  /** Light the tensor cores' 4×4×4 MAC lattice: fn(partition, i, j, k) → 0..1. */
  setTensor(fn: (p: number, i: number, j: number, k: number) => number, color: [number, number, number] = LIN.ember, gain = 1.4) {
    this.tcMeta.forEach((c, n) => {
      const a = Math.max(0, fn(c.p, c.i, c.j, c.k)) * gain;
      this.tcGlow.setColorAt(n, this.col.setRGB(color[0] * a, color[1] * a, color[2] * a));
    });
    this.tcGlow.instanceColor!.needsUpdate = true;
  }

  /** Glow whole units: fn(unit, partition (−1 for SM-level units)) → 0..1. */
  setUnits(fn: (unit: SMUnit, p: number) => number, color: [number, number, number] = LIN.signal, gain = 0.7) {
    for (const u of this.units) {
      const a = Math.max(0, fn(u.unit, u.p)) * gain;
      u.glow.visible = a > 1e-3;
      (u.glow.material as THREE.MeshBasicMaterial).color.setRGB(color[0] * a, color[1] * a, color[2] * a);
    }
  }

  /** SM-local top-centre of a unit (for labels, camera targets, pulse endpoints). */
  anchor(name: string): THREE.Vector3 {
    const b = this.boxes[name];
    if (!b) throw new Error(`SMDetail: no block ${name}`);
    return new THREE.Vector3((b.x0 + b.x1) / 2, 0.006 * this.relief + b.h, (b.z0 + b.z1) / 2);
  }

  /** SM-local position of lane `idx` of `kind` in partition p (top of the pillar). */
  lanePos(p: number, kind: LaneKind, idx: number): THREE.Vector3 {
    const l = this.laneMeta.find((m) => m.p === p && m.kind === kind && m.idx === idx);
    if (!l) throw new Error('SMDetail: no such lane');
    return new THREE.Vector3(l.x, 0.006 * this.relief + (kind === 'fp32' ? 0.05 : 0.04) * this.relief, l.z);
  }
}
