// ROOFLINE ×4 (v5 chorus, 3D). One idea: a kernel's speed is capped either by memory (the slope) or by
// math (the flat roof), and doing more math per byte moves it right, where the roof is higher.
//
// The v4 lesson, built as a structure standing over the GH100 die (shared `_h100.ts`, SMs flickering
// with the music underneath): luminous rails for the axes (x = "math per byte", y = "speed", log-log),
// the roofline as an architectural roof you could stand under (a sloped memory roof and a flat compute
// roof, 20 mm deep, on thin pillars), and the kernel as a glowing orb that climbs, earlier kernels left
// behind as rings. Roofs are the book's RTX 3090 numbers (CH6: 936 GB/s DRAM, 35.6 TFLOP/s FP16 on CUDA
// cores; ridge ≈ 38 FLOP/byte). Each chorus plots one rung of CH6's GEMM ladder (naive 0.3, 1D
// blocktiling 9.3, 2D blocktiling 25.1, vectorized 27.8 TFLOP/s = ACHIEVED / PEAK 0.01 / 0.26 / 0.70 /
// 0.78, landing on "ceiling"). x positions are illustrative (no intensity is printed).
//   line 1 "I'm chasing the roofline,": I'm = the orb, chasing = it takes its place, roofline = the roof
//          (n = 1 builds it; later ones flash it). ROOF / LINE stand on the roof for one beat.
//   line 2 "Memory's the slope and the math's the line,": the slope lights "memory speed limit", the
//          flat roof lights "math speed limit"; the camera frames each in turn.
//   line 3 "Squeeze more math from every byte,": the orb steps right on the four stressed words; a
//          dashed cap shows the roof above it; the camera tracks it (n = 3 swings across the ridge).
//   line 4 "Till I'm hitting the ceiling tonight": the orb lifts on "hitting", ACHIEVED / PEAK lands on
//          "ceiling" (n = 4: the orb breaks through in light).
// Choruses 1 and 2 hold an instrumental recap after the last line. The camera never stops (orbit and
// crane that surge with the drums via _cam.drive), its big moves land on the beat (_lock.q); the
// lyric, labels and readout are screen-space overlays so they stay big and crisp.
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { HEX, LIN, rgba } from '../engine/palette';
import { F, font, fitSize, glyphX, measure } from '../engine/type';
import { PDoom, drawReadout, formatPDoom } from '../engine/hud';
import type { Line, Word } from '../engine/lyrics';
import { clamp, ease, lerp } from '../engine/util';
import { hit, q, snap } from './_lock';
import { drive, kickPush } from './_cam';
import { H100, Pulses, glowMat, type PulseItem } from './_h100';

// ---- the chart in log space (as v4) and its roofs (book numbers, CH6)
const LG = { lo2: -3, hi2: 11, lo10: 11, hi10: 14 };
const BW = 936e9; // B/s, RTX 3090 DRAM (CH6)
const PEAK = 35.6e12; // FLOP/s, RTX 3090 FP16 on CUDA cores (CH6)
const RIDGE = PEAK / BW; // ≈ 38 FLOP/byte
const lx = (ai: number) => Math.log2(ai);
const ly = (fl: number) => Math.log10(fl);
const roofL10 = (l2: number) => ly(Math.min(PEAK, BW * 2 ** l2));

// ---- the structure in mm (y up, die top ≈ 3.4 mm; the chart stands on the module)
const CX0 = -62, CX1 = 62, CY0 = 10, CY1 = 74, DEPTH = 20;
const X3 = (l2: number) => CX0 + ((l2 - LG.lo2) / (LG.hi2 - LG.lo2)) * (CX1 - CX0);
const Y3 = (l10: number) => CY0 + ((l10 - LG.lo10) / (LG.hi10 - LG.lo10)) * (CY1 - CY0);
const PS = new THREE.Vector3(X3(LG.lo2), Y3(roofL10(LG.lo2)), 0); // slope start (left edge)
const PR = new THREE.Vector3(X3(lx(RIDGE)), Y3(ly(PEAK)), 0); // ridge
const PE = new THREE.Vector3(CX1, Y3(ly(PEAK)), 0); // flat end
const CENTER = new THREE.Vector3(0, 38, 0);
const ZF = DEPTH / 2 + 1.5; // the kernel's plane: just in front of the roof's front face, so the slab never hides the orb
const RO3 = new THREE.Vector3(CX0 + 0.727 * (CX1 - CX0), CY0 + 0.13 * (CY1 - CY0), 0); // readout anchor (v4's place)

/** CH6's GEMM ladder, one rung per chorus (x: illustrative math-per-byte position). */
const RUNG = [
  { name: 'naive kernel', short: 'naive', x: 0.43, tf: 0.3 },
  { name: '1D tiling', short: '1D tiles', x: 16, tf: 9.3 },
  { name: '2D tiling', short: '2D tiles', x: 90, tf: 25.1 },
  { name: 'vectorized loads', short: 'vectorized', x: 180, tf: 27.8 },
];
const START = { x: 0.16, tf: 0.13 };
const RECAP: Record<number, [string, string]> = {
  1: ['your first kernel · 0.01 of peak', 'under the slope: memory is the limit'],
  2: ['1D tiling · 0.26 of peak', 'still under the slope: still waiting on memory'],
};
const CAPTION3 = 'past the ridge: now the math is the limit';
const LYR = { x: 150, base: 985, maxW: 1620, maxSize: 68 };

type Col = keyof typeof HEX;
type Pt = { x: number; y: number; ok: boolean };
interface CamShot { az: number; el: number; dist: number; target: THREE.Vector3; fov?: number }

const linC = (k: keyof typeof LIN, s = 1) => new THREE.Color().setRGB(LIN[k][0] * s, LIN[k][1] * s, LIN[k][2] * s, THREE.LinearSRGBColorSpace);

export default class Roofline extends Scene {
  n = 1;
  scene3 = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(32, W / H, 0.5, 5000);
  ms = makeRT(W, H, { samples: 4 });
  gpu!: H100;
  ui = new Layer2D();
  pd!: PDoom;
  lines: Line[] = [];
  lyrSize = 64;
  f = { lyr: F.archivo(100, 800), big: F.archivo(100, 900), mono: F.mono(400), monoM: F.mono(500) };
  T = {
    im: 0, chase: 0, roof: 0,
    mem: 0, slope: 0, math: 0, line: 0,
    sq: [0, 0, 0, 0] as number[],
    hitting: 0, ceil: 0, tonight: 0, lyrEnd: 0,
    recap0: 0, recap1: 0,
  };
  from = { l2: 0, l10: 0 };
  to = { l2: 0, l10: 0 };
  vPrev = 0;
  vNow = 0;

  // 3D parts
  chart = new THREE.Group();
  slope!: { slab: THREE.Mesh; edge: THREE.Mesh; mat: THREE.MeshStandardMaterial; edgeMat: THREE.MeshBasicMaterial };
  flat!: { slab: THREE.Mesh; edge: THREE.Mesh; mat: THREE.MeshStandardMaterial; edgeMat: THREE.MeshBasicMaterial };
  railMat = new THREE.MeshBasicMaterial({ color: linC('bone', 0.5) });
  pillars!: THREE.InstancedMesh;
  gridMat = new THREE.LineBasicMaterial({ color: linC('bone', 1), transparent: true, opacity: 0.11, depthWrite: false });
  orbCore!: THREE.Mesh;
  halos: THREE.Mesh[] = [];
  haloR = 3.2;
  shocks: THREE.Mesh[] = [];
  rays = new THREE.Group();
  orbLight = new THREE.PointLight(linC('signal', 1), 0, 0, 2);
  rings: THREE.Mesh[] = [];
  capLine!: THREE.Line;
  capMat = new THREE.LineDashedMaterial({ color: linC('bone', 0.9), dashSize: 1.6, gapSize: 1.4, transparent: true, opacity: 0.6 });
  tick!: THREE.Mesh;
  flow = new Pulses(256);
  shots: { t: number; fn: (t: number) => CamShot }[] = [];

  override async init() {
    const { lyrics, params, start, end, audio: au } = this.ctx;
    this.n = Number(params.n ?? 1);
    const n = this.n;
    this.pd = new PDoom(lyrics);
    const l1 = lyrics.linesIn(start - 0.3, end).find((l) => /chasing the roofline/i.test(l.text)) ?? lyrics.get("I'm chasing", n - 1);
    this.lines = [0, 1, 2, 3].map((k) => lyrics.lines[l1.i + k]!);
    const [L1, L2, L3, L4] = this.lines as [Line, Line, Line, Line];
    const w = (l: Line, re: RegExp, fb: number) => (l.words.find((x) => re.test(x.w)) ?? l.words[fb]!).start;
    const T = this.T;
    T.im = L1.words[0]!.start; T.chase = w(L1, /chasing/i, 1); T.roof = w(L1, /roofline/i, 3);
    T.mem = w(L2, /memory/i, 0); T.slope = w(L2, /slope/i, 2); T.math = w(L2, /math/i, 5); T.line = w(L2, /^line/i, 7);
    T.sq = [w(L3, /squeeze/i, 0), w(L3, /^math/i, 2), w(L3, /every/i, 4), w(L3, /byte/i, 5)];
    T.hitting = w(L4, /hitting/i, 2); T.ceil = w(L4, /ceiling/i, 4); T.tonight = w(L4, /tonight/i, 5);
    const lastW = L4.words[L4.words.length - 1]!;
    const brk = au.sections.find((s) => s.start >= lastW.start - 0.5 && s.start < end - 0.5 && /break/.test(s.name));
    T.lyrEnd = brk ? brk.start : end + 1;
    T.recap0 = brk ? brk.start : end + 1;
    const db = au.downbeats.find((d) => d > T.recap0 + 0.5) ?? T.recap0 + 1.8;
    T.recap1 = brk ? db : end + 1;
    const prev = n === 1 ? START : RUNG[n - 2]!;
    const cur = RUNG[n - 1]!;
    this.from = { l2: lx(prev.x), l10: ly(prev.tf * 1e12) };
    this.to = { l2: lx(cur.x), l10: ly(cur.tf * 1e12) };
    this.vPrev = this.pd.steps[n - 1]?.v ?? 0;
    this.vNow = this.pd.steps[n]?.v ?? this.vPrev;
    this.lyrSize = Math.min(LYR.maxSize, ...this.lines.map((l) => fitSize(l.text, this.f.lyr, LYR.maxW, LYR.maxSize)));

    // ---- the world
    this.gpu = new H100({ renderer: this.ctx.renderer });
    this.scene3.add(this.gpu.group, H100.rig());
    this.scene3.environment = this.gpu.env;
    this.scene3.background = linC('ink', 1);
    this.scene3.fog = new THREE.Fog(linC('ink', 1), 260, 700);
    this.buildChart();
    this.scene3.add(this.chart, this.flow.mesh);
    this.buildShots();
  }

  // ------------------------------------------------------------------ build
  private buildChart() {
    const g = this.chart;
    // rails (axes) with arrowheads
    const railX = new THREE.Mesh(new THREE.BoxGeometry(CX1 - CX0 + 8, 0.7, 0.7), this.railMat);
    railX.position.set((CX0 + CX1) / 2 + 4, CY0, 0);
    const railY = new THREE.Mesh(new THREE.BoxGeometry(0.7, CY1 - CY0, 0.7), this.railMat);
    railY.position.set(CX0, (CY0 + CY1) / 2, 0);
    const cone = new THREE.ConeGeometry(1.6, 4, 16);
    const ax = new THREE.Mesh(cone, this.railMat); ax.rotation.z = -Math.PI / 2; ax.position.set(CX1 + 9, CY0, 0);
    const ay = new THREE.Mesh(cone, this.railMat); ay.position.set(CX0, CY1 + 1, 0);
    g.add(railX, railY, ax, ay);
    // hairline grid on the back plane and the floor of the structure
    const pts: number[] = [];
    for (let e = LG.lo10 + 1; e < LG.hi10; e++) pts.push(CX0, Y3(e), -DEPTH / 2, CX1, Y3(e), -DEPTH / 2);
    for (let e = LG.lo2 + 2; e < LG.hi2; e += 2) pts.push(X3(e), CY0, -DEPTH / 2, X3(e), CY1, -DEPTH / 2);
    for (let e = LG.lo2 + 2; e < LG.hi2; e += 2) pts.push(X3(e), CY0, -DEPTH / 2, X3(e), CY0, DEPTH / 2);
    const gg = new THREE.BufferGeometry(); gg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    g.add(new THREE.LineSegments(gg, this.gridMat));
    // the roof: two slabs with a glowing front edge
    const seg = (a: THREE.Vector3, b: THREE.Vector3) => {
      const mat = new THREE.MeshStandardMaterial({ color: linC('graphite', 0.3), metalness: 0.9, roughness: 0.32, emissive: linC('blood', 1), emissiveIntensity: 0, envMapIntensity: 0.4 });
      const slab = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat);
      const edgeMat = glowMat(0xffffff);
      const edge = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), edgeMat);
      const holder = new THREE.Group();
      holder.add(slab, edge);
      const L = a.distanceTo(b);
      slab.scale.set(L, 1.4, DEPTH); slab.position.set(L / 2, 0.7, 0);
      edge.scale.set(L, 0.6, 0.6); edge.position.set(L / 2, 0.2, DEPTH / 2 + 0.35);
      holder.position.copy(a);
      holder.rotation.z = Math.atan2(b.y - a.y, b.x - a.x);
      holder.userData.L = L;
      g.add(holder);
      return { slab, edge, mat, edgeMat, holder };
    };
    this.slope = seg(PS, PR);
    this.flat = seg(PR, PE);
    // pillars under the roof, both faces, every 2 steps of math per byte
    const xs: number[] = [];
    for (let e = LG.lo2 + 2; e < LG.hi2; e += 2) xs.push(X3(e));
    xs.push(PR.x);
    this.pillars = new THREE.InstancedMesh(new THREE.BoxGeometry(0.45, 1, 0.45), new THREE.MeshStandardMaterial({ color: linC('graphite', 0.5), metalness: 0.7, roughness: 0.45 }), xs.length * 2);
    const m = new THREE.Object3D();
    let k = 0;
    for (const x of xs) {
      const top = Y3(roofL10((x - CX0) / (CX1 - CX0) * (LG.hi2 - LG.lo2) + LG.lo2));
      for (const z of [-DEPTH / 2 + 0.6, DEPTH / 2 - 0.6]) {
        m.position.set(x, (CY0 + top) / 2, z); m.scale.set(1, top - CY0, 1); m.updateMatrix();
        this.pillars.setMatrixAt(k++, m.matrix);
      }
    }
    g.add(this.pillars);
    // the orb (core, halo, burst) and its light
    this.orbCore = new THREE.Mesh(new THREE.SphereGeometry(1.5, 24, 16), new THREE.MeshBasicMaterial({ color: linC('bone', 3) }));
    const quad = new THREE.PlaneGeometry(2, 2);
    for (let i = 0; i < 2; i++) this.halos.push(new THREE.Mesh(quad, glowMat(0xffffff, { map: glowTex() })));
    const tor = new THREE.TorusGeometry(1, 0.035, 6, 64);
    for (let i = 0; i < 3; i++) this.shocks.push(new THREE.Mesh(tor, glowMat(0xffffff)));
    const ray = new THREE.PlaneGeometry(1, 1).translate(0.5, 0, 0);
    for (let i = 0; i < 14; i++) {
      const r = new THREE.Mesh(ray, glowMat(0xffffff, { map: rayTex() }));
      r.rotation.z = (i / 14) * Math.PI * 2 + 0.3 * hash(i + 20);
      this.rays.add(r);
    }
    g.add(this.orbCore, ...this.halos, ...this.shocks, this.rays, this.orbLight);
    // earlier rungs as rings
    for (let j = 0; j < 3; j++) {
      const r = new THREE.Mesh(new THREE.TorusGeometry(3, 0.22, 8, 48), glowMat(0xffffff));
      r.position.set(X3(lx(RUNG[j]!.x)), Y3(ly(RUNG[j]!.tf * 1e12)), ZF);
      g.add(r); this.rings.push(r);
    }
    // dashed cap from the orb to the roof above it, and a tick on the x rail
    const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 1, 0], 3));
    this.capLine = new THREE.Line(cg, this.capMat);
    this.tick = new THREE.Mesh(new THREE.BoxGeometry(0.9, 5, 0.9), new THREE.MeshBasicMaterial({ color: linC('signal', 2.2) }));
    g.add(this.capLine, this.tick);
  }

  /** Camera keys per chorus: each lands on the grid point of its word; between them the move eases. */
  private buildShots() {
    const au = this.ctx.audio, T = this.T, n = this.n;
    const Q = (t: number) => q(au, t);
    const S = (az: number, el: number, dist: number, target: THREE.Vector3 | ((t: number) => THREE.Vector3), fov?: number) =>
      (t: number): CamShot => ({ az, el, dist, target: typeof target === 'function' ? target(t) : target.clone(), fov });
    const dotT = (dy = 6) => (t: number) => { const d = this.dot3(t); return new THREE.Vector3(d.x * 0.75, d.y * 0.6 + CENTER.y * 0.4 + dy, 0); };
    const wide = (az: number, el = 0.14, dist = 168) => S(az, el, dist, CENTER);
    const slopeT = new THREE.Vector3((PS.x + PR.x) / 2 - 2, (PS.y + PR.y) / 2 - 3, 0);
    const flatT = new THREE.Vector3((PR.x + PE.x) / 2 - 10, PR.y - 18, 0);
    const k: { t: number; fn: (t: number) => CamShot }[] = [];
    const st = this.ctx.start;
    if (n === 1) {
      k.push({ t: st, fn: S(-0.5, 0.04, 62, new THREE.Vector3(CX0 + 14, 20, 0)) });
      k.push({ t: Q(T.im), fn: S(-0.42, 0.06, 70, new THREE.Vector3(CX0 + 16, 22, 0)) });
      k.push({ t: Q(T.roof), fn: wide(-0.14) });
    } else if (n === 2) {
      k.push({ t: st, fn: S(0.62, 0.3, 120, new THREE.Vector3(20, 44, 0)) });
      k.push({ t: Q(T.chase), fn: S(0.3, 0.2, 150, CENTER) });
      k.push({ t: Q(T.roof), fn: wide(0.1) });
    } else if (n === 3) {
      const r = new THREE.Vector3(X3(lx(RUNG[1]!.x)), Y3(ly(RUNG[1]!.tf * 1e12)), 0);
      k.push({ t: st, fn: S(-0.1, 0.1, 48, r) });
      k.push({ t: Q(T.chase), fn: S(-0.25, 0.12, 90, r.clone().lerp(CENTER, 0.5)) });
      k.push({ t: Q(T.roof), fn: wide(-0.18, 0.16, 172) });
    } else {
      k.push({ t: st, fn: S(-1.25, 0.34, 110, CENTER) });
      k.push({ t: Q(T.im) + 0.25, fn: S(-0.3, 0.18, 150, CENTER) });
      k.push({ t: Q(T.roof), fn: wide(0.05, 0.12, 160) });
    }
    // line 2: the slope, then the flat roof
    k.push({ t: Q(T.slope), fn: S(-0.3, 0.1, 128, slopeT) });
    k.push({ t: Q(T.line), fn: S(0.28, 0.16, 128, flatT) });
    // line 3: track the orb (n = 3 swings hard across the ridge)
    // n = 1 careful, n = 2 wider, n = 3 swings from low-left across the ridge to high-right, n = 4 zig-zags
    const swing = [[-0.18, -0.06, 0.06, 0.16], [-0.26, -0.06, 0.1, 0.26], [-0.55, -0.2, 0.32, 0.62], [-0.42, 0.3, -0.25, 0.45]][n - 1]!;
    const lift = n === 3 ? [0.0, 0.08, 0.2, 0.3] : n === 4 ? [0.18, 0.02, 0.2, 0.06] : [0.09, 0.09, 0.09, 0.09];
    const dsq = n === 3 ? [96, 100, 94, 104] : n === 4 ? [98, 92, 100, 94] : [116, 116, 116, 116];
    T.sq.forEach((a, i) => k.push({ t: Q(a), fn: S(swing[i]!, lift[i]!, dsq[i]!, dotT(4)) }));
    // line 4: crane up with the orb on "hitting", pull back to show the readout on "ceiling"
    k.push({ t: Q(T.hitting), fn: S(swing[3]! * 0.6, 0.26, 112, dotT(0)) });
    k.push({ t: Q(T.ceil), fn: wide(n === 4 ? -0.12 : 0.08, 0.15, n === 4 ? 150 : 168) });
    if (n === 4) k.push({ t: Q(T.tonight) + 1.4, fn: S(0.2, 0.2, 96, dotT(-4)) });
    // the break recap: a slow wide orbit
    // the break recap: a slow wide orbit that keeps travelling
    if (T.recap0 < this.ctx.end) {
      const r0 = T.recap0 + 0.9;
      k.push({ t: r0, fn: (t: number) => ({ az: 0.22 + 0.06 * Math.max(0, t - r0), el: 0.2 + 0.012 * Math.max(0, t - r0), dist: 180, target: CENTER.clone() }) });
    }
    k.push({ t: this.ctx.end + 1, fn: k[k.length - 1]!.fn });
    this.shots = k.sort((a, b) => a.t - b.t);
  }

  // ------------------------------------------------------------------ state at t
  private s(t: number, at: number, dur = 0.09) { return snap(this.ctx.audio, t, at, dur); }
  private h(t: number, at: number, hl = 0.09) { return hit(this.ctx.audio, t, at, hl); }

  private dot(t: number): { l2: number; l10: number; shown: number } {
    const T = this.T, n = this.n;
    let l2 = this.from.l2;
    for (let k = 0; k < 4; k++) l2 = lerp(l2, this.from.l2 + ((this.to.l2 - this.from.l2) * (k + 1)) / 4, this.s(t, T.sq[k]!));
    const l10 = lerp(this.from.l10, this.to.l10, this.s(t, T.hitting));
    if (n === 1) {
      const k = this.s(t, T.chase);
      return { l2: lerp(LG.lo2 + 0.25, l2, k), l10: lerp(LG.lo10 + 0.12, l10, k), shown: this.s(t, T.im, 0.06) };
    }
    return { l2, l10, shown: 1 };
  }
  private dot3(t: number) { const d = this.dot(t); return new THREE.Vector3(X3(d.l2), Y3(d.l10), ZF); }

  private camShot(t: number): CamShot {
    const k = this.shots;
    let i = 0;
    while (i + 1 < k.length && k[i + 1]!.t <= t) i++;
    const a = k[i]!, b = k[Math.min(i + 1, k.length - 1)]!;
    const A = a.fn(t);
    if (b === a || t <= a.t) return A;
    const B = b.fn(t);
    const u = ease.inOutCubic(clamp((t - a.t) / Math.max(1e-3, b.t - a.t)));
    return { az: lerp(A.az, B.az, u), el: lerp(A.el, B.el, u), dist: lerp(A.dist, B.dist, u), target: A.target.clone().lerp(B.target, u) };
  }

  // ------------------------------------------------------------------ render
  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const t = f.t, n = this.n, T = this.T;
    const { renderer, comp, audio: au } = this.ctx;
    this.gpu.update(t, f.a);

    // ---- camera: keyed shot + a continuous orbit/crane that surges with the drums
    const sh = this.camShot(t);
    const energy = [0.07, 0.1, 0.13, 0.18][n - 1]!;
    const dv = drive(au, t, 1, 2.2, this.ctx.start);
    const az = sh.az + energy * Math.sin(dv * 0.55);
    const el = sh.el + energy * 0.4 * Math.sin(dv * 0.37 + 1.3);
    const dist = sh.dist * (1 - 0.03 * kickPush(au, t) * (n === 4 ? 1.6 : 1) + 0.03 * Math.sin(dv * 0.29 + 0.7));
    const cam = this.cam;
    const tg = sh.target;
    cam.position.set(tg.x + Math.sin(az) * Math.cos(el) * dist, tg.y + Math.sin(el) * dist, tg.z + Math.cos(az) * Math.cos(el) * dist);
    cam.up.set(0, 1, 0);
    cam.lookAt(tg);
    // the subject sits a little above centre so the lyric band below stays clear
    cam.setViewOffset(W, H, 0, H * 0.07, W, H);
    H100.fitClip(cam, dist);
    cam.far = 2000; cam.updateProjectionMatrix();

    // ---- chart state
    this.updateRoof(t);
    const d = this.dot(t);
    const p3 = new THREE.Vector3(X3(d.l2), Y3(d.l10), ZF);
    this.updateOrb(t, p3, d.shown);
    this.updateRings(t);
    this.updateCap(t, p3, d.l2);
    this.updateFlow(t, dv);

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene3, cam);
    comp.draw(renderer, this.ms.texture, out);

    // ---- overlay
    this.chart.updateMatrixWorld(true);
    cam.updateMatrixWorld(true);
    const c = this.ui.ctx;
    this.ui.clear();
    c.textBaseline = 'alphabetic';
    // a dark band under the lyric so it reads over the board
    const sg = c.createLinearGradient(0, 820, 0, H);
    sg.addColorStop(0, rgba('ink', 0)); sg.addColorStop(0.45, rgba('ink', 0.6)); sg.addColorStop(1, rgba('ink', 0.82));
    c.fillStyle = sg; c.fillRect(0, 820, W, H - 820);
    c.shadowColor = 'rgba(0,0,0,0.85)'; c.shadowBlur = 14;
    this.drawRoofLabels(c, t);
    this.drawAxisTitles(c, t);
    this.drawRungLabels(c, t);
    this.drawDotLabel(c, t, p3, d.shown);
    this.drawReadoutAt(c, t);
    this.drawRecap(c, t);
    this.drawLyric(c, t);
    comp.draw(renderer, this.ui.upload(), out);

    const lift = this.h(t, T.hitting, 0.07);
    const big = [0.8, 1, 1.15, 1.6][n - 1]!;
    return {
      bloomThreshold: 0.85, bloomKnee: 0.4, bloomRadius: 0.7,
      bloom: 0.5 + 0.12 * n + 0.45 * big * this.h(t, T.ceil, 0.3),
      ca: 1.0 + 0.3 * n, vignette: 0.42, grain: 0.05, halation: 0.3,
      zoom: 1 + 0.014 * big * lift,
    };
  }

  // ------------------------------------------------------------------ 3D updates
  private updateRoof(t: number) {
    const T = this.T, n = this.n;
    const on = n === 1 ? this.s(t, T.roof, 0.32) : 1;
    const flash = this.h(t, T.roof, 0.14);
    const slopeLit = n === 1 ? this.s(t, T.slope) : Math.max(0.55, this.s(t, T.slope));
    const flatLit = n === 1 ? this.s(t, T.line) : Math.max(0.55, this.s(t, T.line));
    const slopeHot = this.h(t, T.slope, 0.12), flatHot = this.h(t, T.line, 0.12);
    const drawS = n === 1 ? clamp(on * 1.6) : 1, drawF = n === 1 ? clamp(on * 1.6 - 0.6) : 1;
    const apply = (seg: typeof this.slope, draw: number, lit: number, hot: number) => {
      const holder = seg.slab.parent!;
      const L = holder.userData.L as number;
      holder.visible = draw > 0.001;
      seg.slab.scale.x = Math.max(0.001, L * draw); seg.slab.position.x = (L * draw) / 2;
      seg.edge.scale.x = Math.max(0.001, L * draw); seg.edge.position.x = (L * draw) / 2;
      seg.mat.emissiveIntensity = 0.05 * lit + 0.35 * hot + 0.2 * flash;
      const gb = 0.9 + 1.3 * lit + 2.5 * hot + 1.5 * flash;
      const col = new THREE.Color().copy(linC('bone', 1)).lerp(linC('signal', 1), clamp(lit));
      seg.edgeMat.color.setRGB(col.r * gb, col.g * gb, col.b * gb);
    };
    apply(this.slope, drawS, slopeLit, slopeHot);
    apply(this.flat, drawF, flatLit, flatHot);
    this.pillars.visible = on > 0.5;
    const railK = n === 1 ? 0.35 + 0.35 * this.s(t, T.im) : 0.7;
    const xHot = Math.max(...T.sq.map((a) => this.h(t, a, 0.12)));
    this.railMat.color.copy(linC('bone', railK * 0.8 + 0.8 * xHot));
  }

  private updateOrb(t: number, p: THREE.Vector3, shown: number) {
    const T = this.T, n = this.n;
    const pulseK = Math.max(this.h(t, T.im), this.h(t, T.chase), ...T.sq.map((a) => this.h(t, a)), this.h(t, T.hitting));
    const big = n === 4 ? 1.35 : 1;
    const vis = shown > 0.001;
    const au = this.ctx.audio;
    this.orbCore.visible = vis; this.orbCore.position.copy(p);
    this.orbLight.visible = vis;
    this.orbLight.position.copy(p).add(new THREE.Vector3(0, 0, 3));
    const flick = 0.9 + 0.1 * Math.sin(t * 91.7) * Math.sin(t * 57.3);
    const ce = this.h(t, T.ceil, n === 4 ? 0.6 : 0.25);
    this.orbCore.scale.setScalar(big * shown * (1 + 0.3 * pulseK + 0.5 * ce));
    // halo: two camera-facing soft sprites (tight + wide); bloom does the rest
    const HALO = [[3.2, 1.1], [9, 0.32]] as const;
    this.halos.forEach((o, i) => {
      const [r, g0] = HALO[i]!;
      o.visible = vis; o.position.copy(p);
      o.lookAt(this.cam.position);
      o.scale.setScalar(r * big * shown * (1 + 0.45 * pulseK + (n === 4 ? 1.2 : 0.4) * ce));
      if (i === 0) this.haloR = o.scale.x;
      const g = g0 * (1 + 1.4 * pulseK + 2 * ce) * flick;
      (o.material as THREE.MeshBasicMaterial).color.setRGB(LIN.signal[0] * g, LIN.signal[1] * g, LIN.signal[2] * g);
    });
    // shock rings on the orb's beats: it arrives, steps (squeeze · math · every · byte), lifts, lands
    const evs: [number, number][] = [[T.chase, 0.6], ...T.sq.map((a) => [a, 0.8] as [number, number]), [T.hitting, 1], [T.ceil, n === 4 ? 2.6 : 1.6]];
    const live = evs.map(([a, k]) => [t - q(au, a), k] as [number, number]).filter(([age]) => age >= 0 && age < 0.9).slice(-this.shocks.length);
    this.shocks.forEach((o, i) => {
      const e = live[i];
      o.visible = vis && !!e;
      if (!e) return;
      const [age, k] = e;
      o.position.copy(p);
      o.lookAt(this.cam.position);
      o.scale.setScalar((2.5 + 14 * k * ease.outExpo(clamp(age / 0.7))) * big);
      const g = 1.6 * k * Math.pow(1 - age / 0.9, 2);
      (o.material as THREE.MeshBasicMaterial).color.setRGB(LIN.ember[0] * g, LIN.ember[1] * g, LIN.ember[2] * g);
    });
    // n = 4: the orb breaks through in light (rays that spin out from "ceiling" to the end)
    const age = t - q(au, T.ceil);
    this.rays.visible = vis && n === 4 && age > 0;
    if (this.rays.visible) {
      this.rays.position.copy(p);
      this.rays.lookAt(this.cam.position);
      this.rays.rotateZ(age * 0.35 + 0.6 * ease.outExpo(clamp(age / 0.8)));
      const grow = ease.outExpo(clamp(age / 0.45));
      const g = (0.5 + 1.8 * Math.pow(0.5, age / 0.35)) * (0.85 + 0.15 * f01(Math.sin(t * 40)));
      this.rays.children.forEach((r, i) => {
        const L = (30 + 26 * hash(i)) * grow;
        r.scale.set(L, 1.2 + 1.6 * hash(i + 9), 1);
        const m = (r as THREE.Mesh).material as THREE.MeshBasicMaterial;
        const gi = g * (0.5 + 0.5 * hash(i + 3));
        m.color.setRGB(LIN.ember[0] * gi, LIN.ember[1] * gi, LIN.ember[2] * gi);
      });
    }
    this.orbLight.intensity = vis ? (900 + 2200 * pulseK + 5000 * ce * big) * shown : 0;
  }

  private updateRings(t: number) {
    const n = this.n;
    this.rings.forEach((r, j) => {
      const a = j < n - 1 ? (j === n - 2 ? this.s(t, this.T.sq[0]!) : 1) : 0;
      r.visible = a > 0.001;
      r.lookAt(this.cam.position);
      const g = 0.55 * a;
      (r.material as THREE.MeshBasicMaterial).color.setRGB(LIN.bone[0] * g, LIN.bone[1] * g, LIN.bone[2] * g);
    });
  }

  private updateCap(t: number, p: THREE.Vector3, l2: number) {
    const a = this.s(t, this.T.sq[0]!);
    const top = Y3(roofL10(l2));
    const show = a > 0 && top - p.y > 3;
    this.capLine.visible = show;
    if (show) {
      const pos = this.capLine.geometry.getAttribute('position') as THREE.BufferAttribute;
      pos.setXYZ(0, p.x, p.y + 3.2, ZF); pos.setXYZ(1, p.x, top - 0.6, ZF); pos.needsUpdate = true;
      this.capLine.computeLineDistances();
      this.capMat.opacity = 0.65 * a;
    }
    this.tick.visible = a > 0;
    this.tick.position.set(p.x, CY0, 0.6);
  }

  /** Energy: packets flow along the lit roof (memory bandwidth on the slope, math on the flat) and shoot
   * along the x rail on kicks; faster when the drums hit. */
  private updateFlow(t: number, dv: number) {
    const T = this.T, n = this.n, au = this.ctx.audio;
    const items: PulseItem[] = [];
    const litS = n === 1 ? this.s(t, T.slope) : 1, litF = n === 1 ? this.s(t, T.line) : 1;
    const curveS = new THREE.LineCurve3(PS.clone().add(new THREE.Vector3(0, 1.5, DEPTH / 2 + 0.4)), PR.clone().add(new THREE.Vector3(0, 1.5, DEPTH / 2 + 0.4)));
    const curveF = new THREE.LineCurve3(PR.clone().add(new THREE.Vector3(0, 1.5, DEPTH / 2 + 0.4)), PE.clone().add(new THREE.Vector3(0, 1.5, DEPTH / 2 + 0.4)));
    const rate = 0.09 * (1 + 0.5 * (n - 1));
    const gain = 1.6 + 0.4 * n;
    for (let i = 0; i < 6 + 2 * n; i++) {
      const u = (dv * rate + i / (6 + 2 * n)) % 1;
      if (litS > 0) items.push({ curve: curveS, u, len: 0.03, gain: gain * litS, width: 0.5 });
      if (litF > 0) items.push({ curve: curveF, u: (u + 0.37) % 1, len: 0.025, gain: gain * 0.8 * litF, width: 0.5 });
    }
    const railC = new THREE.LineCurve3(new THREE.Vector3(CX0, CY0 + 0.8, 0.6), new THREE.Vector3(CX1 + 8, CY0 + 0.8, 0.6));
    for (const [kt] of au.events('kick', t - 1.2, t + 0.001)) {
      const age = t - kt;
      if (age < 0 || age > 1.1) continue;
      items.push({ curve: railC, u: ease.outCubic(clamp(age / 0.9)), len: 0.05, gain: 1.8 * (1 - age / 1.1), width: 0.6 });
    }
    this.flow.set(items);
  }

  // ------------------------------------------------------------------ overlay (screen-space, projected anchors)
  private scr(v: THREE.Vector3): Pt {
    const p = this.chart.localToWorld(v.clone()).project(this.cam);
    return { x: (p.x * 0.5 + 0.5) * W, y: (-p.y * 0.5 + 0.5) * H, ok: p.z < 1 && p.z > -1 };
  }

  private drawRoofLabels(c: CanvasRenderingContext2D, t: number) {
    const T = this.T, n = this.n;
    const on = n === 1 ? this.s(t, T.roof) : 1;
    if (on <= 0) return;
    const front = (v: THREE.Vector3) => v.clone().add(new THREE.Vector3(0, 0, DEPTH / 2));
    const a = this.scr(front(PS)), b = this.scr(front(PR)), e = this.scr(front(PE));
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    const word = this.s(t, T.roof) * (1 - this.s(t, T.roof + 60 / this.ctx.audio.bpm, 0.09));
    if (word > 0) {
      const size = 150;
      c.save();
      c.font = font(this.f.big, size);
      c.fillStyle = rgba('bone', 0.92 * word);
      c.translate(lerp(a.x, b.x, 0.55), lerp(a.y, b.y, 0.55)); c.rotate(ang);
      c.fillText('ROOF', -measure('ROOF', this.f.big, size) / 2, -22);
      c.restore();
      c.save();
      c.font = font(this.f.big, size);
      c.fillStyle = rgba('bone', 0.92 * word);
      const angF = Math.atan2(e.y - b.y, e.x - b.x);
      c.translate(lerp(b.x, e.x, 0.5), lerp(b.y, e.y, 0.5)); c.rotate(angF);
      c.fillText('LINE', -measure('LINE', this.f.big, size) / 2, -22);
      c.restore();
    }
    const k0 = n === 1 ? 0 : 0.5, off = 1 - word;
    const memA = Math.max(k0, this.s(t, T.mem)) * off, memB = Math.max(k0, this.s(t, T.slope)) * off;
    const mathA = Math.max(k0, this.s(t, T.math)) * off, mathB = Math.max(k0, this.s(t, T.line)) * off;
    c.save();
    c.translate(lerp(a.x, b.x, 0.5), lerp(a.y, b.y, 0.5)); c.rotate(ang);
    if (memA > 0) {
      c.font = font(this.f.lyr, 46);
      c.fillStyle = rgba('bone', 0.97 * memA);
      const s = 'memory speed limit';
      c.fillText(s, -measure(s, this.f.lyr, 46) / 2, -26 - 8 * (1 - memA));
    }
    if (memB > 0) {
      c.font = font(this.f.mono, 26);
      c.fillStyle = rgba('ash', 0.95 * memB);
      const s = '936 GB/s · RTX 3090 · CH6';
      c.fillText(s, -measure(s, this.f.mono, 26) / 2, 44 + 8 * (1 - memB));
    }
    c.restore();
    const angF = Math.atan2(e.y - b.y, e.x - b.x);
    c.save();
    c.translate(b.x, b.y); c.rotate(angF);
    const Lf = Math.hypot(e.x - b.x, e.y - b.y);
    if (mathA > 0) {
      c.font = font(this.f.lyr, 46);
      c.fillStyle = rgba('bone', 0.97 * mathA);
      c.fillText('math speed limit', 34, -28 - 8 * (1 - mathA));
    }
    if (mathB > 0) {
      c.font = font(this.f.mono, 26);
      c.fillStyle = rgba('ash', 0.95 * mathB);
      const s = '35.6 TFLOP/s · CH6';
      c.fillText(s, Math.max(34, Lf - measure(s, this.f.mono, 26) - 10), 42 + 8 * (1 - mathB));
    }
    c.restore();
  }

  private drawAxisTitles(c: CanvasRenderingContext2D, t: number) {
    const T = this.T;
    const xHot = Math.max(...T.sq.map((a) => this.h(t, a, 0.12)));
    const yHot = this.h(t, T.hitting, 0.15);
    const xe = this.scr(new THREE.Vector3(CX1 + 8, CY0, 0));
    const ye = this.scr(new THREE.Vector3(CX0, CY1 - 1, 0));
    c.font = font(this.f.monoM, 26);
    c.letterSpacing = '2px';
    const xt = 'MATH PER BYTE';
    if (xe.ok) { c.fillStyle = lerpCol('bone', 'signal', xHot, 0.75 + 0.25 * xHot); c.fillText(xt, xe.x - measure(xt, this.f.monoM, 26, 2), xe.y + 46); }
    if (ye.ok) { c.fillStyle = lerpCol('bone', 'signal', yHot, 0.75 + 0.25 * yHot); c.fillText('SPEED', ye.x + 22, ye.y + 8); }
    c.letterSpacing = '0px';
  }

  /** Screen radius of a sphere of radius `r` mm at `v` (for placing labels clear of rings and halos). */
  private scrR(v: THREE.Vector3, r: number): number {
    const a = this.scr(v), up = new THREE.Vector3().setFromMatrixColumn(this.cam.matrixWorld, 1).multiplyScalar(r);
    const b = this.scr(v.clone().add(up));
    return Math.hypot(b.x - a.x, b.y - a.y);
  }

  private drawRungLabels(c: CanvasRenderingContext2D, t: number) {
    const n = this.n;
    for (let j = 0; j < n - 1; j++) {
      const a = j === n - 2 ? this.s(t, this.T.sq[0]!) : 1;
      if (a <= 0) continue;
      const r = RUNG[j]!;
      const v = new THREE.Vector3(X3(lx(r.x)), Y3(ly(r.tf * 1e12)), ZF);
      const p = this.scr(v);
      if (!p.ok) continue;
      const R = this.scrR(v, 3);
      c.font = font(this.f.mono, 24);
      const y = p.y + R + 28, band = clamp((880 - y) / 50);
      if (band <= 0) continue;
      c.fillStyle = rgba('bone', 0.72 * a * band);
      c.fillText(r.short, p.x - measure(r.short, this.f.mono, 24) / 2, y);
    }
  }

  private drawDotLabel(c: CanvasRenderingContext2D, t: number, p3: THREE.Vector3, shown: number) {
    if (shown <= 0) return;
    const T = this.T, n = this.n;
    const p = this.scr(p3);
    if (!p.ok) return;
    const nameA = this.s(t, T.hitting);
    const label = nameA > 0 ? RUNG[n - 1]!.name : 'your kernel';
    const la = nameA > 0 ? nameA : shown * this.s(t, T.chase) * 0.9;
    if (la <= 0) return;
    const size = 32;
    c.font = font(this.f.monoM, size);
    c.fillStyle = rgba(nameA > 0 ? 'signal' : 'bone', 0.98 * la);
    const wl = measure(label, this.f.monoM, size);
    const R = Math.min(170, this.scrR(p3, this.haloR * 0.8)) + 14;
    const right = p.x + R + wl > W - 70;
    const y = p.y + R * 0.55 + 22;
    const band = clamp((880 - y) / 50); // never over the lyric
    if (band <= 0) return;
    c.fillStyle = rgba(nameA > 0 ? 'signal' : 'bone', 0.98 * la * band);
    c.fillText(label, right ? p.x - R - wl : p.x + R, y);
  }

  private drawReadoutAt(c: CanvasRenderingContext2D, t: number) {
    const T = this.T;
    const p = this.scr(RO3);
    if (!p.ok) return;
    const landed = this.s(t, T.ceil);
    const v = landed > 0 ? this.vNow : this.vPrev;
    const flash = this.h(t, T.ceil, 0.14);
    drawReadout(c, clamp(p.x, 120, W - 480), clamp(p.y, 200, 900), v, {
      scale: 1.6, text: formatPDoom(v),
      digits: flash > 0.02 ? lerpCol('bone', 'signal', flash, 1) : rgba('bone', 0.94),
      label: rgba('bone', 0.62),
    });
  }

  private drawRecap(c: CanvasRenderingContext2D, t: number) {
    const T = this.T, n = this.n;
    const lines: [string, number][] = [];
    if (RECAP[n]) lines.push([RECAP[n]![0], this.s(t, T.recap0)], [RECAP[n]![1], this.s(t, T.recap1)]);
    else if (n === 3) lines.push([CAPTION3, this.s(t, T.tonight)]);
    let y = 940;
    for (const [s, a] of lines) {
      if (a <= 0) { y += 56; continue; }
      c.font = font(this.f.lyr, 44);
      c.fillStyle = rgba(y === 940 ? 'bone' : 'ash', 0.97 * a);
      c.fillText(s, LYR.x, y - 36 + 6 * (1 - a));
      y += 56;
    }
  }

  private drawLyric(c: CanvasRenderingContext2D, t: number) {
    const T = this.T;
    if (t >= T.lyrEnd) return;
    let li = -1;
    for (let k = 0; k < 4; k++) if (t >= this.lines[k]!.start - 0.4) li = k;
    if (li < 0) li = 0;
    if (this.n === 3 && this.s(t, T.tonight) > 0) return;
    const line = this.lines[li]!;
    const size = this.lyrSize;
    c.font = font(this.f.lyr, size);
    const text = line.text;
    let ci = 0;
    for (const w of line.words) {
      const at = text.indexOf(w.w, ci);
      const i0 = at < 0 ? ci : at;
      ci = i0 + w.w.length;
      const x = LYR.x + glyphX(text, i0, this.f.lyr, size);
      c.fillStyle = this.wordCol(w, t, line);
      c.fillText(w.w, x, LYR.base);
    }
  }

  private wordCol(w: Word, t: number, line: Line): string {
    if (t < w.start) return rgba('bone', 0.32);
    const next = line.words[w.index + 1];
    const cur = t < (next ? next.start : w.end + 0.25);
    const k = clamp((t - w.start) / 0.05);
    return cur ? lerpCol('bone', 'signal', k, 1) : rgba('bone', 0.96);
  }
}

// ---------------------------------------------------------------- helpers
function hexRGB(k: Col): [number, number, number] {
  const v = parseInt(HEX[k].slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
function lerpCol(a: Col, b: Col, k: number, alpha: number): string {
  const A = hexRGB(a), B = hexRGB(b);
  const m = (i: number) => Math.round(A[i]! + (B[i]! - A[i]!) * clamp(k));
  return `rgba(${m(0)},${m(1)},${m(2)},${alpha})`;
}
function hash(i: number): number {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}
function f01(x: number): number { return 0.5 + 0.5 * x; }

let GLOW_TEX: THREE.CanvasTexture | null = null;
/** Soft radial falloff in RGB (additive: black = nothing). */
function glowTex(): THREE.CanvasTexture {
  if (GLOW_TEX) return GLOW_TEX;
  const N = 128, cv = document.createElement('canvas');
  cv.width = cv.height = N;
  const c = cv.getContext('2d')!, img = c.createImageData(N, N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const r = Math.hypot((x + 0.5) / N * 2 - 1, (y + 0.5) / N * 2 - 1);
    const v = r >= 1 ? 0 : Math.exp(-r * r * 7) * (1 - r * r) + 0.35 * Math.exp(-r * r * 60);
    const b = Math.round(255 * clamp(v)), i = (y * N + x) * 4;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = b; img.data[i + 3] = 255;
  }
  c.putImageData(img, 0, 0);
  return (GLOW_TEX = new THREE.CanvasTexture(cv));
}
let RAY_TEX: THREE.CanvasTexture | null = null;
/** A light ray: hot at the root, fading along its length, soft across. */
function rayTex(): THREE.CanvasTexture {
  if (RAY_TEX) return RAY_TEX;
  const NX = 256, NY = 32, cv = document.createElement('canvas');
  cv.width = NX; cv.height = NY;
  const c = cv.getContext('2d')!, img = c.createImageData(NX, NY);
  for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    const u = (x + 0.5) / NX, w = (y + 0.5) / NY * 2 - 1;
    const v = Math.pow(1 - u, 1.6) * Math.exp(-w * w * 5) * (1 - w * w);
    const b = Math.round(255 * clamp(v)), i = (y * NX + x) * 4;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = b; img.data[i + 3] = 255;
  }
  c.putImageData(img, 0, 0);
  return (RAY_TEX = new THREE.CanvasTexture(cv));
}
