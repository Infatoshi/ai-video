// roofline (the drop, instrumental, 8 bars, a cut every beat): why decode is slow and why batching fixes it.
// The roofline of the RTX 3090 that measured the run, printed in 3D: x = arithmetic intensity (FLOP per
// byte, log), y = TFLOPS (log), a sloped roof (936 GB/s of memory bandwidth) meeting a flat ceiling (71
// TFLOPS dense bf16) at the ridge, 71e12 / 936e9 = 76 FLOP/byte. Decode at batch 1 uses each 2-byte weight
// for 2 FLOPs: ~1 FLOP/byte, low on the slope, 0.69 TFLOPS (2 x params x the measured 42.8 tok/s). On bar 3
// the batch slider slams 1 -> 100 (the same bytes now serve 100 tokens, ~100 FLOP/byte) and the dot climbs
// to 29.0 TFLOPS (2 x params x the measured 1,805.5 tok/s). Around it: the speed bars, one weight serving B
// tokens, 100 copies writing the greedy picks at the measured rate (never ahead of the story, never past
// the 10 recorded picks), and the tower firing each emitted token out
// of its roof 100 times over (the story emits · word and · ' in the drop).
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { HEX } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, hash } from '../engine/util';
import { showTok } from '../engine/llm';
import { Wash, specks, makeCam, drive, shot, snapIn, portrait, perFrame, inkMat, flatMat, outline, outlineInstanced, tokenTile } from './_kit';
import { Tower, riderTile } from './_tower';
import { PINK, BLUE, INK, PAPER, GPU, RIDGE, runNumbers, fmt1, fmt2, fmtInt, slab, label, labelW, bigNum, rain, toScreen, sheetMats, safeTop } from './memory-parts';

type Kind = 'wide' | 'fly' | 'slider' | 'bars' | 'wall' | 'tower' | 'sheet' | 'ridge';
/** One shot per beat of the drop (32 beats), rotating through the ideas. */
const PLAN: Kind[] = [
  'wide', 'fly', 'sheet', 'ridge', 'bars', 'wide', 'slider', 'fly', // bars 1-2: stuck at batch 1
  'slider', 'wide', 'fly', 'tower', 'wall', 'wide', 'sheet', 'bars', // bars 3-4: the slam, · word
  'ridge', 'wall', 'fly', 'wide', 'slider', 'tower', 'wall', 'bars', // bars 5-6: · '
  'fly', 'sheet', 'wide', 'wall', 'ridge', 'fly', 'bars', 'wide', // bars 7-8
];
/** Batch sizes the slider snaps through on the 16ths after the slam. */
const STEPS = [1, 2, 4, 8, 16, 32, 64, 100];

// chart space: 3 world units per decade
const XL = [-0.5, 3] as const, YL = [-1, 2.2] as const, U = 3;
const xw = (ai: number) => (Math.log10(ai) - XL[0]) * U;
const yw = (tf: number) => (Math.log10(tf) - YL[0]) * U;
const ZF = 3.4; // the front plane: the dot, its stem and the batch rail
const TOWER_AT = new THREE.Vector3(60, 0, 0), BARS_AT = new THREE.Vector3(-60, 0, 0), SHEET_AT = new THREE.Vector3(0, 0, -60);

export default class Roofline extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(45);
  wash = new Wash();
  back = new Layer2D();
  front = new Layer2D();
  fx = new Layer2D();
  N!: ReturnType<typeof runNumbers>;
  chart = new THREE.Group();
  dot!: THREE.Mesh;
  ghost!: THREE.Mesh;
  stem!: THREE.Mesh;
  knob!: THREE.Mesh;
  tower = new Tower();
  towerG = new THREE.Group();
  riders = new Map<number, THREE.Mesh>();
  bursts: { k: number; hero: THREE.Mesh; copies: THREE.InstancedMesh }[] = [];
  bars = new THREE.Group();
  bar1!: THREE.Mesh;
  bar100!: THREE.Mesh;
  barH = 11;
  sheetG = new THREE.Group();
  tiles = new Map<number, THREE.InstancedMesh>();
  wires!: THREE.InstancedMesh;
  b0 = 0;
  tDrop = 0;
  pool: string[] = [];
  private m4 = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private e = new THREE.Euler();
  private v = new THREE.Vector3();
  private s3 = new THREE.Vector3();
  private zero = new THREE.Matrix4().makeScale(0, 0, 0);

  override init() {
    const { llm, audio: au, story } = this.ctx;
    this.N = runNumbers(llm.d);
    const drop = au.sections.find((s) => s.name === 'drop');
    this.tDrop = au.timeOfBeat(Math.round(au.beatAt(drop?.start ?? this.ctx.start + 0.6)));
    this.b0 = Math.round(au.beatAt(this.tDrop));
    this.buildChart();

    // the tower station: each emission fires 100 tiles out of the roof (ours pink, 99 copies)
    this.tower.outlineAll();
    this.towerG.add(this.tower.group);
    this.towerG.position.copy(TOWER_AT);
    for (let k = 0; k < llm.steps.length; k++) {
      const e = story.emits[k]!;
      if (e > this.ctx.start - 20 && e < this.ctx.end + 20) this.riders.set(k, riderTile(llm.steps[k]!.tok.t));
      if (e < this.ctx.start || e > this.ctx.end) continue;
      const t = llm.steps[k]!.tok;
      const hero = tokenTile(t.t, { face: 'pink', text: 'paper', side: 'pink', id: t.id, h: 1.3, line: 3 });
      const proto = tokenTile(t.t, { face: 'paper', text: 'ink', side: 'blue', h: 0.8 });
      const copies = new THREE.InstancedMesh(proto.geometry, proto.material as THREE.Material[], 99);
      copies.frustumCulled = false;
      outlineInstanced(copies, 1.6);
      this.towerG.add(hero, copies);
      this.bursts.push({ k, hero, copies });
    }
    this.scene.add(this.towerG);

    // TFLOPS, linear: the bf16 peak as an empty column, batch 1 (0.69) and batch 100 (29.0) against it
    const peak = new THREE.Mesh(new THREE.BoxGeometry(2.6, this.barH, 2.6).translate(0, this.barH / 2, 0), inkMat({ ink: 'blue', lit: 0.06, shade: 0.28 }));
    peak.position.set(-4, 0, 0);
    outline(peak, 2.4);
    this.bar1 = new THREE.Mesh(new THREE.BoxGeometry(2.6, 1, 2.6).translate(0, 0.5, 0), inkMat({ ink: 'blue', lit: 0.6, shade: 1 }));
    this.bar1.scale.y = (this.barH * this.N.tf1) / (GPU.peak / 1e12);
    this.bar1.position.set(0, 0, 0);
    outline(this.bar1, 2.4);
    this.bar100 = new THREE.Mesh(new THREE.BoxGeometry(2.6, 1, 2.6).translate(0, 0.5, 0), inkMat({ ink: 'pink', lit: 0.6, shade: 1, over: 'blue', overShade: 0.5 }));
    this.bar100.position.set(4, 0, 0);
    outline(this.bar100, 2.4);
    const plate = new THREE.Mesh(new THREE.BoxGeometry(14, 0.3, 6), inkMat({ ink: 'ink', lit: 0.2, shade: 0.9 }));
    plate.position.y = -0.15;
    outline(plate, 2);
    this.bars.add(peak, this.bar1, this.bar100, plate);
    this.bars.position.copy(BARS_AT);
    this.scene.add(this.bars);

    // one weight serving B tokens: a sheet (layer 1's Q) over a 10 x 10 grid of the token being computed
    const sheet = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.08, 4.6), sheetMats('Q', llm.cfg.hidden, llm.cfg.hidden, llm.cfg.hidden));
    outline(sheet, 2.2);
    sheet.position.set(0, 6.2, 0);
    sheet.rotation.set(0.08, 0.3, 0);
    this.sheetG.add(sheet);
    for (let k = 0; k < llm.steps.length; k++) {
      const e0 = k > 0 ? story.emits[k - 1]! : -Infinity, e1 = story.emits[k]!;
      if (e1 < this.ctx.start || e0 > this.ctx.end) continue;
      const proto = tokenTile(llm.steps[k]!.tok.t, { face: 'paper', text: 'ink', side: 'blue', h: 0.5 });
      const inst = new THREE.InstancedMesh(proto.geometry, proto.material as THREE.Material[], 100);
      inst.frustumCulled = false;
      outlineInstanced(inst, 1.4);
      this.sheetG.add(inst);
      this.tiles.set(k, inst);
    }
    this.wires = new THREE.InstancedMesh(new THREE.BoxGeometry(0.05, 1, 0.05).translate(0, 0.5, 0), flatMat('pink'), 100);
    this.wires.frustumCulled = false;
    this.sheetG.add(this.wires);
    this.sheetG.position.copy(SHEET_AT);
    this.scene.add(this.sheetG);

    // the rain: the question and the answer so far, as ids and tokens
    const toks = [...llm.d.prompt.slice(llm.questionStart), ...llm.steps.map((s) => s.tok)];
    this.pool = toks.flatMap((t) => [showTok(t.t), String(t.id)]).concat(['×100', '1,805.5', '42.8', 'TOK/S']);
  }

  private buildChart() {
    const g = this.chart;
    const x0 = 0, x1 = xw(10 ** XL[1]), y1 = yw(10 ** YL[1]);
    const rx = xw(RIDGE), ry = yw(GPU.peak / 1e12);
    const yAt = (x: number) => Math.min(ry, yw((GPU.bw / 1e12) * 10 ** (x / U + XL[0])));
    // the roof: a folded slab, slope then ceiling, 6 deep
    const th = 0.32;
    const sh = new THREE.Shape();
    sh.moveTo(x0, yAt(x0)); sh.lineTo(rx, ry); sh.lineTo(x1, ry); sh.lineTo(x1, ry - th); sh.lineTo(rx, ry - th); sh.lineTo(x0, yAt(x0) - th); sh.closePath();
    const roofGeo = new THREE.ExtrudeGeometry(sh, { depth: 6, bevelEnabled: false });
    roofGeo.translate(0, 0, -3);
    const roof = new THREE.Mesh(roofGeo, inkMat({ ink: 'blue', lit: 0.4, shade: 1, over: 'pink', overLit: 0, overShade: 0.45 }));
    outline(roof, 2.6);
    g.add(roof);
    // the back wall: the chart itself (log grid, attainable region under the roof)
    const px = 140, cw = Math.round(x1 * px), ch = Math.round(y1 * px);
    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    const c = cv.getContext('2d')!;
    c.fillStyle = HEX.paper; c.fillRect(0, 0, cw, ch);
    const X = (x: number) => x * px, Y = (y: number) => ch - y * px;
    c.fillStyle = 'rgba(0,120,191,0.22)';
    c.beginPath(); c.moveTo(X(x0), Y(0)); c.lineTo(X(x0), Y(yAt(x0))); c.lineTo(X(rx), Y(ry)); c.lineTo(X(x1), Y(ry)); c.lineTo(X(x1), Y(0)); c.closePath(); c.fill();
    c.fillStyle = HEX.blue;
    for (let dcd = Math.floor(XL[0]); dcd <= XL[1]; dcd++) for (let m = 1; m < 10; m++) {
      const x = xw(m * 10 ** dcd); if (x < 0 || x > x1) continue;
      c.fillRect(X(x) - (m === 1 ? 4 : 1.5), 0, m === 1 ? 8 : 3, ch);
    }
    for (let dcd = YL[0]; dcd <= YL[1]; dcd++) for (let m = 1; m < 10; m++) {
      const y = yw(m * 10 ** dcd); if (y < 0 || y > y1) continue;
      c.fillRect(0, Y(y) - (m === 1 ? 4 : 1.5), cw, m === 1 ? 8 : 3);
    }
    c.strokeStyle = HEX.ink; c.lineWidth = 12; c.lineJoin = 'round';
    c.beginPath(); c.moveTo(X(x0), Y(yAt(x0))); c.lineTo(X(rx), Y(ry)); c.lineTo(X(x1), Y(ry)); c.stroke();
    c.lineWidth = 10; c.strokeRect(5, 5, cw - 10, ch - 10);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(x1, y1), new THREE.MeshBasicMaterial({ map: tex }));
    wall.position.set(x1 / 2, y1 / 2, -3.02);
    g.add(wall);
    // the floor, with the decades running forward, and the batch rail on the front edge
    const floor = new THREE.Mesh(new THREE.BoxGeometry(x1 + 0.6, 0.2, 7.6), inkMat({ ink: 'blue', lit: 0.1, shade: 0.35 }));
    floor.position.set(x1 / 2, -0.1, 0.5);
    outline(floor, 2);
    g.add(floor);
    for (let dcd = 0; dcd <= XL[1]; dcd++) {
      const t = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.04, 7.2), flatMat('blue'));
      t.position.set(xw(10 ** dcd), 0.02, 0.5);
      g.add(t);
    }
    const rail = new THREE.Mesh(new THREE.BoxGeometry(x1, 0.22, 0.34), inkMat({ ink: 'ink', lit: 0.4, shade: 1 }));
    rail.position.set(x1 / 2, 0.11, ZF);
    outline(rail, 2);
    g.add(rail);
    for (const b of STEPS) {
      const tk = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.5, 0.5), inkMat({ ink: 'ink', lit: 0.5, shade: 1 }));
      tk.position.set(xw(b), 0.25, ZF);
      g.add(tk);
    }
    // the y axis post
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.22, y1, 0.22), inkMat({ ink: 'ink', lit: 0.4, shade: 1 }));
    post.position.set(0, y1 / 2, -3);
    outline(post, 1.6);
    g.add(post);
    // the dot (where decode runs), its stem to the slider knob, and the batch-1 ghost
    this.dot = new THREE.Mesh(new THREE.SphereGeometry(0.42, 32, 16), inkMat({ ink: 'pink', lit: 0.7, shade: 1, over: 'blue', overShade: 0.35 }));
    outline(this.dot, 3);
    this.ghost = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), inkMat({ ink: 'ink', lit: 0.5, shade: 1 }));
    outline(this.ghost, 2);
    this.stem = new THREE.Mesh(new THREE.BoxGeometry(0.07, 1, 0.07).translate(0, 0.5, 0), flatMat('pink'));
    this.knob = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.55, 0.8), inkMat({ ink: 'pink', lit: 0.6, shade: 1 }));
    outline(this.knob, 2.4);
    g.add(this.dot, this.ghost, this.stem, this.knob);
    this.scene.add(g);
  }

  /** Batch size and the dot's (log) position at time t: 1 until the slam, then up the powers of two on the 16ths. */
  private batch(t: number): { B: number; lx: number; ly: number; k: number } {
    const au = this.ctx.audio, N = this.N;
    const slam = this.b0 + 8;
    const b = au.beatAt(t) - slam;
    let s = 0;
    if (b >= 0) s = Math.min(STEPS.length - 1, Math.floor(b * 4) + 1);
    const ts = au.timeOfBeat(slam + (s - 1) / 4);
    const k = s === 0 ? 1 : snapIn(t, ts, 0.06);
    const lb = (i: number) => Math.log10(STEPS[Math.max(0, i)]!);
    const lx = lb(s - 1) + (lb(s) - lb(s - 1)) * k;
    // between the two measured points the dot's height is interpolated (log-log); only the ends are labelled
    const y1 = Math.log10(N.tf1), y100 = Math.log10(N.tf100);
    return { B: STEPS[s]!, lx, ly: y1 + ((y100 - y1) * lx) / 2, k };
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio, llm = this.ctx.llm, story = this.ctx.story;
    const P = portrait();
    const sh = shot(this.ctx, f, { every: 1, from: this.tDrop });
    const i = Math.min(PLAN.length - 1, sh.i);
    const kind = PLAN[i]!;
    const alt = PLAN.slice(0, i).filter((x) => x === kind).length;
    const d = drive(au, f.t, 0.8, 2.2);
    const N = this.N;
    const bt = this.batch(f.t);
    const slamT = au.timeOfBeat(this.b0 + 8);
    const after = f.ft >= slamT;

    // ---- background
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: kind === 'wall' ? 0.35 : 0.55, c1: i % 2 ? 'blue' : 'pink', c2: i % 2 ? 'pink' : 'blue' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => rain(this.back.ctx, this.pool, drive(au, f.ft, 0.8, 2.2) * 0.4, { seed: sh.seed, cols: P ? 6 : 10, color: i % 3 === 1 ? INK : BLUE, size: 22, density: 0.74 })), out);

    // ---- 3D: only the station in shot
    this.chart.visible = kind === 'wide' || kind === 'fly' || kind === 'slider' || kind === 'ridge';
    this.towerG.visible = kind === 'tower' || kind === 'wall';
    this.bars.visible = kind === 'bars';
    this.sheetG.visible = kind === 'sheet';

    // chart: the dot, stem, knob, ghost
    const dx = (bt.lx - XL[0]) * U, dy = (bt.ly - YL[0]) * U;
    this.dot.position.set(dx, dy, ZF);
    this.stem.position.set(dx, 0.22, ZF);
    this.stem.scale.y = Math.max(0.01, dy - 0.22);
    this.knob.position.set(dx, 0.4, ZF);
    this.ghost.visible = after && bt.B > 1;
    this.ghost.position.set(xw(1), yw(N.tf1), ZF);

    // tower: the token in flight, and the bursts
    const st = story.at(f.t);
    const rider = this.riders.get(st.step);
    if (rider) this.tower.ride(rider, st.climb);
    this.tower.light(st.climb);
    const roofY = this.tower.height + 0.4;
    for (const bu of this.bursts) {
      const age = f.t - story.emits[bu.k]!;
      const on = age >= 0 && age < 2.2;
      bu.hero.visible = on;
      bu.copies.visible = on;
      if (!on) continue;
      const u = ease.outExpo(clamp(age / 0.35));
      bu.hero.position.set(0, roofY + 1 + u * 7 + age * 2, 0);
      bu.hero.rotation.set(0, age * 0.8, 0);
      for (let j = 0; j < 99; j++) {
        const th = hash(j, 3) * Math.PI * 2, rr = 0.25 + hash(j, 4) * 0.9, up = 4 + hash(j, 5) * 9;
        const a = Math.min(age, 2), sp = 1 - Math.exp(-a * 7);
        this.v.set(Math.cos(th) * rr * 11 * sp, roofY + up * sp + a * 1.5 - 1.2 * a * a, Math.sin(th) * rr * 11 * sp);
        this.e.set(a * (hash(j, 6) - 0.5) * 6, a * (hash(j, 7) - 0.5) * 6, 0);
        this.q.setFromEuler(this.e);
        const sc = snapIn(age, hash(j, 8) * 0.05, 0.05);
        bu.copies.setMatrixAt(j, this.m4.compose(this.v, this.q, this.s3.set(sc, sc, sc)));
      }
      bu.copies.instanceMatrix.needsUpdate = true;
    }

    // bars: the batch-100 bar slams up on the slam
    this.bar100.scale.y = Math.max(0.001, ((this.barH * N.tf100) / (GPU.peak / 1e12)) * (after ? snapIn(f.t, Math.max(slamT, sh.t0), 0.09) : 0));

    // sheet station: B tiles under one weight, a wire each
    const B = kind === 'sheet' ? (after ? 100 : 1) : 0;
    for (const [k, inst] of this.tiles) {
      const mine = k === st.step && kind === 'sheet';
      inst.visible = mine;
      if (!mine) continue;
      let n = 0;
      for (let j = 0; j < 100; j++) {
        if (j >= B) { inst.setMatrixAt(j, this.zero); this.wires.setMatrixAt(j, this.zero); continue; }
        const gx = B === 1 ? 0 : (j % 10) - 4.5, gz = B === 1 ? 0 : Math.floor(j / 10) - 4.5;
        const kk = snapIn(f.t, sh.t0 + (B === 1 ? 0 : hash(j, 9) * 0.1), 0.06) * (B === 1 ? 2.6 : 1);
        this.v.set(gx * 1.35, (B === 1 ? 0.9 : 0.3) + Math.sin(d * 2 + j) * 0.05, gz * 0.9);
        this.q.identity();
        inst.setMatrixAt(j, this.m4.compose(this.v, this.q, this.s3.set(kk, kk, kk)));
        // wire from the sheet to the tile
        const from = new THREE.Vector3(0, 6.2, 0), to = this.v.clone().setY(B === 1 ? 1.6 : 0.55);
        const dir = from.clone().sub(to);
        this.q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
        this.wires.setMatrixAt(j, this.m4.compose(to, this.q, this.s3.set(1, dir.length() * kk, 1)));
        n++;
      }
      inst.instanceMatrix.needsUpdate = true;
      this.wires.instanceMatrix.needsUpdate = true;
      void n;
    }

    // ---- camera per kind (alt varies the angle each time a kind comes back)
    const a = d * 0.06 + alt * 1.1;
    const Pk = P ? 1.6 : 1;
    let pos: THREE.Vector3, look: THREE.Vector3, fov = 45;
    const cx = xw(10 ** XL[1]) / 2;
    switch (kind) {
      case 'wide': {
        // the whole chart, from a new side each time
        const W5 = [[3, 5.5, 21, 0], [-11, 3.2, 15, 1], [12, 11, 14, -1], [-2, 1.6, 13, 0.6], [6, 7, 24, 0]][alt % 5]!;
        pos = new THREE.Vector3(cx + W5[0]! * (P ? 0.8 : 1) + Math.sin(a) * 1.2, W5[1]!, W5[2]! * Pk);
        look = new THREE.Vector3(cx + W5[3]!, P ? 4.2 : 4.6, 0); fov = P ? 50 : 42; break;
      }
      case 'fly': {
        const rx = xw(RIDGE), ry = yw(GPU.peak / 1e12);
        if (!after && alt === 0) {
          // up the slope from the dot toward the ridge
          const s0 = (d * 0.4) % 0.8;
          pos = new THREE.Vector3(dx - 3.2 + s0, dy + 1.1, ZF + 4.4); look = new THREE.Vector3(dx + 3.2, dy + 2.6, 1.2); fov = P ? 68 : 58;
        } else if (!after) {
          // from the ceiling, down to the dot far below
          pos = new THREE.Vector3(rx + 1.5 + (d * 0.3) % 1, ry + 2.2, ZF + 3.5); look = new THREE.Vector3(xw(1), yw(N.tf1), ZF); fov = P ? 64 : 54;
        } else if (alt % 4 === 2) {
          // alongside the dot, under the ceiling edge, looking back down the slope
          pos = new THREE.Vector3(dx + 2.6, dy + 0.6, ZF + 4.2 + Math.sin(a) * 0.5); look = new THREE.Vector3(xw(1), yw(N.tf1) + 1, 0); fov = P ? 66 : 56;
        } else if (alt % 4 === 3) {
          // over the ceiling, the dot just under its front edge
          pos = new THREE.Vector3(dx - 3 + (d * 0.4) % 2, ry + 2.6, ZF + 3.2); look = new THREE.Vector3(dx + 0.5, dy, ZF); fov = P ? 66 : 56;
        } else if (alt % 4 === 0) {
          // from the floor under the dot, up its stem
          pos = new THREE.Vector3(dx - 1.6, 0.9, ZF + 3.4 + Math.sin(a) * 0.4); look = new THREE.Vector3(dx, dy + 0.4, ZF - 0.4); fov = P ? 70 : 62;
        } else {
          // from the batch-1 ghost, up the whole slope to the dot
          pos = new THREE.Vector3(xw(1) - 1.4, yw(N.tf1) + 0.5, ZF + 2.2); look = new THREE.Vector3(dx, dy, ZF); fov = P ? 62 : 52;
        }
        break;
      }
      case 'slider': {
        const kx = dx;
        pos = new THREE.Vector3(kx - 3 + (alt % 2) * 5, 2.2 + (alt % 2), ZF + (P ? 11 : 7.5));
        look = new THREE.Vector3(kx + (alt % 2 ? -1 : 1.5), 1.6, ZF - 0.5); fov = P ? 58 : 48; break;
      }
      case 'ridge': {
        const rx = xw(RIDGE), ry = yw(GPU.peak / 1e12);
        pos = new THREE.Vector3(rx + (alt % 2 ? -3 : 3) + Math.sin(a) * 0.6, ry + 1.2, ZF + (P ? 11 : 7.5));
        look = new THREE.Vector3(rx - (P ? 0.5 : 1.2), ry - 1.8, 0); fov = P ? 58 : 50; break;
      }
      case 'bars': {
        const o = BARS_AT;
        const sd = alt % 2 ? -1 : 1;
        pos = new THREE.Vector3(o.x + sd * (3 + Math.sin(a)), 2.2, o.z + (P ? 26 : 16));
        look = new THREE.Vector3(o.x + sd * (P ? 0 : 1.2), P ? 5.5 : 5, o.z); fov = P ? 56 : 48; break;
      }
      case 'sheet': {
        const o = SHEET_AT;
        const near = !after;
        pos = new THREE.Vector3(o.x + Math.sin(a + alt) * (near ? 2 : 5), near ? 8.8 : 8.5 + alt, o.z + (P ? (near ? 15 : 19) : near ? 11 : 13));
        look = new THREE.Vector3(o.x + (P ? 0 : near ? -2.6 : 1.5), near ? 3.2 : 2.6, o.z); fov = P ? 58 : 50; break;
      }
      case 'tower': {
        const o = TOWER_AT;
        pos = new THREE.Vector3(o.x + Math.sin(a + alt * 2) * 13, 15, o.z + Math.cos(a + alt * 2) * 13 * (P ? 1.3 : 1));
        look = new THREE.Vector3(o.x, 22.5, o.z); fov = P ? 64 : 56; break;
      }
      default: { // wall: the tower far behind the 100 streams
        const o = TOWER_AT;
        const side = P ? 0 : [0, -1, 1, 0][alt % 4]!; // panel right -> tower left (-1), and back
        const rx = Math.cos(a), rz = -Math.sin(a);
        pos = new THREE.Vector3(o.x + Math.sin(a) * 30, 12, o.z + Math.cos(a) * 30);
        look = new THREE.Vector3(o.x - side * rx * 8.5, 12 + (side ? 2 : 0), o.z - side * rz * 8.5); fov = 46; break;
      }
    }
    this.cam.position.copy(pos);
    this.cam.up.set(0, 1, 0);
    if (this.cam.fov !== fov) { this.cam.fov = fov; this.cam.updateProjectionMatrix(); }
    this.cam.lookAt(look);
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);

    // ---- type (crisp, once per frame)
    this.ctx.comp.draw(r, perFrame(this.front, f, () => this.type(f, kind, alt, sh.t0, after)), out);

    // ---- foreground specks (fast: per sub-frame)
    this.fx.clear();
    specks(this.fx.ctx, d * 1.6, { seed: sh.seed, n: kind === 'wall' ? 50 : 90 });
    this.ctx.comp.draw(r, this.fx.upload(), out);

    return { reg: sh.seed };
  }

  /** Labels, numbers and staged words for the shot. */
  private type(f: Frame, kind: Kind, alt: number, t0: number, after: boolean) {
    const c = this.front.ctx, ft = f.ft, P = portrait(), N = this.N, llm = this.ctx.llm;
    const bt = this.batch(ft);
    const k = snapIn(ft, t0, 0.07);
    const top = safeTop();
    const lab3 = (p: [number, number, number], t: string, o: Parameters<typeof label>[4] = {}) => {
      const [x, y, ok] = toScreen(this.cam, p);
      if (!ok) return;
      const w = labelW(c, t, o.size ?? 24);
      const al = o.align ?? 'left';
      const xl = al === 'center' ? x - w / 2 : al === 'right' ? x - w : x;
      if (xl < 140 || xl + w > W - 40 || y < top || y > H - 120) return;
      label(c, t, x, y, { bg: PAPER, ...o });
    };
    const chartLabels = (full: boolean) => {
      const rx = xw(RIDGE), ry = yw(GPU.peak / 1e12);
      // slope and ceiling
      const sx0 = xw(3), sy0 = yw(0.936 * 3);
      const [ax, ay] = toScreen(this.cam, [sx0, sy0, 3]);
      const [bx, by] = toScreen(this.cam, [sx0 + 1, sy0 + 1, 3]);
      const rot = Math.atan2(by - ay, bx - ax);
      if (ay > top && ay < H - 150 && ax > 150 && ax < W - 300) label(c, `MEMORY ROOF · ${fmtInt(GPU.bw / 1e9)} GB/S`, ax, ay - 26, { size: 26, rot, bg: PAPER, color: BLUE });
      lab3([rx + 0.3, ry + 0.9, 3], `COMPUTE CEILING · ${fmtInt(GPU.peak / 1e12)} TFLOPS`, { size: 26, color: INK });
      lab3([rx, ry + 0.25, 3.1], `◆ RIDGE ${fmtInt(RIDGE)} FLOP/BYTE`, { size: 24, color: PINK, align: 'right' });
      if (full) {
        for (let dcd = 0; dcd <= 3; dcd++) lab3([xw(10 ** dcd), -0.1, ZF + 1.1], fmtInt(10 ** dcd), { size: 24, align: 'center' });
        for (let dcd = -1; dcd <= 2; dcd++) lab3([-0.3, yw(10 ** dcd), -3], dcd < 0 ? '0.1' : fmtInt(10 ** dcd), { size: 24, align: 'right' });
        lab3([xw(1000), -0.1, ZF + 2.4], 'FLOP/BYTE →', { size: 24, color: BLUE, align: 'right' });
        lab3([-0.3, yw(10 ** YL[1]) + 0.3, -3], 'TFLOPS', { size: 24, color: BLUE, align: 'right' });
        lab3([xw(1000), yw(10 ** YL[1]) - 0.2, -3], `${GPU.name} SPEC`, { size: 26, align: 'right', color: INK });
      }
    };
    const dotLabel = () => {
      const [x, y] = toScreen(this.cam, this.dot.position);
      const two = bt.B === 1 || bt.B === 100;
      const l1 = `BATCH ${bt.B} · ≈${bt.B} FLOP/BYTE`;
      const l2 = bt.B === 1 ? `${fmt2(N.tf1)} TFLOPS · FROM ${fmt1(N.s1)} TOK/S` : `${fmt1(N.tf100)} TFLOPS · FROM ${fmt1(N.s100)} TOK/S`;
      const w = Math.max(labelW(c, l1, 28), two ? labelW(c, l2, 24) : 0);
      const lx = clamp(x + 34, 150, W - 60 - w), ly = clamp(y - 30, top + 20, H - 220);
      label(c, l1, lx, ly, { size: 28, color: PINK, bg: PAPER });
      if (two) label(c, l2, lx, ly + 40, { size: 24, bg: PAPER });
      if (two) label(c, 'TOK/S MEASURED · TFLOPS COMPUTED', lx, ly + 74, { size: 22, bg: PAPER, color: BLUE });
      if (after && bt.B === 100) {
        const [gx, gy] = toScreen(this.cam, this.ghost.position);
        if (gy > top && gy < H - 200 && gx > 150) label(c, `BATCH 1 · ${fmt2(N.tf1)}`, gx + 26, gy + 34, { size: 24, bg: PAPER });
      }
    };

    if (kind === 'wide') {
      chartLabels(true);
      dotLabel();
      if (!after && alt === 0) slab(c, 'BATCH 1', P ? W / 2 : W * 0.3, P ? H * 0.2 + 40 : H * 0.28, P ? 150 : 170, k, { maxW: W * 0.8 });
      if (!after && alt === 1) {
        slab(c, '1 FLOP/BYTE', P ? W / 2 : W * 0.3, P ? H * 0.2 + 40 : H * 0.28, P ? 120 : 140, k, { maxW: P ? W * 0.9 : W * 0.5 });
        label(c, 'EACH 2-BYTE WEIGHT: 2 FLOPS PER TOKEN', P ? W / 2 : W * 0.3, (P ? H * 0.2 + 40 : H * 0.28) + (P ? 80 : 95), { size: 26, align: 'center', bg: PAPER });
      }
      if (after && bt.B < 100) {
        // still climbing: the batch size is the hero
        const tx = P ? W / 2 : W * 0.26, ty = P ? H * 0.22 : H * 0.3;
        slab(c, 'BATCH', tx, ty - (P ? 95 : 105), P ? 90 : 100, 1, { width: 125 });
        bigNum(c, String(bt.B), tx, ty + 30, P ? 190 : 210, {});
      } else if (after) {
        // the answer, a different number each time the wide shot comes back
        const tx = P ? W / 2 : W * 0.26, ty = P ? H * 0.22 : H * 0.3;
        const v = Math.max(0, alt - 2) % 5;
        const big = [`×${fmtInt(N.speedup)}`, `${fmt1(N.tf100)} TF`, `${fmtInt((N.tf100 / (GPU.peak / 1e12)) * 100)}%`, fmt1(N.s100), `×${fmtInt(N.speedup)}`][v]!;
        const sub = ['TOKENS PER SECOND · MEASURED', `2 × PARAMS × ${fmt1(N.s100)} TOK/S · COMPUTED`, 'OF THE BF16 PEAK · COMPUTED', 'TOK/S AT BATCH 100 · MEASURED', `SAME ${fmt2(N.read / 1e9)} GB OF WEIGHTS READ PER STEP`][v]!;
        slab(c, big, tx, ty, P ? 170 : 200, k, { maxW: P ? W * 0.9 : W * 0.46 });
        label(c, sub, tx, ty + (P ? 100 : 118), { size: 26, align: 'center', bg: PAPER });
      }
    } else if (kind === 'fly') {
      chartLabels(false);
      dotLabel();
      const [big, sub] = !after
        ? alt === 0 ? ['MEMORY-BOUND', `ON THE SLOPE: SPEED = ${fmtInt(GPU.bw / 1e9)} GB/S × FLOP/BYTE`] : [`${fmtInt((GPU.peak / 1e12) / N.tf1)}× HIGHER`, 'THE CEILING, OVER BATCH 1 · COMPUTED']
        : ([
          [`${fmt1(N.tf100)} TFLOPS`, `2 × PARAMS × ${fmt1(N.s100)} TOK/S · COMPUTED`],
          ['UP THE SLOPE', `BATCH 1 → 100: ${fmt2(N.tf1)} → ${fmt1(N.tf100)} TFLOPS`],
          ['OFF THE SLOPE', `≈100 FLOP/BYTE: PAST THE RIDGE AT ${fmtInt(RIDGE)}`],
          ['UNDER THE CEILING', `${fmt1(N.tf100)} OF ${fmtInt(GPU.peak / 1e12)} TFLOPS · COMPUTED`],
        ] as const)[alt % 4]!;
      // the staged word goes in the half of the frame the dot is not in
      const [, dotY] = toScreen(this.cam, this.dot.position);
      const sy = dotY > H * 0.5 ? (P ? H * 0.3 : H * 0.3) : P ? H * 0.74 : H * 0.72;
      slab(c, big, W / 2, sy, P ? 120 : 150, k, { maxW: W * 0.88, knock: true });
      label(c, sub, W / 2, sy + (P ? 88 : 102), { size: P ? 24 : 28, align: 'center', bg: PAPER });
    } else if (kind === 'slider') {
      // the batch slider: the number is the hero
      const [kx, ky] = toScreen(this.cam, this.knob.position);
      for (const b of STEPS) lab3([xw(b), -0.2, ZF + 0.9], String(b), { size: 26, align: 'center', color: b === bt.B ? PINK : INK });
      const cxT = P ? W / 2 : W * 0.5, cyT = P ? H * 0.34 : H * 0.34;
      slab(c, 'BATCH', cxT, cyT - (P ? 120 : 130), P ? 110 : 120, k, { width: 125 });
      bigNum(c, String(bt.B), cxT, cyT + (P ? 40 : 40), P ? 260 : 280, {});
      label(c, bt.B === 1 ? 'ONE QUESTION AT A TIME' : `${bt.B} COPIES OF THE QUESTION AT ONCE`, cxT, cyT + (P ? 200 : 210), { size: 28, align: 'center', bg: PAPER });
      void kx; void ky;
    } else if (kind === 'ridge') {
      chartLabels(false);
      dotLabel();
      const tx = P ? W / 2 : W * 0.26, ty = P ? H * 0.74 : H * 0.34;
      bigNum(c, `${fmtInt(RIDGE)}`, tx, ty, P ? 200 : 220, {});
      label(c, 'FLOP/BYTE TO LEAVE THE SLOPE', tx, ty + (P ? 125 : 135), { size: 28, align: 'center', bg: PAPER });
      label(c, `${fmtInt(GPU.peak / 1e12)} TFLOPS ÷ ${fmtInt(GPU.bw / 1e9)} GB/S · ${GPU.name} SPEC`, tx, ty + (P ? 165 : 175), { size: 24, align: 'center', bg: PAPER, color: BLUE });
    } else if (kind === 'bars') {
      const colLab = (x: number, h: number, l1: string, l2: string, col: string) => {
        const [px, py] = toScreen(this.cam, [BARS_AT.x + x, h + 0.5, 1.3]);
        const yy = clamp(py - 50, top + 20, H - 260);
        label(c, l1, clamp(px, 220, W - 220), yy, { size: 26, align: 'center', bg: PAPER, color: col });
        label(c, l2, clamp(px, 220, W - 220), yy + 36, { size: 22, align: 'center', bg: PAPER });
      };
      colLab(-4, this.barH, `PEAK ${fmtInt(GPU.peak / 1e12)} TFLOPS`, `${GPU.name} SPEC`, BLUE);
      colLab(0, this.bar1.scale.y, `BATCH 1 · ${fmt2(N.tf1)}`, `${fmt1(N.s1)} TOK/S`, INK);
      if (after) {
        colLab(4, this.bar100.scale.y, `BATCH 100 · ${fmt1(N.tf100)}`, `${fmt1(N.s100)} TOK/S`, PINK);
        const tx = P ? W / 2 : alt % 2 ? W * 0.76 : W * 0.24, ty = P ? H * 0.8 : H * 0.3;
        slab(c, `×${fmt1(N.speedup)}`, tx, ty, P ? 170 : 190, k);
        label(c, 'TOK/S MEASURED · TFLOPS COMPUTED', tx, ty + (P ? 100 : 115), { size: 24, align: 'center', bg: PAPER });
      } else {
        const tx = P ? W / 2 : W * 0.72, ty = P ? H * 0.78 : H * 0.62;
        bigNum(c, `${(N.util1 * 100).toFixed(2)}%`, tx, ty, P ? 170 : 190, {});
        label(c, `OF THE PEAK IN USE AT BATCH 1 · COMPUTED`, tx, ty + (P ? 100 : 110), { size: 26, align: 'center', bg: PAPER });
      }
    } else if (kind === 'sheet') {
      const Bn = after ? 100 : 1;
      const tx = P ? W / 2 : W * 0.25, ty = P ? H * 0.22 : H * 0.3;
      label(c, `ONE WEIGHT · 2 BYTES READ`, tx, ty - 70, { size: 28, align: 'center', bg: PAPER });
      bigNum(c, `${Bn}`, tx - (P ? 60 : 70), ty + 20, P ? 150 : 170, { align: 'right' });
      label(c, Bn === 1 ? 'TOKEN USES IT' : 'TOKENS USE IT', tx - (P ? 40 : 50), ty + 20, { size: 30, bg: PAPER });
      const ex = P ? W / 2 : W * 0.25, ey = P ? H * 0.8 : H * 0.72;
      slab(c, `${Bn} FLOP/BYTE`, ex, ey, P ? 110 : 120, k, { maxW: P ? W * 0.9 : W * 0.44 });
      label(c, `2 × ${Bn} FLOPS ÷ 2 BYTES`, ex, ey + (P ? 80 : 88), { size: 26, align: 'center', bg: PAPER });
    } else if (kind === 'tower') {
      const s = this.ctx.story.at(ft);
      const kk = s.emitted - 1;
      const tok = kk >= 0 ? showTok(llm.steps[kk]!.tok.t) : '';
      const tx = P ? W / 2 : W * 0.3, ty = P ? H * 0.72 : H * 0.7;
      slab(c, `${tok} ×100`, tx, ty, P ? 150 : 170, snapIn(ft, this.ctx.story.emits[kk] ?? t0, 0.07), { maxW: P ? W * 0.9 : W * 0.5, knock: true });
      label(c, 'ONE STEP · ONE READ OF THE WEIGHTS · 100 TOKENS OUT', tx, ty + (P ? 100 : 115), { size: P ? 22 : 26, align: 'center', bg: PAPER });
    } else {
      this.wall(c, ft, alt, t0);
    }
  }

  /**
   * 100 copies writing in lock-step at the measured rate (each copy gets 1/100 of 1,805.5 tok/s). The speed run
   * decoded greedily, so the copies write the 8B's greedy picks (llm.d.spec.rounds[k].big[0]); those are known
   * for k = 0..9 only (after the greedy ' "' the path leaves the recorded one), and the copies never run ahead of
   * the story: at most the main answer's position plus the step in flight, and never past k = 9.
   */
  private wall(c: CanvasRenderingContext2D, ft: number, alt: number, t0: number) {
    const P = portrait(), N = this.N, llm = this.ctx.llm;
    const greedy = llm.d.spec.rounds.slice(0, 10).map((r) => showTok(r.big[0]!));
    const perCopy = N.s100 / 100; // tokens per second per copy
    const limit = Math.min(greedy.length, this.ctx.story.at(ft).step + 1);
    // the last few steps type in at the measured rate (one step every 55 ms), then it holds at the limit
    const n = clamp(limit - 3 + Math.floor((ft - t0) * perCopy), 1, limit);
    const toks = greedy.slice(0, limit);
    const top = safeTop();
    // panel
    const side = P ? 0 : [0, -1, 1, 0][alt % 4]!;
    const x0 = P ? 60 : side < 0 ? W * 0.42 : 150, x1 = P ? W - 60 : side > 0 ? W * 0.58 : W - 150;
    const y0 = top + (P ? 40 : 10), y1 = P ? H - 420 : H - 150;
    c.fillStyle = PAPER;
    c.fillRect(x0 - 16, y0 - 12, x1 - x0 + 32, y1 - y0 + 24);
    c.fillStyle = INK;
    c.fillRect(x0 - 16, y0 - 12, x1 - x0 + 32, 6);
    // header
    const narrow = P || side !== 0;
    label(c, `100 COPIES · ${fmt1(N.s100)} TOK/S · MEASURED · GREEDY`, x0, y0 + 26, { size: narrow ? 24 : 30, color: INK });
    const sub = `${fmtInt(n * 100)} TOKENS WRITTEN · ONE STEP PER ${fmt1(1000 / perCopy)} MS · COMPUTED`;
    label(c, sub, x0, y0 + (narrow ? 62 : 66), { size: narrow ? 22 : 24, color: PINK });
    // copy 1, big, with the real greedy picks: written ones in ink, the newest in pink
    const fs = P ? 30 : 34;
    c.font = font(F.mono(700), fs);
    let x = x0, y = y0 + (narrow ? 120 : 128);
    for (let j = 0; j < n; j++) {
      const w = c.measureText(toks[j]!).width + 16;
      if (x + w > x1) { x = x0; y += fs * 1.6; }
      c.fillStyle = j === n - 1 ? PINK : INK;
      c.fillRect(x, y - fs * 0.85, w, fs * 1.25);
      c.fillStyle = PAPER;
      c.fillText(toks[j]!, x + 8, y + fs * 0.12);
      x += w + 6;
    }
    // the other 99 copies: rows of blocks (token widths from the real strings), same head
    const cols = narrow ? 2 : 3, rowsPer = Math.ceil(99 / cols);
    const gy0 = y + fs * 1.1, gy1 = y1 - 8;
    const rh = (gy1 - gy0) / rowsPer, cw = (x1 - x0 - (cols - 1) * 24) / cols;
    const chars = greedy.reduce((a, t) => a + t.length + 1, 0);
    const px = cw / chars;
    for (let rI = 0; rI < 99; rI++) {
      const col = Math.floor(rI / rowsPer), row = rI % rowsPer;
      let bx = x0 + col * (cw + 24);
      const by = gy0 + row * rh;
      for (let j = 0; j < n; j++) {
        const w = (toks[j]!.length + 1) * px - 2;
        c.fillStyle = j === n - 1 ? PINK : hash(rI, j) < 0.5 ? INK : BLUE;
        c.fillRect(bx, by, w, Math.max(3, rh - 3));
        bx += w + 2;
      }
    }
  }
}
