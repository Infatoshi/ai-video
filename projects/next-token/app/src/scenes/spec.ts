// spec (verse 4): "Stuck on the slope, it's the roofline again, / So read the weights once and serve a
// hundred of them, / Or a little model guesses four words in a row, / And the big one checks all four in
// one go". Cuts every 2 beats, counted from the first line's cut, so each lyric line gets four shots:
//   line 1  the roofline (3D chart): batch 1 stuck on the memory slope, from the front, down the slope,
//           the whole line traced, then the 15.01 GB a decode step reads streaming in for every token
//   line 2  batching: the tower read once, the read fanning out to 100 prompt cards (2D then 3D, a
//           token landing in each), the dot jumping to batch 100 on the chart (measured 1,805.5 tok/s)
//   line 3  speculative decoding: Llama 3.2 1B (16 floors) next to the 8B, its four greedy guesses
//           popping out one pass each, the row, the row flying to the big tower
//   line 4  the 8B checks all four in one pass: the row enters the lobby, climbs the 32 floors together,
//           the check (the 8B's own picks under the guesses, ticks), and the four land on "go"
// Real data: ctx.llm.d.spec (at, draft, big, accepted, rounds), speed_tok_s, config; ctx.story's clock.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { clamp, ease, hash } from '../engine/util';
import { showTok } from '../engine/llm';
import { Wash, specks, makeCam, drive, shot, snapIn, portrait, perFrame, tokenTile, whip, inked } from './_kit';
import { Tower } from './_tower';
import {
  PINK, BLUE, INK, PAPER, CH, chX, buildChart, buildGrid, buildSheets, type Chart, type Grid, mono, arch, txt, box, slab, stamp, tile2, tick, cross, rain,
} from './spec-parts';

const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const SMALL_POS = V(-8, 0, 0);
const SMALL_S = 0.6;
const FRONT_Z = 5.2 / 2 + 0.75; // the big tower's facade, where the four tiles climb

export default class Spec extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(40);
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  front = new Layer2D();
  big = new Tower();
  small = new Tower({ layers: 16 });
  chart!: Chart;
  grid!: Grid;
  sheets!: THREE.InstancedMesh;
  vram!: THREE.Mesh;
  drafts: THREE.Mesh[] = [];
  ctxTiles: THREE.Mesh[] = [];
  /** x offsets of the draft row (centred) and of the context + draft row */
  rowX: number[] = [];
  ctxRowX: number[] = [];
  pool: { id: number; t: string }[] = [];
  tm = { c0: 0, l2: 0, l3: 0, l4: 0, stuck: 0, roofline: 0, hundred: 0, guesses: 0, go: 0, end: 0 };
  kvShown = -1;
  private tmp = new THREE.Object3D();

  override init() {
    const { lyrics: ly, llm, story, audio: au } = this.ctx;
    const line = (q: string) => ly.get(q);
    const word = (q: string, w: string) => (line(q).words.find((x) => x.w.toLowerCase().replace(/[^a-z']/g, '').startsWith(w)) ?? line(q).words[0]!).start;
    const cutOf = (q: string) => au.timeOfBeat(Math.floor(au.beatAt(line(q).words[0]!.start + 0.02)));
    const spec = llm.d.spec;
    this.tm = {
      c0: cutOf('Stuck on the slope'), l2: cutOf('read the weights once'), l3: cutOf('a little model guesses'), l4: cutOf('checks all four'),
      stuck: word('Stuck on the slope', 'stuck'), roofline: word('Stuck on the slope', 'roofline'), hundred: word('read the weights once', 'hundred'),
      guesses: word('a little model guesses', 'guesses'), go: story.emits[spec.at] ?? word('checks all four', 'go'), end: this.ctx.end,
    };
    this.pool = [...llm.d.prompt, ...llm.steps.map((s) => s.tok)];

    // 3D: the chart, the grid, the weight sheets, the two towers, the tiles
    this.chart = buildChart(llm.d.speed_tok_s['1'] ?? 1, llm.d.speed_tok_s['100'] ?? 1);
    this.scene.add(this.chart.group);
    this.sheets = buildSheets();
    this.scene.add(this.sheets);
    // the memory the weights live in: a black-inked block
    this.vram = inked(new THREE.BoxGeometry(3.4, 2.6, 2.6), { ink: 'ink', lit: 0.3, shade: 0.95, over: 'blue', overLit: 0, overShade: 0.5 }, 2.4);
    this.vram.position.set(-11, 4.2, -3);
    this.scene.add(this.vram);
    const stepTok = llm.steps[spec.at]?.tok.t ?? spec.draft[0]!;
    this.grid = buildGrid(tokenTile(stepTok, { face: 'pink', text: 'paper', side: 'pink', h: 0.5 }));
    this.scene.add(this.grid.group);

    this.big.outlineAll();
    this.small.outlineAll();
    this.big.rider.visible = false;
    this.small.rider.visible = false;
    this.small.group.position.copy(SMALL_POS);
    this.small.group.scale.setScalar(SMALL_S);
    this.scene.add(this.big.group, this.small.group);

    const th = 0.7, gap = 0.16;
    this.drafts = spec.draft.map((t) => tokenTile(t, { face: 'pink', text: 'paper', side: 'blue', h: th }));
    const widths = (ms: THREE.Mesh[]) => ms.map((m) => ((m.geometry as THREE.BoxGeometry).parameters.width));
    const offs = (ws: number[]) => { const tot = ws.reduce((a, b) => a + b, 0) + gap * (ws.length - 1); let x = -tot / 2; return ws.map((w) => { const c = x + w / 2; x += w + gap; return c; }); };
    this.rowX = offs(widths(this.drafts));
    const ctxToks = llm.steps.slice(Math.max(0, spec.at - 3), spec.at).map((s) => s.tok.t);
    this.ctxTiles = ctxToks.map((t) => tokenTile(t, { face: 'ink', text: 'paper', side: 'blue', h: th }));
    this.ctxRowX = offs(widths([...this.ctxTiles, ...this.drafts]));
    for (const m of [...this.drafts, ...this.ctxTiles]) this.scene.add(m);
  }

  // ---------------------------------------------------------------- staging (pure in t)

  /**
   * Which idea a shot shows: four per lyric line. (The streak into the finale switches over at the cut after
   * "go", so the last idea holds through the overlap and nothing after it is ever seen.)
   */
  private idea(ft: number, i: number): number {
    const au = this.ctx.audio;
    const b0 = Math.round(au.beatAt(this.tm.c0));
    const first = (tc: number) => Math.max(0, Math.floor((Math.round(au.beatAt(tc)) - b0 + 1e-3) / 2));
    const lineIdx = ft < this.tm.l2 ? 0 : ft < this.tm.l3 ? 1 : ft < this.tm.l4 ? 2 : 3;
    const start = [0, first(this.tm.l2), first(this.tm.l3), first(this.tm.l4)][lineIdx]!;
    return lineIdx * 4 + Math.min(3, Math.max(0, i - start));
  }

  private look(pos: THREE.Vector3, at: THREE.Vector3, fov: number, shiftRight = 0, shiftUp = 0) {
    const cam = this.cam;
    cam.up.set(0, 1, 0);
    cam.position.copy(pos);
    const fwd = at.clone().sub(pos).normalize();
    const right = fwd.clone().cross(cam.up).normalize();
    const up = right.clone().cross(fwd).normalize();
    cam.lookAt(at.clone().addScaledVector(right, shiftRight).addScaledVector(up, shiftUp));
    if (cam.fov !== fov || cam.aspect !== W / H) { cam.fov = fov; cam.aspect = W / H; cam.updateProjectionMatrix(); }
    cam.updateMatrixWorld();
  }

  /** Grid time of the n-th 8th (or 16th with div 4) after t0. */
  private grid8(t0: number, n: number, div = 2) {
    const au = this.ctx.audio;
    return au.timeOfBeat(Math.round(au.beatAt(t0) * div) / div + n / div);
  }

  /** Put every object and the camera where idea `k` wants them at time t. */
  private stage(k: number, t: number, t0: number, t1: number) {
    const au = this.ctx.audio, P = portrait();
    const dd = drive(au, t, 0.6, 1.4) - drive(au, t0, 0.6, 1.4);
    const u = clamp((t - t0) / Math.max(1e-3, t1 - t0));
    const ch = this.chart, big = this.big, small = this.small;
    const Ht = big.height;
    const showChart = [0, 1, 2, 3, 7].includes(k);
    ch.group.visible = showChart;
    this.sheets.visible = k === 3;
    this.vram.visible = k === 3;
    this.grid.group.visible = k === 6;
    big.group.visible = [4, 5, 8, 9, 10, 11, 12, 13, 14, 15].includes(k);
    small.group.visible = [8, 9, 10, 11].includes(k);
    const tilesOn = [9, 10, 11, 12, 13, 14, 15].includes(k);
    this.drafts.forEach((m) => { m.visible = tilesOn; m.rotation.set(0, 0, 0); m.scale.setScalar(1); });
    this.ctxTiles.forEach((m) => { m.visible = k === 10; });
    big.light(-1);
    small.light(-1);

    // KV shelves: the context so far (it grows by four on "go")
    const n = this.ctx.story.at(t).pos;
    if (n !== this.kvShown) { big.setKV(n); small.setKV(n); this.kvShown = n; }

    const smallTop = SMALL_POS.clone().setY(small.height * SMALL_S);
    const bl = au.timeOfBeat(Math.round(au.beatAt(t0)) + 1) - au.timeOfBeat(Math.round(au.beatAt(t0)));

    if (showChart) {
      // batch-1 dot: on every beat it hops up the slope and slides back (stuck)
      const b = au.beatAt(t), bt = au.timeOfBeat(Math.floor(b));
      const hop = k <= 3 ? snapIn(t, bt, 0.06) * (1 - ease.inQuad(clamp((t - bt - 0.06) / (bl * 0.8)))) * 0.28 : 0;
      ch.dot1.position.copy(ch.at(hop, ch.line(hop), 0.25));
      const jump = k === 7 ? snapIn(t, this.grid8(t0, 1), 0.09) : 0;
      ch.dot100.visible = k === 7;
      ch.dot100.position.copy(ch.at(2, ch.v1 + (ch.v100 - ch.v1) * jump, 0.25));
      ch.dot1.visible = true;
      // pen tracing the roofline (idea 2)
      ch.pen.visible = k === 2;
      if (k === 2) {
        const pu = -0.6 + 3.6 * clamp(u * 1.25);
        ch.pen.position.copy(ch.at(pu, ch.line(pu), 0.2)).add(V(0, 0.24, 0));
      }
    }

    switch (k) {
      case 0: { // the chart, straight on: batch 1 stuck at the foot of the slope
        if (P) this.look(V(4.4 - dd * 0.2, 3.2, 30 - dd * 0.7), V(4.4 - dd * 0.2, 3.9, 0), 36);
        else this.look(V(6.2 - dd * 0.3, 3.6, 17.5 - dd * 0.7), V(6.2 - dd * 0.3, 3.6, 0), 36);
        break;
      }
      case 1: { // close on the stuck dot, the slope running out of frame toward the roof
        if (P) this.look(V(0.6 + dd * 0.25, 1.6, 8.5 - dd * 0.4), V(3.2, 3.2, 0), 52);
        else this.look(V(0.8 + dd * 0.3, 1.4, 6.8 - dd * 0.4), V(3.9, 2.8, 0), 48);
        break;
      }
      case 2: { // three-quarter, orbiting, the line traced
        const a = 0.62 + dd * 0.07, R = P ? 25 : 17;
        this.look(V(6 + Math.sin(a) * R, P ? 10 : 8, Math.cos(a) * R), V(6, 3.2, -0.5), P ? 42 : 38);
        break;
      }
      case 3: { // the weights streaming in for every token
        if (P) this.look(V(-17 - dd * 0.3, 7.5, 7), V(-1, 2.6, -1), 55);
        else this.look(V(-6 - dd * 0.3, 4.6, 13), V(-2.5, 2.8, -1.5), 50);
        const src = this.vram.position, dst = ch.dot1.position;
        const dr = drive(au, t, 0.6, 1.4);
        for (let i = 0; i < this.sheets.count; i++) {
          const s = ((dr * 0.55 * (0.8 + hash(i, 2) * 0.5) + hash(i, 1)) % 1 + 1) % 1;
          const p = src.clone().lerp(dst, s);
          p.x += (hash(i, 3) - 0.5) * 3.2 * (1 - s * 0.8);
          p.y += (hash(i, 4) - 0.5) * 2.8 * (1 - s * 0.8) + Math.sin(s * Math.PI) * 1.6;
          p.z += (hash(i, 6) - 0.5) * 3.0 * (1 - s * 0.8);
          this.tmp.position.copy(p);
          this.tmp.rotation.set(dr * (0.8 + hash(i, 7)) + i, dr * 0.6 + i * 0.3, 0.3 * Math.sin(dr + i));
          this.tmp.scale.setScalar(0.35 + 0.65 * Math.min(1, (1 - s) * 5));
          this.tmp.updateMatrix();
          this.sheets.setMatrixAt(i, this.tmp.matrix);
        }
        this.sheets.instanceMatrix.needsUpdate = true;
        break;
      }
      case 4: { // the tower read once, bottom to top
        big.light(clamp(u * 1.1), 1);
        const a = 0.55 + dd * 0.06;
        if (P) this.look(V(Math.sin(a) * 42, 6, Math.cos(a) * 42), V(0, 9.5, 0), 40, 0, 5);
        else this.look(V(Math.sin(a) * 27, 4.5, Math.cos(a) * 27), V(0, 10.3, 0), 42, -6.5);
        break;
      }
      case 5: { // the tower small on the left; the fan is drawn in 2D
        big.light(1, 1);
        const a = -0.35 + dd * 0.05;
        if (P) this.look(V(Math.sin(a) * 124, 4, Math.cos(a) * 124), V(0, 10.3, 0), 40, 0, -25.6);
        else this.look(V(Math.sin(a) * 38, 6, Math.cos(a) * 38), V(0, 10.3, 0), 42, 10.5, 0.5);
        break;
      }
      case 6: { // 100 prompt cards, one read, a token lands in each
        this.stageGrid(t);
        const a = -0.18 + dd * 0.05;
        if (P) this.look(V(Math.sin(a) * 23, 17, Math.cos(a) * 23), V(0, 2.6, -2.2), 52);
        else this.look(V(Math.sin(a) * 16, 11, Math.cos(a) * 16), V(0, 2.8, -2.2), 46);
        break;
      }
      case 7: { // back on the chart: the dot jumps to batch 100
        if (P) this.look(V(-3.5 + dd * 0.3, 1.8, 18), V(5.2, 4.3, -0.5), 44);
        else this.look(V(-4.2 + dd * 0.35, 1.6, 11.5), V(6.6, 4.1, -0.5), 46);
        break;
      }
      case 8: { // the little model next to the big one
        small.light((drive(au, t, 0.6, 1.4) * 0.9) % 1, 1);
        if (P) this.look(V(-2.5 + dd * 0.3, 10, 47), V(-2.5, 9.5, 0), 40);
        else this.look(V(-3.5 + dd * 0.4, 9.5, 40), V(-3.5, 9.2, 0), 40);
        break;
      }
      case 9: { // it guesses: one full pass of the 16 floors per guess, a tile pops out each time
        for (let g = 0; g < 4; g++) {
          const tg = this.grid8(t0, g), te = this.grid8(t0, g + 1);
          if (t >= tg && t < te) small.light(clamp((t - tg) / (te - tg)), 1);
        }
        this.drafts.forEach((m, g) => {
          const kk = snapIn(t, this.grid8(t0, g), 0.09);
          const slot = smallTop.clone().add(V(this.rowX[g]! + 2.2, 1.9, 0.6));
          m.visible = kk > 0;
          m.position.copy(smallTop.clone().add(V(0, 0.2, 0)).lerp(slot, kk));
          m.scale.setScalar(0.2 + 0.8 * kk);
        });
        if (P) this.look(V(SMALL_POS.x + 2.6, 9.4, SMALL_POS.z + 13.5 - dd * 0.4), V(SMALL_POS.x + 2.0, 9.8, SMALL_POS.z), 50);
        else this.look(V(SMALL_POS.x + 4.2, 8.2, SMALL_POS.z + 9.6 - dd * 0.5), V(SMALL_POS.x + 0.4, 6.6, SMALL_POS.z), 46);
        break;
      }
      case 10: { // the row: the context's last three tokens, then the four guesses
        const row = V(-2, 10.5, 9.5);
        [...this.ctxTiles, ...this.drafts].forEach((m, g) => {
          const isDraft = g >= this.ctxTiles.length;
          const kk = isDraft ? snapIn(t, this.grid8(t0, g - this.ctxTiles.length, 4), 0.08) : 1;
          m.position.copy(row).add(V(this.ctxRowX[g]!, (1 - kk) * 1.5, 0));
          m.visible = kk > 0;
        });
        const end = this.ctxRowX[this.ctxRowX.length - 1]!, beg = this.ctxRowX[0]!;
        const dm = (this.ctxRowX[this.ctxTiles.length]! + end) / 2;
        if (P) this.look(V(row.x + dm - 1.6 + dd * 0.3, row.y + 1.3, row.z + 9), V(row.x + dm - 0.3, row.y - 0.2, row.z), 60);
        else this.look(V(row.x + beg + 2.4 + dd * 0.9, row.y + 0.9, row.z + 5.4), V(row.x + beg + 3.6 + dd * 0.9, row.y, row.z), 44);
        break;
      }
      case 11: { // the row flies from the little tower to the big one's door
        this.drafts.forEach((m, g) => {
          const from = smallTop.clone().add(V(this.rowX[g]! + 2.2, 1.9, 0.6));
          const to = V(this.rowX[g]!, 0.75, FRONT_Z + 0.4);
          const ts = this.grid8(t0, 1, 4) + g * 0.03;
          const s = ease.outCubic(clamp((t - ts) / (bl * 1.1)));
          m.position.copy(from.lerp(to, s)).add(V(0, Math.sin(s * Math.PI) * 4.5, 0));
          m.rotation.y = (1 - s) * 0.5;
          m.scale.setScalar(1 + 1.6 * Math.sin(s * Math.PI)); // big as it passes the lens
        });
        const w = whip(au, t, t0, 0.2);
        const a = -0.25 + (1 - w) * 0.9 + dd * 0.04;
        const R = P ? 30 : 19;
        this.look(V(-3.5 + Math.sin(a) * R, P ? 9 : 7.5, Math.cos(a) * R), V(-4, 6.2, 0), P ? 50 : 46);
        break;
      }
      case 12: { // the big one: low, the four at its door
        this.drafts.forEach((m, g) => {
          const push = snapIn(t, this.grid8(t0, 2), 0.09);
          m.position.set(this.rowX[g]!, 0.75, FRONT_Z + 0.4 - push * 0.35);
        });
        big.light(0.02 + 0.03 * snapIn(t, this.grid8(t0, 2), 0.09), 1);
        if (P) this.look(V(-4.5 + dd * 0.25, 2.6, 17), V(0.6, 8, 0), 66);
        else this.look(V(-6.2 + dd * 0.3, 3.0, 13), V(0.5, 5.8, 0), 58, -2.5);
        break;
      }
      case 13: { // all four climb the 32 floors together: one pass
        const c = clamp((t - t0 - 0.03) / Math.max(0.2, (t1 - t0) * 0.92));
        big.light(c, 1);
        const y = big.floorY(c * (big.layers + 1)) + 0.35;
        this.drafts.forEach((m, g) => m.position.set(this.rowX[g]!, y, FRONT_Z));
        if (P) this.look(V(7 + dd * 0.3, y + 3.2, 18), V(0, y + 0.2, 1.5), 50);
        else this.look(V(8 + dd * 0.35, y + 2.2, 12.5), V(0, y + 0.3, 1.5), 44, 2.4);
        break;
      }
      case 14: { // the check (2D table), the tower top behind
        big.light(1, 1);
        this.drafts.forEach((m, g) => m.position.set(this.rowX[g]!, Ht + 0.5, FRONT_Z - 0.4));
        const a = 0.9 + dd * 0.06;
        if (P) this.look(V(Math.sin(a) * 15, Ht + 9, Math.cos(a) * 15), V(0, Ht - 3, 0), 55);
        else this.look(V(Math.sin(a) * 13, Ht + 6, Math.cos(a) * 13), V(0, Ht - 2, 0), 50);
        break;
      }
      case 15: { // on "go": the four come out of the roof together
        big.light(1, 1);
        const go = snapIn(t, this.tm.go, 0.09);
        this.drafts.forEach((m, g) => m.position.set(this.rowX[g]!, Ht + 0.2 + go * (P ? 1.9 : 1.3), 0.4));
        // landscape: the row sits low, under the overlay's ONE PASS band; portrait: above it
        if (P) this.look(V(1.5 + dd * 0.3, Ht + 5.2, 14.5), V(0, Ht - 0.6, 0), 52);
        else this.look(V(1.2 + dd * 0.35, Ht + 4.9, 10.2), V(0, Ht + 3.4, 0), 46);
        break;
      }
    }
  }

  private stageGrid(t: number) {
    const G = this.grid, hundred = this.tm.hundred, au = this.ctx.audio;
    // the read: rays reach out from the weights just before "hundred", tokens land in a wave on it
    const r0 = this.grid8(hundred, -1);
    const land = au.timeOfBeat(Math.round(au.beatAt(hundred) * 4) / 4);
    for (let i = 0; i < 100; i++) {
      const c = G.cell(i);
      const dist = Math.hypot(c.x, c.z) / 8;
      const rk = clamp((t - r0 - dist * 0.08) / 0.1);
      const src = G.source.position.clone().add(V((c.x / 6.5) * 2.4, -0.35, (c.z / 5) * 1.3));
      const dst = c.clone().add(V(0, 0.08, 0));
      this.tmp.position.copy(src);
      this.tmp.lookAt(dst);
      this.tmp.scale.set(0.045, 0.045, src.distanceTo(dst) * ease.outExpo(rk));
      this.tmp.updateMatrix();
      G.rays.setMatrixAt(i, this.tmp.matrix);
      const lk = snapIn(t, land + dist * 0.12, 0.08);
      this.tmp.position.set(c.x, 0.1 + (1 - lk) * 3.5, c.z);
      this.tmp.rotation.set(-Math.PI / 2, 0, 0);
      this.tmp.scale.setScalar(lk > 0 ? 1 : 0);
      this.tmp.updateMatrix();
      G.tiles.setMatrixAt(i, this.tmp.matrix);
      this.tmp.rotation.set(0, 0, 0);
      this.tmp.scale.setScalar(1);
    }
    G.rays.instanceMatrix.needsUpdate = true;
    G.tiles.instanceMatrix.needsUpdate = true;
    ((G.source.material as THREE.ShaderMaterial).uniforms.hot!).value = t >= r0 ? 0.85 : 0;
  }

  /** World point to screen px with the current camera. */
  private scr(v: THREE.Vector3): [number, number] {
    const p = v.clone().project(this.cam);
    return [(p.x + 1) / 2 * W, (1 - p.y) / 2 * H];
  }
  private wpos(o: THREE.Object3D): THREE.Vector3 {
    o.updateWorldMatrix(true, false);
    return new THREE.Vector3().setFromMatrixPosition(o.matrixWorld);
  }

  // ---------------------------------------------------------------- render

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio;
    const sh = shot(this.ctx, f, { every: 2, from: this.tm.c0 });
    const k = this.idea(f.ft, sh.i);
    const P = portrait();
    const d = drive(au, f.t, 0.6, 1.6);
    const line = Math.floor(k / 4);

    // background: wash (reseeded per shot) and a rain of the run's real tokens
    const inkA = line % 2 ? 'blue' : 'pink', inkB = line % 2 ? 'pink' : 'blue';
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: k === 5 || k === 14 ? 0.35 : 0.5, c1: inkA, c2: inkB });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => rain(this.back.ctx, this.pool, drive(au, f.ft, 0.6, 1.6) * 0.45, { seed: sh.seed, cols: P ? 6 : 10 })), out);

    // hero: 3D
    this.stage(k, f.t, sh.t0, sh.t1);
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);

    // type: once per frame, anchored to the 3D at the frame's own time
    this.ctx.comp.draw(r, perFrame(this.mid, f, () => {
      this.stage(k, f.ft, sh.t0, sh.t1);
      this.type(this.mid.ctx, k, f.ft, sh.t0, sh.t1);
    }), out);

    // foreground: ink specks rushing past
    const c = this.front.ctx;
    this.front.clear();
    specks(c, d * (k === 13 || k === 11 ? 1.8 : 1), { seed: sh.seed, n: k === 5 || k === 14 ? 45 : 75 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    return { reg: sh.seed };
  }

  // ---------------------------------------------------------------- type per idea

  private type(c: CanvasRenderingContext2D, k: number, t: number, t0: number, t1: number) {
    const llm = this.ctx.llm, spec = llm.d.spec, P = portrait();
    const ch = this.chart;
    const s1 = llm.d.speed_tok_s['1'] ?? 0, s100 = llm.d.speed_tok_s['100'] ?? 0;
    const GB = llm.cfg.weight_bytes / 1e9; // the model's size
    // what one decode step reads: every weight except the embedding table (one row of it is looked up per token)
    const READ = (llm.cfg.weight_bytes - llm.cfg.vocab * llm.cfg.hidden * 2) / 1e9;
    const k0 = snapIn(t, t0, 0.08);
    const at8 = (n: number, div = 2) => snapIn(t, this.grid8(t0, n, div), 0.08);
    const chartAt = (u: number, v: number, z = 0) => this.scr(ch.group.localToWorld(ch.at(u, v, z)));
    const along = (u0: number, u1: number) => { const a = chartAt(u0, ch.line(u0)), b = chartAt(u1, ch.line(u1)); return Math.atan2(b[1] - a[1], b[0] - a[0]); };
    const rot = (s: string, x: number, y: number, ang: number, f: string, col: string, off = 0) => {
      c.save(); c.translate(x, y); c.rotate(ang); txt(c, s, 0, -off, f, col, 'center', 'bottom'); c.restore();
    };
    const chartLabels = (roof = true, slope = true) => {
      const um = 1.15, [sx, sy] = chartAt(um, ch.line(um));
      if (slope) rot('MEMORY-BOUND', sx, sy, along(0.7, 1.6), arch(P ? 30 : 34), INK, 22);
      if (roof) { const [rx, ry] = chartAt(P ? ch.ridgeU : (ch.ridgeU + 3) / 2, ch.roofV); txt(c, 'COMPUTE ROOF', rx, ry - 26, arch(P ? 26 : 30), INK, 'center', 'bottom'); }
      // tick labels below the tick marks (anchored in 3D under them, so no angle puts ink on ink)
      const under = (u: number) => this.scr(ch.group.localToWorld(V(chX(u), -0.62, 0.05)));
      [0, 1, 2].forEach((u, i) => { const [x, y] = under(u); txt(c, ['1', '10', '100'][i]!, x, y + 6, mono(26), INK, 'center', 'top'); });
      const [ax, ay] = under(2.75);
      txt(c, 'FLOP PER BYTE →', ax, ay + 6, mono(24), INK, 'center', 'top');
      const [yx, yy] = chartAt(CH.uMin, 3.9);
      txt(c, '↑ TOK/S', yx + 16, yy - 6, mono(24), INK, 'left', 'bottom');
    };
    const dotLabel = (m: THREE.Mesh, head: string, big: string, sub: string, bg: string, side: 1 | -1 = 1, dy = 90) => {
      const [x, y0] = this.scr(this.wpos(m));
      const y = y0 + dy;
      const align: CanvasTextAlign = side > 0 ? 'left' : 'right';
      const x0 = x + side * 40;
      box(c, head, x0, y - 30, arch(P ? 34 : 40), bg, PAPER, align);
      box(c, big, x0, y + 26, mono(P ? 30 : 34), PAPER, INK, align, 10, 6);
      if (sub) box(c, sub, x0, y + 70, mono(22), INK, PAPER, align, 10, 5);
    };

    switch (k) {
      case 0: {
        chartLabels();
        dotLabel(ch.dot1, 'BATCH 1', `${s1} TOK/S`, 'MEASURED · RTX 3090', PINK);
        const [x, y] = chartAt(P ? 0.1 : -0.05, P ? 3.45 : 3.3);
        stamp(c, 'STUCK', x, y, P ? 96 : 110, PINK, -0.12, snapIn(t, this.grid8(this.tm.stuck, 0), 0.08));
        break;
      }
      case 1: {
        { const um = 1.15, [sx, sy] = chartAt(um, ch.line(um)); rot('MEMORY-BOUND', sx, sy, along(0.7, 1.6), arch(P ? 30 : 34), INK, 22); }
        dotLabel(ch.dot1, 'BATCH 1', `${s1} TOK/S`, '', PINK);
        const x = P ? W / 2 : 150, y = P ? H * 0.2 : H * 0.22, al: CanvasTextAlign = P ? 'center' : 'left';
        box(c, '1 FLOP PER BYTE', x, y, arch(P ? 64 : 78), INK, PAPER, al, 22, 16);
        box(c, 'BF16: 2 FLOPS PER 2-BYTE WEIGHT · COMPUTED', x, y + (P ? 76 : 86), mono(24), PAPER, INK, al, 12, 8);
        break;
      }
      case 2: {
        const um = 0.7, [sx, sy] = chartAt(um, ch.line(um));
        box(c, 'MEMORY SLOPE', sx - 20, sy - 60, mono(30), PINK, PAPER, 'right');
        const [rx, ry] = chartAt((ch.ridgeU + 3) / 2, ch.roofV);
        box(c, 'COMPUTE ROOF', rx, ry - 64, mono(30), INK, PAPER, 'center');
        const [dx, dy] = this.scr(this.wpos(ch.dot1));
        box(c, 'BATCH 1', dx + 40, dy + 44, arch(34), PINK, PAPER, 'left');
        break;
      }
      case 3: {
        // above the ROOFLINE slam's band (the overlay prints it mid-frame on this shot's first beat)
        const x = P ? W / 2 : W * 0.5, y = P ? H * 0.2 : H * 0.25;
        slab(c, `${READ.toFixed(2)} GB`, x, y, P ? 150 : 160, k0);
        const al: CanvasTextAlign = 'center';
        box(c, 'READ PER TOKEN · COMPUTED', x, y + (P ? 120 : 112), mono(P ? 28 : 30), INK, PAPER, al, 14, 10);
        box(c, `${READ.toFixed(2)} GB × ${s1} TOK/S = ${Math.round(READ * s1)} GB/S · COMPUTED`, x, y + (P ? 176 : 164), mono(P ? 22 : 24), PAPER, INK, al, 10, 7);
        const [vx, vy] = this.scr(this.wpos(this.vram).add(V(0, 1.9, 0)));
        box(c, `WEIGHTS · ${(llm.cfg.params / 1e9).toFixed(2)}B`, vx, vy - 20, mono(28), BLUE, PAPER, 'center');
        break;
      }
      case 4: {
        const x = P ? W / 2 : W * 0.29, y = P ? H * 0.17 : H * 0.42;
        slab(c, 'READ', x, y - (P ? 65 : 90), P ? 150 : 190, k0);
        slab(c, 'ONCE', x, y + (P ? 85 : 100), P ? 150 : 190, at8(1));
        box(c, `${(llm.cfg.params / 1e9).toFixed(2)}B WEIGHTS · ${GB.toFixed(2)} GB`, x, y + (P ? 195 : 250), mono(P ? 26 : 30), INK, PAPER, 'center', 14, 10);
        const u = clamp((t - t0) / Math.max(1e-3, t1 - t0) * 1.1), l = Math.min(32, Math.max(1, Math.round(u * 33)));
        const [lx, ly] = this.scr(V(5.2 / 2 + 0.3, this.big.floorY(l), 5.2 / 2));
        box(c, `L${l}`, lx + 20, ly, mono(30), PINK, PAPER, 'left');
        break;
      }
      case 5: this.fan(c, t, t0); break;
      case 6: {
        const [sx, sy] = this.scr(this.wpos(this.grid.source).add(V(0, 0.6, 0)));
        box(c, `WEIGHTS · ${READ.toFixed(2)} GB READ ONCE`, sx, sy - 34, mono(P ? 26 : 28), BLUE, PAPER, 'center');
        box(c, `100 PROMPTS · 1 READ · 100 TOKENS`, W / 2, P ? H * 0.74 : H * 0.855, mono(P ? 28 : 32), INK, PAPER, 'center', 16, 10);
        break;
      }
      case 7: {
        const jump = snapIn(t, this.grid8(t0, 1), 0.09);
        chartLabels(false, false);
        // the jump's trail: dashed from batch 1 to batch 100
        if (jump > 0) {
          const a = this.scr(this.wpos(ch.dot1)), b = this.scr(this.wpos(ch.dot100));
          c.save(); c.strokeStyle = PINK; c.lineWidth = 7; c.setLineDash([18, 12]);
          c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(a[0] + (b[0] - a[0]) * jump, a[1] + (b[1] - a[1]) * jump); c.stroke(); c.restore();
        }
        dotLabel(ch.dot1, 'BATCH 1', `${s1} TOK/S`, '', INK, 1, 80);
        if (jump > 0.5) dotLabel(ch.dot100, 'BATCH 100', `${s100.toLocaleString('en-US')} TOK/S`, 'MEASURED · SAME 3090', PINK, -1, -150);
        const [x, y] = P ? [W / 2, H * 0.22] : [W * 0.75, H * 0.32];
        slab(c, `${(s100 / s1).toFixed(1)}×`, x, y, P ? 150 : 165, at8(2));
        if (at8(2) > 0) box(c, `${s100.toLocaleString('en-US')} ÷ ${s1} · COMPUTED`, x, y + (P ? 110 : 130), mono(24), PAPER, INK, 'center', 10, 7);
        break;
      }
      case 8: {
        const [sx, sy] = this.scr(SMALL_POS.clone().setY(this.small.height * SMALL_S + 0.4));
        const [bx, by] = this.scr(V(5.2 / 2 + 0.6, this.big.height * 0.72, 0));
        slab(c, '1B', sx, sy - (P ? 150 : 170), P ? 170 : 210, k0);
        box(c, 'LLAMA 3.2 1B · 16 LAYERS', sx, sy - 34, mono(P ? 26 : 30), INK, PAPER, 'center');
        box(c, 'LLAMA 3.1 8B', bx, by, arch(P ? 34 : 40), BLUE, PAPER, P ? 'right' : 'left');
        box(c, '32 LAYERS', bx, by + (P ? 54 : 60), mono(P ? 26 : 30), PAPER, INK, P ? 'right' : 'left', 10, 6);
        break;
      }
      case 9: {
        let n = 0;
        this.drafts.forEach((m, g) => {
          const kk = snapIn(t, this.grid8(t0, g), 0.09);
          if (kk <= 0) return;
          n = g + 1;
          const [x, y] = this.scr(this.wpos(m).add(V(0, 0.62, 0)));
          box(c, String(g + 1), x, y - 16, mono(30), INK, PAPER, 'center', 12, 6);
        });
        const x = P ? W / 2 : 150, y = P ? H * 0.19 : H * 0.23, al: CanvasTextAlign = P ? 'center' : 'left';
        box(c, `GUESS ${n} OF 4`, x, y, arch(P ? 56 : 64), PINK, PAPER, al, 20, 14);
        box(c, '1B · ONE PASS PER GUESS', x, y + (P ? 70 : 78), mono(P ? 26 : 28), INK, PAPER, al, 12, 8);
        break;
      }
      case 10: {
        const nc = this.ctxTiles.length;
        this.drafts.forEach((m, g) => {
          const kk = snapIn(t, this.grid8(t0, g, 4), 0.08);
          if (kk <= 0) return;
          const [x, y] = this.scr(this.wpos(m).add(V(0, -0.64, 0)));
          box(c, String(g + 1), x, y + 30, mono(P ? 30 : 34), PINK, PAPER, 'center', 12, 6);
        });
        const [cx0, cy0] = this.scr(this.wpos(this.ctxTiles[0]!).add(V(0, -0.62, 0)));
        const [cx1] = this.scr(this.wpos(this.ctxTiles[nc - 1]!).add(V(0, -0.62, 0)));
        box(c, 'THE ANSWER SO FAR', (cx0 + cx1) / 2, cy0 + 30, mono(P ? 22 : 24), INK, PAPER, 'center', 10, 6);
        const x = W / 2, y = P ? H * 0.2 : H * 0.22;
        slab(c, '4 GUESSES', x, y, P ? 110 : 140, k0);
        box(c, 'DRAFTED BY LLAMA 3.2 1B, ITS TOP PICK EACH TIME', x, y + (P ? 86 : 100), mono(P ? 22 : 26), INK, PAPER, 'center', 12, 8);
        break;
      }
      case 11: {
        const [sx, sy] = this.scr(SMALL_POS.clone().setY(this.small.height * SMALL_S + 0.3));
        const [bx, by] = this.scr(V(0, this.big.height + 0.4, 0));
        box(c, '1B', sx, sy - 30, arch(40), INK, PAPER, 'center');
        box(c, '8B', bx, by - 30, arch(40), BLUE, PAPER, 'center');
        const x = P ? W / 2 : 150, y = P ? H * 0.2 : H * 0.23, al: CanvasTextAlign = P ? 'center' : 'left';
        box(c, '4 GUESSES → THE 8B', x, y, arch(P ? 52 : 60), PINK, PAPER, al, 20, 14);
        break;
      }
      case 12: {
        const x = P ? W / 2 : W * 0.72, y = P ? H * 0.22 : H * 0.44;
        slab(c, '8B', x, y, P ? 300 : 380, k0);
        box(c, `LLAMA 3.1 8B · ${llm.cfg.layers} LAYERS`, x, y + (P ? 190 : 230), mono(P ? 26 : 30), INK, PAPER, 'center', 14, 10);
        const [tx, ty] = this.scr(this.wpos(this.drafts[0]!).add(V(-0.8, 0.9, 0)));
        box(c, '4 GUESSES IN', tx, ty - 20, mono(28), PINK, PAPER, 'left');
        break;
      }
      case 13: {
        const last = this.drafts[this.drafts.length - 1]!;
        const [x, y] = this.scr(this.wpos(last).add(V(0.7, 0, 0)));
        const cl = clamp((t - t0 - 0.03) / Math.max(0.2, (t1 - t0) * 0.92));
        const l = Math.min(32, Math.max(1, Math.round(cl * 33)));
        if (P) { const [fx, fy] = this.scr(this.wpos(this.drafts[0]!).add(V(0, -0.6, 0))); void fx; box(c, `ALL 4 · ONE PASS · L${l}`, W / 2, fy + 50, mono(30), PINK, PAPER, 'center'); }
        else box(c, `ALL 4 · ONE PASS · L${l}`, x + 24, y, mono(32), PINK, PAPER, 'left');
        const bx = P ? W / 2 : 150, by = P ? H * 0.2 : H * 0.23, al: CanvasTextAlign = P ? 'center' : 'left';
        box(c, 'THE 8B CHECKS', bx, by, arch(P ? 56 : 64), INK, PAPER, al, 20, 14);
        break;
      }
      case 14: this.table(c, t, t0); break;
      case 15: {
        // the verdict (checked on "checks all four", shot 14) stays on; the four are emitted on "go"
        this.drafts.forEach((m, g) => {
          const [x, y] = this.scr(this.wpos(m).add(V(0, 0.66, 0)));
          const kv = snapIn(t, this.grid8(t0, 0, 4) + g * 0.03, 0.07);
          if (g < spec.accepted) tick(c, x, y - 34, 52, PINK, kv);
          else if (g === spec.accepted) cross(c, x, y - 34, 50, INK, kv);
        });
        const x = W / 2, y = P ? H * 0.18 : H * 0.2;
        const kept = spec.accepted;
        const all = spec.rounds.filter((r) => r.accepted === r.draft.length).length;
        box(c, `${spec.draft.length} GUESSED · ${kept} KEPT · 1 PASS`, x, y, arch(P ? 48 : 56), INK, PAPER, 'center', 22, 14);
        box(c, `${all} OF ${spec.rounds.length} ROUNDS IN THIS ANSWER KEPT ALL 4`, x, y + (P ? 66 : 70), mono(P ? 24 : 26), PINK, PAPER, 'center', 12, 8);
        break;
      }
    }
  }

  /** Idea 5: one read of the weights fanning out to a 10 x 10 grid of prompts, a token landing in each. */
  private fan(c: CanvasRenderingContext2D, t: number, t0: number) {
    const P = portrait(), llm = this.ctx.llm;
    const tok = llm.steps[llm.d.spec.at]?.tok.t ?? 'str';
    // landscape: the read leaves the roof; portrait: the tower sits above the grid and the read leaves its lobby
    const [tx, ty] = this.scr(V(0, P ? 0 : this.big.height + 0.3, 0));
    const cell = P ? 70 : 62, pitch = P ? 78 : 68;
    const gx = P ? (W - pitch * 10) / 2 + (pitch - cell) / 2 : W * 0.47, gy = P ? H * 0.4 : H * 0.2;
    const trunkEnd: [number, number] = P ? [W / 2, gy - 50] : [gx - 150, gy + pitch * 5 - (pitch - cell) / 2];
    const tr = snapIn(t, t0, 0.1);
    c.save();
    c.strokeStyle = PINK; c.lineCap = 'butt';
    c.lineWidth = 16;
    c.beginPath(); c.moveTo(tx, ty); c.lineTo(tx + (trunkEnd[0] - tx) * tr, ty + (trunkEnd[1] - ty) * tr); c.stroke();
    // the 100 branches, reaching out on the 16ths (all rays first, then the cards on top of them)
    c.lineWidth = 4;
    const kb = (i: number) => {
      const col = i % 10, row = Math.floor(i / 10);
      const order = P ? row + Math.abs(col - 4.5) * 0.3 : col + Math.abs(row - 4.5) * 0.3;
      return snapIn(t, this.grid8(t0, 1, 4) + order * 0.022, 0.07);
    };
    const cellXY = (i: number) => [gx + (i % 10) * pitch, gy + Math.floor(i / 10) * pitch] as const;
    c.strokeStyle = PINK;
    c.beginPath();
    for (let i = 0; i < 100; i++) {
      const k = kb(i);
      if (k <= 0) continue;
      const [x, y] = cellXY(i);
      const ex = P ? x + cell / 2 : x, ey = P ? y : y + cell / 2;
      c.moveTo(trunkEnd[0], trunkEnd[1]); c.lineTo(trunkEnd[0] + (ex - trunkEnd[0]) * k, trunkEnd[1] + (ey - trunkEnd[1]) * k);
    }
    c.stroke();
    // the prompt cards, then the token landing in each
    c.font = mono(P ? 26 : 22, 700); c.textAlign = 'center'; c.textBaseline = 'middle';
    for (let i = 0; i < 100; i++) {
      const [x, y] = cellXY(i), on = kb(i) >= 0.99;
      c.fillStyle = PAPER; c.fillRect(x, y, cell, cell);
      c.fillStyle = on ? PINK : BLUE; c.fillRect(x, y, cell, cell * 0.2);
      c.strokeStyle = INK; c.lineWidth = 3; c.strokeRect(x, y, cell, cell);
      if (on) { c.fillStyle = INK; c.fillText(showTok(tok), x + cell / 2, y + cell * 0.62); }
    }
    c.restore();
    if (P) {
      box(c, '1 READ', W / 2 + 34, (ty + trunkEnd[1]) / 2, arch(40), PINK, PAPER, 'left');
      txt(c, '100 PROMPTS', gx, gy - 14, arch(44), INK, 'left', 'bottom');
    } else {
      box(c, '1 READ', (tx + trunkEnd[0]) / 2, (ty + trunkEnd[1]) / 2 - 40, arch(44), PINK, PAPER, 'center');
      txt(c, '100 PROMPTS', gx + (pitch * 10) / 2 - (pitch - cell) / 2, gy - 22, arch(48), INK, 'center', 'bottom');
      const read = (llm.cfg.weight_bytes - llm.cfg.vocab * llm.cfg.hidden * 2) / 1e9;
      box(c, `${read.toFixed(2)} GB READ ONCE → 100 TOKENS`, gx + (pitch * 10) / 2, gy + pitch * 10 + 34, mono(26), INK, PAPER, 'center', 12, 8);
    }
  }

  /** Idea 14: the 1B's guesses over the 8B's own picks, a tick for each match, a cross at the first miss. */
  private table(c: CanvasRenderingContext2D, t: number, t0: number) {
    const P = portrait(), spec = this.ctx.llm.d.spec;
    const size = P ? 40 : 52, cellW = P ? 168 : 210, gap = P ? 14 : 20;
    const n = spec.big.length;
    const x0 = P ? (W - (cellW * n + gap * (n - 1))) / 2 : W / 2 - (cellW * n + gap * (n - 1)) / 2 + 110;
    const yA = P ? H * 0.3 : H * 0.3, yB = yA + (P ? 230 : 250);
    const lx = P ? x0 : x0 - 30;
    const al: CanvasTextAlign = P ? 'left' : 'right';
    // row labels
    box(c, '1B GUESSED', lx, P ? yA - 90 : yA, mono(P ? 26 : 28), PINK, PAPER, al, 12, 8);
    box(c, '8B PICKED', lx, P ? yB + 90 : yB, mono(P ? 26 : 28), BLUE, PAPER, al, 12, 8);
    for (let i = 0; i < n; i++) {
      const x = x0 + i * (cellW + gap) + cellW / 2;
      const kA = i < spec.draft.length ? snapIn(t, t0, 0.08) : 0;
      if (kA > 0) tile2(c, spec.draft[i]!, x, yA, size, PINK, PAPER, cellW, 'center');
      else if (i >= spec.draft.length) { c.strokeStyle = INK; c.lineWidth = 4; c.setLineDash([12, 8]); c.strokeRect(x - cellW / 2, yA - size * 0.78, cellW, size * 1.55); c.setLineDash([]); }
      const kB = snapIn(t, this.grid8(t0, 1, 4), 0.08);
      if (kB > 0) tile2(c, spec.big[i]!, x, yB, i === n - 1 ? size * 0.8 : size, i < spec.draft.length ? BLUE : INK, PAPER, cellW, 'center');
      // the verdict, one per 16th
      const kv = snapIn(t, this.grid8(t0, 2 + i, 4), 0.07);
      const ym = (yA + yB) / 2;
      if (i < spec.draft.length) {
        if (i < spec.accepted) tick(c, x, ym, 70, PINK, kv);
        else if (i === spec.accepted) cross(c, x, ym, 66, INK, kv);
      } else {
        const kb = snapIn(t, this.grid8(t0, 2 + i, 4), 0.07);
        if (kb > 0 && spec.accepted === spec.draft.length) txt(c, 'BONUS', x, ym, mono(P ? 24 : 26), INK, 'center', 'middle');
      }
    }
    const ks = snapIn(t, this.grid8(t0, 3), 0.08);
    if (ks > 0) {
      c.save();
      const y = P ? yB + 230 : yB + 150;
      c.translate(W / 2, y); c.scale(1.2 - 0.2 * ks, 1.2 - 0.2 * ks);
      box(c, `${spec.draft.length} GUESSED · ${spec.accepted} KEPT · 1 PASS`, 0, 0, arch(P ? 46 : 60), INK, PAPER, 'center', 22, 14);
      c.restore();
    }
  }
}

