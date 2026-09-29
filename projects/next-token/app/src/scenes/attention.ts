// attention (verse 2, lines 1-4): "Every token looks back at the words that came first, / Asks who matters to
// me, gives a weight to each word, / The strongest ones pull and it blends what they say, / That's attention,
// thirty-two heads, each doing it its own way".
// All real: the step in flight (story.at(t).step), its context (the chat prompt + the answer so far, the newest
// token last and in pink) and llm.attn(step, layer, head), the newest token's weights over that context. Lines
// 1-3 follow one head, LAYER 14 · HEAD 22 (1-based, like the tower's floors; at step 1 "There" puts 67% of it on
// " strawberry"); line 4 is all 32 heads of that layer, then flips up the stack on the beats and ends on the
// tower with every head lit by its weight on the top word. Most heads park most of their weight on the first
// token (an attention sink); every shot that shows it says so.
// Cuts every bar, counted from the beat of line 1's first word, so each lyric line gets two shots.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp } from '../engine/util';
import { LineBatch } from '../engine/lines';
import { showTok } from '../engine/llm';
import { Wash, tokenRain, specks, makeCam, drive, shot, snapIn, portrait, perFrame, tokenTile, inked, type Shot } from './_kit';
import { Tower } from './_tower';
import { drawGrid, drawStamp, drawSlab, headRows, type GridData } from './attention-draw';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

/** The head lines 1-3 follow (data indices, 0-based; shown 1-based). */
const LAYER = 13, HEAD = 21;
/** Layers the head grid flips through on the beats of line 4 (0-based data indices). */
const FLIP = [0, 8, 16, 24, 31];
/** World height of a 100% weight bar. */
const BAR = 8.5;

type Tok = { id: number; t: string };

interface Row {
  s: number;
  toks: Tok[];
  C: number;
  /** The followed head's weights over the context (sums to ~1). */
  w: Float32Array;
  order: number[];
  rank: number[];
  group: THREE.Group;
  tiles: THREE.Mesh[];
  bars: THREE.Mesh[];
  x: number[];
  tw: number[];
  xq: number;
}

export default class Attention extends Scene {
  scene3 = new THREE.Scene();
  cam = makeCam(36);
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  sheetL = new Layer2D();
  front = new Layer2D();
  lines = new LineBatch(8000, { screen2D: false, worldWidth: true, blend: 'normal', depthTest: true });
  rows = new Map<number, Row>();
  sheet!: THREE.Group;
  tower = new Tower();
  from = 0;
  tm = { looks: 0, back: 0, first: 0, matters: 0, weight: 0, pull: 0, blends: 0, attention: 0, heads: 0, tok: 0 };
  grids = new Map<string, GridData>();
  lit = false;
  private v = new THREE.Vector3();

  override init() {
    const { lyrics: ly, audio: au, story } = this.ctx;
    const l1 = ly.get('looks back at the words'), l2 = ly.get('who matters to me'), l3 = ly.get('The strongest ones pull'), l4 = ly.get("That's attention");
    const w = (l: typeof l1, s: string) => (l.words.find((x) => x.w.toLowerCase().replace(/[^a-z-]/g, '').startsWith(s)) ?? l.words[0]!).start;
    this.from = au.timeOfBeat(Math.round(au.beatAt(l1.start)));
    this.tm = {
      tok: w(l1, 'token'), looks: w(l1, 'looks'), back: w(l1, 'back'), first: w(l1, 'first'), matters: w(l2, 'matters'),
      weight: w(l2, 'weight'), pull: w(l3, 'pull'), blends: w(l3, 'blends'), attention: w(l4, 'attention'), heads: w(l4, 'heads'),
    };
    // one row of tiles per step the window can show (the story holds one step through this verse)
    for (let t = this.ctx.start; t <= this.ctx.end; t += 0.25) {
      const s = story.at(t).step;
      if (!this.rows.has(s)) this.rows.set(s, this.build(s));
    }
    // line 4's closing shot: the tower, every head lit by its real weight on the followed head's top word
    this.tower.outlineAll();
    this.tower.ride(tokenTile(this.rowFor(this.ctx.end).toks.at(-1)!.t, { face: 'pink', text: 'paper', side: 'pink', h: 0.5, w: 1.5 }), 0.5);
    this.scene3.add(this.tower.group);
    // the sheet the 32-head grid is printed on for the tilted shot
    this.sheet = new THREE.Group();
    const aspect = W / H, SH = 6;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(SH * aspect, SH), new THREE.MeshBasicMaterial({ map: this.sheetL.texture }));
    const slab = inked(new THREE.BoxGeometry(SH * aspect, SH, 0.18), { ink: 'blue', lit: 0.55, shade: 1 }, 2.4);
    slab.position.z = -0.1;
    plane.position.z = 0.001;
    this.sheet.add(slab, plane);
    this.scene3.add(this.sheet);
  }

  private rowFor(t: number) {
    const s = this.ctx.story.at(t).step;
    let r = this.rows.get(s);
    if (!r) { r = this.build(s); this.rows.set(s, r); }
    return r;
  }

  /** The context as a row of printed tiles along x (newest token last, pink), and a weight bar on each. */
  private build(s: number): Row {
    const llm = this.ctx.llm;
    const toks: Tok[] = [...llm.d.prompt, ...llm.steps.slice(0, s).map((x) => x.tok)];
    const C = toks.length;
    const w = llm.attn(s, LAYER, HEAD);
    const order = [...Array(C).keys()].sort((a, b) => w[b]! - w[a]!);
    const rank: number[] = new Array(C);
    order.forEach((i, r) => (rank[i] = r));
    const group = new THREE.Group();
    const tiles: THREE.Mesh[] = [], bars: THREE.Mesh[] = [], x: number[] = [], tw: number[] = [];
    const GAP = 0.2;
    let acc = 0;
    const barGeo = new THREE.BoxGeometry(1, 1, 1);
    toks.forEach((tk, i) => {
      const q = i === C - 1;
      const len = showTok(tk.t).length;
      const width = q ? Math.max(2.2, 0.3 * len + 0.9) : clamp(0.26 * len + 0.5, 0.95, 2.6);
      const m = q
        ? tokenTile(tk.t, { face: 'pink', text: 'paper', side: 'pink', w: width, h: 1.3, d: 0.5, line: 2.6 })
        : tokenTile(tk.t, { face: 'paper', text: 'ink', side: 'blue', w: width, h: 1, d: 0.35 });
      x.push(acc + width / 2);
      tw.push(width);
      acc += width + GAP;
      tiles.push(m);
      group.add(m);
      const b = inked(barGeo, { ink: rank[i]! < 3 ? 'pink' : 'blue', lit: rank[i]! < 3 ? 0.6 : 0.4, shade: 1 }, 2);
      b.scale.set(Math.min(1.5, width * 0.7), 0.01, 0.8);
      bars.push(b);
      group.add(b);
    });
    const off = acc / 2;
    for (let i = 0; i < C; i++) x[i]! -= off;
    this.scene3.add(group);
    return { s, toks, C, w, order, rank, group, tiles, bars, x, tw, xq: x[C - 1]! };
  }

  /** All 32 heads of one layer at step s (cached). */
  private grid(s: number, layer: number): GridData {
    const k = `${s}:${layer}`;
    let g = this.grids.get(k);
    if (!g) { g = headRows(this.ctx.llm, s, layer); this.grids.set(k, g); }
    return g;
  }

  /** Screen position of a world point (logical px) and whether it is in front of the camera. */
  private proj(x: number, y: number, z: number): [number, number, boolean] {
    this.v.set(x, y, z).project(this.cam);
    return [(this.v.x + 1) * 0.5 * W, (1 - this.v.y) * 0.5 * H, this.v.z < 1];
  }

  /** Where tile i sits at time t (the top three get pulled to the newest token in shot 4). */
  private tilePos(R: Row, i: number, t: number, kind: number): [number, number, number] {
    const q = i === R.C - 1;
    let x = R.x[i]!, y = q ? 0.65 : 0.5, z = 0;
    if (!q && kind === 0) {
      // the context pops in left to right on the scene's first beats
      const k = snapIn(t, this.ctx.start + 0.05 + 0.3 * (i / R.C), 0.07);
      y -= (1 - k) * 3;
    }
    if (kind === 4 && R.rank[i]! < 3 && !q) {
      const k = ease.inOutExpo(clamp((t - this.q8(this.tm.pull)) / 0.22));
      const r = R.rank[i]!;
      const tx = R.xq - 2.6 - r * 3.7, tz = 2.4;
      x = lerp(x, tx, k);
      z = lerp(0, tz, k);
      y += Math.sin(Math.PI * k) * 2.2;
    }
    return [x, y, z];
  }

  /** Nearest 8th-note grid point (arrivals land on the grid). */
  private q8(t: number) {
    const au = this.ctx.audio;
    return au.timeOfBeat(Math.round(au.beatAt(t) * 2) / 2);
  }

  /** The camera for a shot at time t (pure function of time: labels are placed with f.ft, 3D rendered with f.t). */
  private pose(kind: number, t: number, sh: Shot, R: Row) {
    const au = this.ctx.audio, P = portrait();
    const dd = drive(au, t, 0.5, 1.6) - drive(au, sh.t0, 0.5, 1.6);
    const cam = this.cam;
    let pos: THREE.Vector3, look: THREE.Vector3, fov = 36;
    const xq = R.xq, xs = R.x[R.order[0]!]!;
    switch (kind) {
      case 0: // high 3/4 from the newest token, looking back along the row
        pos = P ? new THREE.Vector3(xq + 8 - dd * 0.5, 5, 2) : new THREE.Vector3(xq + 3 - dd * 0.5, 7, 17);
        look = P ? new THREE.Vector3(xq - 14 - dd * 0.5, 0.8, -3) : new THREE.Vector3(xq - 6 - dd * 0.5, 2.2, 0);
        fov = P ? 58 : 40;
        break;
      case 2: // low, in front of the question: the weights stand on the tiles like a skyline
        // (16:9) a dolly back along the skyline from the query to the strongest tile (looking back), rising
        pos = P ? new THREE.Vector3(xq + 10 - dd * 0.3, 2.6, 2.5) : new THREE.Vector3(xq - 2 + (xs + 4 - (xq - 2)) * sh.u, 2.4 + 1.2 * sh.u, 14 - 2 * sh.u);
        look = P ? new THREE.Vector3(xq - 9 - dd * 0.3, 3.4, -1.5) : new THREE.Vector3(xq - 6 + (xs - (xq - 6)) * sh.u, 4.2, 0);
        fov = P ? 58 : 44;
        break;
      case 4: // top-down over the newest token: the strongest tiles get pulled in
        pos = P ? new THREE.Vector3(xq + 4 - dd * 0.3, 4.5, 7) : new THREE.Vector3(xq - 7 - dd * 0.4, 10, 11.5);
        look = P ? new THREE.Vector3(xq - 9 - dd * 0.3, 0.4, 1) : new THREE.Vector3(xq - 7 - dd * 0.4, 0.8, 1.2);
        fov = P ? 56 : 40;
        break;
      case 7: { // the 32-head sheet, tilted in space
        const a = -0.55 + dd * 0.06;
        const r = P ? 11.5 : 9.2;
        pos = new THREE.Vector3(Math.sin(a) * r, (P ? 1.2 : 1.6) - dd * 0.05, Math.cos(a) * r);
        look = new THREE.Vector3(P ? 0.2 : 0.5, 0, 0);
        fov = P ? 44 : 38;
        break;
      }
      default: { // 8: the tower, every head lit
        const Ht = this.tower.height, a = 0.75 + dd * 0.1;
        const r = P ? 22 : 15;
        pos = new THREE.Vector3(Math.cos(a) * r, Ht * 0.72, Math.sin(a) * r);
        look = new THREE.Vector3(0, Ht * 0.5, 0);
        fov = P ? 52 : 48;
      }
    }
    cam.up.set(0, 1, 0);
    cam.position.copy(pos);
    if (cam.fov !== fov) { cam.fov = fov; cam.updateProjectionMatrix(); }
    cam.lookAt(look);
    cam.updateMatrixWorld();
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio;
    const sh = shot(this.ctx, f, { every: 4, from: this.from });
    const kind = Math.min(8, sh.i);
    const P = portrait();
    const R = this.rowFor(f.ft);
    const d = drive(au, f.t, 0.5, 1.6);
    const three = kind === 0 || kind === 2 || kind === 4 || kind === 7 || kind === 8;

    // background: the wash (reseeded per shot) and a rain of the context's real tokens
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: kind === 5 || kind === 6 ? 0.42 : 0.55, c1: kind % 2 ? 'blue' : 'pink', c2: kind % 2 ? 'pink' : 'blue' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => tokenRain(this.back.ctx, R.toks, drive(au, f.ft, 0.5, 1.6) * 0.5, { seed: sh.seed, cols: P ? 8 : 13 })), out);

    // labels and 2D diagrams: once per output frame, placed with the camera at f.ft
    const midTex = perFrame(this.mid, f, () => {
      if (three) this.pose(kind, f.ft, sh, R);
      this.drawMid(this.mid.ctx, f, sh, kind, R);
    });

    if (three) {
      // the sheet's texture (the tilted grid shot)
      if (kind === 7) perFrame(this.sheetL, f, () => this.drawSheet(this.sheetL.ctx, f, sh, R));
      this.stage(kind, f.t, R);
      this.pose(kind, f.t, sh, R);
      r.setRenderTarget(out);
      r.clearDepth();
      r.render(this.scene3, this.cam);
      if (kind === 0 || kind === 4) {
        this.arcs(kind, f.t, R);
        this.lines.render(r, out, this.cam);
      }
    }
    this.ctx.comp.draw(r, midTex, out);

    // foreground: flying ink specks (per sub-frame: they move fast)
    const c = this.front.ctx;
    this.front.clear();
    specks(c, d * (kind >= 6 ? 1.5 : 1), { seed: sh.seed, n: three ? 70 : 90 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    return { reg: sh.seed };
  }

  /** Show what the shot needs in 3D and put it in place for time t. */
  private stage(kind: number, t: number, R: Row) {
    const rowShot = kind === 0 || kind === 2 || kind === 4;
    for (const row of this.rows.values()) row.group.visible = rowShot && row === R;
    this.sheet.visible = kind === 7;
    this.tower.group.visible = kind === 8;
    if (rowShot) {
      for (let i = 0; i < R.C; i++) {
        const [x, y, z] = this.tilePos(R, i, t, kind);
        R.tiles[i]!.position.set(x, y, z);
        if (i === R.C - 1) R.tiles[i]!.scale.setScalar(Math.max(0.001, kind === 0 ? snapIn(t, this.from, 0.08) : 1));
        const b = R.bars[i]!;
        const q = i === R.C - 1;
        // the weights stand on the tiles in shot 2 (they rise on "matters", nearest first)
        const k = kind === 2 && !q ? snapIn(t, this.shotStart(t) + 0.03 + 0.2 * ((R.C - 1 - i) / R.C), 0.09) : 0;
        const hgt = Math.max(0.02, R.w[i]! * BAR) * k;
        b.visible = k > 0.001;
        b.scale.y = Math.max(0.001, hgt);
        b.position.set(x, y + 0.5 + hgt / 2, z);
      }
    }
    if (kind === 8) {
      const st = this.ctx.story.at(t);
      const L = this.tower.layers;
      this.tower.light(st.climb);
      if (!this.lit) {
        // each head's pillar: pink coverage = the share of its weight that is NOT parked on the sink
        for (let l = 1; l <= L; l++) this.tower.setHeads(l, this.grid(R.s, l - 1).rows.map((row) => 1 - row[0]!));
        this.lit = true;
      }
      this.tower.ride(this.tower.rider.children[0]!, st.climb);
    }
  }

  /** The arcs from the newest token back to every earlier one: width = real attention weight. */
  private arcs(kind: number, t: number, R: Row) {
    const L = this.lines;
    L.clear();
    const qi = R.C - 1;
    const [qx, qy, qz] = this.tilePos(R, qi, t, kind);
    const tq = this.q8(this.tm.tok);
    const pink = LIN.pink, blue = LIN.blue, ink = LIN.ink;
    // thin first, heavy last (they draw in order)
    const idx = [...R.order].reverse();
    for (const i of idx) {
      if (i === qi) continue;
      const w = R.w[i]!, rk = R.rank[i]!;
      if (kind === 4 && rk >= 3 && w < 0.004) continue;
      // grow from the newest token back (shot 0: the sweep lands on "back"; later shots: already there)
      let p = 1;
      if (kind === 0) p = clamp((t - (tq + (this.q8(this.tm.back) - tq - 0.08) * ((qi - i) / qi))) / 0.08);
      else p = clamp((t - (this.q8(this.shotStart(t)) + 0.12 * ((qi - i) / qi))) / 0.08);
      if (p <= 0) continue;
      const [tx, ty, tz] = this.tilePos(R, i, t, kind);
      const dx = Math.abs(qx - tx);
      const hgt = Math.min(9, 0.9 + dx * 0.32) * (kind === 4 ? 0.7 : 1);
      const n = 26, top = rk < 3;
      const width = (w >= 0.004 ? 0.03 : 0.009) + w * (kind === 4 ? 0.45 : 0.7);
      const col = top ? pink : w >= 0.004 ? blue : ink;
      const y0 = qy + (kind === 4 ? 0.2 : 0.62), y1 = ty + 0.52;
      let px = qx, py = y0, pz = qz;
      for (let k = 1; k <= n; k++) {
        const u = (k / n) * p;
        const x = lerp(qx, tx, u), y = lerp(y0, y1, u) + Math.sin(Math.PI * u) * hgt, z = lerp(qz, tz, u);
        if (top) L.seg(px, py, pz, x, y, z, width + 0.07, ink[0], ink[1], ink[2], 1);
        L.seg(px, py, pz, x, y, z, width, col[0], col[1], col[2], 1);
        px = x; py = y; pz = z;
      }
    }
  }

  private shotStart(t: number) {
    const au = this.ctx.audio;
    const b0 = Math.round(au.beatAt(this.from));
    const i = Math.max(0, Math.floor((au.beatAt(t) - b0 + 1e-3) / 4));
    return au.timeOfBeat(b0 + i * 4);
  }

  /** The 32-head grid printed on the tilted sheet (shot 7): flips layer on every beat. */
  private drawSheet(c: CanvasRenderingContext2D, f: Frame, sh: Shot, R: Row) {
    const au = this.ctx.audio, P = portrait();
    c.fillStyle = PAPER;
    c.fillRect(0, 0, W, H);
    const beat = Math.max(0, Math.floor(au.beatAt(f.ft) - au.beatAt(sh.t0) + 1e-3));
    const layer = FLIP[Math.min(FLIP.length - 1, beat)]!;
    const k = snapIn(f.ft, au.timeOfBeat(Math.round(au.beatAt(sh.t0)) + beat), 0.08);
    const g = this.grid(R.s, layer);
    const m = P ? 30 : 36;
    drawGrid(c, g, R.toks, m, m, W - 2 * m, H - 2 * m, P ? 4 : 8, P ? 8 : 4, () => k, -1, P ? 30 : 30);
  }

  /** Per-frame type, labels and the 2D diagrams. */
  private drawMid(c: CanvasRenderingContext2D, f: Frame, sh: Shot, kind: number, R: Row) {
    const P = portrait(), ft = f.ft, au = this.ctx.audio;
    const dd = drive(au, ft, 0.5, 1.6) - drive(au, sh.t0, 0.5, 1.6);
    const qi = R.C - 1;
    const pct = (w: number) => `${(w * 100).toFixed(1)}%`;
    const headTag = `LAYER ${LAYER + 1} · HEAD ${HEAD + 1}`;
    const sx = P ? 70 : 110, sy = P ? 250 : 178;
    if (kind <= 5) drawStamp(c, headTag, sx, sy, P ? 28 : 26, INK, PAPER);

    if (kind === 0 || kind === 2 || kind === 4) {
      // weight labels over the strongest tiles, projected from 3D
      const tags: [number, number, string][] = [];
      for (let r = 0; r < (kind === 2 ? 5 : 3); r++) {
        const i = R.order[r]!;
        if (i === qi) continue;
        const [x, y, z] = this.tilePos(R, i, ft, kind);
        const top = kind === 2 ? y + 0.6 + R.w[i]! * BAR : y + 0.75;
        const [X, Y, vis] = this.proj(x, top, z);
        if (!vis || X < 120 || X > W - 120 || Y < 170 || Y > H - 140) continue;
        const k = kind === 0 ? snapIn(ft, this.q8(this.tm.back), 0.08) : kind === 2 ? snapIn(ft, this.q8(this.tm.matters) + 0.1, 0.08) : 1;
        if (k <= 0) continue;
        tags.push([X, Y, i === 0 ? `${pct(R.w[i]!)} SINK` : pct(R.w[i]!)]);
      }
      c.font = font(F.mono(700), P ? 30 : 28);
      c.textAlign = 'center';
      c.textBaseline = 'bottom';
      const placed: [number, number, number, number][] = [];
      for (const [X, Y, s] of tags) {
        const w = c.measureText(s).width + 20;
        // strongest first: a weaker label that would overlap one already printed is dropped
        const box: [number, number, number, number] = [X - w / 2, Y - 46, X + w / 2, Y - 6];
        if (placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
        placed.push(box);
        c.fillStyle = PINK;
        c.fillRect(X - w / 2, Y - 46, w, 40);
        c.fillStyle = PAPER;
        c.fillText(s, X, Y - 12);
      }
      c.textAlign = 'left';
      // the newest token: the one asking
      const [X, Y, vis] = this.proj(R.x[qi]!, this.tilePos(R, qi, ft, kind)[1] - 0.75, this.tilePos(R, qi, ft, kind)[2]);
      if (vis && X > 120 && X < W - 200 && Y < H - 160 && Y > 180 && (kind !== 0 || ft >= this.from)) {
        c.font = font(F.mono(700), P ? 28 : 26);
        c.fillStyle = INK;
        c.textBaseline = 'top';
        c.textAlign = 'center';
        c.fillText(kind === 2 ? 'QUERY' : 'NEWEST TOKEN', X, Y + 10);
        c.textAlign = 'left';
      }
      const sub = kind === 0 ? `LOOKS BACK AT ALL ${qi} EARLIER TOKENS` : kind === 2 ? 'BAR = ITS WEIGHT ON EACH TOKEN' : 'THE TOP 3 PULL';
      drawStamp(c, sub, sx, sy + (P ? 58 : 52), P ? 26 : 24, PINK, PAPER);
    } else if (kind === 1) this.page(c, ft, sh, R, dd);
    else if (kind === 3) this.sortChart(c, ft, sh, R, dd);
    else if (kind === 5) this.blend(c, ft, sh, R, dd);
    else if (kind === 6) {
      const g = this.grid(R.s, LAYER);
      const x0 = P ? 60 : 110, y0 = P ? 300 : 200, gw = W - x0 - (P ? 60 : 70), gh = P ? 1120 : 610;
      const ox = Math.sin(dd * 0.4) * 10;
      drawSlab(c, `LAYER ${LAYER + 1} · ALL 32 HEADS`, x0 + ox, P ? 250 : 168, P ? 56 : 58, 'left');
      drawGrid(c, g, R.toks, x0 + ox, y0, gw, gh, P ? 4 : 8, P ? 8 : 4, (i) => snapIn(ft, sh.t0 + 0.008 * i, 0.08), HEAD);
      c.fillStyle = INK;
      c.font = font(F.mono(700), P ? 26 : 24);
      const ly = y0 + gh + (P ? 44 : 44);
      c.fillText(P ? `BLACK = WEIGHT ON THE FIRST TOKEN,` : `BLACK = WEIGHT ON THE FIRST TOKEN, AN ATTENTION SINK: THE TOP WEIGHT IN ${g.sinkTop} OF 32 HEADS`, x0 + ox, ly);
      if (P) c.fillText(`AN ATTENTION SINK: TOP IN ${g.sinkTop} OF 32 HEADS`, x0 + ox, ly + 36);
      c.fillStyle = PINK;
      c.fillText(P ? 'PINK = TOP WORD AFTER IT, BARS TO ITS PEAK' : 'PINK = EACH HEAD’S TOP WORD AFTER THE SINK · BARS SCALED TO THAT PEAK', x0 + ox, ly + (P ? 72 : 34));
    } else if (kind === 7) {
      const beat = Math.max(0, Math.floor(au.beatAt(ft) - au.beatAt(sh.t0) + 1e-3));
      const layer = FLIP[Math.min(FLIP.length - 1, beat)]!;
      const g = this.grid(R.s, layer);
      drawStamp(c, `FLIPPING UP THE STACK · 32 HEADS PER LAYER`, sx, sy, P ? 28 : 26, INK, PAPER);
      drawStamp(c, `SINK IS THE TOP WEIGHT IN ${g.sinkTop} OF 32`, sx, sy + (P ? 58 : 52), P ? 26 : 24, PINK, PAPER);
      const bk = snapIn(ft, au.timeOfBeat(Math.round(au.beatAt(sh.t0)) + beat), 0.08);
      c.save();
      c.translate(P ? W / 2 : 110, P ? H * 0.76 : H * 0.8);
      c.scale(1.2 - 0.2 * bk, 1.2 - 0.2 * bk);
      drawSlab(c, `LAYER ${layer + 1}`, 0, 0, P ? 150 : 170, P ? 'center' : 'left');
      c.restore();
    } else if (kind === 8) {
      drawStamp(c, `32 LAYERS × 32 HEADS = 1,024 (COMPUTED)`, sx, sy, P ? 28 : 26, INK, PAPER);
      drawStamp(c, P ? 'PINK = WEIGHT NOT ON THE SINK' : 'PINK = EACH HEAD’S WEIGHT NOT PARKED ON THE SINK', sx, sy + (P ? 58 : 52), P ? 26 : 24, PINK, PAPER);
    }
  }

  /** Shot 1: the whole context as a page of tiles, a fan of arcs from the newest token to every one. */
  private page(c: CanvasRenderingContext2D, ft: number, sh: Shot, R: Row, dd: number) {
    const P = portrait(), qi = R.C - 1;
    const size = P ? 30 : 34, pad = 10, gap = 10, lh = P ? 112 : 112;
    const x0 = P ? 70 : 130, maxX = W - (P ? 60 : 110);
    const ox = Math.sin(dd * 0.35) * 14;
    // lay the context out like a page (the newest token, bigger, ends the last line)
    const boxes: { x: number; y: number; w: number; s: number }[] = [];
    let x = x0, y = 0;
    R.toks.forEach((tk, i) => {
      const s = i === qi ? size * 1.5 : size;
      c.font = font(F.mono(700), s);
      const w = c.measureText(showTok(tk.t)).width + pad * 2;
      if (x + w > maxX) { x = x0; y += lh; }
      boxes.push({ x: x + ox, y, w, s });
      x += w + gap;
    });
    const nRows = Math.round(y / lh) + 1;
    const yTop = (P ? H * 0.52 : H * 0.5) - ((nRows - 1) * lh) / 2;
    for (const b of boxes) b.y += yTop;
    const q = boxes[qi]!;
    const qx = q.x + q.w / 2, qy = q.y - q.s * 0.3;
    // arcs behind the tiles, from the newest token to each earlier one: thin first, heavy last
    const idx = [...R.order].reverse();
    c.lineCap = 'round';
    for (const i of idx) {
      if (i === qi) continue;
      const b = boxes[i]!, w = R.w[i]!, rk = R.rank[i]!;
      const p = clamp((ft - (sh.t0 + 0.02 + 0.16 * ((qi - i) / qi))) / 0.07);
      if (p <= 0) continue;
      const bx = b.x + b.w / 2, by = b.y - b.s * 1.05;
      const dist = Math.hypot(qx - bx, qy - by);
      const mx = (qx + bx) / 2, my = Math.min(qy, by) - 30 - dist * 0.28;
      const col = rk < 3 ? PINK : w >= 0.004 ? BLUE : INK;
      const n = 28, m = Math.ceil(n * p);
      const path = () => {
        c.beginPath();
        c.moveTo(qx, qy);
        for (let k = 1; k <= m; k++) {
          const u = Math.min(1, k / n);
          c.lineTo((1 - u) * (1 - u) * qx + 2 * u * (1 - u) * mx + u * u * bx, (1 - u) * (1 - u) * qy + 2 * u * (1 - u) * my + u * u * by);
        }
      };
      const lw = (w >= 0.004 ? 2.4 : 1.6) + w * 48;
      if (rk < 3) { path(); c.strokeStyle = INK; c.lineWidth = lw + 5; c.stroke(); }
      path();
      c.strokeStyle = col;
      c.lineWidth = lw;
      c.stroke();
    }
    // the tiles
    const hot = snapIn(ft, this.q8(this.tm.first), 0.08);
    R.toks.forEach((tk, i) => {
      const b = boxes[i]!, isq = i === qi, rk = R.rank[i]!;
      c.font = font(F.mono(700), b.s);
      if (rk < 3 && !isq) { c.fillStyle = PINK; c.fillRect(b.x - 6, b.y - b.s * 1.05 - 6, b.w + 12, b.s * 1.5 + 12); }
      c.fillStyle = isq ? PINK : INK;
      c.fillRect(b.x, b.y - b.s * 1.05, b.w, b.s * 1.5);
      c.fillStyle = PAPER;
      c.textBaseline = 'alphabetic';
      c.fillText(showTok(tk.t), b.x + pad, b.y + b.s * 0.1);
    });
    // weights of the top three under their tiles; on "first", the first token is called out as the sink
    c.font = font(F.mono(700), 26);
    c.textBaseline = 'top';
    for (let r = 0; r < 3; r++) {
      const i = R.order[r]!;
      if (i === qi) continue;
      const b = boxes[i]!;
      const sink = i === 0 && hot > 0;
      const label = sink ? `${(R.w[i]! * 100).toFixed(1)}% · FIRST TOKEN = ATTENTION SINK` : `${(R.w[i]! * 100).toFixed(1)}%`;
      const w = c.measureText(label).width + 16;
      const lx = Math.max(x0, Math.min(b.x, W - 90 - w));
      c.fillStyle = sink ? INK : PINK;
      c.fillRect(lx, b.y + b.s * 0.62, w, 36);
      c.fillStyle = PAPER;
      c.fillText(label, lx + 8, b.y + b.s * 0.62 + 5);
    }
    c.textBaseline = 'alphabetic';
    c.font = font(F.mono(700), 26);
    c.fillStyle = PINK;
    c.fillText('NEWEST TOKEN', q.x, q.y + q.s * 1.45);
    drawStamp(c, `${qi} EARLIER TOKENS · LINE WIDTH = WEIGHT`, P ? 70 : 110, P ? 308 : 230, P ? 26 : 24, PINK, PAPER);
  }

  /** Shot 3: the weights as bars in context order, sorted in a snap on "weight". */
  private sortChart(c: CanvasRenderingContext2D, ft: number, sh: Shot, R: Row, dd: number) {
    const P = portrait(), qi = R.C - 1, C = R.C;
    const k = ease.outExpo(clamp((ft - this.q8(this.tm.weight)) / 0.09));
    const K = 6;
    const ox = Math.sin(dd * 0.35) * 10;
    const rest = R.order.slice(K).reduce((a, i) => a + R.w[i]!, 0);
    const rise = (i: number) => snapIn(ft, sh.t0 + 0.004 * i, 0.08);
    if (!P) {
      const x0 = 130 + ox, x1 = 1790 + ox, yb = 700, maxH = 560;
      const sw = (x1 - x0) / C, SW = 236, tw = (x1 - x0 - K * SW) / (C - K);
      const slotX = (i: number) => {
        const a = x0 + i * sw + sw * 0.19, rk = R.rank[i]!;
        const b = rk < K ? x0 + rk * SW + SW * 0.12 : x0 + K * SW + (rk - K) * tw + tw * 0.1;
        return lerp(a, b, k);
      };
      const slotW = (i: number) => lerp(sw * 0.62, R.rank[i]! < K ? SW * 0.76 : tw * 0.8, k);
      c.fillStyle = INK;
      c.fillRect(x0 - 10, yb, x1 - x0 + 20, 4);
      for (let i = 0; i < C; i++) {
        const x = slotX(i), w = slotW(i), rk = R.rank[i]!, hh = Math.max(3, R.w[i]! * maxH * rise(i));
        c.fillStyle = i === 0 ? INK : rk < 3 ? PINK : BLUE;
        c.fillRect(x, yb - hh, w, hh);
        // the tile under the bar (before the sort: a stub; after: a labelled tile for the top K)
        c.fillStyle = i === qi ? PINK : INK;
        if (rk >= K || k < 0.5) c.fillRect(x, yb + 12, w, 16);
      }
      // labels: before the sort the top three; after, the top K with their tokens
      c.font = font(F.mono(700), 28);
      c.textAlign = 'center';
      for (let r = 0; r < (k > 0.5 ? K : 3); r++) {
        const i = R.order[r]!, x = slotX(i) + slotW(i) / 2, hh = R.w[i]! * maxH;
        if (rise(i) < 0.5) continue;
        c.fillStyle = INK;
        c.textBaseline = 'bottom';
        c.fillText(`${(R.w[i]! * 100).toFixed(1)}%`, x, yb - hh - 10);
        if (i === 0) { c.fillStyle = PINK; c.fillText('SINK', x, yb - hh - 46); }
        if (k > 0.5) {
          c.font = font(F.mono(700), 24);
          const lab = showTok(R.toks[i]!.t), lw = c.measureText(lab).width + 16;
          c.fillStyle = i === qi ? PINK : INK;
          c.fillRect(x - lw / 2, yb + 12, lw, 38);
          c.fillStyle = PAPER;
          c.textBaseline = 'middle';
          c.fillText(lab, x, yb + 32);
          c.font = font(F.mono(700), 28);
        }
      }
      c.textAlign = 'left';
      c.textBaseline = 'alphabetic';
      c.font = font(F.mono(700), 24);
      if (k > 0.5) {
        c.fillStyle = INK;
        c.textAlign = 'right';
        c.fillText(`+ ${C - K} MORE: ${(rest * 100).toFixed(1)}%`, x1, yb + 64);
        c.textAlign = 'left';
      }
      drawStamp(c, k > 0.5 ? 'SORTED · ALL WEIGHTS ADD UP TO 100%' : `${C} WEIGHTS, IN CONTEXT ORDER`, 110, 230, 24, PINK, PAPER);
    } else {
      const y0 = 330, y1 = 1500, xl = 360 + ox, maxW = W - 90 - xl - 20;
      const sh0 = (y1 - y0) / C, SH = 150, th = (y1 - y0 - K * SH) / (C - K);
      const slotY = (i: number) => {
        const a = y0 + i * sh0, rk = R.rank[i]!;
        const b = rk < K ? y0 + rk * SH : y0 + K * SH + (rk - K) * th;
        return lerp(a, b, k);
      };
      const slotH = (i: number) => lerp(sh0 * 0.72, R.rank[i]! < K ? SH * 0.6 : th * 0.7, k);
      c.fillStyle = INK;
      c.fillRect(xl, y0 - 10, 4, y1 - y0 + 10);
      c.textAlign = 'right';
      c.textBaseline = 'middle';
      for (let i = 0; i < C; i++) {
        const y = slotY(i), h = slotH(i), rk = R.rank[i]!, ww = Math.max(3, R.w[i]! * maxW * 1.3 * rise(i));
        c.fillStyle = i === 0 ? INK : rk < 3 ? PINK : BLUE;
        c.fillRect(xl + 8, y, Math.min(maxW, ww), h);
        const lab = k < 0.5 || rk < K;
        if (lab) {
          c.font = font(F.mono(700), k > 0.5 ? 30 : 22);
          c.fillStyle = i === qi ? PINK : INK;
          c.fillText(showTok(R.toks[i]!.t), xl - 12, y + h / 2);
        }
      }
      c.textAlign = 'left';
      c.font = font(F.mono(700), 30);
      for (let r = 0; r < (k > 0.5 ? K : 3); r++) {
        const i = R.order[r]!, y = slotY(i), h = slotH(i), ww = Math.min(maxW, R.w[i]! * maxW * 1.3);
        c.fillStyle = INK;
        const s = `${(R.w[i]! * 100).toFixed(1)}%${i === 0 ? ' SINK' : ''}`;
        const tx = Math.min(W - 90 - c.measureText(s).width, xl + 22 + ww);
        c.fillText(s, tx, y + h / 2);
      }
      if (k > 0.5) {
        c.font = font(F.mono(700), 26);
        c.fillStyle = INK;
        c.fillText(`+ ${C - K} MORE: ${(rest * 100).toFixed(1)}% TOGETHER`, xl + 20, y0 + K * SH + (C - K) * th * 0.5);
      }
      drawStamp(c, k > 0.5 ? 'SORTED · ADDS UP TO 100%' : `${C} WEIGHTS, IN ORDER`, 70, 308, 26, PINK, PAPER);
    }
  }

  /** Shot 5: the top three words overprint into the newest token, ink coverage = weight. */
  private blend(c: CanvasRenderingContext2D, ft: number, sh: Shot, R: Row, dd: number) {
    const P = portrait(), qi = R.C - 1;
    const cw = P ? 940 : 1480, ch = P ? 600 : 430;
    const cx = W / 2 + Math.sin(dd * 0.35) * 12, cy = P ? H * 0.47 : H * 0.5;
    const tb = this.q8(this.tm.blends);
    // the card: the newest token's update
    c.fillStyle = PAPER;
    c.fillRect(cx - cw / 2, cy - ch / 2, cw, ch);
    c.strokeStyle = PINK;
    c.lineWidth = 14;
    c.strokeRect(cx - cw / 2, cy - ch / 2, cw, ch);
    // its owner, top left of the card
    c.font = font(F.mono(700), P ? 34 : 32);
    const own = showTok(R.toks[qi]!.t), ow = c.measureText(own).width + 24;
    c.fillStyle = PINK;
    c.fillRect(cx - cw / 2, cy - ch / 2 - 52, ow, 52);
    c.fillStyle = PAPER;
    c.textBaseline = 'middle';
    c.fillText(own, cx - cw / 2 + 12, cy - ch / 2 - 25);
    c.fillStyle = INK;
    c.font = font(F.mono(700), P ? 26 : 24);
    c.fillText('+ WHAT IT TAKES FROM THE OTHERS', cx - cw / 2 + ow + 16, cy - ch / 2 - 25);
    // the top three overprint, each landing on an 8th from "blends"; coverage = its real weight
    const inks = [BLUE, INK, PINK];
    const top = R.order.filter((i) => i !== qi).slice(0, 3);
    const au = this.ctx.audio;
    // each word as big as the card allows (mono: 0.6 em per glyph)
    const sizeOf = (i: number) => Math.min(P ? 190 : 220, (cw * 0.92) / (showTok(R.toks[i]!.t).length * 0.6));
    c.save();
    c.globalCompositeOperation = 'multiply';
    c.textAlign = 'center';
    top.forEach((i, r) => {
      const at = au.timeOfBeat(au.beatAt(tb) + r * 0.5);
      const k = snapIn(ft, at, 0.08);
      if (k <= 0) return;
      c.save();
      c.translate(cx, cy + (P ? 10 : 6));
      const s = 1.25 - 0.25 * k;
      c.scale(s, s);
      c.globalAlpha = clamp(R.w[i]!, 0.06, 1);
      c.font = font(F.mono(700), sizeOf(i));
      c.fillStyle = inks[r]!;
      c.fillText(showTok(R.toks[i]!.t), 0, 0);
      c.restore();
    });
    c.restore();
    // before the blend: the three sources wait above the card
    // the legend: weight -> ink
    const ly = cy + ch / 2 + (P ? 60 : 56);
    c.textBaseline = 'middle';
    c.font = font(F.mono(700), P ? 28 : 28);
    let lx = cx - cw / 2;
    const rest = R.order.filter((i) => !top.includes(i)).reduce((a, i) => a + R.w[i]!, 0);
    const items: [string, string][] = top.map((i, r) => [inks[r]!, `${(R.w[i]! * 100).toFixed(1)}% ${showTok(R.toks[i]!.t)}`]);
    items.push(['', `${(rest * 100).toFixed(1)}% THE OTHER ${R.C - 3} (SUM)`]);
    items.forEach(([col, s], n) => {
      const k = n < 3 ? snapIn(ft, au.timeOfBeat(au.beatAt(tb) + n * 0.5), 0.08) : snapIn(ft, au.timeOfBeat(au.beatAt(tb) + 1.5), 0.08);
      if (k <= 0) return;
      const w = c.measureText(s).width;
      if (P && lx + w + 40 > cx + cw / 2) { lx = cx - cw / 2; }
      const yy = P ? ly + n * 46 : ly + Math.floor(n / 2) * 46;
      if (!P && n % 2 === 0) lx = cx - cw / 2;
      if (col) { c.fillStyle = col; c.fillRect(lx, yy - 13, 26, 26); }
      c.fillStyle = INK;
      c.fillText(s, lx + (col ? 36 : 0), yy);
      lx += w + (P ? 0 : 90);
      if (P) lx = cx - cw / 2;
    });
    // before "blends": the card is empty; its three sources hang over it with their weights
    const pre = 1 - snapIn(ft, tb, 0.08);
    if (pre > 0) {
      c.font = font(F.mono(700), P ? 40 : 44);
      c.textAlign = 'center';
      top.forEach((i, r) => {
        const x = cx + (r - 1) * cw * 0.33;
        const k = snapIn(ft, sh.t0 + r * 0.06, 0.08);
        const lab = showTok(R.toks[i]!.t), w = c.measureText(lab).width + 30;
        const y = cy - 30 + (1 - k) * 40;
        c.fillStyle = inks[r]!;
        c.fillRect(x - w / 2, y - 34, w, 68);
        c.fillStyle = PAPER;
        c.fillText(lab, x, y + 2);
      });
      c.textAlign = 'left';
    }
    c.textBaseline = 'alphabetic';
    drawStamp(c, 'INK = WEIGHT: THE NEWEST TOKEN BLENDS THEIR VALUES', P ? 70 : 110, P ? 308 : 230, P ? 24 : 24, PINK, PAPER);
  }
}
