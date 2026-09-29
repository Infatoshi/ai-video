// "Who's It?": the whole video is one picture that keeps transforming. One sentence, printed as its real
// Llama 3.1 tokens; arcs from the word that asks back to every earlier word, each as thick as its real
// attention weight (data/attn.json, layer 8 head 29); the last word morphing tired <-> wide; a reading
// cursor; tags that are drawn in and left up; and, in verse 3, a pull-back from this one head to all
// 1,024 heads. Every state is a function of song time, anchored to the sung lines (SPEC.md "Shot plan").
// No cuts: the camera moves, things are drawn in over >= 0.35 s, and something always drifts.
import type * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp } from '../engine/util';
import { perFrame, Wash } from './_kit';
import type { Lyrics } from '../engine/lyrics';
import type { Attn } from '../engine/attn';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
const P = portrait();
function portrait() { return H > W; }

/** 0 before t0, eased to 1 over d seconds. */
const up = (t: number, t0: number, d = 0.6, fn = ease.inOutCubic) => (Number.isFinite(t0) ? fn(clamp((t - t0) / d)) : 0);
/** On between t0 and t1 with eased edges. */
const span = (t: number, t0: number, t1: number, d = 0.5) => up(t, t0, d) * (1 - up(t, t1, d));
/**
 * A smooth back-and-forth between 0 and 1 over [a, b]: starts at `from`, ends at `to`, about `period` s per
 * full swing (the number of half-swings is rounded so both ends land exactly).
 */
function wave(t: number, a: number, b: number, from: 0 | 1, to: 0 | 1, period: number) {
  if (!(b > a)) return from;
  let k = Math.max(1, Math.round((b - a) / (period / 2)));
  if ((k % 2 === 1) !== (from !== to)) k += 1;
  const u = clamp((t - a) / (b - a));
  const v = 0.5 - 0.5 * Math.cos(Math.PI * k * u);
  return from === 0 ? v : 1 - v;
}
/** The camera's slow float in screen px (two periods per axis, so it never stops; ~30 px/s). */
const float = (t: number): [number, number] => [70 * Math.sin(t * 0.41) + 20 * Math.sin(t * 0.97 + 2.1), 40 * Math.sin(t * 0.33 + 1.3) + 14 * Math.sin(t * 0.79 + 0.5)];
/** Sweep a cursor from token a to token b over [t0, t1], one token at a time (each step eased). */
function sweep(t: number, t0: number, t1: number, a: number, b: number) {
  if (t <= t0) return a;
  if (t >= t1) return b;
  const n = b - a, u = ((t - t0) / (t1 - t0)) * n, k = Math.floor(u);
  return a + k + ease.inOutCubic(clamp((u - k) / 0.55));
}

interface Cues {
  v1: number[]; v1w: { animal: number; wide: number; street: number }; v1end: number;
  ch: number[][]; chEnd: number[];
  v2: number[]; v2end: number;
  br: number[]; brEnd: number;
  v3: number[]; v3end: number;
  out: number[]; end: number;
  tokenIn: number[]; bar: number;
}

function cues(ly: Lyrics, T: number, dur: number, bar: number): Cues {
  const tryGet = (q: string, nth = 0) => { try { return ly.get(q, nth); } catch { return null; } };
  const st = (q: string, nth = 0) => tryGet(q, nth)?.start ?? NaN;
  const en = (q: string, nth = 0) => tryGet(q, nth)?.end ?? NaN;
  const wd = (q: string, w: string, nth = 0) => {
    const l = tryGet(q, nth);
    const x = l?.words.find((y) => y.w.toLowerCase().replace(/[^a-z']/g, '').startsWith(w));
    return x?.start ?? l?.start ?? NaN;
  };
  const V1 = ['The animal didn', 'Because it was too tired', "Who's it? The animal", 'Change one word', "Who's it? Now the street", 'A program reads', 'So how does it know'];
  const CH = ['Every word looks back', 'Asks each one', 'Gives each a weight', "That's attention"];
  const V2 = ["The word that's asking", 'The words that answer', 'The closer they match', 'Each weight is a line', 'And spare weight'];
  const BR = ['But it reads from left', 'When it gets to it', "So it can't tell", 'Tired or wide', 'Then tired comes', 'Wide looks back'];
  const V3 = ['Each way of looking', 'Thirty-two heads', 'Thirty-two layers', "Most don't switch", 'One head is one small', 'Ask it who was tired'];
  const OUT = ["Who's it? Just look back", 'Next time'];
  // tokens appear as they are sung: The animal didn|'t cross the street / because it was too tired .
  const l1 = tryGet(V1[0]!), l2 = tryGet(V1[1]!);
  const w1 = l1?.words ?? [], w2 = l2?.words ?? [];
  const at = (ws: typeof w1, i: number, fb: number) => (ws[i]?.start ?? fb) - 0.12;
  const s1 = st(V1[0]!), s2 = st(V1[1]!);
  const tokenIn = new Array(T).fill(NaN);
  tokenIn[1] = at(w1, 0, s1); tokenIn[2] = at(w1, 1, s1); tokenIn[3] = at(w1, 2, s1); tokenIn[4] = tokenIn[3] + 0.18;
  tokenIn[5] = at(w1, 3, s1); tokenIn[6] = at(w1, 4, s1); tokenIn[7] = at(w1, 5, s1);
  for (let i = 0; i < 5; i++) tokenIn[8 + i] = at(w2, i, s2);
  tokenIn[13] = (w2[4]?.end ?? s2 + 3) - 0.1;
  const ch = [0, 1, 2].map((k) => CH.map((q) => st(q, k)));
  tokenIn[0] = ch[0]![0]! - 0.9;
  return {
    v1: V1.map((q) => st(q)), v1w: { animal: wd(V1[2]!, 'animal'), wide: wd(V1[3]!, 'wide'), street: wd(V1[4]!, 'street') },
    v1end: en(V1[6]!),
    ch, chEnd: [0, 1, 2].map((k) => en(CH[3]!, k)),
    v2: V2.map((q) => st(q)), v2end: en(V2[4]!),
    br: BR.map((q) => st(q)), brEnd: en(BR[5]!),
    v3: V3.map((q) => st(q)), v3end: en(V3[5]!),
    out: OUT.map((q) => st(q)), end: dur, tokenIn, bar,
  };
}

// ------------------------------------------------------------------ layout (world units)
const S = 48;                       // token type size
const ROW = 80;                     // portrait: token row pitch
const FAM = F.archivo(100, 700);

interface Tok { x: number; y: number; w: number; label: string }

export default class Sentence extends Scene {
  wash = new Wash();
  layer = new Layer2D();
  cue!: Cues;
  A!: string[];
  B!: string[];
  a!: Attn;
  measureCtx = document.createElement('canvas').getContext('2d')!;
  wA: number[] = [];
  m = 0;
  wB: number[] = [];
  gap: number[] = [];

  override async init() {
    this.a = this.ctx.attn;
    const d = this.a.d;
    const clean = (s: string) => (s === '<|begin_of_text|>' ? 'start' : s.replace(/^ /, '').replace("'", '’'));
    this.A = d.A.map(clean);
    this.B = d.B.map(clean);
    const au = this.ctx.audio;
    this.cue = cues(this.ctx.lyrics, this.A.length, au.duration, (240 / (au.bpm || 75)));
    const m = this.measureCtx;
    m.font = font(FAM, S);
    const meas = (s: string, i: number) => (i === 0 ? m.measureText('start').width * 0.62 + 30 : m.measureText(s).width);
    this.wA = this.A.map(meas);
    this.wB = this.B.map(meas);
    // no space before a token that continues a word ('t) or punctuation
    this.gap = d.A.map((s, i) => (i === 0 ? 0 : s.startsWith(' ') ? S * 0.34 : S * 0.1));
  }

  /** Token boxes for morph m (0 = "tired" sentence, 1 = "wide"). World units; baseline y = 0 (landscape). */
  layout(m: number): Tok[] {
    const n = this.A.length, out: Tok[] = [];
    if (!P) {
      let x = 0;
      for (let i = 0; i < n; i++) {
        x += this.gap[i]!;
        const w = lerp(this.wA[i]!, this.wB[i]!, m);
        out.push({ x, y: 0, w, label: this.A[i]! });
        x += w;
      }
      const half = x / 2;
      for (const t of out) t.x -= half;
    } else {
      for (let i = 0; i < n; i++) out.push({ x: 0, y: (i - (n - 1) / 2) * ROW, w: lerp(this.wA[i]!, this.wB[i]!, m), label: this.A[i]! });
    }
    return out;
  }

  /** Where an arc attaches to token i (landscape: top centre; portrait: left edge, mid-height). */
  anchor(t: Tok): [number, number] {
    return P ? [t.x - 16, t.y - S * 0.34] : [t.x + t.w / 2, -S * 0.86];
  }

  /** The featured head's weights from the asking word. q: 0 = "it", 1 = "tired"/"wide"; m morph. */
  weights(qk: number, m: number): number[] {
    const n = this.A.length, w = new Array(n).fill(0);
    const it = this.a.row(0, 9), tA = this.a.row(0, 12), tB = this.a.row(1, 12);
    for (let i = 0; i < n; i++) {
      const wi = i <= 9 ? it[i]! : 0;
      const wt = i <= 12 ? lerp(tA[i]!, tB[i]!, m) : 0;
      w[i] = lerp(wi, wt, qk);
    }
    return w;
  }

  // ---------------------------------------------------------------- the state at time t
  state(t: number) {
    const c = this.cue, bar = c.bar;
    // morph: 0 = "...too tired", 1 = "...too wide"
    let m = 0;
    if (t >= c.v1w.wide - 0.15 && t < c.v1[5]!) m = up(t, c.v1w.wide - 0.15, 0.7);
    else if (t >= c.v1[5]! && t < c.v1end + 1) m = 1 - up(t, c.v1[5]!, 0.7);
    else if (t >= c.v1end + 1 && t < c.ch[0]![0]!) m = wave(t, c.v1end + 1, c.ch[0]![0]! - 0.9, 0, 0, 2 * bar);
    else if (t >= c.br[3]! && t < c.br[4]!) m = wave(t, c.br[3]! + 0.25, c.br[4]! - 0.3, 0, 0, 1.8);
    else if (t >= c.br[5]! && t < c.brEnd + 0.8) m = up(t, c.br[5]! + 0.15, 1.4);
    else if (t >= c.brEnd + 0.8 && t < c.v3[0]!) m = wave(t, c.brEnd + 0.8, c.v3[0]! - 0.4, 1, 0, 2 * bar);
    else if (t >= c.v3[3]! && t < c.v3[5]!) m = wave(t, c.v3[3]! + 0.2, c.v3[5]! - 0.3, 0, 0, 3.2);
    else if (t >= c.ch[2]![3]! && t < c.out[0]!) m = up(t, c.ch[2]![3]! + 0.1, 1.2);
    else if (t >= c.out[0]! && t < c.out[1]! + 7) m = 1 - up(t, c.out[0]!, 1.2);
    else if (t >= c.out[1]! + 7) m = wave(t, c.out[1]! + 7, c.end - 3, 0, 0, 2 * bar);

    // arcs: drawn in on each chorus's first line (uniform), grown to the real weights on its second line
    const redraw = (k: number) => {
      const s0 = c.ch[k]![0]!;
      if (t < s0 - 1.0) return k === 0 ? 0 : 1;
      if (t < s0) return k === 0 ? 0 : 1 - up(t, s0 - 1.0, 0.9);
      return up(t, s0 + 0.15, 1.5, ease.inOutQuad);
    };
    let draw = redraw(0);
    if (t >= c.ch[1]![0]! - 1.0) draw = redraw(1);
    if (t >= c.ch[2]![0]! - 1.0) draw = redraw(2);
    draw *= 1 - span(t, c.out[1]!, c.out[1]! + 3.2, 1.6);
    // growth to the real weights (0 = every arc alike)
    const growK = (k: number) => up(t, c.ch[k]![1]! + 0.1, 1.6);
    let g = growK(0);
    // choruses 2 and 3 retract and redraw the arcs at their real weights (the bar never jumps)
    if (t >= c.v2[2]! && t < c.v2[3]!) g = 1 - 0.75 * up(t, c.v2[2]!, 0.9) + 0.75 * up(t, c.v2[2]! + 1.0, 1.6);
    // the asking word: "it", then "tired"/"wide" from the bridge's "Then tired comes"
    const qk = up(t, c.br[4]! + 0.35, 1.3);
    // emphasis on animal + street (bridge: "it looks at both")
    const hl = span(t, c.br[2]! + 0.2, c.br[4]! + 0.2, 0.6);

    // reading cursor (token index) and the unread mask
    let cur = -1, curA = 0, mask = 0;
    if (t >= c.v1[5]! - 0.3 && t < c.v1end + 1.2) {
      cur = sweep(t, c.v1[5]! + 0.1, c.v1[6]! - 0.3, 0, 13);
      curA = span(t, c.v1[5]! - 0.3, c.v1end + 0.4, 0.4); mask = span(t, c.v1[5]! - 0.3, c.v1[6]! - 0.2, 0.8);
    } else if (t >= c.br[0]! - 0.4 && t < c.brEnd + 1.5) {
      cur = t < c.br[4]! ? sweep(t, c.br[0]! + 0.2, c.br[1]! - 0.2, 0, 9) : sweep(t, c.br[4]! + 0.1, c.br[4]! + 1.5, 9, 12);
      curA = span(t, c.br[0]! - 0.4, c.brEnd + 0.6, 0.4); mask = span(t, c.br[0]! - 0.3, c.brEnd + 0.8, 0.8);
    }

    // the "who's it?" answer you give (a dashed arc), verse 1 and its instrumental
    const hA = up(t, c.v1w.animal - 0.25, 0.9) * (1 - up(t, c.v1[3]! + 0.4, 0.6));
    const hS = up(t, c.v1w.street - 0.25, 0.9) * (1 - up(t, c.v1[5]!, 0.6));
    const hI = span(t, c.v1end + 0.6, c.ch[0]![0]! - 1.0, 0.8);

    const tags = up(t, c.v2[0]! + 0.3, 0.6);
    const keyTags = (i: number) => up(t, c.v2[1]! + 0.3 + 0.12 * i, 0.5);
    const barOn = up(t, c.ch[0]![2]! + 0.1, 0.9) * (1 - up(t, c.v3[0]! + 0.5, 0.8)) + up(t, c.ch[2]![2]! + 0.1, 0.9) * (1 - up(t, c.out[0]!, 1));

    // verse 3: this head -> its row of 32 -> all 1,024; then back in
    const row = up(t, c.v3[1]! + 0.2, 2.6);
    const all = up(t, c.v3[2]! + 0.2, 2.8);
    const back = up(t, c.v3[5]! - 0.1, 3.0);
    const dim = up(t, c.v3[3]! + 0.4, 1.0) * (1 - back);   // cells that don't switch dim
    const cleanHi = up(t, c.v3[4]! + 0.3, 0.8) * (1 - back);
    const card = span(t, c.v3[0]! + 0.2, c.v3[5]! + 1.5, 0.6);
    const probe = span(t, c.v3[5]! + 0.8, c.ch[2]![0]! - 0.2, 0.6);
    const grid = Math.max(row, all) * (1 - back);
    return { m, draw, g, qk, hl, cur, curA, mask, hA, hS, hI, tags, keyTags, barOn, row, all, back, dim, cleanHi, card, probe, grid };
  }

  // ---------------------------------------------------------------- camera
  /** World box of the sentence panel (arcs, tokens, tags, bar). */
  panel(toks: Tok[]) {
    if (!P) {
      const x0 = toks[0]!.x, x1 = toks[toks.length - 1]!.x + toks[toks.length - 1]!.w;
      return { x0: x0 - 40, x1: x1 + 40, y0: -560, y1: 170 };
    }
    return { x0: -470, x1: 330, y0: toks[0]!.y - 80, y1: toks[toks.length - 1]!.y + 200 };
  }

  camera(t: number, s: ReturnType<Sentence['state']>, toks: Tok[]) {
    const c = this.cue, pb = this.panel(toks);
    const pw = pb.x1 - pb.x0, ph = pb.y1 - pb.y0;
    const areaW = P ? W * 0.94 : W * 0.9, areaH = P ? H * 0.76 : H * 0.7;
    const fit = Math.min(areaW / pw, areaH / ph);
    const cx0 = (pb.x0 + pb.x1) / 2, cy0 = (pb.y0 + pb.y1) / 2 + (P ? 20 : -10);
    // focus points: [cx, cy, zoom multiplier]
    let cx = cx0, cy = cy0, z = fit;
    const tok = (i: number) => toks[i]!;
    // verse 1 (no arcs yet): the words alone, bigger and centred on the sung ones
    const solo = 1 - up(t, c.ch[0]![0]! - 2.4, 2.6);
    if (solo > 0) {
      const a0 = P ? tok(1).y : tok(1).x, a1 = P ? tok(13).y : tok(13).x + tok(13).w;
      const zs = P ? Math.min(H * 0.6 / (a1 - a0 + 120), W * 0.8 / 420) : W * 0.84 / (a1 - a0);
      const sxc = P ? 60 : (a0 + a1) / 2, syc = P ? (a0 + a1) / 2 : -S * 0.9;
      cx = lerp(cx, sxc, solo); cy = lerp(cy, syc, solo);
      z = Math.exp(lerp(Math.log(z), Math.log(zs), solo));
    }
    // intro -> verse 1: a slow push in
    z *= 1.06 - 0.06 * up(t, c.v1[0]! - 2, 6);
    // bridge: closer on "it" .. "tired"
    const brF = span(t, c.br[1]!, c.brEnd + 1.5, 2.0);
    if (!P) cx = lerp(cx, (tok(2).x + tok(12).x + tok(12).w) / 2, brF * 0.3);
    else cy = lerp(cy, (tok(2).y + tok(12).y) / 2, brF * 0.5);
    z *= 1 + 0.05 * brF;
    // verse 3: pull back to the head's card, its row, the whole grid; then back in
    const cellW = pw + (P ? 60 : 90), cellH = ph + (P ? 80 : 60);
    const L0 = 8, H0 = 29;
    const gx = (15.5 - H0) * cellW, gy = -(15.5 - L0) * cellH;   // grid centre relative to this cell
    const zCard = z * 0.84;
    const zRow = Math.min(W * 0.94 / ((P ? 7 : 9) * cellW), H * 0.5 / cellH);
    const zAll = Math.min(W * (P ? 0.92 : 0.86) / (32 * cellW), H * (P ? 0.62 : 0.72) / (32 * cellH));
    let lz = Math.log(z);
    const k1 = up(t, c.v3[0]! + 0.1, 1.6), k2 = s.row, k3 = s.all, k4 = s.back;
    lz = lerp(lz, Math.log(zCard), k1 * (1 - k4));
    const lzRow = Math.log(zRow), lzAll = Math.log(zAll);
    // row, then all: centre moves in proportion to the visible extent so the pull-back reads as one move
    const ext = (zz: number) => 1 / zz;
    let tx = cx, ty = cy;
    if (k2 > 0) {
      const l2 = lerp(Math.log(zCard), lzRow, k2);
      const f = (ext(Math.exp(l2)) - ext(zCard)) / (ext(zRow) - ext(zCard));
      lz = lerp(lz, l2, 1);
      tx = lerp(cx0, cx0 + gx, f); ty = cy0;
      if (k3 > 0) {
        const l3 = lerp(lzRow, lzAll, k3);
        const f3 = (ext(Math.exp(l3)) - ext(zRow)) / (ext(zAll) - ext(zRow));
        lz = l3;
        tx = cx0 + gx; ty = lerp(cy0, cy0 + gy, f3);
      }
      if (k4 > 0) {
        const lFrom = lz, lTo = Math.log(fit);
        const l4 = lerp(lFrom, lTo, k4);
        const f4 = (ext(Math.exp(l4)) - ext(Math.exp(lFrom))) / (ext(fit) - ext(Math.exp(lFrom)) || 1);
        tx = lerp(tx, cx0, f4); ty = lerp(ty, cy0, f4);
        lz = l4;
      }
      cx = tx; cy = ty;
    }
    z = Math.exp(lz);
    // outro: a slow pull back
    z *= 1 - 0.1 * up(t, c.out[0]!, 8, ease.inOutQuad);
    // always a gentle drift
    // (a slow float, two periods so it never stops: ~20 px/s on screen, enough for the gate to see motion)
    const fl = float(t);
    cx += fl[0] / z;
    cy += fl[1] / z;
    z *= 1 + 0.02 * Math.sin(t * 0.19 + 0.4);
    return { cx, cy, z, cellW, cellH, fit };
  }

  // ---------------------------------------------------------------- render
  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const r = this.ctx.renderer;
    this.wash.render(r, out, { seed: 11, drift: f.t * 0.5, amt: 0.2, c1: 'blue', c2: 'pink' });
    const tex = perFrame(this.layer, f, () => this.draw(this.layer.ctx, f.ft));
    this.ctx.comp.draw(r, tex, out);
    const end = this.cue.end;
    const cap = this.caption(f.ft);
    return { reg: 5, fade: up(f.t, end - 2.2, 2.0), caption: cap.txt, captionA: cap.a };
  }

  draw(c: CanvasRenderingContext2D, t: number) {
    const s = this.state(t), cu = this.cue;
    this.m = s.m;
    const toks = this.layout(s.m);
    const cam = this.camera(t, s, toks);
    const z = cam.z;
    const sx = (x: number) => W / 2 + (x - cam.cx) * z;
    const sy = (y: number) => H / 2 + (y - cam.cy) * z;

    this.title(c, t);

    // grid cells (verse 3), under the featured panel
    if (s.grid > 0.001) this.gridCells(c, t, s, toks, cam, sx, sy, z);

    // the featured panel fades to a glyph when it is small
    const detail = clamp((z / cam.fit - 0.12) / 0.2) * (1 - 0.8 * span(t, cu.out[1]! - 0.2, cu.out[1]! + 3.2, 1.4));
    if (s.card > 0.001) {
      const pb = this.panel(toks);
      c.save();
      c.globalAlpha = s.card * clamp(detail * 2);
      c.strokeStyle = INK; c.lineWidth = 2;
      c.strokeRect(sx(pb.x0), sy(pb.y0), (pb.x1 - pb.x0) * z, (pb.y1 - pb.y0) * z);
      c.font = font(F.mono(700), 22);
      c.fillStyle = INK;
      c.fillText('ONE HEAD: LAYER 8, HEAD 29', sx(pb.x0) + 14, sy(pb.y0) + 32);
      c.restore();
    }

    // arcs from the asking word
    const q = Math.round(lerp(9, 12, s.qk));
    const w = this.weights(s.qk, s.m);
    const nk = lerp(10, 13, s.qk);
    const uni = w.map((_, i) => (i <= Math.max(9, q) ? 1 / nk : 0));
    const ws = w.map((x, i) => lerp(uni[i]!, x, s.g) * (i <= 9 || s.qk > 0 ? 1 : 0));
    const qTok = toks[q]!;
    const qa = this.anchor(qTok);
    // the asker's anchor slides from "it" to "tired"
    const itA = this.anchor(toks[9]!), tiA = this.anchor(toks[12]!);
    const Q: [number, number] = [lerp(itA[0], tiA[0], s.qk), lerp(itA[1], tiA[1], s.qk)];
    void qa;
    const calls = this.calls(t);
    const focus = { arc: -1, a: 0 };
    for (const k of calls) { const a = span(t, k.a0, k.a1, 0.45); if (k.arc !== undefined && a > focus.a) { focus.arc = k.arc; focus.a = a; } }
    if (s.draw > 0.001 && detail > 0.01) {
      for (let i = 0; i < toks.length; i++) {
        if (ws[i]! <= 0.0005) continue;
        const K = this.anchor(toks[i]!);
        const self = s.qk < 0.5 ? i === 9 : i === 12;
        let emph = (i === 2 || i === 7) ? 1 : 1 - 0.7 * s.hl;
        if (focus.arc >= 0 && i !== focus.arc) emph *= 1 - 0.65 * focus.a;
        this.ribbon(c, t, Q, K, ws[i]!, s.draw, self, emph * detail, sx, sy, z);
      }
    }

    // tokens
    c.save();
    c.font = font(FAM, S * z);
    c.textBaseline = 'alphabetic';
    for (let i = 0; i < toks.length; i++) {
      const tk = toks[i]!;
      let a = up(t, cu.tokenIn[i]!, 0.4) * detail;
      if (s.cur >= 0 && i > s.cur + 0.02) a *= 1 - 0.78 * s.mask;
      if (a <= 0.003) continue;
      const isQ = (s.qk < 0.5 && i === 9) || (s.qk >= 0.5 && i === 12);
      // pink while it asks: "Who's it?" in verse 1, then while its arcs are drawn (fades with them)
      const ak = Math.max(i === 9 ? span(t, cu.v1[2]! - 0.3, cu.ch[0]![0]! + 1, 0.5) : 0, isQ ? clamp(s.draw * 3) : 0);
      c.globalAlpha = a;
      if (i === 0) { this.startTok(c, sx(tk.x), sy(tk.y), tk.w * z, z); continue; }
      const X = sx(tk.x), Y = sy(tk.y) + (P ? S * 0.34 * z : 0);
      for (const [col, al] of [[INK, 1 - ak], [PINK, ak]] as const) {
        if (al <= 0.003) continue;
        c.globalAlpha = a * al;
        c.fillStyle = col;
        if (i === 12) this.morphWord(c, X, Y, s.m, z, a * al);
        else c.fillText(tk.label, X, Y);
      }
      if (ak > 0.003) { c.globalAlpha = a * ak; c.fillStyle = PINK; c.fillRect(X, Y + 10 * z, tk.w * z, 5 * z); }
    }
    c.restore();

    // the reading cursor
    if (s.curA > 0.003 && s.cur >= 0) {
      const i0 = Math.floor(s.cur), fr = s.cur - i0;
      const a0 = toks[Math.min(i0, toks.length - 1)]!, a1 = toks[Math.min(i0 + 1, toks.length - 1)]!;
      c.save();
      c.globalAlpha = s.curA;
      c.fillStyle = PINK;
      if (!P) {
        const x = lerp(a0.x, a1.x, fr), wv = lerp(a0.w, a1.w, fr);
        c.beginPath();
        const X = sx(x + wv / 2), Y = sy(S * 0.42);
        c.moveTo(X, Y); c.lineTo(X - 14 * z, Y + 22 * z); c.lineTo(X + 14 * z, Y + 22 * z); c.fill();
      } else {
        const y = lerp(a0.y, a1.y, fr), wv = lerp(a0.w, a1.w, fr);
        const X = sx(wv + 24), Y = sy(y);
        c.beginPath(); c.moveTo(X, Y); c.lineTo(X + 22 * z, Y - 14 * z); c.lineTo(X + 22 * z, Y + 14 * z); c.fill();
      }
      c.restore();
    }

    // "who's it?": your answer, a dashed arc
    const human = (target: number, k: number) => {
      if (k <= 0.003) return;
      const A0 = this.anchor(toks[9]!);
      const tA = this.anchor(toks[2]!), tS = this.anchor(toks[7]!);
      const K: [number, number] = [lerp(tA[0], tS[0], target), lerp(tA[1], tS[1], target)];
      this.dashed(c, A0, K, k, sx, sy, z);
    };
    human(0, s.hA);
    human(1, s.hS);
    human(s.m, s.hI);

    // tags: query (the asker), keys (every earlier word)
    if (s.tags > 0.003 && detail > 0.05) {
      c.save();
      const fs = 17 * Math.min(1.2, z / cam.fit);
      c.font = font(F.mono(700), fs);
      c.textBaseline = 'middle';
      for (let i = 0; i < toks.length; i++) {
        const isQ = (s.qk < 0.5 && i === 9) || (s.qk >= 0.5 && i === 12);
        const isK = i <= (s.qk < 0.5 ? 9 : 12) && !isQ;
        const a = (isQ ? s.tags : isK ? s.keyTags(i) : 0) * detail * (1 - s.grid);
        if (a <= 0.003) continue;
        const tk = toks[i]!;
        const label = isQ ? 'QUERY' : 'KEY';
        const tw = c.measureText(label).width + 12;
        const X = P ? sx(tk.x + tk.w) + 18 * z + (i === 0 ? 0 : 0) : sx(tk.x + tk.w / 2) - tw / 2;
        const Y = P ? sy(tk.y) : sy(S * 0.95);
        c.globalAlpha = a;
        c.fillStyle = isQ ? PINK : BLUE;
        c.fillRect(X, Y - fs * 0.75, tw, fs * 1.5);
        c.fillStyle = PAPER;
        c.fillText(label, X + 6, Y + 1);
      }
      c.restore();
    }

    // the weights, stacked into one bar that adds up to 1
    if (s.barOn > 0.003) this.weightBar(c, ws, toks, s.barOn * detail, sx, sy, z);

    // the probe: the model's own next-word guess
    if (s.probe > 0.003) this.probeLine(c, t, s.probe, toks, sx, sy, z);

    // one number at a time
    this.callouts(c, t, s, toks, calls, Q, sx, sy, z, cam);

    // final chorus: every piece at once, including where this head sits among all of them
    const inset = span(t, cu.ch[2]![0]! + 0.5, cu.out[1]!, 1.0);
    if (inset > 0.003) this.miniGrid(c, inset);

    // outro: the next episode
    this.outro(c, t);
  }

  /** All 32 x 32 heads as small squares (the 4 clean switches pink, this head outlined): "one head of many". */
  miniGrid(c: CanvasRenderingContext2D, a: number) {
    const n = this.a.d.config.layers, cell = 5, gap = 1.5, gw = n * (cell + gap);
    const x0 = P ? W - 50 - gw : W - 70 - gw, y0 = P ? 250 : 170;
    c.save();
    for (let l = 0; l < n; l++) for (let h = 0; h < n; h++) {
      const k = this.a.kind(l, h);
      c.globalAlpha = a * (k === 2 ? 1 : k === 0 ? 0.22 : 0.45);
      c.fillStyle = k === 2 || k === 1 ? PINK : k === -1 ? BLUE : INK;
      c.fillRect(x0 + h * (cell + gap), y0 + (n - 1 - l) * (cell + gap), cell, cell);
    }
    const L0 = this.a.d.featured.layer, H0 = this.a.d.featured.head;
    c.globalAlpha = a;
    c.strokeStyle = INK; c.lineWidth = 2;
    c.strokeRect(x0 + H0 * (cell + gap) - 3, y0 + (n - 1 - L0) * (cell + gap) - 3, cell + 6, cell + 6);
    c.font = font(F.mono(700), P ? 17 : 15);
    c.fillStyle = INK;
    c.textAlign = 'left';
    c.fillText('ONE HEAD OF MANY', x0, y0 + gw + 22);
    c.restore();
  }

  // ---------------------------------------------------------------- pieces
  title(c: CanvasRenderingContext2D, t: number) {
    const cu = this.cue;
    // centred until the singing starts, then it glides into a small header and waits there for the first chorus
    const k = up(t, cu.v1[0]! - 1.8, 3.0);
    const a = up(t, 0.4, 1.4) * (1 - up(t, cu.ch[0]![0]! - 1.4, 1.2));
    if (a <= 0.003) return;
    c.save();
    c.globalAlpha = a;
    const fl = float(t);
    c.translate(-fl[0] * (1 - k), -fl[1] * (1 - k));
    const size = P ? 150 : 190;
    c.font = font(F.archivo(112, 900), size);
    c.textAlign = 'left';
    c.textBaseline = 'alphabetic';
    const t1 = 'Who’s ', t2 = 'It?';
    const w1 = c.measureText(t1).width, w2 = c.measureText(t2).width;
    const x0 = W / 2 - (w1 + w2) / 2, y0 = H * (P ? 0.46 : 0.5) + size * 0.2 - 12 * up(t, 0.4, 6);
    const sc = lerp(1, P ? 0.4 : 0.34, k);
    const X = lerp(x0, P ? W / 2 - (w1 + w2) * 0.2 : 58, k), Y = lerp(y0, P ? 300 : 150, k);
    c.save();
    c.translate(X, Y); c.scale(sc, sc);
    c.fillStyle = INK; c.fillText(t1, 0, 0);
    c.fillStyle = PINK; c.fillText(t2, w1, 0);
    c.restore();
    c.globalAlpha = a * (1 - k);
    c.font = font(F.mono(600), P ? 28 : 26);
    c.fillStyle = INK;
    c.textAlign = 'center';
    c.fillText('how attention works, one sentence at a time', W / 2, y0 + (P ? 90 : 84));
    c.restore();
  }

  startTok(c: CanvasRenderingContext2D, X: number, Y: number, w: number, z: number) {
    const h = S * 0.78 * z;
    const y0 = P ? Y - h / 2 + 2 * z : Y - h + 6 * z;
    c.save();
    c.strokeStyle = INK; c.lineWidth = Math.max(1.2, 3 * z);
    c.strokeRect(X + 4 * z, y0, w - 8 * z, h);
    c.fillStyle = INK;
    c.font = font(F.mono(700), 20 * z);
    c.textBaseline = 'middle';
    c.textAlign = 'center';
    c.fillText('START', X + w / 2, y0 + h / 2 + 1 * z);
    c.restore();
  }

  /** The last word: "tired" rolls up and out as "wide" rolls in (and back). */
  morphWord(c: CanvasRenderingContext2D, X: number, Y: number, m: number, z: number, a: number) {
    const k = ease.inOutCubic(clamp(m));
    const d = S * 0.55 * z;
    c.save();
    if (k < 0.999) { c.globalAlpha = a * (1 - k); c.fillText(this.A[12]!, X, Y - k * d); }
    if (k > 0.001) { c.globalAlpha = a * k; c.fillText(this.B[12]!, X, Y + (1 - k) * d); }
    c.restore();
  }

  /** One attention arc as a tapered ribbon (thin at the asker, full width at the word it looks at). */
  ribbon(c: CanvasRenderingContext2D, t: number, Q: [number, number], K: [number, number], w: number, draw: number, self: boolean,
    alpha: number, sx: (x: number) => number, sy: (y: number) => number, z: number) {
    const N = 44;
    const dx = K[0] - Q[0], dy = K[1] - Q[1], dist = Math.hypot(dx, dy);
    const h = 40 + 0.33 * dist;
    let p1: [number, number], p2: [number, number];
    if (self) {
      p1 = P ? [Q[0] - 95, Q[1] - 45] : [Q[0] - 48, Q[1] - 105];
      p2 = P ? [Q[0] - 95, Q[1] + 45] : [Q[0] + 48, Q[1] - 105];
    } else if (!P) { p1 = [Q[0], Q[1] - h]; p2 = [K[0], K[1] - h]; }
    else { p1 = [Q[0] - h, Q[1]]; p2 = [K[0] - h, K[1]]; }
    const bez = (u: number): [number, number] => {
      const v = 1 - u;
      return [v * v * v * Q[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u * u * u * K[0],
        v * v * v * Q[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u * u * u * K[1]];
    };
    const Wk = 2.5 + 84 * w;           // world width at the key end
    const pts: [number, number][] = [], wid: number[] = [];
    const n = Math.max(2, Math.ceil(N * draw));
    for (let j = 0; j <= n; j++) {
      const u = (j / n) * draw;
      const [x, y] = bez(u);
      pts.push([sx(x), sy(y)]);
      wid.push(Math.max(1.3, Wk * (0.35 + 0.65 * u) * z));
    }
    // outline polygon
    const L: [number, number][] = [], R: [number, number][] = [];
    for (let j = 0; j < pts.length; j++) {
      const a = pts[Math.max(0, j - 1)]!, b = pts[Math.min(pts.length - 1, j + 1)]!;
      let nx = -(b[1] - a[1]), ny = b[0] - a[0];
      const l = Math.hypot(nx, ny) || 1; nx /= l; ny /= l;
      const hw = wid[j]! / 2;
      L.push([pts[j]![0] + nx * hw, pts[j]![1] + ny * hw]);
      R.push([pts[j]![0] - nx * hw, pts[j]![1] - ny * hw]);
    }
    c.save();
    c.globalAlpha = clamp(alpha);
    c.fillStyle = PINK;
    c.beginPath();
    c.moveTo(L[0]![0], L[0]![1]);
    for (const p of L) c.lineTo(p[0], p[1]);
    for (let j = R.length - 1; j >= 0; j--) c.lineTo(R[j]![0], R[j]![1]);
    c.closePath();
    c.fill();
    // flow: paper dots drifting from the word back into the asker (what it takes from each word)
    const endW = wid[wid.length - 1]!;
    if (draw > 0.9 && endW > 9) {
      c.globalAlpha = clamp(alpha) * clamp((draw - 0.9) / 0.1);
      let len = 0;
      const acc = [0];
      for (let j = 1; j < pts.length; j++) { len += Math.hypot(pts[j]![0] - pts[j - 1]![0], pts[j]![1] - pts[j - 1]![1]); acc.push(len); }
      const gap = 70, off = (t * 46) % gap;
      c.fillStyle = PAPER;
      for (let d0 = len - off; d0 > 0; d0 -= gap) {
        let j = 1;
        while (j < acc.length - 1 && acc[j]! < d0) j++;
        const f0 = (d0 - acc[j - 1]!) / Math.max(1e-3, acc[j]! - acc[j - 1]!);
        const x = lerp(pts[j - 1]![0], pts[j]![0], f0), y = lerp(pts[j - 1]![1], pts[j]![1], f0);
        const r = lerp(wid[j - 1]!, wid[j]!, f0) * 0.16;
        if (r < 1.2) continue;
        c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill();
      }
    }
    c.restore();
  }

  /** Your answer to "who's it?": a hand-drawn dashed arc (ink), with a small "you" label. */
  dashed(c: CanvasRenderingContext2D, A0: [number, number], K: [number, number], k: number, sx: (x: number) => number, sy: (y: number) => number, z: number) {
    const dist = Math.hypot(K[0] - A0[0], K[1] - A0[1]), h = 60 + 0.36 * dist;
    const p1: [number, number] = P ? [A0[0] - h, A0[1]] : [A0[0], A0[1] - h];
    const p2: [number, number] = P ? [K[0] - h, K[1]] : [K[0], K[1] - h];
    c.save();
    c.strokeStyle = INK;
    c.lineWidth = Math.max(2, 4.5 * z);
    c.setLineDash([14 * z, 10 * z]);
    c.lineCap = 'round';
    c.beginPath();
    const N = 50, n = Math.max(1, Math.ceil(N * k));
    for (let j = 0; j <= n; j++) {
      const u = (j / n) * k, v = 1 - u;
      const x = v * v * v * A0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u * u * u * K[0];
      const y = v * v * v * A0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u * u * u * K[1];
      if (j === 0) c.moveTo(sx(x), sy(y)); else c.lineTo(sx(x), sy(y));
    }
    c.stroke();
    if (k > 0.97) {
      // arrow head at the answer
      c.setLineDash([]);
      const X = sx(K[0]), Y = sy(K[1]);
      c.fillStyle = INK;
      c.beginPath();
      if (!P) { c.moveTo(X, Y + 2 * z); c.lineTo(X - 11 * z, Y - 16 * z); c.lineTo(X + 11 * z, Y - 16 * z); }
      else { c.moveTo(X + 2 * z, Y); c.lineTo(X - 16 * z, Y - 11 * z); c.lineTo(X - 16 * z, Y + 11 * z); }
      c.fill();
    }
    // label
    const mid = 0.5, v = 1 - mid;
    const lx = v * v * v * A0[0] + 3 * v * v * mid * p1[0] + 3 * v * mid * mid * p2[0] + mid * mid * mid * K[0];
    const ly = v * v * v * A0[1] + 3 * v * v * mid * p1[1] + 3 * v * mid * mid * p2[1] + mid * mid * mid * K[1];
    c.globalAlpha = clamp((k - 0.5) * 2);
    c.font = font(F.mono(700), 24 * z);
    c.fillStyle = INK;
    c.textAlign = 'center';
    c.fillText('YOU', sx(lx) - (P ? 30 * z : 0), sy(ly) - (P ? 0 : 16 * z));
    c.restore();
  }

  weightBar(c: CanvasRenderingContext2D, ws: number[], toks: Tok[], a: number, sx: (x: number) => number, sy: (y: number) => number, z: number) {
    const tot = ws.reduce((p, x) => p + x, 0) || 1;
    let x0: number, x1: number, y: number;
    if (!P) { x0 = toks[0]!.x; x1 = toks[toks.length - 1]!.x + toks[toks.length - 1]!.w; y = S * 1.9; }
    else { x0 = -440; x1 = 300; y = toks[toks.length - 1]!.y + 130; }
    const hgt = 34;
    c.save();
    c.globalAlpha = a;
    let x = x0;
    const Wb = x1 - x0;
    c.font = font(F.mono(700), 15);
    c.textBaseline = 'top';
    for (let i = 0; i < ws.length; i++) {
      const wv = (ws[i]! / tot) * Wb;
      if (wv <= 0.01) continue;
      c.fillStyle = PINK;
      c.globalAlpha = a * (i % 2 === 0 ? 1 : 0.55);
      c.fillRect(sx(x), sy(y), Math.max(0.5, wv * z - 2), hgt * z);
      c.globalAlpha = a;
      if (wv * z > 62) {
        c.fillStyle = INK;
        c.fillText(i === 0 ? 'START' : i === 12 ? (this.m > 0.5 ? this.B[12]! : this.A[12]!) : toks[i]!.label, sx(x) + 3, sy(y + hgt) + 6);
      }
      x += wv;
    }
    c.strokeStyle = INK; c.lineWidth = 2;
    c.strokeRect(sx(x0), sy(y), Wb * z, hgt * z);
    c.restore();
  }

  probeLine(c: CanvasRenderingContext2D, t: number, a: number, toks: Tok[], sx: (x: number) => number, sy: (y: number) => number, z: number) {
    const d = this.a.d;
    const lead = 'The one that was too tired was the';
    const pick = d.probe.A.top10[0]![0].trim();
    c.save();
    c.globalAlpha = a;
    const fs = (P ? 34 : 36);
    c.font = font(F.archivo(100, 500), fs);
    c.textBaseline = 'alphabetic';
    const X = P ? W * 0.08 : sx(toks[1]!.x);
    const Y = P ? H * 0.86 : sy(S * 2.4);
    c.fillStyle = INK;
    c.fillText(lead, X, Y);
    const lw = c.measureText(lead + ' ').width;
    const k = up(t, this.cue.v3[5]! + 1.6, 0.6);
    c.globalAlpha = a * k;
    c.font = font(F.archivo(100, 900), fs);
    c.fillStyle = PINK;
    c.fillText(pick, X + lw, Y);
    c.fillRect(X + lw, Y + 8, c.measureText(pick).width, 4);
    c.restore();
  }

  /** The single number on screen: [from, to, text, where]. */
  /** The numbers, one at a time: when, what, and which arc it belongs to (or none: printed low centre). */
  calls(t: number): { a0: number; a1: number; text: string; arc?: number }[] {
    const cu = this.cue, d = this.a.d;
    const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
    const n = (x: number) => x.toLocaleString('en-US');
    const rowIt = this.a.row(0, 9), rowA = this.a.row(0, 12), rowB = this.a.row(1, 12);
    const g = d.grid.change;
    const L: { a0: number; a1: number; text: string; arc?: number; bar?: boolean }[] = [];
    for (let k = 0; k < 3; k++) L.push({ a0: cu.ch[k]![2]! + 0.9, a1: cu.ch[k]![3]! + 0.3, text: '= 1', bar: true });
    L.push({ a0: cu.v2[2]! + 1.2, a1: cu.v2[3]! + 0.2, text: pct(rowIt[2]!), arc: 2 });
    L.push({ a0: cu.v2[4]! + 0.4, a1: cu.v2end + 2.0, text: pct(rowIt[0]!), arc: 0 });
    L.push({ a0: cu.br[3]! + 0.4, a1: cu.br[4]!, text: `all ${n(d.config.layers * d.config.heads)} heads: the same weights` });
    L.push({ a0: cu.br[4]! + 1.6, a1: cu.br[5]!, text: pct(rowA[2]!), arc: 2 });
    L.push({ a0: cu.br[5]! + 1.4, a1: cu.brEnd + 2.0, text: pct(rowB[7]!), arc: 7 });
    L.push({ a0: cu.v3[3]! + 0.6, a1: cu.v3[4]!, text: `${n(g.no_flip)} of ${n(g.heads)} don’t switch` });
    L.push({ a0: cu.v3[4]! + 0.5, a1: cu.v3[5]! - 0.2, text: `${g.clean} switch clearly` });
    L.push({ a0: cu.v3[5]! + 2.0, a1: cu.ch[2]![0]! - 0.3, text: pct(d.probe.A.words.animal ?? 0) });
    L.push({ a0: cu.ch[2]![3]! + 1.3, a1: cu.out[0]! + 1.0, text: pct(rowB[7]!), arc: 7 });
    return L.filter((x) => t > x.a0 - 1 && t < x.a1 + 1);
  }

  callouts(c: CanvasRenderingContext2D, t: number, s: ReturnType<Sentence['state']>, toks: Tok[], calls: ReturnType<Sentence['calls']>, Q: [number, number],
    sx: (x: number) => number, sy: (y: number) => number, z: number, cam: { fit: number }) {
    const arcPeak = (i: number): [number, number] => {
      const K = this.anchor(toks[i]!);
      const dist = Math.hypot(K[0] - Q[0], K[1] - Q[1]), h = 40 + 0.33 * dist;
      return P ? [lerp(Q[0], K[0], 0.5) - h * 0.75 - 60, lerp(Q[1], K[1], 0.5)] : [lerp(Q[0], K[0], 0.5), Q[1] - h * 0.75 - 44];
    };
    const barEnd = (): [number, number] => (P ? [300, toks[toks.length - 1]!.y + 225] : [toks[toks.length - 1]!.x + toks[toks.length - 1]!.w + 70, S * 1.9 + 18]);
    const list: [number, number, string, () => [number, number], boolean][] = calls.map((k) =>
      [k.a0, k.a1, k.text, k.arc !== undefined ? () => arcPeak(k.arc!) : barEnd, k.arc === undefined && !(k as { bar?: boolean }).bar]);
    for (const [a0, a1, text, where, screen] of list) {
      const a = span(t, a0, a1, 0.35);
      if (a <= 0.003) continue;
      let X: number, Y: number;
      if (screen) { X = W / 2; Y = P ? H * 0.925 : H * 0.9; }
      else { const p = where(); X = sx(p[0]); Y = sy(p[1]); }
      if (!screen && s.grid > 0.5) continue;
      if (screen && text.includes('%')) { X = W * 0.5; Y = P ? H * 0.925 : H * 0.93; }
      c.save();
      c.globalAlpha = a;
      const fs = screen ? (P ? 34 : 34) : 34 * Math.min(1.25, z / cam.fit);
      c.font = font(F.mono(700), fs);
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      const tw = c.measureText(text).width;
      c.fillStyle = PAPER;
      c.fillRect(X - tw / 2 - 12, Y - fs * 0.72, tw + 24, fs * 1.44);
      c.strokeStyle = INK; c.lineWidth = 2;
      c.strokeRect(X - tw / 2 - 12, Y - fs * 0.72, tw + 24, fs * 1.44);
      c.fillStyle = INK;
      c.fillText(text, X, Y + 1);
      c.restore();
    }
  }

  /** The source of what is on screen (printed crisp by the overlay); dips out while it changes. */
  caption(t: number): { txt: string; a: number } {
    const cu = this.cue, s = this.state(t), d = this.a.d;
    const on = up(t, cu.ch[0]![0]! - 0.5, 0.8) * (1 - span(t, cu.out[1]!, cu.out[1]! + 3.2, 1));
    const model = 'LLAMA 3.1 8B INSTRUCT';
    if (s.probe > 0.001) return { txt: `${model} · ITS OWN GUESS AT THE NEXT WORD`, a: on * s.probe };
    if (s.grid > 0.5) return { txt: `${model} · EVERY HEAD: ${d.config.layers} LAYERS × ${d.config.heads} HEADS`, a: on * (s.grid - 0.5) * 2 };
    const back = t > cu.v3[5]! ? 1 - up(t, cu.v3[5]! + 0.3, 0.4) : 1;
    return { txt: `${model} · LAYER ${d.featured.layer}, HEAD ${d.featured.head} · REAL ATTENTION WEIGHTS`, a: on * (1 - s.grid * 2) * (t > cu.v3[5]! && t < cu.ch[2]![0]! ? back : 1) };
  }

  outro(c: CanvasRenderingContext2D, t: number) {
    const cu = this.cue;
    const a = up(t, cu.out[1]! + 0.2, 1.2);
    if (a <= 0.003) return;
    c.save();
    c.globalAlpha = a;
    const fl = float(t);
    c.translate(-fl[0] * 0.6, -fl[1] * 0.6);
    // after 6 s it shrinks out of the way (top in 16:9, bottom in 9:16) and the sentence comes back under it
    const k = up(t, cu.out[1]! + 3.2, 1.6);
    const yMid = H * (P ? 0.44 : 0.4), yEnd = P ? H * 0.855 : H * 0.155;
    c.translate(W / 2, lerp(yMid, yEnd, k));
    c.scale(1 - 0.4 * k, 1 - 0.4 * k);
    c.translate(-W / 2, -yMid);
    c.textAlign = 'center';
    const title = 'Watch It Learn to Write';
    let fs = P ? 76 : 96;
    c.font = font(F.archivo(112, 900), fs);
    const tw = c.measureText(title).width;
    if (tw > W * 0.88) { fs *= (W * 0.88) / tw; c.font = font(F.archivo(112, 900), fs); }
    c.fillStyle = PAPER;
    c.globalAlpha = a * 0.9;
    c.fillRect(W / 2 - Math.min(tw, W * 0.88) / 2 - 24, H * (P ? 0.44 : 0.4) - 50, Math.min(tw, W * 0.88) + 48, fs * 1.2 + 120);
    c.globalAlpha = a;
    c.fillStyle = INK;
    c.font = font(F.mono(700), P ? 26 : 24);
    c.fillText('NEXT · 5/5', W / 2, H * (P ? 0.44 : 0.4));
    c.font = font(F.archivo(112, 900), fs);
    c.fillStyle = PINK;
    c.fillText(title, W / 2, H * (P ? 0.44 : 0.4) + fs * 0.6 + 64);
    c.restore();
  }

  gridCells(c: CanvasRenderingContext2D, t: number, s: ReturnType<Sentence['state']>, toks: Tok[], cam: ReturnType<Sentence['camera']>,
    sx: (x: number) => number, sy: (y: number) => number, z: number) {
    const a = this.a, L0 = 8, H0 = 29, cu = this.cue;
    const nL = a.d.config.layers, nH = a.d.config.heads;
    const anchors = toks.map((tk) => this.anchor(tk));
    const Q = anchors[12]!;
    const cellW = cam.cellW, cellH = cam.cellH;
    const pb = this.panel(toks);
    c.save();
    for (let l = 0; l < nL; l++) {
      for (let h = 0; h < nH; h++) {
        const featured = l === L0 && h === H0;
        const rowIn = l === L0 ? up(t, cu.v3[1]! + 0.3 + 0.07 * Math.abs(h - H0), 0.6) : 0;
        const allIn = up(t, cu.v3[2]! + 0.3 + 0.1 * Math.abs(l - L0), 0.7);
        let vis = Math.max(rowIn, allIn) * (1 - s.back);
        if (featured) vis = (1 - clamp((z / cam.fit - 0.12) / 0.2)) * s.grid;
        if (vis <= 0.003) continue;
        const ox = (h - H0) * cellW, oy = -(l - L0) * cellH;
        const x0 = sx(pb.x0 + ox), y0 = sy(pb.y0 + oy), cw = (pb.x1 - pb.x0) * z, chh = (pb.y1 - pb.y0) * z;
        if (x0 > W || y0 > H || x0 + cw < 0 || y0 + chh < 0) continue;
        const kind = a.kind(l, h);
        let alpha = vis;
        if (kind === 0) alpha *= 1 - 0.75 * s.dim;
        const col = kind === 2 || kind === 1 ? PINK : kind === -1 ? BLUE : INK;
        const colA = kind === 2 ? 1 : kind === 1 ? 0.6 : kind === -1 ? 0.8 : 0.5;
        // clean flips get a solid pink frame when highlighted
        if (kind === 2 && s.cleanHi > 0.01) {
          c.globalAlpha = alpha * s.cleanHi;
          c.strokeStyle = PINK; c.lineWidth = 3;
          c.strokeRect(x0 - 2, y0 - 2, cw + 4, chh + 4);
        }
        if (featured) {
          c.globalAlpha = alpha;
          c.strokeStyle = INK; c.lineWidth = 2.5;
          c.strokeRect(x0 - 3, y0 - 3, cw + 6, chh + 6);
        }
        // the head's arcs from "tired"/"wide", morphing with the word
        const rA = a.gridRow(l, h, 0), rB = a.gridRow(l, h, 1);
        c.globalAlpha = alpha * colA;
        c.strokeStyle = col;
        c.lineCap = 'round';
        const qx = sx(Q[0] + ox), qy = sy(Q[1] + oy);
        for (let i = 0; i < rA.length; i++) {
          const wv = lerp(rA[i]!, rB[i]!, s.m);
          if (wv < 0.01) continue;
          const K = anchors[i]!;
          const kx = sx(K[0] + ox), ky = sy(K[1] + oy);
          const dist = Math.hypot(K[0] - Q[0], K[1] - Q[1]), hh = (40 + 0.33 * dist) * z;
          c.lineWidth = Math.max(0.6, wv * 7 * Math.min(1.6, z / 0.03 * 0.6));
          c.beginPath();
          c.moveTo(qx, qy);
          if (!P) c.bezierCurveTo(qx, qy - hh, kx, ky - hh, kx, ky);
          else c.bezierCurveTo(qx - hh, qy, kx - hh, ky, kx, ky);
          c.stroke();
        }
        // the asker as a small ink mark; big cells also show the sentence as a row of ticks
        c.globalAlpha = vis * 0.7;
        c.fillStyle = INK;
        c.fillRect(qx - 1.5, qy - 1, 3, 3);
        const cd = clamp((cw - 70) / 90);
        if (cd > 0.01) {
          c.globalAlpha = vis * cd;
          for (let i = 0; i < anchors.length; i++) {
            const K = anchors[i]!, X = sx(K[0] + ox), Y = sy(K[1] + oy);
            c.fillStyle = i === 12 ? PINK : INK;
            if (!P) c.fillRect(X - 2.5, Y + 3, 5, 9 * cd + 3);
            else c.fillRect(X + 3, Y - 2.5, 9 * cd + 3, 5);
          }
        }
      }
    }
    // axis labels and legend (screen space)
    const lab = up(t, cu.v3[2]! + 1.5, 0.8) * (1 - s.back);
    if (lab > 0.003) {
      c.globalAlpha = lab;
      c.fillStyle = INK;
      c.font = font(F.mono(700), P ? 18 : 16);
      const gx0 = sx(pb.x0 + (0 - H0) * cellW), gx1 = sx(pb.x1 + (nH - 1 - H0) * cellW);
      const gyTop = sy(pb.y0 - (nL - 1 - L0) * cellH), gyBot = sy(pb.y1 + L0 * cellH);
      c.textAlign = 'left';
      c.fillText('HEAD 0', gx0, gyBot + 26);
      c.textAlign = 'right';
      c.fillText('HEAD 31', gx1, gyBot + 26);
      c.save();
      c.translate(gx0 - 14, gyBot); c.rotate(-Math.PI / 2); c.textAlign = 'left'; c.fillText('LAYER 0', 0, 0); c.restore();
      c.save();
      c.translate(gx0 - 14, gyTop); c.rotate(-Math.PI / 2); c.textAlign = 'right'; c.fillText('LAYER 31', 0, 0); c.restore();
      const leg = up(t, cu.v3[4]! + 0.3, 0.8) * (1 - s.back);
      if (leg > 0.003) {
        c.globalAlpha = leg;
        const items: [string, string, number][] = [[PINK, 'SWITCH CLEARLY', 1], [PINK, 'LEAN RIGHT', 0.6], [BLUE, 'LEAN WRONG', 0.8], [INK, 'DON’T SWITCH', 0.5]];
        c.textAlign = 'left';
        let x = P ? W * 0.08 : gx0;
        const y = gyTop - (P ? 70 : 30);
        for (const [col, txt, al] of items) {
          c.globalAlpha = leg * al;
          c.fillStyle = col;
          c.fillRect(x, y - 12, 22, 14);
          c.globalAlpha = leg;
          c.fillStyle = INK;
          c.fillText(txt, x + 30, y);
          x += c.measureText(txt).width + 70;
          if (P && x > W * 0.7) { x = W * 0.08; }
        }
      }
    }
    c.restore();
  }
}
