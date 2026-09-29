// "Tokens" (ML, slowly #2): the whole video is one sheet of paper and one camera. Every idea lives at a
// station on the sheet and stays there; the camera travels between them (no cuts), labels draw in and
// stay up, and the last chorus pulls back over all of it.
//   Q  the question as a line of type: letters -> Llama's 8 chunks -> 8 numbers -> into the model
//   B  a line from this song under our own BPE: letters, then the most frequent pair glued, again and again
//   V  Llama 3.1's vocabulary in id order: " strawberry" among its neighbours, then all 128,256 as dots
//   S  " strawberry": one chunk, one number (73700); T the test: one chunk -> it says 2, letters -> 3
//   D  same letters, different numbers (no space: str|aw|berry; capital S: 89077)
// Every number is from data/tok.json. Times come from the sung words (data/lyrics.json).
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { F, font } from '../engine/type';
import { clamp, ease, lerp, smoothstep } from '../engine/util';
import { hudInfo } from '../engine/hud';
import type { Lyrics } from '../engine/lyrics';
import { Wash, perFrame, portrait } from './_kit';
import {
  PINK, BLUE, INK, PAPER, pinkA, blueA, inkA, k, win, flash, rrect,
  type RowGeom, type RowState, newState, TIGHT, drawRow, chunkBox, charBox, gapOf, label, drawLine, modelBox, slip, bracket, tally,
} from './world-draw';

type Rect = [number, number, number, number];
interface CamKey { t: number; d: number; r: Rect }
interface Cam { cx: number; cy: number; z: number }

const normW = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');
const show = (t: string) => t.replace(/ /g, '·');

/** Chunks (char ranges) of a token list shown as one string (spaces as '·'). */
function chunksOf(toks: string[]): { chars: string[]; chunks: [number, number][] } {
  const chars: string[] = [], chunks: [number, number][] = [];
  for (const t of toks) {
    const s = show(t);
    chunks.push([chars.length, chars.length + s.length - 1]);
    chars.push(...s.split(''));
  }
  return { chars, chunks };
}

export default class World extends Scene {
  wash = new Wash();
  layer = new Layer2D();
  A: Record<string, number> = {};
  C: Record<string, number>[] = [];
  merges: number[] = [];
  keys: CamKey[] = [];
  P = portrait();
  // stations
  gQ!: RowGeom; qChunks: [number, number][] = []; qIds: number[] = [];
  gB!: RowGeom; bWords: string[] = []; bWordStart: number[] = [];
  gS!: RowGeom;
  gD: RowGeom[] = []; dChunks: [number, number][][] = []; dIds: number[][] = [];
  M = { x: 0, y: 0, w: 0, h: 0 };
  V = { ox: 0, oy: 0, cw: 28, chh: 10, cols: 358, rows: 359, sx: 0, sy: 0 };
  field?: HTMLCanvasElement;
  T = { y1: 0, y2: 0, x: 0 };
  X = 0;
  R: Record<string, Rect> = {};

  override async init() {
    const tok = this.ctx.tok, ly = this.ctx.lyrics;
    this.anchors(ly);
    this.layout();
    this.camera();
    this.buildField();
    void tok;
  }

  // ---------------------------------------------------------------- times from the sung words
  private anchors(ly: Lyrics) {
    const line = (q: string, nth = 0) => { try { return ly.get(q, nth); } catch { return null; } };
    const w = (q: string, p: string, nth = 0, occ = 0) => {
      const l = line(q, nth);
      if (!l) return NaN;
      const ws = l.words.filter((x) => normW(x.w).startsWith(p));
      return ws[occ]?.start ?? ws[ws.length - 1]?.start ?? l.start;
    };
    const lend = (q: string, nth = 0) => line(q, nth)?.end ?? NaN;
    const A: Record<string, number> = {};
    const v0 = ly.lines[0]?.start ?? 12;
    A.type0 = Math.max(1.0, Math.min(v0 - 9, v0 * 0.35));
    A.type1 = A.type0 + Math.min(5, (v0 - A.type0) * 0.5);
    A.ans = A.type1 + 0.8;
    A.asked = w('You asked', 'you');
    A.model1 = w('The model, that', 'model'); A.two1 = w('The model, that', 'two');
    A.count = w('Count them yourself', 'count'); A.yourself = w('Count them yourself', 'yourself'); A.three1 = w('Count them yourself', 'three');
    A.never1 = w('It never saw', 'never'); A.letters1 = w('It never saw', 'letters');
    A.before = w('Before it reads', 'before'); A.cut = w('Before it reads', 'cut'); A.chunks1 = w('Before it reads', 'chunks');
    A.swapped = w('is swapped for a number', 'swapped'); A.number1 = w('is swapped for a number', 'number');
    A.eight = w('Eight little chunks', 'eight'); A.eightNum = w('Eight little chunks', 'eight', 0, 1); A.row = w('Eight little chunks', 'row');
    A.shown = w('the numbers are all', 'numbers'); A.shownEnd = lend('the numbers are all');
    for (let n = 0; n < 3; n++) {
      this.C.push({
        start: line('A token is a chunk', n)?.start ?? NaN,
        token: w('A token is a chunk', 'token', n), text: w('A token is a chunk', 'text', n),
        number: w('Every chunk gets a number', 'number', n), reads: w('The model reads the numbers', 'reads', n),
        numbers: w('The model reads the numbers', 'numbers', n), never: w('Never the letters', 'never', n),
        end: lend('Never the letters', n),
      });
    }
    A.where = w('Where do the chunks', 'where'); A.take = w('Take this song', 'take'); A.break = w('Take this song', 'break');
    A.lettersB = w('Take this song', 'letters');
    A.pair = w('Find the pair', 'pair'); A.most = w('Find the pair', 'most');
    A.eAndR = w('E and R', 'e'); A.glue1 = w('E and R', 'glue'); A.together = lend('E and R');
    A.countB = w('Count again and glue', 'count'); A.glueB = w('Count again and glue', 'glue');
    A.till = w('Till the words', 'till'); A.whole = w('Till the words', 'whole'); A.wholeEnd = lend('Till the words');
    A.made = w("chunks were made this way", 'model'); A.way = w('chunks were made this way', 'way');
    A.hundred = w('A hundred twenty-eight', 'hundred'); A.told = lend('A hundred twenty-eight');
    A.look = w('Look at strawberry', 'look'); A.straw3 = w('Look at strawberry', 'strawberry');
    A.space = w('With a space in front', 'space'); A.oneChunk = w('With a space in front', 'one');
    A.oneNumber = w('One number', 'number');
    const dig = ['seven', 'three', 'seven', 'oh', 'oh'];
    const numLine = line('One number');
    const echo = numLine ? numLine.words.slice(2) : [];
    dig.forEach((d, i) => {
      const hit = echo.filter((x) => normW(x.w).startsWith(d === 'oh' ? 'o' : d));
      const occ = dig.slice(0, i).filter((x) => x === d).length;
      A[`dig${i}`] = hit[occ]?.start ?? NaN;
    });
    A.whereR = w('Where are the', 'where'); A.rsIn = w('Where are the', 'rs');
    A.give = w('Give it one chunk', 'give'); A.says2 = w('Give it one chunk', 'says');
    A.spaceOut = w('Space out the letters', 'space'); A.says3 = w('Space out the letters', 'says');
    A.hiding = w('were hiding in the chunk', 'hiding'); A.onlySaw = w('It only ever saw', 'saw'); A.onlyNum = w('It only ever saw', 'number');
    A.wrote = w('When it wrote the word back', 'wrote'); A.noSpace = w('When it wrote the word back', 'space');
    A.came = w('It came out in three', 'came'); A.str = w('It came out in three', 'str'); A.aw = w('It came out in three', 'aw');
    A.berry = w('It came out in three', 'berry');
    A.capital = w('A capital S', 'capital'); A.another = w('A capital S', 'another');
    A.same = w('Same letters, different', 'same'); A.different = w('Same letters, different', 'different');
    A.tag = w('Never the letters', 'never', 3);
    A.what = w('What does a number mean', 'what'); A.mean = w('What does a number mean', 'mean');
    A.next = w('Next time', 'next'); A.direction = w('Next time', 'direction'); A.last = ly.lines[ly.lines.length - 1]?.end ?? NaN;
    A.end = this.ctx.audio.duration;
    // anything the take dropped: a second and a half after the anchor before it (keeps the order)
    let prev = 0;
    for (const key of Object.keys(A)) { if (!Number.isFinite(A[key]!)) A[key] = prev + 1.5; prev = A[key]!; }
    for (const c of this.C) { let p = c.start!; for (const key of Object.keys(c)) { if (!Number.isFinite(c[key]!)) c[key] = p + 1.5; p = c[key]!; } }
    for (let i = 1; i < 5; i++) if (!(A[`dig${i}`]! > A[`dig${i - 1}`]!)) A[`dig${i}`] = A[`dig${i - 1}`]! + 0.35;
    this.A = A;
    // our BPE's merges, one at a time: 1 on "glue", 2 on "Count", 3 on "glue" (again), then a run through
    // "Till the words we sing the most are whole" up to merge 29 ("never" whole), the rest in the
    // instrumental after chorus 2 (the line is finished by merge 56).
    const M = this.ctx.tok.bpe.merges.length;
    const m: number[] = new Array(M + 1).fill(Infinity);
    m[1] = A.glue1!; m[2] = A.countB!; m[3] = A.glueB!;
    const r0 = A.till!, r1 = Math.max(A.wholeEnd!, A.till! + 3) + 0.6;
    for (let i = 4; i <= 29; i++) m[i] = lerp(r0, r1, ease.inOutQuad((i - 4) / 25));
    const c2end = this.C[1]!.end!, v3 = A.look! - 0.8;
    const i0 = c2end + 1.2, i1 = Math.max(i0 + 3, v3 - 2.0), im = lerp(i0, i1, 0.62);
    const last = this.ctx.tok.bpe.line.steps[this.ctx.tok.bpe.line.steps.length - 1]!.k;
    for (let i = 30; i <= M; i++) m[i] = i <= last ? lerp(i0, im, (i - 30) / Math.max(1, last - 30)) : lerp(im, i1, (i - last) / Math.max(1, M - last));
    this.merges = m;
  }

  // ---------------------------------------------------------------- where things are on the sheet
  private layout() {
    const P = this.P, tok = this.ctx.tok;
    // Q: the question, as Llama cuts it
    const q = chunksOf(tok.llama.question_tokens.map((x) => x.t));
    this.qChunks = q.chunks; this.qIds = tok.llama.question_tokens.map((x) => x.id);
    const qBreak = P ? [q.chunks[6]![0]] : [];
    this.gQ = { chars: q.chars, x: 0, y: 0, cw: P ? 48 : 46, ch: P ? 70 : 64, breaks: qBreak, lineGap: 46 };
    this.M = P ? { x: -450, y: 250, w: 900, h: 230 } : { x: -520, y: 190, w: 1040, h: 190 };
    // B: a line of this song under our BPE (words separated by a gap, never fused)
    const words = tok.bpe.line.words;
    this.bWords = words;
    const chars: string[] = [];
    words.forEach((wd) => { this.bWordStart.push(chars.length); chars.push(...wd.split('')); });
    const bBreaks = P ? [this.bWordStart[3]!, this.bWordStart[5]!] : [];
    this.gB = { chars, x: 0, y: P ? 1350 : 1080, cw: P ? 44 : 38, ch: P ? 62 : 54, breaks: bBreaks, lineGap: 40 };
    // S, T, D sit in a second column in landscape (the overview stays compact), below in portrait
    const X = P ? 0 : 2500;
    this.X = X;
    // V: Llama's vocabulary as a grid in id order, " strawberry" at the station's centre, clear of the rest
    const f = tok.llama.field;
    const [r0, c0] = f.strawberry;
    const vx = P ? -5012 + (c0 + 0.5) * 28 : 3700 + (c0 + 0.5) * 28, vy = P ? 5500 + (r0 + 0.5) * 10 : 700;
    this.V = { ...this.V, cols: f.cols, rows: Math.ceil(tok.llama.vocab_total / f.cols), ox: vx - (c0 + 0.5) * 28, oy: vy - (r0 + 0.5) * 10, sx: vx, sy: vy };
    // S: " strawberry" big
    const s = chunksOf([' strawberry']);
    this.gS = { chars: s.chars, x: X, y: P ? 2350 : 0, cw: P ? 80 : 84, ch: P ? 112 : 116, breaks: [], lineGap: 0 };
    // T: the test, two input rows
    this.T = P ? { x: 0, y1: 3050, y2: 3650 } : { x: X, y1: 620, y2: 950 };
    // D: same letters, different numbers
    const ans = tok.llama.answer_tokens;
    const rows = [[' strawberry'], ans.slice(8, 14).map((x) => x.t), [' Strawberry']];
    const ids = [[73700], ans.slice(8, 14).map((x) => x.id), [tok.llama.words[' Strawberry']![0]!.id]];
    const dy0 = P ? 4400 : 1450, dstep = P ? 330 : 250;
    rows.forEach((r, i) => {
      const c = chunksOf(r);
      this.dChunks.push(c.chunks); this.dIds.push(ids[i]!);
      this.gD.push({ chars: c.chars, x: X + (P ? 0 : 180), y: dy0 + i * dstep, cw: P ? 44 : 54, ch: P ? 64 : 76, breaks: [], lineGap: 0 });
    });
    // camera rectangles
    const Vall: Rect = [this.V.ox - 300, this.V.oy - 400, this.V.ox + f.cols * 28 + 300, this.V.oy + this.V.rows * 10 + 1250];
    this.R = P ? {
      intro: [-560, -520, 560, 560], Q: [-540, -170, 540, 700], Qs: [-520, -30, 540, 140], QM: [-520, -120, 520, 560],
      B: [-560, 1080, 560, 1880], QB: [-560, -170, 560, 1600], Vp: [vx - 250, vy - 150, vx + 250, vy + 150], Vall,
      S: [-520, 2050, 520, 2700], T: [-520, 2800, 520, 4050], D: [-520, 4120, 520, 5250],
      all: [-560, -260, 560, 5250], O: [-500, 1950, 500, 2750],
    } : {
      intro: [-900, -460, 900, 470], Q: [-910, -150, 910, 520], Qs: [-120, -120, 780, 110], QM: [-880, -160, 880, 500],
      B: [-940, 900, 940, 1420], QB: [-960, -170, 960, 1330], Vp: [vx - 230, vy - 85, vx + 230, vy + 85], Vall,
      S: [X - 760, -250, X + 760, 300], T: [X - 900, 470, X + 900, 1120], D: [X - 900, 1340, X + 900, 2140],
      all: [-960, -260, X + 900, 2150], O: [X - 600, -420, X + 1500, 260],
    };
  }

  // ---------------------------------------------------------------- the camera path
  private camera() {
    const A = this.A, C = this.C, R = this.R;
    const key = (t: number, r: Rect, d = 2.2) => this.keys.push({ t, d, r });
    key(0, R.intro!, 0.01);
    key(A.asked! - 1.2, R.Q!, 2.0);
    key(A.count! - 0.6, R.Qs!, 1.6);
    key(A.never1! - 0.2, R.Q!, 1.8);
    key(C[0]!.start! - 1.0, R.Q!, 1.0);
    key(C[0]!.end! + 0.6, R.QM!, 3.0);
    key(A.where! - 1.4, R.B!, 2.6);
    key(A.made! - 0.8, R.Vp!, 3.2);
    key(A.hundred! - 0.3, R.Vall!, 3.6);
    key(C[1]!.start! - 1.6, R.QB!, 3.2);
    key(C[1]!.end! + 0.4, R.B!, 2.4);
    key(A.look! - 1.2, R.S!, 2.6);
    key(A.give! - 0.8, R.T!, 2.2);
    key(A.hiding! - 0.6, R.S!, 2.0);
    key(A.wrote! - 1.0, R.D!, 2.4);
    key(C[2]!.start! - 1.4, R.Q!, 2.8);
    key(C[2]!.reads! - 0.2, R.all!, 4.5);
    key(A.what! - 0.8, R.O!, 3.0);
    // a long instrumental tail: one last slow pull back over the whole sheet (every chunk a number now)
    if (A.end! - A.last! > 12) key(A.last! + 4, R.all!, 9.0);
    // ...then settles on what the model got: the question as numbers
    if (A.end! - A.last! > 24) key(A.last! + 17, R.Q!, 9.0);
    this.keys.sort((a, b) => a.t - b.t);
  }

  private fit(r: Rect): Cam {
    const P = this.P;
    const mx = P ? 50 : 90, top = P ? 250 : 150, bot = P ? 150 : 90;
    const z = Math.min((W - 2 * mx) / (r[2] - r[0]), (H - top - bot) / (r[3] - r[1]));
    return { cx: (r[0] + r[2]) / 2, cy: (r[1] + r[3]) / 2, z };
  }

  /** Slow push-in and pan while a framing holds, so the picture never freezes. */
  private drift(c: Cam, dt: number): Cam {
    const s = Math.max(0, dt);
    return { cx: c.cx, cy: c.cy, z: c.z * (1 + Math.min(0.05, 0.0035 * s)) };
  }

  /**
   * The camera never stops: a slow float on a circle (~12 px/s on screen, one turn every 18 s) and a
   * breathing zoom, on top of the keyed framing. A pure function of song time, so it is continuous across moves.
   */
  private float(c: Cam, t: number): Cam {
    const R = this.P ? 48 : 60, w = (2 * Math.PI) / 16;
    return { cx: c.cx + (R / c.z) * Math.sin(w * t), cy: c.cy + (0.6 * R / c.z) * Math.cos(w * t), z: c.z * (1 + 0.035 * Math.sin((2 * Math.PI * t) / 20)) };
  }

  private cam(t: number): Cam { return this.float(this.camKeyed(t), t); }

  private camKeyed(t: number): Cam {
    const ks = this.keys;
    let cur = this.fit(ks[0]!.r), curEnd = ks[0]!.t + ks[0]!.d;
    for (let i = 1; i < ks.length; i++) {
      const kk = ks[i]!;
      if (t < kk.t) break;
      const from = this.drift(cur, kk.t - curEnd), to = this.fit(kk.r);
      if (t < kk.t + kk.d) {
        const u = ease.inOutCubic((t - kk.t) / kk.d);
        // far moves zoom out on the way (both ends in view at the midpoint)
        const span = Math.max(Math.abs(to.cx - from.cx) / (W / from.z), Math.abs(to.cy - from.cy) / (H / from.z),
          Math.abs(to.cx - from.cx) / (W / to.z), Math.abs(to.cy - from.cy) / (H / to.z));
        const zl = lerp(Math.log(from.z), Math.log(to.z), u);
        const bump = span > 0.9 ? Math.log(Math.min(from.z, to.z)) - Math.log(Math.min(from.z, to.z) / (span * 1.1)) : 0;
        // centre follows the zoom so a far move reads as a flight, not a smear
        const uc = ease.inOutCubic(clamp((t - kk.t) / kk.d * 1.0));
        return { cx: lerp(from.cx, to.cx, uc), cy: lerp(from.cy, to.cy, uc), z: Math.exp(zl - bump * Math.sin(Math.PI * u)) };
      }
      cur = to; curEnd = kk.t + kk.d;
    }
    return this.drift(cur, t - curEnd);
  }

  // ---------------------------------------------------------------- the vocabulary field (prerendered)
  private buildField() {
    const V = this.V, n = this.ctx.tok.llama.vocab_total, reg = this.ctx.tok.llama.vocab_regular;
    const px = 4, py = 2;
    const cv = document.createElement('canvas');
    cv.width = V.cols * px; cv.height = V.rows * py;
    const c = cv.getContext('2d')!;
    c.fillStyle = blueA(0.55);
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / V.cols), q = i % V.cols;
      if (i === reg) c.fillStyle = INK;
      c.fillRect(q * px, r * py, px - 1, py - 0.6);
    }
    this.field = cv;
  }

  // ---------------------------------------------------------------- render
  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, t = f.t;
    this.wash.render(r, out, { seed: 3, drift: t * 0.07, amt: 0.14, c1: 'pink', c2: 'blue' });
    const tex = perFrame(this.layer, f, () => this.draw(f.ft));
    this.ctx.comp.draw(r, tex, out);
    const A = this.A;
    return { reg: 7, kinetic: 0, fade: clamp((f.ft - (A.end! - 3.2)) / 2.4) };
  }

  private draw(t: number) {
    const c = this.layer.ctx, cam = this.cam(t);
    const P = this.P, top = P ? 250 : 150, bot = P ? 150 : 90;
    const sx = W / 2, sy = top + (H - top - bot) / 2;
    c.save();
    c.setTransform(cam.z, 0, 0, cam.z, sx - cam.cx * cam.z, sy - cam.cy * cam.z);
    const view: Rect = [cam.cx - sx / cam.z, cam.cy - sy / cam.z, cam.cx + (W - sx) / cam.z, cam.cy + (H - sy) / cam.z];
    const vis = (r: Rect) => !(r[2] < view[0] || r[0] > view[2] || r[3] < view[1] || r[1] > view[3]);
    this.drawField(c, t, cam, view);
    if (vis(this.R.QB!) || vis(this.R.intro!)) this.drawQ(c, t);
    if (vis(this.R.B!)) this.drawB(c, t);
    if (vis(this.R.S!) || vis(this.R.O!)) this.drawS(c, t);
    if (vis(this.R.T!)) this.drawT(c, t);
    if (vis(this.R.D!)) this.drawD(c, t);
    c.restore();
    this.sources(t);
  }

  /** The source line (HUD, bottom left) for what is on screen. */
  private sources(t: number) {
    const A = this.A, C = this.C, tok = this.ctx.tok;
    const st = tok.spell_test;
    const spans: [number, string][] = [
      [0, `split and numbers: the Llama 3.1 tokenizer (${tok.llama.tokenizer})\nits answer: Llama 3.1 8B Instruct, seed 7, as recorded for "Next Token"`],
      [A.where! - 1, `our own byte-pair encoding (BPE), ~60 lines of Python, trained on this song's lyrics\n${tok.bpe.corpus.words} words, ${tok.bpe.merges.length} merges until no pair repeats`],
      [A.made! - 0.5, `Llama 3.1's vocabulary in id order: ${tok.llama.vocab_regular.toLocaleString('en-US')} chunks of text + ${tok.llama.special} special tokens\none dot per token (${this.V.cols} per row)`],
      [C[1]!.start! - 1.5, `split and numbers: the Llama 3.1 tokenizer / our BPE on this song`],
      [A.look! - 1, `the Llama 3.1 tokenizer (${tok.llama.tokenizer})`],
      [A.give! - 0.5, `${st.model.split('/')[1]}, ${st.gpu.replace('NVIDIA GeForce ', '')}, bf16: each question asked ${st.n} times\n(temperature ${st.temperature}, top-p ${st.top_p}, seed ${st.seed}); the total each answer states`],
      [A.hiding! - 0.5, `the Llama 3.1 tokenizer (${tok.llama.tokenizer})`],
      [A.wrote! - 1, `"str aw berry": the model's own answer, recorded for "Next Token"; ids from the Llama 3.1 tokenizer`],
      [C[2]!.start! - 1, `split and numbers: the Llama 3.1 tokenizer / our BPE on this song`],
      [A.what! - 1, ''],
    ];
    let cur = '', since = -1e9;
    for (const [a, s] of spans) if (t >= a) { cur = s; since = a; }
    hudInfo.source = cur;
    hudInfo.sourceA = cur ? clamp((t - since) / 0.5) * (1 - clamp((t - (A.end! - 4)) / 1)) : 0;
    hudInfo.marker = 1 - clamp((t - (A.end! - 3)) / 1.5);
  }

  // ---------------------------------------------------------------- Q: the question
  private qState(t: number): RowState {
    const A = this.A, C = this.C, g = this.gQ, n = g.chars.length;
    const s = newState(n, this.qChunks);
    s.ids = this.qIds;
    const cw = g.cw, g0 = cw * 0.13, gc = cw * 0.5;
    const ty = (A.type1! - A.type0!) / n;
    for (let i = 0; i < n; i++) s.alpha[i] = k(t, A.type0! + i * ty, 0.3);
    // cut into Llama's chunks on "cut", each chunk fusing in turn
    const chunkOf: number[] = [];
    this.qChunks.forEach(([a, b], ci) => { for (let i = a; i <= b; i++) chunkOf[i] = ci; });
    for (let i = 0; i < n - 1; i++) {
      const same = chunkOf[i] === chunkOf[i + 1];
      const tc = A.cut! + chunkOf[i]! * 0.09;
      const fu = same ? k(t, tc, 0.6) : 0, sp = same ? 0 : k(t, tc, 0.6);
      s.fuse[i] = fu;
      s.gaps[i] = gapOf(g0, gc, fu, sp, cw);
    }
    // flips: to numbers on "swapped", back to letters on "shown" (the numbers leave as slips)
    this.qChunks.forEach((_, ci) => {
      s.hot[ci] = flash(t, A.cut! + ci * 0.09 + 0.2, 0.35);
      const on = A.swapped! + ci * 0.22, off = A.shown! + 0.2 + ci * 0.08;
      let fl = k(t, on, 0.55) * (1 - k(t, off, 0.55));
      // last chorus: every letter on the sheet turns into its number
      fl = Math.max(fl, k(t, C[2]!.never! + ci * 0.05, 0.55));
      s.flip[ci] = fl;
    });
    // "count them": strawberry's r's in pink
    const rs = [22, 27, 28].map((i) => i);
    const rOn = win(t, A.count! - 0.2, A.cut! - 0.3, 0.3);
    rs.forEach((i) => (s.pinkChar[i] = rOn > 0.5 ? 1 : 0));
    // chorus: "never the letters" dims the letters (the model never gets them)
    return s;
  }

  private drawQ(c: CanvasRenderingContext2D, t: number) {
    const A = this.A, C = this.C, g = this.gQ, P = this.P, M = this.M, tok = this.ctx.tok;
    const s = this.qState(t);
    const top = g.y - (g.breaks.length ? (g.ch * 2 + g.lineGap) / 2 : g.ch / 2);
    // title (the intro), then gone when the singing starts
    const ta = k(t, 0.3, 1.0) * (1 - k(t, A.asked! - 1.6, 1.0));
    if (ta > 0.003) {
      c.save(); c.globalAlpha = ta;
      c.fillStyle = INK; c.textBaseline = 'alphabetic';
      c.font = font(F.archivo(112, 900), P ? 150 : 150);
      const tx = P ? -480 : -820, ty = P ? -300 : -250;
      c.fillText('Tokens', tx, ty);
      c.font = font(F.archivo(100, 600), P ? 38 : 36);
      c.fillStyle = INK;
      c.fillText('why the model can’t see the r’s', tx + 6, ty + (P ? 64 : 58));
      c.restore();
    }
    // brackets over the chunks and the chorus labels
    for (let n = 0; n < 3; n++) {
      const cc = C[n]!, reset = C[n + 1]?.start ?? Infinity;
      const a = win(t, cc.start! - 0.6, n === 2 ? Infinity : reset - 1.0, 0.5);
      if (a <= 0.003) continue;
      c.save(); c.globalAlpha = a;
      const pk = t >= cc.token! - 0.1 && t < cc.number! - 0.2;
      this.qChunks.forEach((_, ci) => {
        const b = chunkBox(g, s, ci);
        bracket(c, t, cc.token! + ci * 0.07, b.x + 3, b.x + b.w - 3, b.y - 16, { color: pk ? PINK : INK, lw: 4, h: 12 });
      });
      const lx = P ? -500 : -800, ly = top - (P ? 58 : 56);
      label(c, t, cc.token!, 'token = a chunk of text', lx, ly, { size: P ? 42 : 38, color: pk ? PINK : INK });
      c.restore();
    }
    drawRow(c, g, s, { fs: g.ch * 0.6 });
    // "count them yourself": 1 2 3 under strawberry's r's
    const rA = win(t, A.count! - 0.1, A.cut! - 0.3, 0.3);
    if (rA > 0.003) {
      [22, 27, 28].forEach((i, j) => {
        const b = charBox(g, s, i);
        const at = [A.count!, A.yourself!, A.three1!][j]!;
        label(c, t, at, String(j + 1), b.cx, b.y + b.h + 34, { size: 40, align: 'center', color: PINK, a: rA, weight: 800 });
      });
    }
    // the model box, and the answer it gave (Next Token's run)
    const ma = k(t, A.type1! - 0.6, 0.8);
    const mhot = flash(t, A.model1!, 0.6) * 0.8;
    modelBox(c, M.x, M.y, M.w, M.h, ma, { sub: 'Llama 3.1 8B Instruct', hot: mhot, size: P ? 44 : 40 });
    const aa = win(t, A.ans!, A.before! - 0.2, 0.6);
    if (aa > 0.003) {
      c.save(); c.globalAlpha = aa;
      const fs = P ? 26 : 24, bx = P ? -450 : -520, by = M.y + M.h + (P ? 58 : 50);
      c.font = font(F.mono(500), fs); c.fillStyle = INK; c.textBaseline = 'middle';
      const pre = 'it answered: ';
      c.fillText(pre, bx, by);
      let x = bx + c.measureText(pre).width;
      const ans = tok.llama.answer; // "There are 2 r's in the word 'strawberry'."
      const i2 = ans.indexOf('2');
      c.font = font(F.mono(600), fs);
      c.fillText(ans.slice(0, i2), x, by); x += c.measureText(ans.slice(0, i2)).width;
      const hot2 = flash(t, A.two1!, 0.8);
      c.fillStyle = PINK; c.font = font(F.mono(700), fs * (1 + 0.35 * hot2));
      c.fillText('2', x, by); x += c.measureText('2').width;
      c.fillStyle = INK; c.font = font(F.mono(600), fs);
      c.fillText(ans.slice(i2 + 1), x, by);
      c.restore();
    }
    // "it never saw the letters": a crossed-out arrow from the letters to the model
    const na = win(t, A.never1! - 0.1, A.cut! - 0.2, 0.35);
    if (na > 0.003) {
      const x0 = P ? 0 : -300, y0 = top + (g.breaks.length ? g.ch * 2 + g.lineGap : g.ch) + 16, y1 = M.y - 10;
      drawLine(c, t, A.never1! - 0.1, x0, y0, x0, y1, { color: inkA(0.8), lw: 4, head: 20, dash: [12, 10], a: na, d: 0.5 });
      const my = (y0 + y1) / 2, xs = 22;
      c.save(); c.globalAlpha = na * k(t, A.letters1!, 0.35); c.strokeStyle = PINK; c.lineWidth = 7; c.lineCap = 'round';
      c.beginPath(); c.moveTo(x0 - xs, my - xs); c.lineTo(x0 + xs, my + xs); c.moveTo(x0 + xs, my - xs); c.lineTo(x0 - xs, my + xs); c.stroke();
      c.restore();
      label(c, t, A.letters1!, 'it never got the letters', x0 + 40, my, { size: P ? 34 : 30, a: na });
    }
    // "eight little chunks, eight numbers": a count under the row
    const ea = win(t, A.eight! - 0.1, A.shown! + 0.5, 0.35);
    if (ea > 0.003) {
      // a brace under the whole row
      const b0 = chunkBox(g, s, 0), b7 = chunkBox(g, s, this.qChunks.length - 1);
      const by = b7.y + b7.h + 16;
      if (!P) bracket(c, t, A.eight!, b0.x, b7.x + b7.w, by, { color: INK, lw: 3, h: -12, d: 0.8 });
      label(c, t, A.row!, `${this.qChunks.length} chunks, ${this.qChunks.length} numbers`, P ? -500 : -800, top + (g.breaks.length ? g.ch * 2 + g.lineGap : g.ch) + 80,
        { size: P ? 36 : 32, a: ea, color: INK });
    }
    // number slips: V1 "the numbers are all that it's shown" (row -> model), and every chorus
    const slipsTo = (t0: number, a: number, under: number) => {
      // under: 0 = at the chunks (just below), 1 = inside the model box
      this.qChunks.forEach((_, ci) => {
        const b = chunkBox(g, s, ci);
        const sw = P ? 96 : 104, sh = P ? 54 : 50, gap = P ? 8 : 10;
        const inX = M.x + M.w / 2 - (8 * sw + 7 * gap) / 2 + ci * (sw + gap) + sw / 2, inY = M.y + M.h - sh / 2 - 26;
        const u = ease.inOutCubic(clamp((under - ci * 0.04) / 0.72));
        const x = lerp(b.cx, inX, u), y = lerp(b.y + b.h + 44, inY, u);
        slip(c, x, y, sw, sh, String(this.qIds[ci]), a * k(t, t0 + ci * 0.06, 0.4), { hot: 0 });
      });
    };
    // V1: from "shown" into the model, emptied before the chorus
    {
      const a = win(t, A.shown!, C[0]!.start! - 0.9, 0.4);
      if (a > 0.003) slipsTo(A.shown!, a, clamp((t - A.shown! - 0.4) / 1.4));
    }
    for (let n = 0; n < 3; n++) {
      const cc = C[n]!, reset = C[n + 1]?.start ?? Infinity;
      const a = win(t, cc.number! - 0.1, (n === 0 ? Math.min(reset, this.A.where!) : reset) - 1.2, 0.45) * (n === 2 ? 1 - k(t, cc.never!, 0.6) : 1);
      if (a <= 0.003) continue;
      slipsTo(cc.number!, a, clamp((t - cc.reads! + 0.1) / 1.4));
      label(c, t, cc.number!, 'every chunk gets a number', 0, M.y + M.h + (P ? 44 : 40), { size: P ? 34 : 30, align: 'center', a, color: t < cc.reads! ? BLUE : INK });
    }
    // chorus: "the model reads the numbers" / "never the letters" labels on the model box and the row
    for (let n = 0; n < 3; n++) {
      const cc = C[n]!, reset = C[n + 1]?.start ?? Infinity;
      const a = win(t, cc.reads! - 0.1, (n === 2 ? Infinity : reset) - 1.2, 0.45);
      if (a <= 0.003) continue;
      label(c, t, cc.reads!, 'reads the numbers', M.x + M.w - 26, M.y + (P ? 52 : 48), { size: P ? 34 : 32, align: 'right', a, color: t < cc.never! ? PINK : INK });
      const la = a * k(t, cc.never!, 0.5);
      // a pink line struck through the letters, left to right
      if (la > 0.003 && s.flip.every((x) => x < 0.5)) {
        const pos = this.qChunks.map((_, ci) => chunkBox(g, s, ci));
        const rows = new Map<number, [number, number]>();
        for (const b of pos) { const r = rows.get(b.y); rows.set(b.y, r ? [Math.min(r[0], b.x), Math.max(r[1], b.x + b.w)] : [b.x, b.x + b.w]); }
        let i = 0;
        for (const [y, [x0, x1]] of rows) {
          drawLine(c, t, cc.never! + i * 0.35, x0 - 10, y + g.ch * 0.5, x1 + 10, y + g.ch * 0.5, { color: PINK, lw: 7, d: 0.7, a: la });
          i++;
        }
      }
      if (la > 0.003) {
        const rx = P ? 500 : 800;
        label(c, t, cc.never!, 'never the letters', rx, top - (P ? 58 : 56), { size: P ? 42 : 38, align: 'right', a: la, color: PINK });
      }
    }
    // after chorus 1: the numbers inside the model, read one by one (no new text)
    const ia = win(t, C[0]!.end! + 0.4, this.A.where! - 1.2, 0.5);
    if (ia > 0.003) {
      this.qChunks.forEach((_, ci) => {
        const sw = P ? 96 : 104, sh = P ? 54 : 50, gap = P ? 8 : 10;
        const x = M.x + M.w / 2 - (8 * sw + 7 * gap) / 2 + ci * (sw + gap) + sw / 2, y = M.y + M.h - sh / 2 - 26;
        const hot = flash(t, C[0]!.end! + 1.2 + ci * 0.6, 0.4);
        if (hot > 0.01) { c.save(); c.globalAlpha = ia * hot; c.fillStyle = pinkA(0.5); rrect(c, x - sw / 2 - 6, y - sh / 2 - 6, sw + 12, sh + 12, 10); c.fill(); c.restore(); }
      });
    }
  }

  // ---------------------------------------------------------------- B: our BPE on a line of this song
  private bSeg(kk: number): string[][] {
    const steps = this.ctx.tok.bpe.line.steps;
    let seg = steps[0]!.words;
    for (const st of steps) if (st.k <= kk) seg = st.words;
    return seg;
  }

  private drawB(c: CanvasRenderingContext2D, t: number) {
    const A = this.A, C = this.C, g = this.gB, P = this.P, tok = this.ctx.tok;
    const n = g.chars.length, mt = this.merges;
    // merge index in progress: the latest merge that has started
    let kk = 0;
    while (kk + 1 < mt.length && mt[kk + 1]! <= t) kk++;
    const seg = this.bSeg(kk);
    // chunks as char ranges, and when each boundary fused (for the animation)
    const chunks: [number, number][] = [];
    const wordOf: number[] = [];
    let ci = 0;
    seg.forEach((wd, wi) => { for (const piece of wd) { chunks.push([ci, ci + piece.length - 1]); ci += piece.length; } for (let j = 0; j < wd.join('').length; j++) wordOf.push(wi); });
    const s = newState(n, chunks);
    const appear = k(t, A.where! - 0.4, 0.8);
    s.a = appear;
    const brk = k(t, A.break!, 0.8); // "break it into letters": the words fall apart into single letters
    const cw = g.cw, g0 = cw * 0.14 * brk, gw = cw * 0.9;
    // fuse time per boundary: the merge whose result first joins chars i and i+1
    const fuseAt = this.bFuseTimes();
    for (let i = 0; i < n - 1; i++) {
      if (wordOf[i] !== wordOf[i + 1]) { s.gaps[i] = gw; s.fuse[i] = 0; continue; }
      const tf = fuseAt[i]!;
      const fu = Number.isFinite(tf) ? k(t, tf, 0.35) : 0;
      // before "break", whole words print as plain text (fused); then letters, then merges
      s.fuse[i] = brk < 1 ? Math.max(1 - brk, fu) : fu;
      s.gaps[i] = g0 * (1 - s.fuse[i]!) - TIGHT * cw * s.fuse[i]!;
    }
    // chunk flash when formed; flips to our BPE ids in chorus 2 and at the end of the instrumental
    chunks.forEach(([a, b], j) => {
      let tf = -Infinity;
      for (let i = a; i < b; i++) tf = Math.max(tf, fuseAt[i]!);
      s.hot[j] = b > a ? flash(t, tf, 0.3) : 0;
      const piece = g.chars.slice(a, b + 1).join('');
      s.ids[j] = tok.bpe.ids[piece] ?? '';
      const c2 = C[1]!;
      const f2 = k(t, c2.number! + j * 0.05, 0.5) * (1 - k(t, c2.end! + 0.3 + j * 0.02, 0.5));
      const fin = k(t, this.merges[tok.bpe.line.steps[tok.bpe.line.steps.length - 1]!.k]! + 1.0 + j * 0.12, 0.5) * (1 - k(t, A.look! - 1.4, 0.5));
      const f3 = k(t, C[2]!.never! + 0.4 + j * 0.05, 0.55);
      s.flip[j] = Math.max(f2, fin, f3);
    });
    // "find the pair": every e-r pair in the line glows pink until it is glued
    const pairOn = win(t, A.most! - 0.1, A.glue1! + 0.4, 0.3);
    if (kk === 0 && pairOn > 0) for (let i = 0; i < n - 1; i++) if (g.chars[i] === 'e' && g.chars[i + 1] === 'r' && wordOf[i] === wordOf[i + 1]) { s.pinkChar[i] = 1; s.pinkChar[i + 1] = 1; }
    // plain words first (no tiles) then tiles: tiles fade in with "break"
    c.save();
    const box = this.R.B!;
    if (appear > 0.003) {
      // label: where the line comes from
      const top = g.y - (g.breaks.length ? (g.ch * (g.breaks.length + 1) + g.lineGap * g.breaks.length) / 2 : g.ch / 2);
      label(c, t, A.where! + 0.2, 'a line from this song', P ? -460 : -900, top - (P ? 50 : 46), { size: P ? 34 : 30, a: appear });
    }
    drawRow(c, g, s, { fs: g.ch * 0.62 });
    // the merge panel under the line: this merge, and how often the pair occurs in the whole song
    const panelA = win(t, A.most! - 0.2, C[1]!.start! - 1.2, 0.4) * appear;
    if (panelA > 0.003) {
      const bottom = g.y + (g.breaks.length ? (g.ch * (g.breaks.length + 1) + g.lineGap * g.breaks.length) / 2 : g.ch / 2);
      const py = bottom + (P ? 110 : 100), px = P ? -460 : -900;
      const m = kk > 0 ? tok.bpe.merges[kk - 1]! : tok.bpe.merges[0]!;
      c.save(); c.globalAlpha = panelA;
      c.font = font(F.mono(700), P ? 40 : 36); c.textBaseline = 'middle'; c.fillStyle = INK;
      const head = kk > 0 ? `merge ${kk}:` : 'most frequent pair:';
      c.fillText(head, px, py);
      let x = px + c.measureText(head).width + 24;
      const tile = (txt: string, col: string) => {
        c.font = font(F.mono(700), P ? 40 : 36);
        const w = c.measureText(txt).width + 26;
        c.fillStyle = PAPER; rrect(c, x, py - 30, w, 60, 8); c.fill();
        c.strokeStyle = col; c.lineWidth = 3; rrect(c, x, py - 30, w, 60, 8); c.stroke();
        c.fillStyle = col; c.fillText(txt, x + 13, py + 2);
        x += w + 16;
      };
      tile(m.a, INK); c.fillStyle = INK; c.fillText('+', x, py); x += 36; tile(m.b, INK);
      c.fillStyle = INK; c.fillText('→', x, py); x += 50; tile(m.ab, PINK);
      c.font = font(F.mono(600), P ? 32 : 30); c.fillStyle = BLUE;
      const cnt = `${m.count} times in the song`;
      if (P) c.fillText(cnt, px, py + 70); else c.fillText(cnt, x + 10, py);
      c.restore();
    }
    // chorus 2: our tokenizer's numbers under the words
    const la = win(t, C[1]!.number!, C[1]!.end! + 0.3, 0.45);
    if (la > 0.003) {
      const bottom = g.y + (g.breaks.length ? (g.ch * (g.breaks.length + 1) + g.lineGap * g.breaks.length) / 2 : g.ch / 2);
      label(c, t, C[1]!.number!, `our tokenizer's numbers (it knows ${tok.bpe.vocab} chunks)`, P ? -460 : -900, bottom + 70, { size: P ? 32 : 30, a: la, color: BLUE });
    }
    c.restore();
    void box;
  }

  private _bFuse?: number[];
  /** For each boundary of the B line: the time of the merge that first joins its two letters. */
  private bFuseTimes(): number[] {
    if (this._bFuse) return this._bFuse;
    const steps = this.ctx.tok.bpe.line.steps, n = this.gB.chars.length;
    const out = new Array(n - 1).fill(Infinity);
    const bounds = (seg: string[][]) => {
      const set = new Set<number>();
      let ci = 0;
      for (const wd of seg) { for (const piece of wd) { for (let j = 0; j < piece.length - 1; j++) set.add(ci + j); ci += piece.length; } }
      return set;
    };
    for (const st of steps) {
      const b = bounds(st.words);
      for (const i of b) if (!Number.isFinite(out[i])) out[i] = this.merges[st.k] ?? Infinity;
    }
    this._bFuse = out;
    return out;
  }

  // ---------------------------------------------------------------- V: Llama's vocabulary
  private drawField(c: CanvasRenderingContext2D, t: number, cam: Cam, view: Rect) {
    const A = this.A, V = this.V, tok = this.ctx.tok;
    const a = win(t, A.made! - 1.0, this.C[1]!.start! - 0.4, 0.9);
    if (a <= 0.003) return;
    const fw = V.cols * V.cw, fh = V.rows * V.chh;
    if (view[2] < V.ox || view[0] > V.ox + fw || view[3] < V.oy || view[1] > V.oy + fh) return;
    c.save();
    c.globalAlpha = a;
    const cellPx = V.cw * cam.z;
    // far: the prerendered dot field; near: live cells, then their text. Each hands over by crossfading across
    // a range of zoom (a switch within one frame reads as a cut); densities matched (~0.29 blue coverage).
    const imgW = 1 - smoothstep(12, 18, cellPx);
    if (imgW > 0.003 && this.field) {
      c.globalAlpha = a * imgW;
      c.imageSmoothingEnabled = true;
      c.drawImage(this.field, V.ox, V.oy, fw, fh);
    }
    if (cellPx >= 12) {
      c.globalAlpha = a * (1 - imgW);
      const q0 = Math.max(0, Math.floor((view[0] - V.ox) / V.cw)), q1 = Math.min(V.cols - 1, Math.ceil((view[2] - V.ox) / V.cw));
      const r0 = Math.max(0, Math.floor((view[1] - V.oy) / V.chh)), r1 = Math.min(V.rows - 1, Math.ceil((view[3] - V.oy) / V.chh));
      const patch = tok.llama.field.patch;
      const textA = smoothstep(36, 52, cellPx), fillA = lerp(0.34, 0.12, smoothstep(30, 52, cellPx));
      for (let r = r0; r <= r1; r++) {
        for (let q = q0; q <= q1; q++) {
          const id = r * V.cols + q;
          if (id >= tok.llama.vocab_total) continue;
          const x = V.ox + q * V.cw, y = V.oy + r * V.chh;
          const straw = id === 73700;
          c.fillStyle = straw ? pinkA(0.85) : id >= tok.llama.vocab_regular ? inkA(0.5) : blueA(fillA);
          c.fillRect(x + 0.6, y + 0.6, V.cw - 1.2, V.chh - 1.2);
          if (textA > 0.003) {
            const s = patch[String(id)];
            if (s !== undefined) {
              const lab = s.replace(/^ /, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥');
              let fs = 4.6;
              c.font = font(F.mono(straw ? 700 : 500), fs);
              const tw = c.measureText(lab).width;
              if (tw > V.cw - 2.4) { fs *= (V.cw - 2.4) / tw; c.font = font(F.mono(straw ? 700 : 500), fs); }
              c.save(); c.globalAlpha *= textA;
              c.fillStyle = INK;
              c.textBaseline = 'middle';
              c.fillText(lab, x + 1.2, y + V.chh / 2 + 0.2);
              c.restore();
            }
          }
        }
      }
    }
    c.globalAlpha = a;
    // labels: the id of " strawberry" while the patch is in view, the whole count when zoomed out
    const sx = V.sx, sy = V.sy;
    const pa = win(t, A.made! + 0.8, A.hundred! + 0.2, 0.5);
    if (pa > 0.003) {
      c.save();
      c.globalAlpha = a * pa;
      c.strokeStyle = PINK; c.lineWidth = 1.4;
      rrect(c, sx - V.cw / 2 - 2, sy - V.chh / 2 - 2, V.cw + 4, V.chh + 4, 2); c.stroke();
      c.font = font(F.mono(700), 6); c.textBaseline = 'middle';
      const lab = 'id 73700', lw = c.measureText(lab).width;
      c.fillStyle = PAPER; rrect(c, sx - lw / 2 - 3, sy - V.chh * 1.9 - 4, lw + 6, 8, 1.5); c.fill();
      c.fillStyle = BLUE; c.textAlign = 'center';
      c.fillText(lab, sx, sy - V.chh * 1.9);
      c.restore();
    }
    const za = k(t, A.hundred! + 0.6, 0.8) * a;
    // zoomed out: a pink ring shows where " strawberry" sits among all of them
    const ra = k(t, A.hundred!, 0.8) * a * clamp((12 - V.cw * cam.z) / 6);
    if (ra > 0.003) {
      c.save(); c.globalAlpha = ra; c.strokeStyle = PINK; c.lineWidth = 5 / cam.z;
      c.beginPath(); c.arc(sx, sy, 34 / cam.z, 0, Math.PI * 2); c.stroke(); c.restore();
    }
    if (za > 0.003) {
      const fs = fw * 0.06;
      c.save(); c.globalAlpha = za;
      c.fillStyle = PAPER;
      c.font = font(F.archivo(112, 900), fs);
      const txt = tok.llama.vocab_total.toLocaleString('en-US');
      const tw = c.measureText(txt).width;
      const cx = V.ox + fw / 2, cy = V.oy + fh + fs * 0.9;
      rrect(c, cx - tw / 2 - fs * 0.3, cy - fs * 0.8, tw + fs * 0.6, fs * 1.25, fs * 0.1); c.fill();
      c.fillStyle = BLUE; c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(txt, cx, cy);
      c.font = font(F.archivo(100, 700), fs * 0.3); c.fillStyle = INK;
      c.fillText('chunks in the model’s vocabulary', cx, cy + fs * 0.62);
      c.restore();
    }
    c.restore();
  }

  // ---------------------------------------------------------------- S: " strawberry", one chunk, one number
  private sState(t: number): RowState {
    const A = this.A, C = this.C, g = this.gS, n = g.chars.length;
    const s = newState(n, [[0, n - 1]]);
    s.ids = [''];
    s.a = k(t, A.look! - 0.8, 0.8);
    const cw = g.cw, g0 = cw * 0.13;
    const fu = k(t, A.oneChunk!, 0.7);
    for (let i = 0; i < n - 1; i++) { s.fuse[i] = fu; s.gaps[i] = g0 * (1 - fu) - TIGHT * cw * fu; }
    s.hot[0] = flash(t, A.oneChunk! + 0.3, 0.4);
    s.pinkChar[0] = win(t, A.space! - 0.1, A.oneChunk! + 0.6, 0.2) > 0.5 ? 1 : 0;
    // digits arrive with "seven three seven oh oh"
    const digs = '73700';
    let shown = '';
    for (let i = 0; i < 5; i++) if (t >= A[`dig${i}`]! - 0.05) shown += digs[i];
    const flipOn = k(t, A.oneNumber! - 0.35, 1.2);
    // "the r's were hiding in the chunk": back to letters, r's in pink; "it only ever saw the number": flip again
    const back = k(t, A.hiding! - 0.35, 1.2) * (1 - k(t, A.onlyNum! - 0.45, 1.2));
    s.flip[0] = Math.max(0, flipOn - back);
    s.ids[0] = t < A.hiding! ? shown : digs;
    const rOn = win(t, A.hiding!, A.onlyNum! - 0.1, 0.3);
    [3, 8, 9].forEach((i) => (s.pinkChar[i] = rOn > 0.5 ? 1 : 0));
    void C;
    return s;
  }

  private drawS(c: CanvasRenderingContext2D, t: number) {
    const A = this.A, g = this.gS, P = this.P;
    const s = this.sState(t);
    if (s.a <= 0.003) return;
    drawRow(c, g, s, { fs: g.ch * 0.58, numFs: g.ch * 0.66, lw: 4 });
    const b = chunkBox(g, s, 0);
    // "with a space in front"
    const spA = win(t, A.space! - 0.1, A.oneNumber! - 0.2, 0.35) * s.a;
    if (spA > 0.003) {
      const cb = charBox(g, s, 0);
      drawLine(c, t, A.space!, cb.cx, cb.y - 70, cb.cx, cb.y - 12, { color: PINK, lw: 4, head: 16, a: spA });
      label(c, t, A.space!, 'a space', cb.cx - 10, cb.y - 96, { size: 34, a: spA, color: PINK });
      label(c, t, A.oneChunk!, `= 1 chunk`, cb.cx + 180, cb.y - 96, { size: 34, a: spA * k(t, A.oneChunk!, 0.4) });
    }
    // "where are the r's in that?": ghost letters float above the number, r's in pink, then fade
    const gA = win(t, A.whereR! - 0.1, A.give! - 0.4, 0.4) * s.a;
    if (gA > 0.003) {
      c.save(); c.globalAlpha = gA;
      c.font = font(F.mono(600), 44); c.textAlign = 'center'; c.textBaseline = 'middle';
      const word = 'strawberry';
      for (let i = 0; i < word.length; i++) {
        const x = b.x + (b.w / word.length) * (i + 0.5), y = b.y - 60 - 10 * Math.sin(t * 1.3 + i);
        c.fillStyle = 'rs'.includes(word[i]!) && word[i] === 'r' ? PINK : inkA(0.35);
        c.fillText(word[i]!, x, y);
      }
      c.restore();
      label(c, t, A.rsIn!, 'no r in 73700', b.x + b.w + 30, b.y + b.h / 2, { size: 36, a: gA, color: INK });
    }
    // "it only ever saw the number"
    const oA = win(t, A.onlyNum! - 0.1, A.wrote! - 0.8, 0.4) * s.a;
    if (oA > 0.003) label(c, t, A.onlyNum!, 'what the model got', b.cx, b.y + b.h + 50, { size: 34, align: 'center', a: oA, color: BLUE });
    // outro: what does a number mean? an arrow out of the number (next episode: meaning is a direction)
    const oa = k(t, A.next! - 0.2, 0.6);
    if (oa > 0.003) {
      const x0 = P ? b.cx + 120 : b.x + b.w + 20, y0 = P ? b.y - 16 : b.cy;
      drawLine(c, t, A.next! - 0.2, x0, y0, x0 + (P ? 200 : 420), y0 - (P ? 300 : 260), { color: PINK, lw: 9, head: 42, d: 1.4 });
      label(c, t, A.direction! - 0.2, 'next: meaning is a direction', P ? b.x : x0 + 40, P ? b.y - 420 : y0 - 330, { size: P ? 44 : 44, weight: 800, a: oa });
      label(c, t, A.direction! + 0.6, 'ML, slowly #3', P ? b.x : x0 + 40, P ? b.y - 370 : y0 - 280, { size: P ? 32 : 32, mono: true, a: oa, color: BLUE });
    }
    const qa = win(t, A.what! - 0.2, A.end!, 0.5);
    if (qa > 0.003) label(c, t, A.what!, 'what does a number mean?', b.cx, b.y + b.h + 60, { size: 38, align: 'center', a: qa * (1 - k(t, A.next!, 0.5)) });
  }

  // ---------------------------------------------------------------- T: the test
  private drawT(c: CanvasRenderingContext2D, t: number) {
    const A = this.A, P = this.P, st = this.ctx.tok.spell_test;
    const a = Math.max(k(t, A.give! - 1.0, 0.8) * (1 - k(t, A.wrote! + 2.5, 0.8)), k(t, this.C[2]!.start!, 1.0));
    if (a <= 0.003) return;
    const rows = [
      { y: this.T.y1, t0: A.give!, said: st.word.said, ids: [73700], lab: 'as one chunk' },
      { y: this.T.y2, t0: A.spaceOut!, said: st.spaced.said, ids: st.spaced.ids, lab: 'letters spaced out' },
    ];
    c.save(); c.globalAlpha = a;
    rows.forEach((r, ri) => {
      const ra = k(t, r.t0 - 0.3, 0.6);
      if (ra <= 0.003) return;
      c.save(); c.globalAlpha = a * ra;
      const x0 = this.T.x + (P ? -500 : -880);
      // the question as the model gets it: the words before, then the word's numbers
      c.font = font(F.mono(500), P ? 30 : 28); c.fillStyle = inkA(0.8); c.textBaseline = 'middle';
      c.fillText('How many r’s are in', x0, r.y - (P ? 70 : 60));
      label(c, t, r.t0 - 0.2, r.lab, x0, r.y - (P ? 115 : 102), { size: P ? 34 : 32, weight: 800 });
      const n = r.ids.length, sw = n === 1 ? 150 : P ? 88 : 62, sh = P ? 60 : 52, gap = P ? 8 : 8;
      const perRow = P ? 5 : 10;
      r.ids.forEach((id, i) => {
        const rr = Math.floor(i / perRow), q = i % perRow;
        const x = x0 + sw / 2 + q * (sw + gap), y = r.y + rr * (sh + gap);
        slip(c, x, y, sw, sh, String(id), k(t, r.t0 + i * 0.07, 0.4));
      });
      const endX = x0 + Math.max(Math.min(n, perRow) * (sw + gap), P ? 0 : 400);
      // -> the model -> 100 answers
      const slipRows = Math.ceil(n / perRow);
      const mx = P ? x0 : endX + 90, my = P ? r.y + slipRows * (sh + gap) + 20 : r.y - 50;
      const tx = P ? x0 + 240 : mx + 260, ty = P ? my : my - 60;
      if (!P) drawLine(c, t, r.t0 + 0.4, endX + 10, r.y, mx - 10, r.y, { lw: 4, head: 18, d: 0.4 });
      modelBox(c, mx, my, 200, 100, k(t, r.t0 + 0.3, 0.5), { title: 'model', size: 30 });
      const tp = r.t0 + 0.7;
      const cols: string[] = [];
      const said = r.said as Record<string, number>;
      for (let i = 0; i < (said['2'] ?? 0); i++) cols.push(INK);
      for (let i = 0; i < (said['3'] ?? 0); i++) cols.push(PINK);
      for (let i = 0; i < (said['none'] ?? 0) + (said['other'] ?? 0); i++) cols.push(inkA(0.18));
      // keep the answers in the order they came? (ours: grouped, so the count reads at a glance)
      const cell = P ? 20 : 18;
      tally(c, t, tp, tx, ty, cell, ri === 0 ? cols : [...cols.filter((x) => x === PINK), ...cols.filter((x) => x === INK), ...cols.filter((x) => x !== PINK && x !== INK)], { d: 1.4 });
      const main = ri === 0 ? said['2']! : said['3']!;
      const lx = tx + 10 * cell * 1.22 + 26;
      // one number at a time: the first result steps back when the second arrives
      const back = ri === 0 ? 1 - 0.6 * k(t, rows[1]!.t0 + 1.9, 0.6) * (1 - k(t, this.C[2]!.start!, 0.8)) : 1;
      label(c, t, tp + 1.2, `${main} of ${st.n}`, lx, ty + cell * 3, { size: P ? 44 : 48, weight: 900, color: ri === 0 ? INK : PINK, a: back });
      label(c, t, tp + 1.5, ri === 0 ? 'said 2' : 'said 3', lx, ty + cell * 3 + (P ? 50 : 52), { size: P ? 32 : 34, weight: 700, a: back });
      c.restore();
    });
    c.restore();
  }

  // ---------------------------------------------------------------- D: same letters, different numbers
  private drawD(c: CanvasRenderingContext2D, t: number) {
    const A = this.A, C = this.C, P = this.P;
    const labs = ['with a space in front', 'no space (its own answer)', 'a capital S'];
    const starts = [A.wrote! - 0.6, A.noSpace!, A.capital! - 0.2];
    const flipAt = [A.wrote! + 0.3, A.str!, A.another!];
    this.gD.forEach((g, i) => {
      const n = g.chars.length;
      const s = newState(n, this.dChunks[i]!);
      s.ids = this.dIds[i]!;
      s.a = k(t, starts[i]!, 0.7);
      if (s.a <= 0.003) return;
      const chunkOf: number[] = [];
      this.dChunks[i]!.forEach(([a, b], ci) => { for (let j = a; j <= b; j++) chunkOf[j] = ci; });
      for (let j = 0; j < n - 1; j++) {
        const same = chunkOf[j] === chunkOf[j + 1];
        s.fuse[j] = same ? 1 : 0;
        s.gaps[j] = same ? -TIGHT * g.cw : g.cw * 0.45;
      }
      // row 2: str | aw | berry light up as they are sung, then everything flips
      const pieceT = [A.str!, A.aw!, A.berry!];
      this.dChunks[i]!.forEach((_, ci) => {
        let on = flipAt[i]!;
        if (i === 1) on = ci >= 2 && ci <= 4 ? pieceT[ci - 2]! + 0.2 : A.berry! + 0.6;
        s.flip[ci] = Math.max(k(t, on, 0.55) * (1 - k(t, C[2]!.start! - 1.0, 0.5)), k(t, C[2]!.never! + 0.8 + ci * 0.05, 0.55));
        if (i === 1 && ci >= 2 && ci <= 4) s.hot[ci] = flash(t, pieceT[ci - 2]!, 0.5);
      });
      const tx = this.X + (P ? -500 : -880);
      label(c, t, starts[i]!, labs[i]!, tx, g.y - g.ch / 2 - (P ? 40 : 0) + (P ? 0 : g.ch / 2), { size: P ? 32 : 32, weight: 700, a: s.a });
      drawRow(c, g, s, { fs: g.ch * 0.58 });
    });
    const sa = k(t, A.same! - 0.1, 0.6);
    if (sa > 0.003) {
      const g0 = this.gD[0]!, g2 = this.gD[2]!;
      label(c, t, A.same!, 'same letters,', this.X + (P ? -500 : -880), g2.y + g2.ch / 2 + 80, { size: P ? 44 : 46, weight: 900 });
      label(c, t, A.different!, 'different numbers', this.X + (P ? -500 + 330 : -880 + 340), g2.y + g2.ch / 2 + 80, { size: P ? 44 : 46, weight: 900, color: BLUE });
      void g0;
    }
  }
}
