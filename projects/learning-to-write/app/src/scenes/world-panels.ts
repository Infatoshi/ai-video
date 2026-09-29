// The pieces of the world, each drawn in world coordinates (the camera transform is already set) as a pure
// function of song time. Colours: pink = what is being taught while it is sung; blue = the data (Shakespeare,
// the prompt, the weights' values); ink = structure and type. Tints print as halftone; small type is solid.
import { rgba } from '../engine/palette';
import { W, H } from '../engine/gl';
import { F, font } from '../engine/type';
import { clamp, ease, hash, lerp } from '../engine/util';
import type { Run, ProbeStep } from '../engine/run';
import { showChar } from '../engine/run';
import type { Rect } from './world-layout';

export const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
export const tint = (k: 'pink' | 'blue' | 'ink', a: number) => rgba(k, a);
export const fmt = (n: number) => Math.round(n).toLocaleString('en-US');

/** Numbers fade out in wide shots (one number on screen at a time): world.ts sets this from the zoom each frame. */
export const NUM = { a: 1, z: 1 };

/** 0..1 draw-in progress of something that starts at t0 and takes dur (0 before; eased). */
export const inP = (t: number, t0: number, dur = 0.6) => ease.outCubic(clamp((t - t0) / dur));

/** Text written in left to right: each character fades in over `fade` s, `cps` characters per second. */
export function writeText(c: CanvasRenderingContext2D, s: string, x: number, y: number, t: number, t0: number, o: { cps?: number; fade?: number; color?: string } = {}) {
  if (t < t0) return;
  const cps = o.cps ?? 28, fade = o.fade ?? 0.3;
  const n = s.length;
  const done = t - t0 > n / cps + fade;
  if (done) { if (o.color) c.fillStyle = o.color; c.fillText(s, x, y); return; }
  const a0 = c.globalAlpha;
  const align = c.textAlign;
  const total = c.measureText(s).width;
  let xx = align === 'center' ? x - total / 2 : align === 'right' ? x - total : x;
  c.textAlign = 'left';
  if (o.color) c.fillStyle = o.color;
  for (let i = 0; i < n; i++) {
    const a = clamp((t - t0 - i / cps) / fade);
    const w = c.measureText(s.slice(0, i + 1)).width - c.measureText(s.slice(0, i)).width;
    if (a > 0) { c.globalAlpha = a0 * a; c.fillText(s[i]!, xx, y); }
    xx += w;
  }
  c.globalAlpha = a0;
  c.textAlign = align;
}

/** A label: small caps mono over a big number or word, both written in. */
export function label(c: CanvasRenderingContext2D, x: number, y: number, small: string, big: string, t: number, t0: number, o: { size?: number; color?: string; align?: CanvasTextAlign; cps?: number } = {}) {
  if (t < t0) return;
  const size = o.size ?? 64;
  c.save();
  c.textAlign = o.align ?? 'left';
  c.textBaseline = 'alphabetic';
  c.font = font(F.mono(600), size * 0.3);
  c.letterSpacing = '2px';
  writeText(c, small, x, y - size * 0.95, t, t0, { color: INK, cps: o.cps ?? 40 });
  c.letterSpacing = '0px';
  c.font = font(F.archivo(100, 900), size);
  if (/\d/.test(big)) c.globalAlpha *= NUM.a;
  writeText(c, big, x, y, t, t0 + 0.15, { color: o.color ?? INK, cps: o.cps ?? 30 });
  c.restore();
}

/** A sheet of paper: ink outline, a blue plate printed off-register behind it. */
export function sheet(c: CanvasRenderingContext2D, r: Rect, a = 1) {
  if (a <= 0) return;
  c.save();
  c.globalAlpha *= a;
  c.fillStyle = tint('blue', 0.28);
  c.fillRect(r.x - r.w / 2 + 14, r.y - r.h / 2 + 14, r.w, r.h);
  c.fillStyle = PAPER;
  c.fillRect(r.x - r.w / 2, r.y - r.h / 2, r.w, r.h);
  c.strokeStyle = INK;
  c.lineWidth = 2.5;
  c.strokeRect(r.x - r.w / 2, r.y - r.h / 2, r.w, r.h);
  c.restore();
}

// ------------------------------------------------------------------ the page (the model's writing)

export const COLS = 52, CELL = 15, ROWH = 33, TEXT_PX = 25;

export interface Cell { ch: string; r: number; c: number; i: number }
/** A sample laid out on the page grid: newline -> next row, wrap at COLS. `i` = index in the text. */
export function grid(text: string): Cell[] {
  const out: Cell[] = [];
  let r = 0, col = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    if (ch === '\n') { out.push({ ch: '↵', r, c: col, i }); r++; col = 0; continue; }
    if (col >= COLS) { r++; col = 0; }
    out.push({ ch, r, c: col, i });
    col++;
  }
  return out;
}

export interface PageState {
  /** checkpoint shown per cell: the morph from checkpoint `from`'s character to checkpoint k's */
  ckptAt: (cellIndex: number) => { k: number; from: number; u: number };
  /** characters typed so far (Infinity = all) */
  typed: number;
  /** highlighted span [a, b) of the current sample's text and how much of it has been sung (0..1) */
  hi?: { a: number; b: number; sung: number; step: number; t0: number };
  cursorOn: boolean;
  stepLabel: string;
  t: number;
}

const gridCache = new Map<string, Map<string, string>>();
function cellMap(text: string) {
  let m = gridCache.get(text);
  if (!m) { m = new Map(grid(text).map((g) => [`${g.r},${g.c}`, g.ch])); gridCache.set(text, m); }
  return m;
}

export function drawPage(c: CanvasRenderingContext2D, run: Run, r: Rect, s: PageState) {
  sheet(c, r);
  const x0 = r.x - r.w / 2 + 64, y0 = r.y - r.h / 2 + 96;
  c.save();
  c.textBaseline = 'alphabetic';
  c.font = font(F.mono(600), 20);
  c.letterSpacing = '2px';
  c.fillStyle = INK;
  c.fillText('THE PROMPT', x0, y0 - 44);
  c.textAlign = 'right';
  c.font = font(F.mono(700), 22);
  c.fillText(s.stepLabel, r.x + r.w / 2 - 64, y0 - 44);
  c.textAlign = 'left';
  c.letterSpacing = '0px';
  c.font = font(F.mono(700), TEXT_PX + 4);
  c.fillStyle = BLUE;
  c.fillText('ROMEO:', x0, y0);
  c.font = font(F.mono(600), 20);
  c.letterSpacing = '2px';
  c.fillStyle = INK;
  c.fillText('WHAT THE MODEL WROTE NEXT', x0, y0 + 50);
  c.letterSpacing = '0px';
  // the text grid (clipped to the sheet)
  c.save();
  c.beginPath(); c.rect(r.x - r.w / 2, r.y - r.h / 2, r.w, r.h - 24); c.clip();
  const ty = y0 + 104;
  c.font = font(F.mono(600), TEXT_PX);
  const S = run.d.samples;
  const cells = new Map<string, { r: number; c: number }>();
  // every cell any of the two samples in play uses
  const idx = s.ckptAt(0);
  const cur = cellMap(S[idx.k]!.text), prv = cellMap(S[idx.from]!.text);
  for (const key of new Set([...cur.keys(), ...prv.keys()])) { const [rr, cc] = key.split(',').map(Number); cells.set(key, { r: rr!, c: cc! }); }
  const curGrid = grid(S[idx.k]!.text);
  const typedCells = new Set<string>();
  if (Number.isFinite(s.typed)) for (const g of curGrid) if (g.i < s.typed) typedCells.add(`${g.r},${g.c}`);
  let hiCells: Set<string> | null = null, sungCells: Set<string> | null = null;
  if (s.hi) {
    hiCells = new Set(); sungCells = new Set();
    const n = s.hi.b - s.hi.a;
    for (const g of curGrid) if (g.i >= s.hi.a && g.i < s.hi.b) {
      hiCells.add(`${g.r},${g.c}`);
      if ((g.i - s.hi.a) / n < s.hi.sung) sungCells.add(`${g.r},${g.c}`);
    }
  }
  let lastTyped: { r: number; c: number } | null = null;
  for (const [key, pos] of cells) {
    if (Number.isFinite(s.typed) && !typedCells.has(key)) continue;
    const ci = pos.r * COLS + pos.c;
    const { k, from, u } = s.ckptAt(ci);
    const a = cellMap(S[from]!.text).get(key), b = cellMap(S[k]!.text).get(key);
    const x = x0 + pos.c * CELL, y = ty + pos.r * ROWH;
    const sung = sungCells?.has(key);
    if (sung) { c.fillStyle = PINK; c.fillRect(x - 1, y + 7, CELL + 1, 4); }
    const col = sung ? PINK : INK;
    const vis = (ch: string | undefined) => ch && ch !== '↵';
    if (a === b || u >= 1) { if (vis(b)) { c.fillStyle = col; c.fillText(b!, x, y); } }
    else {
      if (vis(a)) { c.globalAlpha = 1 - u; c.fillStyle = col; c.fillText(a!, x, y); }
      if (vis(b)) { c.globalAlpha = u; c.fillStyle = col; c.fillText(b!, x, y); }
      c.globalAlpha = 1;
    }
    if (typedCells.has(key) && (!lastTyped || pos.r > lastTyped.r || (pos.r === lastTyped.r && pos.c > lastTyped.c))) lastTyped = pos;
  }
  // the cursor: after the last typed character (or after the prompt when nothing is typed)
  if (s.cursorOn && Number.isFinite(s.typed)) {
    const p = lastTyped ? { r: lastTyped.r, c: lastTyped.c + 1 } : { r: 0, c: 0 };
    const x = x0 + p.c * CELL, y = ty + p.r * ROWH;
    c.fillStyle = PINK;
    c.fillRect(x + 1, y - TEXT_PX * 0.8, CELL - 2, TEXT_PX * 0.98);
  }
  c.restore();
  // "step N, verbatim" beside the heading of what it wrote (the excerpt starts at its first character)
  if (s.hi) {
    c.font = font(F.mono(700), 22);
    c.letterSpacing = '2px';
    c.textAlign = 'right';
    writeText(c, `STEP ${fmt(s.hi.step)}, VERBATIM`, r.x + r.w / 2 - 64, y0 + 50, s.t, s.hi.t0, { color: PINK, cps: 40 });
    c.textAlign = 'left';
    c.letterSpacing = '0px';
  }
  c.restore();
}

/** Rows a sample takes on the page (for framing the excerpt). */
export function excerptBox(run: Run, r: Rect, sample: number, a: number, b: number): Rect {
  const g = grid(run.d.samples[sample]!.text).filter((x) => x.i >= a && x.i < b);
  const x0 = r.x - r.w / 2 + 64, ty = r.y - r.h / 2 + 96 + 104;
  const rows = g.map((x) => x.r), cols = g.map((x) => x.c);
  const r0 = Math.min(...rows), r1 = Math.max(...rows);
  const c0 = r0 === r1 ? Math.min(...cols) : 0, c1 = r0 === r1 ? Math.max(...cols) : COLS;
  void c0; void c1;
  // the full width of the page's text (so the "STEP N, VERBATIM" label on the right stays in frame)
  return { x: r.x, y: ty + ((r0 + r1) / 2) * ROWH - TEXT_PX * 1.2, w: r.w - 60, h: (r1 - r0 + 1) * ROWH + TEXT_PX * 4 };
}

// ------------------------------------------------------------------ the weights (a corner of them)

let embCanvas: HTMLCanvasElement | null = null;
let embScale = 0;
/** Ink scale for the weights: |value| at the 95th percentile of the last checkpoint prints as full ink. */
function weightScale(run: Run) {
  if (!embScale) {
    const v = run.p.steps[run.p.steps.length - 1]!.emb.flat().map(Math.abs).sort((a, b) => a - b);
    embScale = v[Math.floor(v.length * 0.95)] || 0.1;
  }
  return embScale;
}
/** The first 48 numbers of each character's row in the embedding table, blended between two checkpoints. */
export function drawWeights(c: CanvasRenderingContext2D, run: Run, r: Rect, t: number, k0: number, k: number, u: number, o: { t0: number; countT0: number; oneT0: number; oneHot: boolean }) {
  sheet(c, r);
  const P = run.p.steps, A = P[k0]!.emb, B = P[k]!.emb;
  const rows = A.length, cols = A[0]!.length;
  if (!embCanvas) { embCanvas = document.createElement('canvas'); embCanvas.width = cols; embCanvas.height = rows; }
  const ec = embCanvas.getContext('2d')!;
  const img = ec.createImageData(cols, rows);
  const blue = [0, 120, 191], ink = [35, 31, 32], paper = [242, 239, 230], sc = weightScale(run);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const v = lerp(A[i]![j]!, B[i]![j]!, u);
    const a = clamp(Math.abs(v) / sc);
    const col = v >= 0 ? blue : ink;
    const o = (i * cols + j) * 4;
    for (let q = 0; q < 3; q++) img.data[o + q] = Math.round(paper[q]! + (col[q]! - paper[q]!) * a);
    img.data[o + 3] = 255;
  }
  ec.putImageData(img, 0, 0);
  const cell = 10.5, gw = cols * cell, gh = rows * cell;
  const gx = r.x - r.w / 2 + 110, gy = r.y + r.h / 2 - gh - 50;
  c.save();
  c.imageSmoothingEnabled = false;
  c.drawImage(embCanvas, gx, gy, gw, gh);
  c.strokeStyle = INK; c.lineWidth = 1.5; c.strokeRect(gx, gy, gw, gh);
  // row labels: the characters
  c.font = font(F.mono(600), 9);
  c.textAlign = 'right';
  c.textBaseline = 'middle';
  c.fillStyle = INK;
  const V = run.d.vocab;
  for (let i = 0; i < rows; i++) c.fillText(showChar(V[i]!), gx - 6, gy + (i + 0.5) * cell);
  c.textAlign = 'left';
  c.textBaseline = 'alphabetic';
  // heading: the whole count (the one number)
  label(c, r.x - r.w / 2 + 50, r.y - r.h / 2 + 150, 'THE WEIGHTS: EVERY NUMBER IN THE MODEL', fmt(run.d.n_params), t, o.countT0, { size: 92, color: INK });
  c.font = font(F.mono(500), 15);
  writeText(c, 'a corner of them: 48 numbers for each of the 65 characters', r.x - r.w / 2 + 50, r.y - r.h / 2 + 200, t, o.countT0 + 0.8, { color: INK, cps: 45 });
  // one weight, followed: the character "e", its first number
  if (t >= o.oneT0) {
    const ei = V.indexOf('e');
    const x = gx + cell * 0.5, y = gy + (ei + 0.5) * cell;
    const a = inP(t, o.oneT0, 0.5);
    c.strokeStyle = PINK; c.lineWidth = 3;
    c.beginPath(); c.arc(x, y, 9 + 20 * (1 - a), 0, Math.PI * 2); c.stroke();
    const v = lerp(P[k0]!.one, P[k]!.one, u);
    // a leader up from the circled cell to the value, written above the grid
    c.beginPath(); c.moveTo(x, y - 10); c.lineTo(x, gy - 12); c.lineTo(x + 20, gy - 30); c.stroke();
    c.textAlign = 'left';
    c.globalAlpha = a;
    c.font = font(F.mono(500), 17);
    c.fillStyle = INK;
    c.fillText('one weight (“e”, number 1):', x + 28, gy - 24);
    c.font = font(F.mono(700), 34);
    c.fillStyle = o.oneHot ? PINK : INK;
    c.globalAlpha = a * NUM.a;
    c.fillText(`${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(4)}`, x + 330, gy - 22);
    c.globalAlpha = 1;
  }
  c.restore();
}

// ------------------------------------------------------------------ the book (the training text)

export function drawBook(c: CanvasRenderingContext2D, run: Run, r: Rect, t: number, t0: number, text: string) {
  sheet(c, r, inP(t, t0 - 0.6, 0.6));
  if (t < t0 - 0.6) return;
  const x0 = r.x - r.w / 2 + 44, y0 = r.y - r.h / 2 + 64;
  const d = run.d.dataset;
  c.save();
  label(c, x0, y0 + 50, 'THE BOOK: TINY SHAKESPEARE', `${fmt(d.chars)} characters`, t, t0, { size: 50, color: BLUE });
  // the split: what it practises on, what is held back
  const bw = r.w - 88, by = y0 + 100, frac = d.train_chars / d.chars;
  const a = inP(t, t0 + 1.2, 1.2);
  c.fillStyle = tint('blue', 0.85); c.fillRect(x0, by, bw * frac * a, 26);
  c.strokeStyle = INK; c.lineWidth = 1.5; c.strokeRect(x0, by, bw, 26);
  c.font = font(F.mono(600), 15);
  c.fillStyle = INK;
  writeText(c, `IT PRACTISES ON ${fmt(d.train_chars)} (90%)`, x0, by + 50, t, t0 + 1.6, { color: INK, cps: 50 });
  c.textAlign = 'right';
  writeText(c, `HELD BACK: ${fmt(d.val_chars)}`, x0 + bw, by + 50, t, t0 + 2.2, { color: INK, cps: 50 });
  c.textAlign = 'left';
  // a few real lines of it, drifting up
  c.font = font(F.mono(400), 17);
  c.fillStyle = tint('blue', 0.9);
  const lines = text.split('\n');
  const scroll = (t - t0) * 0.35;
  c.beginPath(); c.rect(x0, by + 70, bw, r.h - (by + 70 - (r.y - r.h / 2)) - 24); c.clip();
  for (let i = 0; i < 14; i++) {
    const li = Math.floor(scroll) + i;
    const y = by + 100 + (i - (scroll % 1)) * 26;
    const s = lines[li % lines.length] ?? '';
    c.globalAlpha = inP(t, t0 + 0.5 + i * 0.08, 0.5);
    c.fillText(s.slice(0, 80), x0, y);
  }
  c.restore();
}

// ------------------------------------------------------------------ the guess (65 bars)

export function drawGuess(c: CanvasRenderingContext2D, run: Run, r: Rect, t: number, A: ProbeStep, B: ProbeStep, u: number, pos: number, o: { t0: number; answerT0: number; pT0: number }) {
  sheet(c, r, inP(t, o.t0 - 0.6, 0.6));
  if (t < o.t0 - 0.6) return;
  const V = run.d.vocab, line = run.p.line;
  const x0 = r.x - r.w / 2 + 44, y0 = r.y - r.h / 2 + 70;
  c.save();
  c.font = font(F.mono(600), 15);
  c.letterSpacing = '2px';
  writeText(c, 'A LINE IT NEVER PRACTISED ON (THE TAMING OF THE SHREW)', x0, y0, t, o.t0, { color: INK, cps: 50 });
  c.letterSpacing = '0px';
  // the line, typed up to the guess (character `pos` of the line), then the blank
  c.font = font(F.mono(500), 30);
  const cw = 18;
  const shown = line.slice(0, pos);
  c.fillStyle = BLUE;
  c.fillText(shown, x0, y0 + 56);
  const bx = x0 + shown.length * cw;
  const ans = line[pos]!;
  const reveal = inP(t, o.answerT0, 0.5);
  c.strokeStyle = PINK; c.lineWidth = 3;
  c.strokeRect(bx, y0 + 56 - 26, cw, 34);
  if (reveal > 0) { c.globalAlpha = reveal; c.fillStyle = PINK; c.fillText(ans === ' ' ? '·' : ans, bx, y0 + 56); c.globalAlpha = 1; }
  c.font = font(F.mono(600), 15);
  c.letterSpacing = '2px';
  writeText(c, `WHAT COMES NEXT? ITS GUESS, ONE BAR FOR EACH OF THE ${V.length} CHARACTERS`, x0, y0 + 110, t, o.t0 + 0.8, { color: INK, cps: 50 });
  c.letterSpacing = '0px';
  // the bars
  const pa = A.chars[pos]!.dist, pb = B.chars[pos]!.dist;
  const n = V.length, bw = (r.w - 88) / n, base = r.y + r.h / 2 - 60, hmax = 400;
  const grow = inP(t, o.t0 + 1.0, 1.2);
  const ai = V.indexOf(ans);
  for (let i = 0; i < n; i++) {
    const p = lerp(pa[i]!, pb[i]!, u);
    const h = Math.max(1.5, p * hmax * grow);
    const right = i === ai && reveal > 0;
    c.fillStyle = right ? PINK : tint('blue', 0.85);
    c.fillRect(x0 + i * bw + 1, base - h, bw - 2, h);
  }
  c.strokeStyle = INK; c.lineWidth = 1.5;
  c.beginPath(); c.moveTo(x0, base); c.lineTo(x0 + n * bw, base); c.stroke();
  c.font = font(F.mono(500), 11);
  c.textAlign = 'center';
  c.fillStyle = INK;
  for (let i = 0; i < n; i++) c.fillText(showChar(V[i]!), x0 + (i + 0.5) * bw, base + 16);
  // the right answer's chance: the one number
  if (t >= o.pT0) {
    const p = lerp(pa[ai]!, pb[ai]!, u);
    const x = x0 + (ai + 0.5) * bw, h = Math.max(1.5, p * hmax * grow);
    c.font = font(F.archivo(100, 900), 44);
    c.textAlign = 'center';
    c.globalAlpha = inP(t, o.pT0, 0.4) * NUM.a;
    c.fillStyle = PINK;
    c.fillText(`${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`, x, base - h - 18);
    c.globalAlpha = 1;
  }
  c.restore();
}

// ------------------------------------------------------------------ the surprise (the loss on every letter)

export function drawSurprise(c: CanvasRenderingContext2D, run: Run, r: Rect, t: number, A: ProbeStep, B: ProbeStep, u: number, pos: number, o: { t0: number; allT0: number; meanT0: number }) {
  sheet(c, r, inP(t, o.t0 - 0.6, 0.6));
  if (t < o.t0 - 0.6) return;
  const line = run.p.line.replace(/\n$/, '');
  const x0 = r.x - r.w / 2 + 44, y0 = r.y - r.h / 2 + 64;
  c.save();
  c.font = font(F.mono(600), 15);
  c.letterSpacing = '2px';
  writeText(c, 'THE LOSS: HOW SURPRISED IT WAS BY EACH RIGHT LETTER', x0, y0, t, o.t0, { color: INK, cps: 50 });
  c.letterSpacing = '0px';
  const n = line.length, cw = (r.w - 88) / (n + 1);
  const ty = y0 + 58, base = ty + 20, hmax = 40; // px per unit of loss
  c.font = font(F.mono(500), Math.min(28, cw * 1.55));
  c.textAlign = 'center';
  for (let i = 0; i < n; i++) {
    const x = x0 + (i + 0.5) * cw;
    const on = i === pos ? inP(t, o.t0, 0.5) : inP(t, o.allT0 + i * 0.05, 0.4);
    c.fillStyle = BLUE;
    c.fillText(line[i]!, x, ty);
    if (on <= 0) continue;
    // chars[i] is the guess for character i of the line (the first one, "G", follows the speaker's name)
    const L = lerp(A.chars[i]!.loss, B.chars[i]!.loss, u);
    c.fillStyle = i === pos ? PINK : tint('pink', 0.75);
    c.fillRect(x - cw * 0.38, base, cw * 0.76, L * hmax * on);
  }
  // the loss of the one guess (verse 2's example), until the line's average takes over
  if (t >= o.t0 + 0.5) {
    const i = pos, x = x0 + (i + 0.5) * cw;
    const L = lerp(A.chars[i]!.loss, B.chars[i]!.loss, u);
    c.save();
    c.textAlign = 'center';
    c.globalAlpha = inP(t, o.t0 + 0.5, 0.4) * (1 - inP(t, o.meanT0 - 0.4, 0.4)) * NUM.a;
    c.font = font(F.mono(600), 14);
    c.fillStyle = INK;
    c.fillText('LOSS', x, base + L * hmax + 26);
    c.font = font(F.archivo(100, 900), 40);
    c.fillStyle = PINK;
    c.fillText(L.toFixed(2), x, base + L * hmax + 66);
    c.restore();
  }
  // the mean: the loss of the whole line (the one number)
  if (t >= o.meanT0) {
    const L = lerp(A.line_loss, B.line_loss, u);
    c.textAlign = 'right';
    c.globalAlpha = inP(t, o.meanT0, 0.4) * NUM.a;
    c.font = font(F.mono(600), 15);
    c.fillStyle = INK;
    c.fillText('AVERAGE LOSS ON THIS LINE', r.x + r.w / 2 - 44, r.y + r.h / 2 - 84);
    c.font = font(F.archivo(100, 900), 56);
    c.fillStyle = PINK;
    c.fillText(L.toFixed(2), r.x + r.w / 2 - 44, r.y + r.h / 2 - 28);
    c.globalAlpha = 1;
  }
  c.restore();
}

// ------------------------------------------------------------------ the loss curve

export function drawCurve(c: CanvasRenderingContext2D, run: Run, r: Rect, t: number, step: number, o: { t0: number; trainT0: number; valT0: number; minT0: number; memT0: number }) {
  sheet(c, r, inP(t, o.t0 - 0.6, 0.6));
  if (t < o.t0 - 0.6) return;
  const x0 = r.x - r.w / 2 + 120, x1 = r.x + r.w / 2 - 60, y0 = r.y - r.h / 2 + 170, y1 = r.y + r.h / 2 - 110;
  const N = run.maxStep, LMAX = 4.5;
  const X = (s: number) => x0 + (s / N) * (x1 - x0), Y = (l: number) => y1 - (l / LMAX) * (y1 - y0);
  const E = run.d.evals;
  c.save();
  label(c, r.x - r.w / 2 + 50, r.y - r.h / 2 + 100, 'THE LOSS AT EVERY STEP (EVALUATED EVERY 250 STEPS)', 'LOSS', t, o.t0, { size: 56, color: PINK });
  // axes
  const ax = inP(t, o.t0, 1.0);
  c.strokeStyle = INK; c.lineWidth = 2;
  c.beginPath(); c.moveTo(x0, y1 - (y1 - y0) * ax); c.lineTo(x0, y1); c.lineTo(x0 + (x1 - x0) * ax, y1); c.stroke();
  c.font = font(F.mono(500), 17); c.fillStyle = INK;
  c.textAlign = 'center';
  c.globalAlpha = ax;
  for (const st of [0, 1000, 2000, 3000, 4000, 5000]) c.fillText(fmt(st), X(st), y1 + 28);
  c.fillText('STEP', (x0 + x1) / 2, y1 + 60);
  c.textAlign = 'right';
  for (const l of [0, 1, 2, 3, 4]) c.fillText(String(l), x0 - 12, Y(l) + 6);
  c.globalAlpha = 1;
  // past the lowest unseen loss: it is learning the lines by heart (shaded, labelled)
  const best = E.reduce((m, e) => (e.val < m.val ? e : m), E[0]!);
  if (t >= o.memT0 && step > best.step) {
    const a = inP(t, o.memT0, 1.2);
    c.fillStyle = tint('pink', 0.13 * a);
    c.fillRect(X(best.step), y0, (X(Math.min(step, N)) - X(best.step)), y1 - y0);
    c.globalAlpha = a;
    c.font = font(F.mono(700), 20); c.fillStyle = INK; c.textAlign = 'left';
    c.letterSpacing = '2px';
    writeText(c, 'LEARNING THE LINES BY HEART', X(best.step) + 18, y0 + 34, t, o.memT0, { color: INK, cps: 30 });
    c.letterSpacing = '0px';
    c.globalAlpha = 1;
  }
  const line = (k: 'train' | 'val', col: string, w: number) => {
    const pts = E.filter((e) => e.step <= step);
    c.strokeStyle = col; c.lineWidth = w; c.lineJoin = 'round';
    c.beginPath();
    pts.forEach((e, i) => { if (i === 0) c.moveTo(X(e.step), Y(e[k])); else c.lineTo(X(e.step), Y(e[k])); });
    if (pts.length) c.lineTo(X(step), Y(run.evalAt(step, k)));
    c.stroke();
  };
  // the practice text (training loss, no dropout): blue; the pen rides it
  line('train', BLUE, 5);
  // the text it never practised on (validation): pink, drawn in when sung
  if (t >= o.valT0 - 0.5) { c.globalAlpha = inP(t, o.valT0 - 0.5, 0.5); line('val', PINK, 5); c.globalAlpha = 1; }
  c.fillStyle = BLUE;
  c.beginPath(); c.arc(X(step), Y(run.evalAt(step, 'train')), 10, 0, Math.PI * 2); c.fill();
  // one number at a time: the practice loss at the end, then the unseen text's lowest point and where it ended
  const num = (v: number, x: number, y: number, col: string, sub: string, a: number, align: CanvasTextAlign = 'right') => {
    if (a <= 0) return;
    c.globalAlpha = a * NUM.a;
    c.textAlign = align;
    c.font = font(F.archivo(100, 900), 50); c.fillStyle = col;
    c.fillText(v.toFixed(2), x, y);
    c.font = font(F.mono(600), 16); c.fillStyle = INK;
    c.fillText(sub, x, y + 26);
    c.globalAlpha = 1;
  };
  const e1 = run.finalEval();
  num(e1.train, X(N), Y(e1.train) + 58, BLUE, 'PRACTICE TEXT', inP(t, o.trainT0, 0.4) * (1 - 0.65 * inP(t, o.valT0, 0.4)));
  if (t >= o.minT0) {
    c.strokeStyle = PINK; c.lineWidth = 3;
    c.beginPath(); c.arc(X(best.step), Y(best.val), 14 * inP(t, o.minT0, 0.4), 0, Math.PI * 2); c.stroke();
  }
  num(best.val, X(best.step), Y(best.val) + 64, PINK, `LOWEST, STEP ${fmt(best.step)}`, inP(t, o.minT0, 0.4) * (1 - 0.65 * inP(t, o.memT0, 0.4)), 'center');
  num(e1.val, X(N), Y(e1.val) - 40, PINK, 'UNSEEN TEXT, STEP 5,000', inP(t, o.memT0 + 0.8, 0.4));
  c.restore();
}

// ------------------------------------------------------------------ the loop (the chorus's picture)

export const LOOP = ['GUESS THE NEXT LETTER', 'SEE HOW WRONG IT WAS', 'NUDGE EVERY WEIGHT A LITTLE', 'AND DO IT AGAIN'];

/** The ring around the page: four labels (drawn in on their first chorus, then left up), the one being sung in
 *  pink, and a pink bead that goes round once per `lap` of the clock. */
export function drawRing(c: CanvasRenderingContext2D, r: Rect, t: number, labelT0: number[], hot: number, bead: number, a: number) {
  if (a <= 0) return;
  const rx = r.w / 2 - 70, ry = r.h / 2 - 60;
  c.save();
  c.globalAlpha *= a;
  c.strokeStyle = INK; c.lineWidth = 3;
  c.setLineDash([2, 12]); c.lineCap = 'round';
  c.beginPath(); c.ellipse(r.x, r.y, rx, ry, 0, 0, Math.PI * 2); c.stroke();
  c.setLineDash([]);
  // labels at 12, 3, 6, 9 o'clock (16:9) or on the diagonals (9:16, where 3 and 9 o'clock would leave the frame)
  const off = H > W ? -Math.PI / 4 : 0;
  // arrowheads between the labels (clockwise)
  for (let q = 0; q < 4; q++) {
    const th = -Math.PI / 2 + off + (q + 0.5) * (Math.PI / 2);
    const x = r.x + rx * Math.cos(th), y = r.y + ry * Math.sin(th);
    const dx = -rx * Math.sin(th), dy = ry * Math.cos(th), L = Math.hypot(dx, dy);
    const ux = dx / L, uy = dy / L;
    c.fillStyle = INK;
    c.beginPath(); c.moveTo(x + ux * 16, y + uy * 16); c.lineTo(x - ux * 8 - uy * 10, y - uy * 8 + ux * 10); c.lineTo(x - ux * 8 + uy * 10, y - uy * 8 - ux * 10); c.fill();
  }
  // the bead
  const th = -Math.PI / 2 + bead * Math.PI * 2;
  c.fillStyle = PINK;
  c.beginPath(); c.arc(r.x + rx * Math.cos(th), r.y + ry * Math.sin(th), 13, 0, Math.PI * 2); c.fill();
  // the labels at 12, 3, 6 and 9 o'clock, on paper plates
  c.font = font(F.archivo(100, 900), H > W ? 27 : 34);
  c.textBaseline = 'middle';
  c.textAlign = 'center';
  LOOP.forEach((s, q) => {
    const t0 = labelT0[q]!;
    if (!(t >= t0 - 0.2)) return;
    const th2 = -Math.PI / 2 + off + q * (Math.PI / 2);
    const w = c.measureText(s).width;
    let x = r.x + rx * Math.cos(th2), y = r.y + ry * Math.sin(th2);
    // 9:16: the four labels sit above and below the page (on the diagonals they would cover its corners)
    if (H > W) { x = r.x + (q === 0 || q === 3 ? -1 : 1) * 250; y = q < 2 ? r.y - 20 - 500 - 80 : r.y - 20 + 500 + 190; }
    // in the close-ups of the page (zoomed in past 1.0) the labels would sit under the sung line: they step back
    const pa = inP(t, t0 - 0.2, 0.3) * (1 - clamp((NUM.z - 1.0) / 0.2));
    if (pa <= 0.002) return;
    c.globalAlpha = a * pa;
    c.fillStyle = PAPER; c.fillRect(x - w / 2 - 16, y - 28, w + 32, 56);
    c.strokeStyle = q === hot ? PINK : INK; c.lineWidth = q === hot ? 4 : 2; c.strokeRect(x - w / 2 - 16, y - 28, w + 32, 56);
    c.globalAlpha = a;
    c.globalAlpha = a * pa;
    writeText(c, s, x, y + 2, t, t0 - 0.2, { color: q === hot ? PINK : INK, cps: 30 });
  });
  c.restore();
}

/** Deterministic jitter for "something always moves" (seeded, smooth). */
export const wob = (t: number, seed: number, amp = 1) => amp * (Math.sin(t * 0.37 + hash(seed) * 6.28) * 0.6 + Math.sin(t * 0.23 + hash(seed, 2) * 6.28) * 0.4);

/** Chorus 2's added layer: under the page, the loss (on the held-back text) at the step the page is showing. */
export function drawLossGauge(c: CanvasRenderingContext2D, run: Run, page: Rect, t: number, step: number, t0: number) {
  if (t < t0) return;
  const a = inP(t, t0, 0.8);
  const x0 = page.x - page.w / 2, y = page.y + page.h / 2 + 64, w = page.w;
  const L = run.evalAt(step, 'val'), LMAX = 4.5;
  c.save();
  c.globalAlpha *= a;
  c.font = font(F.mono(600), 20);
  c.letterSpacing = '2px';
  c.fillStyle = INK;
  c.fillText('ITS LOSS ON TEXT IT NEVER PRACTISED ON', x0, y);
  c.letterSpacing = '0px';
  c.strokeStyle = INK; c.lineWidth = 2;
  c.strokeRect(x0, y + 16, w - 150, 30);
  c.fillStyle = PINK;
  c.fillRect(x0, y + 16, (w - 150) * clamp(L / LMAX) * a, 30);
  c.font = font(F.archivo(100, 900), 48);
  c.textAlign = 'right';
  c.globalAlpha *= NUM.a;
  c.fillText(L.toFixed(2), x0 + w, y + 46);
  c.restore();
}
