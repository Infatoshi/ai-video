// Drawing kit for the "Tokens" sheet (world.ts): everything is Canvas2D in world units under the camera
// transform. Letter tiles and chunks (type sorts on a line), split-flap flips to numbers, labels that draw
// in and stay, the model box, number slips, tally grids. Inks: pink = what is being taught right now,
// blue = the numbers (the data the model acts on), ink = structure, letters, outlines.
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp } from '../engine/util';

export const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
export const pinkA = (a: number) => rgba('pink', a);
export const blueA = (a: number) => rgba('blue', a);
export const inkA = (a: number) => rgba('ink', a);

/** Eased 0..1 from time a over d seconds (inOutCubic). */
export const k = (t: number, a: number, d = 0.5) => ease.inOutCubic(clamp((t - a) / d));
/** Eased 0..1 in over [a, a+d], back to 0 over [b, b+d]. */
export const win = (t: number, a: number, b: number, d = 0.45) => Math.min(k(t, a, d), 1 - k(t, b, d));
/** A flash: rises over 0.3 s from a, decays with half-life hl. */
export const flash = (t: number, a: number, hl = 0.5) => (t < a ? 0 : Math.min(1, (t - a) / 0.3) * Math.pow(0.5, Math.max(0, t - a - 0.3) / hl));

export function rrect(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

// ------------------------------------------------------------------ the line of type

/** A line of type sorts: characters (a space shows as '·'), laid out in world units, wrapping at `breaks`. */
export interface RowGeom {
  chars: string[];
  /** Centre of the block. */
  x: number;
  y: number;
  cw: number;
  ch: number;
  /** Visual line starts (char indices) for wrapped layouts (portrait). */
  breaks: number[];
  lineGap: number;
}

/** Per-frame state of a row. gaps[i] = world gap after char i. */
export interface RowState {
  gaps: number[];
  /** Per boundary i (between char i and i+1): 0 separate tiles, 1 fused into one chunk. */
  fuse: number[];
  /** Chunks as [first, last] char ranges (inclusive), for chunk decorations. */
  chunks: [number, number][];
  /** Per chunk: flip progress 0 (letters) .. 1 (number), the number, pink flash. */
  flip: number[];
  ids: (number | string)[];
  hot: number[];
  /** Per char: pink letter amount, visibility. */
  pinkChar: number[];
  alpha: number[];
  /** Whole-row opacity and letter dimming (0..1, 1 = normal ink). */
  a: number;
  ink: number;
}

export function newState(n: number, chunks: [number, number][] = []): RowState {
  return {
    gaps: new Array(Math.max(0, n - 1)).fill(0), fuse: new Array(Math.max(0, n - 1)).fill(0), chunks,
    flip: chunks.map(() => 0), ids: chunks.map(() => ''), hot: chunks.map(() => 0),
    pinkChar: new Array(n).fill(0), alpha: new Array(n).fill(1), a: 1, ink: 1,
  };
}

/** Char cell positions (top-left) for the gaps. */
export function layoutRow(g: RowGeom, gaps: number[]): { x: number; y: number }[] {
  const n = g.chars.length;
  const starts = [0, ...g.breaks.filter((b) => b > 0 && b < n), n];
  const lines = starts.length - 1;
  const out: { x: number; y: number }[] = new Array(n);
  const totalH = lines * g.ch + (lines - 1) * g.lineGap;
  for (let li = 0; li < lines; li++) {
    const a = starts[li]!, b = starts[li + 1]!;
    let w = 0;
    for (let i = a; i < b; i++) w += g.cw + (i < b - 1 ? gaps[i] ?? 0 : 0);
    let x = g.x - w / 2;
    const y = g.y - totalH / 2 + li * (g.ch + g.lineGap);
    for (let i = a; i < b; i++) { out[i] = { x, y }; x += g.cw + (gaps[i] ?? 0); }
  }
  return out;
}

/** Letters fused into one chunk sit tighter than their cells (the chunk reads as a word, not spaced type). */
export const TIGHT = 0.2;
/** Gap for a boundary: separate letters g0, fused -TIGHT*cw, a chunk border gc (mixed by fuse and split). */
export const gapOf = (g0: number, gc: number, fuse: number, split: number, cw = 0) => lerp(g0 * (1 - fuse) - TIGHT * cw * fuse, gc, split);

/** Is boundary i a line wrap? */
const wraps = (g: RowGeom, i: number) => g.breaks.includes(i + 1);

/**
 * Draw a row: tiles (paper with a blue slab shadow), letters, chunk outlines crossfading from per-letter
 * outlines as they fuse, and chunks flipping (split-flap, about the row's horizontal axis) to their number.
 */
export function drawRow(c: CanvasRenderingContext2D, g: RowGeom, s: RowState, o: { fs?: number; numFs?: number; slab?: number; lw?: number } = {}) {
  const n = g.chars.length;
  if (s.a <= 0.003 || n === 0) return;
  const pos = layoutRow(g, s.gaps);
  const fs = o.fs ?? g.ch * 0.6, lw = o.lw ?? Math.max(1.5, g.cw * 0.05), slab = o.slab ?? g.cw * 0.1, R = g.cw * 0.14;
  c.save();
  c.globalAlpha = s.a;
  // chunk of each char (for flips)
  const chunkOf = new Array(n).fill(-1);
  s.chunks.forEach(([a, b], ci) => { for (let i = a; i <= b; i++) chunkOf[i] = ci; });
  // runs of chars to draw as one box: split where the boundary is not fused or wraps
  const runs: [number, number][] = [];
  let r0 = 0;
  for (let i = 0; i < n - 1; i++) if (s.fuse[i]! < 0.999 || wraps(g, i) || chunkOf[i] !== chunkOf[i + 1]) { runs.push([r0, i]); r0 = i + 1; }
  runs.push([r0, n - 1]);
  const flipOf = (i: number) => (chunkOf[i]! >= 0 ? s.flip[chunkOf[i]!]! : 0);

  // 1) slabs (blue shadow) and paper faces, per char (so partly fused tiles read as separate sorts)
  for (let i = 0; i < n; i++) {
    const a = s.alpha[i]!;
    if (a <= 0.003) continue;
    const p = pos[i]!;
    const f = flipOf(i), sy = Math.abs(Math.cos(Math.PI * f));
    const right = i < n - 1 && !wraps(g, i) ? (s.gaps[i]! * clamp(s.fuse[i]! * 1.02)) : 0;
    const left = i > 0 && !wraps(g, i - 1) ? (s.gaps[i - 1]! * clamp(s.fuse[i - 1]! * 1.02)) * 0 : 0;
    const h = g.ch * sy, y = p.y + (g.ch - h) / 2;
    c.globalAlpha = s.a * a;
    c.fillStyle = blueA(0.45);
    c.fillRect(p.x - left + slab, y + slab, g.cw + right, h);
    c.fillStyle = f >= 0.5 ? PAPER : PAPER;
    c.fillRect(p.x - left, y, g.cw + right, h);
  }
  // 2) chunk tints (flash pink, number side blue tint)
  s.chunks.forEach(([a, b], ci) => {
    const f = s.flip[ci]!, hot = s.hot[ci]!;
    const sy = Math.abs(Math.cos(Math.PI * f));
    const p0 = pos[a]!, p1 = pos[b]!;
    if (p0.y !== p1.y) return; // wrapped chunk: skip the tint
    const h = g.ch * sy, y = p0.y + (g.ch - h) / 2;
    const alpha = s.a * Math.min(...Array.from({ length: b - a + 1 }, (_, j) => s.alpha[a + j]!));
    if (f >= 0.5) { c.globalAlpha = alpha; c.strokeStyle = BLUE; c.lineWidth = Math.max(2, g.cw * 0.06); rrect(c, p0.x + 5, y + 5 * (h / g.ch), p1.x + g.cw - p0.x - 10, h - 10 * (h / g.ch), R * 0.6); c.stroke(); }
    if (hot > 0.01) { c.globalAlpha = alpha * hot; c.fillStyle = pinkA(0.55); c.fillRect(p0.x, y, p1.x + g.cw - p0.x, h); }
  });
  // 3) letters (face up) or numbers (flipped)
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {
    const a = s.alpha[i]!, f = flipOf(i);
    if (a <= 0.003 || f >= 0.5) continue;
    const p = pos[i]!, sy = Math.abs(Math.cos(Math.PI * f));
    if (sy < 0.02) continue;
    c.save();
    c.globalAlpha = s.a * a;
    c.translate(p.x + g.cw / 2, p.y + g.ch / 2);
    c.scale(1, sy);
    const ch = g.chars[i]!;
    c.fillStyle = s.pinkChar[i]! > 0.5 ? PINK : ch === '·' ? BLUE : s.ink >= 0.999 ? INK : inkA(0.25 + 0.75 * s.ink);
    c.font = font(F.mono(600), fs);
    c.fillText(ch, 0, fs * 0.04);
    c.restore();
  }
  s.chunks.forEach(([a, b], ci) => {
    const f = s.flip[ci]!;
    if (f < 0.5) return;
    const p0 = pos[a]!, p1 = pos[b]!;
    if (p0.y !== p1.y) return;
    const sy = Math.abs(Math.cos(Math.PI * f));
    if (sy < 0.02) return;
    const w = p1.x + g.cw - p0.x;
    const label = String(s.ids[ci] ?? '');
    let nf = o.numFs ?? fs * 0.92;
    c.font = font(F.mono(700), nf);
    const tw = c.measureText(label).width;
    if (tw > w * 0.86) { nf *= (w * 0.86) / tw; c.font = font(F.mono(700), nf); }
    c.save();
    c.globalAlpha = s.a * Math.min(...Array.from({ length: b - a + 1 }, (_, j) => s.alpha[a + j]!));
    c.translate(p0.x + w / 2, p0.y + g.ch / 2);
    c.scale(1, sy);
    c.fillStyle = BLUE;
    c.fillText(label, 0, nf * 0.04);
    c.restore();
  });
  // 4) outlines: per-letter boxes fade out as the letters fuse into the chunk's one box
  c.lineWidth = lw;
  c.strokeStyle = INK;
  for (const [a, b] of runs) {
    const p0 = pos[a]!, p1 = pos[b]!;
    let alpha = 1;
    for (let i = a; i <= b; i++) alpha = Math.min(alpha, s.alpha[i]!);
    if (alpha <= 0.003) continue;
    const f = flipOf(a), sy = Math.abs(Math.cos(Math.PI * f));
    const h = g.ch * sy, y = p0.y + (g.ch - h) / 2;
    const right = b < n - 1 && !wraps(g, b) ? s.gaps[b]! * clamp(s.fuse[b]!) : 0;
    c.globalAlpha = s.a * alpha;
    rrect(c, p0.x, y, p1.x + g.cw - p0.x + right, h, R * sy);
    c.stroke();
  }
  // seams inside a fused run that is still settling (fuse < 1 never reaches here: those split runs)
  c.restore();
}

/** Screen-independent: centre of a chunk (world). */
export function chunkBox(g: RowGeom, s: RowState, ci: number) {
  const pos = layoutRow(g, s.gaps);
  const [a, b] = s.chunks[ci]!;
  const p0 = pos[a]!, p1 = pos[b]!;
  return { x: p0.x, y: p0.y, w: p1.x + g.cw - p0.x, h: g.ch, cx: (p0.x + p1.x + g.cw) / 2, cy: p0.y + g.ch / 2 };
}
export function charBox(g: RowGeom, s: RowState, i: number) {
  const p = layoutRow(g, s.gaps)[i]!;
  return { x: p.x, y: p.y, w: g.cw, h: g.ch, cx: p.x + g.cw / 2, cy: p.y + g.ch / 2 };
}

// ------------------------------------------------------------------ labels, boxes, slips

/**
 * A label that draws in (a left-to-right wipe over `d` s from `t0`) and stays. `size` world units. Pink
 * while `pinkUntil` has not passed (it is the thing being sung), then ink.
 */
export function label(c: CanvasRenderingContext2D, t: number, t0: number, text: string, x: number, y: number,
  o: { size?: number; align?: CanvasTextAlign; mono?: boolean; weight?: number; color?: string; d?: number; a?: number; pinkFrom?: number; pinkTo?: number; width?: number } = {}) {
  const p = k(t, t0, o.d ?? 0.6);
  const a = o.a ?? 1;
  if (p <= 0 || a <= 0.003) return 0;
  const size = o.size ?? 34;
  c.save();
  c.font = o.mono ? font(F.mono(o.weight ?? 600), size) : font(F.archivo(o.width ?? 100, o.weight ?? 700), size);
  c.textAlign = o.align ?? 'left';
  c.textBaseline = 'middle';
  const w = c.measureText(text).width;
  const x0 = (o.align ?? 'left') === 'center' ? x - w / 2 : (o.align ?? 'left') === 'right' ? x - w : x;
  c.beginPath();
  c.rect(x0 - 4, y - size, (w + 8) * p, size * 2);
  c.clip();
  const pk = o.pinkFrom !== undefined ? (t >= o.pinkFrom && t < (o.pinkTo ?? Infinity) ? 1 : 0) : 0;
  const tail = o.pinkTo !== undefined ? clamp((t - o.pinkTo) / 0.4) : 1;
  c.globalAlpha = a * (pk ? 1 : 1);
  c.fillStyle = o.color ?? (pk ? PINK : INK);
  if (!pk && o.pinkTo !== undefined && tail < 1) c.fillStyle = PINK;
  c.fillText(text, x, y);
  c.restore();
  return w;
}

/** A line that draws itself from (x0,y0) to (x1,y1) over d seconds from t0, optional arrow head. */
export function drawLine(c: CanvasRenderingContext2D, t: number, t0: number, x0: number, y0: number, x1: number, y1: number,
  o: { d?: number; color?: string; lw?: number; head?: number; dash?: number[]; a?: number } = {}) {
  const p = k(t, t0, o.d ?? 0.5);
  if (p <= 0) return;
  const x = lerp(x0, x1, p), y = lerp(y0, y1, p);
  c.save();
  c.globalAlpha = o.a ?? 1;
  c.strokeStyle = o.color ?? INK;
  c.fillStyle = o.color ?? INK;
  c.lineWidth = o.lw ?? 3;
  c.lineCap = 'round';
  if (o.dash) c.setLineDash(o.dash);
  c.beginPath(); c.moveTo(x0, y0); c.lineTo(x, y); c.stroke();
  if (o.head && p > 0.6) {
    c.setLineDash([]);
    const ang = Math.atan2(y1 - y0, x1 - x0), hh = o.head * clamp((p - 0.6) / 0.4);
    c.beginPath();
    c.moveTo(x, y);
    c.lineTo(x - Math.cos(ang - 0.45) * hh, y - Math.sin(ang - 0.45) * hh);
    c.lineTo(x - Math.cos(ang + 0.45) * hh, y - Math.sin(ang + 0.45) * hh);
    c.closePath(); c.fill();
  }
  c.restore();
}

/** The model: a heavy ink box, "the model" in display type and its name in mono under it. */
export function modelBox(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, a: number, o: { title?: string; sub?: string; hot?: number; size?: number } = {}) {
  if (a <= 0.003) return;
  const s = o.size ?? 34;
  c.save();
  c.globalAlpha = a;
  c.fillStyle = blueA(0.45);
  rrect(c, x + 10, y + 12, w, h, 18); c.fill();
  c.fillStyle = PAPER;
  rrect(c, x, y, w, h, 18); c.fill();
  if ((o.hot ?? 0) > 0.01) { c.globalAlpha = a * (o.hot ?? 0); c.fillStyle = pinkA(0.4); rrect(c, x, y, w, h, 18); c.fill(); c.globalAlpha = a; }
  c.lineWidth = 5; c.strokeStyle = INK;
  rrect(c, x, y, w, h, 18); c.stroke();
  c.fillStyle = INK;
  c.textBaseline = 'alphabetic';
  c.font = font(F.archivo(100, 800), s);
  c.fillText(o.title ?? 'the model', x + 26, y + s * 1.25);
  if (o.sub) { c.font = font(F.mono(500), s * 0.52); c.fillText(o.sub, x + 28, y + s * 1.25 + s * 0.8); }
  c.restore();
}

/** A number slip: a small blue-tinted card with a number (the model's input). */
export function slip(c: CanvasRenderingContext2D, cx: number, cy: number, w: number, h: number, text: string, a: number, o: { fs?: number; hot?: number } = {}) {
  if (a <= 0.003) return;
  c.save();
  c.globalAlpha = a;
  c.fillStyle = blueA(0.5);
  rrect(c, cx - w / 2 + 5, cy - h / 2 + 6, w, h, 8); c.fill();
  c.fillStyle = PAPER;
  rrect(c, cx - w / 2, cy - h / 2, w, h, 8); c.fill();
  if ((o.hot ?? 0) > 0.01) { c.fillStyle = pinkA(0.5 * (o.hot ?? 0)); rrect(c, cx - w / 2, cy - h / 2, w, h, 8); c.fill(); }
  c.lineWidth = 3; c.strokeStyle = BLUE;
  rrect(c, cx - w / 2, cy - h / 2, w, h, 8); c.stroke();
  let fs = o.fs ?? h * 0.5;
  c.font = font(F.mono(700), fs);
  const tw = c.measureText(text).width;
  if (tw > w * 0.84) { fs *= (w * 0.84) / tw; c.font = font(F.mono(700), fs); }
  c.fillStyle = BLUE;
  c.textAlign = 'center'; c.textBaseline = 'middle';
  c.fillText(text, cx, cy + fs * 0.04);
  c.restore();
}

/** A bracket over [x0, x1] at y (opening downward), drawing in from the middle. */
export function bracket(c: CanvasRenderingContext2D, t: number, t0: number, x0: number, x1: number, y: number, o: { h?: number; color?: string; lw?: number; up?: boolean; d?: number } = {}) {
  const p = k(t, t0, o.d ?? 0.45);
  if (p <= 0) return;
  const h = (o.h ?? 14) * (o.up ? -1 : 1), m = (x0 + x1) / 2, hw = ((x1 - x0) / 2) * p;
  c.save();
  c.strokeStyle = o.color ?? PINK; c.lineWidth = o.lw ?? 4; c.lineJoin = 'round';
  c.beginPath();
  c.moveTo(m - hw, y + h); c.lineTo(m - hw, y); c.lineTo(m + hw, y); c.lineTo(m + hw, y + h);
  c.stroke();
  c.restore();
}

/** 10 x 10 answers: cells fill in order over `d` s from t0, coloured per answer. */
export function tally(c: CanvasRenderingContext2D, t: number, t0: number, x: number, y: number, cell: number, cols: string[], o: { d?: number; a?: number } = {}) {
  const d = o.d ?? 1.4, gap = cell * 0.22;
  c.save();
  c.globalAlpha = o.a ?? 1;
  for (let i = 0; i < cols.length; i++) {
    const r = Math.floor(i / 10), q = i % 10;
    const p = clamp((t - t0 - (i / cols.length) * d) / 0.3);
    const xx = x + q * (cell + gap), yy = y + r * (cell + gap);
    c.lineWidth = 1.5; c.strokeStyle = inkA(0.5);
    c.strokeRect(xx, yy, cell, cell);
    if (p <= 0) continue;
    c.globalAlpha = (o.a ?? 1) * p;
    c.fillStyle = cols[i]!;
    c.fillRect(xx, yy, cell, cell);
    c.globalAlpha = o.a ?? 1;
  }
  c.restore();
}
