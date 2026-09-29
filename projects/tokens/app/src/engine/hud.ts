// Global overlay for the course ("ML, slowly"), drawn on every frame, crisp and solid (never halftoned):
// the printer's crop and registration marks, the series marker top left ("ML, SLOWLY 2/5"), the sung line
// top centre (beginners read along; it fades between lines and never pops), and a small source line bottom
// left naming where the number on screen comes from (set by the scene through `hudInfo`). No word slams.
import { Layer2D, W, H } from './gl';
import { rgba } from './palette';
import { F, font } from './type';
import type { Lyrics, Line } from './lyrics';
import { clamp, smoothstep } from './util';

export interface HudState {
  /** 0..1 overall opacity (scenes lower it with post.hud). */
  opacity: number;
  kinetic: number;
  lyric: number;
}

/** Written by the scene each frame (a pure function of the frame's time): the source line and its opacity. */
export const hudInfo = { source: '', sourceA: 0, marker: 1 };

export const EPISODE = { n: 2, of: 5 };

const PINK = rgba('pink'), INK = rgba('ink'), PAPER = rgba('paper'), BLUE = rgba('blue');
const mix = (a: number[], b: number[], k: number) => `rgb(${a.map((x, i) => Math.round(x + (b[i]! - x) * k)).join(',')})`;
const INK3 = [0x23, 0x1f, 0x20], PINK3 = [0xff, 0x48, 0xb0];

/** Display form of a sung word: letters sung one by one are written as one word. */
const shown = (w: string) => w.replace(/\bA-I\b/g, 'AI').replace(/\bS-T-R\b/g, 'STR').replace(/\bA-W\b/g, 'AW');

export class Hud {
  layer = new Layer2D();
  constructor(public lyrics: Lyrics) {}

  draw(t: number, st: HudState) {
    const L = this.layer;
    L.clear();
    const c = L.ctx;
    if (st.opacity <= 0.001) return L.upload();
    c.globalAlpha = clamp(st.opacity);
    this.marks(c);
    if (hudInfo.marker > 0.01) this.marker(c);
    if (st.lyric > 0.5) this.lyricLines(c, t);
    if (hudInfo.sourceA > 0.01 && hudInfo.source) this.sourceLine(c);
    return L.upload();
  }

  /** Crop marks at the corners, a registration target and the three ink swatches (the sheet's print marks). */
  private marks(c: CanvasRenderingContext2D) {
    c.save();
    c.strokeStyle = INK;
    c.lineWidth = 1.5;
    const m = 22, l = 26;
    c.beginPath();
    for (const [x, y, sx, sy] of [[m, m, 1, 1], [W - m, m, -1, 1], [m, H - m, 1, -1], [W - m, H - m, -1, -1]] as const) {
      c.moveTo(x + sx * l, y); c.lineTo(x - sx * 8, y); c.moveTo(x, y + sy * l); c.lineTo(x, y - sy * 8);
    }
    c.stroke();
    const rx = W / 2, ry = m;
    c.beginPath(); c.arc(rx, ry, 7, 0, Math.PI * 2); c.moveTo(rx - 13, ry); c.lineTo(rx + 13, ry); c.moveTo(rx, ry - 13); c.lineTo(rx, ry + 13); c.stroke();
    const sw = 12, sx0 = W / 2 - (3 * sw + 2 * 6) / 2, sy = H - m - sw / 2;
    [PINK, BLUE, INK].forEach((col, i) => { c.fillStyle = col; c.fillRect(sx0 + i * (sw + 6), sy, sw, sw); });
    c.restore();
  }

  /** The series marker: a small solid-ink label, top left. */
  private marker(c: CanvasRenderingContext2D) {
    const portrait = H > W, pad = 58, fs = portrait ? 18 : 16;
    c.save();
    c.globalAlpha *= hudInfo.marker;
    c.font = font(F.mono(700), fs);
    c.letterSpacing = '2px';
    const a = 'ML, SLOWLY', b = `${EPISODE.n}/${EPISODE.of}`;
    const wa = c.measureText(a).width, wb = c.measureText(b).width;
    const h = fs * 1.9, x = pad - 12, y = pad - fs * 0.9;
    c.fillStyle = INK;
    c.fillRect(x, y, wa + wb + 50, h);
    c.fillStyle = PAPER;
    c.textBaseline = 'middle';
    c.fillText(a, x + 12, y + h / 2 + 1);
    c.fillStyle = PINK;
    c.fillText(b, x + 12 + wa + 22, y + h / 2 + 1);
    c.restore();
  }

  /**
   * The sung line, top centre, on a paper strip. A line shows from 0.25 s before its first word and holds
   * until the next line comes in (crossfade 0.35 s) or 2.5 s after its end (then fades over 0.5 s). Sung
   * words ease from ink to pink over 0.15 s. Nothing pops in within a frame.
   */
  private lyricLines(c: CanvasRenderingContext2D, t: number) {
    const ls = this.lyrics.lines, LEAD = 0.25, X = 0.35, HOLD = 2.5, OUT = 0.5;
    for (let i = 0; i < ls.length; i++) {
      const l = ls[i]!, nx = ls[i + 1];
      const a0 = l.start - LEAD;
      if (t < a0 - X || (nx && t > nx.start - LEAD + X)) continue;
      const fin = smoothstep(a0 - X, a0, t);
      const nextIn = nx ? smoothstep(nx.start - LEAD - X, nx.start - LEAD, t) : 0;
      const lateOut = 1 - smoothstep(l.end + HOLD, l.end + HOLD + OUT, t);
      // a line replaced by the next one hands over with a crossfade; one followed by a gap fades out on its own
      const gapBefore = nx ? nx.start - LEAD - (l.end + HOLD) : Infinity;
      const a = fin * (gapBefore > 0 ? lateOut : 1 - nextIn);
      if (a > 0.003) this.lyricLine(c, t, l, a);
    }
  }

  private lyricLine(c: CanvasRenderingContext2D, t: number, l: Line, alpha: number) {
    const portrait = H > W, size = portrait ? 34 : 30, maxW = W * (portrait ? 0.84 : 0.6);
    c.save();
    c.globalAlpha *= alpha;
    c.font = font(F.archivo(100, 700), size);
    c.textBaseline = 'alphabetic';
    const words = l.words.map((w) => ({ ...w, w: shown(w.w) }));
    const rows: typeof words[] = [[]];
    let wsum = 0;
    const space = c.measureText(' ').width;
    for (const w of words) {
      const ww = c.measureText(w.w).width;
      if (wsum + ww > maxW && rows[rows.length - 1]!.length) { rows.push([]); wsum = 0; }
      rows[rows.length - 1]!.push(w);
      wsum += ww + space;
    }
    const y0 = portrait ? 170 : 100;
    const widest = Math.max(...rows.map((row) => row.reduce((a, w) => a + c.measureText(w.w).width, 0) + space * (row.length - 1)));
    c.fillStyle = PAPER;
    c.fillRect(W / 2 - widest / 2 - 18, y0 - size * 1.05, widest + 36, size * 1.25 * rows.length + size * 0.36);
    c.fillStyle = INK;
    c.fillRect(W / 2 - widest / 2 - 18, y0 - size * 1.05, widest + 36, 2);
    rows.forEach((row, ri) => {
      const total = row.reduce((a, w) => a + c.measureText(w.w).width, 0) + space * (row.length - 1);
      let x = W / 2 - total / 2;
      for (const w of row) {
        c.fillStyle = mix(INK3, PINK3, smoothstep(w.start - 0.02, w.start + 0.13, t));
        c.fillText(w.w, x, y0 + ri * size * 1.25);
        x += c.measureText(w.w).width + space;
      }
    });
    c.restore();
  }

  /** Where the number on screen comes from: small mono, bottom left, above the crop mark. */
  private sourceLine(c: CanvasRenderingContext2D) {
    const portrait = H > W, fs = portrait ? 17 : 15, pad = 58;
    c.save();
    c.globalAlpha *= hudInfo.sourceA;
    c.font = font(F.mono(500), fs);
    c.letterSpacing = '0.5px';
    c.fillStyle = INK;
    c.textBaseline = 'alphabetic';
    // wrap to the frame (portrait lines are long)
    const maxW = W - 2 * pad, lines: string[] = [];
    for (const para of hudInfo.source.split('\n')) {
      let cur = '';
      for (const w of para.split(' ')) {
        const next = cur ? `${cur} ${w}` : w;
        if (c.measureText(next).width > maxW && cur) { lines.push(cur); cur = w; } else cur = next;
      }
      lines.push(cur);
    }
    lines.forEach((s, i) => c.fillText(s, pad, H - pad + 6 - (lines.length - 1 - i) * fs * 1.45));
    c.restore();
  }
}
