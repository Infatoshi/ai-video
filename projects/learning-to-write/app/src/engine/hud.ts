// Global overlay, drawn on every frame: printer's crop and registration marks, the course marker
// ("ML, SLOWLY 5/5", solid ink, top left), the training clock (top right: the step the video is showing), and
// the sung line (top centre on a paper strip; it stays up between lines so beginners can read along, and
// crossfades when the next line starts: nothing pops in within a frame). No concept-word slams in the course.
// Everything is solid ink (small type must never halftone).
import { Layer2D, W, H } from './gl';
import { rgba } from './palette';
import { F, font } from './type';
import type { Run } from './run';
import type { Clock } from '../clock';
import type { Lyrics, Line } from './lyrics';
import { clamp } from './util';

export interface HudState {
  /** 0..1 overall opacity (scenes lower it with post.hud). */
  opacity: number;
  kinetic: number;
  lyric: number;
}

const PINK = rgba('pink'), INK = rgba('ink'), PAPER = rgba('paper');
/** Seconds a line takes to fade in / out, and how long it stays after it was sung when no line follows soon. */
const FADE = 0.35, HOLD = 5;

export class Hud {
  layer = new Layer2D();
  constructor(public run: Run | null, public clock: Clock | null, public lyrics: Lyrics) {}

  draw(t: number, st: HudState) {
    const L = this.layer;
    L.clear();
    const c = L.ctx;
    if (st.opacity <= 0.001) return L.upload();
    c.globalAlpha = clamp(st.opacity);
    this.marks(c);
    this.marker(c);
    if (this.run && this.clock) this.readout(c, t);
    if (st.lyric > 0.5) this.lyricLines(c, t);
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
    [PINK, rgba('blue'), INK].forEach((col, i) => { c.fillStyle = col; c.fillRect(sx0 + i * (sw + 6), sy, sw, sw); });
    c.restore();
  }

  /** The course marker, top left: solid ink block, paper type. */
  private marker(c: CanvasRenderingContext2D) {
    const portrait = H > W, pad = 58, fs = portrait ? 19 : 16;
    c.save();
    c.font = font(F.mono(700), fs);
    c.letterSpacing = '2px';
    const txt = 'ML, SLOWLY  5/5';
    const w = c.measureText(txt).width;
    c.fillStyle = INK;
    c.fillRect(pad - 10, pad - fs - 4, w + 22, fs + 16);
    c.fillStyle = PAPER;
    c.textBaseline = 'alphabetic';
    c.fillText(txt, pad, pad + 4);
    c.restore();
  }

  /** Top right: the training step the video is showing, and what it is. */
  private readout(c: CanvasRenderingContext2D, t: number) {
    const run = this.run!, step = Math.round(this.clock!.step(t)), portrait = H > W;
    const pad = 58, fs = portrait ? 19 : 16;
    c.save();
    c.textAlign = 'right';
    c.textBaseline = 'alphabetic';
    c.letterSpacing = '1.5px';
    c.fillStyle = INK;
    c.font = font(F.mono(700), fs);
    c.fillText(`STEP ${step.toLocaleString('en-US')} / ${run.maxStep.toLocaleString('en-US')}`, W - pad, pad + 4);
    c.font = font(F.mono(400), fs * 0.82);
    c.fillText('A SMALL GPT LEARNING SHAKESPEARE', W - pad, pad + 4 + fs * 1.45);
    c.restore();
  }

  /** The sung line, top centre: sung words in pink, the rest in ink. Crossfades line to line. */
  private lyricLines(c: CanvasRenderingContext2D, t: number) {
    const ls = this.lyrics.lines;
    // the current line: the last one started (with its fade-in beginning FADE before its first word)
    let cur = -1;
    for (let i = 0; i < ls.length; i++) if (ls[i]!.start - FADE <= t) cur = i;
    if (cur < 0) return;
    const a = ls[cur]!, next = ls[cur + 1];
    const fin = clamp((t - (a.start - FADE)) / FADE);
    // it fades out when the next line comes in, or HOLD s after it ends when nothing follows soon
    const outAt = next && next.start - FADE < a.end + HOLD ? next.start - FADE : a.end + HOLD;
    const fout = 1 - clamp((t - outAt) / FADE);
    const prev = ls[cur - 1];
    if (prev && fin < 1 && prev.end + HOLD > a.start - FADE) this.lyricLine(c, prev, t, 1 - fin);
    this.lyricLine(c, a, t, Math.min(fin, fout));
  }

  private lyricLine(c: CanvasRenderingContext2D, l: Line, t: number, alpha: number) {
    if (alpha <= 0.002) return;
    const portrait = H > W, size = portrait ? 34 : 30, maxW = W * (portrait ? 0.84 : 0.6);
    c.save();
    c.globalAlpha *= alpha;
    c.font = font(F.archivo(100, 700), size);
    c.textBaseline = 'alphabetic';
    const rows: typeof l.words[] = [[]];
    let wsum = 0;
    const space = c.measureText(' ').width;
    for (const w of l.words) {
      const ww = c.measureText(w.w).width;
      if (wsum + ww > maxW && rows[rows.length - 1]!.length) { rows.push([]); wsum = 0; }
      rows[rows.length - 1]!.push(w);
      wsum += ww + space;
    }
    const y0 = portrait ? 168 : 112;
    const widest = Math.max(...rows.map((row) => row.reduce((a, w) => a + c.measureText(w.w).width, 0) + space * (row.length - 1)));
    c.fillStyle = PAPER;
    c.fillRect(W / 2 - widest / 2 - 18, y0 - size * 1.05, widest + 36, size * 1.25 * rows.length + size * 0.36);
    c.fillStyle = INK;
    c.fillRect(W / 2 - widest / 2 - 18, y0 - size * 1.05, widest + 36, 2);
    rows.forEach((row, ri) => {
      const total = row.reduce((a, w) => a + c.measureText(w.w).width, 0) + space * (row.length - 1);
      let x = W / 2 - total / 2;
      for (const w of row) {
        c.fillStyle = t >= w.start ? PINK : INK;
        c.fillText(w.w, x, y0 + ri * size * 1.25);
        x += c.measureText(w.w).width + space;
      }
    });
    c.restore();
  }
}
