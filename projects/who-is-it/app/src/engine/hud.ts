// Global overlay, drawn on every frame after the print (crisp, solid ink): printer's crop and registration
// marks, the course marker ("ML, SLOWLY  4/5", top left) and the sung line on a paper strip (top centre),
// which beginners read along. The course rules: no concept-word slams; nothing pops in within a frame, so
// each sung line slides in (0.35 s) and stays up until the next one replaces it.
import { Layer2D, W, H } from './gl';
import { rgba } from './palette';
import { F, font } from './type';
import type { Lyrics, Line } from './lyrics';
import { clamp, ease } from './util';

export interface HudState {
  /** 0..1 overall opacity (scenes lower it with post.hud). */
  opacity: number;
  kinetic: number;
  lyric: number;
  caption?: string;
  captionA?: number;
}

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
/** The course marker: which episode of "ML, slowly" this is. */
export const MARKER = 'ML, SLOWLY  4/5';
const FADE = 0.35;   // a line slides in over this long, starting this long before its first word
const HOLD = 3.0;    // a line stays up this long after its last word when no line follows

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
    this.marker(c);
    if (st.lyric > 0.5) this.lyricLines(c, t);
    if (st.caption && (st.captionA ?? 0) > 0.003) this.caption(c, st.caption, st.captionA!);
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

  /** "ML, SLOWLY  4/5": a small solid-ink label, top left. */
  private marker(c: CanvasRenderingContext2D) {
    const portrait = H > W, fs = portrait ? 19 : 16, x = 58, y = portrait ? 64 : 50;
    c.save();
    c.font = font(F.mono(700), fs);
    c.letterSpacing = '2px';
    const w = c.measureText(MARKER).width;
    c.fillStyle = INK;
    c.fillRect(x - 10, y - fs * 1.05, w + 18, fs * 1.6);
    c.fillStyle = PAPER;
    c.textBaseline = 'alphabetic';
    c.fillText(MARKER, x, y + fs * 0.28);
    c.restore();
  }

  /** The source of what is on screen, bottom left, small solid ink. */
  private caption(c: CanvasRenderingContext2D, txt: string, a: number) {
    const portrait = H > W;
    c.save();
    c.globalAlpha *= clamp(a);
    c.font = font(F.mono(600), portrait ? 18 : 16);
    c.letterSpacing = '1.5px';
    c.fillStyle = INK;
    c.textBaseline = 'alphabetic';
    c.fillText(txt, 58, portrait ? H - 72 : H - 50);
    c.restore();
  }

  /** The line being sung (and the one it replaces, sliding out), top centre, on a paper strip. */
  private lyricLines(c: CanvasRenderingContext2D, t: number) {
    const ls = this.lyrics.lines;
    let k = -1;
    for (let i = 0; i < ls.length; i++) if (ls[i]!.start - FADE <= t) k = i;
    if (k < 0) return;
    const cur = ls[k]!, prev = ls[k - 1], next = ls[k + 1];
    const until = (l: Line, nx?: Line) => (nx ? Math.min(nx.start - FADE, l.end + HOLD) : l.end + HOLD);
    const inK = ease.inOutCubic(clamp((t - (cur.start - FADE)) / FADE));
    const outK = ease.inOutCubic(clamp((t - until(cur, next)) / FADE));
    if (prev && inK < 1 && t < until(prev, cur) + FADE) this.lyricLine(c, t, prev, 1 - inK, -1);
    if (outK < 1) this.lyricLine(c, t, cur, inK * (1 - outK), outK > 0 ? -1 : 1);
  }

  /** One line: sung words in pink, the rest in ink. `a` 0..1 visibility, `dir` where it slides from/to. */
  private lyricLine(c: CanvasRenderingContext2D, t: number, l: Line, a: number, dir: number) {
    if (a <= 0.004) return;
    const portrait = H > W, size = portrait ? 34 : 30, maxW = W * (portrait ? 0.84 : 0.6);
    c.save();
    c.globalAlpha *= a;
    c.translate(0, dir * (1 - a) * 18);
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
    const y0 = portrait ? 170 : 104;
    const widest = Math.max(...rows.map((row) => row.reduce((s, w) => s + c.measureText(w.w).width, 0) + space * (row.length - 1)));
    c.fillStyle = PAPER;
    c.fillRect(W / 2 - widest / 2 - 18, y0 - size * 1.05, widest + 36, size * 1.3 * rows.length + size * 0.3);
    c.fillStyle = INK;
    c.fillRect(W / 2 - widest / 2 - 18, y0 - size * 1.05, widest + 36, 2);
    rows.forEach((row, ri) => {
      const total = row.reduce((s, w) => s + c.measureText(w.w).width, 0) + space * (row.length - 1);
      let x = W / 2 - total / 2;
      for (const w of row) {
        c.fillStyle = t >= w.start ? PINK : INK;
        c.fillText(w.w, x, y0 + ri * size * 1.3);
        x += c.measureText(w.w).width + space;
      }
    });
    c.restore();
  }
}
