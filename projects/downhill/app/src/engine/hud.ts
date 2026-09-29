// Global overlay, drawn on every frame: the printer's crop and registration marks, the course marker
// ("ML, SLOWLY 1/5", top left) and the sung line (top centre, on a paper strip). Everything is solid ink
// (small type must never halftone). ML, slowly: no concept-word slams; the sung line stays up while it is
// the current line and moves off (slides and fades over 0.45 s) when the next one comes, so the overlay never
// pops. In instrumental gaps it leaves the picture alone.
import { Layer2D, W, H } from './gl';
import { rgba } from './palette';
import { F, font } from './type';
import type { Model } from './model';
import type { Story } from '../story';
import type { Lyrics, Line } from './lyrics';
import type { AudioData } from './audio';
import { clamp, ease } from './util';

export interface HudState {
  /** 0..1 overall opacity (scenes lower it with post.hud). */
  opacity: number;
  kinetic: number;
  lyric: number;
}

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
/** How long a line change takes (s), and how long a line may stay up after it ends before an instrumental clears it. */
const SWAP = H > W ? 0.7 : 0.45, HOLD = 2.2, CLEAR = 0.7;

/** A scene's small type (labels, captions), drawn onto the overlay so it prints crisp, never halftoned. */
let overlay: HTMLCanvasElement | null = null;
export const setOverlay = (cv: HTMLCanvasElement | null) => { overlay = cv; };

export class Hud {
  layer = new Layer2D();
  constructor(public model: Model | null, public story: Story | null, public lyrics: Lyrics, public audio?: AudioData) {}

  draw(t: number, st: HudState) {
    const L = this.layer;
    L.clear();
    const c = L.ctx;
    if (st.opacity <= 0.001) return L.upload();
    c.globalAlpha = st.opacity > 0.5 ? 1 : st.opacity * 2;
    this.marks(c);
    if (overlay) c.drawImage(overlay, 0, 0, W, H);
    this.marker(c);
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
    [PINK, BLUE, INK].forEach((col, i) => { c.fillStyle = col; c.fillRect(sx0 + i * (sw + 6), sy, sw, sw); });
    c.restore();
  }

  /** The course marker: a small solid-ink label, top left. */
  private marker(c: CanvasRenderingContext2D) {
    const portrait = H > W, fs = portrait ? 18 : 16, pad = 52;
    c.save();
    c.font = font(F.mono(700), fs);
    c.letterSpacing = '2px';
    const txt = 'ML, SLOWLY', num = '1/5';
    const w1 = c.measureText(txt).width, w2 = c.measureText(num).width;
    const x = pad, y = pad, h = fs * 1.9, gap = fs * 1.2;
    c.fillStyle = INK;
    c.fillRect(x, y, w1 + w2 + gap + fs * 1.6, h);
    c.fillStyle = PAPER;
    c.textBaseline = 'middle';
    c.fillText(txt, x + fs * 0.8, y + h / 2 + 1);
    c.fillText(num, x + fs * 0.8 + w1 + gap, y + h / 2 + 1);
    c.restore();
  }

  /** Which lines are up at t and how (a line slides up and out while the next slides in under it). */
  private lyricLines(c: CanvasRenderingContext2D, t: number) {
    const ls = this.lyrics.lines;
    let i = -1;
    for (let k = 0; k < ls.length; k++) if (ls[k]!.start - SWAP <= t) i = k;
    if (i < 0) return;
    const cur = ls[i]!, prev = ls[i - 1];
    // the incoming slide: over SWAP before the line starts
    const u = ease.inOutCubic(clamp((t - (cur.start - SWAP)) / SWAP));
    // an instrumental gap after the line: clear it
    const next = ls[i + 1];
    const gapClear = next && next.start - cur.end > HOLD + CLEAR + SWAP ? clamp((t - (cur.end + HOLD)) / CLEAR) : !next ? clamp((t - (cur.end + HOLD)) / CLEAR) : 0;
    const prevVisible = prev && !(cur.start - prev.end > HOLD + CLEAR + SWAP);
    if (prevVisible && u < 1) this.lyricLine(c, t, prev!, -34 * u, 1 - u);
    if (gapClear < 1) this.lyricLine(c, t, cur, 34 * (1 - u) - 34 * gapClear, u * (1 - gapClear));
  }

  /** One sung line, top centre, on a paper strip: sung words in pink, the rest in ink. */
  private lyricLine(c: CanvasRenderingContext2D, t: number, l: Line, dy: number, alpha: number) {
    if (alpha <= 0.002) return;
    const portrait = H > W, size = portrait ? 34 : 30, maxW = W * (portrait ? 0.84 : 0.6);
    c.save();
    c.globalAlpha *= alpha;
    c.font = font(F.archivo(100, 700), size);
    c.textBaseline = 'alphabetic';
    const rows: (typeof l.words)[] = [[]];
    let wsum = 0;
    const space = c.measureText(' ').width;
    for (const w of l.words) {
      const ww = c.measureText(w.w).width;
      if (wsum + ww > maxW && rows[rows.length - 1]!.length) { rows.push([]); wsum = 0; }
      rows[rows.length - 1]!.push(w);
      wsum += ww + space;
    }
    const y0 = (portrait ? 168 : 112) + dy;
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
