// Global overlay (ML, slowly #3), drawn after the print so it stays crisp: printer's crop and registration marks,
// the course marker top left ("ML, SLOWLY  3/5"), the source tag bottom left, and the sung line top centre (beginners
// read along). Nothing here pops: the sung line fades out and the next fades in (>= 0.2 s each), sung words go from
// a light tint to solid ink. No concept-word slams (the course's pace rules).
import { Layer2D, W, H } from './gl';
import { rgba } from './palette';
import { F, font } from './type';
import type { Lyrics, Line } from './lyrics';
import type { AudioData } from './audio';
import { clamp } from './util';

export interface HudState {
  /** 0..1 overall opacity (scenes lower it with post.hud). */
  opacity: number;
  kinetic: number;
  lyric: number;
}

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
export const COURSE_TAG = 'ML, SLOWLY';
export const EPISODE = '3/5';
export const SOURCE_TAG = 'LLAMA 3.1 8B INSTRUCT · ITS INPUT EMBEDDING TABLE';

/** Opacity of lyric line i at time t: in over [start-0.1, start+0.25], out before the next line comes in. */
export function lineAlpha(lines: Line[], i: number, t: number): number {
  const l = lines[i]!, n = lines[i + 1];
  const inA = clamp((t - (l.start - 0.1)) / 0.35);
  // out: just before the next line (if it comes within 2.5 s of this one's end), else 1.4 s after the end
  const outAt = n && n.start - l.end < 2.5 ? n.start - 0.1 - 0.25 : l.end + 1.4;
  const outA = 1 - clamp((t - outAt) / (n && n.start - l.end < 2.5 ? 0.25 : 0.5));
  return Math.min(inA, outA);
}

/** Scene overlays drawn with the HUD (after the print, crisp): labels, numbers, arrows. Scenes register in init(). */
export const OVERLAYS: ((c: CanvasRenderingContext2D, t: number) => void)[] = [];

export class Hud {
  layer = new Layer2D();
  constructor(public lyrics: Lyrics, public audio: AudioData) {}

  draw(t: number, st: HudState) {
    const L = this.layer;
    L.clear();
    const c = L.ctx;
    if (st.opacity <= 0.001) return L.upload();
    c.globalAlpha = clamp(st.opacity);
    this.marks(c);
    for (const o of OVERLAYS) { c.save(); o(c, t); c.restore(); }
    this.tags(c);
    if (st.lyric > 0.001) this.lyricLine(c, t, st.lyric);
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

  /** The course marker (top left, a solid ink tag) and the data source (bottom left). */
  private tags(c: CanvasRenderingContext2D) {
    const portrait = H > W, pad = 58, fs = portrait ? 19 : 17;
    c.save();
    c.textBaseline = 'middle';
    c.letterSpacing = '2px';
    c.font = font(F.mono(700), fs);
    const a = COURSE_TAG, b = EPISODE;
    const wa = c.measureText(a).width, wb = c.measureText(b).width, gap = fs * 1.2, px = fs * 0.7, h = fs * 2;
    c.fillStyle = INK;
    c.fillRect(pad - px, pad - h / 2, wa + gap + wb + 2 * px, h);
    c.fillStyle = PAPER;
    c.fillText(a, pad, pad + 1);
    c.fillText(b, pad + wa + gap, pad + 1);
    c.font = font(F.mono(500), portrait ? 15 : 13);
    c.letterSpacing = '1.5px';
    c.fillStyle = INK;
    c.textBaseline = 'alphabetic';
    c.fillText(SOURCE_TAG, pad - px, H - pad + 6);
    c.restore();
  }

  /** The sung line(s), top centre on a paper strip: sung words in solid ink, the rest a light tint. */
  private lyricLine(c: CanvasRenderingContext2D, t: number, k: number) {
    const lines = this.lyrics.lines;
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i]!;
      if (t < l.start - 0.2 || t > l.end + 2.5) continue;
      const a = lineAlpha(lines, i, t) * k;
      if (a <= 0.002) continue;
      this.drawLine(c, l, t, a);
    }
  }

  private drawLine(c: CanvasRenderingContext2D, l: Line, t: number, alpha: number) {
    const portrait = H > W, size = portrait ? 40 : 34, maxW = W * (portrait ? 0.84 : 0.62);
    c.save();
    c.globalAlpha *= alpha;
    const base = c.globalAlpha;
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
    c.fillRect(W / 2 - widest / 2 - 20, y0 - size * 1.05, widest + 40, size * 1.25 * rows.length + size * 0.4);
    c.fillStyle = INK;
    c.fillRect(W / 2 - widest / 2 - 20, y0 - size * 1.05, widest + 40, 2);
    rows.forEach((row, ri) => {
      const total = row.reduce((s, w) => s + c.measureText(w.w).width, 0) + space * (row.length - 1);
      let x = W / 2 - total / 2;
      for (const w of row) {
        const sung = clamp((t - w.start) / 0.12);
        c.fillStyle = INK;
        c.globalAlpha = base * (0.38 + 0.62 * sung);
        c.fillText(w.w, x, y0 + ri * size * 1.25);
        x += c.measureText(w.w).width + space;
      }
    });
    c.restore();
  }
}
