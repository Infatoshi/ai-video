// 2D pieces of the attention scene: the 32-head grid (one real attention row per head), stamps and slab type.
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { showTok, type Llm } from '../engine/llm';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

export interface GridData {
  layer: number;
  /** Per head: the newest token's weights over the context. */
  rows: Float32Array[];
  /** Per head: the index of its strongest weight after the first token (the sink). */
  topNS: number[];
  /** Heads whose single strongest weight is on the first token. */
  sinkTop: number;
}

/** All heads of one layer at step s. */
export function headRows(llm: Llm, s: number, layer: number): GridData {
  const Hn = llm.cfg.heads;
  const rows: Float32Array[] = [], topNS: number[] = [];
  let sinkTop = 0;
  for (let h = 0; h < Hn; h++) {
    const r = llm.attn(s, layer, h);
    rows.push(r);
    let best = 1, arg = 0;
    for (let i = 1; i < r.length; i++) if (r[i]! > r[best]!) best = i;
    for (let i = 1; i < r.length; i++) if (r[i]! > r[arg]!) arg = i;
    if (r[0]! >= r[arg]!) sinkTop++;
    topNS.push(best);
  }
  return { layer, rows, topNS, sinkTop };
}

/**
 * The heads as a cols x rows grid of panels. Each panel: the head's weight on the first token (the sink) as a
 * black bar at true scale on the left, then a strip over the rest of the context scaled to its own peak (the
 * peak and anything over 15% in pink, the rest blue), the head number and its top word after the sink.
 * `appear(i)` 0..1 per panel (snap-ins); `hi` = a head to frame in pink; `ts` = type size.
 */
export function drawGrid(c: CanvasRenderingContext2D, g: GridData, toks: { t: string }[], x0: number, y0: number, gw: number, gh: number,
  cols: number, rows: number, appear: (i: number) => number, hi: number, ts = 24) {
  const pw = gw / cols, ph = gh / rows, m = 7;
  c.save();
  for (let h = 0; h < g.rows.length; h++) {
    const k = appear(h);
    if (k <= 0) continue;
    const cx = h % cols, cy = Math.floor(h / cols);
    const px = x0 + cx * pw + m, py = y0 + cy * ph + m + (1 - k) * 30, w = pw - 2 * m, hh = ph - 2 * m;
    c.fillStyle = PAPER;
    c.fillRect(px, py, w, hh);
    c.strokeStyle = h === hi ? PINK : INK;
    c.lineWidth = h === hi ? 7 : 3;
    c.strokeRect(px, py, w, hh);
    // head number
    c.fillStyle = h === hi ? PINK : INK;
    c.font = font(F.mono(700), Math.max(22, ts - 2));
    c.textBaseline = 'top';
    c.fillText(`H${h + 1}`, px + 8, py + 6);
    const r = g.rows[h]!, n = r.length;
    const top = py + ts + 14, base = py + hh - ts - 14, sh = Math.max(10, base - top);
    // the sink: black, true scale (the panel's strip height = 100%)
    const sinkW = Math.max(12, w * 0.07);
    const sk = r[0]! * k;
    c.fillStyle = INK;
    c.fillRect(px + 8, base - sk * sh, sinkW, Math.max(2, sk * sh));
    // the rest, scaled to the head's own peak
    const tn = g.topNS[h]!, peak = Math.max(1e-4, r[tn]!);
    const sx = px + 8 + sinkW + 8, sw = px + w - 8 - sx;
    const bw = sw / (n - 1);
    for (let i = 1; i < n; i++) {
      const v = (r[i]! / peak) * k;
      const bh = Math.max(2, v * sh);
      c.fillStyle = i === tn || r[i]! >= 0.15 ? PINK : BLUE;
      c.fillRect(sx + (i - 1) * bw, base - bh, Math.max(1.5, bw - 1), bh);
    }
    c.fillStyle = INK;
    c.fillRect(px + 8, base, w - 16, 2);
    // its top word after the sink
    let lab = showTok(toks[tn]?.t ?? '');
    let size = ts;
    c.font = font(F.mono(700), size);
    while (c.measureText(lab).width > w - 14 && size > 22) { size -= 1; c.font = font(F.mono(700), size); }
    while (c.measureText(lab).width > w - 14 && lab.length > 3) lab = lab.slice(0, -2) + '…';
    c.fillStyle = PINK;
    c.textBaseline = 'bottom';
    c.fillText(lab, px + 8, py + hh - 6);
  }
  c.restore();
}

/** A label printed as a solid block: ink box, paper type (or any two inks). */
export function drawStamp(c: CanvasRenderingContext2D, s: string, x: number, y: number, size: number, bg: string, fg: string) {
  c.save();
  c.font = font(F.mono(700), size);
  c.textBaseline = 'middle';
  const w = c.measureText(s).width + size * 0.8;
  c.fillStyle = bg;
  c.fillRect(x, y - size * 0.8, w, size * 1.6);
  c.fillStyle = fg;
  c.fillText(s, x + size * 0.4, y + 1);
  c.restore();
}

/** Big two-ink slab type: blue underprint, pink on top with multiply (prints purple where they overlap). */
export function drawSlab(c: CanvasRenderingContext2D, s: string, x: number, y: number, size: number, align: CanvasTextAlign = 'center') {
  c.save();
  c.font = font(F.archivo(125, 900), size);
  c.textAlign = align;
  c.textBaseline = 'middle';
  c.fillStyle = BLUE;
  c.fillText(s, x + size * 0.04, y + size * 0.035);
  c.globalCompositeOperation = 'multiply';
  c.fillStyle = PINK;
  c.fillText(s, x, y);
  c.restore();
}
