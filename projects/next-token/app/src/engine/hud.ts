// Global overlay, drawn on every frame so no shot is ever bare: printer's crop and registration marks,
// and the model's live readout from the story clock (story.ts) and the recorded run (data/llm.json):
// model tag, context position, KV cache size, measured speed, the floor the token in flight has reached,
// the next-token odds, and the answer so far. Everything is solid ink (small type must never halftone).
import { Layer2D, W, H } from './gl';
import { rgba } from './palette';
import { F, font } from './type';
import { showTok, type Llm } from './llm';
import type { Story } from '../story';
import type { Lyrics } from './lyrics';
import { SLAMS } from '../words';
import { clamp, ease, hash } from './util';

export interface HudState {
  /** 0..1 overall opacity (scenes lower it with post.hud). */
  opacity: number;
  kinetic: number;
  lyric: number;
}

interface SlamAt { t0: number; t1: number; show: string; slot: number }

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

export class Hud {
  layer = new Layer2D();
  slams: SlamAt[] = [];
  constructor(public llm: Llm | null, public story: Story | null, public lyrics: Lyrics) {
    const found: { t: number; show: string }[] = [];
    for (const sl of SLAMS) {
      try {
        const l = lyrics.get(sl.line, sl.nth ?? 0);
        const w = l.words.find((x) => x.w.toLowerCase().replace(/[^a-z0-9']/g, '').startsWith(sl.word));
        if (w) found.push({ t: w.start, show: sl.show });
      } catch { /* line not in this take's alignment */ }
    }
    found.sort((a, b) => a.t - b.t);
    this.slams = found.map((x, i) => ({ t0: x.t, t1: Math.min(x.t + 0.62, found[i + 1]?.t ?? Infinity), show: x.show, slot: Math.floor(hash(i, 17) * 4) }));
  }

  draw(t: number, st: HudState) {
    const L = this.layer;
    L.clear();
    const c = L.ctx;
    if (st.opacity <= 0.001) return L.upload();
    c.globalAlpha = st.opacity > 0.5 ? 1 : st.opacity * 2; // ink is on or off: fade by dropping out, never by tint
    this.marks(c);
    if (this.llm && this.story) this.readout(c, t);
    if (st.lyric > 0.5) this.lyricLine(c, t);
    if (st.kinetic > 0.5) this.slam(c, t);
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
    // registration target, top centre
    const rx = W / 2, ry = m;
    c.beginPath(); c.arc(rx, ry, 7, 0, Math.PI * 2); c.moveTo(rx - 13, ry); c.lineTo(rx + 13, ry); c.moveTo(rx, ry - 13); c.lineTo(rx, ry + 13); c.stroke();
    // ink swatches, bottom centre
    const sw = 12, sx0 = W / 2 - (3 * sw + 2 * 6) / 2, sy = H - m - sw / 2;
    [PINK, BLUE, INK].forEach((col, i) => { c.fillStyle = col; c.fillRect(sx0 + i * (sw + 6), sy, sw, sw); });
    c.restore();
  }

  private readout(c: CanvasRenderingContext2D, t: number) {
    const llm = this.llm!, s = this.story!.at(t), cfg = llm.cfg, portrait = H > W;
    const pad = 58, fs = portrait ? 17 : 15;
    const mono = (w: number, px = fs) => font(F.mono(w), px);
    c.save();
    c.textBaseline = 'alphabetic';
    c.letterSpacing = '1.5px';

    // model tag (top left)
    c.fillStyle = INK;
    c.font = mono(600);
    c.fillText('LLAMA 3.1 8B INSTRUCT', pad, pad + 8);
    c.font = mono(400);
    c.fillText(`BF16 · ${cfg.layers} LAYERS · ${(cfg.params / 1e9).toFixed(1)}B PARAMS`, pad, pad + 8 + fs * 1.5);

    // position, KV cache, speed (top right)
    const kv = (s.pos * cfg.kv_bytes_per_token) / 2 ** 20;
    const sp = llm.d.speed_tok_s;
    const speed = s.mode === 'batch' ? `${Math.round(sp['100'] ?? 0)} TOK/S ×100` : `${(sp['1'] ?? 0).toFixed(0)} TOK/S`;
    const right = [
      `POS ${String(s.pos).padStart(4, '0')}`,
      `KV ${kv.toFixed(1)} MB`,
      `${speed} · 3090`,
    ];
    c.textAlign = 'right';
    right.forEach((txt, i) => { c.fillStyle = i === 0 ? PINK : INK; c.font = mono(i === 0 ? 700 : 500); c.fillText(txt, W - pad, pad + 8 + i * fs * 1.5); });
    c.textAlign = 'left';

    // the tower's floors (left edge): 33 ticks, embedding at the bottom; the token in flight lights its floor
    const x0 = 30, y1 = H * (portrait ? 0.3 : 0.24), y0 = H * (portrait ? 0.7 : 0.76);
    const n = cfg.layers + 1;
    c.fillStyle = INK;
    for (let i = 0; i < n; i++) {
      const y = y0 + (y1 - y0) * (i / (n - 1));
      const on = i === s.layer && s.mode !== 'prefill';
      const passed = i < s.layer;
      c.fillStyle = on ? PINK : passed ? BLUE : INK;
      c.fillRect(x0, y - (on ? 2 : 1), on ? 26 : passed ? 12 : 7, on ? 4 : 2);
    }
    c.fillStyle = PINK;
    c.font = mono(700);
    const ly = y0 + (y1 - y0) * (s.layer / (n - 1));
    c.fillText(s.layer === 0 ? 'EMB' : `L${s.layer}`, x0 + 32, ly + fs * 0.35);

    // next-token odds (bottom right): the nucleus the model samples from
    const st = llm.steps[s.step]!;
    const bw = portrait ? 300 : 260, bx = W - pad - bw, rows = 5, rh = fs * 1.55;
    const by = H - pad - (portrait ? 150 : 70) - rows * rh;
    c.fillStyle = INK;
    c.font = mono(600);
    c.fillText(s.mode === 'prefill' ? `PREFILL · ${llm.d.prompt.length} TOKENS` : `NEXT TOKEN · T ${llm.d.sampling.temperature}`, bx, by - rh * 0.6);
    const reveal = s.mode === 'prefill' ? 0 : ease.outCubic(clamp((s.climb - 0.55) / 0.35));
    const justPicked = s.since < 0.7 && s.emitted > 0;
    const shown = justPicked ? llm.steps[s.emitted - 1]! : st;
    if (s.mode !== 'prefill') shown.nucleus.slice(0, rows).forEach(([tok, p], i) => {
      const y = by + i * rh;
      const k = justPicked ? 1 : reveal;
      const picked = justPicked && i === shown.pick;
      c.fillStyle = picked ? PINK : BLUE;
      c.fillRect(bx + 118, y - fs * 0.75, Math.max(2, (bw - 170) * p * k), fs * 0.85);
      c.fillStyle = picked ? PINK : INK;
      c.font = mono(picked ? 700 : 500);
      c.fillText(showTok(tok).slice(0, 11), bx, y);
      c.textAlign = 'right';
      c.fillText(k > 0.01 ? `${(p * 100 * k).toFixed(0)}%` : '··', W - pad, y);
      c.textAlign = 'left';
    });

    // the answer so far (bottom left): emitted tokens as boxed pieces, newest in pink
    const toks = this.story!.textAt(t);
    c.font = mono(600, fs + 2);
    let x = pad, y = H - pad - 4;
    const maxX = portrait ? W - pad : bx - 40;
    const lineH = fs * 2.2;
    if (portrait) y = H - pad - 4 - lineH;
    const boxes: { x: number; y: number; w: number; tok: string; last: boolean }[] = [];
    toks.forEach((tk, i) => {
      const label = showTok(tk);
      const w = c.measureText(label).width + 14;
      if (x + w > maxX) { x = pad; y += lineH; }
      boxes.push({ x, y, w, tok: label, last: i === toks.length - 1 });
      x += w + 6;
    });
    // keep the strip on screen: shift everything up if it wrapped past the bottom
    const over = boxes.length ? Math.max(0, boxes[boxes.length - 1]!.y - (H - pad - 4)) : 0;
    for (const b of boxes) {
      const yy = b.y - over, hot = b.last && s.since < 1.2;
      c.fillStyle = hot ? PINK : INK;
      c.fillRect(b.x, yy - fs - 5, b.w, fs + 13);
      c.fillStyle = PAPER;
      c.fillText(b.tok, b.x + 7, yy);
    }
    c.restore();
  }

  /** The line being sung, top centre: sung words in pink, the rest in ink. */
  private lyricLine(c: CanvasRenderingContext2D, t: number) {
    const l = this.lyrics.lineAt(t);
    if (!l) return;
    const portrait = H > W, size = portrait ? 30 : 26, maxW = W * (portrait ? 0.84 : 0.56);
    c.save();
    c.font = font(F.archivo(100, 700), size);
    c.textBaseline = 'alphabetic';
    // wrap into rows that fit, then centre each row
    const rows: typeof l.words[] = [[]];
    let wsum = 0;
    const space = c.measureText(' ').width;
    for (const w of l.words) {
      const ww = c.measureText(w.w).width;
      if (wsum + ww > maxW && rows[rows.length - 1]!.length) { rows.push([]); wsum = 0; }
      rows[rows.length - 1]!.push(w);
      wsum += ww + space;
    }
    const y0 = (portrait ? 150 : 96);
    // a paper strip behind the line (a knockout label) so it reads over anything
    const widest = Math.max(...rows.map((row) => row.reduce((a, w) => a + c.measureText(w.w).width, 0) + space * (row.length - 1)));
    c.fillStyle = PAPER;
    c.fillRect(W / 2 - widest / 2 - 16, y0 - size * 1.02, widest + 32, size * 1.25 * rows.length + size * 0.32);
    c.fillStyle = INK;
    c.fillRect(W / 2 - widest / 2 - 16, y0 - size * 1.02, widest + 32, 2);
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

  /** The concept word being sung, slammed in huge: pink over a misregistered blue, overprinting purple. */
  private slam(c: CanvasRenderingContext2D, t: number) {
    const s = this.slams.find((x) => t >= x.t0 && t < x.t1);
    if (!s) return;
    const k = ease.outExpo(clamp((t - s.t0) / 0.07));
    const portrait = H > W;
    c.save();
    const fam = F.archivo(125, 900);
    let size = portrait ? 200 : 250;
    c.font = font(fam, size);
    const maxW = W * (portrait ? 0.9 : 0.8);
    const w0 = c.measureText(s.show).width;
    if (w0 > maxW) size *= maxW / w0;
    const slots = portrait ? [[0.5, 0.42], [0.5, 0.58], [0.5, 0.36], [0.5, 0.64]] : [[0.5, 0.56], [0.36, 0.44], [0.62, 0.66], [0.5, 0.4]];
    const [sx, sy] = slots[s.slot]!;
    const scale = 1.3 - 0.3 * k;
    c.translate(W * sx!, H * sy!);
    c.scale(scale, scale);
    c.font = font(fam, size);
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillStyle = BLUE;
    c.fillText(s.show, size * 0.035, size * 0.03);
    c.globalCompositeOperation = 'multiply';
    c.fillStyle = PINK;
    c.fillText(s.show, 0, 0);
    c.restore();
  }
}
