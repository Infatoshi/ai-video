// YouTube thumbnails (not part of the edit): three printed compositions, rendered with
// `render.ts stills --only thumbA,thumbB,thumbC --t 0.5,1.5,2.5` (timeline.ts adds the entries only then).
// Each has at most 3 big words and reads at 168x94: A "WHO'S IT?" over the real arc from "wide" to "street",
// B the 68.0% arc alone, C all 1,024 heads with the 4 that switch clearly in pink. Real data only (data/attn.json).
import type * as THREE from 'three';
import type { Frame, PostOverrides } from '../engine/scene';
import { W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { perFrame } from './_kit';
import Sentence from './sentence';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink');

export default class Thumb extends Sentence {
  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const r = this.ctx.renderer, v = this.ctx.params.v as 'A' | 'B' | 'C';
    this.wash.render(r, out, { seed: { A: 11, B: 23, C: 37 }[v], drift: 0, amt: v === 'C' ? 0.12 : 0.4, c1: 'pink', c2: 'blue' });
    const tex = perFrame(this.layer, f, () => {
      const c = this.layer.ctx;
      if (v === 'A') this.thumbA(c);
      else if (v === 'B') this.thumbB(c);
      else this.thumbC(c);
    });
    this.ctx.comp.draw(r, tex, out);
    return { reg: { A: 3, B: 7, C: 9 }[v], hud: 0 };
  }

  private big(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, col = INK, align: CanvasTextAlign = 'left') {
    c.save();
    c.font = font(F.archivo(125, 900), size);
    c.textAlign = align; c.textBaseline = 'alphabetic';
    c.fillStyle = col; c.fillText(text, x, y);
    c.restore();
  }

  /** The sentence's end with the real arcs from "wide" (layer 8, head 29), framed on "street ... wide". */
  private arcs(c: CanvasRenderingContext2D, cx: number, cy: number, z: number, only7 = false) {
    const toks = this.layout(1);
    const sx = (x: number) => W / 2 + (x - cx) * z, sy = (y: number) => H / 2 + (y - cy) * z;
    const w = this.weights(1, 1);
    const Q = this.anchor(toks[12]!);
    for (let i = 0; i <= 12; i++) {
      if (w[i]! < 0.004) continue;
      const a = i === 7 ? 1 : only7 ? 0.12 : 0.45;
      this.ribbon(c, 0, Q, this.anchor(toks[i]!), w[i]!, 1, i === 12, a, sx, sy, z);
    }
    c.save();
    c.font = font(F.archivo(100, 700), 48 * z);
    for (let i = 1; i < toks.length; i++) {
      const tk = toks[i]!;
      c.fillStyle = i === 12 ? PINK : INK;
      c.globalAlpha = i === 7 || i === 12 || i === 13 ? 1 : 0.85;
      c.fillText(i === 12 ? this.B[12]! : tk.label, sx(tk.x), sy(tk.y));
    }
    c.restore();
    return { toks, sx, sy };
  }

  private thumbA(c: CanvasRenderingContext2D) {
    const toks = this.layout(1);
    const cx = (toks[6]!.x + toks[13]!.x + toks[13]!.w) / 2;
    this.arcs(c, cx, -300, 1.7);
    c.font = font(F.archivo(125, 900), 250);
    const w1 = c.measureText('WHO’S ').width, w2 = c.measureText('IT?').width;
    const x = (W - w1 - w2) / 2;
    this.big(c, 'WHO’S', x, 260, 250);
    this.big(c, 'IT?', x + w1, 260, 250, PINK);
  }

  private thumbB(c: CanvasRenderingContext2D) {
    const toks = this.layout(1);
    const cx = (toks[7]!.x + toks[12]!.x + toks[12]!.w) / 2;
    this.arcs(c, cx, -210, 2.1, true);
    const pct = `${(this.a.row(1, 12)[7]! * 100).toFixed(0)}%`;
    this.big(c, pct, W / 2, 330, 330, INK, 'center');
  }

  private thumbC(c: CanvasRenderingContext2D) {
    const n = 32, cell = 26, gap = 4, gw = n * (cell + gap);
    const x0 = W - gw - 90, y0 = (H - gw) / 2;
    for (let l = 0; l < n; l++) for (let h = 0; h < n; h++) {
      const k = this.a.kind(l, h);
      c.fillStyle = k === 2 ? PINK : k === -1 ? BLUE : INK;
      c.globalAlpha = k === 2 ? 1 : k === 1 ? 0.3 : k === -1 ? 0.3 : 0.2;
      if (k === 1) c.fillStyle = PINK;
      const x = x0 + h * (cell + gap), y = y0 + (n - 1 - l) * (cell + gap);
      c.fillRect(x, y, cell, cell);
      if (k === 2) { c.globalAlpha = 1; c.strokeStyle = INK; c.lineWidth = 4; c.strokeRect(x - 5, y - 5, cell + 10, cell + 10); }
    }
    c.globalAlpha = 1;
    const g = this.a.d.grid.change;
    this.big(c, `${g.clean} OF`, 70, 430, 230);
    this.big(c, (g.heads).toLocaleString('en-US'), 70, 690, 230, PINK);
    c.font = font(F.mono(700), 34);
    c.fillStyle = INK;
    c.fillText('HEADS SWITCH', 78, 770);
  }
}
