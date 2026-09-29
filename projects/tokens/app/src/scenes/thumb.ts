// YouTube thumbnails (not part of the edit), rendered with `render.ts stills --only thumbA,thumbB,thumbC
// --t 0.5,1.5,2.5`. At most 3 big words each, built to read at 168x94 (the phone grid). Next Token already uses
// "AI SEES 73700" and the sliced strawberry, so these show this episode's own findings:
//   A "SPACE IT OUT": one chunk -> it said 2 (67 of 100); the letters spaced out -> it said 3 (90 of 100)
//   B "FIND THE R": the word as the model gets it, one split-flap tile, 73700
//   C "IT READS": the question as the model gets it, eight numbers
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { Wash, specks } from './_kit';
import { drawStrawberry } from './thumb-berry';
import { rrect } from './world-draw';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

export default class Thumb extends Scene {
  wash = new Wash();
  layer = new Layer2D();

  /** Two-ink slab type: an offset under-ink, the top ink multiplied over it. */
  private slab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, top = INK, under = PINK, align: CanvasTextAlign = 'left') {
    c.save();
    c.font = font(F.archivo(125, 900), size);
    c.textAlign = align; c.textBaseline = 'middle';
    c.fillStyle = under; c.fillText(text, x + size * 0.045, y + size * 0.04);
    c.globalCompositeOperation = 'multiply';
    c.fillStyle = top; c.fillText(text, x, y);
    c.restore();
  }

  /** A type sort / chunk tile: paper face, blue slab, ink outline, text in mono. */
  private tile(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, text: string, o: { color?: string; fs?: number; lw?: number; num?: boolean } = {}) {
    c.save();
    c.fillStyle = rgba('blue', 0.55); rrect(c, x + h * 0.08, y + h * 0.1, w, h, h * 0.14); c.fill();
    c.fillStyle = PAPER; rrect(c, x, y, w, h, h * 0.14); c.fill();
    c.strokeStyle = INK; c.lineWidth = o.lw ?? h * 0.07; rrect(c, x, y, w, h, h * 0.14); c.stroke();
    if (o.num) { c.strokeStyle = BLUE; c.lineWidth = h * 0.05; rrect(c, x + h * 0.1, y + h * 0.1, w - h * 0.2, h * 0.8, h * 0.08); c.stroke(); }
    c.fillStyle = o.color ?? INK;
    c.font = font(F.mono(700), o.fs ?? h * 0.62);
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(text, x + w / 2, y + h / 2 + h * 0.03);
    c.restore();
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, v = this.ctx.params.v as 'A' | 'B' | 'C';
    const tok = this.ctx.tok, st = tok.spell_test;
    this.wash.render(r, out, { seed: { A: 29, B: 31, C: 43 }[v], drift: 0, amt: 0.5, c1: v === 'B' ? 'blue' : 'pink', c2: v === 'B' ? 'pink' : 'blue' });
    const c = this.layer.ctx;
    this.layer.clear();
    specks(c, 2.9, { seed: 9, n: 36, size: 8 });
    if (v === 'A') {
      this.slab(c, 'SPACE IT OUT', W / 2, H * 0.16, 190, INK, PINK, 'center');
      // one chunk -> 2
      const y1 = H * 0.36, th = 150;
      this.tile(c, W * 0.06, y1, 900, th, '·strawberry', { fs: 104 });
      this.slab(c, '2', W * 0.8, y1 + th / 2, 300, INK, BLUE, 'center');
      // spaced out -> 3
      const y2 = H * 0.66, tw = 74, gap = 18;
      'strawberry'.split('').forEach((ch, i) => this.tile(c, W * 0.06 + i * (tw + gap), y2, tw, th, ch, { color: ch === 'r' ? PINK : INK, fs: 96 }));
      this.slab(c, '3', W * 0.8, y2 + th / 2, 300, PINK, BLUE, 'center');
      c.font = font(F.mono(700), 42); c.fillStyle = INK; c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(`${st.word.said['2']} of ${st.n}`, W * 0.8, y1 + th / 2 + 150);
      c.fillText(`${st.spaced.said['3']} of ${st.n}`, W * 0.8, y2 + th / 2 + 150);
    } else if (v === 'B') {
      this.slab(c, 'FIND THE R', W * 0.05, H * 0.2, 230, INK, PINK, 'left');
      // the word, as the model gets it: one tile, one number
      const x = W * 0.05, y = H * 0.42, w = W * 0.72, h = 380;
      drawStrawberry(c, W * 0.87, H * 0.7, 360, 0.35);
      this.tile(c, x, y, w, h, String(tok.llama.words[' strawberry']![0]!.id), { color: BLUE, fs: 330, lw: 16, num: true });
    } else {
      this.slab(c, 'IT READS', W * 0.05, H * 0.17, 230, INK, PINK, 'left');
      c.font = font(F.mono(600), 50); c.fillStyle = INK; c.textBaseline = 'middle';
      c.fillText('"How many r\'s are in strawberry?" =', W * 0.055, H * 0.35);
      const ids = tok.llama.question_tokens.map((x) => String(x.id));
      const tw = 420, th = 170, gx = 30, gy = 36, x0 = (W - (4 * tw + 3 * gx)) / 2, y0 = H * 0.45;
      ids.forEach((id, i) => this.tile(c, x0 + (i % 4) * (tw + gx), y0 + Math.floor(i / 4) * (th + gy), tw, th, id, { color: id === '73700' ? PINK : BLUE, fs: 120, num: true }));
    }
    this.ctx.comp.draw(r, this.layer.upload(), out);
    void f;
    return { reg: { A: 3, B: 7, C: 13 }[v], hud: 0, lyric: 0, kinetic: 0, misreg: 2.0 };
  }
}
