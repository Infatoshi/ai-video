// YouTube thumbnails (render.ts stills --only thumbA,thumbB,thumbC): a frame of the real scene at a chosen song
// time, the HUD off, and at most three big words. Checked at 168x94 (the feed's smallest size).
import type * as THREE from 'three';
import type { Frame, PostOverrides } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import World from './world';

const PINK = rgba('pink'), INK = rgba('ink'), PAPER = rgba('paper');

export default class Thumb extends World {
  words = new Layer2D();
  override lite = true;
  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides | void {
    const v = this.ctx.params.v as string, T = this.T;
    // A: the ball rolling into the valley, "HOW AI LEARNS"; B: the untangled spirals, "3,121 STEPS";
    // C: the too-big jump, "TOO BIG A STEP"
    const at = v === 'A' ? (T.c2 ?? 200) + 12 : v === 'B' ? (T.outb ?? 230) + 3 : (T.brc ?? 180) + 1.5;
    const g = { ...f, t: at, ft: at };
    super.render(g, out);
    const c = this.words.ctx;
    this.words.clear();
    const lines = v === 'A' ? ['HOW AI', 'LEARNS'] : v === 'B' ? ['3,121', 'STEPS'] : ['STEP', 'TOO BIG'];
    const size = 190;
    c.font = font(F.archivo(112, 900), size);
    c.textBaseline = 'alphabetic';
    const x = 70, y0 = v === 'C' ? 230 : 250;
    lines.forEach((l, i) => {
      const w = c.measureText(l).width;
      c.fillStyle = PAPER; c.fillRect(x - 18, y0 + i * size * 0.95 - size * 0.8, w + 36, size * 0.93);
      c.fillStyle = i === 1 ? PINK : INK;
      c.fillText(l, x, y0 + i * size * 0.95);
    });
    this.ctx.comp.draw(this.ctx.renderer, this.words.upload(), out);
    return { reg: 3, hud: 0, lyric: 0 };
  }
}
void W;
