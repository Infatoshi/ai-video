// YouTube thumbnails (1920x1080 stills, scaled to 1280x720; checked at 168x94): at most 3 big words, the
// model's real writing as the picture. Rendered with `render.ts stills --t 0.5,1.5,2.5 --only thumbA,thumbB,thumbC`.
//  A: "{N} MINUTES": what it wrote at step 0 vs step 5,000
//  B: "GIBBERISH" -> "SHAKESPEARE": the same page, before and after
//  C: "82 MILLION GUESSES": the loss curve's cliff
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H, clearRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

export default class Thumb extends Scene {
  layer = new Layer2D();

  /** Big type printed pink over a misregistered blue plate (the course's slam look, for the thumbnail only). */
  private big(c: CanvasRenderingContext2D, s: string, x: number, y: number, size: number, align: CanvasTextAlign = 'left', col = PINK) {
    c.save();
    c.font = font(F.archivo(112, 900), size);
    c.textAlign = align;
    c.textBaseline = 'alphabetic';
    c.fillStyle = BLUE;
    c.fillText(s, x + size * 0.035, y + size * 0.03);
    c.globalCompositeOperation = 'multiply';
    c.fillStyle = col;
    c.fillText(s, x, y);
    c.restore();
  }

  private sheet(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number) {
    c.fillStyle = rgba('blue', 0.3); c.fillRect(x + 16, y + 16, w, h);
    c.fillStyle = PAPER; c.fillRect(x, y, w, h);
    c.strokeStyle = INK; c.lineWidth = 4; c.strokeRect(x, y, w, h);
  }

  /** The first `n` characters of a sample on one line (newlines as ↵). */
  private first(step: number, n: number) {
    const s = this.ctx.run.sampleAt(step).text.replace(/^\n+/, '');
    return s.slice(0, n).replace(/\n/g, ' ↵ ');
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const { renderer, comp, run } = this.ctx;
    clearRT(renderer, out, LIN.paper);
    const L = this.layer, c = L.ctx;
    L.clear();
    const v = this.ctx.params.v as string;
    const last = run.maxStep;
    if (v === 'A') {
      this.big(c, `${run.ex.minutes} MINUTES`, 90, 290, 250);
      c.font = font(F.mono(700), 40); c.fillStyle = INK;
      c.fillText('OF TRAINING, FROM RANDOM', 100, 360);
      this.sheet(c, 90, 410, W - 180, 580);
      c.font = font(F.mono(700), 34); c.fillStyle = INK;
      c.fillText('STEP 0', 140, 480);
      c.font = font(F.mono(500), 76); c.fillStyle = INK;
      const g = run.sampleAt(0).text.replace(/\n/g, ' ').slice(0, 22);
      c.fillText(g, 140, 575);
      c.strokeStyle = PINK; c.lineWidth = 12;
      c.beginPath(); c.moveTo(130, 550); c.lineTo(140 + c.measureText(g).width + 10, 550); c.stroke();
      c.font = font(F.mono(700), 34); c.fillStyle = INK;
      c.fillText(`STEP ${last.toLocaleString('en-US')}`, 140, 690);
      c.font = font(F.mono(700), 84); c.fillStyle = BLUE;
      const w = run.sampleAt(last).text.replace(/^\n+/, '').split('\n')[0]!;
      const cut = w.lastIndexOf(' ', 30);
      c.fillText(w.slice(0, cut), 140, 790);
      c.fillText(w.slice(cut + 1), 140, 900);
    } else if (v === 'B') {
      this.big(c, 'GOSPEL', W / 2, 330, 300, 'center');
      this.sheet(c, 140, 400, W - 280, 250);
      c.font = font(F.mono(700), 30); c.fillStyle = INK; c.textAlign = 'left';
      c.fillText('STEP 100, SUNG VERBATIM', 190, 460);
      c.font = font(F.mono(700), 80); c.fillStyle = BLUE;
      const e = run.ex.excerpts[0]!.lines.join(' ').split(' ').slice(2, 6).join(' ');
      c.fillText(e, 190, 580);
      this.big(c, 'GIBBERISH', W / 2, 960, 270, 'center', INK);
    } else {
      // the loss curve's cliff: every step's own loss, huge
      const S = run.d.steps, x0 = 120, x1 = W - 120, y0 = 360, y1 = H - 110;
      const X = (s: number) => x0 + (s / last) * (x1 - x0), Y = (l: number) => y1 - (l / 4.5) * (y1 - y0);
      this.sheet(c, 70, 300, W - 140, H - 360);
      c.strokeStyle = rgba('blue', 0.8); c.lineWidth = 3;
      c.beginPath();
      S.loss.forEach((l, i) => { if (i === 0) c.moveTo(X(S.step[i]!), Y(l)); else c.lineTo(X(S.step[i]!), Y(l)); });
      c.stroke();
      c.strokeStyle = PINK; c.lineWidth = 14; c.lineJoin = 'round';
      c.beginPath();
      run.d.evals.forEach((e, i) => { if (i === 0) c.moveTo(X(e.step), Y(e.val)); else c.lineTo(X(e.step), Y(e.val)); });
      c.stroke();
      this.big(c, `${Math.round((last * run.d.chars_per_step) / 1e6)} MILLION`, 90, 180, 170);
      this.big(c, 'GUESSES', 90, 330, 170, 'left', INK);
      c.font = font(F.mono(700), 44); c.fillStyle = INK; c.textAlign = 'right';
      c.fillText(`LOSS ${run.d.evals[0]!.val.toFixed(2)} → ${run.finalEval().val.toFixed(2)}`, x1, 440);
    }
    comp.draw(renderer, L.upload(), out);
    return { reg: 3, hud: 0, kinetic: 0, lyric: 0 };
  }
}
