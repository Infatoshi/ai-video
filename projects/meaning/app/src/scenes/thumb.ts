// YouTube thumbnails (render.ts stills --only thumbA,thumbB,thumbC --t 0.5,1.5,2.5): at most 3 big words each,
// readable at 168x94. Every number and position is data/emb.json.
//  A "QUEEN CAME 4TH": king - man + woman, the real top 5 (three spellings of King first)
//  B "RAIN = 4,096 NUMBERS": ·rain's real row as a bar chart
//  C "MEANING = DIRECTION": the man -> woman arrow carried to boy and father (real 3D squash, side view)
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { Wash } from './_kit';
import { loadEmb, toWorld, tokLabel, type Emb } from './flight-data';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

export default class Thumb extends Scene {
  emb!: Emb;
  wash = new Wash();
  layer = new Layer2D();

  override async init() { this.emb = await loadEmb(); }

  private fit(c: CanvasRenderingContext2D, text: string, fam: string, size: number, maxW: number) {
    c.font = font(fam, size);
    const w = c.measureText(text).width;
    if (w > maxW) { size *= maxW / w; c.font = font(fam, size); }
    return size;
  }

  render(_f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, v = this.ctx.params.v as string;
    this.wash.render(r, out, { seed: 5, drift: 0, amt: 0.28, c1: 'blue', c2: 'blue' });
    const c = this.layer.ctx;
    this.layer.clear();
    c.save();
    if (v === 'A') this.a(c); else if (v === 'B') this.b(c); else this.cc(c);
    c.restore();
    this.ctx.comp.draw(r, this.layer.upload(), out);
    return { hud: 0, lyric: 0, kinetic: 0, reg: 3 };
  }

  private a(c: CanvasRenderingContext2D) {
    const e = this.emb, top = e.analogies.king!.top5;
    const big = F.archivo(125, 900);
    c.textBaseline = 'alphabetic';
    c.fillStyle = PINK;
    let s = this.fit(c, 'QUEEN', big, 300, W * 0.5);
    c.fillText('QUEEN', 70, 360);
    c.fillStyle = INK;
    s = this.fit(c, 'CAME 4TH', big, 230, W * 0.5);
    c.fillText('CAME 4TH', 70, 360 + s * 1.05);
    // the list
    const x = W * 0.6, y0 = 170, rh = 150;
    c.font = font(F.archivo(100, 700), 44); c.fillStyle = INK;
    c.fillText('king − man + woman =', x, y0 - 40);
    top.slice(0, 4).forEach((t, i) => {
      const q = i === 3;
      c.fillStyle = q ? PINK : INK;
      c.font = font(F.mono(700), 110);
      c.fillText(String(i + 1), x, y0 + 110 + i * rh);
      c.font = font(F.archivo(112.5, 900), 124);
      c.fillText(tokLabel(t.t).replace('·', ''), x + 110, y0 + 110 + i * rh);
    });
    c.fillStyle = INK; c.font = font(F.mono(600), 26);
    c.fillText('LLAMA 3.1 8B · ITS EMBEDDING TABLE · REAL TOP 4', 70, H - 60);
  }

  private b(c: CanvasRenderingContext2D) {
    const e = this.emb, vals = e.rows[' rain']!.values.slice(0, 384);
    const vmax = Math.max(...vals.map(Math.abs));
    const y0 = H * 0.74, hh = 190, x0 = 60, x1 = W - 60;
    const n = vals.length, bw = (x1 - x0) / n;
    for (let i = 0; i < n; i++) {
      const v = vals[i]! / vmax;
      const h = Math.sign(v) * Math.pow(Math.abs(v), 0.85) * hh;
      c.fillStyle = PINK;
      c.fillRect(x0 + i * bw, h > 0 ? y0 - h : y0, bw * 0.72, Math.abs(h));
    }
    c.fillStyle = INK; c.fillRect(x0, y0 - 2, x1 - x0, 4);
    const big = F.archivo(125, 900);
    c.textBaseline = 'alphabetic';
    c.fillStyle = INK;
    this.fit(c, 'RAIN =', big, 280, W * 0.9);
    c.fillText('RAIN =', 60, 270);
    c.fillStyle = BLUE;
    this.fit(c, '4,096 NUMBERS', big, 210, W * 0.9);
    c.fillText('4,096 NUMBERS', 60, 270 + 205);
    c.fillStyle = INK; c.font = font(F.mono(600), 26);
    c.fillText('THE FIRST 384 OF THE 4,096 NUMBERS IN “·rain”’S ROW (TOKEN 11,422), LLAMA 3.1 8B', 60, H - 50);
  }

  private cc(c: CanvasRenderingContext2D) {
    const e = this.emb;
    const P = (t: string) => toWorld(e.space.featured.find((f) => f.t === t)!.p);
    const people = [' man', ' woman', ' boy', ' girl', ' father', ' mother'];
    const ctr = people.reduce((a, t) => a.add(P(t)), new THREE.Vector3()).multiplyScalar(1 / people.length);
    const cam = new THREE.PerspectiveCamera(30, W / H, 0.1, 100);
    cam.position.copy(ctr).add(new THREE.Vector3(6.4, 1.0, 0.8));
    cam.lookAt(ctr);
    cam.updateMatrixWorld();
    const pr = (p: THREE.Vector3) => { const q = p.clone().project(cam); return [W * 0.52 + q.x * W * 0.5, H * 0.72 - q.y * H * 0.8] as const; };
    const vec = P(' woman').sub(P(' man'));
    const arrow = (a: THREE.Vector3, col: string, w: number) => {
      const [x0, y0] = pr(a), [x1, y1] = pr(a.clone().add(vec));
      const ang = Math.atan2(y1 - y0, x1 - x0), hl = 34;
      c.strokeStyle = col; c.fillStyle = col; c.lineWidth = w; c.lineCap = 'round';
      c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1 - Math.cos(ang) * hl * 0.8, y1 - Math.sin(ang) * hl * 0.8); c.stroke();
      c.beginPath(); c.moveTo(x1, y1); c.lineTo(x1 - Math.cos(ang - 0.45) * hl, y1 - Math.sin(ang - 0.45) * hl);
      c.lineTo(x1 - Math.cos(ang + 0.45) * hl, y1 - Math.sin(ang + 0.45) * hl); c.closePath(); c.fill();
    };
    arrow(P(' man'), INK, 12);
    arrow(P(' boy'), PINK, 16);
    arrow(P(' father'), PINK, 16);
    c.font = font(F.archivo(100, 800), 58);
    for (const t of people) {
      const [x, y] = pr(P(t));
      c.fillStyle = BLUE; c.beginPath(); c.arc(x, y, 20, 0, Math.PI * 2); c.fill();
      c.strokeStyle = INK; c.lineWidth = 3; c.stroke();
      c.fillStyle = INK; c.fillText(tokLabel(t).replace('·', ''), x + 22, y - 18);
    }
    const big = F.archivo(125, 900);
    c.fillStyle = INK; c.textBaseline = 'alphabetic';
    this.fit(c, 'MEANING', big, 230, W * 0.46);
    c.fillText('MEANING', 60, 250);
    c.fillStyle = PINK;
    this.fit(c, '= DIRECTION', big, 230, W * 0.6);
    c.fillText('= DIRECTION', 60, 250 + 200);
    c.fillStyle = PAPER;
    c.fillStyle = INK; c.font = font(F.mono(600), 26);
    c.fillText('ONE ARROW, man → woman, CARRIED TO boy AND father · LLAMA 3.1 8B', 60, H - 60);
  }
}
