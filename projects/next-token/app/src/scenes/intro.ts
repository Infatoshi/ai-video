// intro (instrumental, ~8 bars): the reference scene for the kit. A new shot every bar, each a different
// idea: the title, the tower waking floor by floor, the empty prompt, the lobby and its shaft, the real
// parameter count typing in, the tower from above, the floor numbers racing, the title again over the
// whole tower. Three layers in every shot (wash / token rain behind, the hero, specks or tiles in front).
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, frameIdx } from '../engine/util';
import { Wash, tokenRain, specks, makeCam, drive, shot, snapIn, portrait } from './_kit';
import { Tower, riderTile } from './_tower';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

export default class Intro extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(34);
  wash = new Wash();
  back = new Layer2D();
  front = new Layer2D();
  tower = new Tower();

  override init() {
    this.scene.add(this.tower.group);
    this.tower.outlineAll();
    this.tower.ride(riderTile('?'), 0);
  }

  /** Big two-ink type: blue underprint, pink on top with multiply (prints purple where they overlap). */
  private slab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, k = 1, align: CanvasTextAlign = 'center') {
    if (k <= 0) return; // not landed yet
    c.save();
    c.font = font(F.archivo(125, 900), size);
    c.textAlign = align;
    c.textBaseline = 'middle';
    c.translate(x, y);
    const s = 1.25 - 0.25 * k;
    c.scale(s, s);
    c.fillStyle = BLUE;
    c.fillText(text, size * 0.04, size * 0.035);
    c.globalCompositeOperation = 'multiply';
    c.fillStyle = PINK;
    c.fillText(text, 0, 0);
    c.restore();
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio, llm = this.ctx.llm;
    const sh = shot(this.ctx, f);
    const kind = sh.i % 8;
    const d = drive(au, f.t, 1.1, 1.6); // the intro is quiet: a higher base keeps the camera travelling
    const P = portrait();
    const pool = llm.d.prompt;

    // background: the wash, reseeded per shot, plus token rain from the real prompt
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: kind === 4 ? 0.8 : 0.5, c1: kind % 2 ? 'blue' : 'pink', c2: kind % 2 ? 'pink' : 'blue' });
    const b = this.back.ctx;
    this.back.clear();
    tokenRain(b, pool, d * 0.6, { seed: sh.seed, cols: P ? 9 : 14 });
    this.ctx.comp.draw(r, this.back.upload(), out);

    // the hero: the tower, framed per shot
    const L = this.tower.layers;
    const sweep = (d * 0.35) % 1;
    this.tower.light(kind === 1 || kind === 6 ? sweep : kind === 7 ? 1 : -1);
    this.tower.ride(this.tower.rider.children[0]!, kind === 3 ? clamp(sh.u * 0.3) : sweep);
    const Ht = this.tower.height;
    const dist = P ? 1.45 : 1;
    const orbitA = d * 0.12 + sh.i * 0.9;
    let pos: THREE.Vector3, look: THREE.Vector3, fov = 34;
    switch (kind) {
      case 0: pos = new THREE.Vector3(Math.cos(orbitA) * 30 * dist, Ht * 0.45, Math.sin(orbitA) * 30 * dist); look = new THREE.Vector3(0, Ht * 0.5, 0); break;
      case 1: pos = new THREE.Vector3(Math.cos(orbitA) * 7 * dist, 0.6, Math.sin(orbitA) * 7 * dist); look = new THREE.Vector3(0, Ht * 0.55, 0); fov = 58; break;
      case 2: pos = new THREE.Vector3(18 * dist, Ht * 0.3, 18 * dist); look = new THREE.Vector3(0, Ht * 0.3, 0); break;
      case 3: pos = new THREE.Vector3(Math.cos(orbitA) * 9 * dist, 0.4 + sh.u * 2, Math.sin(orbitA) * 9 * dist); look = new THREE.Vector3(0, 2.2 + sh.u * 3, 0); fov = 48; break;
      case 4: pos = new THREE.Vector3(Math.cos(orbitA) * 26 * dist, Ht * 0.9, Math.sin(orbitA) * 26 * dist); look = new THREE.Vector3(0, Ht * 0.6, 0); break;
      case 5: pos = new THREE.Vector3(Math.cos(orbitA) * 12 * dist, Ht * 0.62 + sh.u * 4, Math.sin(orbitA) * 12 * dist); look = new THREE.Vector3(0, Ht * 0.45 + sh.u * 4, 0); fov = 52; break;
      case 6: pos = new THREE.Vector3(-6 * dist, Ht * sweep + 1, 5 * dist); look = new THREE.Vector3(0, Ht * sweep, 0); fov = 44; break;
      default: pos = new THREE.Vector3(Math.cos(orbitA) * 34 * dist, Ht * 0.35, Math.sin(orbitA) * 34 * dist); look = new THREE.Vector3(0, Ht * 0.48, 0); break;
    }
    this.cam.up.set(0, 1, 0);
    this.cam.position.copy(pos);
    this.cam.fov = fov;
    this.cam.updateProjectionMatrix();
    this.cam.lookAt(look);
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);

    // front: per-shot type and specks
    const c = this.front.ctx;
    this.front.clear();
    specks(c, d, { seed: sh.seed, n: kind === 4 ? 40 : 80 });
    const k = snapIn(f.t, sh.t0, 0.08);
    const cx = W / 2, cy = H / 2;
    if (kind === 0 || kind === 7) {
      const size = P ? 230 : 300;
      this.slab(c, 'NEXT', cx, cy - size * 0.48, size * (kind === 7 ? 0.7 : 1), k);
      this.slab(c, 'TOKEN', cx, cy + size * 0.48, size * (kind === 7 ? 0.7 : 1), snapIn(f.t, sh.t0 + (sh.t1 - sh.t0) / 4, 0.08));
    } else if (kind === 2) {
      // the empty prompt: a box, a caret blinking on the 8ths
      const bw = Math.min(W * 0.8, 1100), bh = 150, bx = cx - bw / 2, by = cy - bh / 2;
      c.fillStyle = PAPER; c.fillRect(bx, by, bw, bh);
      c.strokeStyle = INK; c.lineWidth = 6; c.strokeRect(bx, by, bw, bh);
      c.fillStyle = INK; c.font = font(F.mono(600), 58); c.textBaseline = 'middle';
      c.fillText('>', bx + 40, cy);
      if (Math.floor(au.beatAt(f.ft) * 2) % 2 === 0) { c.fillStyle = PINK; c.fillRect(bx + 100, cy - 32, 30, 64); }
      c.fillStyle = BLUE; c.font = font(F.mono(500), 22);
      c.fillText('ASK LLAMA 3.1 8B ANYTHING', bx, by - 26);
    } else if (kind === 4) {
      // the real parameter count, one digit per 8th note
      const digits = llm.cfg.params.toLocaleString('en-US');
      const eighths = Math.floor((au.beatAt(f.ft) - au.beatAt(sh.t0)) * 2) + 1;
      let shown = 0, n = 0;
      for (const ch of digits) { if (/\d/.test(ch)) { if (n >= eighths * 2) break; n++; } shown++; }
      const size = P ? 120 : 170;
      c.font = font(F.mono(700), size);
      c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillStyle = INK;
      c.fillText(digits.slice(0, shown), cx, cy - (P ? 20 : 30));
      c.font = font(F.archivo(125, 900), size * 0.42);
      c.fillStyle = PINK;
      c.fillText('PARAMETERS', cx, cy + size * 0.75);
    } else if (kind === 5) {
      // the tower's real dimensions, one per 8th note
      const lines = [`${llm.cfg.layers} LAYERS`, `${llm.cfg.heads} HEADS EACH`, `${llm.cfg.hidden.toLocaleString('en-US')} WIDE`, `${llm.cfg.vocab.toLocaleString('en-US')} WORDS`];
      const n = Math.min(lines.length, Math.floor((au.beatAt(f.ft) - au.beatAt(sh.t0)) * 2) + 1);
      c.font = font(F.archivo(125, 900), P ? 96 : 118);
      c.textAlign = P ? 'center' : 'left'; c.textBaseline = 'middle';
      lines.slice(0, n).forEach((l, i) => {
        const x = P ? cx : W * 0.06, y = (P ? H * 0.3 : H * 0.26) + i * (P ? 120 : 150);
        c.fillStyle = i === n - 1 ? PINK : BLUE;
        c.fillText(l, x, y);
      });
    } else if (kind === 6) {
      // floor numbers racing up on the 16ths
      const sixteenth = Math.floor(au.beatAt(f.ft) * 4);
      const l = (sixteenth % L) + 1;
      c.font = font(F.archivo(125, 900), P ? 260 : 340);
      c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillStyle = INK;
      c.fillText(`L${l}`, P ? cx : W * 0.7, P ? H * 0.64 : cy);
      c.font = font(F.mono(600), 26); c.fillStyle = PINK;
      c.fillText(`OF ${L} LAYERS`, P ? cx : W * 0.7, (P ? H * 0.64 : cy) + (P ? 170 : 210));
    }
    // a tile crossing the frame in the lobby shot and the top view (foreground motion)
    if (kind === 3 || kind === 5) {
      const x = ((f.t - sh.t0) / Math.max(0.2, sh.t1 - sh.t0)) * (W + 400) - 200;
      c.fillStyle = PINK; c.fillRect(x, H * 0.72, 150, 54);
      c.fillStyle = PAPER; c.font = font(F.mono(700), 30); c.textBaseline = 'middle';
      c.fillText(String(pool[frameIdx(sh.t0) % pool.length]!.id), x + 18, H * 0.72 + 27);
    }
    this.ctx.comp.draw(r, this.front.upload(), out);
    return { reg: sh.seed, kinetic: 0 };
  }
}
