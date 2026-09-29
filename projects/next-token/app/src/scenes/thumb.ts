// YouTube thumbnails (not part of the edit): three printed compositions for Test & Compare, rendered with
// `render.ts stills --only thumb --t 0.5,1.5,2.5` (timeline.ts adds the entries only for ?only=thumb).
// Each has at most 3 big words and reads at 168x94 (the phone grid): A the strawberry, B the die, C the tower.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { Wash, makeCam, tokenFaceTexture, outline, specks } from './_kit';
import { Tower, riderTile } from './_tower';
import { drawStrawberry, drawSliced } from './thumb-berry';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

export default class Thumb extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(30);
  wash = new Wash();
  layer = new Layer2D();
  tower = new Tower();
  die!: THREE.Mesh;

  override init() {
    const llm = this.ctx.llm, st = llm.steps[3]!;
    // a die of the two candidates the model drew from at step 3: "2" (pink, front and top) and "3" (blue)
    const two = st.nucleus[0]![0], three = st.nucleus[1]![0];
    const faces = [three, three, two, two, two, three].map((t) =>
      new THREE.MeshBasicMaterial({ map: tokenFaceTexture(t, t === two ? 'pink' : 'blue', 'paper', { w: 256, h: 256 }) }));
    this.die = new THREE.Mesh(new THREE.BoxGeometry(2.4, 2.4, 2.4), faces);
    outline(this.die, 5);
    this.scene.add(this.die, this.tower.group);
    this.tower.outlineAll();
    this.tower.ride(riderTile('?'), 1);
  }

  /** Two-ink slab type: blue under, pink over with multiply. */
  private slab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, top = PINK, under = BLUE, align: CanvasTextAlign = 'center') {
    c.save();
    c.font = font(F.archivo(125, 900), size);
    c.textAlign = align; c.textBaseline = 'middle';
    c.fillStyle = under; c.fillText(text, x + size * 0.045, y + size * 0.04);
    c.globalCompositeOperation = 'multiply';
    c.fillStyle = top; c.fillText(text, x, y);
    c.restore();
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, v = this.ctx.params.v as 'A' | 'B' | 'C' | 'D' | 'E' | 'I';
    this.wash.render(r, out, { seed: { A: 11, B: 23, C: 37, D: 41, E: 53, I: 53 }[v], drift: 0, amt: v === 'D' || v === 'E' ? 0.5 : 0.65, c1: v === 'B' || v === 'E' || v === 'I' ? 'blue' : 'pink', c2: v === 'B' || v === 'E' || v === 'I' ? 'pink' : 'blue' });

    this.die.visible = v === 'B';
    this.tower.group.visible = v === 'C';
    if (v === 'B') {
      this.die.rotation.set(-0.42, 0.62, 0.08);
      this.die.position.set(-2.2, -0.1, 0);
      this.cam.position.set(0, 0.4, 10.5); this.cam.lookAt(0, 0, 0);
    } else if (v === 'C') {
      const Ht = this.tower.height;
      this.tower.light(1);
      this.cam.fov = 34; this.cam.updateProjectionMatrix();
      this.cam.position.set(10, Ht * 0.8, 36); this.cam.lookAt(-10.5, Ht * 0.52, 0);
    }
    this.cam.up.set(0, 1, 0);
    this.scene.updateMatrixWorld(true);
    r.setRenderTarget(out); r.clearDepth(); r.render(this.scene, this.cam);

    const c = this.layer.ctx;
    this.layer.clear();
    specks(c, 3.7, { seed: 5, n: 45, size: 9 });
    if (v === 'A') {
      // "strawberry" with its three r's in pink, and the model's answer stamped over it
      const word = 'strawberry', size = 250;
      c.font = font(F.archivo(125, 900), size);
      c.textBaseline = 'middle';
      const w = c.measureText(word).width, x0 = W / 2 - w / 2, y = H * 0.36;
      let x = x0;
      for (const ch of word) {
        c.fillStyle = ch === 'r' ? PINK : INK;
        c.fillText(ch, x, y);
        x += c.measureText(ch).width;
      }
      c.save();
      c.translate(W * 0.5, H * 0.74); c.rotate(-0.06);
      c.font = font(F.archivo(125, 900), 270); c.textAlign = 'center'; c.textBaseline = 'middle';
      const bw = c.measureText('AI: 2').width + 110;
      c.fillStyle = PINK; c.fillRect(-bw / 2, -165, bw, 330);
      c.strokeStyle = INK; c.lineWidth = 14; c.strokeRect(-bw / 2, -165, bw, 330);
      c.fillStyle = PAPER; c.fillText('AI: 2', 0, 14);
      c.restore();
    } else if (v === 'B') {
      this.slab(c, '69.7%', W * 0.68, H * 0.44, 270);
      c.fillStyle = INK; c.font = font(F.archivo(125, 900), 96); c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText('ROLLED A 2', W * 0.68, H * 0.74);
    } else if (v === 'D') {
      // the three chunks the model writes the word as, each labelled with its token and id
      const llm = this.ctx.llm, toks = llm.d.strawberry['strawberry']!;
      const ids = llm.steps.slice(10, 13).map((st) => st.tok.id);
      this.slab(c, 'WHAT AI SEES', W / 2, H * 0.14, 165, INK, PINK);
      const centres = drawSliced(c, W / 2 + 40, H * 0.52, 640, [0.3, 0.2, 0.5], 120);
      centres.forEach((sx, i) => {
        const sy = H * 0.86;
        c.font = font(F.mono(700), 96);
        const w = c.measureText(toks[i]!).width + 60;
        c.fillStyle = INK; c.fillRect(sx - w / 2, sy - 62, w, 124);
        c.fillStyle = PAPER; c.textAlign = 'center'; c.textBaseline = 'middle';
        c.fillText(toks[i]!, sx, sy + 4);
        void ids;
      });
    } else if (v === 'I') {
      // app icon: the strawberry alone, centred (crop the middle square)
      drawStrawberry(c, W / 2, H / 2 + 20, 820, 0.18);
    } else if (v === 'E') {
      // the whole word, as the model got it in the question: one token, one number
      const id = this.ctx.llm.d.prompt.find((x) => x.t === ' strawberry')?.id ?? 0;
      const bx = W * 0.72, by = H * 0.56;
      drawStrawberry(c, bx, by, 700, 0.25);
      this.slab(c, 'AI SEES', W * 0.05, H * 0.28, 230, INK, PINK, 'left');
      const tx = W * 0.06, ty = H * 0.5, tw = 720, th = 300;
      c.strokeStyle = INK; c.lineWidth = 7;
      c.beginPath(); c.moveTo(tx + tw - 40, ty + 36); c.quadraticCurveTo(bx - 250, by - 60, bx - 70, by - 230); c.stroke();
      c.save(); c.translate(tx, ty); c.rotate(-0.05);
      c.fillStyle = PAPER; c.fillRect(0, 0, tw, th);
      c.strokeStyle = INK; c.lineWidth = 10; c.strokeRect(0, 0, tw, th);
      c.beginPath(); c.arc(tw - 40, 40, 16, 0, Math.PI * 2); c.stroke();
      c.fillStyle = PINK; c.font = font(F.mono(700), 210); c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(String(id), tw / 2, th / 2 + 14);
      c.restore();
    } else {
      this.slab(c, 'HOW AI', W * 0.05, H * 0.34, 250, PINK, BLUE, 'left');
      this.slab(c, 'WRITES', W * 0.05, H * 0.64, 250, PINK, BLUE, 'left');
    }
    this.ctx.comp.draw(r, this.layer.upload(), out);
    return { reg: { A: 3, B: 7, C: 13, D: 17, E: 19, I: 19 }[v], hud: 0, lyric: 0, kinetic: 0, misreg: 2.2 };
  }
}
