// YouTube thumbnails (not part of the edit), rendered with `render.ts stills --only thumbD,thumbE --t 0.5,1.5`
// (timeline.ts adds the entries only for ?only=thumb...). Both on the shared H100 model:
//   D  one lit SM out of 132 (0.8% of the chip, the book's naive matmul at 0.3 of 35.6 TFLOPS): "YOU'RE USING 1%"
//   E  every SM lit: "93x FASTER" (the book's CH6 matmul, 0.3 -> 27.8 TFLOPS on an RTX 3090, 92.7x)
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H, clearRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { H100 } from './_h100';

const BONE = rgba('bone'), SIGNAL = rgba('signal'), ASH = rgba('ash');

export default class Thumb extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(30, W / H, 1, 5000);
  gpu!: H100;
  text = new Layer2D();

  override init() {
    this.gpu = new H100({ renderer: this.ctx.renderer });
    this.scene.add(this.gpu.group, H100.rig());
    this.scene.environment = this.gpu.env;
  }

  render(_f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, v = this.ctx.params.v as 'D' | 'E';
    // the lit SM for D: an enabled site near the middle of the die
    let one = 0, best = Infinity;
    for (let i = 0; i < this.gpu.smCount; i++) {
      if (!this.gpu.smEnabled(i)) continue;
      const p = this.gpu.smPos(i), d = Math.hypot(p.x - 3, p.z + 2);
      if (d < best) { best = d; one = i; }
    }
    this.gpu.setSMActivity(v === 'D' ? (i) => (i === one ? 4.5 : 0) : (i) => (this.gpu.smEnabled(i) ? 1.1 + 0.25 * ((i * 7) % 5) / 5 : 0));
    this.gpu.setHBMActivity(v === 'D' ? () => 0.05 : () => 1.2);
    this.gpu.update(0);
    clearRT(r, out, LIN.ink);
    const dist = v === 'D' ? 95 : 150;
    const target = v === 'D' ? this.gpu.smPos(one) : new THREE.Vector3(0, 0, 0);
    this.cam.position.copy(target).add(new THREE.Vector3(dist * 0.55, dist * 0.62, dist * 0.56));
    this.cam.lookAt(target.clone().add(new THREE.Vector3(v === 'D' ? -dist * 0.18 : -dist * 0.42, 0, v === 'D' ? 0 : dist * 0.1)));
    H100.fitClip(this.cam, dist);
    r.setRenderTarget(out); r.clearDepth(); r.render(this.scene, this.cam);

    const c = this.text.ctx;
    this.text.clear();
    c.textBaseline = 'alphabetic';
    // a dark wash on the left so the type holds over the bloom
    const g = c.createLinearGradient(0, 0, W * 0.62, 0);
    g.addColorStop(0, 'rgba(10,10,11,0.9)'); g.addColorStop(1, 'rgba(10,10,11,0)');
    c.fillStyle = g; c.fillRect(0, 0, W, H);
    if (v === 'D') {
      c.font = font(F.archivo(125, 900), 150); c.fillStyle = BONE;
      c.fillText("YOU'RE USING", 90, 330);
      c.font = font(F.archivo(125, 900), 470); c.fillStyle = SIGNAL;
      c.fillText('1%', 70, 760);
      c.font = font(F.mono(500), 40); c.fillStyle = ASH;
      c.fillText('one SM of 132 · a naive matmul runs at 0.8% of peak', 96, 850);
    } else {
      c.font = font(F.archivo(125, 900), 420); c.fillStyle = SIGNAL;
      c.fillText('93×', 70, 520);
      c.font = font(F.archivo(125, 900), 190); c.fillStyle = BONE;
      c.fillText('FASTER', 80, 740);
      c.font = font(F.mono(500), 40); c.fillStyle = ASH;
      c.fillText('same GPU, same matmul · 0.3 → 27.8 TFLOPS (RTX 3090, CH6)', 90, 840);
    }
    this.ctx.comp.draw(r, this.text.upload(), out);
    return { hud: 0, pdoom: 0, frame: 0, bloom: 0.9, vignette: 0.5 };
  }
}
