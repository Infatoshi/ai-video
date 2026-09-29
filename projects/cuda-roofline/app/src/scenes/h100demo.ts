// Preview of the shared H100 model (_h100.ts) for review, played after the song (timeline entry
// `h100demo`). 0–5 s orbit the module, 5–9 s explode the stack, 9–13 s dive onto the die with data
// pulses HBM → L2 → SM, 13–20 s into one SM (lanes and tensor cores lighting).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp, smoothstep } from '../engine/util';
import { H100, H100_Y, type PulseItem } from './_h100';

const SM = 58; // a mid-die SM site (GPC 3)

export default class H100Demo extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(32, W / H, 0.05, 5000);
  gpu!: H100;
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();
  famM = F.mono(500);

  override async init() {
    this.gpu = new H100({ renderer: this.ctx.renderer });
    this.scene.add(this.gpu.group, H100.rig());
    this.scene.environment = this.gpu.env;
    this.scene.background = new THREE.Color().setRGB(LIN.ink[0], LIN.ink[1], LIN.ink[2], THREE.LinearSRGBColorSpace);
  }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const t = f.lt;
    const g = this.gpu;
    // explode 5–9 s (up 5→6.5, hold, down 8→9)
    const ex = smoothstep(5, 6.5, t) * (1 - smoothstep(8, 9, t));
    g.explode(ex);
    g.update(f.t);
    const cam = this.cam;
    const tgt = new THREE.Vector3();
    let dist = 170;
    if (t < 5) {
      const az = lerp(-0.9, 0.5, ease.inOutCubic(t / 5)), el = 0.62;
      dist = 175;
      cam.position.set(Math.sin(az) * Math.cos(el) * dist, Math.sin(el) * dist, Math.cos(az) * Math.cos(el) * dist);
      tgt.set(0, 0, 0);
    } else if (t < 9) {
      const k = ease.inOutCubic(clamp((t - 5) / 4));
      const az = lerp(0.5, 0.95, k), el = lerp(0.62, 0.42, smoothstep(5, 6.5, t)) - 0.07 * smoothstep(8, 9, t);
      dist = lerp(175, 120, smoothstep(5, 6.5, t));
      tgt.set(0, lerp(0, 22, ex), 0);
      cam.position.set(Math.sin(az) * Math.cos(el) * dist, tgt.y + Math.sin(el) * dist, Math.cos(az) * Math.cos(el) * dist);
    } else if (t < 13) {
      const k = ease.inOutCubic(clamp((t - 9) / 3.2));
      const smp = g.smPos(SM);
      dist = lerp(120, 22, k);
      const el = lerp(0.35, 1.0, k), az = lerp(0.95, 0.25, k);
      tgt.copy(new THREE.Vector3(0, 3, 0).lerp(smp, k * 0.6));
      cam.position.set(tgt.x + Math.sin(az) * Math.cos(el) * dist, tgt.y + Math.sin(el) * dist, tgt.z + Math.cos(az) * Math.cos(el) * dist);
    } else {
      const k = ease.inOutCubic(clamp((t - 13) / 3.5));
      const smp = g.smPos(SM);
      dist = lerp(22, 1.9, k);
      const el = lerp(1.0, 0.62, k) , az = lerp(0.25, -0.35, clamp((t - 13) / 7));
      tgt.copy(smp).add(new THREE.Vector3(0, 0.03, 0));
      cam.position.set(tgt.x + Math.sin(az) * Math.cos(el) * dist, tgt.y + Math.sin(el) * dist, tgt.z + Math.cos(az) * Math.cos(el) * dist);
    }
    cam.lookAt(tgt);
    H100.fitClip(cam, dist);

    // pulses: HBM → SM during the die dive, faded out before the camera is inside the SM (their length is a
    // fraction of a ~40 mm path, so at SM scale they would be bars across the view)
    const items: PulseItem[] = [];
    const pg = 2.2 * (1 - smoothstep(13, 14, t));
    if (t > 9 && pg > 0) {
      for (let k = 0; k < 6; k++) {
        if (!g.hbm(k).active) continue;
        const path = g.pulsePath(k, SM + (k % 3) * 18 - 18 + (k < 3 ? 0 : 4));
        for (let n = 0; n < 5; n++) {
          const u = ((t - 9) * 0.35 + n / 5 + k * 0.13) % 1;
          items.push({ curve: path, u, len: 0.035, gain: pg, width: 0.16 });
        }
      }
    }
    g.pulses.set(items);

    // SM close-up 12.5 s on
    if (t > 12.5) {
      const d = g.smDetail(SM);
      const w = t - 13;
      d.setLanes((p, kind, idx) => {
        const lane = kind === 'fp32' ? idx : idx * 2;
        return 0.15 + 0.85 * (Math.sin(w * 6 - lane * 0.35 - p * 0.8) > 0.2 ? 1 : 0);
      });
      d.setTensor((p, i, j, k) => (((Math.floor(w * 4) + i + j + k + p) % 4) === 0 ? 1 : 0.05));
      d.setUnits((u) => (u === 'regfile' ? 0.25 + 0.2 * Math.sin(w * 5) : u === 'sched' ? 0.3 : 0));
    } else g.hideSMDetail();

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    comp.draw(renderer, this.ms.texture, out);

    // labels (screen-space), to judge the model
    const c = this.ui.ctx;
    this.ui.clear();
    c.font = font(this.famM, 22);
    const label = (p: THREE.Vector3, s: string, a: number) => {
      if (a <= 0) return;
      const v = g.group.localToWorld(p.clone()).project(cam);
      if (v.z > 1) return;
      const x = (v.x * 0.5 + 0.5) * W, y = (-v.y * 0.5 + 0.5) * H;
      c.fillStyle = rgba('bone', 0.85 * a);
      c.fillText(s, x + 10, y - 10);
      c.fillStyle = rgba('signal', a);
      c.fillRect(x - 3, y - 3, 6, 6);
    };
    const la = t < 5 ? smoothstep(1, 2, t) : t < 9 ? 1 - smoothstep(8.6, 9, t) : 0;
    label(new THREE.Vector3(-4, H100_Y.dieTop + ex * 38, 6), 'GH100 die · 814 mm²', la);
    label(g.hbm(0).pos, 'HBM3 stack', la);
    label(g.hbm(5).pos, 'spare site', la * 0.8);
    label(new THREE.Vector3(-45, 5, 0), 'VRM', la * 0.8);
    if (t > 9 && t < 13) label(g.smPos(SM), 'SM', smoothstep(10.5, 11.5, t));
    if (t > 14.5) {
      const d = g.smDetail(SM);
      const at = (n: string) => d.group.localToWorld(d.anchor(n).clone());
      const inv = new THREE.Matrix4().copy(g.group.matrixWorld).invert();
      const L = (n: string, s: string) => label(at(n).applyMatrix4(inv), s, smoothstep(14.5, 15.3, t));
      L('p0.regfile', 'register file');
      L('p0.tensor', 'tensor core');
      L('p1.sched', 'warp scheduler');
      L('l1', 'L1 / shared memory · 256 KB');
    }
    c.fillStyle = rgba('ash', 0.8);
    c.fillText(`h100demo  ${t.toFixed(2)} s  explode ${ex.toFixed(2)}`, 60, H - 50);
    comp.draw(renderer, this.ui.upload(), out);
    return { bloom: 0.55, bloomThreshold: 0.85 };
  }
}
