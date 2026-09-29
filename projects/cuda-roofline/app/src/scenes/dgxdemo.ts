// Preview of the shared DGX H100 model (scenes/_dgx.ts); not part of the song. 20 s:
// 0-4 orbit around the closed node, 4-9 teardown (explode), 9-13 dive to the baseboard while traffic
// ramps idle -> all-reduce, 13-18 fly along one NVLink trace as the flicker speeds up, 18-20 pull out.
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp, smoothstep } from '../engine/util';
import { DGX } from './_dgx';

type K = [number, [number, number, number], [number, number, number]];
// [local time, camera position, look-at] in mm
const KEYS: K[] = [
  [0, [980, 560, 1380], [0, 170, 0]],
  [4, [-760, 620, 1260], [0, 180, 20]],
  [9, [260, 1450, 1650], [0, 260, 360]],
  [13, [120, 470, 980], [0, 250, 560]],
  [18, [40, 330, 760], [0, 230, 600]],
  [20, [620, 1000, 1700], [0, 240, 320]],
];

export default class DgxDemo extends Scene {
  dgx = new DGX();
  cam = new THREE.PerspectiveCamera(32, W / H, 0.2, 8000);
  text = new Layer2D();
  famM = F.mono(500);

  override init() { this.dgx.init(this.ctx.renderer); }

  private keyed(lt: number) {
    let i = 0;
    while (i < KEYS.length - 2 && KEYS[i + 1]![0] <= lt) i++;
    const [t0, p0, l0] = KEYS[i]!, [t1, p1, l1] = KEYS[i + 1]!;
    const k = ease.inOutCubic(clamp((lt - t0) / (t1 - t0)));
    const P = new THREE.Vector3(lerp(p0[0], p1[0], k), lerp(p0[1], p1[1], k), lerp(p0[2], p1[2], k));
    const L = new THREE.Vector3(lerp(l0[0], l1[0], k), lerp(l0[1], l1[1], k), lerp(l0[2], l1[2], k));
    return { P, L };
  }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const lt = f.lt, d = this.dgx;
    d.setExplode(ease.inOutCubic(clamp((lt - 4) / 5)));
    const ramp = clamp((lt - 9) / 4), dive = clamp((lt - 13) / 5);
    d.setTraffic(lt < 9 ? { pattern: 'idle', intensity: 0.3, rate: 1 }
      : { pattern: 'allreduce', intensity: 0.25 + 0.75 * ramp, rate: 1 + 2 * ramp + 5 * dive });
    d.update(f.t);
    // camera: keyframes, blended into a trace-following dive between 13 and 18 s
    let { P, L } = this.keyed(lt);
    const wIn = smoothstep(12.2, 13.6, lt) * (1 - smoothstep(17.2, 18.4, lt));
    if (wIn > 0) {
      const curve = d.link(5, 1, 1);
      const u = 0.06 + 0.8 * ease.inOutQuad(clamp((lt - 12.6) / 5.6));
      const cp = curve.getPoint(Math.max(0, u - 0.07)).add(new THREE.Vector3(0, 13, 0));
      const cl = curve.getPoint(Math.min(1, u + 0.08));
      P = P.lerp(cp, wIn); L = L.lerp(cl, wIn);
    }
    this.cam.position.copy(P);
    this.cam.lookAt(L);
    this.cam.near = wIn > 0.5 ? 0.2 : 2;
    this.cam.updateProjectionMatrix();
    d.render(renderer, out, this.cam);

    // judging labels
    const c = this.text.ctx;
    this.text.clear();
    c.font = font(this.famM, 24);
    const lab = (s: string, a: number, x = 120, y = 980) => { if (a > 0) { c.fillStyle = rgba('bone', 0.85 * a); c.fillText(s, x, y); } };
    lab('DGX H100 · 8U · 482 × 356 × 897 mm', 1 - smoothstep(3.5, 4.2, lt));
    lab('8 × H100 SXM5  ·  4 × NVSwitch', smoothstep(5, 5.6, lt) * (1 - smoothstep(9, 9.6, lt)));
    lab('18 NVLinks per GPU: 5 + 4 + 4 + 5 to the four switches · all-reduce', smoothstep(9.6, 10.2, lt) * (1 - smoothstep(13, 13.5, lt)));
    lab('one NVLink = 2 lanes × 100 Gb/s · bits as they flicker by', smoothstep(13.6, 14.2, lt) * (1 - smoothstep(17.8, 18.3, lt)));
    comp.draw(renderer, this.text.upload(), out);
    return { bloom: 0.75, bloomThreshold: 0.88 };
  }
}
