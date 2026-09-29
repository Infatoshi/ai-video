// Shared 3D model: an NVIDIA DGX H100 node (v5, the multi-GPU world). Procedural three.js, units = mm.
//
// Verified structure (NVIDIA primary sources):
//   - 8U rackmount, 482.3 mm wide, 356 mm high, 897.1 mm deep; 8 H100 GPUs (640 GB); 4 NVSwitches;
//     2 Intel Xeon 8480C; 2 TB as 32 DIMMs; 8 U.2 NVMe cache drives + 2 M.2 boot drives; 6 x 3.3 kW PSUs;
//     4 OSFP ports for 8 single-port ConnectX-7 + 2 dual-port ConnectX-7 cards
//     (docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html).
//   - HGX H100 8-GPU: eight H100 + four third-gen NVSwitch; every GPU connects to all four switches
//     (developer.nvidia.com/blog/introducing-nvidia-hgx-h100-...).
//   - Each GPU's 18 NVLink4 links split 5 + 4 + 4 + 5 across the four switches (the outer two switches
//     take 5 per GPU, the inner two 4); each NVLink is x2 lanes at 100 Gb/s per diff pair
//     (NVIDIA, "NVLink4 NVSwitch", Hot Chips 34, hc34.hotchips.org).
//   Not verified, drawn generic and never labelled: the fan count/arrangement, exact board placement
//   of GPUs and switches, heatsink shapes, the motherboard layout.
//
// Coordinates: x = width (left -, right +), y = up (0 = chassis floor), z = depth (front of the box at
// +z, rear at -z). The GPU tray (HGX baseboard) is the top half, the motherboard tray below it.
//
// API (deterministic: everything is a function of the state you set + the time you pass to update()):
//   const dgx = new DGX();              // builds all geometry (sync)
//   dgx.init(renderer);                 // environment lighting (PMREM), once
//   dgx.setExplode(k);                  // 0..1: 0-0.3 lid lifts off, 0.25-0.65 GPU tray slides out the
//                                       // front and rises, 0.55-1 heatsinks lift off GPUs and switches
//   dgx.setTraffic({ pattern, intensity, rate });   // 'idle'|'ring'|'allreduce'|'broadcast'
//   dgx.update(t);                      // apply explode + traffic + fan spin for song time t
//   dgx.render(renderer, out, camera, { clear?: true, bg?: [r,g,b] });  // scene + traces + packets
//   Anchors: dgx.gpu(i) / dgx.nvswitch(j) (Object3D, i 0..7, j 0..3), dgx.link(i, j, l)
//   (world-space THREE.Curve along link l of GPU i -> switch j, live with the explode state; l indexes
//   the 5/4/4/5 links), dgx.gpuTray, dgx.moboTray, dgx.lid, dgx.gpuHeatsinks[i],
//   dgx.switchHeatsinks[j], dgx.worldPos(obj) helper, DGX.DIM, LINKS_PER_SWITCH.
//   Traffic: rate ~ packets speed and flicker speed (1 = calm, 4+ = frantic); intensity 0..1 = how
//   many links carry packets and how bright. Packets are drawn as short runs of "bits" (dashes) that
//   read individually when the camera is within a few mm of a trace.
import * as THREE from 'three';
import { LineBatch } from '../engine/lines';
import { LIN } from '../engine/palette';
import { clamp, ease, frameIdx, hash, mulberry32 } from '../engine/util';

export const LINKS_PER_SWITCH = [5, 4, 4, 5] as const;
export type TrafficPattern = 'idle' | 'ring' | 'allreduce' | 'broadcast';
export interface Traffic { pattern: TrafficPattern; intensity: number; rate: number }

const DIM = { w: 482.3, h: 356, d: 897.1 };
const X0 = -DIM.w / 2, X1 = DIM.w / 2, Z0 = -DIM.d / 2, Z1 = DIM.d / 2;
// GPU tray (top half)
const TRAY_Y = 184;                    // tray pan top
const BOARD_Y = TRAY_Y + 5;            // baseboard top surface
const BOARD = { x0: -226, x1: 226, z0: -432, z1: 360 };
const GPU_X = [-168, -56, 56, 168];
const GPU_Z = [-318, -150];           // rear row, front row (module centres)
const MOD = { w: 102, d: 150, t: 3 };  // SXM module board footprint
const SW_Z = 118, SW_X = [-168, -56, 56, 168];
const PKG = { gpu: 58, sw: 46 };
const HS = { gpuH: 128, swH: 70 };

const col = (k: keyof typeof LIN, m = 1) => new THREE.Color().setRGB(LIN[k][0] * m, LIN[k][1] * m, LIN[k][2] * m, THREE.LinearSRGBColorSpace);
const colRGB = (r: number, g: number, b: number) => new THREE.Color().setRGB(r, g, b, THREE.LinearSRGBColorSpace);

/** Curve in a group's local space, read in world space through the group's current matrixWorld. */
class WorldCurve extends THREE.Curve<THREE.Vector3> {
  constructor(private local: THREE.Curve<THREE.Vector3>, private owner: THREE.Object3D) { super(); }
  override getPoint(u: number, target = new THREE.Vector3()) {
    return this.local.getPoint(u, target).applyMatrix4(this.owner.matrixWorld);
  }
}

interface Link { gpu: number; sw: number; l: number; curve: THREE.Curve<THREE.Vector3>; pts: Float32Array; cum: Float32Array; len: number; lanes: [Float32Array, Float32Array] }

export class DGX {
  static DIM = DIM;
  readonly scene = new THREE.Scene();
  readonly root = new THREE.Group();
  readonly chassis = new THREE.Group();
  readonly lid = new THREE.Group();
  readonly gpuTray = new THREE.Group();
  readonly moboTray = new THREE.Group();
  readonly gpus: THREE.Group[] = [];
  readonly switches: THREE.Group[] = [];
  readonly gpuHeatsinks: THREE.Group[] = [];
  readonly switchHeatsinks: THREE.Group[] = [];
  private links: Link[] = [];
  private lines = new LineBatch(60000, { screen2D: false, worldWidth: true, depthTest: true, blend: 'add' });
  private fans: THREE.InstancedMesh | null = null;
  private fanCenters: THREE.Vector3[] = [];
  private leds: THREE.InstancedMesh | null = null;
  private explode = 0;
  private traffic: Traffic = { pattern: 'idle', intensity: 0.2, rate: 1 };
  private t = 0;
  private mats: Record<string, THREE.Material> = {};
  private edgeMat = new THREE.LineBasicMaterial({ color: col('bone', 0.22), transparent: true, opacity: 1, depthWrite: false });
  private m4 = new THREE.Matrix4();
  private v = new THREE.Vector3();
  private w = new THREE.Vector3();

  constructor() {
    this.scene.background = null;
    this.scene.add(this.root);
    this.root.add(this.chassis, this.lid, this.gpuTray, this.moboTray);
    this.buildMaterials();
    this.buildChassis();
    this.buildLid();
    this.buildMobo();
    this.buildGpuTray();
    this.buildLights();
    this.root.updateMatrixWorld(true);
  }

  /** Environment reflections for the metals (RoomEnvironment through PMREM). Call once. */
  init(renderer: THREE.WebGLRenderer) {
    if (this.scene.environment) return;
    // a dark studio: a box room with a few soft bone light panels and one warm strip, for reflections only
    const env = new THREE.Scene();
    const room = new THREE.Mesh(new THREE.BoxGeometry(10, 10, 10), new THREE.MeshBasicMaterial({ color: colRGB(0.012, 0.012, 0.013), side: THREE.BackSide }));
    env.add(room);
    const panel = (w: number, h: number, c: THREE.Color, x: number, y: number, z: number, ry = 0, rx = 0) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: c, side: THREE.DoubleSide }));
      m.position.set(x, y, z); m.rotation.set(rx, ry, 0); env.add(m);
    };
    panel(6, 1.2, col('bone', 1.6), 0, 4.9, 0, 0, Math.PI / 2);        // overhead softbox
    panel(1.2, 4, col('bone', 0.9), -4.9, 1, 0, Math.PI / 2);          // left strip
    panel(3, 0.3, col('ember', 0.35), 0, 0.5, -4.9);                   // warm rear strip
    panel(2, 2, col('ash', 0.5), 4.9, 2, 2, -Math.PI / 2);             // right bounce
    const pm = new THREE.PMREMGenerator(renderer);
    this.scene.environment = pm.fromScene(env, 0.03).texture;
    this.scene.environmentIntensity = 0.55;
    pm.dispose();
  }

  // ------------------------------------------------------------------ public state
  setExplode(k: number) { this.explode = clamp(k, 0, 1); }
  setTraffic(tr: Partial<Traffic>) { this.traffic = { ...this.traffic, ...tr }; }
  gpu(i: number): THREE.Object3D { return this.gpus[i]!; }
  nvswitch(j: number): THREE.Object3D { return this.switches[j]!; }
  /** World-space curve along link `l` (0..LINKS_PER_SWITCH[j]-1) from GPU i's module edge to switch j. */
  link(i: number, j: number, l = 0): THREE.Curve<THREE.Vector3> {
    const L = this.links.find((x) => x.gpu === i && x.sw === j && x.l === Math.min(l, LINKS_PER_SWITCH[j]! - 1))!;
    return L.curve;
  }
  worldPos(o: THREE.Object3D, target = new THREE.Vector3()) { return o.getWorldPosition(target); }

  /** Apply explode, fan spin and LED state for time t (and remember t for the traffic draw). */
  update(t: number) {
    this.t = t;
    const k = this.explode;
    const kl = ease.inOutCubic(clamp(k / 0.3));
    this.lid.position.set(0, 330 * kl, -40 * kl);
    this.lid.rotation.set(-0.12 * kl, 0, 0);
    const kt = ease.inOutCubic(clamp((k - 0.25) / 0.4));
    this.gpuTray.position.set(0, 90 * kt, 560 * kt);
    const kh = clamp((k - 0.55) / 0.45);
    this.gpuHeatsinks.forEach((h, i) => {
      const s = ease.outCubic(clamp(kh * 1.6 - (i % 4) * 0.12 - Math.floor(i / 4) * 0.05));
      h.position.y = h.userData.y0 + 190 * s;
      h.rotation.x = -0.05 * s;
    });
    this.switchHeatsinks.forEach((h, j) => {
      const s = ease.outCubic(clamp(kh * 1.6 - 0.25 - j * 0.1));
      h.position.y = h.userData.y0 + 140 * s;
    });
    // side walls part a little so the exploded box reads as a kit of parts
    const ks = ease.inOutCubic(clamp((k - 0.1) / 0.4));
    this.chassis.children.forEach((c) => {
      if (c.userData.side) c.position.x = (c.userData.x0 ?? 0) + c.userData.side * 36 * ks;
    });
    // fans: blades spin (a few rev/s; deterministic)
    if (this.fans) {
      const B = 7, rot = t * 7.5;
      this.fanCenters.forEach((c, f) => {
        for (let b = 0; b < B; b++) {
          const a = rot + (b / B) * Math.PI * 2 + f * 0.37;
          this.m4.makeRotationZ(a).setPosition(c.x, c.y, c.z);
          this.fans!.setMatrixAt(f * B + b, this.m4);
        }
      });
      this.fans.instanceMatrix.needsUpdate = true;
    }
    this.root.updateMatrixWorld(true);
  }

  // ------------------------------------------------------------------ render
  render(renderer: THREE.WebGLRenderer, out: THREE.WebGLRenderTarget, camera: THREE.Camera, o: { clear?: boolean; bg?: [number, number, number] } = {}) {
    if (o.clear ?? true) {
      const bg = o.bg ?? LIN.ink;
      renderer.setRenderTarget(out);
      renderer.setClearColor(colRGB(bg[0], bg[1], bg[2]), 1);
      renderer.clear(true, true, true);
    } else {
      renderer.setRenderTarget(out);
      renderer.clearDepth();
    }
    renderer.render(this.scene, camera);
    this.drawTraces(camera);
    this.lines.render(renderer, out, camera);
  }

  // ------------------------------------------------------------------ traces + packets
  private drawTraces(camera: THREE.Camera) {
    const L = this.lines;
    L.clear();
    const tr = this.traffic, t = this.t;
    const M = this.gpuTray.matrixWorld;
    const cam = camera.getWorldPosition(this.w);
    const a = this.tmpA, b = this.tmpB;
    // base traces: every lane as a dim copper-orange hairline (2 lanes per link), brighter when busy
    const base = LIN.blood;
    for (const lk of this.links) {
      const glow = 0.06 + 0.26 * this.linkActivity(lk, t);
      for (const lane of lk.lanes) {
        a.set(lane[0]!, lane[1]!, lane[2]!).applyMatrix4(M);
        for (let i = 3; i < lane.length; i += 3) {
          b.set(lane[i]!, lane[i + 1]!, lane[i + 2]!).applyMatrix4(M);
          L.seg(a.x, a.y, a.z, b.x, b.y, b.z, 0.26, base[0] * glow, base[1] * glow, base[2] * glow, 1);
          a.copy(b);
        }
      }
    }
    // packets: runs of bits sliding along the lanes. Flicker is keyed to the 60 fps frame index so it
    // never changes inside one frame's shutter; its cadence rises with the rate.
    const rate = Math.max(0.05, tr.rate), inten = clamp(tr.intensity);
    const flick = Math.floor((frameIdx(t) * Math.min(30, 6 * rate)) / 60);
    const em = LIN.ember, sg = LIN.signal;
    for (const lk of this.links) {
      const act = this.linkActivity(lk, t);
      if (act <= 0.02) continue;
      const nPk = Math.max(1, Math.round(1 + 5 * act * inten));
      const speed = (70 + 150 * rate) / lk.len;                 // curve fractions per second
      const bright = (1.4 + 2.6 * inten) * act;
      for (let d = 0; d < 2; d++) {
        const dir = this.linkDir(lk, t, d);
        if (dir === 0) continue;
        const lane = lk.lanes[d]!;
        for (let k = 0; k < nPk; k++) {
          let u = (hash(lk.gpu, lk.sw, lk.l, d, k) + speed * t) % 1;
          if (dir < 0) u = 1 - u;
          const s0 = u * lk.len;
          // bits resolve only when the camera is close to this packet
          this.pointAt(lk, lane, s0, a).applyMatrix4(M);
          const near = a.distanceTo(cam) < 160;
          const bits = 12, bitL = 0.7, gap = 0.45;
          if (!near) {
            this.segAlong(lk, lane, s0, s0 - dir * bits * (bitL + gap), 0.34, [em[0] * bright, em[1] * bright, em[2] * bright]);
            continue;
          }
          for (let q = 0; q < bits; q++) {
            if (hash(lk.gpu * 31 + lk.sw, lk.l * 7 + k, q, flick) <= 0.42) continue;
            const sa = s0 - dir * q * (bitL + gap), sb = sa - dir * bitL;
            const c = q === 0 ? sg : em, m = bright * (q === 0 ? 1.4 : 1);
            this.segAlong(lk, lane, sa, sb, 0.3, [c[0] * m, c[1] * m, c[2] * m]);
          }
        }
      }
    }
  }

  /** 0..1: how busy a link is at t under the current traffic pattern. */
  private linkActivity(lk: Link, t: number): number {
    const tr = this.traffic, inten = clamp(tr.intensity);
    switch (tr.pattern) {
      case 'idle': return hash(lk.gpu, lk.sw, lk.l, Math.floor(t * 0.7 * tr.rate)) > 0.72 ? 0.35 * Math.max(0.3, inten) : 0;
      case 'allreduce': return inten;
      case 'ring': {
        const hop = Math.floor(t * tr.rate * 1.5) % 8;
        return lk.gpu === hop || lk.gpu === (hop + 1) % 8 ? inten : 0.12 * inten;
      }
      case 'broadcast': {
        const ph = (t * tr.rate * 0.8) % 1;
        return lk.gpu === 0 ? inten : ph > 0.35 ? inten : 0.08 * inten;
      }
    }
  }

  /** Packet direction on lane d: +1 GPU -> switch, -1 switch -> GPU, 0 = none. */
  private linkDir(lk: Link, t: number, d: number): number {
    const tr = this.traffic;
    switch (tr.pattern) {
      case 'idle': return d === 0 ? 1 : -1;
      case 'allreduce': {
        // reduce-scatter (out) then all-gather (in), alternating; the two lanes run opposite ways
        const phase = Math.floor(t * tr.rate * 0.5) % 2;
        return (d === 0) === (phase === 0) ? 1 : -1;
      }
      case 'ring': {
        const hop = Math.floor(t * tr.rate * 1.5) % 8;
        if (lk.gpu === hop) return d === 0 ? 1 : 0;
        if (lk.gpu === (hop + 1) % 8) return d === 1 ? -1 : 0;
        return d === 0 ? 1 : -1;
      }
      case 'broadcast': return lk.gpu === 0 ? (d === 0 ? 1 : 0) : (d === 1 ? -1 : 0);
    }
  }

  private pointAt(lk: Link, lane: Float32Array, s: number, out: THREE.Vector3) {
    const n = lk.cum.length;
    s = clamp(s, 0, lk.len);
    let i = 1;
    while (i < n - 1 && lk.cum[i]! < s) i++;
    const s0 = lk.cum[i - 1]!, s1 = lk.cum[i]!, f = s1 > s0 ? (s - s0) / (s1 - s0) : 0;
    out.set(
      lane[(i - 1) * 3]! + (lane[i * 3]! - lane[(i - 1) * 3]!) * f,
      lane[(i - 1) * 3 + 1]! + (lane[i * 3 + 1]! - lane[(i - 1) * 3 + 1]!) * f,
      lane[(i - 1) * 3 + 2]! + (lane[i * 3 + 2]! - lane[(i - 1) * 3 + 2]!) * f,
    );
    return out;
  }

  private tmpA = new THREE.Vector3(); private tmpB = new THREE.Vector3();
  private tmpC = new THREE.Vector3(); private tmpD = new THREE.Vector3();
  private segAlong(lk: Link, lane: Float32Array, sa: number, sb: number, w: number, c: [number, number, number]) {
    if ((sa < 0 && sb < 0) || (sa > lk.len && sb > lk.len)) return;
    const M = this.gpuTray.matrixWorld, A = this.tmpC, B = this.tmpD;
    this.pointAt(lk, lane, sa, A).applyMatrix4(M);
    this.pointAt(lk, lane, sb, B).applyMatrix4(M);
    this.lines.seg(A.x, A.y + 0.05, A.z, B.x, B.y + 0.05, B.z, w, c[0], c[1], c[2], 1);
  }

  // ------------------------------------------------------------------ build: materials
  private buildMaterials() {
    const S = (o: THREE.MeshStandardMaterialParameters) => new THREE.MeshStandardMaterial(o);
    this.mats = {
      shell: S({ color: col('graphite', 0.16), metalness: 0.55, roughness: 0.62 }),
      shellIn: S({ color: col('graphite', 0.1), metalness: 0.6, roughness: 0.7 }),
      pcb: S({ color: col('ink2', 0.9), metalness: 0.1, roughness: 0.78 }),
      pcbEdge: S({ color: col('graphite', 0.25), metalness: 0.2, roughness: 0.6 }),
      alu: S({ color: col('ash', 0.55), metalness: 1.0, roughness: 0.32 }),
      aluDark: S({ color: col('graphite', 0.45), metalness: 1.0, roughness: 0.4 }),
      die: S({ color: col('graphite', 0.12), metalness: 1.0, roughness: 0.12 }),
      substrate: S({ color: col('graphite', 0.18), metalness: 0.2, roughness: 0.55 }),
      hbm: S({ color: col('graphite', 0.3), metalness: 0.6, roughness: 0.3 }),
      comp: S({ color: col('graphite', 0.28), metalness: 0.5, roughness: 0.45 }),
      plastic: S({ color: col('ink', 1.2), metalness: 0.0, roughness: 0.85 }),
      dimm: S({ color: col('graphite', 0.22), metalness: 0.4, roughness: 0.5 }),
      fan: S({ color: col('ink2', 1.4), metalness: 0.0, roughness: 0.6, side: THREE.DoubleSide }),
      port: S({ color: col('ash', 0.35), metalness: 0.9, roughness: 0.35 }),
      led: new THREE.MeshBasicMaterial({ color: col('ember', 2.2) }),
      ledGreenless: new THREE.MeshBasicMaterial({ color: col('bone', 1.4) }),
    };
  }

  private box(w: number, h: number, d: number, mat: THREE.Material, x: number, y: number, z: number, parent: THREE.Object3D, edges = true) {
    const g = new THREE.BoxGeometry(w, h, d);
    const m = new THREE.Mesh(g, mat);
    m.position.set(x, y, z);
    parent.add(m);
    if (edges) {
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), this.edgeMat);
      m.add(e);
    }
    return m;
  }

  // ------------------------------------------------------------------ build: chassis
  private buildChassis() {
    const P = this.chassis, th = 3;
    this.box(DIM.w, th, DIM.d, this.mats.shell!, 0, th / 2, 0, P);                              // floor
    const sL = this.box(th, DIM.h, DIM.d, this.mats.shell!, X0 + th / 2, DIM.h / 2, 0, P); sL.userData.side = -1; sL.userData.x0 = sL.position.x;
    const sR = this.box(th, DIM.h, DIM.d, this.mats.shell!, X1 - th / 2, DIM.h / 2, 0, P); sR.userData.side = 1; sR.userData.x0 = sR.position.x;
    // rack ears
    for (const s of [-1, 1]) { const e = this.box(18, DIM.h - 10, 3, this.mats.shell!, s * (DIM.w / 2 + 9), DIM.h / 2, Z1 - 20, P); e.userData.side = s; e.userData.x0 = e.position.x; }
    // rear panel with PSU bays, OSFP cages, RJ45 and a mid divider
    const rear = new THREE.Group(); P.add(rear);
    this.box(DIM.w - 6, DIM.h - 6, th, this.mats.shellIn!, 0, DIM.h / 2, Z0 + th / 2, rear);
    // PSUs: 6 across the bottom rear (bodies run into the box)
    for (let i = 0; i < 6; i++) {
      const x = X0 + 10 + 38 + i * 77;
      this.box(72, 40, 280, this.mats.aluDark!, x, 26, Z0 + 140, P);
      this.box(60, 6, 4, this.mats.port!, x, 26, Z0 - 1, P, false);                               // handle
      this.box(10, 3, 1, this.mats.led as THREE.Material, x + 25, 38, Z0 - 0.5, P, false);          // status LED
    }
    // OSFP cages (4 for the 8 single-port CX7) + 2 dual-port CX7 + RJ45s on the motherboard I/O band
    for (let i = 0; i < 4; i++) this.box(24, 14, 30, this.mats.port!, -150 + i * 34, 70, Z0 + 12, P);
    for (let i = 0; i < 4; i++) this.box(18, 12, 26, this.mats.port!, 30 + i * 26, 70, Z0 + 12, P);
    for (let i = 0; i < 3; i++) this.box(14, 12, 20, this.mats.plastic!, 150 + i * 20, 70, Z0 + 10, P);
    // front: bezel frame, fan wall for the GPU tray, drive bays for the U.2 cache below
    const fr = new THREE.Group(); P.add(fr);
    const fz = Z1 - 6;
    this.box(DIM.w - 6, 8, 12, this.mats.shell!, 0, DIM.h - 4, fz, fr);
    this.box(DIM.w - 6, 8, 12, this.mats.shell!, 0, 178, fz, fr);
    // fans (count not verified: drawn as a generic wall, never labelled)
    const nx = 6, ny = 2, fw = 74;
    const fanGeo = new THREE.BoxGeometry(fw * 0.47, 2, 1.2);
    fanGeo.translate(fw * 0.24, 0, 0);
    this.fans = new THREE.InstancedMesh(fanGeo, this.mats.fan!, nx * ny * 7);
    for (let yy = 0; yy < ny; yy++) for (let xx = 0; xx < nx; xx++) {
      const x = (xx - (nx - 1) / 2) * 78, y = 222 + yy * 84, z = Z1 - 24;
      this.fanCenters.push(new THREE.Vector3(x, y, z));
      // fan housing: a square frame + hub + grille rings
      const hz = z;
      const frame = new THREE.Mesh(new THREE.BoxGeometry(fw + 2, fw + 2, 28), this.mats.plastic!);
      frame.position.set(x, y, hz - 10);
      const hole = new THREE.Mesh(new THREE.CylinderGeometry(fw * 0.47, fw * 0.47, 29, 40, 1, true), this.mats.fan!);
      hole.rotation.x = Math.PI / 2; hole.position.set(x, y, hz - 10);
      fr.add(frame, hole);
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(12, 12, 8, 24), this.mats.plastic!);
      hub.rotation.x = Math.PI / 2; hub.position.set(x, y, z + 1); fr.add(hub);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(fw * 0.36, 0.7, 6, 48), this.mats.port!);
      ring.position.set(x, y, z + 4); fr.add(ring);
      const ring2 = new THREE.Mesh(new THREE.TorusGeometry(fw * 0.2, 0.7, 6, 40), this.mats.port!);
      ring2.position.set(x, y, z + 4); fr.add(ring2);
    }
    this.fans.frustumCulled = false;
    fr.add(this.fans);
    // 8 U.2 drive sleds (front, lower) + front console
    for (let i = 0; i < 8; i++) {
      const x = -196 + i * 50;
      this.box(44, 18, 12, this.mats.aluDark!, x, 150, fz - 4, fr);
      this.box(4, 3, 1, this.mats.ledGreenless as THREE.Material, x + 15, 146, fz + 2.5, fr, false);
    }
    for (let i = 0; i < 2; i++) this.box(DIM.w - 12, 2, 10, this.mats.port!, 0, 120 + i * 12, fz - 2, fr, false);   // vent slats band
    this.box(120, 30, 8, this.mats.plastic!, 150, 100, fz, fr);                                     // console
    this.box(6, 3, 1, this.mats.led as THREE.Material, 190, 104, fz + 4.5, fr, false);
    // mid shelf between the two trays
    this.box(DIM.w - 6, 3, DIM.d - 40, this.mats.shellIn!, 0, 178, 0, P);
  }

  private buildLid() {
    const L = this.lid;
    this.box(DIM.w, 3, DIM.d, this.mats.shell!, 0, DIM.h - 1.5, 0, L);
    // a few stamped ribs so the lid reads as sheet metal
    for (let i = 0; i < 5; i++) this.box(DIM.w - 40, 1.5, 6, this.mats.shell!, 0, DIM.h + 0.6, -300 + i * 150, L, false);
  }

  // ------------------------------------------------------------------ build: motherboard tray
  private buildMobo() {
    const P = this.moboTray;
    const y0 = 60;
    this.box(DIM.w - 12, 3, DIM.d - 80, this.mats.shellIn!, 0, y0, -10, P);
    this.box(DIM.w - 30, 2, 560, this.mats.pcb!, 0, y0 + 2.5, -60, P);
    // two CPUs with fin-stack heatsinks, 16 DIMMs around each (32 total, as in the spec)
    const dimmGeo = new THREE.BoxGeometry(2.2, 32, 133);
    const dimms = new THREE.InstancedMesh(dimmGeo, this.mats.dimm!, 32);
    let di = 0;
    for (const cx of [-110, 110]) {
      const cz = -80;
      this.box(78, 6, 78, this.mats.substrate!, cx, y0 + 6, cz, P);
      const hs = new THREE.Group(); hs.position.set(cx, y0 + 9, cz); P.add(hs);
      this.box(84, 5, 84, this.mats.alu!, 0, 2.5, 0, hs);
      const fin = new THREE.BoxGeometry(84, 60, 0.8);
      const fins = new THREE.InstancedMesh(fin, this.mats.alu!, 34);
      for (let f = 0; f < 34; f++) { this.m4.makeTranslation(0, 35, -41 + f * 2.5); fins.setMatrixAt(f, this.m4); }
      hs.add(fins);
      for (const side of [-1, 1]) for (let k = 0; k < 8; k++) {
        this.m4.makeTranslation(cx + side * (52 + k * 5.2), y0 + 19, cz);
        dimms.setMatrixAt(di++, this.m4);
      }
    }
    P.add(dimms);
    // PCIe risers + network cards (ConnectX-7 boards), M.2 boot drives
    for (let i = 0; i < 4; i++) {
      this.box(2, 70, 160, this.mats.pcb!, -180 + i * 22, y0 + 40, -300, P);
      this.box(12, 20, 30, this.mats.aluDark!, -180 + i * 22 + 6, y0 + 40, -300, P, false);
    }
    for (let i = 0; i < 2; i++) this.box(2, 60, 150, this.mats.pcb!, 150 + i * 24, y0 + 36, -310, P);
    for (let i = 0; i < 2; i++) this.box(22, 2, 80, this.mats.comp!, 20 + i * 30, y0 + 5, 120, P);
    // VRM rows
    const vrm = new THREE.InstancedMesh(new THREE.BoxGeometry(8, 7, 8), this.mats.comp!, 24);
    for (let i = 0; i < 24; i++) { this.m4.makeTranslation(-110 + (i % 12) * 20 - (i < 12 ? 0 : -2), y0 + 6, i < 12 ? 0 : -160); vrm.setMatrixAt(i, this.m4); }
    P.add(vrm);
  }

  // ------------------------------------------------------------------ build: GPU tray (HGX baseboard)
  private buildGpuTray() {
    const T = this.gpuTray;
    this.box(DIM.w - 14, 4, DIM.d - 30, this.mats.shellIn!, 0, TRAY_Y - 2, -10, T);
    // the baseboard: solder-mask top with a procedural via field, copper pours and silkscreen outlines
    const bw = BOARD.x1 - BOARD.x0, bd = BOARD.z1 - BOARD.z0;
    const top = new THREE.MeshStandardMaterial({ map: this.boardTexture(bw, bd), color: col('bone', 0.55), metalness: 0.05, roughness: 0.8 });
    const bg = new THREE.BoxGeometry(bw, 3, bd);
    const board = new THREE.Mesh(bg, [this.mats.pcbEdge!, this.mats.pcbEdge!, top, this.mats.pcb!, this.mats.pcbEdge!, this.mats.pcbEdge!]);
    board.position.set(0, BOARD_Y - 1.5, (BOARD.z0 + BOARD.z1) / 2);
    board.add(new THREE.LineSegments(new THREE.EdgesGeometry(bg), this.edgeMat));
    T.add(board);
    // board-edge connectors to the motherboard (rear)
    for (let i = 0; i < 6; i++) this.box(60, 10, 10, this.mats.plastic!, -170 + i * 68, BOARD_Y + 5, BOARD.z0 + 8, T);
    // GPUs: 2 rows x 4
    for (let r = 0; r < 2; r++) for (let c = 0; c < 4; c++) {
      const i = r * 4 + c;
      const g = this.buildGpuModule(i);
      g.position.set(GPU_X[c]!, BOARD_Y, GPU_Z[r]!);
      T.add(g); this.gpus[i] = g;
      const hs = this.buildHeatsink(MOD.w - 6, MOD.d - 8, HS.gpuH, 34);
      hs.position.set(GPU_X[c]!, BOARD_Y + 16, GPU_Z[r]!);
      hs.userData.y0 = hs.position.y;
      T.add(hs); this.gpuHeatsinks[i] = hs;
    }
    // NVSwitches: a row of 4 in front of the GPUs
    for (let j = 0; j < 4; j++) {
      const s = this.buildSwitch(j);
      s.position.set(SW_X[j]!, BOARD_Y, SW_Z);
      T.add(s); this.switches[j] = s;
      const hs = this.buildHeatsink(70, 70, HS.swH, 22);
      hs.position.set(SW_X[j]!, BOARD_Y + 8, SW_Z);
      hs.userData.y0 = hs.position.y;
      T.add(hs); this.switchHeatsinks[j] = hs;
    }
    this.buildLinks();
    // board furniture: capacitor/VRM fields around every package (instanced)
    const n = 8 * 20 + 4 * 10;
    const caps = new THREE.InstancedMesh(new THREE.BoxGeometry(5, 4, 5), this.mats.comp!, n);
    const rng = mulberry32(7);
    let ci = 0;
    for (let i = 0; i < 8; i++) {
      const c = i % 4, r = Math.floor(i / 4);
      for (let k = 0; k < 20; k++) {
        const side = k < 10 ? -1 : 1;
        this.m4.makeTranslation(GPU_X[c]! + side * (MOD.w / 2 - 7), BOARD_Y + MOD.t + 2, GPU_Z[r]! - MOD.d / 2 + 12 + (k % 10) * 13 + rng() * 2);
        caps.setMatrixAt(ci++, this.m4);
      }
    }
    for (let j = 0; j < 4; j++) for (let k = 0; k < 10; k++) {
      this.m4.makeTranslation(SW_X[j]! - 30 + (k % 5) * 15, BOARD_Y + 2, SW_Z + (k < 5 ? -32 : 32));
      caps.setMatrixAt(ci++, this.m4);
    }
    T.add(caps);
  }

  /** Board-top texture (sRGB canvas): dark solder mask, via field, copper pours, silkscreen outlines. */
  private boardTexture(bw: number, bd: number) {
    const ppm = 5;                                     // px per mm
    const cv = document.createElement('canvas');
    cv.width = Math.round(bw * ppm); cv.height = Math.round(bd * ppm);
    const c = cv.getContext('2d')!;
    const X = (x: number) => (x - BOARD.x0) * ppm, Zc = (z: number) => (z - BOARD.z0) * ppm;
    c.fillStyle = '#16161a'; c.fillRect(0, 0, cv.width, cv.height);
    // copper pours (slightly lighter planes) under the power areas
    c.fillStyle = 'rgba(94,91,87,0.12)';
    for (let r = 0; r < 2; r++) for (let k = 0; k < 4; k++) c.fillRect(X(GPU_X[k]! - 58), Zc(GPU_Z[r]! - 82), 116 * ppm, 164 * ppm);
    // via field
    const rng = mulberry32(11);
    c.fillStyle = 'rgba(156,151,143,0.35)';
    for (let i = 0; i < 26000; i++) {
      const x = rng() * cv.width, y = rng() * cv.height;
      c.beginPath(); c.arc(x, y, 0.9 + rng() * 0.8, 0, Math.PI * 2); c.fill();
    }
    // via rows along the trace channels
    c.fillStyle = 'rgba(238,233,223,0.28)';
    for (let k = 0; k < 4; k++) for (let i = 0; i < 90; i++) {
      c.beginPath(); c.arc(X(SW_X[k]! - 24 + (i % 30) * 1.6), Zc(SW_Z - 40 - Math.floor(i / 30) * 4), 1.2, 0, Math.PI * 2); c.fill();
    }
    // silkscreen: module and switch footprints, corner marks, a board-edge ruler
    c.strokeStyle = 'rgba(238,233,223,0.32)'; c.lineWidth = 2;
    for (let r = 0; r < 2; r++) for (let k = 0; k < 4; k++) c.strokeRect(X(GPU_X[k]! - MOD.w / 2 - 3), Zc(GPU_Z[r]! - MOD.d / 2 - 3), (MOD.w + 6) * ppm, (MOD.d + 6) * ppm);
    for (let k = 0; k < 4; k++) c.strokeRect(X(SW_X[k]! - 30), Zc(SW_Z - 30), 60 * ppm, 60 * ppm);
    c.lineWidth = 1.2;
    for (let i = 0; i < bd; i += 10) { c.beginPath(); c.moveTo(0, Zc(BOARD.z0 + i)); c.lineTo((i % 50 === 0 ? 6 : 3) * ppm, Zc(BOARD.z0 + i)); c.stroke(); }
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    return tex;
  }

  /** H100 SXM5 module, lite: module board, package substrate, dark mirror die, 6 HBM sites (3 a side). */
  private buildGpuModule(i: number) {
    const g = new THREE.Group(); g.name = `gpu${i}`;
    this.box(MOD.w, MOD.t, MOD.d, this.mats.pcb!, 0, MOD.t / 2 + 6, 0, g);
    // mezzanine connectors underneath (lift the module off the board)
    this.box(MOD.w - 20, 6, 24, this.mats.plastic!, 0, 3, -MOD.d / 2 + 22, g, false);
    this.box(MOD.w - 20, 6, 24, this.mats.plastic!, 0, 3, MOD.d / 2 - 22, g, false);
    const top = 6 + MOD.t;
    this.box(PKG.gpu, 2, PKG.gpu, this.mats.substrate!, 0, top + 1, 0, g);
    const die = this.box(26, 0.9, 31, this.mats.die!, 0, top + 2.45, 0, g);
    die.name = 'die';
    for (const s of [-1, 1]) for (let k = 0; k < 3; k++) {
      this.box(8, 1.3, 10.5, this.mats.hbm!, s * 19, top + 2.65, -12 + k * 12, g);
    }
    // stiffener frame
    const fr = PKG.gpu + 6;
    for (const s of [-1, 1]) {
      this.box(fr, 1.6, 3, this.mats.aluDark!, 0, top + 0.8, s * (fr / 2 - 1.5), g, false);
      this.box(3, 1.6, fr, this.mats.aluDark!, s * (fr / 2 - 1.5), top + 0.8, 0, g, false);
    }
    return g;
  }

  private buildSwitch(j: number) {
    const g = new THREE.Group(); g.name = `nvswitch${j}`;
    this.box(PKG.sw, 2, PKG.sw, this.mats.substrate!, 0, 1, 0, g);
    this.box(22, 0.9, 22, this.mats.die!, 0, 2.45, 0, g);
    return g;
  }

  private buildHeatsink(w: number, d: number, h: number, nFins: number) {
    const g = new THREE.Group();
    this.box(w, 7, d, this.mats.alu!, 0, 3.5, 0, g);
    // fins run front-to-back (airflow), spaced across the width; heat pipes as rounded bars
    const fin = new THREE.BoxGeometry(0.9, h - 12, d);
    const fins = new THREE.InstancedMesh(fin, this.mats.alu!, nFins);
    for (let f = 0; f < nFins; f++) {
      this.m4.makeTranslation(-w / 2 + 1 + (f * (w - 2)) / (nFins - 1), 7 + (h - 12) / 2, 0);
      fins.setMatrixAt(f, this.m4);
    }
    g.add(fins);
    const top = this.box(w, 2, d, this.mats.aluDark!, 0, h - 4, 0, g);
    top.name = 'cap';
    for (let p = 0; p < 4; p++) {
      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(3, 3, d + 6, 12), this.mats.alu!);
      pipe.rotation.x = Math.PI / 2; pipe.position.set(-w / 2 + 10 + p * ((w - 20) / 3), 10, 0);
      g.add(pipe);
    }
    return g;
  }

  /** 18 NVLinks per GPU (5/4/4/5 to switches 0..3), each x2 lanes, routed on the baseboard top. */
  private buildLinks() {
    const y = BOARD_Y + 0.12;
    for (let i = 0; i < 8; i++) {
      const c = i % 4, r = Math.floor(i / 4);
      const gx = GPU_X[c]!, gz = GPU_Z[r]! + MOD.d / 2;          // module front edge
      let slot = 0;
      for (let j = 0; j < 4; j++) {
        for (let l = 0; l < LINKS_PER_SWITCH[j]!; l++, slot++) {
          // launch points spread across the module's front edge; landing points spread along the
          // switch package's rear edge, grouped per GPU
          const sx = gx - 40 + slot * (80 / 17);
          const ex = SW_X[j]! - 21 + (i * 5.2) + l * 1.0;
          const ez = SW_Z - PKG.sw / 2 - 1.5;
          // rear-row GPUs route down the gap between the front-row modules
          const lane = r === 0 ? -1 : 0;
          const pts: THREE.Vector3[] = [];
          const zA = gz + 3;
          if (r === 0) {
            const gapX = (c < 2 ? -112 : 112) + (c === 0 || c === 3 ? 0 : 0) + (sx - gx) * 0.12 + (c % 2 === 0 ? -1 : 1) * 2;
            const jog = GPU_Z[1]! - MOD.d / 2 - 14;
            pts.push(new THREE.Vector3(sx, y, zA), new THREE.Vector3(sx, y, jog - 6), new THREE.Vector3(gapX, y, jog + 6),
              new THREE.Vector3(gapX, y, GPU_Z[1]! + MOD.d / 2 + 10), new THREE.Vector3(ex, y, ez - 26), new THREE.Vector3(ex, y, ez));
          } else {
            pts.push(new THREE.Vector3(sx, y, zA), new THREE.Vector3(sx, y, zA + 18), new THREE.Vector3(ex, y, ez - 26), new THREE.Vector3(ex, y, ez));
          }
          void lane;
          const local = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.2);
          const N = 48;
          const center = local.getSpacedPoints(N);
          const lanes: [Float32Array, Float32Array] = [new Float32Array((N + 1) * 3), new Float32Array((N + 1) * 3)];
          const cum = new Float32Array(N + 1);
          for (let k = 0; k <= N; k++) {
            const p = center[k]!;
            const q = center[Math.min(N, k + 1)]!, pp = center[Math.max(0, k - 1)]!;
            const tx = q.x - pp.x, tz = q.z - pp.z, tl = Math.hypot(tx, tz) || 1;
            const nx = -tz / tl, nz = tx / tl;                      // board-plane normal to the path
            for (let d = 0; d < 2; d++) {
              const o = (d === 0 ? -0.32 : 0.32);
              lanes[d][k * 3] = p.x + nx * o; lanes[d][k * 3 + 1] = p.y + (j % 2) * 0.02; lanes[d][k * 3 + 2] = p.z + nz * o;
            }
            if (k > 0) cum[k] = cum[k - 1]! + p.distanceTo(center[k - 1]!);
          }
          const flat = new Float32Array((N + 1) * 3);
          center.forEach((p, k) => { flat[k * 3] = p.x; flat[k * 3 + 1] = p.y; flat[k * 3 + 2] = p.z; });
          this.links.push({ gpu: i, sw: j, l, curve: new WorldCurve(local, this.gpuTray), pts: flat, cum, len: cum[N]!, lanes });
        }
      }
    }
  }

  // ------------------------------------------------------------------ lights
  private buildLights() {
    const key = new THREE.DirectionalLight(col('bone', 1), 2.2);
    key.position.set(-400, 900, 700);
    const fill = new THREE.DirectionalLight(col('ash', 1), 0.5);
    fill.position.set(700, 300, 200);
    const rim = new THREE.DirectionalLight(col('ember', 1), 0.3);
    rim.position.set(350, -60, -900);
    const amb = new THREE.AmbientLight(col('bone', 1), 0.06);
    this.scene.add(key, fill, rim, amb);
  }
}
