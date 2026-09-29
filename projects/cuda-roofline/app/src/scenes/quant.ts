// Store each number in 4 bits instead of 32, and there is 8× less data to move, while the value barely changes.
//
// `quant` (v5, CH9, 3D on the shared H100), 143.5-150.7 s. Camera always moving (scenes/_cam.ts), changes land
// on the 8th grid (scenes/_lock.ts).
//  A  "Thirty-two bits down to four, and it's fine,"
//     one number (CH09's 1.2) as a physical bar of 32 bit blocks hovering over the GH100 die while the
//     camera orbits it. "two": the "32 bits · FP32" label lights. "four" (one hit): every 8 blocks slam into
//     one, 32 -> 4, lit orange, reading 0011 (q = round(1.2 / (2.5 / 7)) = 3). "fine": 1.20 -> 1.07.
//  B  "Eight times less to move down the line,"
//     a whip down onto the interposer: the camera rides the HBM3 -> die link. Before "less" the lanes carry
//     a dense stream of long FP32 packets and the HBM stack burns; on "less" it thins to a sparse stream of
//     short INT4 packets, the stack calms, "8× less to move" (CH09: per billion parameters FP32 4 GB ->
//     INT4 500 MB). "line": the ride lands at the die's memory PHY.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { Lyrics, type Line } from '../engine/lyrics';
import { clamp, ease, frameIdx, hash, lerp } from '../engine/util';
import { q, snap, hit } from './_lock';
import { cumEnv, kickPush, whip } from './_cam';
import { H100, H100_Y, DIM, glowMat, type PulseItem } from './_h100';

// CH09's symmetric INT4 example: weight 1.2 in a group with absmax 2.5, clamp [-7, 7] -> s = 2.5 / 7,
// q = round(1.2 / s) = 3, dequantized 3 × s = 1.071. 1.2f = 0x3F99999A.
const FP32_BITS = '0' + '01111111' + '00110011001100110011010';
const INT4_BITS = '0011';

// the bit bar (mm, H100 group space): 32 blocks over the die centre, collapsing into 4
const BW = 0.28, BH = 0.26, BD = 0.46, BP = 0.335;          // small block: width, height, depth, pitch
const B4W = 1.3, B4P = 1.55;                              // merged block width and pitch
const BAR_Y = H100_Y.dieTop + 1.9;
const BAR_C = new THREE.Vector3(0, BAR_Y, 0.5);

const HBM_K = 1;          // the west middle HBM3 stack
const SM_I = 40;          // a west-side SM the link routes to (via L2)
const LANES = 8;          // parallel lanes of the ride (z offsets of the path)

interface Times {
  bracket: number; four: number; fine: number;
  swap: number;
  eight: number; less: number; move: number; land: number;
}

const lin = (k: keyof typeof LIN, s = 1): [number, number, number] => [LIN[k][0] * s, LIN[k][1] * s, LIN[k][2] * s];

export default class Quant extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(34, W / H, 0.05, 5000);
  gpu!: H100;
  rim!: THREE.DirectionalLight;
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();
  L1!: Line;
  L2!: Line;
  T!: Times;
  // bar
  body!: THREE.InstancedMesh;
  glow!: THREE.InstancedMesh;
  edges: THREE.LineSegments[] = [];
  private tmp = new THREE.Object3D();
  private col = new THREE.Color();
  // ride
  lanes: THREE.CurvePath<THREE.Vector3>[] = [];
  ride!: THREE.CurvePath<THREE.Vector3>;
  uI = 0; uEdge = 0; uEnd = 0;

  override async init() {
    const ly = this.ctx.lyrics, au = this.ctx.audio;
    this.L1 = ly.get('Thirty-two bits');
    this.L2 = ly.get('Eight times less');
    const w1 = this.L1.words, w2 = this.L2.words;
    const two = w1[0]!.syl?.[1]?.[0] ?? w1[0]!.start; // "Thir-ty-TWO"
    this.T = {
      bracket: q(au, two), four: q(au, w1[4]!.start), fine: q(au, w1[7]!.start),
      swap: au.timeOfBeat(Math.ceil(au.beatAt(w2[0]!.start - 0.4) * 2) / 2),
      eight: q(au, w2[0]!.start), less: q(au, w2[2]!.start), move: q(au, w2[4]!.start), land: q(au, w2[7]!.start),
    };

    this.gpu = new H100({ renderer: this.ctx.renderer });
    const rig = H100.rig();
    this.rim = rig.children.find((o) => o instanceof THREE.DirectionalLight && o.color.g < 0.5) as THREE.DirectionalLight;
    this.scene.add(this.gpu.group, rig);
    this.scene.environment = this.gpu.env;
    this.scene.environmentIntensity = 0.5;   // grazing views pick up a lot of environment otherwise
    this.scene.background = new THREE.Color().setRGB(LIN.ink[0], LIN.ink[1], LIN.ink[2], THREE.LinearSRGBColorSpace);

    // the bar: rounded dark-graphite blocks, a glow plate on top of each, bone hairline edges
    const geo = new RoundedBoxGeometry(1, 1, 1, 2, 0.08);
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(0.05, 0.05, 0.052), metalness: 0.7, roughness: 0.32 });
    this.body = new THREE.InstancedMesh(geo, mat, 32);
    this.body.frustumCulled = false;
    this.glow = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), glowMat(), 32);
    this.glow.frustumCulled = false;
    this.glow.setColorAt(0, this.col.setRGB(0, 0, 0));
    this.gpu.group.add(this.body, this.glow);
    const eg = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    for (let i = 0; i < 32; i++) {
      const e = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: new THREE.Color().setRGB(...lin('bone', 0.55)), transparent: true, opacity: 0.8 }));
      e.frustumCulled = false;
      this.edges.push(e);
      this.gpu.group.add(e);
    }

    // the ride: lanes = the HBM -> die path shifted across its corridor
    this.ride = this.gpu.pulsePath(HBM_K, SM_I);
    for (let l = 0; l < LANES; l++) {
      const dz = (l - (LANES - 1) / 2) * 0.2;
      const cp = new THREE.CurvePath<THREE.Vector3>();
      for (const c of this.ride.curves) {
        const lc = c as THREE.LineCurve3;
        cp.add(new THREE.LineCurve3(lc.v1.clone().add(new THREE.Vector3(0, 0, dz)), lc.v2.clone().add(new THREE.Vector3(0, 0, dz))));
      }
      this.lanes.push(cp);
    }
    // key points along the ride: where it reaches the interposer, the die edge, and where we stop
    const yI = H100_Y.intTop + 0.03;
    const N = 400;
    let uI = -1, uE = -1;
    for (let k = 0; k <= N; k++) {
      const u = k / N, p = this.ride.getPointAt(u);
      if (uI < 0 && Math.abs(p.y - yI) < 0.02) uI = u;
      if (uI >= 0 && uE < 0 && Math.abs(p.x) < DIM.die.w / 2 - 0.9) uE = u;
    }
    this.uI = Math.max(0, uI - 0.012);
    this.uEdge = uE < 0 ? 0.3 : uE;
    this.uEnd = Math.min(0.95, this.uEdge + 0.07);
  }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const au = this.ctx.audio;
    const t = f.t, T = this.T, g = this.gpu;
    const kFour = snap(au, t, T.four);
    const kLess = snap(au, t, T.less);
    const kLand = snap(au, t, T.land);
    const hFour = hit(au, t, T.four, 0.1);
    const hLand = hit(au, t, T.land, 0.12);
    const partB = t >= T.swap;

    // ── activity: the die breathes with the music; the HBM stack burns under FP32 and calms on INT4
    // (a calmer die than the default so the bar and the link stay the brightest things)
    g.update(t, { ...f.a, rms: f.a.rms * 0.55, drums: f.a.drums * 0.55, kick: f.a.kick * 0.5 });
    this.scene.environmentIntensity = partB ? 0.32 : 0.34;
    if (this.rim) this.rim.intensity = partB ? 0.35 : 1.3;   // in B the rim lights the very wall we face
    const fi = frameIdx(t) >> 1;
    const burn = partB ? lerp(1.35, 0.22, kLess) : 0.9;
    g.setHBMActivity((k) => (k === HBM_K ? burn : partB ? burn * 0.6 : 0.45) * (0.7 + 0.3 * (hash(k, fi, 11) < 0.6 ? 1 : 0.4)));

    // ── the bar (part A only)
    this.drawBar(t, kFour, hFour, partB);

    // ── packets along the lanes (dense FP32 until "less", then sparse, short INT4)
    this.drawTraffic(t, partB, kLess);

    // ── camera
    const shotA = this.shotA(t, kFour);
    const shotB = this.shotB(t);
    const k = partB ? whip(au, t, T.swap, 0.3) : 0;
    const pos = shotA.pos.clone().lerp(shotB.pos, k);
    const tgt = shotA.tgt.clone().lerp(shotB.tgt, k);
    const cam = this.cam;
    cam.position.copy(g.group.localToWorld(pos));
    cam.up.set(0, 1, 0);
    cam.lookAt(g.group.localToWorld(tgt));
    const fov = lerp(34, 50, k);
    if (cam.fov !== fov) cam.fov = fov;
    H100.fitClip(cam, pos.distanceTo(tgt));

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    comp.draw(renderer, this.ms.texture, out);

    // ── screen-space type
    const c = this.ui.ctx;
    this.ui.clear();
    if (!partB) this.drawBarType(c, t, kFour);
    else this.drawRideType(c, t, kLess, kLand, hLand);
    this.drawTag(c);
    this.drawLyric(c, t, partB ? this.L2 : this.L1, partB && t < this.L2.words[0]!.start);
    comp.draw(renderer, this.ui.upload(), out);

    return { bloom: 0.45, bloomThreshold: 0.9, vignette: 0.4, exposure: 0.92 + 0.2 * hFour + 0.12 * hLand };
  }

  // ─────────────────────────────────────────────────────────────── part A: the 32-bit bar
  private barLayout(i: number, kFour: number) {
    const slot = Math.floor(i / 8);
    const x0 = (i - 15.5) * BP, x1 = (slot - 1.5) * B4P;
    const x = lerp(x0, x1, kFour);
    const w = lerp(BW, B4W, kFour), h = lerp(BH, BH * 1.9, kFour), d = lerp(BD, BD * 1.5, kFour);
    return { x, w, h, d, slot, merged: kFour >= 1 && i % 8 !== 0 };
  }

  private drawBar(t: number, kFour: number, hFour: number, partB: boolean) {
    const au = this.ctx.audio;
    // the bar leaves on the swap (it's behind the camera by then); a gentle bob with the bass while it's up
    const gone = partB ? snap(au, t, this.T.swap, 0.25) : 0;
    const bob = 0.05 * Math.sin(t * 2.2) + 0.08 * hFour;
    for (let i = 0; i < 32; i++) {
      const L = this.barLayout(i, kFour);
      const s = L.merged || gone >= 1 ? 0 : 1 - gone;
      this.tmp.position.set(BAR_C.x + L.x, BAR_C.y + bob + L.h / 2, BAR_C.z);
      this.tmp.rotation.set(0, 0, 0);
      this.tmp.scale.set(L.w * s || 1e-5, L.h * s || 1e-5, L.d * s || 1e-5);
      this.tmp.updateMatrix();
      this.body.setMatrixAt(i, this.tmp.matrix);
      this.edges[i]!.matrixAutoUpdate = false;
      this.edges[i]!.matrix.copy(this.tmp.matrix);
      this.edges[i]!.visible = s > 0;
      // the glow plate on top: bone for a 1 bit, a faint ember for a 0, all signal once merged
      this.tmp.position.y = BAR_C.y + bob + L.h + 0.006;
      this.tmp.scale.set(L.w * 0.82 * s || 1e-5, 0.01, L.d * 0.8 * s || 1e-5);
      this.tmp.updateMatrix();
      this.glow.setMatrixAt(i, this.tmp.matrix);
      const bit = FP32_BITS[i] === '1';
      let c: [number, number, number];
      if (kFour > 0) c = lin('signal', 1.1 + 1.4 * hFour);
      else c = bit ? lin('bone', 0.5) : lin('ember', 0.02);
      this.glow.setColorAt(i, this.col.setRGB(c[0], c[1], c[2]));
    }
    this.body.instanceMatrix.needsUpdate = true;
    this.glow.instanceMatrix.needsUpdate = true;
    if (this.glow.instanceColor) this.glow.instanceColor.needsUpdate = true;
  }

  private shotA(t: number, kFour: number) {
    const au = this.ctx.audio;
    const t0 = this.ctx.start;
    // continuous orbit whose speed surges with the drums; a push in on "four" and a small dolly on each kick
    const ang = -0.42 + 0.06 * (t - t0) + 0.07 * (cumEnv(au, 'drums', t) - cumEnv(au, 'drums', t0));
    const r = lerp(11.5, 8.6, kFour) - 0.5 * kickPush(au, t);
    const el = lerp(0.46, 0.4, kFour);
    const tgt = BAR_C.clone().add(new THREE.Vector3(0, 0.25, 0));
    const pos = new THREE.Vector3(tgt.x + Math.sin(ang) * Math.cos(el) * r, tgt.y + Math.sin(el) * r, tgt.z + Math.cos(ang) * Math.cos(el) * r);
    return { pos, tgt };
  }

  // ─────────────────────────────────────────────────────────────── part B: riding the HBM -> die link
  private rideS(t: number) {
    const au = this.ctx.audio, T = this.T;
    const t0 = T.swap, t1 = T.land;
    // lands exactly on "line"; the drums add surge on the way; a short coast after
    const d = cumEnv(au, 'drums', t) - cumEnv(au, 'drums', t0);
    const dTot = Math.max(1e-3, cumEnv(au, 'drums', t1) - cumEnv(au, 'drums', t0));
    const lin01 = clamp((t - t0) / Math.max(1e-3, t1 - t0));
    if (t <= t1) return ease.inOutQuad(clamp(0.5 * lin01 + 0.5 * clamp(d / dTot)));
    return 1 + 0.2 * ease.outCubic(clamp((t - t1) / 1.2));
  }

  private shotB(t: number) {
    // the camera backs down the link over the die, facing the HBM3 stack the data pours out of: tight on
    // the burning wall with the stream rushing at the lens, pulling back until the whole stack is in frame
    const s = this.rideS(t);
    const wallX = -(DIM.hbm.x - DIM.hbm.w / 2);          // the west stacks' die-facing wall
    const push = 0.25 * kickPush(this.ctx.audio, t);
    const pos = new THREE.Vector3(wallX + lerp(2.2, 6.6, s) - push, H100_Y.dieTop + lerp(0.9, 2.6, s), lerp(1.1, 2.2, s));
    const tgt = new THREE.Vector3(wallX - lerp(0.5, 1.4, s), H100_Y.intTop + 0.35, lerp(0.3, 0, s));
    return { pos, tgt };
  }

  private drawTraffic(t: number, partB: boolean, kLess: number) {
    const items: PulseItem[] = [];
    const g = this.gpu;
    const fi = frameIdx(t);
    // part A: ambient HBM -> SM traffic on all active stacks, far below the bar
    if (!partB) {
      for (let kk = 0; kk < 6; kk++) {
        if (!g.hbm(kk).active) continue;
        const path = g.pulsePath(kk, (SM_I + kk * 17) % 144);
        for (let n = 0; n < 4; n++) items.push({ curve: path, u: ((t - this.ctx.start) * 0.3 + n / 4 + kk * 0.13) % 1, len: 0.03, gain: 1.0, width: 0.1 });
      }
    } else {
      // the ride's lanes, packets cycling only through the corridor the camera rides (HBM edge -> past the PHY):
      // FP32 = 32 long packets per lane (a solid stream); INT4 = 8 packets per lane, each half as long:
      // 4× sparser × 2× shorter = 8× less light on the wire, same as the bytes
      const dense = kLess < 0.5;
      const per = dense ? 32 : 8;
      const u0 = this.uI - 0.03, u1 = this.uEnd + 0.12, span = u1 - u0;
      for (let l = 0; l < LANES; l++) {
        const lane = this.lanes[l]!;
        for (let n = 0; n < per; n++) {
          const ph = hash(l, n, 3) * 0.35;
          const v = ((t - this.T.swap) * 0.42 + (n + ph) / per + l * 0.061) % 1;
          const fl = hash(l, n, fi >> 1) < 0.82 ? 1 : 0.3;
          items.push({
            curve: lane, u: u0 + v * span,
            len: dense ? 0.0075 : 0.0075 / 2,
            gain: (dense ? 2.2 : 4) * fl,
            width: dense ? 0.07 : 0.11,
            color: dense ? LIN.ember : LIN.signal,
          });
        }
      }
    }
    g.pulses.set(items);
  }

  // ─────────────────────────────────────────────────────────────── screen-space type
  private project(p: THREE.Vector3) {
    const v = this.gpu.group.localToWorld(p.clone()).project(this.cam);
    return { x: (v.x * 0.5 + 0.5) * W, y: (-v.y * 0.5 + 0.5) * H, behind: v.z > 1 };
  }

  private drawBarType(c: CanvasRenderingContext2D, t: number, kFour: number) {
    const au = this.ctx.audio, T = this.T;
    const kBr = snap(au, t, T.bracket);
    const kFine = snap(au, t, T.fine);
    // digits on the blocks
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    for (let i = 0; i < 32; i++) {
      const L = this.barLayout(i, kFour);
      if (L.merged) continue;
      if (kFour > 0 && i % 8 !== 0) continue;
      const p = this.project(new THREE.Vector3(BAR_C.x + L.x, BAR_C.y + L.h * 0.5, BAR_C.z + L.d * 0.5 + 0.01));
      if (p.behind) continue;
      if (kFour > 0) {
        c.font = font(F.mono(700), 76);
        c.fillStyle = rgba('bone', 0.97);
        c.fillText(INT4_BITS[L.slot]!, p.x, p.y + 4);
      } else {
        c.font = font(F.mono(700), 30);
        c.fillStyle = FP32_BITS[i] === '1' ? rgba('bone', 0.97) : rgba('ash', 0.75);
        c.fillText(FP32_BITS[i]!, p.x, p.y + 1);
      }
    }
    // the size label under the bar: "32 bits · FP32" (lit on "two") -> "4 bits · INT4"
    const four = kFour >= 0.5;
    const lp = this.project(new THREE.Vector3(BAR_C.x, BAR_C.y - 0.2, BAR_C.z + 0.9));
    c.font = font(F.archivo(100, 700), 64);
    c.fillStyle = four ? rgba('signal', 1) : rgba('bone', 0.35 + 0.6 * kBr);
    c.fillText(four ? '4 bits' : '32 bits', lp.x, lp.y + 40);
    c.font = font(F.mono(500), 24);
    c.fillStyle = rgba('ash', 0.9 * Math.max(0.4, kBr));
    c.fillText(four ? 'INT4 · one number' : 'FP32 · one number', lp.x, lp.y + 80);

    // the value, bottom centre: 1.20, then on "fine" 1.20 -> 1.07
    const vy = 965;
    this.band(c, vy - 110, vy + 80);
    c.textBaseline = 'alphabetic';
    c.font = font(F.mono(500), 96);
    const s0 = '1.20', arrow = '  →  ', s1 = '1.07';
    const wA = c.measureText(s0).width, wArr = c.measureText(arrow).width, wB = c.measureText(s1).width;
    const cx = lerp(960, 960 - (wArr + wB) / 2, kFine);
    c.textAlign = 'left';
    c.fillStyle = rgba('bone', 0.95);
    c.fillText(s0, cx - wA / 2, vy);
    if (kFine > 0) {
      c.globalAlpha = kFine;
      c.fillStyle = rgba('ash', 0.9);
      c.fillText(arrow, cx + wA / 2, vy);
      c.fillStyle = rgba('signal', 1);
      c.fillText(s1, cx + wA / 2 + wArr, vy);
      c.font = font(F.mono(400), 24);
      c.fillStyle = rgba('ash', 0.9);
      c.textAlign = 'center';
      c.fillText('rounded to the nearest 4-bit step · close enough · CH09 example', 960, vy + 48);
      c.globalAlpha = 1;
    }
  }

  private drawRideType(c: CanvasRenderingContext2D, t: number, kLess: number, kLand: number, hLand: number) {
    const au = this.ctx.audio, T = this.T;
    const kEight = snap(au, t, T.eight);
    c.textAlign = 'center';
    c.textBaseline = 'alphabetic';
    // the stream's caption: what is on the wire
    c.font = font(F.mono(500), 30);
    if (kLess < 0.5) {
      c.globalAlpha = Math.max(0.5, kEight);
      c.fillStyle = rgba('bone', 0.9);
      c.fillText('HBM3 → die · 8 numbers in FP32 = 8 loads', 960, 330);
    } else {
      c.fillStyle = rgba('bone', 0.9);
      c.fillText('the same 8 numbers in INT4 = 1 load', 960, 330);
    }
    c.globalAlpha = 1;
    if (kLess > 0) {
      c.globalAlpha = kLess;
      this.band(c, 720, 950);
      c.font = font(F.archivo(100, 800), 132);
      const a8 = '8× ', rest = 'less to move';
      const w8 = c.measureText(a8).width, wr = c.measureText(rest).width;
      c.textAlign = 'left';
      c.fillStyle = rgba('signal', 1);
      c.fillText(a8, 960 - (w8 + wr) / 2, 860);
      c.fillStyle = rgba('bone', 0.97);
      c.fillText(rest, 960 - (w8 + wr) / 2 + w8, 860);
      c.textAlign = 'center';
      c.font = font(F.mono(400), 26);
      c.fillStyle = rgba('ash', 0.95);
      c.fillText('per billion parameters: FP32 4 GB → INT4 500 MB · CH09', 960, 918);
      c.globalAlpha = 1;
    }
    // "line": the ride lands at the die's memory PHY
    if (kLand > 0) {
      const p = this.project(new THREE.Vector3(-DIM.die.w / 2, H100_Y.dieTop + 0.03, 0));
      if (!p.behind) {
        c.strokeStyle = rgba('signal', 0.6 + 0.4 * hLand);
        c.lineWidth = 2;
        c.beginPath(); c.arc(p.x, p.y, 26 + 60 * hLand, 0, Math.PI * 2); c.stroke();
        c.font = font(F.mono(500), 24);
        c.fillStyle = rgba('bone', 0.9 * kLand);
        c.fillText('die edge · memory PHY', p.x, Math.min(p.y - 44 - 60 * hLand, 690));
      }
    }
  }

  /** A soft full-width ink band so screen type reads over the 3D. */
  private band(c: CanvasRenderingContext2D, y0: number, y1: number) {
    const grd = c.createLinearGradient(0, y0, 0, y1);
    grd.addColorStop(0, rgba('ink', 0));
    grd.addColorStop(0.3, rgba('ink', 0.72));
    grd.addColorStop(0.7, rgba('ink', 0.72));
    grd.addColorStop(1, rgba('ink', 0));
    c.fillStyle = grd;
    c.fillRect(0, y0, W, y1 - y0);
  }

  private drawLyric(c: CanvasRenderingContext2D, t: number, line: Line, preview: boolean) {
    const size = 76, x0 = 128, y = 200;
    c.save();
    c.font = font(F.archivo(100, 500), size);
    c.textAlign = 'left';
    c.textBaseline = 'alphabetic';
    // a soft ink backing so the lyric reads over the 3D
    const tw = c.measureText(line.text).width;
    const grd = c.createLinearGradient(0, y - size - 40, 0, y + 40);
    grd.addColorStop(0, rgba('ink', 0));
    grd.addColorStop(0.35, rgba('ink', 0.55));
    grd.addColorStop(1, rgba('ink', 0));
    c.fillStyle = grd;
    c.fillRect(x0 - 60, y - size - 40, tw + 120, size + 80);
    let x = x0;
    const space = c.measureText(' ').width;
    for (const w of line.words) {
      const p = preview ? 0 : Lyrics.wordProgress(w, t);
      const ww = c.measureText(w.w).width;
      c.fillStyle = rgba('bone', preview ? 0.3 : 0.4);
      c.fillText(w.w, x, y);
      if (p > 0) {
        c.save();
        c.beginPath(); c.rect(x - 2, y - size, (ww + 4) * clamp(p), size * 1.4); c.clip();
        c.fillStyle = p < 1 ? rgba('signal') : rgba('bone', 0.97);
        c.fillText(w.w, x, y);
        c.restore();
      }
      x += ww + space;
    }
    c.restore();
  }

  private drawTag(c: CanvasRenderingContext2D) {
    c.font = font(F.mono(500), 20);
    c.textAlign = 'right';
    c.textBaseline = 'alphabetic';
    c.fillStyle = rgba('ash', 0.75);
    c.fillText('CH09 · QUANTIZATION · H100', 1792, 110);
  }
}
