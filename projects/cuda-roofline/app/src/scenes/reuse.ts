// One idea: every time you reuse data you already fetched, you do more math per byte, and more math
// per byte moves you right on the roofline (toward a higher speed limit).
//
// v5 `reuse` (pre-chorus 3, 108.3-116.4 s), in 3D inside the shared H100 (`_h100.ts`):
//   "Fetch it once and use it more,"   the camera orbits one SM close-up (its drift surges with the drums).
//     Fetch      one tile drops from far above (the only trip to memory) into L1 / shared memory. Counter 1.
//     once       "trips to memory: 1"; the trip beam goes dark and never lights again.
//     use        counter 2: a stream of data pulses from the tile to partition 1's tensor core.
//     more       counter 4: two more streams (partitions 2, 3).
//     (gap beat) counter 8: streams to all four partitions' lanes as well; every tensor core lit.
//   "That's what the roofline's for"
//     That's     the camera rises fast out of the SM; a small 3D roofline stands over the die.
//     roofline's the roof lights.
//     for        the dot steps right along the slope 1 -> 2 -> 4 -> 8 on 16ths (rings stay behind).
// The counter is illustrative (no book number). Changes snap on the 8th grid (_lock.ts) and hold; the
// camera never stops (_cam.ts drive/kickPush/beatEase).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { Lyrics, type Line, type Word } from '../engine/lyrics';
import { clamp, ease, lerp } from '../engine/util';
import { hit, q, snap } from './_lock';
import { beatEase, drive, kickPush } from './_cam';
import { H100, H100_Y, Pulses, glowMat, type PulseItem } from './_h100';

const SM = 58;                                            // a mid-die SM site (GPC 3)
const LYR = { x: 120, y: 206, size: 84, fam: F.archivo(100, 800) };
const MONO = F.mono(500), MONO_L = F.mono(400);
const CNT = { x: 1390, y: 420 };                          // counter panel (screen space, line 1)

// the 3D roofline over the die (group space, mm): a vertical chart in the x-y plane facing +z
const CH = { x0: -11, x1: 11, y0: H100_Y.dieTop + 2.2, y1: H100_Y.dieTop + 13.2, z: 3 };
const RIDGE = 16;                                         // illustrative ridge, math per byte
const cx = (i: number) => lerp(CH.x0, CH.x1, (Math.log2(i) + 1) / 7);     // x: 0.5 .. 64
const cy = (s: number) => lerp(CH.y0, CH.y1, (Math.log2(s) + 1) / 5.5);   // y: 0.5 .. ~22.6
const roof = (i: number) => Math.min(i, RIDGE);
const HOPS = [1, 2, 4, 8];

const hash = (a: number, b: number) => { const x = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return x - Math.floor(x); };
const lin = (k: keyof typeof LIN, s = 1) => new THREE.Color().setRGB(LIN[k][0] * s, LIN[k][1] * s, LIN[k][2] * s, THREE.LinearSRGBColorSpace);

export default class Reuse extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(34, W / H, 0.01, 5000);
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();
  gpu!: H100;
  beams = new Pulses(256);
  tile!: THREE.Mesh;
  tileGlow!: THREE.Mesh;
  trip!: THREE.Mesh;
  chart = new THREE.Group();
  roofMat = new THREE.MeshBasicMaterial({ color: lin('bone', 0.55) });
  roofGlow!: THREE.Mesh;
  dot!: THREE.Mesh;
  dotGlow!: THREE.Mesh;
  rings: THREE.Mesh[] = [];
  lA!: Line; lB!: Line;
  T: Record<string, number> = {};
  /** Reuse streams: target anchor name in the SM and the time it starts. */
  streams: { to: string; lanes: number; at: number }[] = [];

  override async init() {
    const au = this.ctx.audio, ly = this.ctx.lyrics;
    this.lA = ly.get('Fetch it once');
    this.lB = ly.get("That's what the roofline");
    const w = (l: Line, s: string) => l.words.find((x) => x.w.toLowerCase().startsWith(s))!.start;
    const A = this.lA, B = this.lB;
    this.T = {
      fetch: w(A, 'fetch'), once: w(A, 'once'), use: w(A, 'use'), more: w(A, 'more'),
      eight: au.timeOfBeat(Math.round(au.beatAt(q(au, w(A, 'more')))) + 2),   // the instrumental beat after "more"
      thats: w(B, 'that'), roof: w(B, 'roofline'), for: w(B, 'for'),
    };
    const T = this.T;
    // use 1 at Fetch (tensor p0), 2 at use (p1), 3-4 at more (p2, p3), 5-8 at the gap beat (lanes p0-p3)
    this.streams = [
      { to: 'p0.tensor', lanes: -1, at: T.fetch }, { to: 'p1.tensor', lanes: -1, at: T.use },
      { to: 'p2.tensor', lanes: -1, at: T.more }, { to: 'p3.tensor', lanes: -1, at: T.more },
      { to: 'p0.regfile', lanes: 0, at: T.eight }, { to: 'p1.regfile', lanes: 1, at: T.eight },
      { to: 'p2.regfile', lanes: 2, at: T.eight }, { to: 'p3.regfile', lanes: 3, at: T.eight },
    ];

    this.gpu = new H100({ renderer: this.ctx.renderer });
    this.scene.add(this.gpu.group, H100.rig());
    this.scene.environment = this.gpu.env;
    this.scene.background = lin('ink');
    this.gpu.smDetail(SM);
    this.scene.add(this.beams.mesh);

    // the tile (lands on L1 / shared memory) and the one trip down to it
    this.tile = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: lin('signal', 0.6), emissive: lin('signal', 0.9), metalness: 0.2, roughness: 0.4 }));
    this.tileGlow = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), glowMat(0x000000));
    this.trip = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 10, 1, true), glowMat(0x000000));
    this.scene.add(this.tile, this.tileGlow, this.trip);

    // the roofline chart
    const axes = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(CH.x0, CH.y1 + 0.8, CH.z), new THREE.Vector3(CH.x0, CH.y0, CH.z),
      new THREE.Vector3(CH.x0, CH.y0, CH.z), new THREE.Vector3(CH.x1 + 0.8, CH.y0, CH.z),
      ...[1, 2, 4, 8].flatMap((i) => [new THREE.Vector3(cx(i), CH.y0, CH.z), new THREE.Vector3(cx(i), CH.y0 - 0.45, CH.z)]),
    ]);
    this.chart.add(new THREE.LineSegments(axes, new THREE.LineBasicMaterial({ color: lin('bone', 0.8), transparent: true })));
    const roofPath = new THREE.CurvePath<THREE.Vector3>();
    const a = new THREE.Vector3(cx(0.5), cy(roof(0.5)), CH.z), r = new THREE.Vector3(cx(RIDGE), cy(RIDGE), CH.z), e = new THREE.Vector3(cx(64), cy(RIDGE), CH.z);
    roofPath.add(new THREE.LineCurve3(a, r));
    roofPath.add(new THREE.LineCurve3(r, e));
    const tube = (rad: number) => new THREE.TubeGeometry(roofPath, 64, rad, 10, false);
    this.chart.add(new THREE.Mesh(tube(0.11), this.roofMat));
    this.roofGlow = new THREE.Mesh(tube(0.2), glowMat(0x000000));
    this.chart.add(this.roofGlow);
    this.dot = new THREE.Mesh(new THREE.SphereGeometry(0.34, 20, 14), new THREE.MeshBasicMaterial({ color: lin('signal', 1.6) }));
    this.dotGlow = new THREE.Mesh(new THREE.SphereGeometry(0.75, 20, 14), glowMat(0x000000));
    this.chart.add(this.dot, this.dotGlow);
    for (let k = 0; k < 3; k++) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.045, 8, 32), new THREE.MeshBasicMaterial({ color: lin('signal', 0.7) }));
      ring.visible = false;
      this.rings.push(ring);
      this.chart.add(ring);
    }
    this.chart.visible = false;
    this.gpu.group.add(this.chart);
  }

  /** Math-per-byte counter value (1 -> 2 -> 4 -> 8), snapped. */
  private count(t: number): number {
    const au = this.ctx.audio, T = this.T;
    if (snap(au, t, T.eight) > 0) return 8;
    if (snap(au, t, T.more) > 0) return 4;
    if (snap(au, t, T.use) > 0) return 2;
    if (snap(au, t, T.fetch) > 0) return 1;
    return 0;
  }

  /** The dot's hop state on "for": index into HOPS and the snap within the current hop. */
  private hop(t: number): { at: number; k: number; ts: number } {
    const au = this.ctx.audio;
    const t0 = q(au, this.T.for, 2), P = (au.timeOfBeat(Math.round(au.beatAt(t0)) + 1) - t0) / 4;
    let at = 0;
    for (let k = 1; k < HOPS.length; k++) if (t >= t0 + (k - 1) * P) at = k;
    const ts = t0 + Math.max(0, at - 1) * P;
    const k = at > 0 ? clamp((t - ts) / 0.07) : 1;
    return { at, k: 1 - Math.pow(1 - k, 4), ts };
  }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp, audio: au } = this.ctx;
    const t = f.t, T = this.T, g = this.gpu;
    // the other SMs keep flickering, but low while we sit inside one (at this scale a neighbour's glow
    // fills a quarter of the frame); in the wide shot they come back, and every dot hop lights the die
    const uWide = beatEase(au, t, T.thats, T.roof, ease.inOutExpo);
    const hp = this.hop(t);
    const hopHit = hp.at > 0 && t >= hp.ts ? Math.exp(-(t - hp.ts) / 0.12) * 0.35 : 0;
    const fi = Math.floor(t * 15);
    g.setSMActivity((i) => {
      if (i === SM || !g.smEnabled(i)) return 0;
      const fl = hash(i, fi) < 0.55 ? 1 : 0.25;
      const lvl = 0.3 + 0.9 * clamp(f.a.rms * 1.2 + f.a.drums * 0.4);
      return lvl * lerp(0.03 + 0.07 * fl, 0.04 + 0.16 * fl, uWide) + hopHit;
    });
    // HBM (the memory the tile came from) flares on the one fetch and then stays nearly dark while the
    // SM reuses it; it wakes again in the wide shot
    const hMem = hit(au, t, T.fetch, 0.18);
    g.setHBMActivity(() => lerp(0.04, 0.1 + 0.12 * f.a.drums, uWide) + 1.6 * hMem * (1 - snap(au, t, T.once)));
    g.update(t, f.a);
    const d = g.smDetail(SM);
    this.scene.updateMatrixWorld(true);
    const smW = (name: string) => d.group.localToWorld(d.anchor(name).clone());     // world

    const kFetch = snap(au, t, T.fetch), kOnce = snap(au, t, T.once);
    const kThats = snap(au, t, T.thats), kRoof = snap(au, t, T.roof);
    const n = this.count(t);

    // ---------------------------------------------------------------- the tile and the one trip
    const l1 = smW('l1');
    const tileSize = new THREE.Vector3(0.32, 0.035, 0.1);
    const tilePos = l1.clone().add(new THREE.Vector3(-0.22, tileSize.y / 2 + 0.004, 0));
    const drop = kFetch > 0 ? (1 - kFetch) * 9 : 9;
    this.tile.visible = this.tileGlow.visible = kFetch > 0;
    this.tile.position.copy(tilePos).add(new THREE.Vector3(0, drop, 0));
    this.tile.scale.copy(tileSize);
    const hTile = hit(au, t, T.fetch, 0.1);
    const reuseHits = this.streams.reduce((m, s) => Math.max(m, hit(au, t, s.at, 0.09)), 0);
    const tg = kFetch > 0 ? 0.35 + 1.4 * hTile + 0.5 * reuseHits + 0.25 * f.a.drums : 0;
    this.tileGlow.position.copy(this.tile.position);
    this.tileGlow.scale.copy(tileSize).multiplyScalar(1.25);
    (this.tileGlow.material as THREE.MeshBasicMaterial).color.setRGB(LIN.signal[0] * tg, LIN.signal[1] * tg, LIN.signal[2] * tg);
    // the trip: a vertical beam from far above to the tile, one hit on Fetch, a faint thread until "once" cuts it
    const tripA = kFetch > 0 ? (1 - kOnce) * (0.12 + 2.8 * hit(au, t, T.fetch, 0.16)) : 0;
    this.trip.visible = tripA > 0.01;
    this.trip.position.copy(tilePos).add(new THREE.Vector3(0, 2, 0));
    this.trip.scale.set(0.009, 4, 0.009);
    (this.trip.material as THREE.MeshBasicMaterial).color.setRGB(LIN.ember[0] * tripA, LIN.ember[1] * tripA, LIN.ember[2] * tripA);

    // ---------------------------------------------------------------- reuse streams (data pulses) and what they light
    const items: PulseItem[] = [];
    const lit = [0, 0, 0, 0], laneLit = [0, 0, 0, 0];
    const fade = 1 - kThats * 0.6;
    for (const s of this.streams) {
      const k = snap(au, t, s.at);
      if (k <= 0) continue;
      const p = Number(s.to[1]);
      const h = hit(au, t, s.at, 0.1);
      if (s.lanes < 0) lit[p] = Math.max(lit[p]!, 0.55 + 1.6 * h); else laneLit[p] = Math.max(laneLit[p]!, 0.5 + 1.4 * h);
      const to = smW(s.to);
      const from = tilePos.clone().add(new THREE.Vector3(0, tileSize.y / 2, 0));
      const mid = from.clone().lerp(to, 0.5).add(new THREE.Vector3(0, 0.28 + 0.05 * p, 0));
      const curve = new THREE.QuadraticBezierCurve3(from, mid, to);
      // a burst on the snap, then a steady stream: 4 packets riding the arc, faster when the drums hit
      const ph = (t - q(au, s.at)) * 1.6 + p * 0.21;
      for (let j = 0; j < 4; j++) {
        const u = ((ph + j / 4) % 1 + 1) % 1;
        items.push({ curve, u: u * k, len: 0.09, gain: (1.6 + 2.5 * h) * fade, width: 0.018 });
      }
      if (h > 0.05) items.push({ curve, u: 0.5, len: 1.0, gain: 2.2 * h * fade, width: 0.012 });
    }
    this.beams.set(items);
    const idle = 0.08 + 0.06 * f.a.rms;
    d.setTensor((p, i, j, k) => {
      const v = lit[p]!;
      if (v <= 0) return idle * 0.6;
      // the MAC lattice pulses in diagonal waves while it consumes the tile
      const wave = (Math.floor((t - T.fetch) * 8) + i + j + k) % 4 === 0 ? 1 : 0.45;
      return v * wave;
    });
    d.setLanes((p, kind, idx) => {
      const v = laneLit[p]!;
      if (v <= 0) return idle * (kind === 'fp32' ? 1 : 0.6);
      return v * (((Math.floor(t * 10) + idx + p) % 3) === 0 ? 1 : 0.55);
    });
    d.setUnits((u, p) => {
      if (u === 'l1') return kFetch > 0 ? 0.55 + 1.3 * hTile + 0.6 * reuseHits : 0.1;
      if (u === 'regfile') return laneLit[p]! > 0 ? 0.35 : 0.05;
      if (u === 'sched') return 0.12 + 0.2 * f.a.kick;
      return 0;
    });

    // ---------------------------------------------------------------- the 3D roofline (line 2)
    this.chart.visible = kThats > 0;
    if (kThats > 0) {
      const s = lerp(0.6, 1, kThats);
      const c0 = new THREE.Vector3(0, (CH.y0 + CH.y1) / 2, CH.z);
      this.chart.position.copy(c0).multiplyScalar(1 - s);
      this.chart.scale.setScalar(s);
      this.roofMat.color.copy(kRoof > 0 ? lin('signal', lerp(0.7, 1.4, kRoof)) : lin('bone', 0.55));
      const rg = kRoof * (0.9 + 1.4 * hit(au, t, T.roof, 0.12) + 0.3 * f.a.drums);
      (this.roofGlow.material as THREE.MeshBasicMaterial).color.setRGB(LIN.signal[0] * rg, LIN.signal[1] * rg, LIN.signal[2] * rg);
      const { at, k } = this.hop(t);
      const from = HOPS[Math.max(0, at - 1)]!, to = HOPS[at]!;
      const x = lerp(cx(from), cx(to), k), y = lerp(cy(roof(from)), cy(roof(to)), k);
      this.dot.position.set(x, y, CH.z + 0.05);
      this.dotGlow.position.copy(this.dot.position);
      const dg = 1.2 + 2.5 * hopHit;
      (this.dotGlow.material as THREE.MeshBasicMaterial).color.setRGB(LIN.signal[0] * dg * 0.5, LIN.signal[1] * dg * 0.5, LIN.signal[2] * dg * 0.5);
      this.rings.forEach((ring, j) => {
        ring.visible = j < at;
        ring.position.set(cx(HOPS[j]!), cy(roof(HOPS[j]!)), CH.z);
      });
    }

    // ---------------------------------------------------------------- camera
    const cam = this.cam;
    // close shot: orbiting the SM, its speed surging with the drums, a small push on each kick
    const smC = smW('p0.dispatch').lerp(smW('p3.dispatch'), 0.5).lerp(l1, 0.4);
    const az0 = -0.55 + 0.06 * drive(au, t, 1, 2.5, f.start);
    const dist0 = 2.8 * (1 - 0.045 * kickPush(au, t));
    const el0 = 1.0;                                                                  // steep: keeps the rim light's grazing glare off the die out of frame
    const right0 = new THREE.Vector3(Math.cos(az0), 0, -Math.sin(az0));
    const tgt0 = smC.clone().addScaledVector(right0, 0.42);                          // the SM sits left of centre
    const pos0 = tgt0.clone().add(new THREE.Vector3(Math.sin(az0) * Math.cos(el0), Math.sin(el0), Math.cos(az0) * Math.cos(el0)).multiplyScalar(dist0));
    // high shot: in front of the chart, a little above it, the die spread out below; slow drift
    // (aimed above the chart's centre so the chart sits below the lyric, the die filling the bottom)
    const chartC = g.group.localToWorld(new THREE.Vector3(0.6, (CH.y0 + CH.y1) / 2 + 1.9, CH.z));
    const az1 = 0.16 * Math.sin(0.35 * drive(au, t, 1, 1.5, f.start)) - 0.12;
    const dist1 = 37 * (1 - 0.02 * kickPush(au, t));
    const pos1 = chartC.clone().add(new THREE.Vector3(Math.sin(az1) * dist1, 9, Math.cos(az1) * dist1));
    const pos = pos0.clone().lerp(pos1, uWide), tgt = tgt0.clone().lerp(chartC, uWide);
    cam.position.copy(pos);
    cam.up.set(0, 1, 0);
    cam.lookAt(tgt);
    H100.fitClip(cam, pos.distanceTo(tgt));

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    comp.draw(renderer, this.ms.texture, out);

    // ---------------------------------------------------------------- screen-space layer
    const c = this.ui.ctx;
    this.ui.clear();
    const showB = t >= this.lB.start - 0.4;
    karaoke(c, (showB ? this.lB : this.lA).words, t, LYR.x, LYR.y, LYR.size, LYR.fam);

    // counter (line 1), gone as the camera rises
    const cA = 1 - kThats;
    if (cA > 0 && n > 0) {
      c.save();
      c.globalAlpha = cA;
      label(c, 'MATH PER BYTE', CNT.x, CNT.y, 28, MONO, 'bone', 0.85);
      const born = [T.fetch, T.use, T.more, T.eight][[1, 2, 4, 8].indexOf(n)] ?? T.fetch;
      const sz = 300 * lerp(1.14, 1, snap(au, t, born));
      c.font = font(F.archivo(100, 900), sz); c.textAlign = 'left'; c.textBaseline = 'alphabetic';
      c.fillStyle = rgba(n > 0 ? 'signal' : 'bone', n > 0 ? 1 : 0.25);
      c.fillText(String(n), CNT.x - 8, CNT.y + 320);
      label(c, n <= 1 ? 'uses of one fetched tile' : `${n} uses, still 1 fetch`, CNT.x, CNT.y + 382, 26, MONO_L, 'bone', 0.8);
      if (kOnce > 0) stamp(c, 'trips to memory: 1', CNT.x + 170, CNT.y + 460, lerp(1.25, 1, kOnce));
      c.restore();
    }
    // labels in the SM (line 1): the tile and the tensor cores it feeds
    const proj = (vW: THREE.Vector3) => {
      const v = vW.clone().project(cam);
      return { x: (v.x * 0.5 + 0.5) * W, y: (-v.y * 0.5 + 0.5) * H, ok: v.z < 1 && v.z > -1 };
    };
    if (kFetch > 0 && kThats < 1) {
      const a = (1 - kThats) * kFetch;
      const pt = proj(this.tile.position.clone().add(new THREE.Vector3(0, 0.06, 0)));
      if (pt.ok) tag(c, 'one tile · L1 / shared memory', pt.x, pt.y, a);
      if (n >= 2) {
        const pp = proj(smW('p1.tensor').add(new THREE.Vector3(0, 0.1, 0)));
        if (pp.ok) tag(c, 'tensor core', pp.x, pp.y, a * snap(au, t, T.use));
      }
    }
    // chart labels (line 2), projected from the 3D chart
    if (kThats > 0) {
      const P = (x: number, y: number) => proj(g.group.localToWorld(new THREE.Vector3(x, y, CH.z)));
      const aC = kThats;
      c.save(); c.globalAlpha = aC;
      const ya = P(CH.x0, CH.y1 + 0.8), xa = P(CH.x1 + 0.8, CH.y0);
      label(c, 'speed ↑', ya.x - 10, ya.y - 16, 28, MONO, 'bone', 0.9);
      label(c, 'math per byte →', xa.x, xa.y + 48, 28, MONO, 'bone', 0.9, 'right');
      for (const i of [1, 2, 4, 8]) { const p = P(cx(i), CH.y0 - 0.45); label(c, String(i), p.x, p.y + 30, 24, MONO_L, 'bone', 0.8, 'center'); }
      if (kRoof > 0) {
        c.globalAlpha = aC * kRoof;
        const rl = P(cx(RIDGE), cy(RIDGE));
        label(c, 'the roofline', rl.x - 18, rl.y - 44, 34, MONO, 'signal', 1, 'right');
        const ms = P(cx(2.2), cy(roof(2.2)));
        const me = P(cx(8), cy(roof(8)));
        c.save(); c.translate(ms.x, ms.y); c.rotate(Math.atan2(me.y - ms.y, me.x - ms.x));
        label(c, 'memory speed limit', 0, -30, 26, MONO_L, 'bone', 0.9); c.restore();
        const fl = P(cx(40), cy(RIDGE));
        label(c, 'math speed limit', fl.x, fl.y - 24, 26, MONO_L, 'bone', 0.9, 'center');
      }
      c.globalAlpha = aC;
      const { at } = this.hop(t);
      const dp = proj(this.dot.getWorldPosition(new THREE.Vector3()));
      label(c, `${HOPS[at]} math per byte`, dp.x + 34, dp.y + 50, 28, MONO, 'bone', 0.95);
      c.restore();
    }
    comp.draw(renderer, this.ui.upload(), out);
    return { bloom: 0.6, bloomThreshold: 0.85 };
  }
}

// ---------------------------------------------------------------- 2D helpers
function label(c: CanvasRenderingContext2D, s: string, x: number, y: number, size: number, fam: string,
  col: string, a: number, align: CanvasTextAlign = 'left') {
  c.font = font(fam, size); c.fillStyle = rgba(col, a);
  c.textAlign = align; c.textBaseline = 'alphabetic';
  c.fillText(s, x, y);
}

/** A leader tag: a small orange square on the point and a mono label up-right of it. */
function tag(c: CanvasRenderingContext2D, s: string, x: number, y: number, a: number) {
  if (a <= 0) return;
  c.save();
  c.globalAlpha = a;
  c.strokeStyle = rgba('bone', 0.7); c.lineWidth = 1.5;
  c.beginPath(); c.moveTo(x, y); c.lineTo(x + 34, y - 34); c.lineTo(x + 44, y - 34); c.stroke();
  c.fillStyle = rgba('signal', 1); c.fillRect(x - 4, y - 4, 8, 8);
  label(c, s, x + 50, y - 26, 24, MONO, 'bone', 0.95);
  c.restore();
}

/** A rubber-stamp label (orange frame, slight tilt), snapped in with scale k. */
function stamp(c: CanvasRenderingContext2D, s: string, x: number, y: number, k: number) {
  c.save();
  c.translate(x, y); c.rotate(-0.035); c.scale(k, k);
  c.font = font(MONO, 26);
  const w = c.measureText(s).width + 36;
  c.strokeStyle = rgba('signal', 0.95); c.lineWidth = 3;
  c.strokeRect(-w / 2, -26, w, 46);
  c.fillStyle = rgba('signal', 0.95); c.textAlign = 'center'; c.textBaseline = 'middle';
  c.fillText(s, 0, -2);
  c.restore();
}

/** One lyric line with a per-word karaoke wipe: sung bone, the word being sung signal, unsung dim. */
function karaoke(c: CanvasRenderingContext2D, words: Word[], t: number, x: number, y: number, size: number, fam: string) {
  c.font = font(fam, size); c.textAlign = 'left'; c.textBaseline = 'alphabetic';
  let cx_ = x;
  const space = c.measureText(' ').width;
  for (const w of words) {
    const s = w.w, wd = c.measureText(s).width;
    const p = Lyrics.wordProgress(w, t);
    c.fillStyle = rgba('bone', 0.3); c.fillText(s, cx_, y);
    if (p > 0) {
      c.save(); c.beginPath(); c.rect(cx_ - 4, y - size, (wd + 8) * p, size * 1.4); c.clip();
      c.fillStyle = p >= 1 ? rgba('bone', 1) : rgba('signal', 1); c.fillText(s, cx_, y);
      c.restore();
    }
    cx_ += wd + space;
  }
}

