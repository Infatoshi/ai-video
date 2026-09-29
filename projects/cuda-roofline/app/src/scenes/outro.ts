// ONE IDEA: you just walked from one thread to eight GPUs; here is that path, in order, and the book it came from.
//
// OUTRO (v5, 3D; the timeline starts it on the outro's second bar, 175.07 → 190.0, iris in; full band to the
// last kick at ~186.80, drums gone by ~187.3, silence from ~188.6). The zoom-in of the whole video, reversed:
//   175.07  inside the SM close-up of the shared H100 (scenes/_h100.ts): "threads" (every lane sparks on
//           its own), "warps" (each partition's lanes pulse in lockstep on the beat)
//   176.87  pulling back off the SM onto the die: "memory" (LD/ST + L1 light, data pulses HBM → SM),
//           "tiles" (the L1 / shared memory block holds)
//   178.68  die → module: "tensor cores" (every SM flares at once), "flash attention" (a band of tiles
//           streams across the die, a step per 8th)
//   180.48  the whole module: "4 bits" (the HBM stacks pump short bursts: bits)
//   181.39  cut on the snare to the DGX H100 (scenes/_dgx.ts), exploded with all-reduce traffic: "8 GPUs";
//           the teardown runs backwards (heatsinks down, tray in, lid on) while the camera pulls out
//   184.09  the node, closed and idling, darkens; the end card: the title a word per 8th, the author on the
//           last kick; the camera settles and holds still through the silence.
// Each recap word snaps in big as its level passes and then joins the running list (scenes/_lock.ts);
// the camera is always moving until the card (scenes/_cam.ts: an orbit that surges with the drums).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font, layout } from '../engine/type';
import { clamp, ease, frameIdx, hash, lerp, smoothstep } from '../engine/util';
import { snap, hit, q } from './_lock';
import { drive, kickPush } from './_cam';
import { H100, H100_Y, type PulseItem } from './_h100';
import { DGX } from './_dgx';

/** The recap path: one plain word per stop and its chapter. */
const ITEMS = [
  { word: 'threads', ch: 'CH02' },
  { word: 'warps', ch: 'CH03' },
  { word: 'memory', ch: 'CH06' },
  { word: 'tiles', ch: 'CH06' },
  { word: 'tensor cores', ch: 'CH07' },
  { word: 'flash attention', ch: 'CH08' },
  { word: '4 bits', ch: 'CH09' },
  { word: '8 GPUs', ch: 'CH10' },
] as const;

const SM = 58;                         // the SM site the video dove into (mid-die, enabled)
const mixEase = (x: number) => 0.65 * ease.inOutCubic(x) + 0.35 * x;   // lands on the beat, never stops dead

type V3 = [number, number, number];
/** DGX camera keys (mm, world). */
interface DKey { t: number; p: V3; l: V3; fn: (x: number) => number }

export default class Outro extends Scene {
  private scene = new THREE.Scene();
  private cam = new THREE.PerspectiveCamera(34, W / H, 0.05, 20000);
  private gpu!: H100;
  private dgx!: DGX;
  private ms = makeRT(W, H, { samples: 4 });
  private ui = new Layer2D();
  private paths: THREE.Curve<THREE.Vector3>[] = [];
  /** Downbeats from the outro section start, the recap stop times, the card beats, the cut to the node. */
  private D: number[] = [];
  private tItem: number[] = [];
  private tWord: number[] = [];
  private tAuthor = 0;
  private tSwitch = 0;
  private dKeys: DKey[] = [];

  override async init() {
    const au = this.ctx.audio, r = this.ctx.renderer;
    this.gpu = new H100({ renderer: r });
    this.scene.add(this.gpu.group, H100.rig());
    this.scene.environment = this.gpu.env;
    this.scene.background = new THREE.Color().setRGB(LIN.ink[0], LIN.ink[1], LIN.ink[2], THREE.LinearSRGBColorSpace);
    this.gpu.explode(0);
    for (let k = 0; k < 6; k++) if (this.gpu.hbm(k).active) {
      for (const d of [-20, -2, 14]) {
        let s = clamp(SM + d + k * 3, 0, 143);
        while (!this.gpu.smEnabled(s)) s = (s + 1) % 144;
        this.paths.push(this.gpu.pulsePath(k, s));
      }
    }
    this.dgx = new DGX();
    this.dgx.init(r);
    // a cool key and a warm rim so the closed box reads as an object, not a silhouette
    const key = new THREE.DirectionalLight(new THREE.Color().setRGB(LIN.bone[0], LIN.bone[1], LIN.bone[2]), 2.0);
    key.position.set(1500, 1100, 1700);
    const rim = new THREE.DirectionalLight(new THREE.Color().setRGB(LIN.ember[0], LIN.ember[1], LIN.ember[2]), 0.6);
    rim.position.set(1400, 500, -1500);
    this.dgx.scene.add(key, rim);

    // bars count from the outro *section*; the timeline starts this scene on its second bar
    const s = au.sections.find((x) => x.name === 'outro')?.start ?? this.ctx.start;
    this.D = au.downbeats.filter((d) => d >= s - 0.02);
    const b1 = Math.round(au.beatAt(this.D[1]!));
    this.tItem = ITEMS.map((_, i) => au.timeOfBeat(b1 + 2 * i));
    this.tSwitch = this.tItem[7]!;
    const b6 = Math.round(au.beatAt(this.D[6]!));
    this.tWord = [0, 1, 2, 3].map((j) => au.timeOfBeat(b6 + 0.5 * j));
    const kicks = au.events('kick', this.D[6]!, this.ctx.end);
    const last = kicks.length ? kicks[kicks.length - 1]![0] : au.timeOfBeat(b6 + 6);
    this.tAuthor = Math.max(q(au, last, 2), this.tWord[3]! + 0.45);
    this.dKeys = [
      { t: this.tSwitch, p: [150, 560, 1010], l: [0, 280, 470], fn: mixEase },
      { t: this.D[5]!, p: [430, 700, 1300], l: [0, 250, 400], fn: mixEase },
      { t: this.D[6]!, p: [1080, 700, 1700], l: [0, 170, 30], fn: mixEase },
      { t: this.tAuthor, p: [1200, 660, 1790], l: [0, 170, 20], fn: ease.outCubic },
    ];
  }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp, audio: au } = this.ctx;
    const t = f.t;
    if (t >= q(au, this.tSwitch, 2)) this.renderNode(t, out);
    else this.renderChip(f, out);

    this.ui.clear();
    const c = this.ui.ctx;
    const card = t >= q(au, this.D[6]!, 2);
    if (card) this.drawCard(c, t);
    else this.drawRecap(c, t);
    comp.draw(renderer, this.ui.upload(), out);

    const cut = hit(au, t, this.tSwitch, 0.09, 2);
    const node = t >= q(au, this.tSwitch, 2) && !card ? 1 : 0;   // the open node is dark metal: lift it a little
    const post: PostOverrides = { bloom: 0.6, bloomThreshold: 0.86, hud: 1, exposure: 1 + 0.25 * node + 0.9 * cut };
    if (card) post.frame = snap(au, t, this.D[6]!, 0.09);
    return post;
  }

  // ------------------------------------------------------------------ the chip: SM → die → module
  private renderChip(f: Frame, out: THREE.WebGLRenderTarget) {
    const { renderer, comp, audio: au } = this.ctx;
    const t = f.t, g = this.gpu, it = this.tItem, D = this.D;
    const beat = au.timeOfBeat(1) - au.timeOfBeat(0);

    // camera distance: log-interpolated keys that land on the downbeats
    const KD: [number, number][] = [[this.ctx.start, 1.45], [D[1]!, 1.9], [D[2]!, 3.4], [D[3]!, 26], [D[4]!, 105], [this.tSwitch, 175]];
    let ld = Math.log(KD[0]![1]);
    for (let n = 0; n + 1 < KD.length; n++) {
      const [ta, da] = KD[n]!, [tb, db] = KD[n + 1]!;
      if (t >= ta) ld = lerp(Math.log(da), Math.log(db), mixEase(clamp((t - ta) / (tb - ta))));
    }
    const dist = Math.exp(ld) * (1 - 0.035 * kickPush(au, t));
    const w = clamp((Math.log(dist) - Math.log(1.9)) / (Math.log(175) - Math.log(1.9)));   // 0 SM → 1 module
    const tgt = g.smPos(SM).lerp(new THREE.Vector3(0, H100_Y.dieTop * 0.6, 0), smoothstep(0.15, 0.9, w));
    const az = 0.6 + drive(au, t, 0.07, 0.35, this.ctx.start);
    const el = lerp(0.68, 0.5, w);
    const cam = this.cam;
    cam.position.set(tgt.x + Math.sin(az) * Math.cos(el) * dist, tgt.y + Math.sin(el) * dist, tgt.z + Math.cos(az) * Math.cos(el) * dist);
    cam.up.set(0, 1, 0);
    cam.lookAt(tgt);
    H100.fitClip(cam, dist);

    // die-level activity: SMs flare for "tensor cores", a band of tiles streams for "flash attention"
    const fi = frameIdx(t) >> 2;
    if (t >= q(au, it[5]!) && t < q(au, it[6]!)) {
      const step = Math.floor((t - q(au, it[5]!)) / (beat * 0.5));      // one column step per 8th
      const bx = -12 + (step % 7) * 4;
      g.setSMActivity((i) => (g.smEnabled(i) ? 0.18 + 1.3 * Math.exp(-Math.pow((g.smPos(i).x - bx) / 2.2, 2)) : 0));
    } else if (t >= q(au, it[4]!) && t < q(au, it[5]!)) {
      const k = hit(au, t, it[4]!, 0.22);
      g.setSMActivity((i) => (g.smEnabled(i) ? 0.45 + 1.6 * k + 0.35 * f.a.kick * (hash(i, fi) < 0.5 ? 1 : 0) : 0));
    } else if (t >= q(au, it[3]!)) {
      // tiles: the die works through its tiles, one slice of SMs per 8th
      const step = Math.floor((t - q(au, it[3]!)) / (beat * 0.5));
      g.setSMActivity((i) => (g.smEnabled(i) ? (i % 6 === step % 6 ? 1.25 : 0.14) : 0));
    } else {
      // inside the SM: the neighbours stay low so the close-up reads
      g.setSMActivity((i) => (g.smEnabled(i) && i !== SM ? 0.08 + 0.3 * f.a.kick * (hash(i, fi) < 0.3 ? 1 : 0) : 0));
    }
    if (t >= q(au, it[6]!)) {
      const hb = hit(au, t, au.timeOfBeat(Math.floor(au.beatAt(t) * 2) / 2), 0.1);
      g.setHBMActivity(() => 0.7 + 1.6 * hb);
    } else if (t < q(au, it[4]!)) {
      // the HBM PHYs sit next to this SM: keep them low inside it, breathing on the beat once memory is named
      const on = t >= q(au, it[2]!) ? 1 : 0;
      g.setHBMActivity(() => 0.1 + on * (0.2 + 0.35 * hit(au, t, au.timeOfBeat(Math.floor(au.beatAt(t))), 0.15, 1)));
    } else g.setHBMActivity(null);
    g.update(t, f.a);

    // data pulses once the camera is off the SM (they are long at SM scale): steady from "memory", short
    // dense bursts ("bits") from "4 bits"
    const items: PulseItem[] = [];
    const pg = smoothstep(3.5, 7, dist) * (t >= q(au, it[2]!) ? 1 : 0);
    if (pg > 0) {
      const bits = t >= q(au, it[6]!), ws = clamp(dist / 26, 1, 5);   // keep them a visible width as we pull out
      this.paths.forEach((p, n) => {
        const per = bits ? 9 : 3;
        for (let m = 0; m < per; m++) {
          const u = (((t - it[2]!) * (bits ? 0.55 : 0.32) + m / per + n * 0.137) % 1 + 1) % 1;
          items.push({ curve: p, u, len: bits ? 0.012 : 0.03, gain: 2.3 * pg, width: (bits ? 0.11 : 0.15) * ws });
        }
      });
    }
    g.pulses.set(items);

    // the SM close-up, lit by what the words name
    if (dist < 40) {
      const d = g.smDetail(SM);
      const tb = t - it[0]!;
      const onBeat = hit(au, t, au.timeOfBeat(Math.floor(au.beatAt(t))), 0.16, 1);
      d.setLanes((p, kind, idx) => {
        const lane = kind === 'fp32' ? idx : kind === 'int32' ? 32 + idx : 64 + idx;
        if (t < q(au, it[0]!)) return hash(p, lane, fi) < 0.25 ? 0.5 : 0.08;
        if (t < q(au, it[1]!)) return hash(p, lane, fi) < 0.45 ? 1.1 : 0.1;           // threads: each on its own
        return kind === 'fp32' ? 0.15 + 1.2 * onBeat : 0.08 + 0.3 * onBeat;         // warps: lockstep
      });
      d.setUnits((u) => {
        let v = u === 'sched' ? 0.3 + 0.3 * f.a.kick : 0;
        if (t >= q(au, it[2]!) && (u === 'ldst' || u === 'l1')) v = 0.9 + 0.6 * onBeat;
        if (t >= q(au, it[3]!) && u === 'l1') v = 1.5;
        if (t >= q(au, it[3]!) && u === 'regfile') v = 0.55;
        return v;
      });
      d.setTensor((p, i, j, k) => (t >= q(au, it[4]!) ? (((Math.floor(tb * 8) + i + j + k + p) % 4) === 0 ? 1.1 : 0.08) : 0.04));
    } else g.hideSMDetail();

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    comp.draw(renderer, this.ms.texture, out);
  }

  // ------------------------------------------------------------------ the node: 8 GPUs, teardown reversed
  private renderNode(t: number, out: THREE.WebGLRenderTarget) {
    const { renderer, audio: au } = this.ctx;
    const d = this.dgx, D = this.D;
    // explode: fully open on the cut, closing back up by the card's downbeat
    const EK: [number, number][] = [[this.tSwitch, 0.92], [D[5]!, 0.62], [D[6]!, 0]];
    let ex = EK[0]![1];
    for (let n = 0; n + 1 < EK.length; n++) {
      if (t >= EK[n]![0]) ex = lerp(EK[n]![1], EK[n + 1]![1], ease.inOutCubic(clamp((t - EK[n]![0]) / (EK[n + 1]![0] - EK[n]![0]))));
    }
    d.setExplode(ex);
    const k6 = clamp((t - this.tSwitch) / (D[6]! - this.tSwitch));
    if (t < D[6]!) d.setTraffic({ pattern: 'allreduce', intensity: lerp(1, 0.45, k6), rate: lerp(4.2, 1.3, k6) });
    else d.setTraffic({ pattern: 'idle', intensity: t < this.tAuthor ? 0.3 : 0.14, rate: 1 });
    d.update(t);

    // camera keys; after the last one it holds still
    const K = this.dKeys;
    const P = new THREE.Vector3(...K[0]!.p), L = new THREE.Vector3(...K[0]!.l);
    for (let n = 0; n + 1 < K.length; n++) {
      const a = K[n]!, b = K[n + 1]!;
      if (t >= a.t) {
        const u = b.fn(clamp((t - a.t) / (b.t - a.t)));
        P.set(...a.p).lerp(new THREE.Vector3(...b.p), u);
        L.set(...a.l).lerp(new THREE.Vector3(...b.l), u);
      }
    }
    // while the node closes, the camera also swings round it, surging with the drums (lands at 0 on the card)
    const dr = (x: number) => drive(au, x, 0.1, 0.5, this.tSwitch);
    const rot = -0.34 * (1 - clamp(dr(Math.min(t, D[6]!)) / dr(D[6]!)));
    P.sub(L).applyAxisAngle(new THREE.Vector3(0, 1, 0), rot).add(L);
    // a push along the view axis on the kicks while the band plays (a nudge, not a shake)
    if (t < this.tAuthor) P.add(L.clone().sub(P).normalize().multiplyScalar(18 * kickPush(au, t)));
    const cam = this.cam;
    cam.position.copy(P);
    cam.up.set(0, 1, 0);
    cam.lookAt(L);
    cam.near = 2; cam.far = 20000;
    cam.updateProjectionMatrix();
    d.render(renderer, out, cam);
  }

  // ------------------------------------------------------------------ the recap words
  private drawRecap(c: CanvasRenderingContext2D, t: number) {
    const au = this.ctx.audio;
    const cur = this.tItem.reduce((m, ti, i) => (t >= q(au, ti) ? i : m), -1);
    if (cur < 0) return;
    const LX = 120, LY = 150, LS = 50;
    // a soft ink pool under the type so it reads over a lit die
    const k0 = snap(au, t, this.tItem[0]!, 0.09);
    for (const [cx, cy, r, a] of [[LX + 380, H - 170, 820, 0.62], [LX + 120, LY + 150, 560, 0.45]] as const) {
      const gr = c.createRadialGradient(cx, cy, 0, cx, cy, r);
      gr.addColorStop(0, rgba('ink', a * k0));
      gr.addColorStop(1, rgba('ink', 0));
      c.fillStyle = gr;
      c.fillRect(cx - r, cy - r, 2 * r, 2 * r);
    }
    // the running list (top left): past stops dim, the current one in signal
    for (let i = 0; i <= cur; i++) {
      const k = snap(au, t, this.tItem[i]!, 0.09);
      c.globalAlpha = k;
      c.font = font(F.mono(500), 20);
      c.fillStyle = rgba(i === cur ? 'signal' : 'ash', 0.9);
      c.fillText(String(i + 1).padStart(2, '0'), LX, LY + i * LS);
      c.font = font(F.archivo(100, 700), 34);
      c.fillStyle = i === cur ? rgba('signal', 0.95) : rgba('bone', 0.55);
      c.fillText(ITEMS[i]!.word, LX + 48, LY + i * LS + 2);
    }
    // the current stop, big, bottom left
    const it = ITEMS[cur]!;
    const k = snap(au, t, this.tItem[cur]!, 0.09);
    const fam = F.archivo(it.word.length > 11 ? 75 : 100, 900);
    let px = 190;
    c.font = font(fam, px);
    const wd = c.measureText(it.word).width;
    if (wd > 1500) { px *= 1500 / wd; c.font = font(fam, px); }
    c.globalAlpha = k;
    c.fillStyle = rgba('bone', 0.97);
    c.fillText(it.word, LX, H - 120 + (1 - k) * 24);
    c.font = font(F.mono(500), 22);
    c.fillStyle = rgba('signal', 0.95);
    c.fillText(`${String(cur + 1).padStart(2, '0')} / 08`, LX, H - 120 - px * 0.78);
    c.fillStyle = rgba('ash', 0.85);
    c.fillText(it.ch, LX + 130, H - 120 - px * 0.78);
    c.globalAlpha = 1;
  }

  // ------------------------------------------------------------------ the end card, over the darkened node
  private drawCard(c: CanvasRenderingContext2D, t: number) {
    const au = this.ctx.audio;
    const dark = 0.5 * snap(au, t, this.D[6]!, 0.09) + 0.25 * snap(au, t, this.tAuthor, 0.09);
    c.fillStyle = rgba('ink', dark);
    c.fillRect(0, 0, W, H);
    const TITLE = 'CUDA for Deep Learning';
    const fT = F.serif(600);
    c.font = font(fT, 100);
    const ST = Math.min(150, (100 * 1300) / c.measureText(TITLE).width);
    const lay = layout(TITLE, fT, ST);
    const x0 = W / 2 - lay.width / 2, YT = 500;
    c.font = font(fT, ST); c.fillStyle = rgba('bone'); c.textBaseline = 'alphabetic';
    let word = 0;
    for (const gl of lay.glyphs) {
      if (gl.ch === ' ') { word++; continue; }
      const k = snap(au, t, this.tWord[word]!, 0.09);
      if (k <= 0) continue;
      c.globalAlpha = k;
      c.fillText(gl.ch, x0 + gl.x, YT + (1 - k) * 28);
    }
    c.globalAlpha = 1;
    const ka = snap(au, t, this.tAuthor, 0.09);
    if (ka > 0) {
      c.globalAlpha = ka;
      c.font = font(F.serif(400), 46); c.fillStyle = rgba('bone', 0.85); c.textAlign = 'center';
      c.letterSpacing = '6px';
      c.fillText('ELLIOT ARLEDGE', W / 2, YT + 110);
      c.letterSpacing = '0px'; c.textAlign = 'left';
      c.fillStyle = rgba('signal', 0.9);
      c.fillRect(W / 2 - 130 * ka, YT + 160, 260 * ka, 2);
      c.globalAlpha = 1;
    }
  }
}
