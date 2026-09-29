// Launching ten thousand threads is easy; the real question is how fast they run.
//
// LAUNCH (v5, pre-chorus 1, CH02), in 3D on the shared GH100 die (_h100.ts).
//   "Ten thousand threads all launch at once,": the camera circles the whole die at a low angle, its
//   big moves landing on the downbeats; the dim SMs pulse together on the 8th grid, then on 16ths from
//   "all" (the tension). On "launch" the host / GigaThread block on the die's north edge hands the blocks
//   out: a streak flies from it to every enabled SM, `kernel<<<blocks, threads>>>();` snaps in small,
//   and every streak lands together on "once", where all 132 SMs flare in one hit (CH02: 132 SMs on an
//   H100 SXM) and stay busy, flickering, with data flowing in from the HBM stacks.
//   "So tell me, how fast did it run?": the camera pulls up and away fast (landing on the downbeat
//   after "So"), keeps rising to a near top-down view with the busy die centred, while the question
//   builds on screen: HOW / FAST, a "speed ? FLOP/s" readout whose digits roll while it "measures", and
//   on "run?" a huge "?" hits dead centre, which the dive transition into the chorus flies into.
// Continuous motion via scenes/_cam.ts (orbit driven by the drums), accents via scenes/_lock.ts.
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, clearRT, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font, measure } from '../engine/type';
import { Lyrics, type Line } from '../engine/lyrics';
import { clamp, ease, frameIdx, hash, lerp } from '../engine/util';
import { H100, H100_Y, DIM, type PulseItem } from './_h100';
import { DIE_LAYOUT } from './_h100-tex';
import { snap, hit, q } from './_lock';
import { beatEase, drive, kickPush } from './_cam';

const LYR_X = 112, LYR_Y = 150, LYR_SIZE = 76; // the sung line always sits here

export default class Launch extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(32, W / H, 0.05, 5000);
  gpu!: H100;
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();
  l1!: Line; l2!: Line;
  /** Dispatch curves: host / GigaThread block → every enabled SM (with each one's SM index). */
  dispatch: { curve: THREE.Curve<THREE.Vector3>; sm: number; d: number }[] = [];
  mem: THREE.Curve<THREE.Vector3>[] = [];
  enabled: number[] = [];
  hostPos = new THREE.Vector3();
  // grid-quantized event times
  tAll = 0; tLaunch = 0; tOnce = 0; tSwap = 0; tPull = 0; tPullEnd = 0; tHow = 0; tFast = 0; tDid = 0; tRun = 0;
  db: number[] = [];

  override async init() {
    const { lyrics, audio: au } = this.ctx;
    this.gpu = new H100({ renderer: this.ctx.renderer });
    const rig = H100.rig();
    // the low orbit looks toward the ember rim light: at the shared 2.2 its specular on the east HBM stacks
    // and the board parts behind them blooms into a flare, so this scene runs it much softer
    rig.children.forEach((l) => { if (l instanceof THREE.DirectionalLight && l.position.z < 0) l.intensity = 0.12; });
    this.scene.add(this.gpu.group, rig);
    this.scene.environment = this.gpu.env;
    // grazing low shots catch the environment's light panel on the mirror-dark die top as a blown sheen
    this.scene.environmentIntensity = 0.5;
    this.scene.background = new THREE.Color().setRGB(LIN.ink[0], LIN.ink[1], LIN.ink[2], THREE.LinearSRGBColorSpace);

    this.l1 = lyrics.get('Ten thousand threads');
    this.l2 = lyrics.get('So tell me');
    const w1 = (s: string) => this.l1.words.find((w) => w.w.toLowerCase().startsWith(s))!;
    const w2 = (s: string) => this.l2.words.find((w) => w.w.toLowerCase().startsWith(s))!;
    this.tAll = q(au, w1('all').start);
    this.tLaunch = q(au, w1('launch').start);
    this.tOnce = q(au, w1('once').start);
    // the lyric swaps on the grid point nearest "So", after line 1 has ended, between two frames
    const sw = Math.max(q(au, w2('so').start), this.l1.end + 0.02);
    this.tSwap = (Math.floor(sw * 60) + 0.5) / 60;
    this.tHow = w2('how').start;
    this.tFast = w2('fast').start;
    this.tDid = w2('did').start;
    this.tRun = w2('run').start;
    this.db = au.downbeats.filter((d) => d > this.ctx.start - 0.5 && d < this.ctx.end + 0.5);
    // the pull-back: starts on the grid point nearest "So", lands on the next downbeat
    this.tPull = q(au, w2('so').start);
    this.tPullEnd = this.db.find((d) => d > this.tPull + 0.3) ?? this.tPull + 0.9;

    // the host interface + GigaThread engine block along the die's north edge (DIE_LAYOUT.host)
    const g = this.gpu, { w, d } = DIM.die, hr = DIE_LAYOUT.host;
    const yTop = H100_Y.dieTop;
    this.hostPos.set(0, yTop + 0.05, ((hr.v0 + hr.v1) / 2) * d - d / 2);
    for (let i = 0; i < g.smCount; i++) {
      if (!g.smEnabled(i)) continue;
      this.enabled.push(i);
      const p = g.smPos(i);
      // leave from a point along the host block under the SM's column, arc up over the die, land on the SM
      const a = new THREE.Vector3(clamp(p.x, (hr.u0 - 0.5) * w + 1, (hr.u1 - 0.5) * w - 1), yTop + 0.05, this.hostPos.z);
      const b = new THREE.Vector3(p.x, yTop + 0.05, p.z);
      const dist = a.distanceTo(b);
      const m = a.clone().lerp(b, 0.5); m.y += 1.2 + dist * 0.12;
      this.dispatch.push({ curve: new THREE.QuadraticBezierCurve3(a, m, b), sm: i, d: dist });
    }
    // memory traffic: each active HBM stack → a few SMs
    for (let k = 0; k < 6; k++) {
      if (!g.hbm(k).active) continue;
      for (let n = 0; n < 3; n++) this.mem.push(g.pulsePath(k, this.enabled[Math.floor(hash(k, n, 11) * this.enabled.length)]!));
    }
  }

  /** SM glow: dim pulses on the grid (8ths, then 16ths from "all"), one flare on "once", then busy. */
  private smLevel(i: number, t: number): number {
    const au = this.ctx.audio;
    const amp = 0.6 + 0.4 * hash(i, 3);
    if (t < this.tOnce) {
      const div = t < this.tAll ? 2 : 4;
      const b = au.beatAt(t) * div;
      const tick = au.timeOfBeat(Math.floor(b) / div);
      const pulse = Math.exp(-(t - tick) / (div === 2 ? 0.09 : 0.05));
      // the stored pulse grows through the line (more waiting, more tension)
      const grow = lerp(0.12, 0.3, clamp((t - this.ctx.start) / Math.max(0.1, this.tOnce - this.ctx.start)));
      return 0.04 + grow * pulse * amp;
    }
    const flare = 2.2 * hit(au, t, this.tOnce, 0.16);
    const fi = frameIdx(t) >> 2;
    const busy = 0.45 + 0.4 * (hash(i, fi) < 0.6 ? 1 : 0.35) + 0.3 * au.hit('kick', t, 0.1);
    return Math.max(flare, busy * snap(au, t, this.tOnce, 0.05));
  }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, audio: au, comp } = this.ctx;
    const t = f.t;
    const g = this.gpu;

    g.explode(0);
    g.setSMActivity((i) => (g.smEnabled(i) ? this.smLevel(i, t) : 0));
    g.update(t, f.a);

    // ---- camera: a low orbit that surges with the drums and steps on downbeats; then the pull-back
    const die = new THREE.Vector3(0, H100_Y.dieTop, 0);
    let az = -0.55 + 0.055 * drive(au, t, 1, 2, this.ctx.start);
    for (const db of this.db) if (db < this.tPull) az += 0.24 * beatEase(au, t, db - 0.45, db);
    const kFire = hit(au, t, this.tOnce, 0.12);
    const pull = beatEase(au, t, this.tPull, this.tPullEnd, ease.inOutCubic);
    const rise = clamp((t - this.tPullEnd) / Math.max(0.1, this.ctx.end - this.tPullEnd));
    let el = lerp(0.34, 1.18, pull) + 0.26 * ease.outCubic(rise);
    let dist = lerp(64, 150, pull) + 26 * ease.outCubic(rise);
    dist *= 1 - 0.05 * kFire - 0.012 * kickPush(au, t) * (1 - pull);
    el = Math.min(el, 1.48);
    const tgt = die.clone();
    tgt.y += lerp(-1.5, 0, pull); // low shots look slightly over the die's far edge
    const cam = this.cam;
    cam.position.set(tgt.x + Math.sin(az) * Math.cos(el) * dist, tgt.y + Math.sin(el) * dist, tgt.z + Math.cos(az) * Math.cos(el) * dist);
    cam.up.set(0, 1, 0);
    cam.lookAt(tgt);
    H100.fitClip(cam, dist);

    // ---- data: dispatch streaks (launch → once), then memory traffic HBM → SMs (always some flowing)
    const items: PulseItem[] = [];
    if (t >= this.tLaunch - 0.02 && t < this.tOnce + 0.12) {
      const span = this.tOnce - this.tLaunch;
      for (const dp of this.dispatch) {
        // nearer SMs leave later so every streak lands on "once"
        const leave = this.tLaunch + span * 0.35 * (1 - dp.d / 30);
        const u = clamp((t - leave) / Math.max(0.05, this.tOnce - leave));
        if (t < leave) continue;
        const land = clamp((t - this.tOnce) / 0.12);
        items.push({ curve: dp.curve, u: Math.min(u, 0.999), len: 0.18, gain: 3.2 * (1 - land), width: 0.14 });
      }
    }
    const memGain = t < this.tOnce ? 0.9 : 2.0;
    this.mem.forEach((c, n) => {
      for (let j = 0; j < 3; j++) {
        const u = (t * (t < this.tOnce ? 0.22 : 0.45) + j / 3 + n * 0.071) % 1;
        items.push({ curve: c, u, len: 0.03, gain: memGain, width: 0.15 });
      }
    });
    g.pulses.set(items);

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    clearRT(renderer, out, LIN.ink);
    comp.draw(renderer, this.ms.texture, out);

    // ---- screen-space overlay
    const L = this.ui; L.clear();
    const c = L.ctx;
    this.overlay(c, t);
    this.karaoke(c, t < this.tSwap ? this.l1 : this.l2, t);
    comp.draw(renderer, L.upload(), out);

    const hRun = hit(au, t, this.tRun, 0.08);
    return { bloom: 0.6 + 0.25 * kFire + 0.2 * hRun, bloomThreshold: 0.8, vignette: 0.4, flash: 0.04 * kFire + 0.03 * hRun };
  }

  private project(p: THREE.Vector3) {
    const v = this.gpu.group.localToWorld(p.clone()).project(this.cam);
    return { x: (v.x * 0.5 + 0.5) * W, y: (-v.y * 0.5 + 0.5) * H, ok: v.z < 1 };
  }

  private overlay(c: CanvasRenderingContext2D, t: number) {
    const au = this.ctx.audio;
    const fM = F.mono(500);
    const kLaunch = snap(au, t, this.tLaunch);
    const kOnce = snap(au, t, this.tOnce);
    const kSwap = t >= this.tSwap ? snap(au, t, this.tSwap) : 0;
    const l1a = 1 - kSwap;

    // line 1: the host block label, then the launch line under the frame, then "all at once · 132 SMs"
    if (l1a > 0) {
      const hp = this.project(this.hostPos);
      if (hp.ok) {
        c.save(); c.globalAlpha = l1a;
        c.fillStyle = rgba('signal', 1); c.fillRect(hp.x - 4, hp.y - 4, 8, 8);
        c.strokeStyle = rgba('bone', 0.7); c.lineWidth = 1.5;
        c.beginPath(); c.moveTo(hp.x, hp.y - 6); c.lineTo(hp.x, hp.y - 70); c.lineTo(hp.x + 16, hp.y - 70); c.stroke();
        c.font = font(fM, 28); c.textBaseline = 'middle';
        c.fillStyle = rgba('bone', 0.95);
        c.fillText(kLaunch > 0 ? 'host · hands out the blocks' : '10,000 threads · waiting', hp.x + 24, hp.y - 70);
        c.restore();
      }
      if (kLaunch > 0) {
        const code = 'kernel<<<blocks, threads>>>();';
        c.font = font(fM, 36); c.textBaseline = 'alphabetic';
        const cw = measure(code, fM, 36);
        c.globalAlpha = l1a * kLaunch;
        c.fillStyle = rgba('bone', 0.95);
        c.fillText(code, W / 2 - cw / 2, 972);
        c.globalAlpha = 1;
      }
      if (kOnce > 0) {
        const s = 'all at once · 132 SMs', s2 = '  (CH02: an H100 SXM has 132)';
        c.font = font(F.archivo(100, 800), 58); c.textBaseline = 'alphabetic';
        const sw = measure(s, F.archivo(100, 800), 58);
        c.globalAlpha = l1a * kOnce;
        c.fillStyle = rgba('ink', 0.7); c.fillRect(W / 2 - sw / 2 - 30, 842, sw + 60, 76);
        c.fillStyle = rgba('bone', 1);
        c.fillText(s, W / 2 - sw / 2, 900);
        c.font = font(fM, 22); c.fillStyle = rgba('ash', 0.9);
        c.fillText(s2.trim(), W / 2 - measure(s2.trim(), fM, 22) / 2, 1012);
        c.globalAlpha = 1;
      }
    }

    // line 2: HOW / FAST, the readout that rolls while it measures, then the "?" dead centre
    if (kSwap > 0) {
      const kHow = snap(au, t, this.tHow, 0.09, 4), kFast = snap(au, t, this.tFast, 0.09, 4), kDid = snap(au, t, this.tDid, 0.09, 4), kRun = snap(au, t, this.tRun, 0.09, 4);
      const hRun = hit(au, t, this.tRun, 0.08);
      const dimAll = 1 - 0.75 * kRun;
      const fT = F.archivo(100, 900);
      const sz = 150;
      const how = 'HOW', fast = 'FAST';
      const wH = measure(how, fT, sz), wF = measure(fast, fT, sz), gap = 44;
      const x0 = W / 2 - (wH + gap + wF) / 2, y = 470;
      c.font = font(fT, sz); c.textBaseline = 'alphabetic';
      if (kHow > 0) { c.globalAlpha = kHow * dimAll; c.fillStyle = rgba('bone', 1); c.fillText(how, x0, y); }
      if (kFast > 0) { c.globalAlpha = kFast * dimAll; c.fillStyle = rgba('bone', 1); c.fillText(fast, x0 + wH + gap, y); }
      c.globalAlpha = 1;
      // readout: "speed  ?  FLOP/s", and from "did" a measuring bar that steps on 16ths until "run?"
      if (kFast > 0) {
        const yr = 670, fR = F.mono(600);
        const lab = 'speed', unit = 'FLOP/s', qm = '?';
        const wl = measure(lab, fR, 52), wd = measure(qm, fR, 88), wu = measure(unit, fR, 52);
        const tot = wl + 30 + wd + 30 + wu;
        let x = W / 2 - tot / 2;
        c.globalAlpha = kFast * dimAll;
        c.fillStyle = rgba('ink', 0.55 * c.globalAlpha); c.fillRect(W / 2 - tot / 2 - 28, yr - 92, tot + 56, 150);
        c.font = font(fR, 52); c.fillStyle = rgba('ash', 1); c.fillText(lab, x, yr); x += wl + 30;
        c.font = font(fR, 88); c.fillStyle = rgba(kDid > 0 ? 'signal' : 'bone', 1); c.fillText(qm, x, yr + 12); x += wd + 30;
        c.font = font(fR, 52); c.fillStyle = rgba('bone', 1); c.fillText(unit, x, yr);
        if (kDid > 0) {
          const n = 16, bw = tot / n;
          const k0 = Math.floor(au.beatAt(this.tDid) * 4);
          const lit = kRun > 0 ? n : Math.min(n, Math.max(0, Math.floor(au.beatAt(t) * 4) - k0 + 1) * 2);
          for (let j = 0; j < n; j++) {
            c.fillStyle = rgba(j < lit ? 'signal' : 'graphite', j < lit ? 0.9 : 0.6);
            c.fillRect(W / 2 - tot / 2 + j * bw + 2, yr + 30, bw - 4, 10);
          }
          c.font = font(fR, 22); c.fillStyle = rgba('ash', 0.9);
          c.fillText('measuring…', W / 2 - tot / 2, yr + 72 - 4);
        }
        c.globalAlpha = 1;
      }
      // "run?": one huge question mark, dead centre (the dive into the chorus flies into it)
      if (kRun > 0) {
        const qs = 380 * (1 + 0.06 * hRun);
        c.font = font(fT, qs); c.textBaseline = 'middle';
        const wq = measure('?', fT, qs);
        c.globalAlpha = kRun;
        c.fillStyle = rgba('signal', 1);
        c.fillText('?', W / 2 - wq / 2, H / 2 + qs * 0.04);
        c.globalAlpha = 1;
        c.textBaseline = 'alphabetic';
      }
    }
  }

  /** Per-word karaoke: unsung dim bone, the word being sung wipes in signal, sung words bone. */
  private karaoke(c: CanvasRenderingContext2D, line: Line, t: number) {
    const fam = F.archivo(100, 800);
    c.font = font(fam, LYR_SIZE); c.textBaseline = 'alphabetic';
    const sp = measure(' ', fam, LYR_SIZE);
    // a soft ink scrim behind the line so it reads over the lit die
    c.fillStyle = rgba('ink', 0.45);
    c.fillRect(0, LYR_Y - LYR_SIZE - 40, W, LYR_SIZE + 80);
    let x = LYR_X;
    for (const w of line.words) {
      const txt = w.w, wd = measure(txt, fam, LYR_SIZE);
      const p = Lyrics.wordProgress(w, t);
      c.fillStyle = rgba('bone', p >= 1 ? 1 : 0.3);
      c.fillText(txt, x, LYR_Y);
      if (p > 0 && p < 1) {
        c.save(); c.beginPath(); c.rect(x - 4, LYR_Y - LYR_SIZE, (wd + 8) * clamp(p), LYR_SIZE * 1.4); c.clip();
        c.fillStyle = rgba('signal', 1); c.fillText(txt, x, LYR_Y); c.restore();
      }
      x += wd + sp;
    }
  }
}
