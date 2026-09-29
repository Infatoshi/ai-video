// Idea: a slow kernel spends most of its time waiting for memory, not doing math.
//
// v5 scene `slow` (pre-chorus 2, "Same old code, but it's crawling slow, / Tell me, where'd all my
// speed go?"), on the shared H100 model. Three shots, cut on downbeats, the camera always moving:
//   A  (to the 2nd downbeat)  chase: one SM (lanes grey, idle) sends a request, and the camera flies
//      beside it the long way across the die and interposer to an HBM stack, while the held "Same"
//      drags on.
//   B  (to "Tell")  wide orbit over the package: the data crawls back, the SM does its math in a
//      blink on the downbeat, and the next request leaves at once. Every SM is doing the same (slow
//      traffic everywhere). A bar under it tallies the time: almost all grey (waiting), a sliver of
//      orange (math), with CH6's measurement under it.
//   C  (from "Tell")  a whip to the HBM stack, the camera pushing in; the DRAM dies part and MEMORY
//      slams on "speed", held through "go?". The stack sits at frame centre for the streak into chorus 2.
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font, layout, type TextLayout } from '../engine/type';
import type { Line, Word } from '../engine/lyrics';
import { clamp, ease, lerp, smoothstep, hash } from '../engine/util';
import { H100, type PulseItem } from './_h100';
import { drive, kickPush } from './_cam';
import { hit, q, snap } from './_lock';

// ---- screen layout (logical px) ------------------------------------------------------------------
const LYR_X = 120, LYR_Y = 180, LYR_FAM = F.archivo(87.5, 800), LYR_SIZE = 76;
const BAR_X0 = 480, BAR_X1 = 1440, BAR_Y = 944, BAR_H = 26;   // the waiting / math tally (shot B)
const MEM_Y = 958;                                           // MEMORY baseline (shot C)
const HERO_HBM = 1;                                          // west, middle stack
const MATH = 0.12;                                           // the math blink (s)
const N_AMB = 26;                                            // other SMs' slow traffic

const up = (s: string) => s.replace(/[“”"]/g, '');
const layCache = new Map<string, TextLayout>();
function lay(text: string, fam: string, size: number) {
  const k = `${text}|${fam}|${size}`;
  let l = layCache.get(k);
  if (!l) { l = layout(text, fam, size, 0); layCache.set(k, l); }
  return l;
}
function xAt(l: TextLayout, n: number) {
  if (n <= 0) return 0;
  const g = l.glyphs;
  if (n >= g.length) return l.width;
  const i = Math.floor(n), f = n - i;
  return lerp(g[i]!.x, i + 1 < g.length ? g[i + 1]!.x : l.width, f);
}

/** One lyric line with a per-glyph karaoke wipe (done bone, singing signal, unsung dim). */
function karaoke(c: CanvasRenderingContext2D, words: Word[], x: number, y: number, t: number, alpha = 1) {
  const texts = words.map((w) => up(w.w));
  const text = texts.join(' ');
  const L = lay(text, LYR_FAM, LYR_SIZE);
  let off = 0, doneN = 0, sungN = 0;
  for (let i = 0; i < words.length; i++) {
    const w = words[i]!, s = texts[i]!;
    if (t >= w.end) { doneN = off + s.length; sungN = doneN; }
    else if (t > w.start) { doneN = off; sungN = off + (s.length * (t - w.start)) / Math.max(1e-3, w.end - w.start); break; }
    else break;
    off += s.length + 1;
  }
  c.save();
  c.globalAlpha = alpha;
  c.font = font(LYR_FAM, LYR_SIZE);
  c.textBaseline = 'alphabetic';
  c.fillStyle = rgba('bone', 0.3);
  c.fillText(text, x, y);
  const top = y - LYR_SIZE * 1.05, h = LYR_SIZE * 1.4;
  const xd = x + xAt(L, doneN), xs = x + xAt(L, sungN);
  if (doneN > 0) {
    c.save(); c.beginPath(); c.rect(x - 20, top, xd - x + 20, h); c.clip();
    c.fillStyle = rgba('bone', 1); c.fillText(text, x, y); c.restore();
  }
  if (sungN > doneN) {
    c.save(); c.beginPath(); c.rect(xd, top, xs - xd, h); c.clip();
    c.fillStyle = rgba('signal', 1); c.fillText(text, x, y); c.restore();
  }
  c.restore();
}

function word(l: Line, s: string): Word {
  const k = s.replace(/[^a-z]/g, '');
  const w = l.words.find((x) => x.w.toLowerCase().replace(/[^a-z]/g, '').startsWith(k));
  if (!w) throw new Error(`slow: word not found: ${s}`);
  return w;
}

interface Amb { path: THREE.CurvePath<THREE.Vector3>; ph: number; per: number; out: boolean }

export default class Slow extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(34, W / H, 0.05, 5000);
  gpu!: H100;
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();

  private L1!: Line; private L2!: Line;
  private hero = 0;
  private path!: THREE.CurvePath<THREE.Vector3>;
  private amb: Amb[] = [];
  // schedule (song time)
  private tReq0 = 0; private d1 = 0; private d2 = 0; private d3 = 0; private d4 = 0;
  private tWhip = 0; private tWhere = 0; private tSpeed = 0;

  override async init() {
    const ly = this.ctx.lyrics, au = this.ctx.audio;
    this.L1 = ly.get('crawling slow');
    this.L2 = ly.get("where'd all my speed");
    this.gpu = new H100({ renderer: this.ctx.renderer });
    this.scene.add(this.gpu.group, H100.rig());
    this.scene.environment = this.gpu.env;
    this.scene.environmentIntensity = 0.28;   // grazing views across the die mirror the room env as a white glare
    this.scene.background = new THREE.Color().setRGB(LIN.ink[0], LIN.ink[1], LIN.ink[2], THREE.LinearSRGBColorSpace);
    const g = this.gpu;
    // the hero SM: the far (east) side of the die from the west HBM stack, in the north SM rows
    let best = -1e9;
    for (let i = 0; i < g.smCount; i++) {
      if (!g.smEnabled(i)) continue;
      const p = g.smPos(i);
      const s = p.x - 0.6 * Math.abs(p.z + 7);
      if (s > best) { best = s; this.hero = i; }
    }
    this.path = g.pulsePath(HERO_HBM, this.hero);
    // everyone else's slow traffic
    const act = [0, 1, 2, 3, 4, 5].filter((k) => g.hbm(k).active);
    const ens = Array.from({ length: g.smCount }, (_, i) => i).filter((i) => g.smEnabled(i) && i !== this.hero);
    for (let n = 0; n < N_AMB; n++) {
      const k = act[Math.floor(hash(n, 11) * act.length)]!;
      const i = ens[Math.floor(hash(n, 23) * ens.length)]!;
      this.amb.push({ path: g.pulsePath(k, i), ph: hash(n, 37), per: 3.2 + 1.4 * hash(n, 41), out: hash(n, 53) < 0.5 });
    }
    // schedule: request leaves on "Same" (grid), arrives at HBM on the next downbeat, data returns by the
    // following one (math blink), the next request leaves at once; downbeats of this stretch
    this.tReq0 = q(au, word(this.L1, 'same').start);
    const db = au.downbeats.filter((d) => d > this.tReq0 + 0.5);
    [this.d1, this.d2, this.d3, this.d4] = [db[0]!, db[1]!, db[2]!, db[3]!];
    this.tWhip = q(au, word(this.L2, 'tell').start);
    this.tWhere = q(au, word(this.L2, "where'd").start);
    this.tSpeed = q(au, word(this.L2, 'speed').start, 2);
  }

  /** The hero pulse: path parameter (u: 0 at the HBM stack, 1 at the SM), or -1 when it's the math blink. */
  private heroU(t: number): { u: number; dir: number } {
    const { tReq0, d1, d2, d3, d4 } = this;
    if (t < tReq0) return { u: 1, dir: 0 };
    if (t < d1) return { u: 1 - (t - tReq0) / (d1 - tReq0), dir: -1 };
    if (t < d2) return { u: (t - d1) / (d2 - d1), dir: 1 };
    if (t < d2 + MATH) return { u: -1, dir: 0 };
    if (t < d3) return { u: 1 - (t - d2 - MATH) / (d3 - d2 - MATH), dir: -1 };
    if (t < d4) return { u: (t - d3) / (d4 - d3), dir: 1 };
    return { u: 1 - (t - d4) / (d4 - d3), dir: -1 };
  }

  private pt(u: number) { return this.path.getPointAt(clamp(u)); }

  /** Shot A: beside the request, looking ahead at where it still has to go. */
  private shotA(t: number, au: Frame['a'] | undefined): { pos: THREE.Vector3; tgt: THREE.Vector3; dist: number } {
    const { u } = this.heroU(t);
    const uu = u < 0 ? 1 : u;
    const sm = this.gpu.smPos(this.hero);
    // opening: close on the idle SM, drifting
    const ang = 0.9 + 0.12 * (t - this.tReq0);
    const open = { pos: sm.clone().add(new THREE.Vector3(Math.cos(ang) * 2.6, 2.1, Math.sin(ang) * 2.6)), tgt: sm.clone().add(new THREE.Vector3(-0.4, 0, 0)) };
    // chase: trail behind the pulse (toward the SM), look ahead (toward the HBM stack)
    const back = this.pt(uu + 0.06), ahead = this.pt(uu - 0.14), here = this.pt(uu);
    const dirv = ahead.clone().sub(back).setY(0).normalize();
    const side = new THREE.Vector3(-dirv.z, 0, dirv.x);
    const sw = 0.5 + 0.5 * Math.sin(drive(this.ctx.audio, t, 0.6, 1.2, this.tReq0) * 0.9);
    const chase = {
      pos: back.clone().add(side.multiplyScalar(lerp(0.9, 2.2, sw))).add(new THREE.Vector3(0, 1.7 + 0.6 * sw, 0)),
      tgt: ahead.clone().lerp(here, 0.35).add(new THREE.Vector3(0, 0.1, 0)),
    };
    // arriving: pull up and back so the stack it's headed for is in frame (not its bare top at 1 mm)
    const arr = smoothstep(0.42, 0.04, uu);
    const hb = this.gpu.hbm(HERO_HBM).pos;
    chase.pos.add(new THREE.Vector3(0, 5.5 * arr, 0)).add(dirv.clone().multiplyScalar(-5 * arr));
    chase.tgt.lerp(hb.clone().add(new THREE.Vector3(0, -0.3, 0)), arr * 0.8);
    const k = smoothstep(this.tReq0 - 0.05, this.tReq0 + 0.75, t);
    void au;
    return { pos: open.pos.lerp(chase.pos, k), tgt: open.tgt.lerp(chase.tgt, k), dist: 3.5 + 6 * arr };
  }

  /** Shot B: wide 3/4 over the package, orbiting with the drums. */
  private shotB(t: number): { pos: THREE.Vector3; tgt: THREE.Vector3; dist: number } {
    const au = this.ctx.audio;
    const az = -0.55 + 0.045 * drive(au, t, 1, 2, this.d1);
    const el = 0.78, dist = 78 - 4 * kickPush(au, t);
    const tgt = new THREE.Vector3(-3.5, 3, 1);
    const pos = tgt.clone().add(new THREE.Vector3(Math.sin(az) * Math.cos(el) * dist, Math.sin(el) * dist, Math.cos(az) * Math.cos(el) * dist));
    return { pos, tgt, dist };
  }

  /** Shot C: around the HBM stack, pushing in. */
  private shotC(t: number): { pos: THREE.Vector3; tgt: THREE.Vector3; dist: number } {
    const au = this.ctx.audio;
    const hb = this.gpu.hbm(HERO_HBM).pos;
    const k = clamp((t - this.tWhip) / Math.max(0.1, this.ctx.end - this.tWhip));
    const dist = lerp(19, 13, ease.outCubic(k)) - 0.8 * kickPush(au, t);
    const az = -2.05 + 0.06 * drive(au, t, 1, 2, this.tWhip);
    const el = lerp(0.36, 0.24, k);
    const tgt = hb.clone().add(new THREE.Vector3(0, -1.2, 0));
    const pos = tgt.clone().add(new THREE.Vector3(Math.sin(az) * Math.cos(el) * dist, Math.sin(el) * dist, Math.cos(az) * Math.cos(el) * dist));
    return { pos, tgt, dist };
  }

  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp, audio: au } = this.ctx;
    const t = f.t;
    const g = this.gpu;
    const hu = this.heroU(t);

    // ---- activity: every SM near idle (grey); the hero blinks orange for its math; the stack flares when a
    // request lands
    const blink = t >= this.d2 && t < this.d2 + MATH ? 1 : 0;
    const blinkHit = hit(au, t, this.d2, 0.07);
    g.setSMActivity((i) => (i === this.hero ? 0.05 + 1.6 * Math.max(blink, blinkHit) : g.smEnabled(i) ? 0.025 : 0));
    const land = Math.max(hit(au, t, this.d1, 0.18), hit(au, t, this.d3, 0.18));
    g.setHBMActivity((k) => (k === HERO_HBM ? 0.22 + 1.4 * land + 0.9 * hit(au, t, this.tSpeed, 0.2) : g.hbm(k).active ? 0.14 : 0));
    // on "speed" the memory stack blows apart into its DRAM layers (the whole module lifts a little with it)
    g.explode(0.34 * snap(au, t, this.tSpeed, 0.12));
    g.update(t, f.a);

    // ---- pulses: the hero (bright, with a short trail) and everyone else's slow traffic (dim)
    const items: PulseItem[] = [];
    const shotAOn = t < this.d1;
    if (hu.u >= 0) {
      const w = shotAOn ? 0.2 : 0.34, len = shotAOn ? 0.022 : 0.03;
      items.push({ curve: this.path, u: hu.u, len, gain: 3.2, width: w, color: LIN.signal });
      for (let j = 1; j <= 5; j++) {
        const uj = hu.u - hu.dir * j * 0.012;
        items.push({ curve: this.path, u: uj, len: len * 0.7, gain: 1.6 * Math.pow(0.62, j), width: w * 0.8 });
      }
    }
    for (const a of this.amb) {
      const ph = ((t - this.ctx.start) / a.per + a.ph) % 1;
      const u = a.out ? 1 - ph : ph;
      items.push({ curve: a.path, u, len: 0.02, gain: shotAOn ? 0.5 : 0.9, width: 0.16 });
    }
    g.pulses.set(items);

    // ---- the SM close-up only in shot A (grey, idle lanes)
    if (shotAOn) {
      const d = g.smDetail(this.hero);
      d.setLanes(() => 0.015);
      d.setTensor(() => 0.0);
      d.setUnits((u) => (u === 'sched' ? 0.18 + 0.12 * (Math.sin(t * 7) > 0 ? 1 : 0) : 0));
    } else g.hideSMDetail();

    // ---- camera: A until the first landing downbeat, B until "Tell", then a whip into C
    const cam = this.cam;
    let s: { pos: THREE.Vector3; tgt: THREE.Vector3; dist: number };
    if (t < this.d1) s = this.shotA(t, f.a);
    else if (t < this.tWhip) s = this.shotB(t);
    else {
      const wk = ease.inOutExpo(clamp((t - this.tWhip) / 0.2));
      const b = this.shotB(this.tWhip), c = this.shotC(t);
      s = { pos: b.pos.lerp(c.pos, wk), tgt: b.tgt.lerp(c.tgt, wk), dist: lerp(b.dist, c.dist, wk) };
    }
    cam.position.copy(s.pos);
    cam.up.set(0, 1, 0);
    cam.lookAt(s.tgt);
    H100.fitClip(cam, s.dist);

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    comp.draw(renderer, this.ms.texture, out);

    // ---- screen-space: lyric, labels, tally, MEMORY
    const c = this.ui.ctx;
    this.ui.clear();
    // a soft ink backing behind the lyric so it reads over the model
    const bg = c.createLinearGradient(0, 0, 0, 290);
    bg.addColorStop(0, rgba('ink', 0.78)); bg.addColorStop(1, rgba('ink', 0));
    c.fillStyle = bg; c.fillRect(0, 0, W, 290);
    const l2on = t >= this.L2.words[0]!.start - 0.4;
    if (!l2on) karaoke(c, this.L1.words, LYR_X, LYR_Y, t);
    else karaoke(c, this.L2.words, LYR_X, LYR_Y, t, t < this.L2.words[0]!.start ? 0.55 : 1);

    const project = (p: THREE.Vector3) => {
      const v = g.group.localToWorld(p.clone()).project(cam);
      return { x: (v.x * 0.5 + 0.5) * W, y: (-v.y * 0.5 + 0.5) * H, ok: v.z < 1 };
    };
    const tag = (p: THREE.Vector3, lines: string[], a: number, color: 'bone' | 'signal' = 'bone', dx = 22, dy = -22) => {
      if (a <= 0) return;
      const s2 = project(p);
      if (!s2.ok) return;
      c.save();
      c.globalAlpha = a;
      c.strokeStyle = rgba(color, 0.9); c.lineWidth = 1.5;
      c.beginPath(); c.moveTo(s2.x, s2.y); c.lineTo(s2.x + dx * 0.7, s2.y + dy * 0.7); c.stroke();
      c.fillStyle = rgba(color, 1); c.fillRect(s2.x - 4, s2.y - 4, 8, 8);
      c.font = font(F.mono(500), 30); c.textBaseline = 'alphabetic';
      lines.forEach((ln, i) => { c.fillStyle = rgba(i === 0 ? color : 'ash', i === 0 ? 1 : 0.9); c.fillText(ln, s2.x + dx, s2.y + dy + i * 34); });
      c.restore();
    };
    const smP = g.smPos(this.hero), hbP = g.hbm(HERO_HBM).pos;
    // shot A: the idle SM (from the first frame), then "memory" far ahead once the chase is under way
    if (t < this.d1) {
      tag(smP, ['SM', 'lanes idle, waiting'], 1 - smoothstep(this.tReq0 + 0.3, this.tReq0 + 0.5, t));
      tag(hbP, ['HBM · memory', 'the long way'], snap(au, t, this.tReq0 + 0.9, 0.09), 'bone', 26, -30);
    }
    // shot B: both ends labelled, the math blink called out
    if (t >= this.d1 && t < this.tWhip) {
      tag(smP, ['SM', blink || blinkHit > 0.3 ? 'math!' : 'waiting'], 1, blink || blinkHit > 0.3 ? 'signal' : 'bone', 28, -34);
      tag(hbP, ['HBM · memory'], 1, 'bone', -300, -44);
      // the tally: waiting (grey) vs math (orange) since the first request
      const el = Math.max(0, t - this.tReq0);
      const math = t >= this.d2 ? Math.min(MATH, t - this.d2) : 0;
      const tot = Math.max(1e-3, el);
      const wx = (BAR_X1 - BAR_X0) * ((tot - math) / tot);
      c.save();
      c.fillStyle = rgba('ink', 0.6); c.fillRect(BAR_X0 - 24, BAR_Y - 70, BAR_X1 - BAR_X0 + 48, 150);
      c.fillStyle = rgba('ash', 0.55); c.fillRect(BAR_X0, BAR_Y, wx, BAR_H);
      if (math > 0) { c.fillStyle = rgba('signal', 1); c.fillRect(BAR_X0 + wx + 3, BAR_Y - 4, Math.max(8, (BAR_X1 - BAR_X0) - wx - 3), BAR_H + 8); }
      c.font = font(F.mono(500), 24); c.textBaseline = 'alphabetic';
      c.fillStyle = rgba('bone', 0.95); c.fillText('waiting for memory', BAR_X0, BAR_Y - 18);
      c.textAlign = 'right'; c.fillStyle = rgba('signal', math > 0 ? 1 : 0.35); c.fillText('math', BAR_X1, BAR_Y - 18);
      c.textAlign = 'center'; c.fillStyle = rgba('bone', 0.85 * snap(au, t, this.d1 + 0.45));
      c.fillText('CH06 · the naive GEMM spends over 90% of its time stalled on memory', (BAR_X0 + BAR_X1) / 2, BAR_Y + 64);
      c.restore();
    }
    // shot C: MEMORY on "speed", held through "go?" into the streak
    const kSpeed = snap(au, t, this.tSpeed, 0.07);
    if (kSpeed > 0) {
      const slam = hit(au, t, this.tSpeed, 0.09);
      const fam = F.archivo(112.5, 900), size = 150;
      const sc = (1 + 0.06 * slam) * lerp(1.3, 1, kSpeed);
      c.save();
      c.translate(W / 2, MEM_Y);
      c.scale(sc, sc);
      c.globalAlpha = kSpeed;
      c.font = font(fam, size); c.textAlign = 'center'; c.textBaseline = 'alphabetic';
      c.fillStyle = rgba('signal', 1);
      c.fillText('MEMORY', 0, 0);
      c.restore();
    }
    // the whip into C lands with a flash of the stack label
    if (t >= this.tWhip + 0.2) tag(hbP, ['HBM stack'], 1 - kSpeed, 'bone', 34, -40);
    comp.draw(renderer, this.ui.upload(), out);

    void this.tWhere;
    return { bloom: 0.6, bloomThreshold: 0.85 };
  }
}
