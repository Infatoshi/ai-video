// A CPU has a few big cores; a GPU has thousands of small ones.
//
// `cpugpu` (v5, CH1), 0–13.56 s, in 3D on the shared H100 model. Intro: the H100 SXM5 module rises out
// of the dark, rim-lit, under the title card (CHASING THE / ROOFLINE; subtitle on downbeat 2, off on
// downbeat 4) while the camera orbits it with the drums. "A CPU thinks with a few big cores,": a
// lidless CPU package beside it, four fat cores snapping on (CPU / thinks / big / cores) as the camera
// swings across to frame both. "A GPU just throws ten thousand more,": the camera lunges onto the GH100
// die (landing on the downbeat at "GPU") and the SMs light in waves outward from the centre, one wave
// per 8th, with a counter to 10,000; then it keeps diving toward the die surface into the whip to
// `index`. The camera never stops; its big moves land on downbeats (scenes/_cam.ts), accents snap
// (scenes/_lock.ts). Numbers: 10,000 is the lyric; 132 SMs × 128 FP32 cores = 16,896 is the H100 SXM5
// (NVIDIA H100 whitepaper, as in _h100.ts).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font, layout, type TextLayout } from '../engine/type';
import { Lyrics, norm, type Line, type Word } from '../engine/lyrics';
import { clamp, ease, lerp, smoothstep } from '../engine/util';
import { q, snap, hit } from './_lock';
import { drive, beatEase, kickPush } from './_cam';
import { H100, H100_Y, glowMat, type PulseItem } from './_h100';

const LYR = { x: 150, base: 978, px: 66 };        // the karaoke line: always here (bottom-left)
const CPU_POS = new THREE.Vector3(-150, 0, 6);     // the CPU package, beside the module (mm)
const MID = new THREE.Vector3(-78, 0, 0);          // between the two chips

const lin = (k: keyof typeof LIN, s = 1) => new THREE.Color().setRGB(LIN[k][0] * s, LIN[k][1] * s, LIN[k][2] * s, THREE.LinearSRGBColorSpace);

interface KWord { w: Word; x: number; lay: TextLayout }
interface KLine { l: Line; words: KWord[]; show: number; hide: number }

/** A simple lidless CPU package: substrate, die, four fat cores (control / math / cache), shared cache. */
class CPUModel {
  group = new THREE.Group();
  coreGlow: THREE.Mesh[] = [];     // one per core (additive, over the core's top)
  mathGlow: THREE.Mesh[] = [];     // each core's math block
  coreBody: THREE.Group[] = [];
  constructor() {
    const edge = new THREE.LineBasicMaterial({ color: lin('bone', 0.55), transparent: true, opacity: 0.8 });
    const box = (w: number, h: number, d: number, mat: THREE.Material, x: number, y: number, z: number, parent: THREE.Object3D = this.group) => {
      const geo = new THREE.BoxGeometry(w, h, d);
      const m = new THREE.Mesh(geo, mat);
      m.position.set(x, y + h / 2, z);
      parent.add(m);
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edge);
      e.position.copy(m.position);
      parent.add(e);
      return m;
    };
    const sub = new THREE.MeshStandardMaterial({ color: lin('graphite', 0.35), roughness: 0.75, metalness: 0.25 });
    const sil = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(0.012, 0.012, 0.014), roughness: 0.22, metalness: 0.7 });
    const blk = new THREE.MeshStandardMaterial({ color: lin('ink2', 1.2), roughness: 0.4, metalness: 0.5 });
    box(66, 1.6, 66, sub, 0, -2, 0);                                  // substrate (y -2 .. -0.4)
    // pad field around the die
    const pads = new THREE.InstancedMesh(new THREE.BoxGeometry(1.2, 0.2, 1.2), new THREE.MeshStandardMaterial({ color: lin('ash', 0.4), roughness: 0.4, metalness: 0.9 }), 22 * 4);
    const o = new THREE.Object3D();
    let n = 0;
    for (let s = 0; s < 4; s++) for (let i = 0; i < 22; i++) {
      const a = -29 + i * 2.75;
      const [x, z] = s === 0 ? [a, -30] : s === 1 ? [a, 30] : s === 2 ? [-30, a] : [30, a];
      o.position.set(x, -0.3, z); o.updateMatrix(); pads.setMatrixAt(n++, o.matrix);
    }
    this.group.add(pads);
    box(40, 0.8, 34, sil, 0, -0.4, 0);                                // the die (y -0.4 .. 0.4)
    // four cores, 2 × 2, with the shared cache strip between the rows
    const CW = 17, CD = 12.5, CH = 1.6;
    const cx = [-9.5, 9.5], cz = [-9.2, 9.2];
    for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) {
      const g = new THREE.Group();
      g.position.set(cx[c]!, 0.4, cz[r]!);
      box(CW, CH, CD, blk, 0, 0, 0, g);
      // inner blocks on the core's top: control (front), math (middle, big), cache (back)
      const top = CH + 0.02;
      const inner = (wd: number, dd: number, z: number, lum: number) => {
        const pg = new THREE.PlaneGeometry(wd, dd).rotateX(-Math.PI / 2);
        const m = new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ color: lin('bone', lum) }));
        m.position.set(0, top, z); g.add(m);
        const e = new THREE.LineSegments(new THREE.EdgesGeometry(pg), edge);
        e.position.copy(m.position); g.add(e);
      };
      inner(CW - 2, 2.2, -CD / 2 + 2.1, 0.03);
      inner(CW - 2, 5.4, 0.3, 0.02);
      inner(CW - 2, 2.2, CD / 2 - 2.1, 0.025);
      const mg = new THREE.Mesh(new THREE.PlaneGeometry(CW - 2.4, 5.0).rotateX(-Math.PI / 2), glowMat());
      mg.position.set(0, top + 0.03, 0.3); g.add(mg);
      const cg = new THREE.Mesh(new THREE.PlaneGeometry(CW + 1.2, CD + 1.2).rotateX(-Math.PI / 2), glowMat());
      cg.position.set(0, top + 0.05, 0); g.add(cg);
      this.group.add(g);
      this.coreBody.push(g); this.mathGlow.push(mg); this.coreGlow.push(cg);
    }
    box(38, 0.5, 3.2, blk, 0, 0.4, 0);                                // shared cache between the rows
    this.group.position.copy(CPU_POS);
  }
  /** Core k's presence (0..1: drops into place) and its glows. */
  setCore(k: number, present: number, glow: number, math: number) {
    const g = this.coreBody[k]!;
    g.visible = present > 0.001;
    g.position.y = 0.4 + (1 - present) * 14;
    g.scale.setScalar(0.85 + 0.15 * present);
    const cg = this.coreGlow[k]!.material as THREE.MeshBasicMaterial;
    cg.color.setRGB(LIN.signal[0] * glow, LIN.signal[1] * glow, LIN.signal[2] * glow);
    const mg = this.mathGlow[k]!.material as THREE.MeshBasicMaterial;
    mg.color.setRGB(LIN.ember[0] * math, LIN.ember[1] * math, LIN.ember[2] * math);
  }
}

export default class CpuGpu extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(34, W / H, 0.1, 20000);
  gpu!: H100;
  cpu = new CPUModel();
  rig = H100.rig();
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();
  famL = F.archivo(100, 700);
  famT = F.archivo(112, 800);
  famM = F.mono(400);
  famTitle = F.archivo(125, 900);
  kl: KLine[] = [];
  db: number[] = [];
  w: Record<string, Word> = {};
  chunks: number[] = [];          // grid times of the SM waves
  order: number[] = [];           // enabled SMs, centre-out
  perChunk: number[] = [];        // SMs lit by each wave
  T1 = 0;

  override async init() {
    const au = this.ctx.audio, ly = this.ctx.lyrics;
    this.T1 = this.ctx.end;
    this.gpu = new H100({ renderer: this.ctx.renderer });
    this.scene.add(this.gpu.group, this.cpu.group, this.rig);
    this.scene.environment = this.gpu.env;
    this.scene.background = lin('ink');
    const L1 = ly.get('A CPU thinks'), L2 = ly.get('A GPU just throws');
    const find = (l: Line, s: string) => {
      const x = l.words.find((y) => norm(y.w) === norm(s));
      if (!x) throw new Error(`cpugpu: word not found: ${s}`);
      return x;
    };
    for (const s of ['CPU', 'thinks', 'big', 'cores']) this.w[s] = find(L1, s);
    for (const s of ['GPU', 'throws', 'ten', 'more']) this.w[s] = find(L2, s);
    this.w.L1 = L1.words[0]!;
    this.w.L2 = L2.words[0]!;
    this.db = au.downbeats.filter((d) => d < 14).slice(0, 8);
    // waves: "throws", then every 8th from "ten" up to "more" (which lands the last one)
    const e8 = (au.timeOfBeat(1) - au.timeOfBeat(0)) / 2;
    const tTen = q(au, this.w.ten!.start), tMore = q(au, this.w.more!.start);
    const ts = [q(au, this.w.throws!.start)];
    for (let t = tTen; t < tMore - e8 * 0.5; t += e8) ts.push(t);
    ts.push(tMore);
    this.chunks = ts;
    // enabled SMs ordered centre-out, split over the waves
    const g = this.gpu;
    this.order = Array.from({ length: g.smCount }, (_, i) => i).filter((i) => g.smEnabled(i))
      .sort((a, b) => g.smPos(a).setY(0).length() - g.smPos(b).setY(0).length());
    const n = this.order.length;
    this.perChunk = ts.map((_, i) => Math.floor(n / ts.length) + (i < n % ts.length ? 1 : 0));
    // karaoke lines: one at a time, the next pre-shows dim 0.4 s early
    const lines = [L1, L2];
    lines.forEach((l, i) => {
      const words: KWord[] = [];
      const full = layout(l.text, this.famL, LYR.px);
      let cursor = 0;
      for (const wd of l.words) {
        const at = l.text.indexOf(wd.w, cursor);
        cursor = at + wd.w.length;
        words.push({ w: wd, x: LYR.x + (full.glyphs[at]?.x ?? 0), lay: layout(wd.w, this.famL, LYR.px) });
      }
      const next = lines[i + 1];
      this.kl.push({ l, words, show: l.start - 0.4, hide: next ? next.start - 0.4 : Infinity });
    });
  }

  /** Waves fired by t: count, SMs lit, and the latest wave's hit envelope. */
  waves(t: number) {
    const au = this.ctx.audio;
    let n = 0, lit = 0, k = 0;
    this.chunks.forEach((tc, i) => {
      if (snap(au, t, tc) <= 0) return;
      n = i + 1; lit += this.perChunk[i]!; k = hit(au, t, tc, 0.12);
    });
    return { n, lit, k };
  }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const au = this.ctx.audio, t = f.t, g = this.gpu, w = this.w;
    const S = (at: number, dur = 0.09) => snap(au, t, at, dur);
    const [d0, d1, , d3, d4, , d6] = this.db;
    const tCpu = q(au, w.L1!.start), tGpu = d6 ?? q(au, w.GPU!.start);

    // ── the module rises out of the dark over the first two bars; the key light comes up with it ──
    const rise = ease.outCubic(clamp((t - (d0 ?? 0)) / ((d1 ?? 1.8) + 1.8 - (d0 ?? 0))));
    g.group.position.y = -26 * (1 - rise);
    const key = this.rig.children[0] as THREE.DirectionalLight, rim = this.rig.children[1] as THREE.DirectionalLight;
    key.intensity = 0.25 + 1.25 * smoothstep(0.5, d3 ?? 5.4, t);
    // grazing rim: lights the board edges and component sides, not a specular hot spot on the flat tops
    rim.position.set(90, 10, -140);
    rim.intensity = 1.8 + 1.0 * kickPush(au, t, 0.18);

    // ── SM activity: default music flicker until the GPU line, then the waves ──
    const wv = this.waves(t);
    if (t < this.chunks[0]! - 0.3) {
      g.setSMActivity(null);
    } else {
      const litSet = new Set(this.order.slice(0, wv.lit));
      const fresh = new Set(this.order.slice(wv.lit - (this.perChunk[wv.n - 1] ?? 0), wv.lit));
      const fi = Math.floor(t * 15);
      g.setSMActivity((i) => {
        if (!g.smEnabled(i)) return 0;
        if (!litSet.has(i)) return 0.04;
        const flick = ((i * 7 + fi) % 5) / 5;
        return fresh.has(i) ? 0.55 + 1.6 * wv.k : 0.5 + 0.25 * flick + 0.35 * f.a.kick;
      });
    }
    g.update(t, f.a);

    // data always flowing: pulses HBM → SM
    const items: PulseItem[] = [];
    for (let k = 0; k < 6; k++) {
      if (!g.hbm(k).active) continue;
      const path = g.pulsePath(k, this.order[(k * 17) % this.order.length]!);
      for (let n = 0; n < 4; n++) {
        const u = (drive(au, t, 0.22, 0.5) + n / 4 + k * 0.17) % 1;
        items.push({ curve: path, u, len: 0.04, gain: 1.8, width: 0.16 });
      }
    }
    g.pulses.set(items);

    // ── the CPU: the package drops in on "CPU", its cores on the stressed words ──
    const when = [w.thinks!.start, w.thinks!.start, w.big!.start, w.big!.start];
    const cpuOn = S(w.CPU!.start, 0.12);
    this.cpu.group.visible = t > tCpu - 0.6;
    this.cpu.group.position.y = CPU_POS.y - 20 * (1 - cpuOn);
    for (let k = 0; k < 4; k++) {
      const p = S(when[k]!, 0.09);
      const h = hit(au, t, when[k]!, 0.14);
      this.cpu.setCore(k, p, p * (0.18 + 1.4 * h + 0.25 * f.a.kick), p * (0.6 + 2.2 * h));
    }

    // ── camera: orbit with the drums, swing to both chips on "A CPU", lunge onto the die on "GPU" ──
    const cam = this.cam;
    const ang = -0.55 + drive(au, t, 0.035, 0.12);
    const orbitShot = () => {
      const r = lerp(215, 175, clamp(t / 7));
      const c = new THREE.Vector3(0, 2, 0);
      return { pos: new THREE.Vector3(c.x + Math.sin(ang) * r, 92 - 10 * clamp(t / 7), c.z + Math.cos(ang) * r), tgt: c };
    };
    const bothShot = () => {
      const a2 = -0.35 + drive(au, t, 0.03, 0.08, tCpu);
      const r = 205;
      return { pos: new THREE.Vector3(MID.x + Math.sin(a2) * r, 105, MID.z + Math.cos(a2) * r), tgt: MID.clone().setY(-4) };
    };
    const smTarget = g.smPos(this.order[4]!);
    const dieShot = () => {
      // lands above the die at 62 mm, then keeps diving toward the surface through the end
      const u = clamp((t - tGpu) / Math.max(0.5, this.T1 + 0.2 - tGpu));
      const dist = lerp(62, 13, ease.inQuad(u)) * (1 - 0.04 * kickPush(au, t, 0.12));
      const a3 = 0.55 + drive(au, t, 0.05, 0.1, tGpu);
      const el = lerp(0.95, 1.18, u);
      const tg = new THREE.Vector3(0, H100_Y.dieTop, 0).lerp(smTarget, u);
      return { pos: new THREE.Vector3(tg.x + Math.sin(a3) * Math.cos(el) * dist, tg.y + Math.sin(el) * dist, tg.z + Math.cos(a3) * Math.cos(el) * dist), tgt: tg };
    };
    let pos: THREE.Vector3, tgt: THREE.Vector3;
    const A = orbitShot();
    if (t < tCpu - 0.25) {
      pos = A.pos; tgt = A.tgt;
    } else if (t < tGpu - 0.5) {
      const B = bothShot();
      const k = beatEase(au, t, tCpu - 0.25, d4 ?? tCpu + 0.3, ease.inOutCubic);
      pos = A.pos.clone().lerp(B.pos, k); tgt = A.tgt.clone().lerp(B.tgt, k);
    } else {
      const B = bothShot(), C = dieShot();
      const k = beatEase(au, t, tGpu - 0.5, tGpu, ease.inOutExpo);
      pos = B.pos.clone().lerp(C.pos, k); tgt = B.tgt.clone().lerp(C.tgt, k);
    }
    cam.position.copy(pos);
    cam.up.set(0, 1, 0);
    cam.lookAt(tgt);
    H100.fitClip(cam, pos.distanceTo(tgt));

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    comp.draw(renderer, this.ms.texture, out);

    // ── screen-space overlays ──
    const c = this.ui.ctx;
    this.ui.clear();
    this.drawCpuLabel(c, t, S);
    this.drawGpuLabels(c, t, S, wv);
    this.drawLyrics(c, t);
    this.drawTitle(c, t, S, d1, d3);
    comp.draw(renderer, this.ui.upload(), out);
    return { bloom: 0.6, bloomThreshold: 0.85 };
  }

  /** Screen position of a world point (null behind the camera). */
  proj(p: THREE.Vector3): { x: number; y: number } | null {
    const v = p.clone().project(this.cam);
    if (v.z > 1) return null;
    return { x: (v.x * 0.5 + 0.5) * W, y: (-v.y * 0.5 + 0.5) * H };
  }

  drawCpuLabel(c: CanvasRenderingContext2D, t: number, S: (at: number, d?: number) => number) {
    const w = this.w;
    const named = S(w.CPU!.start);
    const fade = 1 - clamp((t - (w.L2!.start - 0.1)) / 0.2);
    if (named <= 0 || fade <= 0) return;
    const p = this.proj(this.cpu.group.localToWorld(new THREE.Vector3(-33, 4, -33)));
    if (!p) return;
    c.textAlign = 'left'; c.textBaseline = 'alphabetic';
    c.fillStyle = rgba('bone', named * fade);
    c.font = font(this.famT, 64);
    c.fillText('CPU', p.x, p.y - 18);
    const cap = S(w.cores!.start);
    if (cap > 0) {
      c.font = font(this.famM, 28);
      c.fillStyle = rgba('bone', 0.9 * cap * fade);
      c.fillText('a few big cores ·', p.x, p.y + 22);
      c.fillStyle = rgba('signal', cap * fade);
      c.fillText('4', p.x + c.measureText('a few big cores · ').width, p.y + 22);
    }
  }

  drawGpuLabels(c: CanvasRenderingContext2D, t: number, S: (at: number, d?: number) => number, wv: { n: number; lit: number; k: number }) {
    const w = this.w;
    const named = S(w.GPU!.start);
    if (named <= 0) return;
    const X = W - 150, Y0 = 170;
    // an ink bed under the block: the lit SM tiles behind it are as bright as the text
    const bed = c.createRadialGradient(X - 220, Y0 + 90, 40, X - 220, Y0 + 90, 640);
    bed.addColorStop(0, rgba('ink', 0.88 * named));
    bed.addColorStop(0.5, rgba('ink', 0.6 * named));
    bed.addColorStop(1, rgba('ink', 0));
    c.fillStyle = bed;
    c.fillRect(0, 0, W, H);
    c.textAlign = 'right'; c.textBaseline = 'alphabetic';
    c.font = font(this.famT, 64);
    c.fillStyle = rgba('bone', named);
    c.fillText('GPU', X, Y0);
    if (wv.n > 0) {
      const val = Math.round((10000 * wv.n) / this.chunks.length / 100) * 100;
      c.font = font(this.famT, 112);
      c.fillStyle = wv.k > 0.05 ? mix('signal', 'bone', 1 - wv.k * 0.8) : rgba('bone', 0.96);
      c.fillText(val.toLocaleString('en-US'), X, Y0 + 124);
    }
    const cap = S(w.more!.start);
    if (wv.n > 0 && cap < 1) {
      c.font = font(this.famM, 24);
      c.fillStyle = rgba('ash', 0.85 * (1 - cap));
      c.fillText('small cores', X, Y0 + 162);
    }
    if (cap > 0) {
      c.font = font(this.famM, 28);
      c.fillStyle = rgba('bone', 0.92 * cap);
      c.fillText('thousands of small cores', X, Y0 + 170);
      c.font = font(this.famM, 20);
      c.fillStyle = rgba('ash', 0.8 * cap);
      c.fillText('an H100 has 132 SMs × 128 = 16,896 · H100 whitepaper', X, Y0 + 206);
    }
  }

  // ── the title card: complete on the first frame; subtitle on downbeat 2; snaps away on downbeat 4 ──
  drawTitle(c: CanvasRenderingContext2D, t: number, S: (at: number, d?: number) => number, d1?: number, d3?: number) {
    const gone = S(d3 ?? 5.44);
    if (gone >= 1) return;
    const a = 1 - gone;
    c.save();
    // a soft dark bed behind the type so it reads over the module
    const gr = c.createRadialGradient(W / 2, H / 2 + 30, 60, W / 2, H / 2 + 30, 820);
    gr.addColorStop(0, rgba('ink', 0.78 * a));
    gr.addColorStop(1, rgba('ink', 0));
    c.fillStyle = gr;
    c.fillRect(0, 0, W, H);
    c.textAlign = 'center';
    c.textBaseline = 'alphabetic';
    c.font = font(this.famTitle, 132);
    c.fillStyle = rgba('bone', a);
    c.fillText('CHASING THE', W / 2, H / 2 - 40);
    c.fillStyle = rgba('signal', a);
    c.fillText('ROOFLINE', W / 2, H / 2 + 100);
    const sub = S(d1 ?? 1.83);
    if (sub > 0) {
      c.font = font(this.famM, 28);
      c.fillStyle = rgba('ash', 0.95 * a * sub);
      c.fillText('a song about CUDA  ·  from the book CUDA for Deep Learning', W / 2, H / 2 + 180);
    }
    c.restore();
  }

  // ── karaoke (one line, fixed place) ──
  drawLyrics(c: CanvasRenderingContext2D, t: number) {
    const kl = this.kl.find((k) => t >= k.show && t < k.hide);
    if (!kl) return;
    const pre = clamp((t - kl.show) / 0.12);
    // a soft shade under the line so it reads over the 3D
    const shade = c.createLinearGradient(0, LYR.base - 150, 0, H);
    shade.addColorStop(0, rgba('ink', 0));
    shade.addColorStop(1, rgba('ink', 0.62 * pre));
    c.fillStyle = shade;
    c.fillRect(0, LYR.base - 150, W, H - LYR.base + 150);
    c.textBaseline = 'alphabetic';
    c.textAlign = 'left';
    c.font = font(this.famL, LYR.px);
    const last = kl.words[kl.words.length - 1]!.w;
    for (const kw of kl.words) {
      const p = kw.w === last && kw.w.end > this.T1 - 0.05
        ? clamp((t - kw.w.start) / Math.max(0.05, this.T1 - 0.1 - kw.w.start))
        : Lyrics.wordProgress(kw.w, t);
      const settle = clamp((t - kw.w.end) / 0.25);
      const n = kw.lay.glyphs.length;
      for (const gl of kw.lay.glyphs) {
        const gx = kw.x + gl.x;
        const gp = clamp(p * n - gl.i);
        if (gp < 1) {
          c.fillStyle = rgba('ash', 0.36 * pre);
          c.fillText(gl.ch, gx, LYR.base);
        }
        if (gp > 0) {
          c.save();
          if (gp < 1) { c.beginPath(); c.rect(gx - 4, LYR.base - LYR.px, gl.w * gp + 4, LYR.px * 1.4); c.clip(); }
          c.fillStyle = settle < 1 ? mix('signal', 'bone', settle) : rgba('bone', 1);
          c.fillText(gl.ch, gx, LYR.base);
          c.restore();
        }
      }
    }
  }
}

function mix(a: string, b: string, k: number, alpha = 1) {
  const pa = rgba(a).match(/\d+/g)!.map(Number), pb = rgba(b).match(/\d+/g)!.map(Number);
  return `rgba(${[0, 1, 2].map((i) => Math.round(pa[i]! + (pb[i]! - pa[i]!) * clamp(k))).join(',')},${alpha})`;
}
