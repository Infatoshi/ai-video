// ONE IDEA: when one GPU can't hold the job, split it into 8 pieces on 8 GPUs, then add their partial
// answers together (all-reduce).
// v5 scene `multi` (CH10), 3D on the shared models: "When one GPU is not enough, / Split it eight
// ways, then sum it up". The window runs into the song's stop: from ~155.9 s the band drops out and
// "then sum it up" is sung over near-silence until the final chorus hits at 159.28 s, so the
// all-reduce traffic and the ride down one NVLink trace are the riser into that drop.
//  1. 150.5-153.4 (one H100, `_h100.ts`): the camera circles close over the module while the job, a
//     slab of 8 slices of work tiles, hangs over the die and swells on the beats. "not": it snaps to
//     its real size, far wider than the module, the camera kicks back, every SM saturates.
//  2. "enough" (the downbeat): hard cut to the whole DGX H100 (`_dgx.ts`), lit hard; the camera pulls
//     back off its fan wall to a 3/4 view with the job (at server scale) hanging over the box.
//  3. "Split it eight ways": the teardown on the beats (lid on "Split", GPU tray on "eight", heatsinks
//     on "ways") while the job breaks into its 8 slices, each riding over one GPU; then one slice per
//     16th drops onto its GPU, whose die lights.
//  4. "then sum it up": all-reduce over the four NVSwitches from "sum", full on "up" (the downbeat),
//     where the camera drops onto one trace and rides it through the silence, the packets flickering
//     faster and faster, to the brightest frame on the cut (the timeline's 1.5 s dive into roof4
//     zooms into frame centre, which is where the trace runs).
// Camera always moving (scenes/_cam.ts); events snap on the beat grid (scenes/_lock.ts).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font, layout, type TextLayout } from '../engine/type';
import type { Line, Word } from '../engine/lyrics';
import { clamp, ease, hash, lerp } from '../engine/util';
import { hit, q, snap } from './_lock';
import { cumEnv } from './_cam';
import { H100, glowMat } from './_h100';
import { DGX } from './_dgx';

const col = (k: keyof typeof LIN, m = 1) => new THREE.Color().setRGB(LIN[k][0] * m, LIN[k][1] * m, LIN[k][2] * m, THREE.LinearSRGBColorSpace);
const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);

// ------------------------------------------------------------------ karaoke (per-word wipe, screen space)
const LYR = { x: 120, y: 196, size: 92 };
const layCache = new Map<string, TextLayout>();
function lay(text: string, fam: string, size: number) {
  const k = `${text}|${fam}|${size}`;
  let l = layCache.get(k);
  if (!l) { l = layout(text, fam, size); layCache.set(k, l); }
  return l;
}
function xAt(l: TextLayout, n: number) {
  if (n <= 0) return 0;
  const g = l.glyphs;
  if (n >= g.length) return l.width;
  const i = Math.floor(n), f = n - i;
  return lerp(g[i]!.x, i + 1 < g.length ? g[i + 1]!.x : l.width, f);
}
/** Characters of `text` sung by t, following syllable times (G / P / U) when they match. */
function sungChars(w: Word, text: string, t: number): number {
  if (t <= w.start) return 0;
  if (t >= w.end) return text.length;
  if (w.syl && w.syl.length === text.replace(/[^A-Za-z]/g, '').length && w.syl.length > 1) {
    let n = 0;
    for (const [a, b] of w.syl) {
      if (t < a) break;
      n += t < b ? (t - a) / Math.max(1e-3, b - a) : 1;
    }
    return n;
  }
  return (text.length * (t - w.start)) / Math.max(1e-3, w.end - w.start);
}
function karaoke(c: CanvasRenderingContext2D, line: Line, t: number, alpha: number) {
  const fam = F.archivo(100, 800), size = LYR.size;
  const texts = line.words.map((w) => w.w);
  const text = texts.join(' ');
  const L = lay(text, fam, size);
  let off = 0, doneN = 0, sungN = 0;
  for (let i = 0; i < line.words.length; i++) {
    const w = line.words[i]!, s = texts[i]!;
    if (t >= w.end) { doneN = off + s.length; sungN = doneN; }
    else if (t > w.start) { doneN = off; sungN = off + sungChars(w, s, t); break; }
    else break;
    off += s.length + 1;
  }
  const { x, y } = LYR;
  c.save();
  c.globalAlpha = alpha;
  c.font = font(fam, size);
  c.letterSpacing = '0px';
  c.textBaseline = 'alphabetic';
  c.textAlign = 'left';
  // a soft ink shadow so the line reads over bright 3D (no outline, no box)
  c.shadowColor = 'rgba(10,10,11,0.9)';
  c.shadowBlur = 28;
  c.fillStyle = rgba('bone', 0.34);
  c.fillText(text, x, y);
  c.shadowBlur = 0;
  const top = y - size * 1.05, h = size * 1.4;
  const xd = x + xAt(L, doneN), xs = x + xAt(L, sungN);
  if (doneN > 0) { c.save(); c.beginPath(); c.rect(x - 20, top, xd - x + 20, h); c.clip(); c.fillStyle = rgba('bone', 1); c.fillText(text, x, y); c.restore(); }
  if (sungN > doneN) { c.save(); c.beginPath(); c.rect(xd, top, xs - xd, h); c.clip(); c.fillStyle = rgba('signal', 1); c.fillText(text, x, y); c.restore(); }
  c.restore();
}

// ------------------------------------------------------------------ the job: 8 slices of work tiles
const NS = 8, NZ = 4, NY = 3, PER = NZ * NY;
class Job {
  mesh: THREE.InstancedMesh;
  mat: THREE.MeshStandardMaterial;
  private m4 = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private v = new THREE.Vector3();
  private s = new THREE.Vector3();
  private c = new THREE.Color();
  constructor() {
    this.mat = new THREE.MeshStandardMaterial({ color: col('ember', 0.7), emissive: col('signal', 1), emissiveIntensity: 1, roughness: 0.4, metalness: 0.05 });
    this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.mat, NS * PER);
    this.mesh.frustumCulled = false;
    for (let i = 0; i < NS * PER; i++) this.mesh.setColorAt(i, this.c.setRGB(1, 1, 1));
  }
  /**
   * Lay out the tiles: slice k centred on `center(k)` (world units), tile edge `cube`, pitch
   * `cube * 1.18`, scaled by `sk(k)` (0 hides a slice); `heat(k, i)` scales each tile's lit colour.
   */
  set(center: (k: number) => THREE.Vector3, cube: number, sk: (k: number) => number, heat: (k: number, i: number) => number) {
    const p = cube * 1.18;
    for (let k = 0; k < NS; k++) {
      const c0 = center(k), s = sk(k);
      for (let i = 0; i < PER; i++) {
        const iz = i % NZ, iy = Math.floor(i / NZ);
        this.v.set(c0.x, c0.y + (iy - (NY - 1) / 2) * p * s, c0.z + (iz - (NZ - 1) / 2) * p * s);
        const e = s > 0.001 ? cube * s : 0.0001;
        this.s.set(e, e, e);
        this.m4.compose(this.v, this.q, this.s);
        this.mesh.setMatrixAt(k * PER + i, this.m4);
        const h = heat(k, i);
        this.mesh.setColorAt(k * PER + i, this.c.setRGB(0.5 + h, 0.45 + 0.7 * h, 0.4 + 0.45 * h));
      }
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
}

export default class Multi extends Scene {
  ui = new Layer2D();
  ms = makeRT(W, H, { samples: 4 });
  cam = new THREE.PerspectiveCamera(34, W / H, 0.5, 9000);
  famM = F.mono(500);
  famMB = F.mono(600);
  // act 1: one H100
  hScene = new THREE.Scene();
  gpu!: H100;
  jobH = new Job();
  // acts 2-4: the DGX H100
  dgx = new DGX();
  jobD = new Job();
  dies: THREE.Mesh[] = [];
  beams: THREE.Mesh[] = [];             // the light each slice rides down onto its die
  lights: [THREE.DirectionalLight, number][] = [];   // our added lights and their full intensity
  trayC = new THREE.Vector3();          // centre of the 8 GPUs with the tray fully out (world mm)

  private L1!: Line; private L2!: Line;
  private tBar1 = 0;                    // first downbeat in the window (the push-in lands)
  private tGPU = 0; private tIs = 0; private tNot = 0; private tEnough = 0;
  private tSplit = 0; private tEight = 0; private tWays = 0; private tLand: number[] = [];
  private tSum = 0; private tUp = 0; private tFade = 0; private tDrop = 0;

  override async init() {
    const { lyrics, audio: au, renderer } = this.ctx;
    this.L1 = lyrics.get('When one GPU');
    this.L2 = lyrics.get('Split it eight ways');
    const w1 = this.L1.words, w2 = this.L2.words;
    this.tBar1 = au.downbeats.find((d) => d > this.ctx.start + 0.4) ?? w1[2]!.start;
    this.tGPU = q(au, w1[2]!.start);
    this.tIs = q(au, w1[3]!.start);
    this.tNot = q(au, w1[4]!.start);
    this.tEnough = q(au, w1[5]!.start, 1);  // the downbeat under "enough": the cut to the server
    this.tSplit = q(au, w2[0]!.start);
    this.tEight = q(au, w2[2]!.start);
    this.tWays = q(au, w2[3]!.start);
    const b0 = Math.ceil(au.beatAt(this.tWays + 0.05));
    this.tLand = Array.from({ length: NS }, (_, k) => au.timeOfBeat(b0 + k / 4)); // one slice per 16th
    this.tSum = q(au, w2[5]!.start);
    this.tUp = q(au, w2[7]!.start);
    this.tFade = au.downbeats.find((d) => d > this.tUp + 0.5) ?? this.ctx.end;
    // the final chorus hits on the beat at/before its first word (the timeline cuts there too)
    const next = lyrics.get("I'm chasing", 3).words[0]!.start;
    this.tDrop = au.timeOfBeat(Math.floor(au.beatAt(next + 0.02)));

    // act 1: the H100 module on its own
    this.gpu = new H100({ renderer });
    this.hScene.add(this.gpu.group, H100.rig(), this.jobH.mesh);
    this.hScene.environment = this.gpu.env;
    this.hScene.background = col('ink');

    // acts 2-4: the server, lit hard so the closed box reads
    const d = this.dgx;
    d.init(renderer);
    d.scene.environmentIntensity = 1.1;
    const key = new THREE.DirectionalLight(col('bone', 1), 5);
    key.position.set(-700, 1200, 1100);
    const rim = new THREE.DirectionalLight(col('ember', 1), 7);
    rim.position.set(900, 400, -1000);
    const rim2 = new THREE.DirectionalLight(col('signal', 1), 3);
    rim2.position.set(-1000, 200, -600);
    const face = new THREE.DirectionalLight(col('bone', 1), 2.2);
    face.position.set(300, 150, 1500);
    d.scene.add(key, rim, rim2, face, this.jobD.mesh);
    this.lights = [key, rim, rim2, face].map((l) => [l, l.intensity]);
    // a glowing plate on each GPU die (hidden under its heatsink until the teardown)
    for (let i = 0; i < 8; i++) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(26, 31).rotateX(-Math.PI / 2), glowMat(0x000000));
      m.position.set(0, 12.05, 0);
      d.gpu(i).add(m);
      this.dies.push(m);
      const b = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), glowMat(0x000000));
      b.frustumCulled = false;
      d.scene.add(b);
      this.beams.push(b);
    }
    d.setExplode(1); d.update(0);
    for (let i = 0; i < 8; i++) this.trayC.add(d.worldPos(d.gpu(i)).multiplyScalar(1 / 8));
    d.setExplode(0); d.update(0);
  }

  /** Orbit phase: steady plus a surge on the drums (cumulative envelope). */
  private orbit(t: number, base: number, gain: number) {
    const au = this.ctx.audio;
    return base * (t - this.ctx.start) + gain * (cumEnv(au, 'drums', t) - cumEnv(au, 'drums', this.ctx.start));
  }

  /** Shift the image down by `oy` px so the subject sits under the lyric band (0 = centred). */
  private offset(oy: number) {
    if (Math.abs(oy) < 0.5) { if (this.cam.view?.enabled) this.cam.clearViewOffset(); }
    else this.cam.setViewOffset(W, H, 0, -oy, W, H);
  }

  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp, audio: au } = this.ctx;
    const t = f.t;
    const act1 = t < this.tEnough;
    const post = act1 ? this.renderH100(t, f) : this.renderDGX(t);
    comp.draw(renderer, this.ms.texture, out, { mode: 'replace' });

    // ---------------------------------------------------------------- screen-space type
    const U = this.ui; U.clear();
    const c = U.ctx;
    const pre = this.L2.start - 0.35;
    const lyrA = 1 - snap(au, t, this.tFade, 0.12);
    if (t < pre) karaoke(c, this.L1, t, 1);
    else if (lyrA > 0) karaoke(c, this.L2, t, (t < this.L2.start ? 0.6 : 1) * lyrA);
    const label = (s: string, a: number, y: number, em: boolean) => {
      if (a <= 0) return;
      c.save();
      c.globalAlpha = a;
      c.font = font(em ? this.famMB : this.famM, em ? 36 : 30);
      c.textBaseline = 'alphabetic'; c.textAlign = 'left'; c.letterSpacing = '0px';
      c.shadowColor = 'rgba(10,10,11,0.95)'; c.shadowBlur = 20;
      c.fillStyle = rgba(em ? 'signal' : 'bone', 1);
      c.fillText(s, LYR.x, y);
      c.restore();
    };
    const sNot = snap(au, t, this.tNot), sSplit = snap(au, t, this.tSplit);
    const sLand = snap(au, t, this.tLand[0]!, 0.09, 4), sSum = snap(au, t, this.tSum), sUp = snap(au, t, this.tUp);
    if (act1) {
      label('one GPU', 1 - sNot, 990, true);
      label("the job doesn't fit", sNot, 990, true);
    } else if (t < this.tSplit) {
      label('one server: 8 GPUs', 1, 990, true);
    } else if (t < this.tSum) {
      label(sLand > 0 ? '8 pieces → 8 GPUs' : 'take it apart', sSplit, 990, true);
    } else {
      label('add the pieces = all-reduce', sSum * lyrA, 990, true);
      label('NVLink: 900 GB/s per GPU (CH10)', sUp * lyrA, 1040, false);
    }
    comp.draw(renderer, U.upload(), out);
    return post;
  }

  // ---------------------------------------------------------------- act 1: one H100, overflowing
  private renderH100(t: number, f: Frame): PostOverrides {
    const { renderer, audio: au } = this.ctx;
    const g = this.gpu;
    const sNot = snap(au, t, this.tNot, 0.1);
    const kNot = hit(au, t, this.tNot, 0.14);
    // straining: the job swells a notch on each beat from "GPU" to "is", the SMs heat with it
    const steps = [this.tGPU, q(au, this.tGPU + 0.45, 1), this.tIs].map((x) => snap(au, t, x, 0.12));
    const strain = (steps[0]! + steps[1]! + steps[2]!) / 3;
    g.setSMActivity((i) => {
      const busy = 0.25 + 0.5 * strain * hash(i, Math.floor(t * 12));
      const hot = 0.8 + 0.2 * hash(i * 7, Math.floor(t * 30));
      return lerp(busy, hot, sNot);
    });
    g.setHBMActivity(() => 0.3 + 0.4 * strain + 0.3 * sNot);
    g.update(t, f.a);
    // camera: circle the module (always moving, surging with the drums), push in to land on the
    // first downbeat, kick back and up on "not" to show the job overflowing
    const pushIn = ease.inOutCubic(clamp((t - this.ctx.start) / Math.max(0.2, this.tBar1 - this.ctx.start)));
    const kickBack = ease.outExpo(clamp((t - this.tNot) / 0.3));
    const dist = lerp(lerp(230, 150, pushIn), 520, kickBack) - 8 * au.hit('kick', t, 0.12);
    const az = -1.05 + this.orbit(t, 0.16, 0.5) + 0.35 * kickBack;
    const el = lerp(lerp(0.3, 0.52, pushIn), 0.5, kickBack);
    const tgt = V(0, lerp(10, 58, kickBack), 0);
    this.cam.position.set(tgt.x + Math.sin(az) * Math.cos(el) * dist, tgt.y + Math.sin(el) * dist, tgt.z + Math.cos(az) * Math.cos(el) * dist);
    this.cam.up.set(0, 1, 0);
    this.cam.lookAt(tgt);
    this.offset(90);
    H100.fitClip(this.cam, dist);
    // the job: hangs over the die at die size, swells, then snaps to its real size on "not"
    // (1.6x the module's width): it does not fit
    const cube = lerp(3 * (1 + 0.6 * strain), 22, sNot);
    const shake = (0.3 + 1.2 * strain) * (1 - sNot) + 1.5 * kNot;
    const y = lerp(17 + 4 * strain, 80, sNot) + 1.5 * Math.sin(t * 3.1);
    this.jobH.set(
      (k) => V((k - (NS - 1) / 2) * cube * 1.18 + shake * (hash(k, Math.floor(t * 40)) - 0.5), y, 0),
      cube,
      () => 1,
      (k, i) => 0.15 + 0.3 * hash(k * 13 + i, Math.floor(t * 8)) + 0.3 * strain * (1 - sNot) + 0.6 * kNot,
    );
    this.jobH.mat.emissiveIntensity = lerp(0.9 + 0.5 * strain, 0.28, sNot) + 0.5 * kNot;
    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(col('ink'), 1);
    renderer.clear(true, true, true);
    renderer.render(this.hScene, this.cam);
    return { bloom: 0.8, bloomThreshold: 0.9, zoom: 1 + 0.03 * kNot };
  }

  // ---------------------------------------------------------------- acts 2-4: the DGX H100
  private renderDGX(t: number): PostOverrides {
    const { renderer, audio: au } = this.ctx;
    const d = this.dgx;
    // teardown on the beats: lid on "Split", tray on "eight", heatsinks on "ways"
    const ex = 0.25 * ease.inOutCubic(clamp((t - this.tSplit) / 0.36))
      + 0.31 * ease.inOutCubic(clamp((t - this.tEight) / 0.4))
      + 0.44 * ease.inOutCubic(clamp((t - this.tWays) / 0.5));
    d.setExplode(ex);
    // traffic: idle hum; all-reduce from "sum", full on "up", then faster and faster into the drop
    const sSum = snap(au, t, this.tSum), sUp = snap(au, t, this.tUp, 0.12);
    const build = clamp((t - this.tUp) / Math.max(0.5, this.tDrop - this.tUp));
    const bi = ease.inCubic(build);
    if (sSum <= 0) d.setTraffic({ pattern: 'idle', intensity: 0.25, rate: 1 });
    else d.setTraffic({ pattern: 'allreduce', intensity: lerp(0.45, 1, sUp), rate: lerp(1.8, 4, sUp) + 12 * bi });
    d.update(t);
    // the GPUs light as their slice lands
    this.dies.forEach((m, i) => {
      const on = snap(au, t, this.tLand[i]!, 0.05, 4);
      const h = hit(au, t, this.tLand[i]!, 0.1, 4);
      const hum = 0.75 + 0.25 * hash(i, Math.floor(t * 24));
      const k = on * (1.1 * hum + 2.5 * h + 0.6 * sSum + 1.2 * bi);
      (m.material as THREE.MeshBasicMaterial).color.setRGB(LIN.signal[0] * k, LIN.signal[1] * k, LIN.signal[2] * k);
    });
    // the job at server scale: hangs over the box; on "Split" it breaks into 8 slices, which ride over
    // their GPUs as the tray comes out ("eight"), then drop one per 16th onto the dies
    const cube = 20, pitch = cube * 1.18;
    const sSplit = snap(au, t, this.tSplit, 0.12);
    const kSplit = hit(au, t, this.tSplit, 0.12);
    const bob = 8 * Math.sin(t * 2.3);
    // (it drops in from above on the beat after the cut, so the close shot on the fans stays clean)
    const fall = 1 - ease.outBack(clamp((t - q(au, this.tEnough + 0.3, 1)) / 0.3));
    const home = (k: number) => V((k - (NS - 1) / 2) * (pitch + 46 * sSplit), 450 + bob + 700 * fall, -40);
    // over each GPU, clear of its heatsink once that lifts (128 mm tall, raised 190 mm)
    const over = (k: number) => d.worldPos(d.gpu(k)).add(V(0, 400 + 0.5 * bob, 0));
    const toOver = ease.inOutCubic(clamp((t - this.tEight) / 0.42));
    const drop = (k: number) => ease.inCubic(clamp((t - (this.tLand[k]! - 0.15)) / 0.15));
    this.jobD.set(
      (k) => home(k).lerp(over(k), toOver).lerp(d.worldPos(d.gpu(k)).add(V(0, 14, 0)), drop(k)),
      cube,
      (k) => (t >= this.tLand[k]! ? 0 : lerp(1, 0.35, drop(k))),
      (k, i) => 0.2 + 0.3 * hash(k * 13 + i, Math.floor(t * 8)) + 0.8 * kSplit,
    );
    this.jobD.mat.emissiveIntensity = 0.55 + 0.6 * kSplit;
    // each slice falls down a beam of light onto its die (the beam hides the pass through the fins)
    this.beams.forEach((b, k) => {
      const a = t < this.tLand[k]! - 0.15 ? 0 : t < this.tLand[k]! ? 0.6 + 0.4 * drop(k) : hit(au, t, this.tLand[k]!, 0.05, 4);
      b.visible = a > 0.04;
      if (!b.visible) return;
      const g0 = d.worldPos(d.gpu(k)), top = over(k).y;
      b.position.set(g0.x, (g0.y + 12 + top) / 2, g0.z);
      b.scale.set(8 + 10 * a, top - g0.y - 12, 8 + 10 * a);
      const e = 2.2 * a;
      (b.material as THREE.MeshBasicMaterial).color.setRGB(LIN.signal[0] * e, LIN.signal[1] * e, LIN.signal[2] * e);
    });

    // ---------------------------------------------------------------- camera
    const tc = this.trayC;
    type K = [number, THREE.Vector3, THREE.Vector3];
    // the cut: tight on the fan wall; pull back to 3/4 with the job overhead (lands on "Split");
    // rise over the tray as it slides out (lands on the beat under "eight"); a high look down the
    // tray for "sum"; "up" hands over to the trace ride
    const keys: K[] = [
      [this.tEnough, V(150, 300, 760), V(-10, 250, 440)],
      [this.tSplit, V(820, 780, 1180), V(0, 330, 40)],
      [q(au, this.tEight, 1), V(tc.x + 420, tc.y + 660, tc.z + 1120), V(tc.x, tc.y + 200, tc.z - 40)],
      [this.tSum, V(tc.x + 250, tc.y + 560, tc.z + 900), V(tc.x, tc.y + 160, tc.z - 60)],
      [this.tUp, V(-380, 760, 1050), V(0, 250, 200)],
    ];
    let i = 0;
    while (i < keys.length - 2 && keys[i + 1]![0] <= t) i++;
    const [t0, p0, l0] = keys[i]!, [t1, p1, l1] = keys[i + 1]!;
    const u = clamp((t - t0) / Math.max(0.2, t1 - t0));
    const k = i === 0 ? ease.outQuad(u) : ease.inOutCubic(u);
    const P = p0.clone().lerp(p1, k), L = l0.clone().lerp(l1, k);
    // always moving: a slow swing around the node's vertical axis that surges with the drums
    const swing = 0.16 * Math.sin(this.orbit(t, 0.35, 0.9));
    P.sub(L).applyAxisAngle(V(0, 1, 0), swing).add(L);
    P.y += 30 * Math.sin(t * 1.3);
    // the ride: from "up" the camera swings over one trace, drops straight down onto it (just in front
    // of GPU 5's lifted heatsink, so it never passes through a part) and rides it, accelerating into
    // the drop
    const wIn = ease.inOutCubic(clamp((t - this.tUp) / 0.4));
    const wDn = ease.inOutCubic(clamp((t - this.tUp - 0.2) / 0.55));
    let near = 2;
    if (wIn > 0) {
      const curve = d.link(5, 1, 1);
      const uu = 0.06 + 0.86 * ease.inCubic(clamp((t - this.tUp) / Math.max(0.5, this.tDrop + 0.75 - this.tUp)));
      const h = lerp(24, 13, bi);
      const cp = curve.getPoint(Math.max(0, uu - 0.07)).add(V(0, h, 0));
      const cl = curve.getPoint(Math.min(1, uu + 0.08));
      P.lerp(cp.clone().add(V(0, 320, 20)), wIn).lerp(cp, wDn);
      L.lerp(cl, wIn);
      near = lerp(2, 0.2, wDn);
    }
    // on the trace the board goes dark so the lanes carry the frame
    for (const [l, i0] of this.lights) l.intensity = i0 * (1 - 0.85 * wIn);
    d.scene.environmentIntensity = lerp(1.1, 0.25, wIn);
    this.cam.position.copy(P);
    this.cam.up.set(Math.sin(0.35 * bi * Math.sin(t * 1.7)), 1, 0).normalize();
    this.cam.lookAt(L);
    this.cam.near = near; this.cam.far = 9000;
    this.offset(120 * (1 - wIn));
    this.cam.updateProjectionMatrix();
    d.render(renderer, this.ms, this.cam);
    // punch on the cut, the split and each landing; the ride brightens into the drop
    const hitE = hit(au, t, this.tEnough, 0.1);
    let hitL = 0;
    for (const tl of this.tLand) hitL = Math.max(hitL, hit(au, t, tl, 0.07, 4));
    return {
      bloom: 0.85 + 0.25 * hitL - 0.25 * wIn + 0.1 * bi, bloomThreshold: 0.88 + 0.04 * wIn,
      exposure: 1.05 + 0.3 * hitE + 0.12 * hitL - 0.1 * wIn + 0.1 * bi,
      zoom: 1 + 0.05 * hitE + 0.02 * kSplit,
      vignette: 0.35 + 0.15 * bi,
    };
  }
}
