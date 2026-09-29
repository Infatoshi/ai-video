// ffn (verse 2, lines 5-8): "Then the feed-forward fires where the facts are kept, / Fourteen thousand
// neurons every single step, / Attention, feed-forward, that's one floor of the tower, / Stack it
// thirty-two high, that's where it gets its power". A new shot every bar, two per line:
//  line 1: the 14,336 feed-forward neurons of floor 16 as a city of bars (112 x 128), heights = the REAL
//          |activations| of the step in flight, the ones above 10% of max in pink; a low flight, then
//          from above with the strongest neurons tagged
//  line 2: floors 1, 16 and 32 side by side with their real active counts; then one city flipping
//          through all 15 steps of the answer on the 16ths
//  line 3: one floor of the tower close up (ATTENTION ring, FEED-FORWARD block); then that floor pulled
//          out of the tower
//  line 4: the tower built two floors per 16th, each floor printing its REAL logit-lens guess, then the
//          top floors close as the guess sharpens into the token the model picks on "power"
// Every number is llm.json (ff activations and active counts, logit lens, attention, config) or arithmetic.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease } from '../engine/util';
import { showTok } from '../engine/llm';
import { Wash, tokenRain, specks, makeCam, drive, shot, snapIn, portrait, perFrame, inkMat, outlineInstanced, type Shot } from './_kit';
import { Tower, riderTile, SLAB_W, FLOOR_H } from './_tower';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
/** City grid: 112 x 128 = 14,336 neurons. */
const GX = 112, GZ = 128, PITCH = 0.1, FOOT = 0.074, HS = 6.5;
const CITY_W = GX * PITCH, CITY_D = GZ * PITCH;
const FF_LAYERS = [0, 15, 31] as const;
const comma = (n: number) => n.toLocaleString('en-US');

/** One layer's 14,336 neurons: a blue carpet of bars, the active ones (> 10% of max) a second, outlined mesh. */
class City {
  group = new THREE.Group();
  base: THREE.InstancedMesh;
  hot: THREE.InstancedMesh;
  hotLine: THREE.InstancedMesh;
  hotPink = inkMat({ ink: 'pink', lit: 0.5, shade: 1 });
  hotBlue = inkMat({ ink: 'blue', lit: 0.45, shade: 1 });
  key = '';
  /** Active neuron indices, strongest first. */
  top: number[] = [];
  constructor() {
    const g = new THREE.BoxGeometry(FOOT, 1, FOOT);
    g.translate(0, 0.5, 0);
    this.base = new THREE.InstancedMesh(g, inkMat({ ink: 'blue', lit: 0.45, shade: 1 }), GX * GZ);
    this.hot = new THREE.InstancedMesh(g, this.hotPink, 2048);
    this.hotLine = outlineInstanced(this.hot, 1.3);
    for (const m of [this.base, this.hot]) { m.frustumCulled = false; this.group.add(m); }
    // ground plate
    const plate = new THREE.Mesh(new THREE.BoxGeometry(CITY_W + 0.3, 0.04, CITY_D + 0.3), inkMat({ ink: 'blue', lit: 0.12, shade: 0.5 }));
    plate.position.y = -0.02;
    this.group.add(plate);
  }
  static pos(i: number): [number, number] {
    return [((i % GX) + 0.5) * PITCH - CITY_W / 2, (Math.floor(i / GX) + 0.5) * PITCH - CITY_D / 2];
  }
  /** Heights from |activation| (uint8 of the layer's max): linear, 255 = HS world units. */
  set(key: string, u8: Uint8Array) {
    if (this.key === key) return;
    this.key = key;
    const a = this.base.instanceMatrix.array as Float32Array, b = this.hot.instanceMatrix.array as Float32Array;
    let n = 0;
    const act: number[] = [];
    for (let i = 0; i < u8.length; i++) {
      const [x, z] = City.pos(i);
      const on = u8[i]! > 25.5;
      const h = Math.max(0.025, (u8[i]! / 255) * HS);
      const o = i * 16;
      a[o] = on ? 0 : 1; a[o + 1] = 0; a[o + 2] = 0; a[o + 3] = 0;
      a[o + 4] = 0; a[o + 5] = on ? 0 : h; a[o + 6] = 0; a[o + 7] = 0;
      a[o + 8] = 0; a[o + 9] = 0; a[o + 10] = on ? 0 : 1; a[o + 11] = 0;
      a[o + 12] = x; a[o + 13] = 0; a[o + 14] = z; a[o + 15] = 1;
      if (on && n < 2048) {
        const p = n * 16;
        b.fill(0, p, p + 16);
        b[p] = 1.25; b[p + 5] = h; b[p + 10] = 1.25; b[p + 12] = x; b[p + 14] = z; b[p + 15] = 1;
        n++;
        act.push(i);
      }
    }
    this.hot.count = n;
    this.hotLine.count = n;
    this.base.instanceMatrix.needsUpdate = true;
    this.hot.instanceMatrix.needsUpdate = true;
    this.top = act.sort((p, q) => u8[q]! - u8[p]!);
  }
  fire(on: boolean) { this.hot.material = on ? this.hotPink : this.hotBlue; }
}

interface Pose { pos: THREE.Vector3; look: THREE.Vector3; fov: number }

export default class Ffn extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(40);
  camL = makeCam(40);
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  front = new Layer2D();
  cities: City[] = [];
  tower = new Tower();
  single = new Tower({ layers: 1 });
  rider!: THREE.Mesh;
  headsRest!: Float32Array;
  tm = { l: [0, 0, 0, 0], feed: 0, fires: 0, facts: 0, neurons: 0, attention: 0, ff2: 0, floor: 0, stack: 0, power: 0 };
  /** The step whose numbers line 4 shows: the token in flight while the tower is stacked. */
  lensStep = 1;
  private lineOf: number[] = [];

  override init() {
    const { lyrics: ly, llm, story } = this.ctx;
    const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9'-]/g, '');
    const lines = ['Then the feed-forward', 'Fourteen thousand neurons', 'one floor of the tower', 'Stack it thirty-two high'].map((q) => ly.get(q));
    const word = (li: number, w: string, nth = 0) => (lines[li]!.words.filter((x) => norm(x.w).startsWith(w))[nth] ?? lines[li]!.words[0]!).start;
    this.tm = {
      l: lines.map((l) => l.start), feed: word(0, 'feed'), fires: word(0, 'fires'), facts: word(0, 'facts'), neurons: word(1, 'neurons'),
      attention: word(2, 'attention'), ff2: word(2, 'feed'), floor: word(2, 'floor'), stack: word(3, 'stack'), power: word(3, 'power'),
    };
    this.lensStep = story.at(this.tm.l[3]! + 0.05).step;

    for (let k = 0; k < 3; k++) { const c = new City(); this.cities.push(c); this.scene.add(c.group); }

    this.tower.outlineAll();
    this.headsRest = (this.tower.heads.instanceMatrix.array as Float32Array).slice();
    this.rider = riderTile(llm.steps[this.lensStep]!.tok.t);
    this.rider.scale.setScalar(1.6);
    this.tower.ride(this.rider, 1);
    this.scene.add(this.tower.group);
    // one floor on its own (a one-layer tower without lobby, roof, shaft or number plate)
    this.single.outlineAll();
    this.single.slabs[0]!.visible = false;
    this.single.roof.visible = false;
    this.single.shaft.visible = false;
    for (const m of this.single.labels) m.visible = false;
    this.scene.add(this.single.group);
  }

  private shotLine(i: number): { line: number; sub: number } {
    const au = this.ctx.audio;
    if (!this.lineOf.length) {
      const b0 = Math.round(au.beatAt(this.ctx.start));
      for (let k = 0; k < 16; k++) {
        const mid = au.timeOfBeat(b0 + k * 4 + 2);
        let li = 0;
        this.tm.l.forEach((s, j) => { if (mid >= s - 0.05) li = j; });
        this.lineOf.push(li);
      }
    }
    const k = Math.min(i, this.lineOf.length - 1);
    const line = this.lineOf[k]!;
    let sub = 0;
    for (let j = 0; j < k; j++) if (this.lineOf[j] === line) sub++;
    return { line, sub: Math.min(sub, 1) };
  }

  private beatLen() { const au = this.ctx.audio; return au.timeOfBeat(1) - au.timeOfBeat(0); }
  private q8(t: number) { const au = this.ctx.audio; return au.timeOfBeat(Math.round(au.beatAt(t) * 2) / 2); }

  /** The floor shown in close-up: where the token in flight is when line 3 starts. */
  private floorN() { return clamp(this.ctx.story.at(this.tm.l[2]!).layer, 2, 31); }

  /** Floors built in line 4's first shot: two per 16th, snapping in. */
  private built(t: number, sh0: number) {
    const b = (this.ctx.audio.beatAt(t) - this.ctx.audio.beatAt(sh0)) * 4;
    if (b < 0) return 0;
    return Math.min(32, (Math.floor(b) + 1) * 2);
  }

  private pose(line: number, sub: number, t: number, sh: Shot): Pose {
    const au = this.ctx.audio, P = portrait();
    const d = drive(au, t, 0.5, 1.6) - drive(au, sh.t0, 0.5, 1.6);
    const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
    const c1 = this.cities[1]!;
    if (line === 0) {
      if (sub === 0) {
        // low flight toward the strongest neuron
        const [tx, tz] = City.pos(c1.top[0] ?? 7000);
        const dir = V(tx, 0, tz).normalize();
        if (dir.lengthSq() < 0.5) dir.set(0, 0, 1);
        const back = (P ? 6.2 : 5.6) - d * 0.55;
        const p0 = V(tx, 0, tz).addScaledVector(dir, -back);
        const side = V(-dir.z, 0, dir.x).multiplyScalar(0.7);
        return { pos: V(p0.x + side.x, 0.62 + d * 0.04, p0.z + side.z), look: V(p0.x + dir.x * 5, 0.78, p0.z + dir.z * 5), fov: P ? 64 : 54 };
      }
      const a = 0.6 + d * 0.08;
      const r = P ? 15 : 11.5;
      return { pos: V(Math.sin(a) * r, P ? 12 : 9, Math.cos(a) * r), look: V(0, 0.6, 0), fov: 40 };
    }
    if (line === 1) {
      if (sub === 0) {
        const a = -0.25 + d * 0.05;
        return P ? { pos: V(Math.sin(a) * 12 + 3, 46, 28), look: V(0, 0, 0.5), fov: 44 }
          : { pos: V(Math.sin(a) * 31, 17, Math.cos(a) * 31), look: V(0, -1.2, 0), fov: 40 };
      }
      const a = -0.35 + d * 0.06;
      const r = P ? 18 : 13;
      return { pos: V(Math.sin(a) * r, P ? 5.5 : 4.2, Math.cos(a) * r), look: V(0, 1.6, 0), fov: 40 };
    }
    const fy = this.tower.floorY(this.floorN());
    if (line === 2) {
      if (sub === 0) {
        const a = -0.5 + d * 0.07;
        const r = P ? 8.2 : 5.8;
        return { pos: V(Math.sin(a) * r, fy + (P ? 5.2 : 3.9), Math.cos(a) * r), look: V(P ? 0 : 0.6, fy - (P ? 0.4 : 0.1), 0), fov: 46 };
      }
      const a = -0.3 + d * 0.05;
      const r = P ? 19 : 12;
      const cx = -SLAB_W * 0.85;
      return { pos: V(cx + Math.sin(a) * r, fy + (P ? 6 : 4.2), Math.cos(a) * r), look: V(cx, fy - (P ? 1.2 : 0.4), 0), fov: 40 };
    }
    const Ht = this.tower.height;
    if (sub === 0) {
      const nC = clamp((au.beatAt(t) - au.beatAt(sh.t0)) * 8, 2, 32);
      const y = Math.max(this.tower.floorY(nC) - (P ? 1.6 : 3.2), P ? 6 : 4.5);
      const a = -0.5 + d * 0.05;
      const r = P ? 27 : 19;
      return { pos: V(Math.sin(a) * r, y + 3.5, Math.cos(a) * r), look: V(P ? 0 : 1.2, y, 0), fov: 36 };
    }
    void Ht;
    // the top floors, close: the guess sharpening floor by floor, up to the roof
    const a = -0.42 + d * 0.04;
    const r = P ? 19 : 14;
    const y = Math.min(this.tower.floorY(23) + d * 2.2, this.tower.floorY(30));
    return { pos: V(Math.sin(a) * r, y + 1.5, Math.cos(a) * r), look: V(P ? 0 : 1.6, y + 0.4, 0), fov: 40 };
  }

  private applyPose(cam: THREE.PerspectiveCamera, p: Pose) {
    cam.position.copy(p.pos);
    cam.up.set(0, 1, 0);
    cam.lookAt(p.look);
    if (cam.fov !== p.fov) { cam.fov = p.fov; cam.updateProjectionMatrix(); }
    cam.updateMatrixWorld();
  }

  private proj(v: THREE.Vector3): [number, number, boolean] {
    const p = v.clone().project(this.camL);
    return [(p.x * 0.5 + 0.5) * W, (-p.y * 0.5 + 0.5) * H, p.z < 1];
  }

  private slab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, k = 1, align: CanvasTextAlign = 'center', width = 125) {
    c.save();
    c.font = font(F.archivo(width, 900), size);
    c.textAlign = align;
    c.textBaseline = 'middle';
    c.translate(x, y);
    const s = 1.25 - 0.25 * k;
    c.scale(s, s);
    c.fillStyle = BLUE;
    c.fillText(text, size * 0.04, size * 0.035);
    c.globalCompositeOperation = 'multiply';
    c.fillStyle = PINK;
    c.fillText(text, 0, 0);
    c.restore();
  }

  private tag(c: CanvasRenderingContext2D, text: string, x: number, y: number, size = 26, bg = INK, align: 'left' | 'right' | 'center' = 'left', fg = PAPER) {
    c.save();
    c.font = font(F.mono(700), size);
    const w = c.measureText(text).width + size * 0.8;
    const x0 = align === 'left' ? x : align === 'right' ? x - w : x - w / 2;
    c.fillStyle = bg;
    c.fillRect(x0, y - size * 0.8, w, size * 1.3);
    c.fillStyle = fg;
    c.textBaseline = 'alphabetic';
    c.fillText(text, x0 + size * 0.4, y + size * 0.18);
    c.restore();
    return w;
  }

  /** A logit-lens row: floor, token, probability, drawn as separate runs so no token can reorder them. */
  private lensTag(c: CanvasRenderingContext2D, parts: string[], x: number, y: number, size: number, hit: boolean) {
    c.save();
    c.font = font(F.mono(700), size);
    const gap = size * 0.7;
    const ws = parts.map((p) => c.measureText(p).width);
    const w = ws.reduce((a, b) => a + b, 0) + gap * (parts.length - 1) + size * 0.8;
    c.fillStyle = hit ? PINK : PAPER;
    c.fillRect(x, y - size * 0.8, w, size * 1.3);
    c.fillStyle = hit ? PAPER : INK;
    c.textBaseline = 'alphabetic';
    c.direction = 'ltr';
    let xx = x + size * 0.4;
    parts.forEach((p, i) => { c.fillText(p, xx, y + size * 0.18); xx += ws[i]! + gap; });
    c.restore();
  }

  private cap(c: CanvasRenderingContext2D, text: string, x: number, y: number, size = 24, col = INK, align: CanvasTextAlign = 'left') {
    c.save();
    c.font = font(F.mono(700), size);
    const w = c.measureText(text).width, x0 = align === 'left' ? x : align === 'right' ? x - w : x - w / 2;
    c.fillStyle = PAPER;
    c.fillRect(x0 - 8, y - size * 0.95, w + 16, size * 1.3);
    c.fillStyle = col;
    c.textAlign = align;
    c.textBaseline = 'alphabetic';
    c.fillText(text, x, y);
    c.restore();
  }

  private leader(c: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, col = INK) {
    c.save();
    c.strokeStyle = col; c.fillStyle = col; c.lineWidth = 3;
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    c.beginPath(); c.arc(x0, y0, 6, 0, Math.PI * 2); c.fill();
    c.restore();
  }

  /** Every piece of the big tower back to its rest state (the render sets all of it every frame). */
  private resetTower() {
    const T = this.tower;
    for (let l = 1; l <= T.layers; l++) {
      T.slabs[l]!.visible = true; T.ff[l - 1]!.visible = true; T.labels[l - 1]!.visible = true;
      (T.slabs[l]!.material as THREE.ShaderMaterial).uniforms.hot!.value = 0;
      (T.ff[l - 1]!.material as THREE.ShaderMaterial).uniforms.hot!.value = 0;
    }
    (T.roof.material as THREE.ShaderMaterial).uniforms.hot!.value = 0;
    T.roof.visible = true;
    T.shaft.scale.y = 1;
    T.shaft.position.y = (T.floorY(T.layers + 1) + 0.4) / 2 - 0.3;
    (T.heads.instanceMatrix.array as Float32Array).set(this.headsRest);
    T.heads.instanceMatrix.needsUpdate = true;
    T.clearHeads();
  }

  /** Where the k-th of the three recorded floors sits: side by side (16:9), stacked front to back (9:16). */
  private cityAt(k: number) {
    return portrait() ? new THREE.Vector3(0, 0, (1 - k) * (CITY_D + 2.5)) : new THREE.Vector3((k - 1) * (CITY_W + 2.2), 0, 0);
  }

  private kvKey = new Map<Tower, string>();
  /** KV shelf with n card pairs per floor (floors above `cut` emptied), rebuilt only when that changes. */
  private kv(T: Tower, n: number, cut = 0) {
    const key = `${n}:${cut}`;
    if (this.kvKey.get(T) === key) return;
    this.kvKey.set(T, key);
    T.setKV(n);
    if (cut > 0) {
      (T.kv.instanceMatrix.array as Float32Array).fill(0, cut * 96 * 2 * 16);
      T.kv.instanceMatrix.needsUpdate = true;
    }
  }

  /** Shaft from the lobby up to y. */
  private setShaft(top: number) {
    const T = this.tower, full = T.floorY(T.layers + 1) + 0.4, h = top + 0.3;
    T.shaft.scale.y = Math.max(0.01, h / full);
    T.shaft.position.y = -0.3 + h / 2;
  }

  /** Cut the tower open above floor n: everything higher is lifted off (the close-up looks into floor n). */
  private cutAbove(n: number) {
    const T = this.tower;
    for (let l = n + 1; l <= T.layers; l++) { T.slabs[l]!.visible = false; T.ff[l - 1]!.visible = false; T.labels[l - 1]!.visible = false; }
    T.roof.visible = false;
    (T.heads.instanceMatrix.array as Float32Array).fill(0, n * 32 * 16);
    T.heads.instanceMatrix.needsUpdate = true;
    this.setShaft(T.floorY(n) + 0.5);
  }

  /** Hide floor n of the big tower (its slab, FF block and head ring) or restore it (n = 0). */
  private hideFloor(n: number) {
    const T = this.tower;
    for (let l = 1; l <= T.layers; l++) { T.slabs[l]!.visible = l !== n; T.ff[l - 1]!.visible = l !== n; }
    const a = T.heads.instanceMatrix.array as Float32Array;
    a.set(this.headsRest);
    if (n > 0) a.fill(0, (n - 1) * 32 * 16, n * 32 * 16);
    T.heads.instanceMatrix.needsUpdate = true;
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio, llm = this.ctx.llm, story = this.ctx.story;
    const sh = shot(this.ctx, f, { every: 4 });
    const { line, sub } = this.shotLine(sh.i);
    const P = portrait();
    const d = drive(au, f.t, 0.5, 1.6);
    const tm = this.tm;
    const step = story.at(f.ft).step;

    // background
    const blueWash = (line + sub) % 2 === 0;
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: 0.5, c1: blueWash ? 'blue' : 'pink', c2: blueWash ? 'pink' : 'blue' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => tokenRain(this.back.ctx, llm.d.prompt, drive(au, f.ft, 0.5, 1.6) * 0.5, { seed: sh.seed, cols: P ? 8 : 13 })), out);

    // hero
    const cityShot = line === 0 || line === 1;
    const [c0, c1, c2] = this.cities as [City, City, City];
    for (const c of this.cities) c.group.visible = false;
    this.tower.group.visible = !cityShot;
    this.single.group.visible = line === 2 && sub === 1;
    if (cityShot) {
      if (line === 1 && sub === 0) {
        // three floors side by side
        [c0, c1, c2].forEach((c, k) => {
          c.set(`${step}:${FF_LAYERS[k]}`, llm.ff(step, FF_LAYERS[k]));
          c.group.visible = true;
          c.group.position.copy(this.cityAt(k));
          c.group.scale.set(1, Math.max(0.02, snapIn(f.t, sh.t0 + k * this.beatLen() * 0.5, 0.09)), 1);
          c.fire(true);
        });
      } else {
        // one floor: the step in flight, or (line 2, shot 2) every step of the answer on the 16ths
        let s = step;
        if (line === 1) s = clamp(Math.floor((au.beatAt(f.ft) - au.beatAt(sh.t0)) * 4 + 1e-3), 0, llm.steps.length - 1);
        c1.set(`${s}:15`, llm.ff(s, 15));
        c1.group.visible = true;
        c1.group.position.set(0, 0, 0);
        const rise = line === 0 && sub === 0 ? snapIn(f.t, sh.t0, 0.09) : 1;
        c1.group.scale.set(1, Math.max(0.02, rise), 1);
        c1.fire(!(line === 0 && sub === 0) || f.ft >= this.q8(tm.fires));
      }
    } else {
      const T = this.tower, N = this.floorN(), pos = story.at(f.ft).pos;
      this.resetTower();
      if (line === 2) {
        // the floor in close-up: heads lit by how much each looks past the start token (real attention)
        const w = new Float32Array(32);
        for (let h = 0; h < 32; h++) w[h] = 1 - (llm.attn(step, N - 1, h)[0] ?? 1);
        const m = Math.max(1e-3, ...w);
        for (let h = 0; h < 32; h++) w[h] = w[h]! / m;
        this.kv(T, pos, sub === 0 ? N : 0);
        if (sub === 0) {
          this.cutAbove(N);
          T.setHeads(N, w);
          (T.ff[N - 1]!.material as THREE.ShaderMaterial).uniforms.hot!.value = f.ft >= this.q8(tm.ff2) ? 1 : 0;
        } else {
          // that floor pulled out sideways on "floor", snapping
          this.hideFloor(N);
          const S = this.single;
          S.setHeads(1, w);
          this.kv(S, 0);
          (S.ff[0]!.material as THREE.ShaderMaterial).uniforms.hot!.value = 1;
          (S.slabs[1]!.material as THREE.ShaderMaterial).uniforms.hot!.value = 0.35;
          const o = snapIn(f.t, this.q8(tm.floor), 0.1);
          S.group.position.set(-o * (SLAB_W * 1.55), T.floorY(N) - S.floorY(1) + o * 0.9, o * 0.8);
        }
        T.ride(this.rider, N / 33);
        this.rider.scale.setScalar(0.9);
      } else {
        // the stack: two floors per 16th in the first shot; all 32 after
        const n = sub === 0 ? this.built(f.ft, sh.t0) : 32;
        for (let l = 1; l <= T.layers; l++) {
          const on = l <= n;
          T.slabs[l]!.visible = on; T.ff[l - 1]!.visible = on; T.labels[l - 1]!.visible = on;
          (T.slabs[l]!.material as THREE.ShaderMaterial).uniforms.hot!.value = on && l > n - 2 && sub === 0 ? 0.8 : 0;
        }
        const ha = T.heads.instanceMatrix.array as Float32Array;
        if (n < 32) { ha.fill(0, n * 32 * 16); T.heads.instanceMatrix.needsUpdate = true; }
        this.kv(T, n >= 32 ? pos : 0);
        T.roof.visible = n >= 32;
        if (n < 32) this.setShaft(T.floorY(n) + 0.5);
        // the token rides to the top of what is built; on "power" the roof floods pink (the pick)
        T.ride(this.rider, Math.min(1, (n + 0.2) / 33));
        this.rider.scale.setScalar(1.6);
        (T.roof.material as THREE.ShaderMaterial).uniforms.hot!.value = snapIn(f.t, this.q8(tm.power));
      }
    }

    const pose = this.pose(line, sub, f.t, sh);
    this.applyPose(this.cam, pose);
    this.applyPose(this.camL, this.pose(line, sub, f.ft, sh));
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);

    // type
    this.ctx.comp.draw(r, perFrame(this.mid, f, () => this.drawType(f, sh, line, sub, step)), out);

    // foreground
    const c = this.front.ctx;
    this.front.clear();
    specks(c, d * (line === 3 ? 1.6 : 1), { seed: sh.seed, n: line === 0 ? 90 : 70 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    return { reg: sh.seed };
  }

  private drawType(f: Frame, sh: Shot, line: number, sub: number, step: number) {
    const c = this.mid.ctx, ft = f.ft, P = portrait(), llm = this.ctx.llm, au = this.ctx.audio, tm = this.tm;
    const cfg = llm.cfg;
    const L = P ? 110 : 120, R = W - (P ? 70 : 90);
    const k = snapIn(ft, sh.t0, 0.08);
    const bl = this.beatLen();

    if (line === 0) {
      const act = llm.steps[step]!.ff['15']!.active;
      if (sub === 0) {
        const fired = ft >= this.q8(tm.fires);
        const cx = P ? W / 2 : W * 0.72, cy = P ? H * 0.74 : H * 0.62;
        this.slab(c, comma(cfg.ffn), cx, cy, P ? 110 : 130, k, 'center', 100);
        this.cap(c, 'FEED-FORWARD NEURONS · FLOOR 16', cx, cy + (P ? 86 : 100), P ? 24 : 28, INK, 'center');
        if (fired) this.tag(c, `${act} FIRE (> 10% OF MAX)`, cx, cy + (P ? 140 : 156), P ? 28 : 32, PINK, 'center');
      } else {
        // the strongest neurons, tagged one per 8th
        const c1 = this.cities[1]!;
        const u8 = llm.ff(step, 15);
        const n = Math.min(5, c1.top.length);
        for (let j = 0; j < n; j++) {
          const kk = snapIn(ft, sh.t0 + j * bl * 0.5);
          if (kk <= 0) continue;
          const i = c1.top[j]!;
          const [x, z] = City.pos(i);
          const [ax, ay] = this.proj(new THREE.Vector3(x, (u8[i]! / 255) * HS, z));
          const tx = P ? L + 20 : R, ty = P ? H * 0.64 + j * 58 : 220 + j * 62;
          const lab = `#${comma(i)}  ${(u8[i]! / 255).toFixed(2)} × MAX`;
          c.save(); c.font = font(F.mono(700), 28); const w = c.measureText(lab).width + 22; c.restore();
          this.leader(c, ax, ay, P ? tx : tx - w, ty - 8, PINK);
          this.tag(c, lab, tx, ty, P ? 26 : 28, j === 0 ? PINK : INK, P ? 'left' : 'right');
        }
        const y0 = P ? H * 0.34 : H - 170;
        this.tag(c, `FLOOR 16 · ${act} OF ${comma(cfg.ffn)} FIRE`, L, y0, P ? 28 : 30, INK);
        this.cap(c, `|ACTIVATION| OF EACH NEURON · TOKEN ${step + 1} IN FLIGHT`, L, y0 + (P ? 48 : 52), P ? 22 : 24, BLUE);
      }
    }

    if (line === 1) {
      if (sub === 0) {
        // the three recorded floors: 14,336 each, and how many fire (slamming in on the beats)
        FF_LAYERS.forEach((l, j) => {
          const kk = snapIn(ft, sh.t0 + j * bl * 0.5, 0.08);
          if (kk <= 0) return;
          const o = this.cityAt(j);
          const [x, y] = this.proj(o.clone().add(new THREE.Vector3(P ? CITY_W / 2 + 0.4 : 0, 0, P ? 0 : CITY_D / 2 + 0.5)));
          const [xt, yt] = this.proj(o.clone().add(new THREE.Vector3(P ? -CITY_W * 0.2 : 0, HS * 0.75, P ? 0 : -CITY_D / 2)));
          const act = llm.steps[step]!.ff[String(l)]!.active;
          const px = clamp(x, L + 90, R - 90);
          const sx = P ? W * 0.3 : clamp(xt, L + 90, R - 90), sy = P ? clamp(yt, 260, H * 0.72) : clamp(yt, 220, H * 0.45);
          this.slab(c, String(act), sx, sy, P ? 96 : 110, kk, 'center', 100);
          this.cap(c, 'FIRE', sx, sy + (P ? 68 : 76), P ? 24 : 28, PINK, 'center');
          this.tag(c, `L${l + 1} · ${comma(cfg.ffn)}`, P ? clamp(x, W * 0.62, R - 150) : px, P ? clamp(y, 260, H * 0.76) : clamp(y + 40, 250, H - 190), P ? 24 : 28, INK, 'center');
        });
      } else {
        // every step: the same 14,336, a new pattern per token, flipping on the 16ths
        const s = clamp(Math.floor((au.beatAt(ft) - au.beatAt(sh.t0)) * 4 + 1e-3), 0, llm.steps.length - 1);
        const act = llm.steps[s]!.ff['15']!.active;
        const cx = P ? W / 2 : W * 0.74, cy = P ? H * 0.72 : H * 0.36;
        this.slab(c, comma(act), cx, cy, P ? 130 : 150, snapIn(ft, sh.t0 + s * bl / 4, 0.05), 'center', 100);
        this.cap(c, `FIRE AT STEP ${s + 1} OF ${llm.steps.length}`, cx, cy + (P ? 96 : 110), P ? 26 : 30, INK, 'center');
        this.cap(c, `FLOOR 16 · SAME ${comma(cfg.ffn)} NEURONS, NEW PATTERN EVERY TOKEN`, P ? W / 2 : L, P ? cy + 146 : H - 170, P ? 22 : 24, BLUE, P ? 'center' : 'left');
      }
    }

    if (line === 2) {
      const N = this.floorN();
      const T = this.tower;
      if (sub === 0) {
        // ATTENTION (the head ring) and FEED-FORWARD (the block), each on its word
        const fy = T.floorY(N);
        const [hx, hy] = this.proj(new THREE.Vector3(-SLAB_W * 0.08 - SLAB_W * 0.36 * 0.7, fy + 0.3, SLAB_W * 0.08 + SLAB_W * 0.36 * 0.7));
        const [fx, fyy] = this.proj(new THREE.Vector3(SLAB_W * 0.22, fy + 0.05 + FLOOR_H * 0.46, -SLAB_W * 0.22));
        const s1 = P ? 40 : 46;
        const ay = P ? H * 0.3 : 240, by = P ? H * 0.7 : H - 200;
        this.leader(c, hx, hy, P ? W * 0.3 : W * 0.24, ay, INK);
        this.tag(c, 'ATTENTION', P ? W * 0.3 : W * 0.24, ay, s1, INK, 'center');
        this.cap(c, `${cfg.heads} HEADS · PINK = READING THE WORDS`, P ? W * 0.3 : W * 0.24, ay + s1 * 1.25, P ? 22 : 24, INK, 'center');
        if (ft >= this.q8(tm.ff2)) {
          this.leader(c, fx, fyy, P ? W * 0.62 : W * 0.72, by, PINK);
          this.tag(c, 'FEED-FORWARD', P ? W * 0.62 : W * 0.72, by, s1, PINK, 'center');
          this.cap(c, `${comma(cfg.ffn)} NEURONS`, P ? W * 0.62 : W * 0.72, by + s1 * 1.25, P ? 22 : 24, INK, 'center');
        }
      } else {
        const [x, y] = this.proj(new THREE.Vector3(-SLAB_W * 1.55, T.floorY(N) + 1.5, 0.8));
        const out = ft >= this.q8(tm.floor);
        if (out) {
          const tx = clamp(x, L + 200, R - 200), ty = clamp(y - (P ? 170 : 150), 230, H - 300);
          this.tag(c, `1 FLOOR = L${N}`, tx, ty, P ? 40 : 46, PINK, 'center');
          this.cap(c, `ATTENTION (${cfg.heads} HEADS) + FEED-FORWARD (${comma(cfg.ffn)})`, tx, ty + (P ? 56 : 62), P ? 22 : 24, INK, 'center');
        }
        this.cap(c, `L${N} OF ${cfg.layers}`, P ? W / 2 : W * 0.2, P ? H * 0.8 : H - 170, P ? 30 : 34, INK, 'center');
      }
    }

    if (line === 3) {
      // the logit lens: each floor's best guess for the next token (real), printed at its edge
      const st = llm.steps[this.lensStep]!;
      const T = this.tower;
      const n = sub === 0 ? this.built(ft, sh.t0) : 32;
      const pick = st.tok.t;
      const size = sub === 0 ? (P ? 22 : 22) : (P ? 28 : 30);
      const lo = sub === 0 ? 1 : 18;
      for (let l = lo; l <= n; l++) {
        const [top, p] = st.lens[l]!.top[0]!;
        const [x, y, vis] = this.proj(new THREE.Vector3(SLAB_W / 2, T.floorY(l), SLAB_W / 2));
        if (!vis || y < 170 || y > (P ? H * 0.77 : H - 190)) continue;
        const hit = top === pick;
        const xx = clamp(x + 24, L, R - 360);
        c.save();
        c.fillStyle = hit ? PINK : INK;
        c.fillRect(x, y - 1.5, xx - x, 3);
        c.restore();
        if (sub === 1) {
          // probability bar
          c.fillStyle = hit ? PINK : BLUE;
          c.fillRect(xx, y + size * 0.5, Math.max(3, p * 300), size * 0.35);
        }
        this.lensTag(c, [`L${l}`, showTok(top).slice(0, 12), `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`], xx, y + size * 0.2, size, hit);
      }
      if (sub === 0) {
        // ×N counter
        const cx = P ? W / 2 : W * 0.19, cy = P ? H * 0.16 : H * 0.4;
        this.slab(c, `×${n}`, cx, cy + (P ? 40 : 0), P ? 150 : 170, snapIn(ft, sh.t0 + (Math.floor((au.beatAt(ft) - au.beatAt(sh.t0)) * 4)) * bl / 4, 0.05), 'center', 100);
        this.cap(c, 'FLOORS STACKED', cx, cy + (P ? 136 : 120), P ? 24 : 26, INK, 'center');
      }
      const gy = P ? H * 0.8 : H - 165;
      this.tag(c, 'LOGIT LENS', L, gy, P ? 28 : 30, INK);
      this.cap(c, `EACH FLOOR’S BEST GUESS FOR TOKEN ${this.lensStep + 1}`, L, gy + (P ? 46 : 50), P ? 22 : 24, BLUE);
      if (sub === 1 && ft >= this.q8(tm.power)) {
        const [rx, ry] = this.proj(new THREE.Vector3(0, T.floorY(33) + 0.4, 0));
        this.tag(c, `PICKED ${showTok(pick)}`, clamp(rx, L + 150, R - 150), clamp(ry - 60, 220, H - 200), P ? 36 : 40, PINK, 'center');
      }
    }
  }
}
