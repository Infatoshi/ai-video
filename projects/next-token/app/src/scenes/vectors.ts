// vectors (verse 1, lines 5-8): "Every token is a vector, four thousand wide, / Words that mean the same
// thing end up side by side, / Now the sentence is a stack of numbers standing tall, / Thirty-two floors
// in the tower and it's gotta climb them all". A new shot every bar, two per line, each a new framing:
//  line 1: the " strawberry" tile unrolls into a bar code of its REAL first 64 embedding values (of 4,096),
//          then the same bar code end-on with the 64 numbers printed as a list
//  line 2: the 35-word embedding cloud at its real PCA positions, pairs linked, the real cosine printed;
//          then close on the places cluster (Paris/France, London/England, Tokyo/Japan)
//  line 3: the question's 8 tokens stacking as bar-code slabs (real first-64 values), then all 43
//          prompt tokens standing tall
//  line 4: crane up the tower from the lobby to the roof, one floor per 16th note, L1..L32
// Every number is llm.json (embed.first64, embed.pca3, embed.cos, prompt ids, config) or arithmetic on it.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba, HEX } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease } from '../engine/util';
import { showTok } from '../engine/llm';
import {
  Wash, tokenRain, specks, makeCam, drive, shot, snapIn, portrait, perFrame, tokenFaceTexture, inkMat, outline,
  outlineInstanced, flatMat, tokenTile, type Shot,
} from './_kit';
import { Tower, riderTile } from './_tower';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
const HERO = ' strawberry';
/** 3D bar code: slot spacing, bar width / depth, tallest bar. */
const SP = 0.2, BW = 0.14, BD = 0.7, BH = 2.6;
/** Embedding cloud: PCA units -> world units. */
const CS = 7;
/** Sentence stack: slab size and floor pitch. */
const SW = 6.4, SHh = 0.5, SD = 1.0, GAP = 0.68;
const PAIRS: [string, string][] = [['king', 'queen'], ['man', 'woman'], ['cat', 'kitten'], ['dog', 'puppy'], ['Paris', 'France'], ['London', 'England'], ['Tokyo', 'Japan'], ['GPU', 'CPU']];

const fmt = (v: number) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(4);
const comma = (n: number) => n.toLocaleString('en-US');

/** A slab face: the token in an ink box on the left, its 64 values as a bar code (pink up, blue down). */
function barcodeTexture(tok: string, vals: number[], hot: boolean): THREE.CanvasTexture {
  const cw = 2048, ch = Math.round((2048 * SHh) / SW);
  const cv = document.createElement('canvas');
  cv.width = cw; cv.height = ch;
  const c = cv.getContext('2d')!;
  c.fillStyle = HEX.paper; c.fillRect(0, 0, cw, ch);
  const lw = cw * 0.24;
  c.fillStyle = hot ? HEX.pink : HEX.ink; c.fillRect(0, 0, lw, ch);
  let size = ch * 0.62;
  const lab = showTok(tok);
  c.font = font(F.mono(700), size);
  const tw = c.measureText(lab).width;
  if (tw > lw - 36) { size *= (lw - 36) / tw; c.font = font(F.mono(700), size); }
  c.fillStyle = HEX.paper; c.textBaseline = 'middle';
  c.fillText(lab, 18, ch * 0.54);
  const m = Math.max(...vals.map(Math.abs)) || 1;
  const x0 = lw + 18, bw = (cw - x0 - 16) / vals.length, mid = ch / 2;
  vals.forEach((v, i) => {
    const hh = Math.sqrt(Math.abs(v) / m) * (ch / 2 - 8);
    c.fillStyle = v >= 0 ? HEX.pink : HEX.blue;
    c.fillRect(x0 + i * bw + 2, v >= 0 ? mid - hh : mid, bw - 6, Math.max(4, hh));
  });
  c.fillStyle = HEX.ink; c.fillRect(x0, mid - 2, cw - x0 - 16, 4);
  c.strokeStyle = HEX.ink; c.lineWidth = 8; c.strokeRect(4, 4, cw - 8, ch - 8);
  const tx = new THREE.CanvasTexture(cv);
  tx.colorSpace = THREE.SRGBColorSpace;
  tx.anisotropy = 8;
  return tx;
}

/** The empty slots of the rest of the vector: a tick per slot, a heavy tick every 64. */
function slotTexture(): THREE.CanvasTexture {
  const cw = 1024, ch = 64;
  const cv = document.createElement('canvas');
  cv.width = cw; cv.height = ch;
  const c = cv.getContext('2d')!;
  c.fillStyle = HEX.paper; c.fillRect(0, 0, cw, ch);
  c.fillStyle = HEX.blue;
  for (let i = 0; i < 64; i++) c.fillRect((i / 64) * cw + 2, 10, 6, ch - 20);
  c.fillStyle = HEX.ink;
  c.fillRect(0, 0, 10, ch);
  c.fillRect(0, 0, cw, 6); c.fillRect(0, ch - 6, cw, 6);
  const tx = new THREE.CanvasTexture(cv);
  tx.colorSpace = THREE.SRGBColorSpace;
  tx.wrapS = THREE.RepeatWrapping;
  tx.anisotropy = 8;
  return tx;
}

interface Pose { pos: THREE.Vector3; look: THREE.Vector3; fov: number; up?: THREE.Vector3 }

export default class Vectors extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(38);
  /** Same pose at the frame's own time: labels (drawn once per frame) project with it. */
  camL = makeCam(38);
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  front = new Layer2D();

  // line 1: the bar code
  code = new THREE.Group();
  heroTile!: THREE.Mesh;
  barsP!: THREE.InstancedMesh;
  barsB!: THREE.InstancedMesh;
  ribbon!: THREE.Mesh;
  spine!: THREE.Mesh;
  vals: number[] = [];
  vmax = 1;
  // line 2: the cloud
  cloud = new THREE.Group();
  tiles: { w: string; m: THREE.Mesh; face: THREE.Material; hot: THREE.Material; p: THREE.Vector3 }[] = [];
  links: { a: string; b: string; m: THREE.Mesh; cos: number }[] = [];
  // line 3: the stack
  stack = new THREE.Group();
  slabs: THREE.Mesh[] = [];
  qStart = 0;
  // line 4: the tower
  tower = new Tower();
  rider!: THREE.Mesh;

  /** Lyric times (from ctx.lyrics): line starts and the words that trigger beats. */
  tm = { l: [0, 0, 0, 0], vector: 0, thousand: 0, side: 0, stack: 0, numbers: 0 };
  rainVals: { id: number; t: string }[] = [];
  private tmp = new THREE.Object3D();
  private lineOf: number[] = [];

  override init() {
    const { llm, lyrics: ly } = this.ctx;
    const e = llm.d.embed;
    const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9']/g, '');
    const lines = ['Every token is a vector', 'Words that mean the same', 'stack of numbers', 'Thirty-two floors'].map((q) => ly.get(q));
    const word = (li: number, w: string) => (lines[li]!.words.find((x) => norm(x.w).startsWith(w)) ?? lines[li]!.words[0]!).start;
    this.tm = { l: lines.map((l) => l.start), vector: word(0, 'vector'), thousand: word(0, 'thousand'), side: word(1, 'side'), stack: word(2, 'stack'), numbers: word(2, 'numbers') };

    // ---- line 1: the hero tile and its 64 real values
    this.vals = e.first64[HERO]!;
    this.vmax = Math.max(...this.vals.map(Math.abs));
    const heroId = llm.d.prompt.find((p) => p.t === HERO)?.id ?? 0;
    this.heroTile = tokenTile(HERO, { face: 'pink', text: 'paper', side: 'pink', id: heroId, h: 1.1, line: 2.6 });
    const tw = (this.heroTile.geometry as THREE.BoxGeometry).parameters.width;
    this.heroTile.position.set(-0.45 - tw / 2, 0.15, 0);
    this.code.add(this.heroTile);
    const box = new THREE.BoxGeometry(1, 1, 1);
    this.barsP = new THREE.InstancedMesh(box, inkMat({ ink: 'pink', lit: 0.55, shade: 1 }), 64);
    this.barsB = new THREE.InstancedMesh(box, inkMat({ ink: 'blue', lit: 0.45, shade: 1 }), 64);
    for (const b of [this.barsP, this.barsB]) { b.frustumCulled = false; outlineInstanced(b, 1.6); this.code.add(b); }
    const spine = this.spine = new THREE.Mesh(new THREE.BoxGeometry(64 * SP + 0.2, 0.05, 0.14), flatMat('ink'));
    spine.position.set(32 * SP - SP / 2, 0, 0);
    this.code.add(spine);
    const RL = 190;
    const st = slotTexture();
    st.repeat.set(RL / (64 * SP), 1);
    this.ribbon = new THREE.Mesh(new THREE.BoxGeometry(RL, 0.06, BD), [flatMat('ink'), flatMat('ink'), new THREE.MeshBasicMaterial({ map: st }), flatMat('ink'), flatMat('blue'), flatMat('ink')]);
    this.ribbon.geometry.translate(RL / 2, 0, 0);
    this.ribbon.position.set(64 * SP - SP / 2, 0, 0);
    this.code.add(this.ribbon);
    this.setBars(1);
    this.scene.add(this.code);
    // rain of the real values (texture only), keyed by the prompt's ids
    this.rainVals = llm.d.prompt.flatMap((p) => (e.first64[p.t] ?? []).slice(0, 6).map((v) => ({ id: p.id, t: fmt(v) })));

    // ---- line 2: the cloud (every word at its real PCA position)
    e.words.forEach((w, i) => {
      const [x, y, z] = e.pca3[i]!;
      const m = tokenTile(w, { h: 0.44, line: 1.8 });
      const ratio = Math.max(1.2, 0.34 * showTok(w).length + 0.5);
      const mats = m.material as THREE.Material[];
      const hot = new THREE.MeshBasicMaterial({ map: tokenFaceTexture(w, 'pink', 'paper', { w: Math.round(256 * ratio), h: 256 }) });
      const p = new THREE.Vector3(x * CS, y * CS, z * CS);
      m.position.copy(p);
      this.cloud.add(m);
      this.tiles.push({ w, m, face: mats[4]!, hot, p });
    });
    for (const [a, b] of PAIRS) {
      const ia = e.words.indexOf(a), ib = e.words.indexOf(b);
      if (ia < 0 || ib < 0) continue;
      const A = this.tiles[ia]!.p, B = this.tiles[ib]!.p;
      const len = A.distanceTo(B);
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, len, 8, 1, true), flatMat('pink', THREE.DoubleSide));
      m.position.copy(A).add(B).multiplyScalar(0.5);
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), B.clone().sub(A).normalize());
      this.cloud.add(m);
      this.links.push({ a, b, m, cos: e.cos[ia]![ib]! });
    }
    this.scene.add(this.cloud);

    // ---- line 3: the prompt as slabs (bottom = first token)
    this.qStart = llm.questionStart;
    const sideQ = inkMat({ ink: 'pink', lit: 0.5, shade: 1 }), sideT = inkMat({ ink: 'blue', lit: 0.45, shade: 1 });
    const slabGeo = new THREE.BoxGeometry(SW, SHh, SD);
    const texCache = new Map<string, THREE.Texture>();
    llm.d.prompt.forEach((p, i) => {
      const q = i >= this.qStart && i < this.qStart + llm.d.question_tokens.length;
      const key = `${p.t}|${q}`;
      let tx = texCache.get(key);
      if (!tx) { tx = barcodeTexture(p.t, e.first64[p.t] ?? new Array(64).fill(0), q); texCache.set(key, tx); }
      const side = q ? sideQ : sideT;
      const m = new THREE.Mesh(slabGeo, [side, side, side, side, new THREE.MeshBasicMaterial({ map: tx }), side]);
      outline(m, 1.8);
      this.stack.add(m);
      this.slabs.push(m);
    });
    this.scene.add(this.stack);

    // ---- line 4: the tower (the prompt fills every floor's KV shelf on the way up)
    this.tower.outlineAll();
    this.tower.setKV(llm.d.prompt.length);
    this.rider = riderTile(HERO);
    this.rider.scale.setScalar(1.7);
    this.tower.ride(this.rider, 0);
    this.scene.add(this.tower.group);
  }

  /** Bar code instances: bar i grows with reveal(i) (0..1). */
  private setBars(reveal: number | ((i: number) => number)) {
    const r = typeof reveal === 'number' ? () => reveal : reveal;
    this.vals.forEach((v, i) => {
      const h = Math.max(0.03, (Math.abs(v) / this.vmax) * BH) * r(i);
      const on = r(i) > 0.001;
      for (const [mesh, pos] of [[this.barsP, true], [this.barsB, false]] as const) {
        const show = on && (v >= 0) === pos;
        this.tmp.position.set(i * SP, v >= 0 ? h / 2 : -h / 2, 0);
        this.tmp.scale.set(show ? BW : 0, show ? h : 0, show ? BD : 0);
        this.tmp.updateMatrix();
        mesh.setMatrixAt(i, this.tmp.matrix);
      }
    });
    this.barsP.instanceMatrix.needsUpdate = true;
    this.barsB.instanceMatrix.needsUpdate = true;
  }

  /** Which lyric line (0..3) a shot belongs to (the one sung at its midpoint), and its index within it. */
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
    return { line, sub };
  }

  /** Camera pose for (line, sub) at time t: a pure function, so the label camera matches the render camera. */
  private pose(line: number, sub: number, t: number, sh: Shot): Pose {
    const au = this.ctx.audio, P = portrait();
    // distance travelled since the cut (drums-driven): every move starts from its framing on the cut
    const d = drive(au, t, 0.5, 1.6) - drive(au, sh.t0, 0.5, 1.6);
    const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
    if (line === 0) {
      if (sub === 0) {
        // the tile alone, close; on "vector" a whip back to the 3/4 view as the bar code unrolls
        const tc = this.heroTile.position;
        const A: Pose = P ? { pos: V(tc.x + 0.3 + d * 0.1, 0.9, 13.5), look: V(tc.x + 0.3, -1.2, 0), fov: 40 } : { pos: V(tc.x + 1.6 + d * 0.15, 0.9, 6.4), look: V(tc.x + 1.3, 0.1, 0), fov: 40 };
        const B: Pose = P ? { pos: V(-5.8 + d * 0.3, 7.2, 12.5), look: V(4.2 + d * 0.3, -0.6, 0), fov: 46 } : { pos: V(-6.2 + d * 0.35, 3.6, 11.2), look: V(3.6 + d * 0.35, -0.3, 0), fov: 38 };
        const w = ease.inOutExpo(clamp((t - this.unrollT()) / 0.2));
        return { pos: A.pos.clone().lerp(B.pos, w), look: A.look.clone().lerp(B.look, w), fov: A.fov + (B.fov - A.fov) * w };
      }
      // side on, long lens, trucking right along the vector: the bar code slides by, the empty slots follow
      const x = 5.2 + d * 0.9;
      return P ? { pos: V(x + 1.5, 1.8, 24), look: V(x + 1.5, -3.2, 0), fov: 34 } : { pos: V(x, 0.9, 21), look: V(x, -1.4, 0), fov: 30 };
    }
    if (line === 1) {
      if (sub === 0) {
        const a = 0.95 + d * 0.12;
        const r = P ? 13 : 10.5;
        return { pos: V(Math.cos(a) * r, 2.2 + d * 0.2, Math.sin(a) * r), look: V(0.2, 0.2, 0.3), fov: 40 };
      }
      // close on the places cluster
      const c = V(0.55 * CS, 0.37 * CS, -0.08 * CS);
      const a = 0.22 + d * 0.1;
      const r = P ? 5.2 : 3.7;
      return { pos: V(c.x + Math.cos(a) * r, c.y + 1.1, c.z + Math.sin(a) * r), look: c.clone().add(V(0, -0.1, 0)), fov: 40 };
    }
    if (line === 2) {
      if (sub === 0) {
        const n = 8, top = n * GAP;
        const a = -0.4 + d * 0.08;
        const r = P ? 19.5 : 12.5;
        return { pos: V(Math.sin(a) * r, top * 0.55 + 1.6, Math.cos(a) * r), look: V(P ? 0 : 2.9, top * (P ? 0.3 : 0.44), 0), fov: 38 };
      }
      const n = this.slabs.length, top = n * GAP;
      const a = 0.42 + d * 0.07;
      const r = P ? 10 : 8.5;
      return { pos: V(Math.sin(a) * r, 0.2 + d * 0.3, Math.cos(a) * r), look: V(P ? 0 : 1.6, top * 0.55, 0), fov: P ? 74 : 66 };
    }
    // line 4: crane. One floor per 16th from the line's first shot; the camera rides with the tile.
    const y = this.tower.floorY(this.craneFloor(t));
    if (sub === 0) {
      const a = -0.95 + d * 0.06;
      const r = P ? 13.5 : 10;
      return { pos: V(Math.sin(a) * r, y + 1.5, Math.cos(a) * r), look: V(P ? 0 : 1.4, y + 0.2, 0), fov: 42 };
    }
    // worm's eye up the facade: what is left to climb, converging overhead
    const a = -0.62 + d * 0.05;
    const r = P ? 8.2 : 7.4;
    return { pos: V(Math.sin(a) * r, y - 1.2, Math.cos(a) * r), look: V(0, y + 5.5, 0), fov: P ? 70 : 60 };
  }

  /** Continuous floor (0 = lobby .. 33 = roof) for the crane: steps on the 16ths, each step snapping in. */
  private craneFloor(t: number) {
    const au = this.ctx.audio;
    const t0 = this.craneT0();
    const b = (au.beatAt(t) - au.beatAt(t0)) * 4;
    if (b < 0) return 0;
    const k = Math.floor(b);
    return Math.min(33, k + ease.outExpo(clamp((b - k) / 0.45)));
  }
  private craneT0() {
    const au = this.ctx.audio;
    const b0 = Math.round(au.beatAt(this.ctx.start));
    for (let k = 0; k < 16; k++) if (this.shotLine(k).line === 3) return au.timeOfBeat(b0 + k * 4);
    return this.tm.l[3]!;
  }

  private applyPose(cam: THREE.PerspectiveCamera, p: Pose) {
    cam.position.copy(p.pos);
    cam.up.copy(p.up ?? new THREE.Vector3(0, 1, 0));
    cam.lookAt(p.look);
    if (cam.fov !== p.fov) { cam.fov = p.fov; cam.updateProjectionMatrix(); }
    cam.updateMatrixWorld();
  }

  private proj(v: THREE.Vector3): [number, number] {
    const p = v.clone().project(this.camL);
    return [(p.x * 0.5 + 0.5) * W, (-p.y * 0.5 + 0.5) * H];
  }

  /** Big two-ink type: blue underprint, pink on top (multiply prints purple where they overlap). */
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

  /** A solid label: ink (or pink) box with paper mono type. Returns its width. */
  private tag(c: CanvasRenderingContext2D, text: string, x: number, y: number, size = 26, bg = INK, align: 'left' | 'right' | 'center' = 'left') {
    c.save();
    c.font = font(F.mono(700), size);
    const w = c.measureText(text).width + size * 0.8;
    const x0 = align === 'left' ? x : align === 'right' ? x - w : x - w / 2;
    c.fillStyle = bg;
    c.fillRect(x0, y - size * 0.8, w, size * 1.3);
    c.fillStyle = PAPER;
    c.textBaseline = 'alphabetic';
    c.fillText(text, x0 + size * 0.4, y + size * 0.18);
    c.restore();
    return w;
  }

  /** Plain mono caption in solid ink. */
  private cap(c: CanvasRenderingContext2D, text: string, x: number, y: number, size = 24, col = INK, align: CanvasTextAlign = 'left', strip = true) {
    c.save();
    c.font = font(F.mono(700), size);
    if (strip) {
      const w = c.measureText(text).width, x0 = align === 'left' ? x : align === 'right' ? x - w : x - w / 2;
      c.fillStyle = PAPER;
      c.fillRect(x0 - 8, y - size * 0.95, w + 16, size * 1.3);
    }
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

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio, llm = this.ctx.llm;
    const sh = shot(this.ctx, f, { every: 4 });
    const { line, sub } = this.shotLine(sh.i);
    const P = portrait();
    const d = drive(au, f.t, 0.5, 1.6);
    const tm = this.tm;

    // ---------------------------------------------------------------- background: wash + rain
    const blueWash = (line + sub) % 2 === 1;
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: line === 1 ? 0.6 : 0.5, c1: blueWash ? 'blue' : 'pink', c2: blueWash ? 'pink' : 'blue' });
    const rainPool = line === 0 || line === 2 ? this.rainVals : llm.d.prompt;
    this.ctx.comp.draw(r, perFrame(this.back, f, () => tokenRain(this.back.ctx, rainPool, drive(au, f.ft, 0.5, 1.6) * 0.5, { seed: sh.seed, cols: P ? 8 : 13 })), out);

    // ---------------------------------------------------------------- hero (3D)
    this.code.visible = line === 0;
    this.cloud.visible = line === 1;
    this.stack.visible = line === 2;
    this.tower.group.visible = line === 3;

    if (line === 0) {
      // unroll on "vector": one bar per 64th of a beat-and-a-bit, left to right, each snapping in
      const t0 = sub === 0 ? this.unrollT() : -1e9;
      const dt = 0.3 / 64;
      this.setBars((i) => snapIn(f.t, t0 + i * dt, 0.07));
      const rib = sub === 0 ? ease.outExpo(clamp((f.t - (t0 + 0.3)) / 0.25)) : 1;
      this.ribbon.scale.x = Math.max(1e-3, rib);
      this.ribbon.visible = rib > 0.001;
      this.spine.visible = sub !== 0 || f.t >= t0;
      const k = snapIn(f.t, sh.t0, 0.08);
      this.heroTile.scale.setScalar(sub === 0 ? 0.2 + 0.8 * k : 1);
    }
    if (line === 1) {
      // highlighted pair: shot 1 walks the pairs on the beat; shot 2 lights the three place pairs
      const hot = this.hotPairs(sub, f.ft, sh);
      const hotWords = new Set(hot.flatMap((i) => [this.links[i]!.a, this.links[i]!.b]));
      for (const tl of this.tiles) {
        (tl.m.material as THREE.Material[])[4] = hotWords.has(tl.w) ? tl.hot : tl.face;
        tl.m.scale.setScalar(sub === 0 ? 1 : 0.46);
      }
      this.links.forEach((l, i) => { const on = hot.includes(i); l.m.scale.set(on ? 2.6 : 1, 1, on ? 2.6 : 1); });
    }
    if (line === 2) {
      const n = sub === 0 ? llm.d.question_tokens.length : this.slabs.length;
      const base = sub === 0 ? this.qStart : 0;
      const sixteenth = (au.timeOfBeat(1) - au.timeOfBeat(0)) / 4;
      this.slabs.forEach((m, i) => {
        const k = i - base;
        m.visible = k >= 0 && k < n;
        if (!m.visible) return;
        // shot 1: the question lands one slab per 16th; shot 2: the whole prompt, already standing
        const land = sub === 0 ? snapIn(f.t, sh.t0 + k * sixteenth, 0.08) : 1;
        m.visible = sub !== 0 || f.t >= sh.t0 + k * sixteenth;
        m.position.set(0, k * GAP + (1 - land) * 1.4, 0);
      });
    }
    if (line === 3) {
      const fl = this.craneFloor(f.t);
      const climb = fl / 33;
      this.tower.ride(this.rider, climb);
      this.tower.light(climb, 0.4);
    }

    const pose = this.pose(line, sub, f.t, sh);
    this.applyPose(this.cam, pose);
    if (line === 1) for (const tl of this.tiles) tl.m.quaternion.copy(this.cam.quaternion);
    this.applyPose(this.camL, this.pose(line, sub, f.ft, sh));
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);

    // ---------------------------------------------------------------- type (once per frame)
    this.ctx.comp.draw(r, perFrame(this.mid, f, () => this.drawType(f, sh, line, sub)), out);

    // ---------------------------------------------------------------- foreground: specks
    const c = this.front.ctx;
    this.front.clear();
    specks(c, d * (line === 3 ? 1.8 : 1), { seed: sh.seed, n: line === 3 ? 100 : 70 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    return { reg: sh.seed };
  }

  /** The bar code unrolls from the 8th a beat before "vector", so it is complete as the word lands. */
  private unrollT() { return this.q8(this.tm.vector - 0.32); }

  /** Nearest 8th-note grid time. */
  private q8(t: number) {
    const au = this.ctx.audio;
    return au.timeOfBeat(Math.round(au.beatAt(t) * 2) / 2);
  }

  /** Link indices lit in this shot (by beat in the wide shot). */
  private hotPairs(sub: number, ft: number, sh: Shot): number[] {
    const idx = (a: string) => this.links.findIndex((l) => l.a === a);
    if (sub === 0) {
      const beat = Math.floor(this.ctx.audio.beatAt(ft) - this.ctx.audio.beatAt(sh.t0) + 1e-3);
      const seq = [idx('king'), idx('man'), idx('Paris'), idx('GPU')];
      return [seq[clamp(beat, 0, 3)]!];
    }
    return [idx('Paris'), idx('London'), idx('Tokyo')];
  }

  private drawType(f: Frame, sh: Shot, line: number, sub: number) {
    const c = this.mid.ctx, ft = f.ft, P = portrait(), llm = this.ctx.llm, au = this.ctx.audio;
    const e = llm.d.embed, cfg = llm.cfg;
    const k = snapIn(ft, sh.t0, 0.08);
    const L = P ? 110 : 120, R = W - (P ? 70 : 90);

    if (line === 0) {
      const heroId = llm.d.prompt.find((p) => p.t === HERO)?.id ?? 0;
      if (sub === 0) {
        const unrolled = ft >= this.unrollT() + 0.3;
        // the count of real bars, at the end of the bar code
        const [ex, ey] = this.proj(new THREE.Vector3(64 * SP, BH * 0.9, 0));
        if (unrolled) {
          this.slab(c, `64 OF ${comma(cfg.hidden)}`, P ? W / 2 : W * 0.7, P ? H * 0.25 : H * 0.68, P ? 96 : 100, snapIn(ft, this.unrollT() + 0.3), 'center');
          const [rx, ry] = this.proj(new THREE.Vector3(64 * SP + 14, 0, BD / 2));
          const more = cfg.hidden - 64;
          if (rx > 0 && rx < W && ry > 160 && ry < H - 120) this.tag(c, `+${comma(more)} MORE NUMBERS →`, clamp(rx, L, R - 460), ry + 56, P ? 26 : 28, BLUE);
        }
        // what the bars are
        const y0 = P ? H * 0.74 : H * 0.8;
        this.tag(c, `${showTok(HERO)} · ID ${heroId}`, L, y0, P ? 30 : 32, PINK);
        this.cap(c, `INPUT EMBEDDING · PINK = +, BLUE = − · TALLEST ${fmt(this.vals.reduce((a, v) => (Math.abs(v) > Math.abs(a) ? v : a), 0))}`, L, y0 + (P ? 52 : 54), P ? 22 : 24);
      } else {
        // the vector, literally: its first 64 numbers
        const cols = 8, fs = P ? 24 : 26;
        c.font = font(F.mono(700), fs);
        const cw = c.measureText('−0.0000').width + fs * 0.9, rh = fs * 1.42;
        const bw = cols * cw, bh = 8 * rh;
        const x0 = P ? (W - bw) / 2 : L + 10, y0 = P ? H * 0.56 : H - 130 - bh;
        c.fillStyle = PAPER;
        c.fillRect(x0 - 24, y0 - fs * 2.6, bw + 48, bh + fs * 3.9);
        c.fillStyle = INK;
        c.fillRect(x0 - 24, y0 - fs * 2.6, bw + 48, 4);
        this.cap(c, `${showTok(HERO)} = [`, x0, y0 - fs * 1.1, fs, PINK);
        this.cap(c, `FIRST 64 OF ${comma(cfg.hidden)}`, x0 + bw, y0 - fs * 1.1, fs * 0.9, INK, 'right');
        const rows = Math.min(8, Math.floor((ft - sh.t0) / ((sh.t1 - sh.t0) / 16)) + 1);
        this.vals.forEach((v, i) => {
          const rr = Math.floor(i / cols);
          if (rr >= rows) return;
          c.font = font(F.mono(700), fs);
          c.fillStyle = v >= 0 ? PINK : BLUE;
          c.fillText(fmt(v), x0 + (i % cols) * cw, y0 + rr * rh + fs * 0.4);
        });
        if (rows >= 8) this.cap(c, `… ${comma(cfg.hidden - 64)} MORE ]`, x0, y0 + 8 * rh + fs * 0.35, fs, INK);
      }
    }

    if (line === 1) {
      // the lit pair(s): their real cosine similarity, at the pair
      const hot = this.hotPairs(sub, ft, sh);
      hot.forEach((li, j) => {
        const l = this.links[li]!;
        const A = this.tiles.find((x) => x.w === l.a)!.p, B = this.tiles.find((x) => x.w === l.b)!.p;
        const lab = `${l.a.toUpperCase()} · ${l.b.toUpperCase()}  COS ${l.cos.toFixed(3)}`;
        if (sub === 0) {
          // the pair lit on this beat, its cosine above it
          const top = A.y > B.y ? A : B;
          const [x, y] = this.proj(top.clone().add(new THREE.Vector3(0, 0.6, 0)));
          const [ax, ay] = this.proj(top);
          const size = P ? 30 : 36;
          const yy = clamp(y - 30, P ? 280 : 200, H - 260), xx = clamp(x, L + (P ? 250 : 300), R - (P ? 250 : 300));
          this.leader(c, ax, ay, xx, yy + size * 0.5, PINK);
          this.tag(c, lab, xx, yy, size, PINK, 'center');
        } else {
          // the three place pairs, one per beat, in a column with leaders to each pair
          const kk = snapIn(ft, sh.t0 + j * (au.timeOfBeat(1) - au.timeOfBeat(0)));
          if (kk <= 0) return;
          const mid = A.clone().add(B).multiplyScalar(0.5);
          const [ax, ay] = this.proj(mid);
          const size = P ? 28 : 30;
          const tx = P ? W / 2 : R, ty = P ? H * 0.665 + j * 60 : 230 + j * 66;
          c.save();
          c.font = font(F.mono(700), size);
          const w = c.measureText(lab).width + size * 0.8;
          c.restore();
          const lx = P ? tx - w / 2 : tx - w;
          this.leader(c, ax, ay, P ? tx : lx, P ? ty - size * 0.8 : ty - size * 0.3, PINK);
          this.tag(c, lab, tx, ty, size, PINK, P ? 'center' : 'right');
        }
      });
      // honest caption + an unrelated pair for scale
      const cy = P ? H * 0.78 : H - 200;
      const iK = e.words.indexOf('king'), iA = e.words.indexOf('apple');
      this.tag(c, 'PCA OF INPUT EMBEDDINGS', L, cy, P ? 26 : 28, INK);
      this.cap(c, `${e.words.length} WORDS · ${comma(cfg.hidden)} DIMENSIONS SQUASHED TO 3`, L, cy + 46, P ? 22 : 24);
      if (iK >= 0 && iA >= 0) this.cap(c, `UNRELATED: KING · APPLE COS ${e.cos[iK]![iA]!.toFixed(3)}`, L, cy + 80, P ? 22 : 24, BLUE);
    }

    if (line === 2) {
      const n = sub === 0 ? llm.d.question_tokens.length : llm.d.prompt.length;
      const total = n * cfg.hidden;
      const x = P ? W / 2 : W * 0.75, y = P ? (sub === 0 ? H * 0.725 : H * 0.22) : H * 0.5;
      const kk = sub === 0 ? snapIn(ft, this.q8(this.tm.stack)) : k;
      if (sub === 0) {
        const landed = Math.min(n, Math.floor((ft - sh.t0) / ((au.timeOfBeat(1) - au.timeOfBeat(0)) / 4)) + 1);
        this.slab(c, `${landed} × ${comma(cfg.hidden)}`, x, y - (P ? 0 : 40), P ? 92 : 104, snapIn(ft, sh.t0), 'center', 100);
        if (kk > 0) this.cap(c, `= ${comma(landed * cfg.hidden)} NUMBERS`, x, y + (P ? 80 : 50), P ? 30 : 34, INK, 'center');
        this.cap(c, 'THE QUESTION, ONE SLAB PER TOKEN', x, y + (P ? 124 : 96), P ? 22 : 24, BLUE, 'center');
      } else {
        this.slab(c, `${n} × ${comma(cfg.hidden)}`, x, y - (P ? 0 : 40), P ? 92 : 104, k, 'center', 100);
        this.cap(c, `= ${comma(total)} NUMBERS`, x, y + (P ? 80 : 50), P ? 30 : 34, INK, 'center');
        this.cap(c, 'THE WHOLE CHAT PROMPT · PINK = THE QUESTION', x, y + (P ? 124 : 96), P ? 22 : 24, BLUE, 'center');
      }
    }

    if (line === 3) {
      // the floor count, riding next to the tile
      const fl = this.craneFloor(ft);
      const n = clamp(Math.floor(fl + 0.5), 0, 33);
      const lab = n === 0 ? 'LOBBY' : n > 32 ? 'ROOF' : `L${n}`;
      const [x, y] = this.proj(new THREE.Vector3(0, this.tower.floorY(fl) + 0.25, 0));
      this.tag(c, lab, clamp(x + (P ? 110 : 150), L, R - 160), clamp(y + 12, 250, H - 200), P ? 44 : 50, PINK);
      const cx = P ? W / 2 : W * 0.79, cy = P ? H * 0.76 : H * 0.42;
      this.slab(c, `${String(Math.min(32, n)).padStart(2, '0')}/${cfg.layers}`, cx, cy, P ? 120 : 140, 1, 'center', 100);
      this.cap(c, n > 32 ? 'ALL FLOORS CLIMBED' : 'FLOORS CLIMBED', cx, cy + (P ? 96 : 110), P ? 26 : 28, INK, 'center');
    }
  }
}
