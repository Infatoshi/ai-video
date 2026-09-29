// A tensor core computes a whole 16×16 tile of results in one instruction; a normal CUDA core works one result at a time.
//
// `tensor` (v5, CH7, the bridge: a drop): "Tensor cores eat a matrix whole, / Sixteen by sixteen in a single go,".
// Inside one SM of the shared H100 model (_h100.ts), partition 2's 4th-gen tensor core (its 4×4×4 MAC lattice)
// fills the frame. The multiply is drawn as the classic matrix cube: A (16×16, the lattice's front face) and B
// (16×16, its left face) slide in, the lattice fires, and C (16×16) appears on top: all 256 results at once.
// Beside it one FP32 lane of the same partition (a CUDA core) flashes once per 8th and fills its own 16×16
// tile one result at a time; it never gets far.
//   entrance (streak, from 129.30): the camera dives from the whole SM onto the tensor core
//   "Tensor": the lattice powers on, label        "eat": A and B slide in        "matrix": their cells load
//   "whole": the lattice fires in one hit, C appears, "256 / 256 results · 1 instruction"
//   "Sixteen" / "sixteen": C's two edges light as dimension bars "16"   "single": a second full hit, "= 1 instruction"
//   "go": CH07's H100 numbers (67 TFLOP/s on CUDA cores vs 989.5 TFLOP/s on tensor cores)
// The camera orbits the whole time (a base drift plus the drums), with its big moves landing on downbeats.
// Every step starts on the 8th-note grid (never more than 60 ms before the sung word) and snaps in; then holds.
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font, measure } from '../engine/type';
import type { Line, Word } from '../engine/lyrics';
import { clamp, ease, lerp } from '../engine/util';
import { H100, glowMat, type SMDetail } from './_h100';
import { beatEase, drive, whip } from './_cam';
import { hit, snap } from './_lock';

type Ctx2 = CanvasRenderingContext2D;
type V3 = THREE.Vector3;

const SM = 58;          // a mid-die SM site (GPC 3), as in the model preview
const P = 2;            // partition (south-west): its tensor core faces the camera
const LANE = 17;        // the FP32 lane that plays the CUDA core (2nd FP32 row, 2nd column: just north-west of the lattice)
const N = 16;
const LYRIC_X = 180, LYRIC_Y = 972, LYRIC_SIZE = 74;

const lin = (k: keyof typeof LIN, s = 1): [number, number, number] => [LIN[k][0] * s, LIN[k][1] * s, LIN[k][2] * s];

/** A 16×16 tile of cells in SM-local space: solid cells plus additive glow twins. */
class Tile {
  group = new THREE.Group();
  solid: THREE.InstancedMesh;
  glow: THREE.InstancedMesh;
  private col = new THREE.Color();
  /** `center` of the tile, `u`/`v` span vectors (full edge), `n` thickness vector. */
  constructor(center: V3, u: V3, v: V3, n: V3, base = 1.6, matte = false) {
    const geo = new THREE.BoxGeometry(1, 1, 1);
    // matte tiles (the result tiles) stay dark under the key light so their glow reads as the cell's colour
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(...lin('ink2', base)), metalness: matte ? 0 : 0.4, roughness: matte ? 0.95 : 0.45, envMapIntensity: matte ? 0.04 : 0.25 });
    this.solid = new THREE.InstancedMesh(geo, mat, N * N);
    this.glow = new THREE.InstancedMesh(geo, glowMat(), N * N);
    // a basis whose axes are u, v, n: cells are 78% of the pitch, the tile is n thick
    const ux = u.clone().divideScalar(N), vx = v.clone().divideScalar(N);
    const m = new THREE.Matrix4();
    for (let r = 0; r < N; r++) for (let q = 0; q < N; q++) {
      const pos = center.clone().addScaledVector(u, (q + 0.5) / N - 0.5).addScaledVector(v, (r + 0.5) / N - 0.5);
      m.makeBasis(ux.clone().multiplyScalar(0.78), vx.clone().multiplyScalar(0.78), n.clone());
      m.setPosition(pos);
      this.solid.setMatrixAt(r * N + q, m);
      const g = m.clone();
      g.makeBasis(ux.clone().multiplyScalar(0.86), vx.clone().multiplyScalar(0.86), n.clone().multiplyScalar(1.4));
      g.setPosition(pos);
      this.glow.setMatrixAt(r * N + q, g);
      this.glow.setColorAt(r * N + q, this.col.setRGB(0, 0, 0));
    }
    this.solid.frustumCulled = false;
    this.glow.frustumCulled = false;
    // hairline outline of the whole tile
    const c = center, hu = u.clone().multiplyScalar(0.5), hv = v.clone().multiplyScalar(0.5);
    const corners = [c.clone().sub(hu).sub(hv), c.clone().add(hu).sub(hv), c.clone().add(hu).add(hv), c.clone().sub(hu).add(hv)];
    const pts: number[] = [];
    for (let i = 0; i < 4; i++) { const a = corners[i]!, b = corners[(i + 1) % 4]!; pts.push(a.x, a.y, a.z, b.x, b.y, b.z); }
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const edge = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: new THREE.Color().setRGB(...lin('bone')), transparent: true, opacity: 0.55, depthWrite: false }));
    this.group.add(this.solid, this.glow, edge);
  }
  /** fn(row, col) → [r, g, b] linear HDR glow (0 = unlit). */
  set(fn: (r: number, q: number) => [number, number, number]) {
    for (let r = 0; r < N; r++) for (let q = 0; q < N; q++) {
      const [a, b, c] = fn(r, q);
      this.glow.setColorAt(r * N + q, this.col.setRGB(a, b, c));
    }
    this.glow.instanceColor!.needsUpdate = true;
  }
}

export default class Tensor extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(34, W / H, 0.01, 500);
  gpu!: H100;
  det!: SMDetail;
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();

  // SM-local geometry of partition P's tensor core lattice
  lc = new THREE.Vector3();      // lattice centre
  half = 0;                      // half-extent in x/z (2 cells)
  hy = 0;                        // half-height of the lattice
  laneTop = new THREE.Vector3(); // top of the CUDA-core lane
  A!: Tile; B!: Tile; C!: Tile; K!: Tile;
  aGroup = new THREE.Group(); bGroup = new THREE.Group();
  dimX!: THREE.Mesh; dimZ!: THREE.Mesh;
  kc = new THREE.Vector3();      // centre of the CUDA core's tile
  size = 0;                      // tile edge
  shot!: THREE.Mesh;             // the CUDA core's one result, flying up to its cell

  L1!: Line; L2!: Line;
  tTensor = 0; tEat = 0; tMatrix = 0; tWhole = 0; tSix1 = 0; tSix2 = 0; tSingle = 0; tGo = 0;
  db0 = 0; db1 = 0; db2 = 0; db3 = 0;   // the bridge's downbeats
  stress = new Set<Word>();

  override async init() {
    const { lyrics, audio: au } = this.ctx;
    this.gpu = new H100({ renderer: this.ctx.renderer });
    const rig = H100.rig();
    // the orbit looks toward the shared ember rim: at 2.2 its specular hazes the top of frame
    rig.children.forEach((l) => { if (l instanceof THREE.DirectionalLight && l.position.z < 0) l.intensity = 0.6; });
    this.scene.add(this.gpu.group, rig);
    this.scene.environment = this.gpu.env;
    this.scene.background = new THREE.Color().setRGB(...lin('ink'));
    this.det = this.gpu.smDetail(SM);

    // ---- words and their grid points
    this.L1 = lyrics.get('Tensor cores eat');
    this.L2 = lyrics.get('Sixteen by sixteen');
    const w1 = this.L1.words, w2 = this.L2.words;
    const g = (w: Word) => { this.stress.add(w); return this.grid(w.start); };
    this.tTensor = g(w1[0]!); this.tEat = g(w1[2]!); this.tMatrix = g(w1[4]!); this.tWhole = g(w1[5]!);
    this.tSix1 = g(w2[0]!); this.tSix2 = g(w2[2]!); this.tSingle = g(w2[5]!); this.tGo = g(w2[6]!);
    const dbs = au.downbeats.filter((d) => d >= this.L1.start - 0.5);
    [this.db0, this.db1, this.db2, this.db3] = [dbs[0]!, dbs[1]!, dbs[2]!, dbs[3]!];

    // ---- the lattice, exactly as _h100-sm.ts lays it out (4 × 4 × 4 cubes of pitch cs, y pitch 0.9 cs)
    const tb = this.det.boxes[`p${P}.tensor`]!;
    const tcx0 = tb.x0 + 0.015, tcx1 = tb.x1 - 0.015, tcz0 = tb.z0 + 0.01, tcz1 = tb.z1 - 0.01;
    const cs = Math.min((tcx1 - tcx0) / 4, (tcz1 - tcz0) / 4);
    const yb = 0.006 + tb.h;
    this.half = 2 * cs;
    this.hy = 2 * cs * 0.9;
    this.lc.set((tcx0 + tcx1) / 2, yb + this.hy, (tcz0 + tcz1) / 2);
    this.laneTop.copy(this.det.lanePos(P, 'fp32', LANE));

    const s = this.half * 2, sy = this.hy * 2, gap = cs * 0.55, th = cs * 0.12;
    const X = new THREE.Vector3(1, 0, 0), Yv = new THREE.Vector3(0, 1, 0), Z = new THREE.Vector3(0, 0, 1);
    // A (M × K): the front face (x–y plane, facing +z); B (K × N): the left face (y–z plane, facing −x);
    // C (M × N): on top (x–z plane). The CUDA core's tile floats over its lane at C's height.
    // (Every basis is right-handed: instance matrices with a negative determinant would render inside out.)
    const nZ = Z.clone().negate();
    this.A = new Tile(new THREE.Vector3(0, 0, 0), X.clone().multiplyScalar(s), Yv.clone().multiplyScalar(sy), Z.clone().multiplyScalar(th));
    this.B = new Tile(new THREE.Vector3(0, 0, 0), Z.clone().multiplyScalar(s), Yv.clone().multiplyScalar(sy), X.clone().multiplyScalar(-th));
    this.aGroup.add(this.A.group); this.bGroup.add(this.B.group);
    const cy = this.lc.y + this.hy + gap;
    this.C = new Tile(new THREE.Vector3(this.lc.x, cy, this.lc.z), X.clone().multiplyScalar(s), nZ.clone().multiplyScalar(s), Yv.clone().multiplyScalar(th), 0.35, true);
    this.kc.set(this.laneTop.x - cs * 0.4, cy, this.laneTop.z - cs * 0.5);
    this.size = s;
    this.K = new Tile(this.kc, X.clone().multiplyScalar(s), nZ.clone().multiplyScalar(s), Yv.clone().multiplyScalar(th), 0.35, true);
    this.shot = new THREE.Mesh(new THREE.BoxGeometry(s / N * 0.9, th * 2, s / N * 0.9), glowMat(0x000000));
    this.shot.frustumCulled = false;
    // dimension bars along C's front edge (x) and left edge (z)
    const bar = (len: number, axis: 'x' | 'z') => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(axis === 'x' ? len : th * 1.6, th * 1.6, axis === 'z' ? len : th * 1.6), glowMat(0x000000));
      m.frustumCulled = false;
      return m;
    };
    this.dimX = bar(s, 'x'); this.dimZ = bar(s, 'z');
    this.dimX.position.set(this.lc.x, cy, this.lc.z + this.half + gap * 0.9);
    this.dimZ.position.set(this.lc.x - this.half - gap * 0.9, cy, this.lc.z);
    this.det.group.add(this.aGroup, this.bGroup, this.C.group, this.K.group, this.dimX, this.dimZ, this.shot);
    this.scene.updateMatrixWorld(true);
  }

  /** First 8th-note grid point at or after (t − 60 ms): steps never run more than 60 ms ahead of the voice. */
  grid(t: number): number {
    const au = this.ctx.audio;
    return au.timeOfBeat(Math.ceil(au.beatAt(t - 0.06) * 2) / 2);
  }

  /** The CUDA core's progress: results done (the current one included) and the phase 0..1 within its 8th. */
  cuda(t: number): { n: number; ph: number } {
    const au = this.ctx.audio;
    const b = au.beatAt(t) * 2 - Math.round(au.beatAt(this.db0) * 2);
    if (b < 0) return { n: 0, ph: 0 };
    return { n: Math.min(N * N, Math.floor(b) + 1), ph: b - Math.floor(b) };
  }

  /** SM-local centre of the CUDA core tile's cell i (filled row by row from the south-west corner, nearest the camera). */
  kCell(i: number): V3 {
    const r = Math.floor(i / N), q = i % N;   // row 0 is the south edge (the tile's v axis points north)
    return new THREE.Vector3(this.kc.x + this.size * ((q + 0.5) / N - 0.5), this.kc.y, this.kc.z - this.size * ((r + 0.5) / N - 0.5));
  }

  /** SM-local point → world. */
  W(p: V3): V3 { return this.det.group.localToWorld(p.clone()); }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp, audio: au } = this.ctx;
    const t = f.t;
    this.gpu.update(t, f.a);
    this.det = this.gpu.smDetail(SM);
    this.animate(t);
    const dist = this.camera(t);
    H100.fitClip(this.cam, dist);

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, this.cam);
    comp.draw(renderer, this.ms.texture, out);

    const c = this.ui.ctx;
    this.ui.clear();
    this.drawLabels(c, t);
    this.drawLyric(c, t);
    comp.draw(renderer, this.ui.upload(), out);
    const hWhole = hit(au, t, this.tWhole, 0.12), hSingle = hit(au, t, this.tSingle, 0.12);
    return { hud: 1, bloom: 0.6 + 0.35 * Math.max(hWhole, hSingle), bloomThreshold: 0.9, halation: 0.05, vignette: 0.28, grain: 0.05, ca: 0.3, exposure: 1 + 0.1 * Math.max(hWhole, hSingle) };
  }

  // ------------------------------------------------------------------ the model's state at t
  animate(t: number) {
    const au = this.ctx.audio;
    const d = this.det;
    const kOn = snap(au, t, this.tTensor);
    const kEat = snap(au, t, this.tEat, 0.14);
    const kMat = snap(au, t, this.tMatrix);
    const kWhole = snap(au, t, this.tWhole);
    const hWhole = hit(au, t, this.tWhole, 0.16), hSingle = hit(au, t, this.tSingle, 0.16);
    const kSix1 = snap(au, t, this.tSix1), kSix2 = snap(au, t, this.tSix2);

    // the CUDA core: one multiply-add per 8th since the bridge's first downbeat; each result flies
    // up to its cell in the first ~90 ms of its 8th, then the cell holds
    const { n: done, ph } = this.cuda(t);
    const flash = done > 0 ? Math.pow(1 - ph, 2) : 0;
    const fly = ease.outExpo(clamp(ph / 0.4));
    d.setLanes((p, kind, idx) => {
      if (p === P && kind === 'fp32' && idx === LANE) return 0.3 + 1.6 * flash;
      return p === P ? 0.05 : 0.025;
    }, LIN.signal, 1.3);
    this.K.set((r, q) => {
      const i = r * N + q;
      if (i < done - 1) return lin('bone', 1.1);
      if (i === done - 1) return lin('signal', fly * (1.4 + 1.2 * flash));
      return [0, 0, 0];
    });
    const cell = this.kCell(Math.max(0, done - 1));
    this.shot.visible = done > 0 && fly < 0.999;
    this.shot.position.copy(this.laneTop).lerp(cell, fly);
    (this.shot.material as THREE.MeshBasicMaterial).color.setRGB(...lin('signal', 2.2));

    // the tensor cores: partition P's lattice powers on, then fires twice; the others idle dim
    d.setTensor((p, i, j, k) => {
      if (p !== P) return 0.08;
      const wave = kMat > 0 && kWhole <= 0 ? 0.25 + 0.2 * (((i + j + k + Math.floor(t * 8)) % 4) === 0 ? 1 : 0) : 0;
      return 0.12 * kOn + wave + kWhole * (0.45 + 1.5 * hWhole) + 1.3 * hSingle;
    }, LIN.ember, 1.5);
    d.setUnits((u, p) => (p === P && u === 'tensor' ? 0.06 * kOn + 0.12 * Math.max(hWhole, hSingle) : 0));

    // A and B slide in from outside the frame on "eat", load on "matrix", flash and dim on "whole"
    const far = 0.34;
    const aZ = this.lc.z + this.half + this.half * 0.28 + far * (1 - kEat);
    const bX = this.lc.x - this.half - this.half * 0.28 - far * (1 - kEat);
    this.aGroup.position.set(this.lc.x, this.lc.y, aZ);
    this.bGroup.position.set(bX, this.lc.y, this.lc.z);
    this.aGroup.visible = this.bGroup.visible = kEat > 0;
    const data = (r: number, q: number, seed: number) => 0.25 + 0.75 * (((r * 7 + q * 13 + seed) % 5) / 4);
    const load = kMat * (1 - 0.55 * kWhole) + 0.9 * hWhole + 0.5 * hSingle;
    this.A.set((r, q) => lin('bone', 0.42 * load * data(r, q, 1)));
    this.B.set((r, q) => lin('bone', 0.42 * load * data(r, q, 3)));

    // C: all 256 results at once on "whole", flaring again on "single"
    this.C.group.visible = kWhole > 0;
    this.C.group.position.y = (1 - kWhole) * -this.hy;   // rises out of the lattice top in the snap
    const cg = kWhole * (1.5 + 1.4 * hWhole + 1.0 * hSingle);
    this.C.set((r, q) => lin('signal', cg * (0.8 + 0.2 * data(r, q, 5))));

    // dimension bars
    const setBar = (m: THREE.Mesh, k: number) => {
      m.visible = k > 0;
      (m.material as THREE.MeshBasicMaterial).color.setRGB(...lin('bone', 1.2 * k));
    };
    setBar(this.dimX, kSix1);
    setBar(this.dimZ, kSix2);
  }

  // ------------------------------------------------------------------ the camera
  /** Orbit around the tensor core (with the CUDA core beside it during line 1). Returns the distance. */
  camera(t: number): number {
    const au = this.ctx.audio;
    const lat = this.W(new THREE.Vector3(this.lc.x, this.lc.y + this.hy * 0.9, this.lc.z));
    const both = this.W(new THREE.Vector3(this.lc.x, this.lc.y + this.hy, this.lc.z).lerp(this.kc, 0.42));
    // dive in: from the whole SM down onto the tensor core, log-spaced, from just before the cut to "Tensor"
    const kIn = ease.outCubic(clamp((t - (this.db0 - 0.2)) / (this.tTensor - this.db0 + 0.2)));
    const toL2 = snap(this.ctx.audio, t, this.tSix1, 0.3);             // reframe onto the tensor core alone on "Sixteen"
    const out = beatEase(au, t, this.db3 + 0.45, this.ctx.end, ease.inOutCubic); // pull back out, landing on the cut into `flash`
    // each line creeps in by 10% while it plays
    const p1 = clamp((t - this.tTensor) / (this.tSix1 - this.tTensor)), p2 = clamp((t - this.tSix1) / (this.db3 + 0.45 - this.tSix1));
    let dist = Math.exp(lerp(Math.log(4.5), Math.log(0.52), kIn)) * (1 - 0.1 * p1);
    dist = lerp(dist, 0.38 * (1 - 0.1 * p2), toL2);
    dist *= 1 - 0.07 * hit(au, t, this.tWhole, 0.2) - 0.06 * hit(au, t, this.tSingle, 0.2);
    dist = lerp(dist, 1.2, out);
    // orbit from the south-west (A and B faces both in view): a steady drift plus the drums, and a whip on
    // each downbeat that swings 20° and punches the dolly out and back while it travels
    const w1 = whip(au, t, this.db1, 0.3), w2 = whip(au, t, this.db2, 0.35), w3 = whip(au, t, this.db3, 0.3);
    const az = -0.8 + 0.09 * drive(au, t, 1, 1.2, this.ctx.start) - 0.35 * w1 + 0.25 * w2 - 0.35 * w3;
    dist *= 1 + 0.6 * (w1 * (1 - w1) + w2 * (1 - w2) + w3 * (1 - w3));
    const el = lerp(lerp(1.25, 0.66, kIn), 0.52, toL2) + 0.25 * out;
    const tgt = both.lerp(lat, toL2);
    this.cam.position.set(tgt.x + Math.sin(az) * Math.cos(el) * dist, tgt.y + Math.sin(el) * dist, tgt.z + Math.cos(az) * Math.cos(el) * dist);
    this.cam.up.set(0, 1, 0);
    this.cam.lookAt(tgt);
    // the target sits at 40% height so the hero clears the lyric band
    this.cam.setViewOffset(W, H, 0, H * 0.1, W, H);
    this.cam.updateMatrixWorld();
    return dist;
  }

  // ------------------------------------------------------------------ screen-space labels anchored to 3D points
  project(p: V3): { x: number; y: number; ok: boolean } {
    const v = this.W(p).project(this.cam);
    return { x: (v.x * 0.5 + 0.5) * W, y: (-v.y * 0.5 + 0.5) * H, ok: v.z < 1 && Math.abs(v.x) < 1.2 && Math.abs(v.y) < 1.2 };
  }

  drawLabels(c: Ctx2, t: number) {
    const au = this.ctx.audio;
    const kOn = snap(au, t, this.tTensor), kWhole = snap(au, t, this.tWhole);
    const kSix1 = snap(au, t, this.tSix1), kSix2 = snap(au, t, this.tSix2), kSingle = snap(au, t, this.tSingle), kGo = snap(au, t, this.tGo);
    const toL2 = snap(au, t, this.tSix1, 0.3);
    const titleF = F.archivo(100, 800), subF = F.archivo(100, 600), bigF = F.archivo(100, 900);
    c.shadowColor = rgba('ink', 0.85); c.shadowBlur = 14;
    /** a soft dark patch behind a text block (not a panel: blurred edges, no border) */
    const plate = (x: number, y: number, w: number, h: number, a: number) => {
      c.save(); c.shadowBlur = 0; c.filter = 'blur(22px)';
      c.fillStyle = rgba('ink', 0.55 * a); c.fillRect(x - 24, y - 20, w + 48, h + 40);
      c.restore();
    };
    const leader = (ax: number, ay: number, bx: number, by: number, a: number) => {
      c.save(); c.shadowBlur = 0;
      c.strokeStyle = rgba('bone', 0.6 * a); c.lineWidth = 2;
      c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
      c.fillStyle = rgba('signal', a); c.fillRect(ax - 4, ay - 4, 8, 8);
      c.restore();
    };

    // the CUDA core (line 1): label up and to the left of its lane
    const aK = kOn * (1 - toL2);
    if (aK > 0.01) {
      const p = this.project(this.laneTop);
      if (p.ok) {
        const tw = Math.max(measure('CUDA core', titleF, 50), measure('one result at a time', subF, 32));
        const x = clamp(p.x - 120 - tw, 60, W - 60 - tw), y = clamp(p.y + 60, 130, 780);
        plate(x, y - 46, tw, 104, aK);
        leader(p.x, p.y, x + tw + 12, y - 14, aK);
        c.globalAlpha = aK;
        c.font = font(titleF, 50); c.fillStyle = rgba('bone');
        c.fillText('CUDA core', x, y);
        c.font = font(subF, 32); c.fillStyle = rgba('bone', 0.8);
        c.fillText('one result at a time', x, y + 46);
        c.globalAlpha = 1;
      }
    }

    // the tensor core: label to the right of C's east edge; "the whole tile at once" on "whole", "= 1 instruction" on "single"
    if (kOn > 0) {
      const p = this.project(new THREE.Vector3(this.lc.x + this.half, kWhole > 0 ? this.dimX.position.y : this.lc.y + this.hy, this.lc.z - this.half * 0.5));
      if (p.ok) {
        const tw = Math.max(measure('Tensor core', titleF, 56), measure('= 1 instruction', bigF, 64));
        const x = clamp(p.x + 110, 60, W - 60 - tw), y = clamp(p.y - 120, 130, 700);
        plate(x, y - 50, tw, kSingle > 0 ? 196 : kWhole > 0 ? 114 : 60, kOn);
        leader(p.x, p.y, x - 12, y - 16, kOn);
        c.globalAlpha = kOn;
        c.font = font(titleF, 56); c.fillStyle = rgba('bone');
        c.fillText('Tensor core', x, y);
        if (kWhole > 0) {
          c.globalAlpha = kWhole;
          c.font = font(subF, 34); c.fillStyle = rgba('signal');
          c.fillText('the whole tile at once', x, y + 50);
        }
        if (kSingle > 0) {
          c.globalAlpha = kSingle;
          c.font = font(bigF, 64); c.fillStyle = rgba('signal');
          c.fillText('= 1 instruction', x, y + 128);
        }
        c.globalAlpha = 1;
      }
    }

    // "16" on each dimension bar, pushed outward from C's centre
    const pc = this.project(new THREE.Vector3(this.lc.x, this.dimX.position.y, this.lc.z));
    const dimLabel = (m: THREE.Mesh, k: number) => {
      if (k <= 0) return;
      const p = this.project(m.position);
      if (!p.ok) return;
      const dx = p.x - pc.x, dy = p.y - pc.y, l = Math.max(1, Math.hypot(dx, dy));
      const w = measure('16', bigF, 80);
      c.globalAlpha = k;
      c.font = font(bigF, 80); c.fillStyle = rgba('bone');
      c.fillText('16', p.x + (dx / l) * 120 - w / 2, p.y + (dy / l) * 110 + 28);
      c.globalAlpha = 1;
    };
    dimLabel(this.dimX, kSix1);
    dimLabel(this.dimZ, kSix2);

    // "go": CH07's two H100 numbers, top left
    if (kGo > 0) {
      plate(LYRIC_X, 72, measure('989.5 TFLOP/s', bigF, 50) + 18 + measure('on tensor cores', subF, 32), 158, kGo);
      c.globalAlpha = kGo;
      const row = (num: string, what: string, y: number, col: 'bone' | 'signal') => {
        c.font = font(bigF, 50); c.fillStyle = rgba(col);
        c.fillText(num, LYRIC_X, y);
        c.font = font(subF, 32); c.fillStyle = rgba('bone', 0.85);
        c.fillText(what, LYRIC_X + measure(num, bigF, 50) + 18, y);
      };
      row('67 TFLOP/s', 'on CUDA cores', 120, 'bone');
      row('989.5 TFLOP/s', 'on tensor cores', 184, 'signal');
      c.font = font(F.mono(500), 22); c.fillStyle = rgba('bone', 0.7);
      c.fillText('H100  ·  CH07', LYRIC_X, 224);
      c.globalAlpha = 1;
    }
    c.shadowBlur = 0; c.shadowColor = 'transparent';
  }

  // ------------------------------------------------------------------ lyric: one line, fixed place, per-word
  drawLyric(c: Ctx2, t: number) {
    const line = t < this.L2.words[0]!.start - 0.4 ? this.L1 : this.L2;
    // a dark band under the lyric so it reads over the lit model
    const g = c.createLinearGradient(0, 790, 0, H);
    g.addColorStop(0, rgba('ink', 0)); g.addColorStop(0.55, rgba('ink', 0.62)); g.addColorStop(1, rgba('ink', 0.8));
    c.fillStyle = g; c.fillRect(0, 790, W, H - 790);
    const fam = F.archivo(100, 800);
    c.font = font(fam, LYRIC_SIZE);
    const space = measure(' ', fam, LYRIC_SIZE);
    let x = LYRIC_X;
    for (const w of line.words) {
      const k = clamp((t - w.start) / 0.06);   // each word lights exactly at its start, snapping in 60 ms
      const lit = this.stress.has(w) ? 'signal' : 'bone';
      c.fillStyle = k > 0 ? rgba(lit, lerp(0.3, 1, k)) : rgba('bone', 0.26);
      c.fillText(w.w, x, LYRIC_Y);
      x += measure(w.w, fam, LYRIC_SIZE) + space;
    }
  }
}
