// Each thread handles one number, and finds it with block × size + thread.
//
// SCENE `index` (v5, CH2), in 3D: we're zoomed into memory. A row of 16 memory cells (dark silicon
// blocks, bone hairlines, each holding a number on its top) recedes in perspective and the camera
// tracks along it the whole time; bit lines on the floor carry data pulses.
// "One thread for every number you've got,": on "One" the first thread (an orange spark) drops onto
// cell 0 ("1 thread"); from "every" the other 15 drop in, four per 8th note, one per cell, each
// reading its number down a beam; on "got" all 16 pulse ("16 numbers · 16 threads · one each").
// "Block times size plus thread finds its spot": the camera rises to take in the whole array as the
// row splits on "Block" into 4 blocks of 4 (they physically slide apart onto block plates), and the
// formula assembles as big screen-space type word by word: "i = block" · "× size" (size = 4
// bracketed over block 2) · "+ thread" (block 2's thread 1 brightens, the rest dim) · on "spot" cell 9
// fires and the camera whips onto it: i = 2 × 4 + 1 = 9, with the real CUDA line as a CH02 footnote.
// Motion: the camera never stops (scenes/_cam.ts drive/orbit, big moves landing on beats); every
// diagram change is a beat-locked snap (scenes/_lock.ts).
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font, layout, type TextLayout } from '../engine/type';
import { Lyrics, norm, type Line } from '../engine/lyrics';
import { clamp, ease, lerp } from '../engine/util';
import { hit, q, snap } from './_lock';
import { drive, kickPush } from './_cam';
import { H100, Pulses, glowMat, type PulseItem } from './_h100';
import { makeSramTexture } from './_h100-tex';

// ------------------------------------------------------------------ the array (world units ≈ mm)
const N = 16, SIZE = 4, NB = N / SIZE;
const PITCH = 11, CW = 10, CH = 5, BGAP = 10;       // cell pitch, cell width/height, extra gap between blocks
const HOVER = CH + 3.4;                              // where a thread sits above its cell
const BRACKET_Y = CH + 13;                           // the "size = 4" bracket over block 2
/** What sits in the array doesn't matter; each cell just holds a number. */
const VALUES = [7, 2, 9, 4, 1, 8, 3, 6, 5, 0, 9, 2, 4, 7, 1, 3];
/** The worked example: block 2, thread 1 → i = 2 × 4 + 1 = 9. */
const EX = { b: 2, th: 1 };
const EXI = EX.b * SIZE + EX.th;
const cellX = (i: number, kSplit: number) => (i - 7.5) * PITCH + kSplit * (Math.floor(i / SIZE) - 1.5) * BGAP;
const blockX = (b: number, kSplit: number) => (cellX(b * SIZE, kSplit) + cellX(b * SIZE + SIZE - 1, kSplit)) / 2;

// ------------------------------------------------------------------ screen-space type
const LY = { x: 150, y: 196, size: 78 };
const FORM_Y = 872, SUB_Y = 952, FOOT_Y = 1022, FORM_SIZE = 80;
const FORM = ['i =', 'block', '×', 'size', '+', 'thread'] as const;
const famLy = F.archivo(100, 800);
const mono = (w = 400) => F.mono(w);

const lin = (k: keyof typeof LIN, s = 1) => new THREE.Color().setRGB(LIN[k][0] * s, LIN[k][1] * s, LIN[k][2] * s, THREE.LinearSRGBColorSpace);

let ENV: THREE.Texture | null = null;
let HALO: THREE.CanvasTexture | null = null;
const DIGITS: THREE.CanvasTexture[] = [];

/** A soft radial glow (RGB on black, for additive blending). */
function haloTex(): THREE.CanvasTexture {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 128;
  const c = cv.getContext('2d')!;
  c.fillStyle = '#000'; c.fillRect(0, 0, 128, 128);
  const g = c.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgb(255,255,255)'); g.addColorStop(0.18, 'rgb(150,150,150)');
  g.addColorStop(0.5, 'rgb(28,28,28)'); g.addColorStop(1, 'rgb(0,0,0)');
  c.fillStyle = g; c.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** A cell top: dark silicon with an engraved rim and its number in bone. */
function digitTex(d: number): THREE.CanvasTexture {
  const S = 256;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const c = cv.getContext('2d')!;
  c.fillStyle = '#131315'; c.fillRect(0, 0, S, S);
  c.strokeStyle = 'rgba(238,233,223,0.35)'; c.lineWidth = 3; c.strokeRect(14, 14, S - 28, S - 28);
  c.strokeStyle = 'rgba(238,233,223,0.12)'; c.lineWidth = 1;
  for (let k = 1; k < 8; k++) { c.beginPath(); c.moveTo(14, 14 + k * 28.5); c.lineTo(30, 14 + k * 28.5); c.stroke(); }
  c.fillStyle = '#EEE9DF';
  c.font = font(F.archivo(100, 700), 176);
  c.textAlign = 'center'; c.textBaseline = 'middle';
  c.fillText(String(d), S / 2, S / 2 + 10);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

interface Cell { group: THREE.Group; top: THREE.MeshStandardMaterial; edge: THREE.LineBasicMaterial }
interface Thread { core: THREE.Mesh; halo: THREE.Sprite; beam: THREE.Mesh; coreMat: THREE.MeshBasicMaterial; haloMat: THREE.SpriteMaterial; beamMat: THREE.MeshBasicMaterial }

export default class IndexScene extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(36, W / H, 0.5, 3000);
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();
  cells: Cell[] = [];
  threads: Thread[] = [];
  plates: { mesh: THREE.Mesh; edge: THREE.LineBasicMaterial }[] = [];
  bracket!: THREE.LineSegments;
  bracketMat!: THREE.LineBasicMaterial;
  fire!: THREE.Mesh;
  fireMat!: THREE.MeshBasicMaterial;
  column!: THREE.Mesh;
  columnMat!: THREE.MeshBasicMaterial;
  ring!: THREE.Mesh;
  ringMat!: THREE.MeshBasicMaterial;
  pulses = new Pulses(700);
  bitLines: THREE.LineCurve3[] = [];
  beamCurves: THREE.LineCurve3[] = [];
  L2!: Line; L3!: Line;
  layL2!: TextLayout; layL3!: TextLayout; layForm!: TextLayout;
  formX0 = 0;
  formOff: number[] = [];
  // beat-locked event times (grid points)
  tOne = 0; tThreadLbl = 0; tFill: number[] = []; tGot = 0;
  tBlock = 0; tSize = 0; tThr = 0; tSpot = 0; tEnd = 0;
  tokT: number[] = [];

  override async init() {
    const ly = this.ctx.lyrics, au = this.ctx.audio, r = this.ctx.renderer;
    this.L2 = ly.get('One thread for every');
    this.L3 = ly.get('Block times size');
    const w = (l: Line, s: string) => l.words.find((x) => norm(x.w) === norm(s))!;
    const one = w(this.L2, 'One'), thread = w(this.L2, 'thread'), every = w(this.L2, 'every'), got = w(this.L2, 'got,');
    this.tOne = q(au, one.start);
    this.tThreadLbl = q(au, thread.start);
    const bE = Math.round(au.beatAt(every.start) * 2) / 2;
    this.tFill = [0, 1, 2, 3].map((k) => au.timeOfBeat(bE + k / 2));
    this.tGot = q(au, got.start);
    this.tBlock = q(au, w(this.L3, 'Block').start);
    this.tSize = q(au, w(this.L3, 'size').start);
    this.tThr = q(au, w(this.L3, 'thread').start);
    this.tSpot = q(au, w(this.L3, 'spot').start);
    this.tEnd = this.ctx.end;
    this.tokT = [this.tBlock, this.tBlock, this.tSize, this.tSize, this.tThr, this.tThr];
    this.layL2 = layout(this.L2.text, famLy, LY.size);
    this.layL3 = layout(this.L3.text, famLy, LY.size);
    this.layForm = layout(FORM.join(' '), famLy, FORM_SIZE);
    this.formX0 = (W - this.layForm.width) / 2;
    let off = 0;
    for (const tok of FORM) { this.formOff.push(off); off += tok.length + 1; }

    // ---------------------------------------------------------------- world
    if (!ENV) {
      const pm = new THREE.PMREMGenerator(r);
      ENV = pm.fromScene(new RoomEnvironment(), 0.04).texture;
      pm.dispose();
    }
    HALO ??= haloTex();
    for (let d = 0; d < 10; d++) DIGITS[d] ??= digitTex(d);
    const S = this.scene;
    S.environment = ENV;
    S.background = lin('ink');
    S.fog = new THREE.Fog(lin('ink'), 170, 520);
    S.add(H100.rig());

    // the floor: dark die surface with an SRAM bitcell pattern, and bit lines carrying data
    const sram = makeSramTexture(512);
    sram.wrapS = sram.wrapT = THREE.RepeatWrapping;
    sram.repeat.set(26, 26);
    sram.colorSpace = THREE.SRGBColorSpace;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(900, 900).rotateX(-Math.PI / 2),
      new THREE.MeshStandardMaterial({ color: lin('bone', 0.15), map: sram, metalness: 0.2, roughness: 0.85, envMapIntensity: 0.1 }));
    floor.position.y = -0.02;
    S.add(floor);
    const lv: number[] = [];
    for (const z of [-24, -17, -11.5, 11.5, 17, 24, 33]) {
      const c = new THREE.LineCurve3(new THREE.Vector3(-230, 0.05, z), new THREE.Vector3(230, 0.05, z));
      this.bitLines.push(c);
      lv.push(-230, 0.05, z, 230, 0.05, z);
    }
    for (let k = -12; k <= 12; k++) { const x = k * PITCH * 2 + PITCH / 2; lv.push(x, 0.04, -40, x, 0.04, -11.5, x, 0.04, 11.5, x, 0.04, 40); }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(lv, 3));
    S.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: lin('ash', 0.5), transparent: true, opacity: 0.5, depthWrite: false })));
    S.add(this.pulses.mesh);

    // the cells
    const sideTex = makeSramTexture(256);
    sideTex.colorSpace = THREE.SRGBColorSpace;
    const side = new THREE.MeshStandardMaterial({ color: lin('bone', 0.55), map: sideTex, metalness: 0.6, roughness: 0.38, envMapIntensity: 0.28 });
    const box = new THREE.BoxGeometry(CW, CH, CW);
    const edgeGeo = new THREE.EdgesGeometry(box);
    for (let i = 0; i < N; i++) {
      const top = new THREE.MeshStandardMaterial({ color: 0xffffff, map: DIGITS[VALUES[i]!]!, emissiveMap: DIGITS[VALUES[i]!]!, emissive: lin('bone', 0.42), metalness: 0.3, roughness: 0.5, envMapIntensity: 0.2 });
      const m = new THREE.Mesh(box, [side, side, top, side, side, side]);
      m.position.y = CH / 2;
      const edge = new THREE.LineBasicMaterial({ color: lin('bone'), transparent: true, opacity: 0.6, depthWrite: false });
      const e = new THREE.LineSegments(edgeGeo, edge);
      e.position.y = CH / 2;
      const g = new THREE.Group();
      g.add(m, e);
      S.add(g);
      this.cells.push({ group: g, top, edge });
    }
    // block plates (under each block of 4, snap in on "Block")
    const plateGeo = new THREE.BoxGeometry(SIZE * PITCH + 3, 0.5, CW + 6);
    const plateEdge = new THREE.EdgesGeometry(plateGeo);
    const plateMat = new THREE.MeshStandardMaterial({ color: lin('graphite', 0.5), metalness: 0.5, roughness: 0.4, envMapIntensity: 0.25 });
    for (let b = 0; b < NB; b++) {
      const m = new THREE.Mesh(plateGeo, plateMat);
      const edge = new THREE.LineBasicMaterial({ color: lin('bone'), transparent: true, opacity: 0, depthWrite: false });
      m.add(new THREE.LineSegments(plateEdge, edge));
      m.position.y = 0.25;
      m.visible = false;
      S.add(m);
      this.plates.push({ mesh: m, edge });
    }
    // "size = 4" bracket over block 2
    {
      const x0 = -(SIZE * PITCH) / 2 + 0.5, x1 = -x0, y = BRACKET_Y, h = 2.4;
      const v = [x0, y - h, 0, x0, y, 0, x0, y, 0, x1, y, 0, x1, y, 0, x1, y - h, 0];
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
      this.bracketMat = new THREE.LineBasicMaterial({ color: lin('signal', 2.2), transparent: true, opacity: 0, depthWrite: false });
      this.bracket = new THREE.LineSegments(g, this.bracketMat);
      this.bracket.visible = false;
      S.add(this.bracket);
    }
    // the fired cell: a hot plate above cell 9's top, and a shock ring on the floor
    this.fireMat = glowMat(0xffffff);
    this.fire = new THREE.Mesh(new THREE.RingGeometry(CW * 0.5 * Math.SQRT2 - 0.75, CW * 0.5 * Math.SQRT2, 4, 1, Math.PI / 4).rotateX(-Math.PI / 2), this.fireMat);
    this.fire.visible = false;
    S.add(this.fire);
    this.columnMat = glowMat(0xffffff);
    this.column = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 4, 1, true).rotateY(Math.PI / 4).translate(0, 0.5, 0), this.columnMat);
    this.column.visible = false;
    S.add(this.column);
    this.ringMat = glowMat(0xffffff);
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.94, 1, 96).rotateX(-Math.PI / 2), this.ringMat);
    this.ring.visible = false;
    S.add(this.ring);

    // the threads: spark core, halo, and a reading beam down to the cell
    const coreGeo = new THREE.SphereGeometry(0.6, 20, 14);
    const beamGeo = new THREE.CylinderGeometry(0.1, 0.1, 1, 8, 1);
    for (let i = 0; i < N; i++) {
      const coreMat = glowMat(0xffffff);
      const core = new THREE.Mesh(coreGeo, coreMat);
      const haloMat = new THREE.SpriteMaterial({
        map: HALO, color: 0xffffff, transparent: true, depthWrite: false,
        blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
        blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
      });
      const halo = new THREE.Sprite(haloMat);
      const beamMat = glowMat(0xffffff);
      const beam = new THREE.Mesh(beamGeo, beamMat);
      core.visible = halo.visible = beam.visible = false;
      S.add(core, halo, beam);
      this.threads.push({ core, halo, beam, coreMat, haloMat, beamMat });
      this.beamCurves.push(new THREE.LineCurve3(new THREE.Vector3(), new THREE.Vector3()));
    }
  }

  /** When thread i drops in. */
  tApp(i: number) { return i === 0 ? this.tOne : this.tFill[Math.floor((i - 1) / 4)]!; }

  // ---------------------------------------------------------------- camera
  /** The camera for song time t: track along the row, rise to the whole array on "Block", whip onto cell 9 on "spot". */
  shot(t: number) {
    const au = this.ctx.audio;
    const P = au.timeOfBeat(1) - au.timeOfBeat(0);
    /** 0..1 over the `dur` before `at` (a move that lands on that grid point). */
    const land = (at: number, dur: number, fn = ease.inOutCubic) => fn(clamp((t - (at - dur)) / dur));
    const kSize = land(this.tSize, P / 2), kThr = land(this.tThr, P / 2);
    // A: tracking along the row, looking down it so the cells recede; the speed breathes with the drums
    const travel = drive(au, t, 11, 9, this.ctx.start);
    const tx = cellX(0, 0) - 6 + travel;
    const aPos = new THREE.Vector3(tx - 50, 29, 46);
    const aTgt = new THREE.Vector3(tx + 22, 5, -3);
    // B: the whole array from a high, nearly frontal view, slowly orbiting; slides toward block 2 on "size", in on "thread"
    const orb = -0.2 + 0.03 * (drive(au, t, 1, 2.2) - drive(au, this.tBlock, 1, 2.2));
    const bTgt = new THREE.Vector3(lerp(0, blockX(EX.b, 1), lerp(0.3 * kSize, 0.8, kThr)), 2, 0);
    const bDist = lerp(lerp(222, 205, kSize), 150, kThr);
    const bPos = new THREE.Vector3(bTgt.x + Math.sin(orb) * bDist * 0.8, bDist * 0.6, Math.cos(orb) * bDist * 0.8);
    // C: on cell 9, orbiting it; the cell sits above centre so the formula below stays clear
    const c9 = new THREE.Vector3(cellX(EXI, 1), CH, 0);
    const orc = -0.3 + 0.045 * (drive(au, t, 1, 2.2) - drive(au, this.tSpot, 1, 2.2));
    const cDist = lerp(84, 72, clamp((t - this.tSpot) / Math.max(0.1, this.tEnd - this.tSpot)));
    const toCam = new THREE.Vector3(Math.sin(orc), 0, Math.cos(orc));
    const cPos = c9.clone().addScaledVector(toCam, cDist * 0.84).add(new THREE.Vector3(0, cDist * 0.5, 0));
    const cTgt = c9.clone().addScaledVector(toCam, 9).add(new THREE.Vector3(0, -2, 0));
    // blends: the rise lands on "Block" (one beat), the whip lands on "spot" (one 8th)
    const kB = land(this.tBlock, P);
    const kC = land(this.tSpot, P / 2, ease.inOutExpo);
    const pos = aPos.clone().lerp(bPos, kB).lerp(cPos, kC);
    const tgt = aTgt.clone().lerp(bTgt, kB).lerp(cTgt, kC);
    // a small push toward the subject on each kick (lands with the drum; not a shake)
    const push = 0.028 * kickPush(au, t);
    pos.lerp(tgt, push);
    return { pos, tgt, fov: lerp(36, 34, kB) - 2 * kC };
  }

  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const au = this.ctx.audio;
    const t = f.t;
    const kSplit = snap(au, t, this.tBlock, 0.12);
    const kSize = snap(au, t, this.tSize), kThr = snap(au, t, this.tThr), kSpot = snap(au, t, this.tSpot);
    const line3 = t >= this.L3.start - 0.4;
    const drums = au.env('drums', t), rms = au.env('rms', t);
    const pulseAll = hit(au, t, this.tGot, 0.09);

    // ---------------------------------------------------------------- the array
    for (let i = 0; i < N; i++) {
      const c = this.cells[i]!;
      c.group.position.x = cellX(i, kSplit);
      const lit = i === EXI ? kSpot : 0;
      const land = hit(au, t, this.tApp(i), 0.1);
      c.top.emissive.copy(lin('bone', 0.42 + 0.5 * land)).lerp(lin('signal', 2.6), lit);
      c.edge.color.copy(lin('bone')).lerp(lin('signal', 2.2), Math.max(lit, land));
      c.edge.opacity = 0.6 + 0.4 * Math.max(lit, land);
    }
    for (let b = 0; b < NB; b++) {
      const p = this.plates[b]!;
      p.mesh.visible = kSplit > 0;
      p.mesh.position.x = blockX(b, kSplit);
      p.mesh.scale.set(1, 1, kSplit);
      const hot = b === EX.b ? kThr : 0;
      p.edge.opacity = 0.7 * kSplit;
      p.edge.color.copy(lin('bone')).lerp(lin('signal', 2), hot);
    }
    this.bracket.visible = kSize > 0 && kSpot < 1;
    this.bracket.position.x = blockX(EX.b, 1);
    this.bracketMat.opacity = kSize * (1 - kSpot);
    // the fired cell
    this.fire.visible = kSpot > 0;
    this.fire.position.set(cellX(EXI, 1), CH + 0.08, 0);
    const burst = hit(au, t, this.tSpot, 0.16);
    this.fireMat.color.copy(lin('signal', (2.2 + 1.2 * drums) * kSpot + 6 * burst));
    const rt = t - q(au, this.tSpot);
    // the light column: shoots up on the hit, then stays as a thin pillar that breathes with the drums
    this.column.visible = rt > 0;
    if (rt > 0) {
      const up = ease.outExpo(clamp(rt / 0.35));
      this.column.position.set(cellX(EXI, 1), CH, 0);
      const cw = lerp(CW * 0.62, 0.7, up);
      this.column.scale.set(cw, 90 * up, cw);
      this.columnMat.color.copy(lin('signal', 0.2 + 0.25 * drums + 3 * burst));
    }
    this.ring.visible = rt > 0 && rt < 1.2;
    if (this.ring.visible) {
      const R = 6 + 44 * ease.outExpo(clamp(rt / 1.1));
      this.ring.scale.setScalar(R);
      this.ring.position.set(cellX(EXI, 1), 0.08, 0);
      this.ringMat.color.copy(lin('ember', 3.2 * (1 - clamp(rt / 1.1))));
    }

    // ---------------------------------------------------------------- the threads
    const items: PulseItem[] = [];
    for (let i = 0; i < N; i++) {
      const th = this.threads[i]!;
      const k = snap(au, t, this.tApp(i), 0.12);
      const on = k > 0;
      th.core.visible = th.halo.visible = th.beam.visible = on;
      if (!on) continue;
      const isEx = i === EXI;
      const dim = line3 ? lerp(1, isEx ? 1.6 : 0.32, kThr) : 1;
      const I = dim * (0.8 + 0.35 * drums + 0.9 * pulseAll + (isEx ? 1.5 * burst : 0));
      const x = cellX(i, kSplit);
      const y = HOVER + 60 * (1 - k);
      th.core.position.set(x, y, 0);
      th.core.scale.setScalar(isEx ? lerp(1, 1.3, kThr) : 1);
      th.coreMat.color.copy(lin('ember', 2.4 * I));
      th.halo.position.set(x, y, 0);
      th.halo.scale.setScalar(9 * (0.85 + 0.3 * drums) * (isEx ? lerp(1, 1.5, kThr) : 1));
      th.haloMat.color.copy(lin('signal', 1.3 * I));
      // the reading beam: from the thread down to its number, once it has landed
      const b = k >= 1 ? 1 : 0;
      const top = CH + 0.05, h = Math.max(0.01, y - top);
      th.beam.visible = b > 0;
      th.beam.position.set(x, top + h / 2, 0);
      th.beam.scale.set(1, h, 1);
      th.beamMat.color.copy(lin('ember', 0.9 * I));
      if (b > 0) {
        const cv = this.beamCurves[i]!;
        cv.v1.set(x, y - 0.9, 0); cv.v2.set(x, top + 0.2, 0);
        for (let n = 0; n < 2; n++) {
          const u = (t * 1.9 + n / 2 + i * 0.137) % 1;
          items.push({ curve: cv, u, len: 0.22, gain: 3.2 * I, width: 0.34, color: LIN.signal });
        }
      }
    }
    // data on the bit lines: always flowing, faster and brighter with the band
    const flow = drive(au, t, 0.035, 0.05);
    this.bitLines.forEach((cv, li) => {
      for (let n = 0; n < 7; n++) {
        const u = (flow * (li % 2 ? 1 : 1.3) + n / 7 + li * 0.173) % 1;
        items.push({ curve: cv, u, len: 0.012, gain: 1.3 + 2.2 * rms, width: 0.45, color: LIN.ember });
      }
    });
    this.pulses.set(items);

    // ---------------------------------------------------------------- render
    const s = this.shot(t);
    const cam = this.cam;
    cam.position.copy(s.pos);
    cam.lookAt(s.tgt);
    cam.fov = s.fov;
    H100.fitClip(cam, s.pos.distanceTo(s.tgt));
    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    comp.draw(renderer, this.ms.texture, out);

    // ---------------------------------------------------------------- screen-space type
    const c = this.ui.ctx;
    this.ui.clear();
    this.shade(c);
    const P = (v: THREE.Vector3) => {
      const p = v.clone().project(cam);
      return { x: (p.x * 0.5 + 0.5) * W, y: (-p.y * 0.5 + 0.5) * H, ok: p.z < 1 };
    };
    // each cell's index i at its front foot; on "thread" all but block 2 dim, on "spot" cell 9 becomes "i = 9"
    c.textAlign = 'center'; c.textBaseline = 'alphabetic';
    for (let i = 0; i < N; i++) {
      const p = P(new THREE.Vector3(cellX(i, kSplit), 0, CW / 2 + 0.8));
      if (!p.ok) continue;
      const inEx = Math.floor(i / SIZE) === EX.b;
      const big = i === EXI ? kSpot : 0;
      const a = big > 0.5 ? 1 : (inEx ? 0.95 : lerp(0.95, 0.35, kThr)) * (1 - kSpot);
      if (a <= 0) continue;
      c.font = font(mono(big > 0.5 ? 700 : 500), big > 0.5 ? 46 : 26);
      c.fillStyle = big > 0.5 ? rgba('signal', 1) : rgba('ash', a);
      c.fillText(big > 0.5 ? `i = ${i}` : String(i), p.x, p.y + (big > 0.5 ? 58 : 34));
    }
    // "1 thread" by the first one, until the fill starts
    {
      const a = snap(au, t, this.tThreadLbl) * (t >= q(au, this.tFill[0]!) ? 0 : 1);
      if (a > 0) {
        const p = P(new THREE.Vector3(cellX(0, 0), HOVER, 0));
        c.font = font(mono(600), 34); c.fillStyle = rgba('signal', a);
        c.textAlign = 'right';
        c.fillText('1 thread', p.x - 30, p.y + 10);
        c.textAlign = 'center';
      }
    }
    // "16 numbers · 16 threads · one each" on "got", until the split
    {
      const a = snap(au, t, this.tGot) * (kSplit > 0 ? 0 : 1);
      if (a > 0) {
        c.font = font(mono(500), 36); c.fillStyle = rgba('bone', a);
        c.fillText('16 numbers · 16 threads · one each', W / 2, 800);
      }
    }
    // blocks: "block b" under each plate (block 2 in signal on "thread")
    if (kSplit > 0 && kSpot < 1) {
      for (let b = 0; b < NB; b++) {
        const hot = b === EX.b ? kSize : 0;
        const p = P(new THREE.Vector3(blockX(b, kSplit), 0, CW / 2 + 3));
        if (!p.ok) continue;
        c.font = font(mono(hot > 0.5 ? 700 : 500), 32);
        c.fillStyle = hot > 0.5 ? rgba('signal', kSplit * (1 - kSpot)) : rgba('bone', lerp(0.9, 0.5, kThr) * kSplit * (1 - kSpot));
        c.fillText(`block ${b}`, p.x, p.y + 84);
      }
    }
    // on "thread": the threads of block 2 are numbered 0..3 from the block's start; thread 1 is ours
    if (kThr > 0) {
      for (let j = 0; j < SIZE; j++) {
        const i = EX.b * SIZE + j, ours = j === EX.th;
        if (kSpot >= 1) continue;
        const p = P(new THREE.Vector3(cellX(i, 1), HOVER + 1.6, 0));
        if (!p.ok) continue;
        c.font = font(mono(ours ? 700 : 500), ours ? 34 : 26);
        c.fillStyle = ours ? rgba('signal', kThr * (1 - kSpot)) : rgba('ash', 0.95 * kThr * (1 - kSpot));
        c.fillText(ours ? `thread ${j}` : String(j), p.x, p.y - 16);
      }
    }
    if (kSize > 0 && kSpot < 1) {
      const p = P(new THREE.Vector3(blockX(EX.b, 1), BRACKET_Y, 0));
      if (p.ok) { c.font = font(mono(600), 32); c.fillStyle = rgba('signal', kSize * (1 - kSpot)); c.fillText(`size = ${SIZE}`, p.x, p.y - 14); }
    }
    // the formula, word by word
    for (let k = 0; k < FORM.length; k++) {
      const a = snap(au, t, this.tokT[k]!);
      if (a <= 0) continue;
      const tok = FORM[k]!;
      const g0 = this.layForm.glyphs[this.formOff[k]!]!;
      const isOp = tok === '×' || tok === '+' || tok === 'i =';
      const sz = 1 + 0.2 * (1 - a);
      c.save();
      c.translate(this.formX0 + g0.x, FORM_Y); c.scale(sz, sz);
      c.textAlign = 'left';
      c.font = font(famLy, FORM_SIZE);
      c.fillStyle = isOp ? rgba('ash', a) : rgba('bone', a);
      c.fillText(tok, 0, 0);
      c.restore();
    }
    // the worked example and the real line (on "spot")
    if (kSpot > 0) {
      c.textAlign = 'left';
      const s1 = `i = ${EX.b} × ${SIZE} + ${EX.th} = `, s2 = String(EXI);
      c.font = font(mono(500), 48);
      const w1 = c.measureText(s1).width, w2 = c.measureText(s2).width;
      const x = (W - (w1 + w2)) / 2;
      c.fillStyle = rgba('bone', kSpot); c.fillText(s1, x, SUB_Y);
      c.font = font(mono(700), 48); c.fillStyle = rgba('signal', kSpot); c.fillText(s2, x + w1, SUB_Y);
      c.font = font(mono(400), 24); c.fillStyle = rgba('ash', 0.9 * kSpot); c.textAlign = 'center';
      c.fillText('in CUDA:  int i = blockIdx.x * blockDim.x + threadIdx.x;   · CH02', W / 2, FOOT_Y);
    }
    c.textAlign = 'left';
    // the sung line
    if (line3) this.karaoke(c, this.L3, this.layL3, t);
    else this.karaoke(c, this.L2, this.layL2, t);
    comp.draw(renderer, this.ui.upload(), out);

    const imp = hit(au, t, this.tSpot, 0.1) + 0.5 * hit(au, t, this.tBlock, 0.08);
    return { bloom: 0.6, bloomThreshold: 0.9, exposure: 1 + 0.1 * imp };
  }

  /** Soft ink shading at the top and bottom so the lyric and the formula read over the 3D world. */
  shade(c: CanvasRenderingContext2D) {
    let g = c.createLinearGradient(0, 0, 0, 300);
    g.addColorStop(0, rgba('ink', 0.72)); g.addColorStop(1, rgba('ink', 0));
    c.fillStyle = g; c.fillRect(0, 0, W, 300);
    g = c.createLinearGradient(0, H - 300, 0, H);
    g.addColorStop(0, rgba('ink', 0)); g.addColorStop(1, rgba('ink', 0.8));
    c.fillStyle = g; c.fillRect(0, H - 300, W, 300);
  }

  /** The sung line in one fixed place: done words bone, the word being sung signal, unsung dim. */
  karaoke(c: CanvasRenderingContext2D, line: Line, lay: TextLayout, t: number) {
    let off = 0, doneCh = 0, sungCh = 0;
    for (const w of line.words) {
      const n = Array.from(w.w).length;
      const p = Lyrics.wordProgress(w, t);
      if (p >= 1) { doneCh = off + n; sungCh = doneCh; off += n + 1; continue; }
      if (p > 0) { doneCh = off; sungCh = off + p * n; }
      break;
    }
    const xAt = (ch: number) => {
      const i = Math.floor(ch);
      if (i >= lay.glyphs.length) return lay.width;
      const g = lay.glyphs[i]!;
      return g.x + (ch - i) * g.w;
    };
    const { x, y, size } = LY;
    c.font = font(famLy, size);
    c.textBaseline = 'alphabetic';
    c.fillStyle = rgba('bone', 0.3);
    c.fillText(line.text, x, y);
    const top = y - size * 1.05, h = size * 1.45;
    const xd = x + xAt(doneCh), xs = x + xAt(sungCh);
    if (doneCh > 0) {
      c.save(); c.beginPath(); c.rect(x - 20, top, xd - x + 20, h); c.clip();
      c.fillStyle = rgba('bone', 1); c.fillText(line.text, x, y); c.restore();
    }
    if (sungCh > doneCh) {
      c.save(); c.beginPath(); c.rect(xd, top, xs - xd, h); c.clip();
      c.fillStyle = rgba('signal', 1); c.fillText(line.text, x, y); c.restore();
    }
  }
}
