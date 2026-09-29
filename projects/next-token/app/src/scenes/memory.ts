// memory (the breakdown): "Every single token, it reads every weight, / Sixteen gigs from memory while the
// math just waits". Memory holds the 16.06 GB of weights as a pile of printed sheets (each matrix at its real
// proportions: 32 layers x Q K V O GATE UP DOWN, plus the fan-folded embedding and head), and a copy of each
// layer's seven sheets flies into the tower as the token in flight reaches that floor. One decode step reads
// 15.01 GB of it: everything except the embedding table, of which it looks up one row (a thin strip flies).
// The byte counter follows the story clock: it finishes ·in on "weight", then counts ·the from 0 to 15.01 GB
// over the second line, landing on "waits". Then the hardware view: memory -> the 936 GB/s bus (642 GB/s
// achieved) -> the math unit, drawn full size and almost empty (0.97% of the bf16 peak in use, computed from
// the measured 42.8 tok/s), a clock of the measured 23.4 ms per token, and a WAIT stamp. Cuts every 2 beats,
// each on a sung word; the idea changes every bar.
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { F, font } from '../engine/type';
import { HEX } from '../engine/palette';
import { clamp, ease, hash } from '../engine/util';
import { showTok } from '../engine/llm';
import { Wash, specks, makeCam, drive, shot, snapIn, portrait, perFrame, inkMat, flatMat, outline, tokenTile } from './_kit';
import { Tower, riderTile, FLOOR_H } from './_tower';
import {
  PINK, BLUE, INK, PAPER, KINDS, type Kind, type Inventory, inventory, sheetMeshes, sheetSize, slab, label, labelW, bigNum, rain,
  toScreen, stamp, runNumbers, fmt1, fmt2, fmtInt, GPU, safeTop,
} from './memory-parts';

const PILE_T = new THREE.Vector3(-10.5, 0, 0);
const PILE_M = new THREE.Vector3(-10, 0, 0);
const CUBE = { c: new THREE.Vector3(10, 6, 0), s: 12 };
const BELT = { y: 9.2, x0: -7.2, x1: 3.6 };

export default class Memory extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(40);
  wash = new Wash();
  back = new Layer2D();
  front = new Layer2D();
  fx = new Layer2D();
  tower = new Tower();
  inv!: Inventory;
  pile = new THREE.Group();
  fly = new Map<Kind, THREE.InstancedMesh>();
  machine = new THREE.Group();
  busy!: THREE.Mesh;
  riders = new Map<number, THREE.Mesh>();
  heroes = new Map<number, THREE.Mesh>();
  emitted = new Map<number, THREE.Mesh>();
  N!: ReturnType<typeof runNumbers>;
  T0 = 0;
  tm = { weight: 0, sixteen: 0, waits: 0, every: 0, single: 0, token: 0, memory: 0, math: 0 };
  pool: string[] = [];
  private kvAt = -1;
  private m4 = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private e = new THREE.Euler();
  private v = new THREE.Vector3();
  private sc = new THREE.Vector3();
  private zero = new THREE.Matrix4().makeScale(0, 0, 0);

  override init() {
    const { llm, lyrics: ly, audio: au, story } = this.ctx;
    this.N = runNumbers(llm.d);
    this.inv = inventory(llm.d, (l) => this.tower.floorY(l));
    const word = (line: string, w: string) => {
      const l = ly.get(line);
      return (l.words.find((x) => x.w.toLowerCase().replace(/[^a-z']/g, '').startsWith(w)) ?? l.words[0]!).start;
    };
    const l1 = ly.get('Every single token, it reads');
    this.T0 = au.timeOfBeat(Math.floor(au.beatAt(l1.words[0]!.start + 0.02)));
    this.tm = {
      every: word('Every single token, it reads', 'every'), single: word('Every single token, it reads', 'single'),
      token: word('Every single token, it reads', 'token'), weight: word('Every single token, it reads', 'weight'),
      sixteen: word('Sixteen gigs', 'sixteen'), memory: word('Sixteen gigs', 'memory'), math: word('Sixteen gigs', 'math'),
      waits: word('Sixteen gigs', 'waits'),
    };

    // the tower and its riders (the tokens in flight during this scene)
    this.tower.outlineAll();
    this.scene.add(this.tower.group);
    for (let k = 0; k < llm.steps.length; k++) {
      const e = story.emits[k]!;
      if (e < this.ctx.start - 12 || e > this.ctx.end + 6) continue;
      this.riders.set(k, riderTile(llm.steps[k]!.tok.t));
      const hero = tokenTile(llm.steps[k]!.tok.t, { face: 'pink', text: 'paper', side: 'blue', id: llm.steps[k]!.tok.id, h: 2.6, line: 3 });
      hero.visible = false;
      this.scene.add(hero);
      this.heroes.set(k, hero);
      const em = tokenTile(llm.steps[k]!.tok.t, { face: 'pink', text: 'paper', side: 'pink', id: llm.steps[k]!.tok.id, h: 1.1 });
      em.visible = false;
      this.scene.add(em);
      this.emitted.set(k, em);
    }

    // the VRAM pile: every sheet in its slot, the same height as the floor it feeds
    const counts = (k: Kind) => this.inv.byKind.get(k)!.length;
    const pileMeshes = sheetMeshes(this.inv, counts);
    for (const k of KINDS) {
      const m = pileMeshes.get(k)!;
      this.inv.byKind.get(k)!.forEach((s, i) => {
        this.e.set(0, (hash(s.i, 3) - 0.5) * 0.1, 0);
        this.q.setFromEuler(this.e);
        this.v.set((hash(s.i, 1) - 0.5) * 0.3, s.y, (hash(s.i, 2) - 0.5) * 0.3);
        m.setMatrixAt(i, this.m4.compose(this.v, this.q, this.sc.set(1, 1, 1)));
      });
      m.instanceMatrix.needsUpdate = true;
      this.pile.add(m);
    }
    // a base plinth and a spine so the pile reads as one block of memory
    const plinth = new THREE.Mesh(new THREE.BoxGeometry(6.2, 0.5, 6.2), inkMat({ ink: 'ink', lit: 0.3, shade: 1 }));
    plinth.position.y = -0.75;
    outline(plinth, 2);
    this.pile.add(plinth);
    this.scene.add(this.pile);

    // copies in flight (a pool per kind)
    this.fly = sheetMeshes(this.inv, (k) => (k === 'EMBED' || k === 'HEAD' ? 10 : 8), 0.05);
    for (const m of this.fly.values()) this.scene.add(m);

    // the hardware view: the bus and the math unit (a frame cube at full size, nearly empty)
    const s = CUBE.s, b = 0.32, hs = s / 2;
    const beams: THREE.BufferGeometry[] = [];
    for (const u of [-hs, hs]) for (const w of [-hs, hs]) {
      beams.push(new THREE.BoxGeometry(s + b, b, b).translate(0, u, w)); // along x
      beams.push(new THREE.BoxGeometry(b, s + b, b).translate(u, 0, w)); // along y
      beams.push(new THREE.BoxGeometry(b, b, s + b).translate(u, w, 0)); // along z
    }
    const frame = mergeGeometries(beams);
    const cube = new THREE.Mesh(frame, inkMat({ ink: 'blue', lit: 0.55, shade: 1, over: 'pink', overLit: 0, overShade: 0.3 }));
    outline(cube, 2.2);
    cube.position.copy(CUBE.c);
    this.machine.add(cube);
    // the floor, gridded 10 x 10 so one busy column reads as about 1 in 100
    const gc = document.createElement('canvas');
    gc.width = gc.height = 512;
    const g2 = gc.getContext('2d')!;
    g2.fillStyle = HEX.paper; g2.fillRect(0, 0, 512, 512);
    g2.fillStyle = HEX.blue;
    for (let q = 0; q <= 10; q++) { g2.fillRect(q * 51.2 - 3, 0, 6, 512); g2.fillRect(0, q * 51.2 - 3, 512, 6); }
    const gt = new THREE.CanvasTexture(gc);
    gt.colorSpace = THREE.SRGBColorSpace;
    const floor = new THREE.Mesh(new THREE.BoxGeometry(s, 0.12, s), [flatMat('blue'), flatMat('blue'), new THREE.MeshBasicMaterial({ map: gt }), flatMat('blue'), flatMat('blue'), flatMat('blue')]);
    floor.position.set(CUBE.c.x, CUBE.c.y - hs + 0.06, CUBE.c.z);
    outline(floor, 1.6);
    this.machine.add(floor);
    // what is in use: 0.97% of the volume, as one column at full height
    const side = s * Math.sqrt(this.N.util1);
    this.busy = new THREE.Mesh(new THREE.BoxGeometry(side, s - 0.2, side), inkMat({ ink: 'pink', lit: 0.75, shade: 1 }));
    this.busy.position.set(CUBE.c.x - hs + side / 2 + 0.2, CUBE.c.y + 0.02, CUBE.c.z + hs - side / 2 - 0.2);
    outline(this.busy, 2);
    this.machine.add(this.busy);
    const belt = new THREE.Mesh(new THREE.BoxGeometry(BELT.x1 - BELT.x0, 0.35, 2.8), inkMat({ ink: 'ink', lit: 0.35, shade: 1 }));
    belt.position.set((BELT.x0 + BELT.x1) / 2, BELT.y - 0.2, 0);
    outline(belt, 2);
    this.machine.add(belt);
    for (const z of [-1.5, 1.5]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(BELT.x1 - BELT.x0, 0.5, 0.18), inkMat({ ink: 'blue', lit: 0.5, shade: 1 }));
      rail.position.set((BELT.x0 + BELT.x1) / 2, BELT.y + 0.1, z);
      outline(rail, 1.6);
      this.machine.add(rail);
    }
    this.scene.add(this.machine);

    // the rain: the real tensor names and sizes
    const pool = new Set<string>();
    for (const sh of this.inv.sheets) {
      const [rows, cols] = this.inv.shape[sh.kind];
      pool.add(sh.layer >= 1 && sh.layer <= llm.cfg.layers ? `L${sh.layer}.${sh.kind}` : sh.kind);
      pool.add(`${fmtInt(rows)}×${fmtInt(cols)}`);
      pool.add(`${fmt1(this.inv.bytes[sh.kind] / 1e6)} MB`);
    }
    this.pool = [...pool];
  }

  /** Put every copy in flight: sheet s leaves its slot when the token reaches its read position. */
  private flyAt(f: number, layout: 'T' | 'M', pile: THREE.Vector3, target?: THREE.Vector3, big = 1) {
    const FL = layout === 'T' ? 2.4 : 3.4;
    for (const k of KINDS) {
      const m = this.fly.get(k)!, list = this.inv.byKind.get(k)!;
      let n = 0;
      for (const s of list) {
        // the embedding table stays in memory: a decode step looks up one row, drawn as a thin strip of its first panel
        if (s.kind === 'EMBED' && s.panel > 0) continue;
        const p = (f - s.r) / FL;
        if (p <= 0 || p >= 1 || n >= m.count) continue;
        const h1 = hash(s.i, 11), h2 = hash(s.i, 12), h3 = hash(s.i, 13);
        let sc = 1;
        if (layout === 'T') {
          const e = ease.inOutCubic(p);
          const ty = target ? target.y : this.tower.floorY(Math.min(33, s.r + FL)) + 0.3, tx = target ? target.x : 0, tz = target ? target.z : 0;
          this.v.set(pile.x + (tx - pile.x) * e, s.y + (ty - s.y) * e + Math.sin(Math.PI * p) * (1.0 + h1 * 1.6), pile.z + (tz - pile.z) * e + Math.sin(Math.PI * p) * (h2 - 0.3) * 5);
          this.e.set(Math.sin(Math.PI * p) * (h3 - 0.5) * 1.4, p * (h1 - 0.5) * 3, Math.sin(Math.PI * p) * (h2 - 0.5) * 0.9);
          sc = p > 0.78 ? 1 - ((p - 0.78) / 0.22) * 0.75 : 1;
        } else {
          // up from the slot to the bus, along it, and into the math unit
          const zj = (h2 - 0.5) * 1.6;
          if (p < 0.3) {
            const u = ease.inOutCubic(p / 0.3);
            this.v.set(pile.x + (BELT.x0 - pile.x) * u, s.y + (BELT.y + 0.35 - s.y) * u + Math.sin(Math.PI * u) * 1.2, pile.z + zj * u);
            this.e.set(0, (1 - u) * (h1 - 0.5) * 2, Math.sin(Math.PI * u) * (h3 - 0.5));
          } else if (p < 0.8) {
            const u = (p - 0.3) / 0.5;
            this.v.set(BELT.x0 + (BELT.x1 - BELT.x0) * u, BELT.y + 0.35, zj);
            this.e.set(0, 0, 0);
          } else {
            const u = ease.inCubic((p - 0.8) / 0.2);
            this.v.set(BELT.x1 + (CUBE.c.x - BELT.x1) * u, BELT.y + 0.35 + (CUBE.c.y - 3 - BELT.y) * u, zj * (1 - u));
            this.e.set(0, u * (h1 - 0.5) * 2, -u * 0.6);
            sc = 1 - u * 0.8;
          }
        }
        this.q.setFromEuler(this.e);
        sc *= big;
        m.setMatrixAt(n++, this.m4.compose(this.v, this.q, this.sc.set(sc, sc, s.kind === 'EMBED' ? sc * 0.06 : sc)));
      }
      for (let i = n; i < m.count; i++) m.setMatrixAt(i, this.zero);
      m.instanceMatrix.needsUpdate = true;
    }
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio, llm = this.ctx.llm, story = this.ctx.story;
    const P = portrait();
    const sh = shot(this.ctx, f, { every: 2, from: this.T0 });
    const i = Math.min(7, sh.i);
    const idea = i >> 1; // bar: 0 every token, 1 every weight, 2 sixteen gigs from memory, 3 the math waits
    const layout: 'T' | 'M' = idea < 2 ? 'T' : 'M';
    const d = drive(au, f.t, 0.5, 1.6);
    const st = story.at(f.t), stF = story.at(f.ft);
    const fl = st.climb * (llm.cfg.layers + 1);
    const tm = this.tm;

    // ---- background: wash (reseeded per shot) and the tensor-name rain
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: idea === 3 ? 0.7 : 0.5, c1: i % 2 ? 'blue' : 'pink', c2: i % 2 ? 'pink' : 'blue' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => rain(this.back.ctx, this.pool, drive(au, f.ft, 0.5, 1.6) * 0.45, { seed: sh.seed, cols: P ? 6 : 10, color: BLUE })), out);

    // ---- 3D
    const pileP = layout === 'T' ? PILE_T : PILE_M;
    this.pile.position.copy(pileP);
    this.tower.group.visible = layout === 'T' && i !== 1;
    this.machine.visible = layout === 'M';
    const rider = this.riders.get(st.step);
    if (rider) { this.tower.ride(rider, st.climb); rider.scale.setScalar(1.5); }
    this.tower.light(st.climb);
    if (st.pos !== this.kvAt) { this.tower.setKV(st.pos, 1); this.kvAt = st.pos; }
    const yTok = this.tower.floorY(fl) + 0.25;
    // S1: the token alone, every sheet flying into it
    const heroP = new THREE.Vector3(-1.5, yTok + 0.3, 0);
    for (const [k, h] of this.heroes) {
      h.visible = i === 1 && k === st.step;
      if (!h.visible) continue;
      h.position.copy(heroP);
      h.rotation.set(0.12, -0.35 + Math.sin(d * 0.3) * 0.12, 0);
    }
    this.flyAt(fl, layout, pileP, i === 1 ? heroP : undefined, i === 3 ? 1.9 : i === 4 ? 1.7 : 1);
    // the token just emitted shoots out of the roof
    for (const [k, em] of this.emitted) {
      const age = f.t - story.emits[k]!;
      em.visible = layout === 'T' && age >= 0 && age < 0.9;
      if (!em.visible) continue;
      const u = ease.outExpo(clamp(age / 0.5));
      em.position.set(0, this.tower.height + 0.6 + u * 7, 0);
      em.rotation.set(0, age * 1.5, 0);
    }
    const a = d * 0.09 + sh.i * 0.7;
    const push = drive(au, f.t, 1.4, 1.6) - drive(au, sh.t0, 1.4, 1.6); // a steady push-in through the shot
    let pos: THREE.Vector3, look: THREE.Vector3, fov = 40;
    const k = P ? 1.55 : 1;
    switch (i) {
      case 0: // every single token: the pile feeding the tower at the token's floor
        pos = new THREE.Vector3(-5 + Math.sin(a) * 3, yTok + 3, 25 * k); look = new THREE.Vector3(-5, yTok - (P ? 1.5 : 0.5), 0); break;
      case 1: // the token alone, every sheet flying into it
        pos = P ? new THREE.Vector3(0.5 + Math.sin(a) * 0.6, yTok + 1.2, 15) : new THREE.Vector3(2.2 + Math.sin(a) * 0.6, yTok + 0.9, 9.5);
        look = new THREE.Vector3(P ? -3.2 : -3.6, yTok + (P ? -0.4 : 0.1), 0); fov = P ? 50 : 40; break;
      case 2: // in the stream, looking back at the pile
        // (outside the stream's reach: a sheet passing the lens would black out the frame)
        pos = P ? new THREE.Vector3(-3.6, yTok + 2.6, 10.5 + Math.sin(a) * 0.4) : new THREE.Vector3(-3.4, yTok + 2.2, 7.6 + Math.sin(a) * 0.4);
        look = new THREE.Vector3(-9.8, yTok - 0.2, -0.6); fov = P ? 56 : 46; break;
      case 3: // the whole model: pile and tower, top to bottom
        pos = new THREE.Vector3(-5 + Math.sin(a) * 9 + push * 1.5, 11, (P ? 40 : 34) + Math.cos(a) * 2 - push * 3); look = new THREE.Vector3(-5, 10.3, 0); fov = P ? 44 : 40; break;
      case 4: // memory -> bus -> math
        pos = P ? new THREE.Vector3(20 + Math.sin(a) * 2 - push, 18, 30 - push * 2) : new THREE.Vector3(1 + Math.sin(a) * 3 + push, 17, 26 - push * 2.5);
        look = P ? new THREE.Vector3(0, 6.5, 0) : new THREE.Vector3(2, 7.2, 0); fov = P ? 54 : 44; break;
      case 5: // memory, from below
        pos = new THREE.Vector3(-3.5 + Math.sin(a) * 1.2, 2.5, P ? 13 : 10); look = new THREE.Vector3(-9, 11, 0); fov = P ? 62 : 54; break;
      case 6: // the math unit, full size, nearly empty: one busy column
        pos = P ? new THREE.Vector3(CUBE.c.x - 4 + Math.sin(a) * 2, 5, 30) : new THREE.Vector3(CUBE.c.x - 7 + Math.sin(a) * 2, 5, 21);
        look = P ? new THREE.Vector3(CUBE.c.x, 3, 0) : new THREE.Vector3(CUBE.c.x - 5.5, 5.5, 0); fov = P ? 50 : 46; break;
      default: // WAIT: from above the far corner, looking into the empty cage
        pos = P ? new THREE.Vector3(CUBE.c.x + 13, 21, 22) : new THREE.Vector3(CUBE.c.x + 12 + Math.sin(a) * 1.5, 17, 15);
        look = P ? new THREE.Vector3(CUBE.c.x, 2, 0) : new THREE.Vector3(CUBE.c.x - 4, 4, 0); fov = P ? 52 : 48; break;
    }
    this.cam.position.copy(pos);
    this.cam.up.set(0, 1, 0);
    if (this.cam.fov !== fov) { this.cam.fov = fov; this.cam.updateProjectionMatrix(); }
    this.cam.lookAt(look);
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);

    // ---- type (crisp, once per frame)
    const N = this.N, total = this.inv.total, perTok = this.inv.read;
    this.ctx.comp.draw(r, perFrame(this.front, f, () => {
      const c = this.front.ctx, ft = f.ft;
      const s = stF, ff = s.climb * (llm.cfg.layers + 1);
      const read = this.inv.bytesAt(ff);
      const tok = showTok(llm.steps[s.step]!.tok.t);
      const kIn = snapIn(ft, sh.t0);
      const top = safeTop();
      const counter = (x: number, y: number, size: number, align: CanvasTextAlign = 'center', sub = true) => {
        bigNum(c, `${fmt2(read / 1e9)} GB`, x, y, size, { align });
        if (!sub) return;
        const lab = `READ FOR ${tok} · OF ${fmt2(perTok / 1e9)} GB PER TOKEN`;
        const lx = align === 'center' ? x : align === 'right' ? x : x;
        label(c, lab, lx, y + size * 0.72, { size: P ? 26 : 24, align, bg: PAPER });
      };
      if (i === 0) {
        // EVERY / SINGLE / TOKEN, each on its sung word
        const size = P ? 190 : 205;
        const cx = W / 2, cy = P ? H * 0.47 : H * 0.5;
        slab(c, 'EVERY', cx, cy - size * 0.92, size, snapIn(ft, Math.max(sh.t0, tm.every)), { maxW: W * 0.86 });
        slab(c, 'SINGLE', cx, cy, size, snapIn(ft, tm.single), { maxW: W * 0.86 });
        slab(c, 'TOKEN', cx, cy + size * 0.92, size, snapIn(ft, tm.token), { maxW: W * 0.86 });
        label(c, `THE TOKEN IN FLIGHT: ${tok} · ID ${llm.steps[s.step]!.tok.id} · FLOOR ${s.layer}`, cx, cy + size * 1.62, { size: P ? 26 : 26, align: 'center', bg: PAPER });
      } else if (i === 1) {
        // one token: what it has read so far
        const cx = P ? W / 2 : W * 0.3, cy = P ? H * 0.74 : H * 0.7;
        counter(cx, cy, P ? 150 : 140, 'center');
        const tx = P ? W / 2 : W * 0.3, ty = P ? H * 0.25 : H * 0.3;
        slab(c, '1 TOKEN', tx, ty, P ? 170 : 150, kIn, { maxW: P ? W * 0.9 : W * 0.5 });
        label(c, `${fmt2(perTok / 1e9)} GB READ PER TOKEN · COMPUTED`, tx, ty + (P ? 115 : 100), { size: 30, align: 'center', bg: PAPER, color: INK });
        const note = [`${fmt2(total / 1e9)} GB OF WEIGHTS,`, 'EMBEDDING TABLE LOOKED UP (1 ROW)'];
        if (P) note.forEach((t, j) => label(c, t, tx, ty + 160 + j * 38, { size: 24, align: 'center', bg: PAPER, color: BLUE }));
        else label(c, note.join(' '), tx, ty + 146, { size: 22, align: 'center', bg: PAPER, color: BLUE });
      } else if (i === 2) {
        // the inventory: one layer's seven matrices, the one being read now in pink
        const x0 = P ? 90 : 150, y0 = P ? H * 0.52 : top + 70, rowH = P ? 58 : 60, fs = P ? 32 : 34;
        const cur = this.inv.sheets.findLast((q) => q.r <= ff) ?? this.inv.sheets[0]!;
        const lyr = Math.max(1, Math.min(llm.cfg.layers, cur.layer));
        label(c, `LAYER ${lyr} OF ${llm.cfg.layers}: 7 MATRICES`, x0, y0 - rowH * 0.9, { size: fs * 0.8, color: INK, bg: PAPER });
        (['Q', 'K', 'V', 'O', 'GATE', 'UP', 'DOWN'] as Kind[]).forEach((kd, j) => {
          const [rows, cols] = this.inv.shape[kd];
          const on = cur.kind === kd && cur.layer === lyr;
          const y = y0 + j * rowH;
          const kk = snapIn(ft, sh.t0 + j * 0.03);
          if (kk <= 0) return;
          const bw = (this.inv.bytes[kd] / this.inv.bytes.GATE) * (P ? 330 : 420);
          c.fillStyle = on ? PINK : BLUE;
          c.fillRect(x0 + (P ? 150 : 170), y - fs * 0.42, bw * kk, fs * 0.84);
          label(c, kd, x0, y, { size: fs, color: on ? PINK : INK, bg: PAPER });
          label(c, `${fmt1(this.inv.bytes[kd] / 1e6)} MB`, x0 + (P ? 150 : 170) + bw + 16, y, { size: fs * 0.72, bg: PAPER });
          void rows; void cols;
        });
        const yb = y0 + 7 * rowH + 18;
        label(c, `${fmt1(this.inv.layerBytes / 1e6)} MB × ${llm.cfg.layers} + HEAD + 1 EMBED ROW`, x0, yb, { size: fs * 0.8, bg: PAPER });
        counter(P ? W / 2 : W * 0.72, P ? H * 0.3 : H * 0.72, P ? 130 : 140, 'center');
      } else if (i === 3) {
        // the whole pile, and the counter resetting as ·in is written and ·the starts
        counter(P ? W / 2 : W * 0.6, P ? H * 0.78 : H * 0.8, P ? 110 : 100, 'center');
        const [px, py] = toScreen(this.cam, [PILE_T.x - 3, this.tower.height * 0.62, 0]);
        label(c, 'MEMORY', px, py, { size: 30, align: 'right', bg: PAPER, color: BLUE });
        label(c, `${this.inv.sheets.length} SHEETS`, px, py + 40, { size: 26, align: 'right', bg: PAPER });
        const [tx, ty] = toScreen(this.cam, [7.5, this.tower.height * 0.62, 0]);
        label(c, 'THE MODEL', tx, ty, { size: 30, align: 'left', bg: PAPER, color: PINK });
        label(c, `${llm.cfg.layers} FLOORS`, tx, ty + 40, { size: 26, align: 'left', bg: PAPER });
      } else if (i === 4) {
        // memory -> bus -> math, with the counter as the hero
        const lab = (p: [number, number, number], t: string, col = INK, size = P ? 28 : 28, dy = 0) => {
          const [x, y] = toScreen(this.cam, p);
          label(c, t, clamp(x, 150 + labelW(c, t, size) / 2, W - 150 - labelW(c, t, size) / 2), Math.max(top + 30, y) + dy, { size, align: 'center', bg: PAPER, color: col });
        };
        lab([PILE_M.x, this.tower.height + 1.8, 0], `MEMORY · HOLDS ${fmt2(total / 1e9)} GB`);
        lab([(BELT.x0 + BELT.x1) / 2, BELT.y + 2.2, 0], `${fmtInt(GPU.bw / 1e9)} GB/S · ${GPU.name} SPEC`, BLUE);
        lab([(BELT.x0 + BELT.x1) / 2, BELT.y + 2.2, 0], `${fmtInt(N.effBw / 1e9)} GB/S ACHIEVED · COMPUTED`, INK, 24, 40);
        lab([CUBE.c.x, CUBE.c.y + CUBE.s / 2 + 1.4, 0], `MATH · ${fmtInt(GPU.peak / 1e12)} TFLOPS BF16 · SPEC`, PINK);
        counter(W / 2, P ? H * 0.74 : H * 0.8, P ? 150 : 150, 'center');
      } else if (i === 5) {
        // MEMORY
        const cy = P ? H * 0.66 : H * 0.62;
        slab(c, 'MEMORY', W / 2, cy, P ? 200 : 260, snapIn(ft, Math.max(sh.t0, tm.memory)), { maxW: W * 0.9 });
        label(c, `${fmtInt(total)} BYTES OF WEIGHTS IN MEMORY`, W / 2, cy + (P ? 140 : 170), { size: P ? 26 : 30, align: 'center', bg: PAPER });
        label(c, `${fmtInt(perTok)} READ PER TOKEN · COMPUTED`, W / 2, cy + (P ? 184 : 214), { size: P ? 24 : 26, align: 'center', bg: PAPER, color: BLUE });
        counter(P ? W / 2 : W * 0.5, P ? H * 0.3 : H * 0.3, P ? 110 : 110, 'center');
      } else {
        // the math waits: the clock of one token, and the math unit's share
        const waits = i === 7;
        const R = P ? 250 : 215;
        const cx = P ? W / 2 : waits ? W * 0.28 : W * 0.26, cy = P ? (waits ? H * 0.34 : H * 0.36) : H * 0.52;
        this.clock(c, cx, cy, R, s.climb, kIn);
        if (!waits) {
          const tx = P ? W / 2 : W * 0.7, ty = P ? H * 0.66 : H * 0.36;
          slab(c, 'MATH', tx, ty, P ? 190 : 220, snapIn(ft, Math.max(sh.t0, tm.math)), { maxW: P ? W * 0.8 : W * 0.5 });
          label(c, `IN USE: ${(N.util1 * 100).toFixed(2)}% OF THE PEAK`, tx, ty + (P ? 125 : 140), { size: P ? 30 : 32, align: 'center', bg: PAPER, color: PINK });
          label(c, `${fmt2(N.tf1)} OF ${fmtInt(GPU.peak / 1e12)} TFLOPS · COMPUTED`, tx, ty + (P ? 170 : 188), { size: P ? 24 : 26, align: 'center', bg: PAPER });
          // point at the sliver
          const cs = CUBE.s * Math.sqrt(N.util1);
          const [bx, by] = toScreen(this.cam, [CUBE.c.x - CUBE.s / 2 + cs + 0.4, CUBE.c.y - CUBE.s * 0.3, CUBE.c.z + CUBE.s / 2 - cs / 2]);
          label(c, '← BUSY', bx + 14, by, { size: 30, color: PINK, bg: PAPER });
        } else {
          stamp(c, 'WAIT', P ? W / 2 : W * 0.66, P ? H * 0.66 : H * 0.5, P ? 230 : 300, snapIn(ft, sh.t0, 0.07), -0.14);
          label(c, `MATH IDLE ${(100 - N.util1 * 100).toFixed(0)}% OF THE TIME · COMPUTED`, P ? W / 2 : W * 0.66, (P ? H * 0.66 : H * 0.5) + (P ? 190 : 230), { size: P ? 28 : 30, align: 'center', bg: PAPER });
        }
      }
    }), out);

    // ---- foreground: specks (per sub-frame: they move fast)
    const c = this.fx.ctx;
    this.fx.clear();
    specks(c, d * 1.2, { seed: sh.seed, n: 60 });
    this.ctx.comp.draw(r, this.fx.upload(), out);

    return { reg: sh.seed, kinetic: i === 7 ? 0 : 1 }; // the WAIT shot stamps the word itself
  }

  /** One token's clock: a turn is the measured 23.4 ms; blue = reading the weights at spec bandwidth, pink = the math at peak. */
  private clock(c: CanvasRenderingContext2D, x: number, y: number, R: number, climb: number, k: number) {
    const N = this.N, T = N.msTok;
    const a0 = -Math.PI / 2, ang = (ms: number) => a0 + (ms / T) * Math.PI * 2;
    c.save();
    c.translate(x, y);
    const s = 1.2 - 0.2 * k;
    c.scale(s, s);
    // face
    c.fillStyle = PAPER;
    c.beginPath(); c.arc(0, 0, R, 0, Math.PI * 2); c.fill();
    // memory arc (blue) and math wedge (pink)
    c.fillStyle = BLUE;
    c.beginPath(); c.moveTo(0, 0); c.arc(0, 0, R * 0.9, ang(0), ang(N.memMs)); c.closePath(); c.fill();
    c.fillStyle = PAPER;
    c.beginPath(); c.arc(0, 0, R * 0.6, 0, Math.PI * 2); c.fill();
    c.fillStyle = PINK;
    c.beginPath(); c.moveTo(0, 0); c.arc(0, 0, R * 1.02, ang(0) - 0.02, ang(N.mathMs) + 0.02); c.closePath(); c.fill();
    // ticks every ms, labels every 5
    c.strokeStyle = INK; c.lineWidth = 7;
    c.beginPath(); c.arc(0, 0, R, 0, Math.PI * 2); c.stroke();
    for (let ms = 0; ms < T; ms += 1) {
      const t = ang(ms), big = ms % 5 === 0;
      c.lineWidth = big ? 6 : 3;
      c.beginPath(); c.moveTo(Math.cos(t) * R * (big ? 0.84 : 0.9), Math.sin(t) * R * (big ? 0.84 : 0.9)); c.lineTo(Math.cos(t) * R, Math.sin(t) * R); c.stroke();
      if (big && ms > 0) {
        c.font = font(F.mono(700), 24); c.fillStyle = INK; c.textAlign = 'center'; c.textBaseline = 'middle';
        c.fillText(String(ms), Math.cos(t) * R * 1.13, Math.sin(t) * R * 1.13);
      }
    }
    // the hand: the token's progress through its 23.4 ms
    const h = a0 + climb * Math.PI * 2;
    c.lineWidth = 10; c.lineCap = 'round';
    c.beginPath(); c.moveTo(0, 0); c.lineTo(Math.cos(h) * R * 0.95, Math.sin(h) * R * 0.95); c.stroke();
    c.fillStyle = INK; c.beginPath(); c.arc(0, 0, 14, 0, Math.PI * 2); c.fill();
    c.restore();
    // the measured time per token, over the dial; the key under it
    bigNum(c, `${fmt1(T)} MS`, x, y - R * s - 92, 100, {});
    label(c, `PER TOKEN · ${GPU.name}, MEASURED`, x, y - R * s - 30, { size: 24, align: 'center', bg: PAPER });
    const ly = y + R * s + 44;
    const key = `■ MEMORY ${fmt1(N.memMs)} MS  ■ MATH ${fmt2(N.mathMs)} MS · COMPUTED`;
    label(c, key, x, ly, { size: 24, align: 'center', bg: PAPER });
    const tw = labelW(c, key, 24), x0 = x - tw / 2;
    c.fillStyle = BLUE; c.fillRect(x0, ly - 9, 16, 18);
    c.fillStyle = PINK; c.fillRect(x0 + labelW(c, `■ MEMORY ${fmt1(N.memMs)} MS  `, 24), ly - 9, 16, 18);
  }
}
