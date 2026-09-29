// finale (the final chorus, the quiet one: drums out, bass and vocal): a replay of every roll the model
// made. By now 14 of the 15 answer tokens are written; each cut (every 2 beats) replays real steps from
// data/llm.json: that step's nucleus as an odds bar, the needle landing at its real uniform draw u on its
// real pick, the picked tile slamming, in rotating framings (the flat odds bar, the die, a tile on the
// tower) with kit callbacks between them (attention arcs, the tower's floors guessing, a bar-code
// vector). Line 1 rolls one step per cut, line 2 is step 3 ("2" 69.7% vs "3" 30.3%, u = 0.657, the
// biggest beat), line 3 rolls one step per beat, line 4 spins the last die (<END>) and does not land it:
// that happens in the outro.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { clamp, ease, pulse } from '../engine/util';
import { showTok } from '../engine/llm';
import { Wash, specks, makeCam, drive, shot, snapIn, portrait, perFrame, tokenTile, type Shot } from './_kit';
import { Tower, riderTile, SLAB_W } from './_tower';
import {
  PINK, BLUE, INK, slab, label, rain, rainPool, oddsBar, tileRow, arcs, barcode, makeDie, rollDie, worldAt, toScreen,
  flatCam, pct, safeClip, safeBox, type Die,
} from './finale-kit';

type Kind = 'rewind' | 'dice2' | 'tower' | 'arcs' | 'lens' | 'tumble' | 'bigbar' | 'said' | 'roof' | 'bars' | 'die9' | 'fall' | 'code'
  | 'endA' | 'endB' | 'endC' | 'endD' | 'endE';
interface Roll { s: number; start: number; land: number }
interface Plan { kind: Kind; rolls: Roll[] }

const N_STEPS = 15;
/** The attention callback: step 3 (the digit), layer index 18 (floor 19), head index 4: its strongest look at " strawberry". */
const STEP_BIG = 3, ATT_LAYER = 18, ATT_HEAD = 4;

export default class Finale extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(36);
  /** The camera at the frame's own time (labels pinned to 3D points are drawn once per frame). */
  camFT = makeCam(36);
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  front = new Layer2D();
  tower = new Tower();
  rider!: THREE.Mesh;
  dice = new Map<number, Die>();
  tiles: THREE.Mesh[] = [];
  three!: THREE.Mesh;
  /** The token being written (unknown yet): a pink "?" tile. */
  ask!: THREE.Mesh;
  plan: Plan[] = [];
  pool: string[] = [];
  beat = 0.345;
  /** Step 3, layer index 18 (floor 19), head 4: attention over the question + the answer so far. */
  att: number[] = [];
  ctxToks: string[] = [];
  kvShown = -1;
  /** When "dice" is sung in the final chorus (the scene stages that word itself, over step 3's roll). */
  diceT = 0;
  lastKind = '';

  override init() {
    const { llm, audio: au, lyrics: ly } = this.ctx;
    this.pool = rainPool(llm);
    if (llm.steps.length !== N_STEPS) console.warn(`finale: the run has ${llm.steps.length} steps, the scene is laid out for ${N_STEPS}`);
    const b0 = Math.round(au.beatAt(this.ctx.start));
    const bt = (k: number) => au.timeOfBeat(b0 + k);
    this.beat = bt(1) - bt(0);
    const diceW = ly.get('One more roll of the dice', 2).words.find((w) => w.w.toLowerCase().startsWith('dice'));
    this.diceT = diceW?.start ?? bt(13);
    const dice = clamp(this.diceT, bt(12) + 0.25, bt(14) - 0.08);
    const one = (i: number, s: number): Roll[] => [{ s, start: bt(2 * i) + 0.03, land: bt(2 * i + 1) }];
    const two = (i: number, s: number): Roll[] => [{ s, start: bt(2 * i) + 0.02, land: bt(2 * i + 0.5) }, { s: s + 1, start: bt(2 * i + 1), land: bt(2 * i + 1.5) }];
    this.plan = [
      { kind: 'rewind', rolls: [] },
      { kind: 'dice2', rolls: two(1, 0) },
      { kind: 'tower', rolls: one(2, 2) },
      { kind: 'arcs', rolls: [] },
      { kind: 'lens', rolls: [] },
      { kind: 'tumble', rolls: [] },
      { kind: 'bigbar', rolls: [{ s: 3, start: bt(12) + 0.06, land: dice }] },
      { kind: 'said', rolls: [] },
      { kind: 'roof', rolls: two(8, 4) },
      { kind: 'bars', rolls: two(9, 6) },
      { kind: 'die9', rolls: two(10, 8) },
      { kind: 'fall', rolls: two(11, 10) },
      { kind: 'code', rolls: two(12, 12) },
      { kind: 'endA', rolls: [] }, { kind: 'endB', rolls: [] }, { kind: 'endC', rolls: [] }, { kind: 'endD', rolls: [] }, { kind: 'endE', rolls: [] },
    ];

    // 3D: the tower, the dice of the steps that get one, a printed tile per answer token
    this.tower.outlineAll();
    this.rider = riderTile('?');
    this.tower.ride(this.rider, 0.9);
    this.scene.add(this.tower.group);
    for (const s of [0, 1, 3, 9, 14]) { const d = makeDie(llm, s); this.dice.set(s, d); this.scene.add(d.mesh); }
    llm.steps.forEach((st) => {
      const m = tokenTile(st.tok.t, { face: 'pink', text: 'paper', side: 'blue', id: st.tok.id, h: 1.1 });
      this.scene.add(m);
      this.tiles.push(m);
    });
    // step 3's other candidate, "3" (its id from the prompt's "2023")
    const id3 = llm.d.prompt.find((x) => x.t === '3')?.id;
    this.three = tokenTile('3', { face: 'blue', text: 'paper', side: 'ink', id: id3, h: 1.1 });
    this.scene.add(this.three);
    this.ask = tokenTile('?', { face: 'pink', text: 'paper', side: 'ink', h: 1.1 });
    this.scene.add(this.ask);

    // attention callback: step 3 (writing the digit), layer index 18 = floor 19
    const q = llm.d.question_tokens, qs = llm.questionStart, np = llm.d.prompt.length;
    this.ctxToks = [...q, ...llm.steps.slice(0, 3).map((s) => s.tok.t)];
    const row = llm.attn(STEP_BIG, ATT_LAYER, ATT_HEAD);
    this.att = [...q.map((_, i) => row[qs + i]!), ...[0, 1, 2].map((k) => row[np + k]!)];
  }

  // ---------------------------------------------------------------- helpers

  private rollK(ro: Roll, t: number) {
    const sweep = clamp((t - ro.start) / Math.max(0.05, ro.land - ro.start));
    return { needle: t < ro.start ? 0 : ease.outCubic(sweep), landed: t >= ro.land, slam: snapIn(t, ro.land) };
  }
  /** The roll being shown at t: the latest one that has started (or the first). */
  private cur(pl: Plan, t: number): Roll | undefined {
    let r = pl.rolls[0];
    for (const x of pl.rolls) if (t >= x.start - 0.05) r = x;
    return r;
  }

  private hideAll() {
    this.tower.group.visible = false;
    for (const d of this.dice.values()) d.mesh.visible = false;
    for (const t of this.tiles) t.visible = false;
    this.three.visible = false;
    this.ask.visible = false;
    this.rider.visible = false;
  }

  /** A picked tile slamming in at `land`: hidden before, lands 1.6x big and snaps to size in 80 ms. */
  private slamTile(m: THREE.Mesh, t: number, land: number, pos: THREE.Vector3, size: number, d: number, face?: THREE.Vector3) {
    if (t < land) return;
    const k = snapIn(t, land);
    m.visible = true;
    m.position.copy(pos);
    m.scale.setScalar(size * (1 + 0.6 * (1 - k)));
    m.up.set(0, 1, 0);
    if (face) m.lookAt(face);
    else m.rotation.set(0.12 + Math.sin(d * 0.9) * 0.08, Math.sin(d * 0.7) * 0.3, 0);
  }

  private setCam(cam: THREE.PerspectiveCamera, pos: THREE.Vector3, look: THREE.Vector3, fov = 36, shift = 0) {
    cam.position.copy(pos);
    cam.up.set(0, 1, 0);
    cam.lookAt(look);
    if (cam.fov !== fov || cam.filmOffset !== shift) { cam.fov = fov; cam.filmOffset = shift; cam.updateProjectionMatrix(); }
  }
  private flat(cam: THREE.PerspectiveCamera, D = 10) {
    if (cam.filmOffset !== 0) { cam.filmOffset = 0; cam.updateProjectionMatrix(); }
    flatCam(cam, D);
  }

  private resetTower(kind: string) {
    if (kind === this.lastKind) return;
    this.lastKind = kind;
    const tw = this.tower;
    tw.clearHeads();
    for (let l = 1; l <= tw.layers; l++) (tw.ff[l - 1]!.material as THREE.ShaderMaterial).uniforms.hot!.value = 0;
    tw.light(-1);
  }

  private setKV(n: number, newest = 0) {
    const key = n * 10 + newest;
    if (key === this.kvShown) return;
    this.kvShown = key;
    this.tower.setKV(n, newest);
  }

  /** Horizontal unit vector from the tower's axis toward the camera. */
  private toCam(cam: THREE.PerspectiveCamera) {
    return new THREE.Vector3(cam.position.x, 0, cam.position.z).normalize();
  }

  // ---------------------------------------------------------------- 3D staging

  private stage(pl: Plan, sh: Shot, t: number, d: number, P: boolean, cam: THREE.PerspectiveCamera) {
    const au = this.ctx.audio, llm = this.ctx.llm, story = this.ctx.story;
    this.hideAll();
    this.resetTower(pl.kind);
    const tw = this.tower, Ht = tw.height, dist = P ? 1.6 : 1;
    const spin = drive(au, t, 2.2, 2);
    const a = d * 0.12 + sh.i * 1.3;
    const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
    const ro = this.cur(pl, t);
    const np = llm.d.prompt.length;
    switch (pl.kind) {
      case 'rewind': {
        tw.group.visible = true;
        this.rider.visible = true;
        tw.ride(this.rider, story.at(t).climb);
        tw.light((d * 0.5) % 1);
        this.setKV(np + 14);
        this.setCam(cam, V(Math.cos(a) * 34 * dist, Ht * 0.45, Math.sin(a) * 34 * dist), V(0, Ht * 0.5, 0), 40, P ? 0 : -7);
        break;
      }
      case 'dice2': {
        this.flat(cam, P ? 12 : 10);
        pl.rolls.forEach((r, j) => {
          const die = this.dice.get(r.s)!;
          die.mesh.visible = true;
          die.mesh.position.copy(worldAt(cam, P ? 0.5 : j ? 0.7 : 0.3, P ? (j ? 0.6 : 0.29) : 0.46));
          rollDie(die, spin + j * 2, t, r.land, r.s * 5);
          die.mesh.scale.setScalar((P ? 0.62 : 0.95) * (t >= r.land ? 1 + 0.15 * (1 - snapIn(t, r.land, 0.12)) : 1));
        });
        break;
      }
      case 'tower': {
        tw.group.visible = true;
        this.setKV(0);
        const climb = t < ro!.land ? 0.72 + 0.28 * ease.inCubic(clamp((t - sh.t0) / Math.max(0.1, ro!.land - sh.t0))) : 1;
        tw.ride(this.rider, Math.min(climb, 0.985));
        this.rider.visible = t < ro!.land;
        tw.light(climb);
        this.setCam(cam, V(Math.cos(a) * 14 * dist, Ht + 3.5, Math.sin(a) * 14 * dist), V(0, Ht - 3.2, 0), 40, P ? 0 : -6);
        this.slamTile(this.tiles[2]!, t, ro!.land, V(0, Ht + 1.2, 0), 1.3, d, cam.position.clone().setY(Ht + 1.2));
        break;
      }
      case 'arcs': {
        this.flat(cam, 10);
        const [sx, sy] = this.slotAt;
        this.ask.visible = true;
        this.ask.position.copy(worldAt(cam, sx / W, sy / H, 0.4));
        this.ask.scale.setScalar(P ? 0.42 : 0.5);
        this.ask.rotation.set(0.1, Math.sin(d * 0.9) * 0.4, 0);
        break;
      }
      case 'lens': {
        tw.group.visible = true;
        this.setKV(np + 3);
        const lens = llm.steps[3]!.lens;
        for (let l = 1; l <= tw.layers; l++) {
          const g = lens[l]!.top[0]![0].trim();
          (tw.slabs[l]!.material as THREE.ShaderMaterial).uniforms.hot!.value = g === 'two' || g === '2' ? 1 : 0;
          (tw.ff[l - 1]!.material as THREE.ShaderMaterial).uniforms.hot!.value = g === 'three' ? 1 : 0;
        }
        this.rider.visible = true;
        tw.ride(this.rider, (12 + 21 * clamp((t - sh.t0) / (sh.t1 - sh.t0))) / 33);
        this.setCam(cam, V(Math.cos(a) * 36 * (P ? 1.45 : 1), Ht * 0.56, Math.sin(a) * 36 * (P ? 1.45 : 1)), V(0, Ht * 0.58, 0), 40, P ? -1.6 : -5);
        break;
      }
      case 'tumble': {
        this.flat(cam, P ? 11 : 9.5);
        const die = this.dice.get(3)!;
        die.mesh.visible = true;
        die.mesh.position.copy(worldAt(cam, P ? 0.5 : 0.7, P ? 0.34 : 0.5));
        die.mesh.scale.setScalar(P ? 0.95 : 1.1);
        rollDie(die, spin * 1.4, t, Infinity, 3);
        break;
      }
      case 'bigbar': {
        this.flat(cam, 10);
        const [bx, , bw] = this.bigBarBox(P);
        const p0 = llm.steps[3]!.nucleus[0]![1];
        const k = this.rollK(ro!, t);
        const ty = P ? 0.63 : 0.79;
        const two = this.tiles[3]!, three = this.three;
        two.visible = three.visible = true;
        two.position.copy(worldAt(cam, (bx + bw * p0 / 2) / W, ty));
        two.scale.setScalar(k.landed ? 1.35 * (1 + 0.5 * (1 - k.slam)) : 1);
        two.rotation.set(0.1, Math.sin(d * 0.8) * 0.3, 0);
        three.position.copy(worldAt(cam, (bx + bw * (p0 + (1 - p0) / 2)) / W, ty));
        const drop = k.landed ? ease.outCubic(clamp((t - ro!.land) / 0.2)) : 0;
        three.position.y -= drop * 0.45;
        three.scale.setScalar(1 - 0.3 * drop);
        three.rotation.set(0.1 + drop * 0.5, Math.sin(d * 0.8 + 1) * 0.3, drop * -0.3);
        break;
      }
      case 'said': {
        this.flat(cam, 10);
        const two = this.tiles[3]!;
        const k = snapIn(t, sh.t0, 0.09);
        two.visible = true;
        two.position.copy(worldAt(cam, P ? 0.5 : 0.34, P ? 0.42 : 0.5));
        two.scale.setScalar((P ? 2.1 : 2.5) * (1 + 0.7 * (1 - k)));
        two.rotation.set(0.08, Math.sin(d * 0.6) * 0.18 - 0.1, 0);
        break;
      }
      case 'roof': {
        tw.group.visible = true;
        this.setKV(0);
        tw.light(1);
        this.setCam(cam, V(Math.cos(a) * 9 * dist, Ht + 6.5, Math.sin(a) * 9 * dist), V(0, Ht - 0.5, 0), 44, P ? 0 : -6);
        const toC = this.toCam(cam);
        const side = new THREE.Vector3(-toC.z, 0, toC.x);
        pl.rolls.forEach((r, j) => {
          if (t < r.land) return;
          const m = this.tiles[r.s]!;
          const k = snapIn(t, r.land);
          m.visible = true;
          m.position.copy(side.clone().multiplyScalar((j ? -1 : 1) * 1.35)).add(toC.clone().multiplyScalar(0.6));
          m.position.y = Ht + 0.3 + 2.5 * (1 - k);
          m.scale.setScalar(1.1 * (1 + 0.4 * (1 - k)));
          m.up.copy(toC).negate();
          m.lookAt(m.position.x, m.position.y + 1, m.position.z);
        });
        break;
      }
      case 'bars': {
        this.flat(cam, 10);
        const L = this.barsLayout(P);
        pl.rolls.forEach((r, j) => {
          const b = L[j]!;
          this.slamTile(this.tiles[r.s]!, t, r.land, worldAt(cam, b.tx / W, b.ty / H), P ? 0.9 : 1.15, d + j);
        });
        break;
      }
      case 'die9': {
        this.flat(cam, P ? 11 : 9.5);
        const die = this.dice.get(9)!;
        die.mesh.visible = true;
        die.mesh.position.copy(worldAt(cam, P ? 0.5 : 0.76, P ? 0.63 : 0.32));
        const rB = pl.rolls[1]!;
        rollDie(die, spin * 1.2, t, rB.land, 9);
        die.mesh.scale.setScalar((P ? 0.9 : 0.85) * (t >= rB.land ? 1 + 0.15 * (1 - snapIn(t, rB.land, 0.12)) : 1));
        break;
      }
      case 'fall': {
        tw.group.visible = true;
        const landed = pl.rolls.filter((r) => t >= r.land).length;
        this.setKV(np + 10 + landed, 1);
        this.setCam(cam, V(Math.cos(a) * 36 * dist, Ht * 0.52, Math.sin(a) * 36 * dist), V(0, Ht * 0.53, 0), 40, P ? 0 : -4);
        const front = this.toCam(cam).multiplyScalar(SLAB_W * 0.78);
        let lit = -1;
        for (const r of pl.rolls) {
          if (t < r.land) continue;
          const m = this.tiles[r.s]!;
          const fall = ease.inQuad(clamp((t - r.land) / (this.beat * 0.95)));
          m.visible = true;
          m.position.copy(front).setY(Ht + 0.6 - (Ht - 0.1) * fall);
          m.scale.setScalar(1.7);
          m.up.set(0, 1, 0);
          m.lookAt(cam.position.x, m.position.y, cam.position.z);
          lit = m.position.y / (Ht + 0.3);
        }
        tw.light(lit);
        break;
      }
      case 'code': {
        this.flat(cam, 10);
        pl.rolls.forEach((r, j) => {
          this.slamTile(this.tiles[r.s]!, t, r.land, worldAt(cam, P ? 0.3 + 0.4 * j : 0.84, P ? 0.7 : 0.33 + 0.3 * j), P ? 0.85 : 1.05, d + j);
        });
        break;
      }
      case 'endA': case 'endE': {
        this.flat(cam, P ? 11 : 10);
        const die = this.dice.get(14)!;
        die.mesh.visible = true;
        const right = pl.kind === 'endA';
        die.mesh.position.copy(worldAt(cam, P ? 0.5 : right ? 0.76 : 0.25, P ? 0.63 : 0.56));
        die.mesh.scale.setScalar(P ? 1.05 : 1.25);
        rollDie(die, spin * 1.5, t, Infinity, right ? 14 : 3);
        break;
      }
      case 'endB': {
        tw.group.visible = true;
        this.rider.visible = true;
        tw.ride(this.rider, story.at(t).climb);
        tw.light((d * 0.45) % 1);
        this.setKV(np + 14);
        const die = this.dice.get(14)!;
        die.mesh.visible = true;
        die.mesh.position.set(0, Ht + 3.9, 0);
        die.mesh.scale.setScalar(1.4);
        rollDie(die, spin * 1.5, t, Infinity, 14);
        this.setCam(cam, V(Math.cos(a) * 23 * dist, Ht * 0.8, Math.sin(a) * 23 * dist), V(0, Ht * 0.93, 0), 44, P ? 0 : -5);
        break;
      }
      case 'endC': {
        this.flat(cam, P ? 10 : 5.4);
        const die = this.dice.get(14)!;
        die.mesh.visible = true;
        die.mesh.position.set(P ? 0 : 0.9, P ? 1.1 : 0.2, 0);
        die.mesh.scale.setScalar(1);
        rollDie(die, spin * 1.3, t, Infinity, 7);
        break;
      }
      case 'endD': {
        this.flat(cam, P ? 11 : 10);
        const die = this.dice.get(14)!;
        die.mesh.visible = true;
        die.mesh.position.copy(worldAt(cam, P ? 0.5 : 0.27, P ? 0.33 : 0.5));
        die.mesh.scale.setScalar(1.2);
        rollDie(die, spin * 1.5, t, Infinity, 21);
        break;
      }
    }
  }

  // ---------------------------------------------------------------- 2D layouts

  /** Screen position of the attention shot's empty slot (set by draw2D, used by the 3D "?" tile). */
  slotAt: [number, number] = [W * 0.8, H * 0.46];

  private bigBarBox(P: boolean): [number, number, number, number] {
    return P ? [140, H * 0.26, W - 280, 140] : [W * 0.1, H * 0.26, W * 0.78, 150];
  }
  private barsLayout(P: boolean) {
    return P
      ? [{ x: 140, y: H * 0.47, w: W - 280, h: 90, tx: W * 0.3, ty: H * 0.24 }, { x: 140, y: H * 0.66, w: W - 280, h: 90, tx: W * 0.7, ty: H * 0.24 }]
      : [{ x: W * 0.1, y: H * 0.22, w: W * 0.58, h: 100, tx: W * 0.83, ty: H * 0.27 }, { x: W * 0.1, y: H * 0.47, w: W * 0.58, h: 100, tx: W * 0.83, ty: H * 0.52 }];
  }

  /** "ROLL n / 15" and what the draw looked like. */
  private rollHead(c: CanvasRenderingContext2D, s: number, x: number, y: number, align: CanvasTextAlign = 'left', size = 26) {
    const st = this.ctx.llm.steps[s]!;
    const n = st.nucleus_size;
    label(c, `ROLL ${s + 1} / ${N_STEPS} · ${n === 1 ? 'TOP-P LEFT 1 TOKEN' : `${n} TOKENS IN THE DRAW`}`, x, y, size, INK, align);
  }

  /** A compact roll: header, bar, needle. */
  private compact(c: CanvasRenderingContext2D, ro: Roll, t: number, x: number, y: number, w: number, h = 92) {
    const st = this.ctx.llm.steps[ro.s]!;
    const k = this.rollK(ro, t);
    this.rollHead(c, ro.s, x, y - 84);
    oddsBar(c, st, x, y, w, h, { reveal: snapIn(t, ro.start - 0.06), needle: k.needle, landed: k.landed });
  }

  private draw2D(pl: Plan, sh: Shot, t: number, P: boolean) {
    const c = this.mid.ctx, llm = this.ctx.llm, au = this.ctx.audio, story = this.ctx.story;
    c.save();
    safeClip(c);
    const [x0, , x1] = safeBox();
    const ro = this.cur(pl, t);
    const blink = Math.floor(au.beatAt(t) * 2) % 2 === 0;
    switch (pl.kind) {
      case 'rewind': {
        // the answer so far, rewinding tile by tile: the replay starts from the question
        const toks = llm.steps.slice(0, 14).map((s) => s.tok.t);
        const n = Math.round(14 * (1 - clamp((t - (sh.t1 - 0.62)) / 0.56)));
        tileRow(c, toks.slice(0, n), x0, P ? H * 0.2 : H * 0.26, P ? 34 : 40, P ? x1 : W * 0.62, { slot: true, slotOn: blink });
        slab(c, 'REPLAY', P ? W / 2 : x0, P ? H * 0.72 : H * 0.62, P ? 150 : 190, { align: P ? 'center' : 'left', top: INK, under: PINK });
        label(c, `${n} / 14 TOKENS · REWIND`, P ? W / 2 : x0, P ? H * 0.79 : H * 0.76, 30, PINK, P ? 'center' : 'left');
        break;
      }
      case 'dice2': {
        pl.rolls.forEach((r, j) => {
          const st = llm.steps[r.s]!;
          const k = this.rollK(r, t);
          const cx = P ? W / 2 : W * (j ? 0.7 : 0.3), top = P ? H * (j ? 0.465 : 0.155) : H * 0.2;
          const by = P ? H * (j ? 0.72 : 0.41) : H * 0.74, bw = P ? W - 300 : W * 0.3;
          label(c, `ROLL ${r.s + 1} / ${N_STEPS}`, cx, top, 30, INK, 'center');
          oddsBar(c, st, cx - bw / 2, by, bw, 64, { needle: k.needle, landed: k.landed, ruler: false, fs: 30 });
        });
        break;
      }
      case 'tower': {
        if (P) this.compact(c, ro!, t, 140, H * 0.7, W - 280);
        else this.compact(c, ro!, t, x0, H * 0.6, W * 0.36);
        break;
      }
      case 'arcs': {
        // where the next token looks (real attention, step 3, floor 19, head 5): the question, and mostly "strawberry"
        const size = P ? 42 : 38;
        const boxes = tileRow(c, this.ctxToks, x0, P ? H * 0.6 : H * 0.48, size, x1, { slot: true, slotOn: true, center: true, hot: [6] });
        const slot = boxes[boxes.length - 1]!;
        this.slotAt = [slot.x + slot.w / 2, slot.y];
        const k = clamp((t - sh.t0) / 0.2);
        const hot = this.ctxToks.indexOf(' strawberry');
        const lift = P ? 1.1 : 1.3;
        arcs(c, slot, boxes.slice(0, -1), this.att, { k: ease.outCubic(k), hot, lift });
        const sb = boxes[hot]!;
        const x0a = slot.x + slot.w / 2, x1a = sb.x + sb.w / 2;
        const cy = Math.min(slot.y, sb.y) - slot.h / 2 - (80 + Math.hypot(x1a - x0a, slot.y - sb.y) * 0.42) * lift;
        const apexY = 0.25 * (slot.y - slot.h / 2) + 0.5 * cy + 0.25 * (sb.y - sb.h / 2);
        slab(c, pct(this.att[hot]!, 0), (x0a + x1a) / 2, apexY - (P ? 70 : 80), P ? 110 : 140, { k: snapIn(t, sh.t0 + 0.12) });
        const ly = P ? H * 0.2 : H * 0.8;
        label(c, `FLOOR ${ATT_LAYER + 1} · HEAD ${ATT_HEAD + 1} OF ${llm.cfg.heads}`, P ? W / 2 : x0, ly, 30, INK, P ? 'center' : 'left');
        label(c, `WRITING TOKEN ${STEP_BIG + 1}, IT LOOKS AT ·strawberry`, P ? W / 2 : x0, ly + 42, P ? 24 : 30, PINK, P ? 'center' : 'left');
        break;
      }
      case 'lens': {
        const lens = llm.steps[3]!.lens;
        label(c, `WRITING TOKEN ${STEP_BIG + 1}:`, x0, P ? H * 0.17 : H * 0.19, 30, INK);
        label(c, `EACH FLOOR'S BEST GUESS`, x0, (P ? H * 0.17 : H * 0.19) + 40, 30, PINK);
        const right = this.camRight(this.camFT).multiplyScalar(-SLAB_W * 0.62);
        for (const l of [18, 20, 22, 26, 29, 32]) {
          const [g, p] = lens[l]!.top[0]!;
          const [ex, ey] = toScreen(new THREE.Vector3(0, this.tower.floorY(l) + 0.05, 0).add(right), this.camFT);
          if (ey < 200 || ey > H - 150) continue;
          if (t < sh.t0 + ((l - 12) / 21) * (sh.t1 - sh.t0) - 0.02) continue;
          const three = g.trim() === 'three';
          const col = three ? BLUE : l === 32 ? PINK : INK;
          c.fillStyle = col;
          c.fillRect(ex - 40, ey - 2, 40, 4);
          label(c, `L${l}  ${showTok(g)}  ${pct(p)}`, ex - 52, ey, l === 32 || l === 18 ? 36 : 30, col, 'right');
        }
        break;
      }
      case 'tumble': {
        const st = llm.steps[3]!;
        const [a, b] = st.nucleus;
        if (P) {
          slab(c, 'ROLL 4', W / 2, H * 0.6, 96, { under: BLUE, top: INK });
          slab(c, a![0], W * 0.32, H * 0.7, 150, { top: PINK });
          slab(c, b![0], W * 0.68, H * 0.7, 150, { top: BLUE, under: null });
          if (t >= sh.t0 + this.beat) {
            slab(c, pct(a![1]), W * 0.32, H * 0.765, 44, { top: PINK, under: null, k: snapIn(t, sh.t0 + this.beat) });
            slab(c, pct(b![1]), W * 0.68, H * 0.765, 44, { top: BLUE, under: null, k: snapIn(t, sh.t0 + this.beat) });
          }
        } else {
          slab(c, 'ROLL 4', x0, H * 0.24, 110, { align: 'left', under: BLUE, top: INK });
          slab(c, a![0], x0 + 90, H * 0.52, 230, { top: PINK });
          slab(c, b![0], x0 + 420, H * 0.52, 230, { top: BLUE, under: null });
          if (t >= sh.t0 + this.beat) {
            slab(c, pct(a![1]), x0 + 90, H * 0.7, 52, { top: PINK, under: null, k: snapIn(t, sh.t0 + this.beat) });
            slab(c, pct(b![1]), x0 + 420, H * 0.7, 52, { top: BLUE, under: null, k: snapIn(t, sh.t0 + this.beat) });
          }
          label(c, `TOP-P ${llm.d.sampling.top_p} · ${st.nucleus_size} TOKENS IN THE DRAW`, x0, H * 0.8, 26, INK);
        }
        break;
      }
      case 'bigbar': {
        const st = llm.steps[3]!;
        const [bx, by, bw, bh] = this.bigBarBox(P);
        const k = this.rollK(ro!, t);
        // the needle creeps: slow into the landing
        const creep = t < ro!.start ? 0 : ease.outQuart(clamp((t - ro!.start) / (ro!.land - ro!.start)));
        this.rollHead(c, 3, bx, by - (P ? 100 : 108), 'left', 28);
        oddsBar(c, st, bx, by, bw, bh, { needle: creep, landed: k.landed, fs: P ? 40 : 46 });
        // the 2 | 3 boundary (the odds of "2")
        const p0 = st.nucleus[0]![1];
        c.fillStyle = INK;
        c.fillRect(bx + bw * p0 - 2, by + bh + 4, 4, 44);
        label(c, p0.toFixed(3), bx + bw * p0 + 10, by + bh + 64, 26, INK, 'left');
        // the sung word, staged here (the overlay's slam would cover the bar)
        if (t >= this.diceT) slab(c, 'DICE', W / 2, P ? H * 0.47 : H * 0.575, P ? 190 : 200, { k: snapIn(t, this.diceT), top: PINK, under: BLUE });
        break;
      }
      case 'said': {
        const st = llm.steps[3]!;
        const k = snapIn(t, sh.t0, 0.08);
        const [p2, p3] = [st.nucleus[0]![1], st.nucleus[1]![1]];
        if (P) {
          slab(c, 'THE DICE SAID', W / 2, H * 0.2, 88, { k, top: INK, under: BLUE });
          label(c, `2 · ${pct(p2)}`, W * 0.3, H * 0.6, 44, PINK, 'center');
          label(c, `3 · ${pct(p3)}`, W * 0.7, H * 0.6, 44, BLUE, 'center');
        } else {
          slab(c, 'THE DICE SAID', x0, H * 0.2, 118, { k, align: 'left', top: INK, under: BLUE });
          slab(c, '3', W * 0.72, H * 0.46, 300, { top: BLUE, under: null, k: snapIn(t, sh.t0 + this.beat * 0.5) });
          label(c, `${pct(p3)} · NOT PICKED`, W * 0.72, H * 0.63, 34, BLUE, 'center');
          label(c, pct(p2), W * 0.34, H * 0.83, 40, PINK, 'center');
        }
        // the ruler, close up: u landed 0.040 short of the 3
        const lo = 0.6, hi = 0.76;
        const rx = P ? 140 : W * 0.5, rw = P ? W - 280 : W * 0.3, ry = P ? H * 0.68 : H * 0.8 - 40;
        const X = (v: number) => rx + ((v - lo) / (hi - lo)) * rw;
        c.fillStyle = PINK; c.fillRect(rx, ry, X(p2) - rx, 34);
        c.fillStyle = BLUE; c.fillRect(X(p2), ry, rx + rw - X(p2), 34);
        c.fillStyle = INK;
        for (let v = lo; v <= hi + 1e-6; v += 0.01) c.fillRect(X(v) - 1.5, ry + 34, 3, Math.round(v * 100) % 5 === 0 ? 20 : 10);
        const ux = X(st.u);
        c.fillRect(ux - 4, ry - 30, 8, 94);
        label(c, `u ${st.u.toFixed(3)}`, ux - 12, ry - 52, 28, INK, 'right');
        label(c, `${p2.toFixed(3)}`, X(p2) + 12, ry - 52, 28, INK, 'left');
        label(c, `${(p2 - st.u).toFixed(3)} SHORT OF 3 (${p2.toFixed(3)} − ${st.u.toFixed(3)})`, rx, ry + 92, 24, INK, 'left');
        break;
      }
      case 'roof': {
        if (P) this.compact(c, ro!, t, 140, H * 0.7, W - 280);
        else this.compact(c, ro!, t, x0, H * 0.62, W * 0.36);
        break;
      }
      case 'bars': {
        const L = this.barsLayout(P);
        pl.rolls.forEach((r, j) => {
          const b = L[j]!;
          const k = this.rollK(r, t);
          this.rollHead(c, r.s, b.x, b.y - 76);
          oddsBar(c, llm.steps[r.s]!, b.x, b.y, b.w, b.h, { reveal: snapIn(t, r.start - 0.08 - (j ? this.beat * 0.5 : 0)), needle: k.needle, landed: k.landed });
        });
        break;
      }
      case 'die9': {
        const [rA, rB] = pl.rolls as [Roll, Roll];
        const kA = this.rollK(rA, t), kB = this.rollK(rB, t);
        const st = llm.steps[9]!;
        const ax = P ? 140 : x0, ay = P ? H * 0.19 : H * 0.23, aw = P ? W - 280 : W * 0.42;
        this.rollHead(c, 8, ax, ay - 92);
        oddsBar(c, llm.steps[8]!, ax, ay, aw, 64, { needle: kA.needle, landed: kA.landed, ruler: false });
        // step 9's draw as a list: it landed on the second most likely
        const lx = P ? 150 : x0, ly = P ? H * 0.29 : H * 0.7, lw = P ? 360 : 440, rh = P ? 54 : 60;
        this.rollHead(c, 9, lx, ly - 52);
        st.nucleus.forEach(([tok, p], i) => {
          const y = ly + i * rh + 6;
          const hot = kB.landed && i === st.pick;
          tileRow(c, [tok], lx, y, 34, lx + 300, { hot: hot ? [0] : [] });
          c.fillStyle = hot ? PINK : i % 2 ? BLUE : INK;
          c.fillRect(lx + 280, y - 18, lw * p, 36);
          label(c, pct(p), lx + 280 + lw * p + 14, y, 30, hot ? PINK : INK);
        });
        if (kB.needle > 0) label(c, `u = ${st.u.toFixed(3)} · PAST ${pct(st.nucleus[0]![1])}`, P ? lx : lx + 720, P ? ly + 3 * rh + 10 : ly + rh, 28, PINK);
        break;
      }
      case 'fall': {
        if (P) this.compact(c, ro!, t, 140, H * 0.7, W - 280);
        else this.compact(c, ro!, t, x0, H * 0.28, W * 0.34);
        const landed = pl.rolls.filter((r) => t >= r.land).length;
        const n = llm.d.prompt.length + 10 + landed;
        label(c, `FED BACK IN: KV CACHE ${n} TOKENS`, P ? W / 2 : x0, P ? H * 0.2 : H * 0.8, 30, INK, P ? 'center' : 'left');
        break;
      }
      case 'code': {
        const vals = llm.d.embed.first64[' strawberry'] ?? [];
        const id = llm.d.prompt.find((x) => x.t === ' strawberry')?.id ?? 0;
        const hid = llm.cfg.hidden.toLocaleString('en-US');
        const bx = P ? 140 : x0, bw = P ? W - 280 : W * 0.6, by = P ? H * 0.32 : H * 0.4;
        if (P) {
          label(c, `IT READ ·strawberry: 1 TOKEN, ID ${id}`, bx, H * 0.16, 26, INK);
          label(c, `${hid} NUMBERS (FIRST ${vals.length} SHOWN)`, bx, H * 0.16 + 36, 26, INK);
        } else label(c, `IT READ ·strawberry: 1 TOKEN, ID ${id}, ${hid} NUMBERS (FIRST ${vals.length})`, bx, H * 0.19, 28, INK);
        barcode(c, vals, bx, by, bw, P ? 100 : 120, clamp((t - sh.t0) / 0.25));
        const wrote = llm.steps.slice(10, 13).map((s) => s.tok.t);
        const shown = wrote.filter((_, j) => j < 2 || t >= pl.rolls[0]!.land);
        label(c, `IT WROTE IT AS ${wrote.length} TOKENS:`, bx, P ? H * 0.46 : H * 0.62, P ? 26 : 30, INK);
        tileRow(c, shown, bx, P ? H * 0.52 : H * 0.7, P ? 52 : 60, bx + bw, { hot: t >= pl.rolls[0]!.land ? [2] : [] });
        break;
      }
      case 'endA': case 'endE': {
        const toks = llm.steps.slice(0, 14).map((s) => s.tok.t);
        const right = pl.kind === 'endA';
        const ax = P ? x0 : right ? x0 : W * 0.46, amax = P ? x1 : right ? W * 0.6 : x1;
        tileRow(c, toks, ax, P ? H * 0.16 : H * 0.24, P ? 32 : 36, amax, { slot: true, slotOn: blink });
        const n = story.at(t).emitted;
        const cx = P ? W / 2 : (ax + amax) / 2;
        slab(c, `${n} / ${N_STEPS}`, cx, P ? H * 0.35 : H * 0.58, P ? 130 : 190, { k: snapIn(t, sh.t0 + 0.05) });
        label(c, 'TOKENS WRITTEN', cx, P ? H * 0.41 : H * 0.72, 30, INK, 'center');
        break;
      }
      case 'endB': {
        label(c, `ROLL ${N_STEPS} / ${N_STEPS}`, P ? W / 2 : x0, P ? H * 0.16 : H * 0.2, 34, INK, P ? 'center' : 'left');
        label(c, 'STILL CLIMBING THE TOWER', P ? W / 2 : x0, (P ? H * 0.16 : H * 0.2) + 46, 30, PINK, P ? 'center' : 'left');
        if (!P) tileRow(c, llm.steps.slice(0, 14).map((s) => s.tok.t), x0, H * 0.46, 34, W * 0.44, { slot: true, slotOn: blink });
        break;
      }
      case 'endC': {
        const p = llm.steps[14]!.top10_t1[0]![1];
        const tx = P ? W / 2 : x0, ty = P ? H * 0.64 : H * 0.5;
        slab(c, 'P(<END>)', tx, ty, P ? 90 : 110, { align: P ? 'center' : 'left', top: INK, under: BLUE });
        if (t >= sh.t0 + this.beat) {
          slab(c, pct(p, 2), tx, ty + (P ? 110 : 140), P ? 120 : 150, { align: P ? 'center' : 'left', k: snapIn(t, sh.t0 + this.beat) });
          label(c, 'AT TEMPERATURE 1', tx, ty + (P ? 200 : 250), 28, INK, P ? 'center' : 'left');
        }
        break;
      }
      case 'endD': {
        // the last roll, not landed: an empty bar, the needle hovering
        const bx = P ? 140 : W * 0.48, bw = P ? W - 280 : W * 0.38, by = P ? H * 0.6 : H * 0.44, bh = 110;
        label(c, `ROLL ${N_STEPS} / ${N_STEPS} · NOT LANDED YET`, bx, by - 84, 28, INK);
        c.strokeStyle = INK; c.lineWidth = 5; c.setLineDash([16, 10]);
        c.strokeRect(bx, by, bw, bh);
        c.setLineDash([]);
        const hov = 0.5 + 0.42 * Math.sin(drive(au, t, 5, 3));
        const nx = bx + bw * hov;
        c.fillStyle = PINK;
        c.fillRect(nx - 4, by - 36, 8, bh + 58);
        c.beginPath(); c.moveTo(nx - 18, by - 54); c.lineTo(nx + 18, by - 54); c.lineTo(nx, by - 30); c.fill();
        slab(c, '?', bx + bw / 2, by + bh + 130, 160, { top: PINK });
        break;
      }
    }
    c.restore();
  }

  private camRight(cam: THREE.PerspectiveCamera) {
    return new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0).setY(0).normalize();
  }

  // ---------------------------------------------------------------- frame

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio;
    const sh = shot(this.ctx, f, { every: 2 });
    const pl = this.plan[Math.min(sh.i, this.plan.length - 1)]!;
    const P = portrait();
    const d = drive(au, f.t, 0.6, 1.8), dft = drive(au, f.ft, 0.6, 1.8);

    // background: wash (reseeded per shot) + rain of the run's real tokens and ids
    const c1 = sh.i % 2 ? 'blue' : 'pink';
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: pl.kind === 'said' ? 0.8 : 0.55, c1, c2: c1 === 'pink' ? 'blue' : 'pink' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => rain(this.back.ctx, this.pool, dft * 0.5, { seed: sh.seed, density: 0.2 })), out);

    // type and marks, once per output frame (the camera at the frame's own time for pinned labels);
    // drawn first so the 3D pass can read the layout (the "?" tile sits on the attention slot)
    const midTex = perFrame(this.mid, f, () => {
      this.stage(pl, sh, f.ft, dft, P, this.camFT);
      this.camFT.updateMatrixWorld();
      this.draw2D(pl, sh, f.ft, P);
    });

    // the hero in 3D
    this.stage(pl, sh, f.t, d, P, this.cam);
    this.cam.updateMatrixWorld();
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);
    this.ctx.comp.draw(r, midTex, out);

    // foreground: specks
    const c = this.front.ctx;
    this.front.clear();
    specks(c, drive(au, f.t, 1.1, 1.8), { seed: sh.seed, n: 64 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    const said = pl.kind === 'said';
    return { reg: sh.seed, kinetic: said || pl.kind === 'bigbar' ? 0 : 1, flood: said ? 0.5 * pulse(f.t, sh.t0, 0.06) : 0 };
  }
}

