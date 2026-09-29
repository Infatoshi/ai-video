// append (verse 3, lines 5-8): "Roll the dice, one token, stick it on the end, / Feed the whole thing
// back and do it over again, / But the keys and the values that it made, it can keep, / That's the K-V
// cache, so the past comes cheap". Real data: the roll is step 4's nucleus (T 0.6, top-p 0.9) and its
// uniform draw u; the sentence strip is the real context (the 43-token chat prompt + the answer so far);
// the passes are the story's steps 5 and 6; the KV shelves hold one K and one V card per context
// position on each of the 32 floors (n = the story's position), the new pair filed floor by floor on
// "keys" and "values"; the cache grid shades each K by step 6's real attention (mean of the 32 heads);
// KV bytes = positions x llm.cfg.kv_bytes_per_token (derived from the config on screen). A new shot
// every bar, cut on the bar grid from "Roll".
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { HEX } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp } from '../engine/util';
import { showTok } from '../engine/llm';
import { Wash, tokenRain, specks, makeCam, drive, shot, snapIn, portrait, perFrame, inked, flatMat, outline, tokenTile, whip } from './_kit';
import { Tower, riderTile, SLAB_W, FLOOR_H } from './_tower';
import { slab, label, proj, fmtInt, pct, C } from './logits-parts';

const KV_PITCH = 0.1;
const X_END = 3; // right edge of the sentence strip
const STRIP_Z = 7;
const GAP = 0.14;

type Cam = { pos: THREE.Vector3; look: THREE.Vector3; fov: number };

function wheelTexture(nuc: [string, number][], u: number): THREE.CanvasTexture {
  const S = 1024, R = S / 2, cv = document.createElement('canvas');
  cv.width = S; cv.height = S;
  const c = cv.getContext('2d')!;
  const cols = [HEX.pink, HEX.blue, HEX.ink, HEX.blue, HEX.ink];
  let acc = 0;
  const arcs: { a0: number; a1: number }[] = [];
  nuc.forEach(([, p], i) => {
    const a0 = acc * Math.PI * 2, a1 = (acc + p) * Math.PI * 2;
    c.fillStyle = cols[i % cols.length]!;
    c.beginPath(); c.moveTo(R, R); c.arc(R, R, R - 4, a0 - Math.PI / 2, a1 - Math.PI / 2); c.closePath(); c.fill();
    arcs.push({ a0, a1 });
    acc += p;
  });
  // sector edges
  c.strokeStyle = HEX.paper; c.lineWidth = 10;
  for (const a of arcs) { c.beginPath(); c.moveTo(R, R); c.lineTo(R + Math.sin(a.a0) * R, R - Math.cos(a.a0) * R); c.stroke(); }
  // the 0..1 scale round the rim
  c.strokeStyle = HEX.paper;
  for (let i = 0; i < 20; i++) {
    const a = (i / 20) * Math.PI * 2, r0 = R - (i % 2 ? 34 : 60);
    c.lineWidth = i % 2 ? 6 : 10;
    c.beginPath(); c.moveTo(R + Math.sin(a) * r0, R - Math.cos(a) * r0); c.lineTo(R + Math.sin(a) * (R - 6), R - Math.cos(a) * (R - 6)); c.stroke();
  }
  c.lineWidth = 14; c.strokeStyle = HEX.ink;
  c.beginPath(); c.arc(R, R, R - 7, 0, Math.PI * 2); c.stroke();
  // labels: the pick's label sits where the draw lands (upright once the wheel stops), the rest mid-sector
  c.fillStyle = HEX.paper; c.textAlign = 'center'; c.textBaseline = 'middle';
  nuc.forEach(([t, p], i) => {
    const a = i === 0 ? u * Math.PI * 2 : (arcs[i]!.a0 + arcs[i]!.a1) / 2;
    const r = i === 0 ? R * 0.5 : R * 0.66;
    c.save();
    c.translate(R + Math.sin(a) * r, R - Math.cos(a) * r);
    c.rotate(a);
    c.font = font(F.mono(700), i === 0 ? 120 : 76);
    c.fillText(showTok(t), 0, i === 0 ? -34 : -26);
    c.font = font(F.mono(700), i === 0 ? 70 : 52);
    c.fillText(pct(p, 1), 0, i === 0 ? 62 : 34);
    c.restore();
  });
  const tx = new THREE.CanvasTexture(cv);
  tx.colorSpace = THREE.SRGBColorSpace;
  tx.anisotropy = 8;
  return tx;
}

export default class Append extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(36);
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  front = new Layer2D();
  tower = new Tower();
  wheel = new THREE.Group();
  wheelDisc = new THREE.Group();
  pickTile!: THREE.Mesh;
  strip = new THREE.Group();
  tiles: { m: THREE.Mesh; x: number; w: number }[] = [];
  /** The token after the roll ("do it over again": the next append). */
  next!: { m: THREE.Mesh; w: number };
  loop = new THREE.Group();
  loopTube!: THREE.Mesh;
  loopCurve!: THREE.CatmullRomCurve3;
  chevrons: THREE.Mesh[] = [];
  riders = new Map<number, THREE.Mesh>();
  RY = 0;
  k = 4; // the step rolled on this verse
  ctxToks: { id: number; t: string }[] = [];
  att: number[][] = []; // [layer][pos] step 6's attention, mean of heads, normalised per layer
  kvStep = 6;
  emitAgain = 0; // song time the token of "again" is emitted (story, snapped to the beat)
  pool: { id: number; t: string }[] = [];
  tm = { l1: 0, token: 0, stick: 0, end: 0, l2: 0, feed: 0, back: 0, again: 0, l3: 0, keys: 0, values: 0, keep: 0, l4: 0, kv: 0, cache: 0, past: 0, cheap: 0, pick: 0 };
  private tmp = new THREE.Object3D();
  private col = new THREE.Color();

  override init() {
    const { lyrics: ly, llm, story } = this.ctx;
    const word = (line: string, w: string) => {
      const l = ly.get(line);
      return (l.words.find((x) => x.w.toLowerCase().replace(/[^a-z0-9']/g, '').startsWith(w)) ?? l.words[0]!).start;
    };
    const L1 = 'Roll the dice', L2 = 'Feed the whole thing back', L3 = 'keys and the values', L4 = "That's the K";
    this.tm = {
      l1: ly.get(L1).start, token: word(L1, 'token'), stick: word(L1, 'stick'), end: word(L1, 'end'),
      l2: ly.get(L2).start, feed: word(L2, 'feed'), back: word(L2, 'back'), again: word(L2, 'again'),
      l3: ly.get(L3).start, keys: word(L3, 'keys'), values: word(L3, 'values'), keep: word(L3, 'keep'),
      l4: ly.get(L4).start, kv: word(L4, 'kv'), cache: word(L4, 'cache'), past: word(L4, 'past'), cheap: word(L4, 'cheap'), pick: 0,
    };
    // the step whose token lands on "one token": the roll
    let best = 0;
    story.emits.forEach((t, i) => { if (Math.abs(t - this.tm.token) < Math.abs(story.emits[best]! - this.tm.token)) best = i; });
    this.k = best;
    this.tm.pick = story.emits[this.k]!;
    this.kvStep = story.at(this.tm.keys).step;
    this.emitAgain = story.emits.reduce((b, e) => (Math.abs(e - this.tm.again) < Math.abs(b - this.tm.again) ? e : b), story.emits[0]!);
    const st = llm.steps[this.k]!;
    this.pool = [...llm.d.prompt, ...llm.steps.map((s) => s.tok)];

    // tower
    this.RY = this.tower.roof.position.y + 0.15;
    (this.tower.roof.material as THREE.ShaderMaterial).uniforms.litA!.value = 0.08;
    this.tower.outlineAll();
    this.scene.add(this.tower.group);

    // the wheel on the roof: the nucleus as sectors, a pointer at 12 o'clock
    const Rw = 2.2;
    const face = new THREE.Mesh(new THREE.CircleGeometry(Rw, 128), new THREE.MeshBasicMaterial({ map: wheelTexture(st.nucleus, st.u) }));
    face.position.z = 0.16;
    this.wheelDisc.add(face);
    const rim = inked(new THREE.CylinderGeometry(Rw, Rw, 0.3, 96, 1, true), { ink: 'ink', lit: 0.5, shade: 1, side: THREE.DoubleSide }, 2.4);
    rim.rotation.x = Math.PI / 2;
    this.wheelDisc.add(rim);
    const hub = inked(new THREE.CylinderGeometry(0.34, 0.34, 0.5, 32), { ink: 'ink', lit: 0.4, shade: 1 }, 2);
    hub.rotation.x = Math.PI / 2; hub.position.z = 0.25;
    this.wheelDisc.add(hub);
    this.wheel.add(this.wheelDisc);
    const ptr = new THREE.Mesh(new THREE.ConeGeometry(0.32, 0.7, 3), flatMat('ink'));
    ptr.rotation.z = Math.PI; ptr.position.set(0, Rw + 0.2, 0.35);
    outline(ptr, 2);
    this.wheel.add(ptr);
    const stand = inked(new THREE.BoxGeometry(0.5, 2.6, 0.5).translate(0, -1.3, -0.3), { ink: 'blue', lit: 0.5, shade: 1 }, 2);
    this.wheel.add(stand);
    this.pickTile = tokenTile(st.tok.t, { face: 'pink', text: 'paper', side: 'pink', h: 1.1 });
    this.pickTile.position.set(0, 0, 0.75);
    this.wheel.add(this.pickTile);
    this.wheel.position.set(0, this.RY + 2.7, 0.9);
    this.scene.add(this.wheel);

    // the sentence strip: the real context before the roll, then the rolled token
    this.ctxToks = [...llm.d.prompt, ...llm.steps.slice(0, this.k + 1).map((s) => s.tok)];
    let x = X_END;
    const made = this.ctxToks.map((tk, i) => {
      const last = i === this.ctxToks.length - 1;
      const m = tokenTile(tk.t, last ? { face: 'pink', text: 'paper', side: 'pink', h: 1 } : { h: 1, side: i < llm.d.prompt.length ? 'blue' : 'ink' });
      const w = (m.geometry as THREE.BoxGeometry).parameters.width;
      return { m, w };
    });
    for (let i = made.length - 1; i >= 0; i--) {
      const { m, w } = made[i]!;
      this.tiles[i] = { m, x: x - w / 2, w };
      x -= w + GAP;
      this.strip.add(m);
    }
    const nm = tokenTile(llm.steps[this.k + 1]!.tok.t, { face: 'pink', text: 'paper', side: 'pink', h: 1 });
    this.next = { m: nm, w: (nm.geometry as THREE.BoxGeometry).parameters.width };
    this.strip.add(nm);
    this.scene.add(this.strip);

    // the loop: out of the roof, round the outside, back into the lobby
    this.loopCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(1.2, this.RY + 0.4, 1.2), new THREE.Vector3(4.5, this.RY + 2.6, 2.4), new THREE.Vector3(9.5, this.RY - 3, 3.6),
      new THREE.Vector3(11, 10, 4.6), new THREE.Vector3(9.5, 3.2, 5.4), new THREE.Vector3(5.8, 0.9, 5.4), new THREE.Vector3(3.4, 0.5, 4.4),
    ]);
    this.loopTube = inked(new THREE.TubeGeometry(this.loopCurve, 160, 0.55, 12, false), { ink: 'pink', lit: 0.55, shade: 1, over: 'blue', overShade: 0.4 }, 2.4);
    this.loop.add(this.loopTube);
    const head = inked(new THREE.ConeGeometry(1.1, 2.0, 16), { ink: 'pink', lit: 0.55, shade: 1, over: 'blue', overShade: 0.4 }, 2.4);
    const endP = this.loopCurve.getPoint(1), tan = this.loopCurve.getTangent(1);
    head.position.copy(endP).addScaledVector(tan, 0.8);
    head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tan);
    this.loop.add(head);
    for (let i = 0; i < 9; i++) {
      const ch = new THREE.Mesh(new THREE.ConeGeometry(0.62, 1.0, 3), flatMat('ink'));
      this.chevrons.push(ch);
      this.loop.add(ch);
    }
    this.scene.add(this.loop);

    // step 6's attention (the pass whose K/V gets filed on "keys" and "values"): mean of the heads
    const s6 = llm.steps[this.kvStep]!;
    const [Lr, Hh, Cc] = s6.attn.shape;
    for (let l = 0; l < Lr; l++) {
      const row = new Array(Cc).fill(0);
      for (let h = 0; h < Hh; h++) { const a = llm.attn(this.kvStep, l, h); for (let p = 0; p < Cc; p++) row[p] += a[p]! / Hh; }
      const mx = Math.max(...row.slice(1), 1e-6);
      this.att.push(row.map((v) => Math.min(1, v / mx)));
    }
    this.cam.far = 400;
    this.cam.updateProjectionMatrix();
  }

  private rider(step: number): THREE.Mesh {
    let m = this.riders.get(step);
    if (!m) { m = riderTile(this.ctx.llm.steps[step]!.tok.t); m.scale.setScalar(1.25); this.riders.set(step, m); }
    return m;
  }

  /** Wheel angle (CCW radians): spins with the drums, then snaps so the pointer reads u at the pick. */
  private wheelAngle(t: number) {
    const au = this.ctx.audio, u = this.ctx.llm.steps[this.k]!.u;
    const spin = (x: number) => -drive(au, x, 1.2, 3.2) * 2.4;
    const tp = this.tm.pick;
    if (t < tp) return spin(t);
    const a0 = spin(tp), target = u * Math.PI * 2;
    const land = a0 - (((a0 - target) % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
    return lerp(a0, land, ease.outExpo(clamp((t - tp) / 0.12)));
  }

  /** KV cards: `n` settled pairs per floor; the pair after them grows in (K, V 0..1 per floor). */
  private setShelves(n: number, kIn: (l: number) => number, vIn: (l: number) => number) {
    const tw = this.tower, L = tw.layers, M = 96;
    let i = 0;
    for (let l = 1; l <= L; l++) {
      const kk = kIn(l), vv = vIn(l);
      for (let k = 0; k < M; k++) for (let kv = 0; kv < 2; kv++) {
        const isNew = k === n;
        const s = k < n ? 1 : isNew ? (kv ? vv : kk) : 0;
        const big = isNew ? 1.35 : 1;
        this.tmp.position.set(SLAB_W / 2 + 0.16 + k * KV_PITCH, tw.floorY(l) + 0.05 + FLOOR_H * (kv ? 0.54 : 0.2), -SLAB_W / 2 + 0.4);
        this.tmp.scale.set(1.3 * s * big, 0.9 * s * big, 1.6 * s * big);
        this.tmp.updateMatrix();
        tw.kv.setMatrixAt(i, this.tmp.matrix);
        tw.kv.setColorAt(i, this.col.setRGB(kv ? 1 : isNew ? 1 : 0, kv ? 0 : 1, 0));
        i++;
      }
    }
    tw.kv.instanceMatrix.needsUpdate = true;
    tw.kv.instanceColor!.needsUpdate = true;
  }

  private kvState(t: number) {
    const tm = this.tm, au = this.ctx.audio;
    const per = 60 / au.bpm / 32;
    if (t < this.emitAgain) return { n: this.ctx.story.at(t).pos, k: () => 0, v: () => 0 };
    const n = this.ctx.llm.steps[this.kvStep]!.pos - 1;
    return { n, k: (l: number) => snapIn(t, tm.keys + (l - 1) * per, 0.06), v: (l: number) => snapIn(t, tm.values + (l - 1) * per, 0.06) };
  }

  private camAt(kind: number, t: number, t0: number): Cam {
    const au = this.ctx.audio, RY = this.RY, P = portrait(), story = this.ctx.story;
    const d = drive(au, t, 0.5, 1.6), dd = d - drive(au, t0, 0.5, 1.6);
    const dist = P ? 1.6 : 1;
    const Ht = this.tower.height;
    switch (kind) {
      case 0: {
        const a = Math.sin(d * 0.3) * 0.08;
        return { pos: new THREE.Vector3(Math.sin(a) * 13.5 * dist + 0.4, RY + 2.6, Math.cos(a) * 13.5 * dist), look: new THREE.Vector3(0, RY + (P ? 2.5 : 2.65), 0.9), fov: 36 };
      }
      case 1: {
        const s = dd * 0.35, ex = this.tiles[this.tiles.length - 1]!.x;
        return P
          ? { pos: new THREE.Vector3(ex + 1.2 - s * 0.5, 3.0, STRIP_Z + 8.5), look: new THREE.Vector3(ex - 1.4 - s * 0.5, 1.1, STRIP_Z), fov: 44 }
          : { pos: new THREE.Vector3(ex + 2.4 - s, 2.3, STRIP_Z + 6.2), look: new THREE.Vector3(ex - 4.2 - s, 0.8, STRIP_Z), fov: 40 };
      }
      case 2: {
        const s = dd * 0.4;
        return P
          ? { pos: new THREE.Vector3(-7 + s, 3.2, 21), look: new THREE.Vector3(1.5 + s * 0.5, 5.2, 2), fov: 50 }
          : { pos: new THREE.Vector3(-10 + s, 2.4, 15.5), look: new THREE.Vector3(1.8 + s * 0.5, 3.6, 2), fov: 50 };
      }
      case 3: {
        // the sentence face-on: the next token sticks on the end, again
        const s = dd * 0.3;
        return P
          ? { pos: new THREE.Vector3(X_END - 0.6 - s, 2.2, STRIP_Z + 10.5), look: new THREE.Vector3(X_END - 1.4 - s, 1.2, STRIP_Z), fov: 44 }
          : { pos: new THREE.Vector3(X_END - 1.6 - s, 1.9, STRIP_Z + 8.2), look: new THREE.Vector3(X_END - 3.0 - s, 0.95, STRIP_Z), fov: 42 };
      }
      case 4: {
        // the shelves up close, tracking up the floors
        const y = 6 + dd * 2.2;
        return P
          ? { pos: new THREE.Vector3(6.4, y + 0.6, 6.5), look: new THREE.Vector3(6.6, y, -2.2), fov: 44 }
          : { pos: new THREE.Vector3(6.0, y + 0.5, 4.6), look: new THREE.Vector3(5.9, y, -2.2), fov: 42 };
      }
      case 5: {
        const a = 1.0 + d * 0.05;
        return { pos: new THREE.Vector3(Math.cos(a) * 27 * dist + 3, Ht * 0.62, Math.sin(a) * 27 * dist), look: new THREE.Vector3(2.4, Ht * 0.5, -1), fov: 40 };
      }
      case 6: {
        const a = 2.3 + d * 0.04;
        return { pos: new THREE.Vector3(Math.cos(a) * 42, Ht * 0.7, Math.sin(a) * 42), look: new THREE.Vector3(0, Ht * 0.5, 0), fov: 36 };
      }
      case 7: {
        const st = story.at(t), y = this.tower.floorY(st.climb * (this.tower.layers + 1));
        const a = 0.28 + dd * 0.06;
        return { pos: new THREE.Vector3(Math.cos(a) * 11 * dist + 2, y + 0.9, Math.sin(a) * 11 * dist), look: new THREE.Vector3(1.8, y + 0.1, -0.5), fov: 44 };
      }
      default: {
        const a = 0.7 + d * 0.08;
        return { pos: new THREE.Vector3(Math.cos(a) * 30 * dist, Ht * 1.05, Math.sin(a) * 30 * dist), look: new THREE.Vector3(1.5, Ht * 0.5, 0), fov: 38 };
      }
    }
  }

  private applyCam(c: Cam) {
    const cam = this.cam;
    cam.position.copy(c.pos);
    cam.up.set(0, 1, 0);
    cam.lookAt(c.look);
    if (cam.fov !== c.fov) { cam.fov = c.fov; cam.updateProjectionMatrix(); }
    cam.updateMatrixWorld();
  }

  /** Strip tiles for this time: their positions (with the append and the flight into the lobby). */
  private placeTiles(kind: number, t: number) {
    const tm = this.tm, n = this.tiles.length;
    const newW = this.tiles[n - 1]!.w + GAP;
    const shift = snapIn(t, tm.end, 0.09) * newW;
    const lobby = new THREE.Vector3(0, 0.5, 2.8);
    const te = this.emitAgain, shift2 = kind === 3 ? snapIn(t, te, 0.09) * (this.next.w + GAP) : 0;
    const nx = this.next.m;
    nx.visible = kind === 3;
    if (kind === 3) {
      const home = new THREE.Vector3(X_END + GAP + this.next.w / 2 - shift - shift2, 0.55, STRIP_Z);
      const k = snapIn(t, te, 0.1);
      const hover = new THREE.Vector3(home.x + shift2 + 0.4, 2.5 + Math.sin(drive(this.ctx.audio, t, 0.6, 1.8) * 4) * 0.12, STRIP_Z + 0.6);
      nx.position.copy(hover.lerp(home, k));
      nx.rotation.set(0, 0, (1 - k) * -0.22);
    }
    this.tiles.forEach((tl, i) => {
      const m = tl.m;
      m.visible = true;
      m.scale.setScalar(1);
      m.rotation.set(0, 0, 0);
      const home = new THREE.Vector3(tl.x - shift - shift2, 0.55, STRIP_Z);
      if (i === n - 1 && kind <= 1) {
        // the new token: hovers over the empty slot, sticks on "stick"
        const k = snapIn(t, tm.stick, 0.1);
        const hover = new THREE.Vector3(tl.x + 0.2, 2.05 + Math.sin(drive(this.ctx.audio, t, 0.6, 1.8) * 4) * 0.12, STRIP_Z + 1.1);
        m.position.copy(hover.lerp(home, k));
        m.rotation.z = (1 - k) * 0.25;
        return;
      }
      if (kind === 2) {
        // back into the lobby, in order, all in by "back"
        const t0 = lerp(tm.feed + 0.1, tm.back - 0.25, i / (n - 1));
        const e = ease.inCubic(clamp((t - t0) / 0.28));
        if (e >= 1) { m.visible = false; return; }
        m.position.copy(home.clone().lerp(lobby, e));
        m.position.y += Math.sin(e * Math.PI) * 3.2;
        m.scale.setScalar(1 - e * 0.6);
        return;
      }
      m.position.copy(home);
    });
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio, story = this.ctx.story;
    const sh = shot(this.ctx, f, { every: 4, from: this.tm.l1 });
    const kind = Math.min(sh.i, 8);
    const P = portrait();
    const d = drive(au, f.t, 0.6, 1.8);
    const tm = this.tm;

    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: kind === 6 ? 0.35 : 0.52, c1: kind % 2 ? 'blue' : 'pink', c2: kind % 2 ? 'pink' : 'blue' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => tokenRain(this.back.ctx, this.pool, drive(au, f.ft, 0.6, 1.8) * 0.5, { seed: sh.seed, cols: P ? 8 : 13 })), out);

    // 3D
    const st = story.at(f.t);
    this.wheel.visible = kind === 0;
    this.strip.visible = kind >= 1 && kind <= 3;
    this.loop.visible = kind === 2;
    this.tower.group.visible = kind !== 1 && kind !== 3;
    this.wheelDisc.rotation.z = this.wheelAngle(f.t);
    this.pickTile.scale.setScalar(Math.max(1e-3, snapIn(f.t, tm.pick, 0.09)));
    this.placeTiles(kind, f.t);
    // the loop arrow draws on, chevrons run round it with the drums
    const tube = this.loopTube.geometry as THREE.TubeGeometry;
    const reveal = kind === 2 ? snapIn(f.t, sh.t0, 0.14) : 1;
    tube.setDrawRange(0, Math.max(0, Math.floor((tube.index!.count / 6) * reveal) * 6));
    this.chevrons.forEach((ch, i) => {
      const u = (((d * 0.35 + i / this.chevrons.length) % 1) + 1) % 1;
      ch.visible = u < reveal;
      ch.position.copy(this.loopCurve.getPoint(u));
      ch.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), this.loopCurve.getTangent(u));
      ch.position.z += 0.6;
    });
    // the rider: the story's token in flight (hidden while the sentence pours in)
    this.tower.ride(this.rider(st.step), kind === 2 || kind === 0 ? 0 : st.climb);
    this.tower.heads.visible = kind !== 3;
    this.tower.rider.visible = kind !== 2 && kind !== 0;
    let lightAt = st.climb;
    if (kind === 2) lightAt = f.t >= tm.back ? clamp((f.t - tm.back) / 0.35) : -1;
    if (kind === 0) lightAt = 1;
    this.tower.light(lightAt);
    const kv = this.kvState(f.t);
    this.setShelves(kv.n, kv.k, kv.v);
    this.applyCam(this.camAt(kind, f.t, sh.t0));
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);

    this.ctx.comp.draw(r, perFrame(this.mid, f, () => this.drawType(kind, f.ft, sh)), out);

    const c = this.front.ctx;
    this.front.clear();
    specks(c, d * (kind === 2 || kind === 3 ? 1.8 : 1), { seed: sh.seed, n: kind === 6 ? 50 : 75 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    return { reg: sh.seed };
  }

  private drawType(kind: number, t: number, sh: { t0: number; t1: number }) {
    const c = this.mid.ctx, P = portrait(), tm = this.tm, llm = this.ctx.llm, cfg = llm.cfg;
    this.applyCam(this.camAt(kind, t, sh.t0));
    const bottomY = H - (P ? 420 : 160);
    const st = llm.steps[this.k]!;
    switch (kind) {
      case 0: {
        const nuc = st.nucleus_size;
        label(c, `THE DRAW · T ${llm.d.sampling.temperature} · TOP-P ${llm.d.sampling.top_p} · ${nuc} TOKENS LEFT`, W / 2, bottomY, { size: P ? 24 : 28, align: 'center', box: C.PAPER });
        if (t >= tm.pick) {
          const p = proj(this.cam, new THREE.Vector3(0.5, this.RY + 2.7 + 2.6, 0.9));
          if (p) label(c, `u = ${st.u.toFixed(3)}  →  ${showTok(st.tok.t)}`, p[0] + 40, p[1], { size: P ? 34 : 38, box: C.INK, color: C.PAPER });
        } else {
          const p = proj(this.cam, new THREE.Vector3(0.5, this.RY + 2.7 + 2.6, 0.9));
          if (p) label(c, 'u = ?', p[0] + 40, p[1], { size: P ? 34 : 38, box: C.PAPER });
        }
        break;
      }
      case 1: {
        const n = this.tiles.length;
        const last = this.tiles[n - 1]!;
        const stuck = t >= tm.stick;
        if (!stuck) {
          // the empty slot at the end
          const a = proj(this.cam, new THREE.Vector3(last.x - last.w / 2, 0.05, STRIP_Z + 0.2)), b = proj(this.cam, new THREE.Vector3(last.x + last.w / 2, 1.05, STRIP_Z + 0.2));
          if (a && b) {
            c.save(); c.strokeStyle = C.PINK; c.lineWidth = 6; c.setLineDash([16, 10]);
            c.strokeRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]));
            c.restore();
          }
        }
        const p = proj(this.cam, new THREE.Vector3(last.x - (t >= tm.end ? last.w + GAP : 0), stuck ? 1.9 : 3.1, STRIP_Z));
        if (p) label(c, `POS ${st.pos}`, p[0], p[1], { size: 34, align: 'center', box: C.INK, color: C.PAPER });
        label(c, t >= tm.end ? `${n} TOKENS IN CONTEXT` : `${n - 1} TOKENS IN CONTEXT`, P ? W / 2 : W * 0.3, P ? H * 0.2 : H * 0.22, { size: P ? 30 : 34, align: 'center', box: C.PAPER });
        break;
      }
      case 2: {
        const p = proj(this.cam, new THREE.Vector3(-2, 1.2, 4));
        if (p) label(c, `ALL ${this.tiles.length} TOKENS · BACK IN`, clamp(p[0], 120, W - 560), p[1] + 70, { size: P ? 28 : 32, box: C.INK, color: C.PAPER });
        const q = proj(this.cam, this.loopCurve.getPoint(0.42));
        if (q) label(c, 'OUTPUT → INPUT', clamp(q[0] + 50, 120, W - 420), q[1], { size: P ? 26 : 30, box: C.PAPER });
        break;
      }
      case 3: {
        const n = this.ctx.story.at(t).step + 1;
        const k = t >= this.emitAgain ? snapIn(t, this.emitAgain, 0.08) : 1;
        const py = P ? H * 0.24 : H * 0.27;
        slab(c, `PASS ${n}`, P ? W / 2 : W * 0.3, py, P ? 130 : 150, k, 'center', W * 0.46);
        label(c, `OF ${llm.steps.length} · ONE PASS PER TOKEN`, P ? W / 2 : W * 0.3, py + (P ? 100 : 112), { size: P ? 26 : 28, align: 'center', box: C.PAPER });
        const nm = this.next.m;
        const p = proj(this.cam, new THREE.Vector3(nm.position.x, nm.position.y + 1.2, nm.position.z));
        if (p) label(c, `POS ${llm.steps[this.k + 1]!.pos}`, clamp(p[0], 160, W - 160), clamp(p[1], P ? 240 : 190, H - 300), { size: 34, align: 'center', box: C.INK, color: C.PAPER });
        const nCtx = this.tiles.length + (t >= this.emitAgain ? 1 : 0);
        label(c, `${nCtx} TOKENS IN CONTEXT`, W / 2, bottomY, { size: P ? 28 : 32, align: 'center', box: C.PAPER });
        break;
      }
      case 4: {
        const fl = clamp(Math.round((6 + (drive(this.ctx.audio, t, 0.5, 1.6) - drive(this.ctx.audio, sh.t0, 0.5, 1.6)) * 2.2 - 0.15) / FLOOR_H), 1, 32);
        const n = llm.steps[this.kvStep]!.pos - 1;
        const x = SLAB_W / 2 + 0.16 + n * KV_PITCH, y = this.tower.floorY(fl) + 0.05 + FLOOR_H * 0.2, z = -SLAB_W / 2 + 0.4;
        const pk = proj(this.cam, new THREE.Vector3(x, y, z)), pv = proj(this.cam, new THREE.Vector3(x, this.tower.floorY(fl) + 0.05 + FLOOR_H * 0.54, z));
        const big = (txt: string, sub: string, at: [number, number], dx: number, dy: number, col: string, on: number) => {
          if (on <= 0) return;
          c.save();
          c.strokeStyle = C.INK; c.lineWidth = 4;
          const ax = clamp(at[0] + dx, 180, W - 200), ay = clamp(at[1] + dy, P ? 330 : 250, H - 240);
          c.beginPath(); c.moveTo(at[0], at[1]); c.lineTo(ax, ay); c.stroke();
          c.font = font(F.archivo(125, 900), P ? 120 : 140);
          c.textAlign = 'center'; c.textBaseline = 'middle';
          const s = 1.25 - 0.25 * on;
          c.translate(ax, ay - (P ? 70 : 84));
          c.scale(s, s);
          c.fillStyle = col;
          c.fillText(txt, 0, 0);
          c.restore();
          label(c, sub, ax, ay + 4, { size: 28, align: 'center', box: C.INK, color: C.PAPER });
        };
        if (pk) big('K', 'KEYS', pk, P ? -380 : -330, P ? 260 : -150, C.BLUE, snapIn(t, tm.keys, 0.08));
        if (pv) big('V', 'VALUES', pv, P ? -260 : 200, P ? -300 : -260, C.PINK, snapIn(t, tm.values, 0.08));
        label(c, `ONE NEW PAIR ON EVERY FLOOR · ${n} → ${n + 1}`, W / 2, bottomY, { size: P ? 26 : 30, align: 'center', box: C.PAPER });
        break;
      }
      case 5: {
        const per = cfg.layers * cfg.kv_heads * cfg.head_dim * 2 * 2;
        const y0 = P ? H * 0.2 : H * 0.2;
        label(c, `${cfg.layers} FLOORS × ${cfg.kv_heads} KV HEADS × ${cfg.head_dim} NUMBERS × (K + V) × 2 BYTES`, W / 2, y0, { size: P ? 22 : 28, align: 'center', box: C.PAPER });
        label(c, `= ${fmtInt(per)} BYTES = ${per / 1024} KiB PER TOKEN, KEPT`, W / 2, y0 + (P ? 44 : 52), { size: P ? 26 : 32, align: 'center', box: C.INK, color: C.PAPER });
        break;
      }
      case 6: this.cacheGrid(c, t); break;
      case 7: {
        const y = P ? H * 0.24 : H * 0.26;
        const k = snapIn(t, sh.t0, 0.08);
        c.fillStyle = C.PAPER;
        c.fillRect(P ? W * 0.04 : W * 0.14, y - (P ? 66 : 80), P ? W * 0.92 : W * 0.72, P ? 132 : 160);
        c.fillStyle = C.INK;
        c.fillRect(P ? W * 0.04 : W * 0.14, y - (P ? 66 : 80), P ? W * 0.92 : W * 0.72, 5);
        slab(c, 'RECOMPUTED: 0', W / 2, y, P ? 96 : 120, k, 'center', P ? W * 0.86 : W * 0.66);
        label(c, `PAST TOKENS STAY ON THE SHELVES · ONLY THE NEW ONE CLIMBS`, W / 2, y + (P ? 80 : 96), { size: P ? 22 : 28, align: 'center', box: C.INK, color: C.PAPER });
        label(c, `1 TOKEN THROUGH THE TOWER, NOT ${llm.steps[this.kvStep]!.pos}`, W / 2, bottomY, { size: P ? 26 : 30, align: 'center', box: C.PAPER });
        break;
      }
      default: {
        const pos = this.ctx.story.at(t).pos;
        label(c, `KV CACHE ${(pos * cfg.kv_bytes_per_token / 2 ** 20).toFixed(1)} MB · ${pos} × 128 KiB`, W / 2, bottomY, { size: P ? 26 : 30, align: 'center', box: C.INK, color: C.PAPER });
        break;
      }
    }
  }

  /** The cache as a grid: 32 floors × every position, a K and a V card each; K height = real attention. */
  private cacheGrid(c: CanvasRenderingContext2D, t: number) {
    const P = portrait(), llm = this.ctx.llm, cfg = llm.cfg, au = this.ctx.audio;
    const n = llm.steps[this.kvStep]!.pos; // positions cached after this pass
    const L = cfg.layers;
    const gx = P ? W * 0.07 : W * 0.08, gw = P ? W * 0.86 : W * 0.6;
    const gy = P ? H * 0.2 : H * 0.2, gh = P ? H * 0.34 : H * 0.6;
    const cw = gw / n, rh = gh / L;
    c.fillStyle = C.PAPER;
    c.fillRect(gx - 18, gy - 18, gw + 36, gh + 36);
    c.strokeStyle = C.INK; c.lineWidth = 4;
    c.strokeRect(gx - 18, gy - 18, gw + 36, gh + 36);
    // a read head sweeping across the positions with the drums (the new token reading every K)
    const scan = ((drive(au, t, 0.5, 1.6) * 0.9) % 1) * n;
    for (let l = 0; l < L; l++) {
      const y = gy + (L - 1 - l) * rh;
      for (let p = 0; p < n; p++) {
        const x = gx + p * cw;
        const a = this.att[l]![p] ?? 0;
        const hk = (0.2 + 0.8 * Math.sqrt(a)) * rh * 0.86;
        c.fillStyle = p === n - 1 ? C.INK : C.BLUE;
        c.fillRect(x + cw * 0.08, y + rh * 0.92 - hk, cw * 0.4, hk);
        c.fillStyle = C.PINK;
        c.fillRect(x + cw * 0.52, y + rh * 0.92 - rh * 0.3, cw * 0.36, rh * 0.3);
      }
    }
    c.fillStyle = C.PINK;
    c.fillRect(gx + scan * cw - 3, gy - 18, 6, gh + 36);
    // counter
    const mb = (n * cfg.kv_bytes_per_token) / 2 ** 20;
    const cx = P ? W / 2 : W * 0.83, cy = P ? H * 0.64 : H * 0.44;
    slab(c, `${mb.toFixed(1)} MB`, cx, cy, P ? 130 : 110, snapIn(t, this.tm.cache, 0.08) || 1, 'center', P ? W * 0.8 : W * 0.3);
    label(c, `${n} POSITIONS × 128 KiB`, cx, cy + (P ? 96 : 84), { size: P ? 28 : 26, align: 'center', box: C.INK, color: C.PAPER });
    label(c, 'K HEIGHT = REAL ATTENTION', cx, cy + (P ? 146 : 128), { size: 24, align: 'center', box: C.PAPER });
    label(c, `${L} FLOORS`, gx - 18, gy - 44, { size: 24, box: C.PAPER });
    label(c, `POSITION 0 → ${n - 1}`, gx + gw + 18, gy + gh + 46, { size: 24, align: 'right', box: C.PAPER });
  }
}
