// outro: "One more token, / Till it picks the one that says it's done", then the loud instrumental to the
// song's hard stop, then silence. The quiet lines (cuts every 2 beats) bring the last token home: the
// end-of-turn die (step 14, <END>, 99.96% at T 1) spins while its token climbs the tower, slows, and lands on
// "done" exactly when the story clock emits it; the tile slams. When the band comes in, the whole video is
// reprinted at speed, in order: one earlier frame per beat (public/plates, rendered clean, printed here as a
// card with crop and registration marks and its "<id> · <t> s" label), a poster beat (outro-poster.ts, the
// whole exchange as a riso poster) on the 4th beat of every bar, the outro's own objects last, and the full
// poster held for the last 2 bars. On the hard stop: blank paper to the end.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { clamp, ease, hash, pulse } from '../engine/util';
import { showTok } from '../engine/llm';
import { Wash, specks, makeCam, drive, shot, snapIn, portrait, perFrame, tokenTile, flatMat, type Shot } from './_kit';
import { HEX, LIN } from '../engine/palette';
import { F, font } from '../engine/type';
import { Tower, riderTile, SLAB_W } from './_tower';
import {
  PINK, BLUE, INK, slab, label, rain, rainPool, oddsBar, tileRow, makeDie, rollDie, worldAt, toScreen, flatCam, pct, safeClip, safeBox, type Die,
} from './finale-kit';
import { Poster, type Focus, type Inks } from './outro-poster';

type QKind = 'spin' | 'climb' | 'lens' | 'odds' | 'close' | 'roof' | 'land' | 'done';
type BKind = Focus | 'tiles' | 'tower' | 'die';
type Arrive = 'drop' | 'slide' | 'flip' | 'stack';
/** One band beat: an earlier frame as a printed card, a poster zoom, an outro callback, or the final poster. */
type BItem = { kind: 'plate'; p: number; mode: Arrive } | { kind: 'poster'; focus: Focus } | { kind: 'cb'; k: 'tiles' | 'tower' | 'die' } | { kind: 'final' };
interface Plate { id: string; t: number; tex: THREE.Texture; lab: THREE.CanvasTexture; labW: number }

const N_STEPS = 15;
const Q_PLAN: QKind[] = ['spin', 'climb', 'lens', 'odds', 'close', 'roof', 'land', 'done'];
/** Card arrivals, rotated so consecutive plate beats never arrive the same way. */
const MODES: Arrive[] = ['drop', 'slide', 'flip', 'stack'];
/** The poster beats between the plates: each zoom once, the question first, the colophon last. */
const FOCI: Focus[] = ['title', 'q', 'ans1', 'two', 'straw', 'ans2', 'foot', 'footTwo', 'bigTwo', 'footThree', 'three', 'bigThree', 'digits', 'colo'];
/** Beats the full poster holds before the stop (2 bars). */
const FINAL_BEATS = 8;
const CAM_D = 10, CAM_FOV = 36;
/** Ink density gain for the reprinted frames (1 = as rendered). */
const PLATE_GAIN = 1.5;
const INK3 = { paper: LIN.paper };

/** Card geometry in logical px (the group is scaled to world units): image, margins, label height. */
interface CardDims { iw: number; ih: number; m: number; mb: number; sw: number; sh: number; lh: number }

/** A printed card: blue underprint offset behind, the paper sheet (outlined), the frame, marks, the label. */
class Card {
  group = new THREE.Group();
  img: THREE.Mesh;
  lab: THREE.Mesh;
  constructor(public d: CardDims, marks: THREE.Texture) {
    const under = new THREE.Mesh(new THREE.PlaneGeometry(d.sw, d.sh), flatMat('blue'));
    under.position.set(16, -16, -4);
    const sheet = new THREE.Mesh(new THREE.BoxGeometry(d.sw, d.sh, 4), flatMat('paper'));
    this.img = new THREE.Mesh(new THREE.PlaneGeometry(d.iw, d.ih), new THREE.ShaderMaterial({
      uniforms: { plate: { value: null }, paper: { value: new THREE.Vector3(...INK3.paper) }, gain: { value: PLATE_GAIN } },
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      // the plate re-inked: its optical density against the paper scaled up, so the reprint prints as solidly as the original
      fragmentShader: `uniform sampler2D plate; uniform vec3 paper; uniform float gain; varying vec2 vUv;
        void main() { vec3 c = texture2D(plate, vUv).rgb; vec3 od = -log(clamp(c / paper, 1e-3, 1.0)); gl_FragColor = vec4(paper * exp(-od * gain), 1.0); }`,
    }));
    this.img.position.set(0, d.sh / 2 - d.m - d.ih / 2, 2.5);
    const mk = new THREE.Mesh(new THREE.PlaneGeometry(d.sw, d.sh), new THREE.MeshBasicMaterial({ map: marks, transparent: true, depthWrite: false }));
    mk.position.z = 3;
    this.lab = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false }));
    this.lab.position.z = 3;
    this.group.add(under, sheet, this.img, mk, this.lab);
  }
  set(p: Plate) {
    const im = this.img.material as THREE.ShaderMaterial, lm = this.lab.material as THREE.MeshBasicMaterial;
    im.uniforms.plate!.value = p.tex;
    if (lm.map !== p.lab) { lm.map = p.lab; lm.needsUpdate = true; }
    const d = this.d, lw = d.lh * p.labW;
    this.lab.scale.set(lw, d.lh, 1);
    this.lab.position.x = -d.sw / 2 + d.m + lw / 2;
    this.lab.position.y = -d.sh / 2 + d.mb / 2;
  }
}

export default class Outro extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(36);
  camFT = makeCam(36);
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  front = new Layer2D();
  tower = new Tower();
  rider!: THREE.Mesh;
  die!: Die;
  /** The answer's 15 tokens as printed tiles (paper, <END> pink), and the <END> tile that slams. */
  row: THREE.Mesh[] = [];
  endTile!: THREE.Mesh;
  poster!: Poster;
  pool: string[] = [];
  beat = 0.345;
  /** Song times: the quiet lines' cut grid start, the <END> emission, the band's entry, the hard stop. */
  q0 = 0;
  land = 0;
  band = 0;
  stop = 0;
  kvShown = -1;
  lastKind = '';
  /** The band: earlier frames (public/plates or plates_portrait), the beat sequence, the card rigs. */
  plates: Plate[] = [];
  seq: BItem[] = [];
  cards: Card[] = [];
  resident = new Set<number>();

  override async init() {
    const { llm, audio: au, lyrics: ly, story } = this.ctx;
    this.pool = rainPool(llm);
    if (llm.steps.length !== N_STEPS) console.warn(`outro: the run has ${llm.steps.length} steps, the scene is laid out for ${N_STEPS}`);
    this.beat = au.timeOfBeat(Math.round(au.beatAt(150)) + 1) - au.timeOfBeat(Math.round(au.beatAt(150)));
    // the quiet lines cut every 2 beats from the section's start (the entry opens early for the transition)
    const outro = au.sections.find((s) => s.name === 'outro');
    this.q0 = au.timeOfBeat(Math.round(au.beatAt(outro?.start ?? this.ctx.start)));
    this.land = story.emits[N_STEPS - 1]!;
    const done = ly.get('says it', 0).words.find((w) => w.w.toLowerCase().startsWith('done'));
    const after = (done?.end ?? this.land + 1) - 0.1;
    this.band = au.downbeats.find((d) => d >= after) ?? this.land + 1;
    // the hard stop: where the band's level falls away for good, snapped to the beat
    let stop = au.duration;
    for (let t = this.band + 5; t < au.duration - 0.3; t += 0.01) {
      if (au.env('rms', t) < 0.15 && au.env('rms', t + 0.25) < 0.1) { stop = t; break; }
    }
    this.stop = au.timeOfBeat(Math.round(au.beatAt(stop)));

    this.tower.outlineAll();
    this.rider = riderTile('?');
    this.tower.ride(this.rider, 0.9);
    this.scene.add(this.tower.group);
    this.die = makeDie(llm, N_STEPS - 1);
    this.scene.add(this.die.mesh);
    llm.steps.forEach((st, i) => {
      const last = i === llm.steps.length - 1;
      const m = tokenTile(st.tok.t, { face: last ? 'pink' : 'paper', text: last ? 'paper' : 'ink', side: last ? 'ink' : 'blue', id: st.tok.id, h: 1.1 });
      this.scene.add(m);
      this.row.push(m);
    });
    const endSt = llm.steps[N_STEPS - 1]!;
    this.endTile = tokenTile(endSt.tok.t, { face: 'pink', text: 'paper', side: 'ink', id: endSt.tok.id, h: 1.1 });
    this.scene.add(this.endTile);
    this.poster = new Poster(llm);
    await this.initBand();
  }

  /** Load the earlier frames, build the card rigs, and lay the band's beats out in song order. */
  private async initBand() {
    const au = this.ctx.audio, P = portrait();
    const dir = P ? 'plates_portrait' : 'plates';
    let list: { file: string; id: string; t: number }[] = [];
    try {
      const r = await fetch(`${dir}/plates.json`);
      if (r.ok) list = await r.json();
    } catch { list = []; }
    list.sort((a, b) => a.t - b.t);
    const loader = new THREE.TextureLoader();
    const loaded = await Promise.all(list.map(async (x) => {
      try {
        const tex = await loader.loadAsync(x.file);
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = 4;
        const [lab, labW] = this.labelTex(x.id, x.t);
        return { id: x.id, t: x.t, tex, lab, labW } as Plate;
      } catch { console.warn(`outro: plate ${x.file} failed to load`); return null; }
    }));
    this.plates = loaded.filter((x): x is Plate => x !== null);

    // card geometry (px): the frame near full width, a paper margin, a deeper bottom margin for the label
    const iw = Math.round(W * (P ? 0.74 : 0.72)), ih = Math.round(iw * (H / W));
    const m = Math.round(iw * 0.035), mb = m + 64;
    const dims: CardDims = { iw, ih, m, mb, sw: iw + 2 * m, sh: ih + m + mb, lh: 46 };
    const marks = this.marksTex(dims);
    for (let i = 0; i < 3; i++) { const c = new Card(dims, marks); c.group.visible = false; this.scene.add(c.group); this.cards.push(c); }
    if (this.plates[0]) for (const c of this.cards) c.set(this.plates[0]);

    // the beats: plates in song order, a poster zoom on beat 4 of every bar, the outro's own objects after the
    // plates, then the full poster for the last 2 bars
    const n = Math.max(0, Math.round(au.beatAt(this.stop) - au.beatAt(this.band)));
    const fin = Math.min(FINAL_BEATS, n), body = n - fin;
    const items: BItem[] = [...this.plates.map((_, p) => ({ kind: 'plate', p, mode: MODES[p % MODES.length]! }) as BItem),
      { kind: 'cb', k: 'tiles' }, { kind: 'cb', k: 'tower' }, { kind: 'cb', k: 'die' }];
    let fi = 0;
    for (let j = 0; j < body; j++) {
      if (j % 4 === 3 || !items.length) this.seq.push({ kind: 'poster', focus: FOCI[fi++ % FOCI.length]! });
      else this.seq.push(items.shift()!);
    }
    if (items.length) console.warn(`outro: ${items.length} band items did not fit before the stop`);
    for (let j = 0; j < fin; j++) this.seq.push({ kind: 'final' });
  }

  /** "<id> · <t> s" as a transparent label texture (id pink, time ink); returns it and its aspect. */
  private labelTex(id: string, t: number): [THREE.CanvasTexture, number] {
    const h = 96, fs = 64, a = id, b = ` · ${t.toFixed(1)} s`;
    const cv = document.createElement('canvas');
    const c = cv.getContext('2d')!;
    c.font = font(F.mono(700), fs);
    const wa = c.measureText(a).width, wb = c.measureText(b).width;
    cv.width = Math.ceil(wa + wb + 8); cv.height = h;
    c.font = font(F.mono(700), fs);
    c.textBaseline = 'middle';
    c.fillStyle = HEX.pink; c.fillText(a, 2, h / 2);
    c.fillStyle = HEX.ink; c.fillText(b, 2 + wa, h / 2);
    const tx = new THREE.CanvasTexture(cv);
    tx.colorSpace = THREE.SRGBColorSpace;
    return [tx, cv.width / h];
  }

  /** The sheet's print marks (shared by every card): crop marks at the frame's corners, registration targets, ink swatches. */
  private marksTex(d: CardDims): THREE.CanvasTexture {
    const k = 1.5, cv = document.createElement('canvas');
    cv.width = Math.round(d.sw * k); cv.height = Math.round(d.sh * k);
    const c = cv.getContext('2d')!;
    c.scale(k, k);
    c.strokeStyle = HEX.ink; c.fillStyle = HEX.ink;
    c.lineWidth = 4; c.strokeRect(2, 2, d.sw - 4, d.sh - 4); // the sheet's edge
    c.lineWidth = 3;
    const x0 = d.m, y0 = d.m, x1 = d.m + d.iw, y1 = d.m + d.ih, g = 6, L = d.m * 0.8;
    c.beginPath();
    for (const [x, y, sx, sy] of [[x0, y0, -1, -1], [x1, y0, 1, -1], [x0, y1, -1, 1], [x1, y1, 1, 1]] as const) {
      c.moveTo(x + sx * g, y); c.lineTo(x + sx * (g + L), y);
      c.moveTo(x, y + sy * g); c.lineTo(x, y + sy * (g + L));
    }
    c.stroke();
    const target = (x: number, y: number, r: number) => {
      c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2);
      c.moveTo(x - r * 1.7, y); c.lineTo(x + r * 1.7, y); c.moveTo(x, y - r * 1.7); c.lineTo(x, y + r * 1.7); c.stroke();
    };
    target(d.sw / 2, d.m / 2, Math.min(9, d.m * 0.24));
    target(d.sw / 2, d.sh - d.mb / 2, 9);
    const sw = 24;
    [HEX.pink, HEX.blue, HEX.ink].forEach((col, i) => { c.fillStyle = col; c.fillRect(d.sw - d.m - (3 - i) * (sw + 8) + 8, d.sh - d.mb / 2 - sw / 2, sw, sw); });
    const tx = new THREE.CanvasTexture(cv);
    tx.colorSpace = THREE.SRGBColorSpace;
    return tx;
  }

  // ---------------------------------------------------------------- helpers

  private hideAll() {
    this.tower.group.visible = false;
    this.rider.visible = false;
    this.die.mesh.visible = false;
    for (const m of this.row) m.visible = false;
    this.endTile.visible = false;
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
    this.tower.clearHeads();
    this.tower.light(-1);
  }
  private setKV(n: number) {
    if (n === this.kvShown) return;
    this.kvShown = n;
    this.tower.setKV(n);
  }
  private camRight(cam: THREE.PerspectiveCamera) {
    return new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0).setY(0).normalize();
  }
  /** The die's spin distance: drums-driven, then a speed ramp that slows it to rest on the landing. */
  private spin(t: number) {
    const au = this.ctx.audio, T = 2.4, a = this.land - T;
    if (t <= a) return drive(au, t, 2.2, 2);
    const x = clamp((t - a) / T);
    return drive(au, a, 2.2, 2) + 2.2 * T * (1 - (1 - x) * (1 - x)) / 2;
  }

  // ---------------------------------------------------------------- the quiet lines

  private stageQ(k: QKind, sh: Shot, t: number, d: number, P: boolean, cam: THREE.PerspectiveCamera) {
    const llm = this.ctx.llm, story = this.ctx.story, tw = this.tower, Ht = tw.height, dist = P ? 1.6 : 1;
    const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
    const a = d * 0.12 + sh.i * 1.3;
    const np = llm.d.prompt.length;
    const climb = story.at(t).climb;
    this.hideAll();
    for (const c of this.cards) c.group.visible = false;
    this.resetTower(k);
    const dieAt = (pos: THREE.Vector3, s: number) => {
      this.die.mesh.visible = true;
      this.die.mesh.position.copy(pos);
      this.die.mesh.scale.setScalar(s);
      rollDie(this.die, this.spin(t), t, this.land, 14);
    };
    switch (k) {
      case 'spin': {
        this.flat(cam, P ? 11 : 10);
        dieAt(worldAt(cam, P ? 0.5 : 0.3, P ? 0.66 : 0.55), P ? 0.95 : 1.3);
        break;
      }
      case 'climb': {
        tw.group.visible = true;
        this.rider.visible = true;
        tw.ride(this.rider, climb);
        tw.light(climb);
        this.setKV(0);
        const y = tw.floorY(climb * 33);
        this.setCam(cam, V(Math.cos(a) * 12 * dist, y + 2.2, Math.sin(a) * 12 * dist), V(0, y - 0.5, 0), 42, P ? 0 : -6);
        break;
      }
      case 'lens': {
        tw.group.visible = true;
        this.rider.visible = true;
        tw.ride(this.rider, climb);
        this.setKV(0);
        for (let l = 1; l <= tw.layers; l++) (tw.slabs[l]!.material as THREE.ShaderMaterial).uniforms.hot!.value = l === 32 ? 1 : 0;
        const y = tw.floorY(28.5);
        this.setCam(cam, V(Math.cos(a) * 15 * (P ? 1.5 : 1), y + 1.5, Math.sin(a) * 15 * (P ? 1.5 : 1)), V(0, y, 0), 40, P ? -2.5 : -6);
        break;
      }
      case 'odds': {
        this.flat(cam, P ? 11 : 10);
        dieAt(worldAt(cam, P ? 0.5 : 0.78, P ? 0.66 : 0.5), P ? 0.95 : 1.05);
        break;
      }
      case 'close': {
        this.flat(cam, P ? 10 : 5.6);
        dieAt(V(P ? 0 : -0.9, P ? 0.9 : 0.1, 0), 1);
        break;
      }
      case 'roof': {
        tw.group.visible = true;
        this.rider.visible = true;
        tw.ride(this.rider, Math.min(climb, 0.985));
        tw.light(climb);
        this.setKV(0);
        dieAt(V(0, Ht + 3.3, 0), 1.15);
        this.setCam(cam, V(Math.cos(a) * 9.5 * dist, Ht - 1.2, Math.sin(a) * 9.5 * dist), V(0, Ht + 1.4, 0), 50, P ? 0 : -5);
        break;
      }
      case 'land': {
        this.flat(cam, P ? 11 : 10);
        dieAt(worldAt(cam, P ? 0.5 : 0.72, P ? 0.45 : 0.45), (P ? 0.88 : 1.3) * (t >= this.land ? 1 + 0.18 * (1 - snapIn(t, this.land, 0.12)) : 1));
        if (t >= this.land) {
          const kk = snapIn(t, this.land);
          this.endTile.visible = true;
          this.endTile.position.copy(worldAt(cam, P ? 0.5 : 0.3, P ? 0.74 : 0.7, 1));
          this.endTile.scale.setScalar(1.25 * (1 + 0.7 * (1 - kk)));
          this.endTile.up.set(0, 1, 0);
          this.endTile.rotation.set(0.1, Math.sin(d * 0.7) * 0.25, 0);
        }
        break;
      }
      case 'done': {
        this.rowShot(cam, t, d, P, sh.t0, true);
        break;
      }
    }
  }

  /** The answer's 15 tiles in a row, the camera tracking along them (drums-driven). */
  private rowShot(cam: THREE.PerspectiveCamera, t: number, d: number, P: boolean, t0: number, fromEnd = false) {
    const gap = 0.25;
    let x = 0;
    const xs: number[] = [];
    for (const m of this.row) {
      const w = (m.geometry as THREE.BoxGeometry).parameters.width;
      xs.push(x + w / 2);
      x += w + gap;
    }
    const total = x - gap;
    const travel = (drive(this.ctx.audio, t, 0.8, 2) - drive(this.ctx.audio, t0, 0.8, 2));
    if (P) {
      // portrait: the answer as a column, read top to bottom, the camera tracking down it
      const pitch = 1.45, n = this.row.length;
      this.row.forEach((m, i) => {
        m.visible = true;
        m.position.set(0, -i * pitch, 0);
        m.scale.setScalar(0.95);
        m.up.set(0, 1, 0);
        m.rotation.set(0, 0, 0);
      });
      const span = Math.max(1, (n - 4) * pitch), run = (travel * 3.2) % span;
      const cy = fromEnd ? -2.2 - span + run : -2.2 - run;
      this.setCam(cam, new THREE.Vector3(1.2, cy + 1.6, 8.6), new THREE.Vector3(-1.45, cy - 0.4, 0), 50);
      return;
    }
    this.row.forEach((m, i) => {
      m.visible = true;
      m.position.set(xs[i]! - total / 2, 0, 0);
      m.scale.setScalar(1);
      m.up.set(0, 1, 0);
      m.rotation.set(0, 0, 0);
    });
    const span = Math.max(1, total - 6), run = (travel * 2.4) % span;
    const cx = fromEnd ? total / 2 - 3 - run : -total / 2 + 3 + run;
    this.setCam(cam, new THREE.Vector3(cx - 4, 1.8, 6.5), new THREE.Vector3(cx + 1.2, -0.1, 0), 44);
  }

  private drawQ(k: QKind, sh: Shot, t: number, P: boolean) {
    const c = this.mid.ctx, llm = this.ctx.llm, au = this.ctx.audio, story = this.ctx.story;
    const [x0, , x1] = safeBox();
    const blink = Math.floor(au.beatAt(t) * 2) % 2 === 0;
    const st = llm.steps[N_STEPS - 1]!;
    const answered = story.at(t).emitted;
    const toks = llm.steps.slice(0, answered).map((s) => s.tok.t);
    switch (k) {
      case 'spin': {
        tileRow(c, toks, P ? x0 : W * 0.52, P ? H * 0.17 : H * 0.26, P ? 32 : 36, x1, { slot: true, slotOn: blink });
        slab(c, 'ONE MORE', P ? W / 2 : W * 0.52, P ? H * 0.34 : H * 0.56, P ? 104 : 100, { align: P ? 'center' : 'left', top: INK, under: PINK });
        label(c, `ROLL ${N_STEPS} / ${N_STEPS} · ${answered} WRITTEN`, P ? W / 2 : W * 0.52, P ? H * 0.4 : H * 0.68, 30, PINK, P ? 'center' : 'left');
        break;
      }
      case 'climb': {
        const s = story.at(t);
        slab(c, `L${s.layer}`, P ? W / 2 : x0, P ? H * 0.72 : H * 0.52, P ? 200 : 260, { align: P ? 'center' : 'left' });
        label(c, `OF ${llm.cfg.layers} FLOORS · THE LAST TOKEN CLIMBING`, P ? W / 2 : x0, P ? H * 0.8 : H * 0.7, 28, INK, P ? 'center' : 'left');
        break;
      }
      case 'lens': {
        label(c, `WRITING TOKEN ${N_STEPS}:`, x0, P ? H * 0.17 : H * 0.19, 30, INK);
        label(c, `EACH FLOOR'S BEST GUESS`, x0, (P ? H * 0.17 : H * 0.19) + 40, 30, PINK);
        const right = this.camRight(this.camFT).multiplyScalar(-SLAB_W * 0.62);
        for (const l of [26, 28, 30, 31, 32]) {
          const [g, p] = st.lens[l]!.top[0]!;
          const [ex, ey] = toScreen(new THREE.Vector3(0, this.tower.floorY(l) + 0.05, 0).add(right), this.camFT);
          if (ey < 200 || ey > H - 150) continue;
          const col = l === 32 ? PINK : l === 31 ? BLUE : INK;
          c.fillStyle = col;
          c.fillRect(ex - 40, ey - 2, 40, 4);
          label(c, `L${l}  ${showTok(g)}  ${pct(p, l === 32 ? 2 : 1)}`, ex - 52, ey, l >= 31 ? 36 : 28, col, 'right');
        }
        break;
      }
      case 'odds': {
        // the last roll's scoreboard at temperature 1: one token has it all
        const top = st.top10_t1.slice(0, 4);
        const bx = P ? 150 : x0, by = P ? H * 0.2 : H * 0.3, bw = P ? 520 : 760;
        label(c, `THE LAST ROLL · TOP 4 AT TEMPERATURE 1`, bx, by - 60, 28, INK);
        top.forEach(([tok, p], i) => {
          const y = by + i * (P ? 70 : 84);
          tileRow(c, [tok], bx, y, P ? 34 : 40, bx + 400, { hot: i === 0 ? [0] : [] });
          c.fillStyle = i === 0 ? PINK : BLUE;
          c.fillRect(bx + (P ? 180 : 220), y - 20, Math.max(4, bw * p), 40);
          label(c, pct(p, 2), bx + (P ? 180 : 220) + Math.max(4, bw * p) + 16, y, 32, i === 0 ? PINK : INK);
        });
        break;
      }
      case 'close': {
        const p = st.top10_t1[0]![1];
        const tx = P ? W / 2 : W * 0.6, ty = P ? H * 0.66 : H * 0.44;
        slab(c, 'P(<END>)', tx, ty, P ? 90 : 110, { align: P ? 'center' : 'left', top: INK, under: BLUE });
        slab(c, pct(p, 2), tx, ty + (P ? 110 : 140), P ? 120 : 150, { align: P ? 'center' : 'left' });
        break;
      }
      case 'roof': {
        label(c, `THE ROOF: ${llm.cfg.vocab.toLocaleString('en-US')} SCORES`, P ? W / 2 : x0, P ? H * 0.2 : H * 0.2, 30, INK, P ? 'center' : 'left');
        label(c, `ONE OF THEM SAYS STOP`, P ? W / 2 : x0, (P ? H * 0.2 : H * 0.2) + 44, 30, PINK, P ? 'center' : 'left');
        if (!P) tileRow(c, toks, x0, H * 0.5, 32, W * 0.4, { slot: true, slotOn: blink });
        break;
      }
      case 'land': {
        const bx = P ? 140 : x0, bw = P ? W - 280 : W * 0.42, by = P ? H * 0.19 : H * 0.28;
        const ro = { start: Math.max(sh.t0, this.land - 0.3), land: this.land };
        const needle = t < ro.start ? 0 : ease.outCubic(clamp((t - ro.start) / (ro.land - ro.start)));
        label(c, `ROLL ${N_STEPS} / ${N_STEPS} · TOP-P LEFT 1 TOKEN`, bx, by - 84, 28, INK);
        oddsBar(c, st, bx, by, bw, 100, { needle, landed: t >= this.land });
        break;
      }
      case 'done': {
        slab(c, `${answered} / ${N_STEPS}`, x0, P ? H * 0.26 : H * 0.26, P ? 100 : 170, { align: 'left', k: snapIn(t, sh.t0 + 0.04) });
        if (P) {
          label(c, 'TOKENS', x0, H * 0.31, 26, INK);
          label(c, '<END> CLOSES IT', x0, H * 0.31 + 36, 26, PINK);
        } else label(c, 'TOKENS · THE END-OF-TURN TOKEN CLOSES IT', x0, H * 0.37, 28, INK);
        break;
      }
    }
  }

  // ---------------------------------------------------------------- the band: the poster

  private view(k: Focus, j: number, t: number, t0: number, P: boolean) {
    const r = this.poster.rects[k];
    const [x0, y0, x1, y1] = safeBox();
    const sw = x1 - x0, shh = y1 - y0;
    const fill: Record<Focus, number> = {
      full: 0.84, title: 0.96, q: 1.3, ans1: 1.45, ans2: 1.3, two: 0.9, straw: 1.1, foot: 1.6, footTwo: 0.95, footThree: 0.95, three: 0.9, colo: 2.2, digits: 1,
      bigTwo: 0.9, bigThree: 0.9,
    };
    const s = Math.min((sw * fill[k]) / r.w, (shh * (k === 'two' || k === 'three' || k === 'bigTwo' || k === 'bigThree' ? 0.95 : 1.4)) / r.h) * (k === 'full' ? 1 : 1);
    // pan along wide lines, with the drums
    const travel = drive(this.ctx.audio, t, 0.8, 2) - drive(this.ctx.audio, t0, 0.8, 2);
    const over = k === 'full' ? 0 : Math.max(0, r.w * s - sw * 0.9); // the full poster never pans: it must read whole
    const dir = hash(j, 5) < 0.5 ? 1 : -1;
    const panX = over > 0 ? (clamp(travel * 1.1) - 0.5) * over * dir / s : 0;
    const push = 1 + (k === 'full' ? 0.015 : 0.03) * clamp(travel);
    const rot = k === 'full' ? (hash(j, 9) - 0.5) * 0.05 : (hash(j, 9) - 0.5) * 0.03;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2 + (P ? 0 : 0);
    // the question zoom sits a little high so the title above it leaves the frame
    const fy = r.y + r.h / 2 + (k === 'q' ? r.h * 1.3 : 0);
    return { s: s * push, fx: r.x + r.w / 2 + panX, fy, rot, cx, cy };
  }

  private inks(j: number): Inks {
    const v = j % 4;
    return [
      { title: INK, titleUnder: PINK, body: INK, q: BLUE },
      { title: PINK, titleUnder: BLUE, body: BLUE, q: INK },
      { title: BLUE, titleUnder: PINK, body: INK, q: PINK },
      { title: INK, titleUnder: BLUE, body: INK, q: PINK },
    ][v]!;
  }

  private stageB(k: BKind, sh: Shot, t: number, d: number, P: boolean, cam: THREE.PerspectiveCamera) {
    const tw = this.tower, Ht = tw.height, dist = P ? 1.6 : 1;
    const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
    const a = d * 0.12 + sh.i * 1.3;
    this.hideAll();
    for (const c of this.cards) c.group.visible = false;
    this.resetTower(k);
    if (k === 'tiles') this.rowShot(cam, t, d, P, sh.t0);
    else if (k === 'tower') {
      tw.group.visible = true;
      tw.light((d * 0.6) % 1);
      this.setKV(this.ctx.llm.d.prompt.length + N_STEPS);
      this.endTile.visible = true;
      this.endTile.position.set(0, Ht + 0.9, 0);
      this.endTile.scale.setScalar(1.6);
      this.setCam(cam, V(Math.cos(a) * 31 * dist, Ht * 0.5, Math.sin(a) * 31 * dist), V(0, Ht * 0.72, 0), 42, P ? 0 : -5);
      this.endTile.up.set(0, 1, 0);
      this.endTile.lookAt(cam.position.x, Ht + 0.9, cam.position.z);
    } else if (k === 'die') {
      this.flat(cam, P ? 11 : 10);
      this.die.mesh.visible = true;
      this.die.mesh.position.copy(worldAt(cam, P ? 0.5 : 0.66, P ? 0.4 : 0.5));
      this.die.mesh.scale.setScalar(P ? 0.95 : 1.5);
      const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.25 + Math.sin(d * 0.8) * 0.12, -0.5 + Math.sin(d * 0.6) * 0.35, 0));
      this.die.mesh.quaternion.copy(q).multiply(this.die.land);
    }
  }

  /** Where plate p's card lands (px centre, scale, tilt): a new position, size and angle per beat. */
  private pose(p: number, P: boolean) {
    const big = p % 5 === 2;
    const s = big ? 1.2 : 0.92 + 0.14 * hash(p, 1);
    const rot = (hash(p, 2) - 0.5) * (big ? 0.05 : 0.12);
    const cx = W / 2 + (hash(p, 3) - 0.5) * W * (P ? 0.05 : 0.09);
    let cy = H * (P ? 0.47 : 0.475) + (hash(p, 4) - 0.5) * H * 0.03;
    // keep the label (bottom margin) above the HUD's answer strip
    const d = this.cards[0]!.d, labY = cy + (d.sh / 2 - d.mb / 2) * s, maxY = H - (P ? 190 : 128);
    if (labY > maxY) cy -= labY - maxY;
    return { cx, cy, s, rot, dir: hash(p, 5) < 0.5 ? -1 : 1 };
  }

  /** Stage the plate beat: the new card arriving, the previous one (two for a stack) still on the pile. */
  private stagePlate(it: { p: number; mode: Arrive }, t: number, t0: number, P: boolean, cam: THREE.PerspectiveCamera, pile: boolean) {
    this.hideAll();
    for (const c of this.cards) c.group.visible = false;
    this.flat(cam, CAM_D);
    const kpx = (2 * CAM_D * Math.tan(THREE.MathUtils.degToRad(CAM_FOV / 2))) / H;
    // the pile underneath: the previous card (two for a stack); after a poster beat only a stack brings it back
    const under = it.mode === 'stack' ? 2 : pile ? 1 : 0;
    const show: number[] = [];
    for (let b = under; b >= 0; b--) if (it.p - b >= 0) show.push(it.p - b);
    const travel = drive(this.ctx.audio, t, 0.8, 2) - drive(this.ctx.audio, t0, 0.8, 2);
    show.forEach((p, slot) => {
      const card = this.cards[slot]!, pl = this.plates[p]!;
      card.set(pl);
      card.group.visible = true;
      const q = this.pose(p, P);
      let { cx, cy, s, rot } = q;
      let ry = 0;
      const cur = p === it.p;
      if (cur) {
        const k = snapIn(t, t0, 0.09);
        if (it.mode === 'drop') { s *= 1 + 0.5 * (1 - k); rot += 0.12 * (1 - k) * q.dir; }
        else if (it.mode === 'slide') cx += q.dir * W * 1.1 * (1 - k);
        else if (it.mode === 'flip') ry = q.dir * (Math.PI / 2) * (1 - k);
        else { cy -= H * 0.9 * (1 - k); rot -= 0.1 * (1 - k) * q.dir; }
        // after landing: a slow drift and push with the drums
        cx += q.dir * 10 * travel; cy -= 5 * travel; s *= 1 + 0.02 * clamp(travel);
      }
      card.group.position.set((cx - W / 2) * kpx, -(cy - H / 2) * kpx, (slot - show.length + 1) * 0.03);
      card.group.scale.setScalar(kpx * s);
      card.group.rotation.set(0, ry, rot);
    });
    return show;
  }

  /** Keep only the plates on screen resident on the GPU (the rest re-upload if a seek brings them back). */
  private keepResident(show: number[]) {
    const need = new Set(show);
    for (const p of this.resident) if (!need.has(p)) { this.plates[p]!.tex.dispose(); this.resident.delete(p); }
    for (const p of show) this.resident.add(p);
  }

  private drawB(k: BKind, sh: Shot, j: number, t: number, P: boolean, t0 = sh.t0) {
    const c = this.mid.ctx, llm = this.ctx.llm;
    const [x0] = safeBox();
    if (k === 'tiles') {
      if (P) { label(c, 'THE ANSWER', x0, H * 0.2, 30, INK); label(c, `${N_STEPS} TOKENS`, x0, H * 0.2 + 42, 30, PINK); }
      else label(c, `THE ANSWER · ${N_STEPS} TOKENS`, x0, H * 0.2, 32, INK);
      return;
    }
    if (k === 'tower') {
      label(c, `${llm.cfg.layers} FLOORS · ${N_STEPS} CLIMBS`, P ? W / 2 : x0, P ? H * 0.2 : H * 0.2, 32, INK, P ? 'center' : 'left');
      if (!P) tileRow(c, llm.steps.map((s) => s.tok.t), x0, H * 0.4, 36, W * 0.5, { hot: [N_STEPS - 1] });
      return;
    }
    if (k === 'die') {
      const p = llm.steps[N_STEPS - 1]!.top10_t1[0]![1];
      slab(c, '<END>', P ? W / 2 : x0, P ? H * 0.7 : H * 0.4, P ? 130 : 160, { align: P ? 'center' : 'left', top: PINK, under: INK });
      label(c, `${pct(p, 2)} AT TEMPERATURE 1`, P ? W / 2 : x0, P ? H * 0.77 : H * 0.54, 32, INK, P ? 'center' : 'left');
      return;
    }
    // the poster, re-printed: a new view, new inks, new plate offset
    const v = this.view(k, j, t, t0, P);
    const misreg: [number, number] = [(hash(j, 1) - 0.5) * 2, (hash(j, 2) - 0.5) * 2];
    c.save();
    c.translate(v.cx, v.cy);
    c.scale(v.s, v.s);
    c.rotate(v.rot);
    c.translate(-v.fx, -v.fy);
    this.poster.draw(c, this.inks(j), misreg);
    c.restore();
  }

  // ---------------------------------------------------------------- frame

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio;
    const P = portrait();
    // after the hard stop: blank paper
    if (f.ft >= this.stop) {
      this.wash.render(r, out, { seed: 0, drift: 0, amt: 0 });
      return { fade: 1, hud: 0, kinetic: 0, lyric: 0 };
    }
    const isBand = f.ft >= this.band;
    const sh = isBand ? shot(this.ctx, f, { every: 1, from: this.band }) : shot(this.ctx, f, { every: 2, from: this.q0 });
    const j = sh.i;
    const it: BItem = this.seq[Math.min(j, this.seq.length - 1)] ?? { kind: 'final' };
    // the final poster holds one framing (and one ink set) from its first beat; each beat is still a new print
    const finJ = this.seq.findIndex((x) => x.kind === 'final');
    const finT0 = finJ >= 0 ? this.ctx.audio.timeOfBeat(Math.round(this.ctx.audio.beatAt(this.band)) + finJ) : sh.t0;
    const qk: QKind = Q_PLAN[Math.min(sh.i, Q_PLAN.length - 1)]!;
    const d = drive(au, f.t, 0.6, 1.8), dft = drive(au, f.ft, 0.6, 1.8);

    // background: wash + rain
    const c1 = sh.i % 2 ? 'blue' : 'pink';
    const loud = isBand ? 0.7 : 0.55;
    this.wash.render(r, out, { seed: sh.seed + (isBand ? 500 : 0), drift: d, amt: loud, c1, c2: c1 === 'pink' ? 'blue' : 'pink' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => rain(this.back.ctx, this.pool, dft * 0.5, { seed: sh.seed, density: 0.2 })), out);

    // type (once per frame, with the camera at the frame's own time for pinned labels)
    const midTex = perFrame(this.mid, f, () => {
      const c = this.mid.ctx;
      c.save();
      safeClip(c);
      if (isBand) {
        if (it.kind === 'poster') this.drawB(it.focus, sh, j, f.ft, P);
        else if (it.kind === 'final') this.drawB('full', sh, finJ, f.ft, P, finT0);
        else if (it.kind === 'cb') this.drawB(it.k, sh, j, f.ft, P);
      }
      else { this.stageQ(qk, sh, f.ft, dft, P, this.camFT); this.camFT.updateMatrixWorld(); this.drawQ(qk, sh, f.ft, P); }
      c.restore();
    });

    // 3D
    if (isBand) {
      if (it.kind === 'plate') this.keepResident(this.stagePlate(it, f.t, sh.t0, P, this.cam, this.seq[j - 1]?.kind === 'plate'));
      else if (it.kind === 'cb') this.stageB(it.k, sh, f.t, d, P, this.cam);
      else { this.hideAll(); for (const c of this.cards) c.group.visible = false; }
    } else this.stageQ(qk, sh, f.t, d, P, this.cam);
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);
    this.ctx.comp.draw(r, midTex, out);

    // foreground specks (faster once the band is in)
    const c = this.front.ctx;
    this.front.clear();
    specks(c, drive(au, f.t, 1.1, isBand ? 2.6 : 1.8), { seed: sh.seed, n: isBand ? 90 : 64 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    const landHit = !isBand && qk === 'land' ? 0.55 * pulse(f.t, this.land, 0.06) : 0;
    return { reg: sh.seed + (isBand ? 700 : 0), flood: landHit };
  }
}

