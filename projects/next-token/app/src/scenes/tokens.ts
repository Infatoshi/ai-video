// tokens (verse 1, lines 1-4): "Type a question, hit enter, now it's gone to the machine, / Chopped up
// into tokens, little pieces that it's seen, / It don't see letters, only chunks that it knows, / So count
// the R's in strawberry and watch how it goes". Text becomes tokens, with the real run's data: the question
// (llm.d.question) typed on a keyboard, ENTER, into the tower's lobby; chopped at its real token boundaries
// (question_tokens) with real ids (prompt); the full 43-token chat-templated prompt on a conveyor (special
// tokens as black tiles); the tiles flip from letters to ids; the model's view as 43 numbers; strawberry's
// ten letters and three R's collapse into ONE token (73700), and the "?" slot of what comes next.
// Cuts land on beats anchored to lyric words (shotAtTimes): 11 shots, each a different framing or idea.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font, layout, measure, fitSize, type TextLayout } from '../engine/type';
import { clamp, ease, hash, lerp } from '../engine/util';
import { showTok } from '../engine/llm';
import { Wash, tokenRain, specks, makeCam, drive, shotAtTimes, snapIn, portrait, perFrame, tokenTile, type InkName } from './_kit';
import { Tower } from './_tower';
import { buildKeyboard, keyFor, faceTile, faceTexture, canvasTex, Belt, type Key } from './tokens-props';
import { HEX } from '../engine/palette';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

const KINDS = ['type', 'enter', 'lobby', 'chop', 'belt', 'drop', 'flip', 'view', 'letters', 'one', 'next'] as const;
type Kind = typeof KINDS[number];

/** Belt geometry: the belt starts inside the lobby (x = 0) and runs along +x; tiles land at X_DROP. */
const BELT_TOP = 0.36, X_DROP = 16, GAP = 0.32, TILE_H = 0.9;

interface ChopPiece { tok: string; label: string; id: number; o: number; cx: number; cyy: number; tx: number; ty: number; tw: number }

export default class Tokens extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(40);
  cam2 = makeCam(40); // the same framing at the frame's own time, for 2D labels anchored to 3D
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  front = new Layer2D();
  tower = new Tower({ kvMax: 48 });

  kb!: { group: THREE.Group; keys: Key[]; enter: Key };
  card!: THREE.Mesh;
  belt = new Belt(0, 44, BELT_TOP);
  beltTiles: THREE.Mesh[] = [];
  beltW: number[] = [];
  beltP: number[] = [];
  n0 = 17;
  flip = new THREE.Group();
  flipTiles: THREE.Mesh[] = [];
  flipW: number[] = [];
  view = new THREE.Group();
  viewTiles: THREE.Mesh[] = [];
  one!: THREE.Mesh;

  cuts: number[] = [];
  /** Word times (song s) and derived beat-grid times. */
  tm = {
    type: 0, hit: 0, enter: 0, now: 0, gone: 0, chopped: 0, up: 0, into: 0, tokens: 0, dont: 0, see: 0, letters: 0,
    chunks: 0, count: 0, the: 0, rs: 0, straw: 0, watch: 0, flip0: 0, s16: 0, bE: 0,
  };
  qStart = 30;
  strawIdx = 36;
  split: { t: string; id?: number }[] = [];
  chop: ChopPiece[] = [];
  chopSize = 64;
  letters: { ch: string; x: number; y: number; w: number }[] = [];
  letterSize = 200;
  letterY = [0, 0];

  override init() {
    const { lyrics: ly, llm, audio: au } = this.ctx;
    const P = portrait();
    const norm = (s: string) => s.toLowerCase().replace(/[^a-z]/g, '');
    const wd = (line: string, w: string) => {
      const l = ly.get(line);
      return (l.words.find((x) => norm(x.w).startsWith(w)) ?? l.words[0]!).start;
    };
    const L1 = 'Type a question', L2 = 'Chopped up into tokens', L3 = "It don't see letters", L4 = 'count the R';
    const tm = this.tm;
    Object.assign(tm, {
      type: wd(L1, 'type'), hit: wd(L1, 'hit'), enter: wd(L1, 'enter'), now: wd(L1, 'now'), gone: wd(L1, 'gone'),
      chopped: wd(L2, 'chopped'), up: wd(L2, 'up'), into: wd(L2, 'into'), tokens: wd(L2, 'tokens'),
      dont: wd(L3, 'dont'), see: wd(L3, 'see'), letters: wd(L3, 'letters'), chunks: wd(L3, 'chunks'),
      count: wd(L4, 'count'), the: wd(L4, 'the'), rs: wd(L4, 'rs'), straw: wd(L4, 'strawberry'), watch: wd(L4, 'watch'),
    });
    const B = (t: number) => au.beatAt(t), T = (b: number) => au.timeOfBeat(b);
    const near = (t: number) => T(Math.round(B(t)));
    const floorB = (t: number) => T(Math.floor(B(t + 0.02)));
    const q16 = (t: number) => T(Math.round(B(t) * 4) / 4);
    // the cut list: shot 0 from the scene start; the rest on beats anchored to lyric words
    const cB = near(tm.enter), cC = near(tm.gone), cD = floorB(ly.get(L2).start), cE = T(B(cD) + 4), cF = T(B(cE) + 3);
    const cG = floorB(ly.get(L3).start), cH = T(B(cG) + 4), cI = floorB(ly.get(L4).start), cJ = near(tm.straw), cK = T(B(cJ) + 2);
    this.cuts = [this.ctx.start, cB, cC, cD, cE, cF, cG, cH, cI, cJ, cK];
    for (let i = 1; i < this.cuts.length; i++) this.cuts[i] = Math.max(this.cuts[i]!, this.cuts[i - 1]! + 0.1);
    tm.bE = Math.round(B(cE) * 4) / 4;
    tm.s16 = (T(Math.round(B(cD)) + 4) - T(Math.round(B(cD)))) / 16;
    tm.flip0 = q16(tm.dont);

    const prompt = llm.d.prompt;
    this.qStart = llm.questionStart;
    this.strawIdx = prompt.findIndex((p) => p.t === ' strawberry');
    // strawberry without the leading space: str | aw | berry (ids from the answer, where the model spells it so)
    const idOf = (t: string) => llm.steps.find((s) => s.tok.t === t)?.tok.id ?? prompt.find((p) => p.t === t)?.id;
    this.split = (llm.d.strawberry['strawberry'] ?? []).map((t) => ({ t, id: idOf(t) }));

    this.scene.add(this.tower.group);
    this.tower.outlineAll();
    this.tower.setKV(0);

    // keyboard
    this.kb = buildKeyboard();
    this.scene.add(this.kb.group);

    // the question card that flies into the lobby
    const q = llm.d.question;
    const cardTex = canvasTex(1600, 220, (c) => {
      c.fillStyle = HEX.paper; c.fillRect(0, 0, 1600, 220);
      c.strokeStyle = HEX.ink; c.lineWidth = 16; c.strokeRect(8, 8, 1584, 204);
      c.fillStyle = HEX.pink; c.fillRect(40, 60, 22, 100);
      c.fillStyle = HEX.ink; c.font = font(F.mono(700), 92); c.textBaseline = 'middle';
      c.fillText(q, 96, 112);
    });
    this.card = faceTile(cardTex, null, 8, 1.1, 0.12, 'ink', 2.4);
    this.scene.add(this.card);

    // the conveyor and the prompt's 43 tiles (special tokens black, the question pink)
    this.scene.add(this.belt.group);
    prompt.forEach((p, i) => {
      const special = p.t.startsWith('<|');
      const inQ = i >= this.qStart && i < this.qStart + llm.d.question_tokens.length;
      const m = tokenTile(p.t, special ? { face: 'ink', text: 'paper', side: 'ink', id: p.id, h: TILE_H }
        : inQ ? { face: 'pink', text: 'paper', side: 'blue', id: p.id, h: TILE_H } : { face: 'paper', text: 'ink', side: 'blue', id: p.id, h: TILE_H });
      m.rotation.x = -Math.PI / 2;
      const w = (m.geometry as THREE.BoxGeometry).parameters.width;
      this.beltW.push(w);
      this.beltP.push(i === 0 ? 0 : this.beltP[i - 1]! + (this.beltW[i - 1]! + w) / 2 + GAP);
      this.belt.group.add(m);
      this.beltTiles.push(m);
    });

    // the question's 8 tiles, letters on the front, ids on the back
    const qt = llm.d.question_tokens;
    for (let k = 0; k < qt.length; k++) {
      const p = prompt[this.qStart + k]!;
      const label = showTok(p.t);
      const h = 1.1, w = Math.max(1.35, 0.36 * label.length + 0.6) * h;
      const straw = this.qStart + k === this.strawIdx;
      const m = faceTile(faceTexture(label, 'paper', 'ink', w / h), faceTexture(String(p.id), straw ? 'pink' : 'blue', 'paper', w / h), w, h, 0.38, 'blue', 2.2);
      this.flip.add(m);
      this.flipTiles.push(m);
      this.flipW.push(w);
    }
    this.scene.add(this.flip);

    // the model's view: all 43 as ids only, laid flat in a grid
    const cols = P ? 5 : 9;
    prompt.forEach((p, i) => {
      const special = p.t.startsWith('<|');
      const inQ = i >= this.qStart && i < this.qStart + qt.length;
      const bg: InkName = special ? 'ink' : inQ ? 'pink' : 'paper', fg: InkName = special || inQ ? 'paper' : 'ink';
      const m = faceTile(faceTexture(String(p.id), bg, fg, 2.2), null, 2.2, 1, 0.3, special ? 'ink' : 'blue', 2);
      m.rotation.x = -Math.PI / 2;
      const r = Math.floor(i / cols), c = i % cols, rows = Math.ceil(prompt.length / cols);
      m.userData.r = r;
      m.position.set((c - (cols - 1) / 2) * 2.5, 0, (r - (rows - 1) / 2) * 1.32);
      this.view.add(m);
      this.viewTiles.push(m);
    });
    this.scene.add(this.view);

    // ONE token: strawberry as the model gets it
    const sid = prompt[this.strawIdx]?.id ?? 0;
    this.one = faceTile(faceTexture(String(sid), 'pink', 'paper', 4.4 / 1.8), null, 4.4, 1.8, 0.7, 'blue', 3);
    this.scene.add(this.one);

    // chop layout (2D): contiguous (the typed sentence) and tiles (flow-wrapped)
    const size = P ? 76 : 84, fam = F.mono(700);
    this.chopSize = size;
    const pad = size * 0.32, gap = size * 0.34, maxW = W - (P ? 120 : 260);
    let o = 0;
    const pieces: ChopPiece[] = qt.map((t, k) => {
      const piece: ChopPiece = { tok: t, label: showTok(t), id: prompt[this.qStart + k]!.id, o, cx: 0, cyy: 0, tx: 0, ty: 0, tw: 0 };
      o += t.length;
      piece.tw = measure(piece.label, fam, size) + pad * 2;
      return piece;
    });
    const qW = measure(q, fam, size);
    const rowsC: ChopPiece[][] = [[]];
    let acc = 0;
    for (const p of pieces) {
      if (acc + p.tw > maxW && rowsC[rowsC.length - 1]!.length) { rowsC.push([]); acc = 0; }
      rowsC[rowsC.length - 1]!.push(p);
      acc += p.tw + gap;
    }
    const cy = H * (P ? 0.36 : 0.345), rowH = size * 2.7;
    rowsC.forEach((row, ri) => {
      const tot = row.reduce((a, p) => a + p.tw, 0) + gap * (row.length - 1);
      let x = W / 2 - tot / 2;
      for (const p of row) { p.tx = x; p.ty = cy + (ri - (rowsC.length - 1) / 2) * rowH; x += p.tw + gap; }
    });
    // the contiguous sentence (as typed): one line if it fits, otherwise wrapped the same way as the tiles
    const oneLine = qW < maxW;
    rowsC.forEach((row, ri) => {
      const rowText = row.map((p) => p.tok).join('');
      const x0 = oneLine ? W / 2 - qW / 2 : W / 2 - measure(rowText, fam, size) / 2;
      let acc2 = '';
      for (const p of row) {
        p.cx = oneLine ? x0 + measure(q.slice(0, p.o), fam, size) : x0 + measure(acc2, fam, size);
        p.cyy = oneLine ? cy : p.ty;
        acc2 += p.tok;
      }
      void ri;
    });
    this.chop = pieces;
    const lf = F.archivo(125, 900);
    const lines = P ? ['straw', 'berry'] : ['strawberry'];
    const lsz = Math.min(...lines.map((ln) => fitSize(ln, lf, W * (P ? 0.84 : 0.84), P ? 330 : 330)));
    this.letterSize = lsz;
    const yc = H * (P ? 0.44 : 0.47);
    lines.forEach((ln, li) => {
      const L = layout(ln, lf, lsz);
      const y = yc + (li - (lines.length - 1) / 2) * lsz * 1.12;
      for (const g of L.glyphs) this.letters.push({ ch: g.ch, x: W / 2 - L.width / 2 + g.x, y, w: g.w });
    });
    this.letterY = [yc - (lines.length - 1) / 2 * lsz * 1.12, yc + (lines.length - 1) / 2 * lsz * 1.12];
  }

  // ------------------------------------------------------------------ helpers

  private aim(cam: THREE.PerspectiveCamera, pos: THREE.Vector3, look: THREE.Vector3, fov: number, shift = 0, up = new THREE.Vector3(0, 1, 0)) {
    cam.position.copy(pos);
    cam.up.copy(up);
    cam.lookAt(look);
    cam.fov = fov;
    if (shift) cam.setViewOffset(W, H, 0, -shift * H, W, H); else cam.clearViewOffset();
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
  }

  private project(cam: THREE.PerspectiveCamera, p: THREE.Vector3): [number, number] {
    const v = p.clone().project(cam);
    return [(v.x * 0.5 + 0.5) * W, (0.5 - v.y * 0.5) * H];
  }

  /** Big two-ink type: blue underprint, pink over it with multiply (purple where they overlap). */
  private slab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, k = 1, align: CanvasTextAlign = 'center', top: string = PINK) {
    size = Math.min(size, fitSize(text, F.archivo(125, 900), W * (portrait() ? 0.86 : 0.84), size));
    c.save();
    c.font = font(F.archivo(125, 900), size);
    c.textAlign = align;
    c.textBaseline = 'middle';
    c.translate(x, y);
    const s = 1.2 - 0.2 * k;
    c.scale(s, s);
    c.fillStyle = BLUE;
    c.fillText(text, size * 0.035, size * 0.03);
    c.globalCompositeOperation = 'multiply';
    c.fillStyle = top;
    c.fillText(text, 0, 0);
    c.restore();
  }

  /** Mono label, solid ink, bold (never below 22 px: everything goes through the halftone). */
  private lab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, color = INK, align: CanvasTextAlign = 'left') {
    size = Math.max(22, size);
    c.font = font(F.mono(700), size);
    // a paper knockout behind the label so the rain never runs through small type
    const w = c.measureText(text).width, x0 = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
    c.fillStyle = PAPER;
    c.fillRect(x0 - 8, y - size * 0.72, w + 16, size * 1.44);
    c.fillStyle = color;
    c.textAlign = align;
    c.textBaseline = 'middle';
    c.fillText(text, x, y);
    c.textAlign = 'left';
  }

  /** A 2D token chip: filled box with its label (and optional id under it). Returns its width. */
  private chip(c: CanvasRenderingContext2D, label: string, x: number, y: number, size: number, bg: string, fg: string, border = INK): number {
    c.font = font(F.mono(700), size);
    const w = c.measureText(label).width + size * 0.7, h = size * 1.5;
    c.fillStyle = bg; c.fillRect(x, y - h / 2, w, h);
    c.strokeStyle = border; c.lineWidth = Math.max(3, size * 0.1); c.strokeRect(x, y - h / 2, w, h);
    c.fillStyle = fg; c.textBaseline = 'middle'; c.textAlign = 'left';
    c.fillText(label, x + size * 0.35, y + size * 0.04);
    return w;
  }

  /** A strip of chips scrolling sideways (wraps around), colours per item. */
  private ticker(c: CanvasRenderingContext2D, labels: string[], y: number, cs: number, scroll: number, col: (i: number) => string[]) {
    c.font = font(F.mono(700), cs);
    const widths = labels.map((l) => c.measureText(l).width + cs * 0.7 + 10);
    const total = widths.reduce((a, b) => a + b, 0);
    let x = -((scroll % total) + total) % total;
    for (let rep = 0; rep < 4 && x < W; rep++) {
      labels.forEach((l, i) => {
        const w = widths[i]!;
        if (x + w > 0 && x < W) { const [bg, fg] = col(i); this.chip(c, l, x, y, cs, bg!, fg!); }
        x += w;
      });
    }
  }

  /** Chars of the question typed by t: from "Type" to just before ENTER lands. */
  private typed(t: number) {
    const n = this.ctx.llm.d.question.length;
    return Math.floor(n * clamp((t - this.tm.type) / Math.max(0.3, this.cuts[1]! - 0.07 - this.tm.type)) + 1e-6);
  }

  /** Belt: continuous step index since the belt shot (16ths), and the prompt position it has reached. */
  private beltStep(t: number) {
    const au = this.ctx.audio;
    const b = au.beatAt(t) * 4 - this.tm.bE * 4;
    if (b < 0) return 0;
    const k = Math.floor(b);
    return Math.min(this.beltTiles.length - 1 - this.n0, k + ease.outExpo(clamp((b - k) / 0.6)));
  }
  private beltPos(s: number) {
    const x = clamp(this.n0 + s, 0, this.beltP.length - 1), i = Math.floor(x), fr = x - i;
    return lerp(this.beltP[i]!, this.beltP[Math.min(i + 1, this.beltP.length - 1)]!, fr);
  }

  /** Flip time of question tile k (a 16th apart from "don't"). */
  private flipT(k: number) { return this.tm.flip0 + k * this.tm.s16; }

  /** Flood the tower's floors below `level` (0..33) with pink, the front floor solid: the prefill climbing. -1 = off. */
  private fill(level: number) {
    const T = this.tower;
    for (let l = 1; l <= T.layers; l++) {
      const d = level - l;
      const v = level < 0 ? 0 : d >= 0 && d < 1.5 ? 1 : d >= 1.5 ? 0.35 : 0;
      (T.slabs[l]!.material as THREE.ShaderMaterial).uniforms.hot!.value = v;
      (T.ff[l - 1]!.material as THREE.ShaderMaterial).uniforms.hot!.value = v * 0.9;
    }
  }

  // ------------------------------------------------------------------ 3D staging

  /** Pose the 3D world for `kind` at time t and set `cam` (pure function of t). Returns false if nothing 3D shows. */
  private pose(kind: Kind, t: number, cam: THREE.PerspectiveCamera, sh: { t0: number; t1: number }, seed: number): boolean {
    const au = this.ctx.audio, P = portrait(), tm = this.tm;
    const d = drive(au, t, 0.5, 1.6);
    const dl = d - drive(au, sh.t0, 0.5, 1.6); // distance since the cut: small drifts start at 0 on every shot
    const vis = (o: THREE.Object3D, v: boolean) => { o.visible = v; };
    vis(this.kb.group, kind === 'type' || kind === 'enter');
    vis(this.card, kind === 'lobby');
    vis(this.tower.group, kind === 'lobby' || kind === 'belt' || kind === 'drop');
    vis(this.belt.group, kind === 'belt' || kind === 'drop');
    vis(this.flip, kind === 'flip' || kind === 'next');
    vis(this.view, kind === 'view');
    vis(this.one, kind === 'one');
    const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
    const side = hash(seed, 3) < 0.5 ? -1 : 1;

    switch (kind) {
      case 'type': {
        // keys go down as the question is typed on them
        const q = this.ctx.llm.d.question, n = q.length;
        const t0 = tm.type, dt = Math.max(0.3, this.cuts[1]! - 0.07 - t0) / n;
        const press = new Map<Key, number>();
        for (let k = 0; k < n; k++) {
          const key = keyFor(this.kb.keys, q[k]!);
          if (!key) continue;
          const tc = t0 + k * dt, x = t - tc;
          const v = x < 0 ? 0 : x < 0.03 ? x / 0.03 : Math.max(0, 1 - (x - 0.03) / 0.09);
          press.set(key, Math.max(press.get(key) ?? 0, v));
          if (/[A-Z?]/.test(q[k]!)) { const sh2 = this.kb.keys.find((kk) => kk.label === 'SHIFT'); if (sh2) press.set(sh2, Math.max(press.get(sh2) ?? 0, v)); }
        }
        for (const k of this.kb.keys) {
          const v = press.get(k) ?? 0;
          k.mesh.position.y = 0.21 - 0.16 * v;
          k.mat.uniforms.hot!.value = v;
        }
        const a = -0.14 * side + dl * 0.012;
        if (P) this.aim(cam, V(Math.sin(a) * 17 - 0.4, 21, Math.cos(a) * 17), V(-0.4, 0, 0.3), 44, 0.16);
        else this.aim(cam, V(Math.sin(a) * 11.5 - 0.4, 9.5, Math.cos(a) * 11.5), V(-0.4, 0, 0.2), 42, 0.2);
        return true;
      }
      case 'enter': {
        const E = this.kb.enter;
        const down = snapIn(t, sh.t0, 0.06);
        for (const k of this.kb.keys) { k.mesh.position.y = 0.21; k.mat.uniforms.hot!.value = 0; }
        E.mesh.position.y = 0.21 - 0.2 * down;
        E.mat.uniforms.hot!.value = down;
        const a = -0.45 + dl * 0.02;
        const at = V(E.x - 0.1, 0.1, E.z);
        if (P) this.aim(cam, at.clone().add(V(Math.sin(a) * 2.4, 4.6, Math.cos(a) * 2.4)), at.clone().add(V(0, 0, 0.2)), 50, 0.04);
        else this.aim(cam, at.clone().add(V(Math.sin(a) * 2.5, 3.1, Math.cos(a) * 2.5)), at.clone().add(V(0, 0.05, 0.05)), 42, 0.1);
        return true;
      }
      case 'lobby': {
        // the question flies into the lobby on a beat, then the floors light bottom to top (the prefill)
        const a = 0.6 * side + dl * 0.02;
        const R = P ? 19 : 13.5;
        const pos = V(Math.sin(a) * R, P ? 2.4 : 2.0, Math.cos(a) * R);
        const look = V(0, P ? 8 : 5.6, 0);
        this.aim(cam, pos, look, P ? 66 : 60);
        const tArr = au.timeOfBeat(Math.round(au.beatAt(sh.t0)) + 1);
        const u = clamp((t - sh.t0) / Math.max(0.05, tArr - sh.t0));
        const dir = V(Math.sin(a), 0, Math.cos(a));
        const right = V(Math.cos(a), 0, -Math.sin(a));
        const p0 = pos.clone().add(dir.clone().multiplyScalar(-4)).add(right.clone().multiplyScalar(P ? 0.6 : 1.6)).add(V(0, P ? 1.6 : 1.4, 0));
        const p1 = V(0, 0.42, 0);
        const k = ease.inCubic(u);
        this.card.position.copy(p0.clone().lerp(p1, k));
        this.card.lookAt(pos);
        this.card.scale.setScalar(u >= 1 ? 0.0001 : lerp(0.9, 0.12, k));
        this.fill(t < tArr ? -1 : 34 * clamp((t - tArr) / Math.max(0.1, sh.t1 - tArr)));
        return true;
      }
      case 'belt':
      case 'drop': {
        const s = this.beltStep(t), Pb = this.beltPos(s);
        this.belt.setOffset(Pb);
        // on the belt (u <= 0) or waiting in the hopper above the drop point (u = steps until it drops)
        this.beltTiles.forEach((m, i) => {
          const u = i - (this.n0 + s);
          const x = Math.min(X_DROP, X_DROP - (Pb - this.beltP[i]!));
          const inside = x + this.beltW[i]! / 2 < 1.2;
          m.visible = !inside && u <= 3;
          m.position.set(x, BELT_TOP + TILE_H * 0.35 / 2 + 0.02 + 2.2 * clamp(u) + 1.3 * Math.max(0, u - 1), 0);
        });
        this.fill(-1);
        this.tower.light((d * 0.3) % 1);
        if (kind === 'belt') {
          const a = dl * 0.25;
          if (P) this.aim(cam, V(20 + a, 15, 9), V(9.2, 1.2, 0), 56);
          else this.aim(cam, V(16.5 + a, 11.5, 8.2), V(9.4, 0.8, 0), 50);
        } else {
          const a = dl * 0.12;
          if (P) this.aim(cam, V(X_DROP + 0.4 + a, 8.2, 3.6), V(X_DROP - 1.6, 0.3, -0.2), 56);
          else this.aim(cam, V(X_DROP - 0.6 + a, 3.8, 4.4), V(X_DROP - 1.9, 0.3, 0), 42);
        }
        return true;
      }
      case 'flip':
      case 'next': {
        const n = this.flipTiles.length, gap = 0.3;
        const rows: number[][] = kind === 'flip' ? (P ? [[0, 1, 2], [3, 4, 5], [6, 7]] : [[0, 1, 2, 3], [4, 5, 6, 7]]) : [[...Array(n).keys()]];
        this.flip.scale.setScalar(kind === 'flip' ? 1.3 : 1);
        rows.forEach((row, ri) => {
          const tot = row.reduce((a, k) => a + this.flipW[k]!, 0) + gap * (row.length - 1);
          let x = -tot / 2 - (kind === 'next' ? 0.9 : 0);
          for (const k of row) {
            const m = this.flipTiles[k]!, w = this.flipW[k]!;
            const fu = kind === 'next' ? 1 : ease.outExpo(clamp((t - this.flipT(k)) / 0.09));
            m.rotation.set(Math.PI * fu, 0, 0);
            m.position.set(x + w / 2, (ri - (rows.length - 1) / 2) * -1.6 + Math.sin(Math.PI * fu) * 0.35, 0);
            x += w + gap;
          }
        });
        if (kind === 'flip') {
          const a = side * 0.25 + dl * 0.012;
          if (P) this.aim(cam, V(Math.sin(a) * 23, -2.6, Math.cos(a) * 23), V(0, -0.1, 0), 46, 0.07);
          else this.aim(cam, V(Math.sin(a) * 14.5, -2.4, Math.cos(a) * 14.5), V(0, -0.2, 0), 44, 0.05);
        } else {
          // along the row from the empty slot at its end
          const last = this.flipTiles[n - 1]!, xEnd = last.position.x + this.flipW[n - 1]! / 2 + 0.3 + 0.85;
          const a = dl * 0.1;
          if (P) this.aim(cam, V(xEnd - 1.6 - a, 1.8, 15), V(xEnd - 2.5, -0.2, 0), 54, 0.03);
          else this.aim(cam, V(xEnd + 1.2 - a, 1.2, 6.8), V(xEnd - 2.9, -0.1, 0), 46);
        }
        return true;
      }
      case 'view': {
        // tiles land row by row on the 16ths, seen from straight above; the camera turns slowly
        this.viewTiles.forEach((m) => {
          const tr = sh.t0 + (m.userData.r as number) * tm.s16;
          m.visible = t >= tr;
          m.position.y = (1 - snapIn(t, tr, 0.07)) * 3;
        });
        const a = side * 0.06 + dl * 0.01;
        this.aim(cam, V(0, P ? 31 : 23, 0.001), V(0, 0, 0), P ? 46 : 40, P ? 0.02 : 0.07, V(Math.sin(a), 0, -Math.cos(a)));
        return true;
      }
      case 'one': {
        const k = snapIn(t, sh.t0, 0.08);
        this.one.position.set(0, (1 - k) * 1.4, 0);
        this.one.scale.setScalar(1.35 - 0.35 * k);
        this.one.rotation.set(0.05, -0.28 * side + dl * 0.01, 0);
        if (P) this.aim(cam, V(1.4 * side, 1.5, 10.5), V(0, 0.1, 0), 44, -0.02);
        else this.aim(cam, V(2.2 * side, 1.25, 7.4), V(0, -0.1, 0), 38, -0.03);
        return true;
      }
      default:
        return false;
    }
  }

  // ------------------------------------------------------------------ 2D (once per frame, at f.ft)

  private type2D(kind: Kind, ft: number, sh: { t0: number; t1: number }, seed: number) {
    const c = this.mid.ctx, P = portrait(), tm = this.tm, llm = this.ctx.llm, au = this.ctx.audio;
    const q = llm.d.question;
    const blink = Math.floor(au.beatAt(ft) * 2) % 2 === 0;
    switch (kind) {
      case 'type':
      case 'enter': {
        // the prompt box; ENTER sends it away on "now"
        const size = P ? 62 : 66;
        const lines = P ? ['How many r\'s are', 'in strawberry?'] : [q];
        const bw = P ? W - 120 : Math.min(W - 260, 1500), bh = lines.length * size * 1.35 + size * 1.1;
        let bx = (W - bw) / 2;
        const by = P ? H * 0.19 : H * 0.18;
        if (kind === 'enter') {
          const tNow = au.timeOfBeat(Math.round(au.beatAt(tm.now)));
          const k = ease.inExpo(clamp((ft - tNow) / 0.12));
          bx += k * (W + 200);
          if (k > 0 && k < 1) { c.fillStyle = PINK; for (let i = 0; i < 4; i++) c.fillRect(bx - 520 - i * 90, by + bh * (0.2 + i * 0.2), 480, 10); }
        }
        c.fillStyle = PAPER; c.fillRect(bx, by, bw, bh);
        c.strokeStyle = INK; c.lineWidth = 6; c.strokeRect(bx, by, bw, bh);
        this.lab(c, 'PROMPT', bx, by - 22, 24, BLUE);
        const n = kind === 'type' ? this.typed(ft) : q.length;
        c.font = font(F.mono(700), size);
        c.fillStyle = INK; c.textBaseline = 'middle';
        let left = n, cx = bx + 44, cy = by + size * 1.1;
        lines.forEach((ln, li) => {
          const s = ln.slice(0, Math.max(0, left));
          left -= ln.length + (P && li === 0 ? 1 : 0);
          const y = by + size * 0.55 + (li + 0.5) * size * 1.35;
          c.fillStyle = INK;
          c.fillText(s, bx + 44, y);
          if (s.length || li === 0) { cx = bx + 44 + c.measureText(s).width; cy = y; }
        });
        if (kind === 'type' && (blink || (ft > tm.type && n < q.length))) { c.fillStyle = PINK; c.fillRect(cx + 6, cy - size * 0.5, size * 0.5, size); }
        if (kind === 'type') this.lab(c, `${n} / ${q.length} CHARACTERS`, bx + bw, by + bh + 30, 24, INK, 'right');
        break;
      }
      case 'lobby': {
        const [lx, ly] = this.project(this.cam2, new THREE.Vector3(0, 0.1, 0));
        const tx = P ? W * 0.5 : Math.min(W - 520, lx + 380), ty = P ? H * 0.915 : H * 0.8;
        c.strokeStyle = INK; c.lineWidth = 4;
        c.beginPath(); c.moveTo(lx, ly); c.lineTo(tx, ty - 34); c.stroke();
        c.fillStyle = PINK; c.beginPath(); c.arc(lx, ly, 12, 0, Math.PI * 2); c.fill();
        this.lab(c, 'FLOOR 0 · EMBEDDING', tx, ty, 30, INK, P ? 'center' : 'left');
        this.lab(c, `THEN ${llm.cfg.layers} LAYERS UP`, tx, ty + 40, 26, BLUE, P ? 'center' : 'left');
        break;
      }
      case 'chop': {
        const size = this.chopSize, fam = F.mono(700);
        const cutT = au.timeOfBeat(Math.round(au.beatAt(tm.chopped) * 4) / 4);
        const sepT = au.timeOfBeat(Math.round(au.beatAt(tm.up) * 4) / 4);
        const idT = au.timeOfBeat(Math.round(au.beatAt(tm.into) * 4) / 4);
        const inv = ft >= tm.tokens;
        const h = size * 1.5;
        const sep = snapIn(ft, sepT, 0.08);
        this.chop.forEach((p, k) => {
          const x = lerp(p.cx, p.tx, sep), y = lerp(p.cyy, p.ty, sep);
          const w0 = measure(p.tok, fam, size);
          if (sep > 0) {
            const w = lerp(w0, p.tw, sep);
            c.fillStyle = inv ? INK : PAPER; c.fillRect(x, y - h / 2, w, h);
            c.strokeStyle = INK; c.lineWidth = 5; c.strokeRect(x, y - h / 2, w, h);
          }
          c.font = font(fam, size); c.textBaseline = 'middle';
          const lab = sep > 0.5 ? p.label : p.tok;
          const tx = x + (sep > 0.5 ? size * 0.32 : 0);
          if (lab.startsWith('·')) {
            c.fillStyle = PINK; c.fillText('·', tx, y);
            c.fillStyle = inv ? PAPER : INK; c.fillText(lab.slice(1), tx + measure('·', fam, size), y);
          } else { c.fillStyle = inv ? PAPER : INK; c.fillText(lab, tx, y); }
          // the knife: a cut at every token boundary, one per 32nd from "Chopped"
          if (k > 0 && sep <= 0) {
            const kk = snapIn(ft, cutT + (k - 1) * tm.s16 / 2, 0.05);
            if (kk > 0) { c.fillStyle = PINK; const hh = size * 2.2 * kk; c.fillRect(p.cx - 5, y - hh / 2, 10, hh); }
          }
          // the id, printed under its tile
          const ik = snapIn(ft, idT + k * tm.s16 / 2, 0.06);
          if (ik > 0) this.lab(c, String(p.id), p.tx + p.tw / 2, p.ty + h / 2 + size * 0.5, Math.round(size * 0.46), BLUE, 'center');
        });
        if (ft < sepT) this.lab(c, 'YOUR QUESTION', W / 2, this.chop[0]!.cyy - size * 1.5, 28, BLUE, 'center');
        else this.lab(c, `${this.chop.length} TOKENS`, W / 2, Math.min(...this.chop.map((p) => p.ty)) - size * 1.35, 30, PINK, 'center');
        // the whole prompt the model gets, streaming underneath (special tokens black, the question pink), and
        // under it the same 43 as the ids the model reads, streaming the other way
        const prompt = llm.d.prompt, nq = llm.d.question_tokens.length;
        const ys = H * (P ? 0.64 : 0.62), cs = 28;
        const scroll = drive(au, ft, 0.5, 1.6) * 60;
        const col = (i: number) => {
          const special = prompt[i]!.t.startsWith('<|'), inQ = i >= this.qStart && i < this.qStart + nq;
          return special ? [INK, PAPER] : inQ ? [PINK, PAPER] : [PAPER, INK];
        };
        this.lab(c, `ALL ${prompt.length} TOKENS IT GETS: YOUR ${nq} + ${prompt.length - nq} FROM THE CHAT TEMPLATE`, P ? 60 : 130, ys - cs * 1.7, 24, INK);
        this.ticker(c, prompt.map((p) => showTok(p.t)), ys, cs, scroll, col);
        this.ticker(c, prompt.map((p) => String(p.id)), ys + cs * 2.1, cs, -scroll * 0.8, (i) => { const [bg] = col(i); return [bg === PAPER ? BLUE : bg!, PAPER]; });
        this.lab(c, 'AS IDS', P ? 60 : 130, ys + cs * 3.8, 24, BLUE);
        break;
      }
      case 'belt':
      case 'drop': {
        const s = this.beltStep(ft);
        const fed = Math.min(llm.d.prompt.length, this.n0 + Math.round(s) + 1);
        if (kind === 'belt') {
          const x = P ? 70 : 130, y = P ? H * 0.8 : H * 0.8;
          this.slab(c, `${fed}/${llm.d.prompt.length}`, x, y - 40, P ? 110 : 120, 1, 'left');
          this.lab(c, 'TOKENS FED IN', x + 4, y + 40, 28, INK);
          const cy = P ? H * 0.62 : H * 0.6, lx = P ? 70 : 130;
          const cw = this.chip(c, '<eot_id>', lx, cy, 26, INK, PAPER);
          this.lab(c, 'SPECIAL TOKEN', lx + cw + 16, cy, 26, INK);
          const cw2 = this.chip(c, '·many', lx, cy + 52, 26, PINK, PAPER);
          this.lab(c, 'YOUR QUESTION', lx + cw2 + 16, cy + 52, 26, INK);
        } else {
          const i = Math.min(llm.d.prompt.length - 1, this.n0 + Math.round(s));
          const p = llm.d.prompt[i]!;
          const x = P ? W / 2 : 130, y = P ? H * 0.74 : H * 0.8;
          const al: CanvasTextAlign = P ? 'center' : 'left';
          this.lab(c, `TOKEN ${i + 1} OF ${llm.d.prompt.length}`, x, y - 70, 30, INK, al);
          this.slab(c, `ID ${p.id}`, x, y, P ? 96 : 104, 1, al);
        }
        break;
      }
      case 'flip': {
        const flipped = this.flipTiles.filter((_, k) => ft >= this.flipT(k) + 0.045).length;
        const it = flipped >= 4;
        const y = P ? H * 0.2 : H * 0.2;
        this.slab(c, it ? 'WHAT IT SEES' : 'WHAT YOU SEE', W / 2, y, P ? 92 : 96, snapIn(ft, it ? this.flipT(3) + 0.045 : sh.t0, 0.08), 'center', it ? PINK : INK);
        this.lab(c, it ? 'NUMBERS: ONE ID PER TOKEN' : 'LETTERS', W / 2, y + (P ? 90 : 84), 30, it ? BLUE : INK, 'center');
        break;
      }
      case 'view': {
        const yT = P ? H * 0.16 : H * 0.2, yB = P ? H * 0.76 : H * 0.84;
        this.slab(c, `IT READS ${llm.d.prompt.length} NUMBERS`, W / 2, yT, P ? 76 : 88, snapIn(ft, sh.t0, 0.08));
        const k = snapIn(ft, au.timeOfBeat(Math.round(au.beatAt(tm.chunks) * 2) / 2), 0.08);
        if (k > 0) {
          c.fillStyle = PAPER; c.font = font(F.mono(700), P ? 34 : 38);
          const txt = `CHUNKS IT KNOWS: ${llm.cfg.vocab.toLocaleString('en-US')}`;
          const w = c.measureText(txt).width + 40;
          c.fillRect(W / 2 - w / 2, yB - 34, w, 68);
          c.strokeStyle = INK; c.lineWidth = 4; c.strokeRect(W / 2 - w / 2, yB - 34, w, 68);
          this.lab(c, txt, W / 2, yB, P ? 34 : 38, INK, 'center');
        }
        break;
      }
      case 'letters':
      case 'one': {
        const fam = F.archivo(125, 900), size = this.letterSize;
        const [yTop, yBot] = this.letterY as [number, number], yMid = (yTop + yBot) / 2;
        const rT = [tm.count, tm.the, tm.rs];
        let ri = 0;
        const collapse = kind === 'one' ? ease.inQuad(clamp((ft - sh.t0) / 0.1)) : 0;
        if (kind === 'letters') this.lab(c, 'YOU SEE', W / 2, yTop - size * 0.85, 36, INK, 'center');
        c.font = font(fam, size); c.textBaseline = 'middle'; c.textAlign = 'left';
        this.letters.forEach((g, j) => {
          const isR = g.ch === 'r';
          const rIdx = isR ? ri++ : -1;
          if (collapse >= 1) return;
          const arrive = snapIn(ft, sh.t0 + j * 0.018, 0.07);
          if (kind === 'letters' && arrive <= 0) return;
          const counted = isR && ft >= rT[rIdx]!;
          const gx = lerp(g.x, W / 2 - g.w / 2, collapse), gy = lerp(g.y, yMid, collapse) - (1 - arrive) * 60 * (kind === 'letters' ? 1 : 0);
          c.save();
          c.translate(gx + g.w / 2, gy);
          c.scale(1 - collapse, 1 - collapse);
          if (counted) { c.fillStyle = PINK; c.fillText(g.ch, -g.w / 2, -size * 0.04 * snapIn(ft, rT[rIdx]!, 0.07)); }
          else {
            c.fillStyle = BLUE; c.fillText(g.ch, -g.w / 2 + size * 0.03, size * 0.03);
            c.globalCompositeOperation = 'multiply';
            c.fillStyle = INK; c.fillText(g.ch, -g.w / 2, 0);
          }
          c.restore();
          if (kind === 'letters' && counted) {
            const k = snapIn(ft, rT[rIdx]!, 0.07);
            const r = 34 * (1.4 - 0.4 * k), cx = g.x + g.w / 2, cy = g.y + size * 0.6;
            c.fillStyle = PINK; c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2); c.fill();
            c.fillStyle = PAPER; c.font = font(F.archivo(100, 900), 44); c.textAlign = 'center'; c.textBaseline = 'middle';
            c.fillText(String(rIdx + 1), cx, cy + 2);
            c.font = font(fam, size); c.textAlign = 'left';
          }
        });
        if (kind === 'letters') {
          const nR = [...'strawberry'].filter((ch) => ch === 'r').length;
          this.lab(c, `${'strawberry'.length} LETTERS · ${ft >= tm.rs ? `${nR} R'S` : 'COUNT THE R\'S'}`, W / 2, yBot + size * 1.15, 36, INK, 'center');
        } else {
          const k = snapIn(ft, sh.t0, 0.08);
          const yT = P ? H * 0.22 : H * 0.2, yB = P ? H * 0.74 : H * 0.83;
          this.slab(c, 'IT SEES ONE TOKEN', W / 2, yT, P ? 80 : 96, k);
          const sid = llm.d.prompt[this.strawIdx]?.id ?? 0;
          this.lab(c, `·strawberry = ${sid}`, W / 2, yB, P ? 44 : 48, INK, 'center');
          this.lab(c, 'NO LETTERS. NO R\'S. ONE NUMBER.', W / 2, yB + (P ? 60 : 62), 30, BLUE, 'center');
        }
        break;
      }
      case 'next': {
        const sid = llm.d.prompt[this.strawIdx]?.id ?? 0;
        this.slab(c, `COUNT THE R'S IN ${sid}`, W / 2, P ? H * 0.2 : H * 0.2, P ? 70 : 92, snapIn(ft, sh.t0, 0.08));
        // the empty slot at the end of the row: what comes next
        const n = this.flipTiles.length, last = this.flipTiles[n - 1]!;
        const sx = last.position.x + this.flipW[n - 1]! / 2 + 0.3 + 0.85;
        const cs = [[-0.85, 0.55], [0.85, 0.55], [0.85, -0.55], [-0.85, -0.55]].map(([dx, dy]) => this.project(this.cam2, new THREE.Vector3(sx + dx!, dy!, 0.19)));
        c.strokeStyle = PINK; c.lineWidth = 8; c.setLineDash([22, 14]);
        c.lineDashOffset = -Math.floor(au.beatAt(ft) * 4) * 12; // the dashes march on the 16ths
        c.beginPath(); cs.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y))); c.closePath(); c.stroke();
        c.setLineDash([]); c.lineDashOffset = 0;
        const [mx, my] = this.project(this.cam2, new THREE.Vector3(sx, 0, 0.19));
        const hPx = Math.abs(cs[3]![1] - cs[0]![1]);
        c.fillStyle = blink ? PINK : BLUE; c.font = font(F.archivo(125, 900), hPx * 0.95); c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('?', mx, my + hPx * 0.04); c.textAlign = 'left';
        this.lab(c, 'NEXT TOKEN', mx, cs[0]![1] - 30, 26, PINK, 'center');
        // plant: without the leading space it is three tokens (how the model spells it in its answer)
        const k = snapIn(ft, au.timeOfBeat(Math.round(au.beatAt(tm.watch) * 2) / 2), 0.08);
        if (k > 0 && this.split.length) {
          const cs2 = P ? 30 : 32;
          const y = P ? H * 0.74 : H * 0.8;
          c.font = font(F.mono(700), cs2);
          const lab = `WITHOUT THE SPACE, ${this.split.length} TOKENS:`;
          const chipsW = this.split.reduce((a, s) => a + measure(s.t, F.mono(700), cs2) + cs2 * 0.7 + 16, 0);
          const total = P ? chipsW : c.measureText(lab).width + 24 + chipsW;
          let x = W / 2 - total / 2;
          if (P) this.lab(c, lab, W / 2, y - 64, cs2, INK, 'center');
          else { this.lab(c, lab, x, y, cs2, INK); x += c.measureText(lab).width + 24; }
          for (const s of this.split) {
            const w = this.chip(c, s.t, x, y, cs2, PAPER, INK);
            if (s.id !== undefined) this.lab(c, String(s.id), x + w / 2, y + cs2 * 1.45, 26, BLUE, 'center');
            x += w + 16;
          }
        }
        break;
      }
    }
    void seed;
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio, llm = this.ctx.llm;
    const sh0 = shotAtTimes(f, this.cuts);
    const i = clamp(sh0.i, 0, KINDS.length - 1);
    const kind = KINDS[i]!;
    const sh = { t0: this.cuts[i]!, t1: this.cuts[i + 1] ?? this.ctx.end };
    const seed = Math.floor(hash(this.ctx.start * 13.1, i) * 1000);
    const P = portrait();
    const d = drive(au, f.t, 0.5, 1.6);

    // background: the wash (reseeded per shot) and token rain from the real prompt
    const flat = kind === 'type' || kind === 'chop' || kind === 'letters';
    this.wash.render(r, out, { seed, drift: d, amt: flat ? 0.6 : 0.5, c1: i % 2 ? 'blue' : 'pink', c2: i % 2 ? 'pink' : 'blue' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => tokenRain(this.back.ctx, llm.d.prompt, drive(au, f.ft, 0.5, 1.6) * 0.6, { seed, cols: P ? 7 : 12 })), out);

    // the hero in 3D
    if (this.pose(kind, f.t, this.cam, sh, seed)) {
      r.setRenderTarget(out);
      r.clearDepth();
      r.render(this.scene, this.cam);
    }

    // type, labels, 2D heroes: once per output frame at f.ft
    this.ctx.comp.draw(r, perFrame(this.mid, f, () => {
      if (kind === 'lobby' || kind === 'next') this.pose(kind, f.ft, this.cam2, sh, seed);
      this.type2D(kind, f.ft, sh, seed);
      if (kind === 'lobby' || kind === 'next') this.pose(kind, f.t, this.cam, sh, seed);
    }), out);

    // foreground: ink specks rushing past with the drums
    const c = this.front.ctx;
    this.front.clear();
    specks(c, d * 1.2, { seed: seed + 7, n: flat ? 90 : 70 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    const tm = this.tm;
    const staged = (f.ft >= tm.enter && f.ft < this.cuts[2]!) || (f.ft >= tm.straw && f.ft < this.cuts[10]!);
    return { reg: seed, kinetic: staged ? 0 : 1 };
  }
}
