// logits (verse 3, lines 1-4): "At the top of the tower every word gets a score, / A hundred twenty-
// eight thousand, all in the draw, / Softmax turns them into odds with the temperature set, / Keep it
// low and it's safe, turn it up, it's a bet". The step in flight is the story's (step 4, the token after
// "There are 2"). The roof of the tower is a scoreboard: the vocabulary is a ribbon of 128,256 slots
// (llm.cfg.vocab) in rank order, the real top 10 at T=1 (top10_t1) stand up as bars, the rest are
// ticks. Scores are ln p (log-probabilities: the logits up to one shared constant), computed from
// the top 10; softmax squashes them into p; the knob re-weights the top 10 by temperature
// (p_T ∝ p^(1/T), labelled COMPUTED). A new shot every bar, cut on the bar grid from the line's first
// word (two shots per sung line).
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { clamp, ease, lerp } from '../engine/util';
import { showTok } from '../engine/llm';
import { F, font } from '../engine/type';
import { Wash, tokenRain, specks, makeCam, drive, shot, snapIn, portrait, perFrame, inked, whip } from './_kit';
import { Tower, riderTile } from './_tower';
import { ribbonMat, stripGeo, spiralGeo, Knob, tempProbs, slab, label, proj, screenToPlane, fmtInt, pct, C } from './logits-parts';

const SP = 0.6; // bar pitch on the podium
const BW = 0.4; // bar width
const BH = 4.0; // bar height for p = 1 (and for the top of the score scale)
const TW = 0.03; // width of one of the other 128,246 slots
const SH = 1.0; // ribbon height
const LP_FLOOR = -12; // bottom of the score scale (ln p)
const KCAM_Z = 12, KCAM_FOV = 30;

type Cam = { pos: THREE.Vector3; look: THREE.Vector3; fov: number; up?: THREE.Vector3 };

export default class Logits extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(36);
  kscene = new THREE.Scene();
  kcam = makeCam(KCAM_FOV);
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  front = new Layer2D();
  tower = new Tower();
  knob = new Knob();
  bars: THREE.Mesh[] = [];
  ribbon!: THREE.Mesh;
  spiral!: THREE.Mesh;
  k = 4;
  top: { t: string; p: number; lp: number }[] = [];
  rest = 0;
  vocab = 128256;
  RY = 0;
  xEnd = 0;
  pool: { id: number; t: string }[] = [];
  tm = { l1: 0, every: 0, score: 0, l2: 0, thousand: 0, draw: 0, l3: 0, turns: 0, odds: 0, temperature: 0, set: 0, l4: 0, low: 0, safe: 0, turn: 0, up: 0, bet: 0 };

  override init() {
    const { lyrics: ly, llm, story } = this.ctx;
    const word = (line: string, w: string) => {
      const l = ly.get(line);
      return (l.words.find((x) => x.w.toLowerCase().replace(/[^a-z0-9']/g, '').startsWith(w)) ?? l.words[0]!).start;
    };
    const L1 = 'At the top of the tower', L2 = 'A hundred twenty-eight thousand', L3 = 'Softmax turns them', L4 = 'Keep it low';
    this.tm = {
      l1: ly.get(L1).start, every: word(L1, 'every'), score: word(L1, 'score'),
      l2: ly.get(L2).start, thousand: word(L2, 'thousand'), draw: word(L2, 'draw'),
      l3: ly.get(L3).start, turns: word(L3, 'turns'), odds: word(L3, 'odds'), temperature: word(L3, 'temperature'), set: word(L3, 'set'),
      l4: ly.get(L4).start, low: word(L4, 'low'), safe: word(L4, 'safe'), turn: word(L4, 'turn'), up: word(L4, 'up'), bet: word(L4, 'bet'),
    };
    // the step in flight while this verse is sung
    this.k = story.at((this.tm.l1 + this.tm.bet) / 2).step;
    const st = llm.steps[this.k]!;
    this.top = st.top10_t1.map(([t, p]) => ({ t, p, lp: Math.log(Math.max(p, 1e-12)) }));
    this.rest = 1 - this.top.reduce((a, b) => a + b.p, 0);
    this.vocab = llm.cfg.vocab;
    this.pool = [...llm.d.prompt, ...llm.steps.map((s) => s.tok)];

    this.scene.add(this.tower.group);
    this.tower.outlineAll();
    this.tower.ride(riderTile('?'), 0);
    this.RY = this.tower.roof.position.y + 0.15;
    (this.tower.roof.material as THREE.ShaderMaterial).uniforms.litA!.value = 0.08;

    // the podium: the real top 10, one bar each (the leader in pink)
    const geo = new THREE.BoxGeometry(BW, 1, BW).translate(0, 0.5, 0);
    for (let r = 0; r < 10; r++) {
      const m = inked(geo, r === 0 ? { ink: 'pink', lit: 0.55, shade: 1, over: 'blue', overShade: 0.45 } : { ink: 'blue', lit: 0.62, shade: 1 }, 2.2);
      m.position.set(this.xb(r), this.RY, 0.2);
      this.scene.add(m);
      this.bars.push(m);
    }
    // the ribbon: 10 wide podium slots behind the bars, then the other 128,246 thin ones
    this.xEnd = 3 + (this.vocab - 10) * TW;
    this.ribbon = new THREE.Mesh(stripGeo([{ x0: -3, x1: 3, s0: 0, s1: 10 }, { x0: 3, x1: this.xEnd, s0: 10, s1: this.vocab }], this.RY, SH, -0.55), ribbonMat({ nSlots: this.vocab }));
    this.ribbon.frustumCulled = false;
    this.scene.add(this.ribbon);
    // the draw: the same 128,246 slots coiled round the roof (top-down shot)
    this.spiral = new THREE.Mesh(spiralGeo(4.6, 0.95, 8.5, 0.72, 10, this.vocab), ribbonMat({ nSlots: this.vocab, tickH: 0.5, band: 0.18, markEvery: 10000, fogNear: 300, fogFar: 400 }));
    this.spiral.position.y = this.RY + 0.02;
    this.spiral.frustumCulled = false;
    this.scene.add(this.spiral);

    this.kscene.add(this.knob.group);
    this.cam.far = 260;
    this.cam.updateProjectionMatrix();
  }

  private xb(r: number) { return (r - 4.5) * SP; }

  /** Temperature at song time t: 1 until "set" (0.6, the real one), "low" (0.2), "turn it up" (1.5). */
  private temp(t: number) {
    const tm = this.tm;
    let T = 1;
    if (t >= tm.set) T = lerp(1, 0.6, ease.outExpo(clamp((t - tm.set) / 0.16)));
    if (t >= tm.low) T = lerp(0.6, 0.2, ease.outExpo(clamp((t - tm.low) / 0.16)));
    if (t >= tm.turn) T = lerp(0.2, 1.5, ease.inOutCubic(clamp((t - tm.turn) / Math.max(0.2, tm.up - tm.turn))));
    return T;
  }

  /** Bar heights (world units) and the probabilities shown, at song time t. */
  private barsAt(t: number) {
    const au = this.ctx.audio;
    const sixteenth = 60 / au.bpm / 4;
    const sq = snapIn(t, this.tm.turns, 0.12);
    const T = this.temp(t);
    const pT = tempProbs(this.top.map((x) => x.p), T);
    const h = this.top.map((x, r) => {
      const rise = snapIn(t, this.tm.every + r * sixteenth, 0.08);
      const score = BH * clamp((x.lp - LP_FLOOR) / -LP_FLOOR, 0.05, 1);
      const prob = BH * pT[r]!;
      return Math.max(0.035, lerp(score, prob, sq) * rise);
    });
    return { h, T, pT, sq };
  }

  /** The camera for shot `kind` at song time t. */
  private camAt(kind: number, t: number, t0: number): Cam {
    const au = this.ctx.audio, RY = this.RY, P = portrait();
    const d = drive(au, t, 0.5, 1.6), dd = d - drive(au, t0, 0.5, 1.6);
    const dist = P ? 1.55 : 1;
    switch (kind) {
      case 0: {
        // up the tower to the roof: a whip on the downbeat, from the token in flight to the scoreboard
        const climb = this.ctx.story.at(t).climb;
        const yr = this.tower.floorY(climb * (this.tower.layers + 1));
        const db = au.downbeats.find((x) => x >= t0 - 0.01) ?? t0;
        const w = whip(au, t, db, 0.22);
        const a = 0.9 + d * 0.04, rad = 11 * dist;
        const pre = new THREE.Vector3(Math.cos(a) * rad, yr + 0.9, Math.sin(a) * rad);
        const post = P ? new THREE.Vector3(-4.5 + dd * 0.2, RY + 3.2, 13) : new THREE.Vector3(-4.6 + dd * 0.25, RY + 1.7, 6.0);
        const lookPost = P ? new THREE.Vector3(1.2 + dd * 0.2, RY + 0.4, -0.5) : new THREE.Vector3(2.4 + dd * 0.25, RY + 0.55, -0.5);
        return { pos: pre.lerp(post, w), look: new THREE.Vector3(0, yr + 0.25, 0).lerp(lookPost, w), fov: 38 };
      }
      case 1: {
        const s = dd * 0.28;
        return P
          ? { pos: new THREE.Vector3(8.5 - s * 0.4, RY + 2.4, 16), look: new THREE.Vector3(0.2 - s * 0.4, RY + 1.7, 0), fov: 40 }
          : { pos: new THREE.Vector3(4.6 - s, RY + 1.2, 8.6), look: new THREE.Vector3(-0.2 - s, RY + 1.85, 0), fov: 40 };
      }
      case 2: {
        // the rush: a speed ramp along the ribbon, landing on the last slot on "thousand"
        const x = this.xOfSlot(this.slotAt(t, t0));
        const sway = Math.sin(d * 0.6) * 0.25;
        return P
          ? { pos: new THREE.Vector3(x - 1.4, RY + 1.35, 4.3 + sway), look: new THREE.Vector3(x + 2.6, RY + 0.45, -0.55), fov: 46 }
          : { pos: new THREE.Vector3(x - 2.3, RY + 0.95, 2.8 + sway), look: new THREE.Vector3(x + 3.4, RY + 0.4, -0.55), fov: 44 };
      }
      case 3: {
        // straight down on the roof: the whole vocabulary coiled round it
        const a = d * 0.12;
        return { pos: new THREE.Vector3(0.001, RY + (P ? 30 : 25), 0), look: new THREE.Vector3(0, RY, 0), fov: 42, up: new THREE.Vector3(Math.cos(a), 0, Math.sin(a)) };
      }
      case 4: {
        const a = -0.6 + dd * 0.05, r = P ? 15.5 : 7.6;
        return { pos: new THREE.Vector3(Math.sin(a) * r, RY + (P ? 2.8 : 1.7), Math.cos(a) * r), look: new THREE.Vector3(P ? -0.6 : 0.5, RY + (P ? 1.3 : 1.0), 0), fov: 38 };
      }
      case 5: {
        const s = dd * 0.2;
        return P
          ? { pos: new THREE.Vector3(7.5 - s * 0.4, RY + 2.4, 17), look: new THREE.Vector3(0.2 - s * 0.4, RY - 0.4, 0), fov: 36 }
          : { pos: new THREE.Vector3(1.4 - s, RY + 1.5, 9.6), look: new THREE.Vector3(0.7 - s, RY + 1.2, 0), fov: 36 };
      }
      case 6: {
        // wide: the tower's top and the ribbon running off, the knob in front
        const a = 1.05 + d * 0.06, r = 26 * dist;
        return { pos: new THREE.Vector3(Math.cos(a) * r, RY + 7, Math.sin(a) * r), look: new THREE.Vector3(P ? 1 : -4, RY - 2.5, 0), fov: 34 };
      }
      default: {
        const s = dd * 0.25;
        return P
          ? { pos: new THREE.Vector3(7.5 + s * 0.4, RY + 2.2, 17), look: new THREE.Vector3(0.2 + s * 0.4, RY + 0.2, 0), fov: 36 }
          : { pos: new THREE.Vector3(-0.4 + s, RY + 1.1, 6.8), look: new THREE.Vector3(0.6 + s, RY + 0.8, 0), fov: 34 };
      }
    }
  }

  /** The slot the rush has reached: geometric from slot 10 to the last one, landing on "thousand". */
  private slotAt(t: number, t0: number) {
    const v = clamp((t - t0 - 0.08) / Math.max(0.3, this.tm.thousand - t0 - 0.08));
    return 10 * Math.pow(this.vocab / 10, v);
  }
  private xOfSlot(s: number) { return 3 + (s - 10) * TW; }

  private applyCam(c: Cam) {
    const cam = this.cam;
    cam.position.copy(c.pos);
    cam.up.copy(c.up ?? new THREE.Vector3(0, 1, 0));
    cam.lookAt(c.look);
    if (cam.fov !== c.fov) { cam.fov = c.fov; cam.updateProjectionMatrix(); }
    cam.updateMatrixWorld();
  }

  /** Where the knob sits for this shot (screen px, radius px), or null. */
  private knobAt(kind: number, t: number): { x: number; y: number; r: number } | null {
    const P = portrait();
    if (kind === 5) {
      const k = snapIn(t, this.tm.temperature, 0.09);
      if (k <= 0) return null;
      return P ? { x: W * 0.5, y: H * 0.68, r: 230 * (0.7 + 0.3 * k) } : { x: W * 0.77, y: H * 0.56, r: 235 * (0.7 + 0.3 * k) };
    }
    if (kind === 6) return P ? { x: W * 0.5, y: H * 0.42, r: 290 } : { x: W * 0.27, y: H * 0.6, r: 285 };
    if (kind === 7) return P ? { x: W * 0.5, y: H * 0.72, r: 190 } : { x: W * 0.82, y: H * 0.47, r: 185 };
    return null;
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio, llm = this.ctx.llm;
    const sh = shot(this.ctx, f, { every: 4, from: this.tm.l1 });
    const kind = Math.min(sh.i, 7);
    const P = portrait();
    const d = drive(au, f.t, 0.6, 1.8);

    // background: wash + rain of real tokens
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: kind === 2 ? 0.35 : kind === 3 ? 0.7 : 0.5, c1: kind % 2 ? 'blue' : 'pink', c2: kind % 2 ? 'pink' : 'blue' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => tokenRain(this.back.ctx, this.pool, drive(au, f.ft, 0.6, 1.8) * 0.5, { seed: sh.seed, cols: P ? 8 : 13 })), out);

    // 3D: tower, podium, ribbon / spiral
    const bs = this.barsAt(f.t);
    this.bars.forEach((b, i) => { b.scale.y = bs.h[i]!; });
    const story = this.ctx.story.at(f.t);
    this.tower.ride(this.tower.rider.children[0]!, story.climb);
    this.tower.light(kind === 0 ? story.climb : 1, kind === 0 ? 1 : 0.6);
    this.spiral.visible = kind === 3;
    this.ribbon.visible = kind !== 3;
    this.spiral.rotation.y = -d * 0.05;
    const rm = this.ribbon.material as THREE.ShaderMaterial;
    rm.uniforms.tickH!.value = lerp(0.16, 0.03, bs.sq);
    this.applyCam(this.camAt(kind, f.t, sh.t0));
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);

    // the knob (its own camera, composited over the 3D)
    const kn = this.knobAt(kind, f.t);
    if (kn) {
      this.knob.set(bs.T);
      const ppu = (H / 2) / Math.tan((KCAM_FOV * Math.PI) / 360) / KCAM_Z;
      const [kx, ky] = screenToPlane(kn.x, kn.y, KCAM_Z, KCAM_FOV);
      this.knob.group.position.set(kx, ky, 0);
      this.knob.group.scale.setScalar(kn.r / (1.7 * ppu));
      this.knob.group.rotation.set(-0.22 + Math.sin(d * 0.4) * 0.04, (kn.x < W / 2 ? 0.28 : -0.28) + Math.sin(d * 0.3) * 0.05, 0);
      this.kcam.position.set(0, 0, KCAM_Z);
      this.kcam.lookAt(0, 0, 0);
      r.setRenderTarget(out);
      r.clearDepth();
      r.render(this.kscene, this.kcam);
    }

    // type: labels, counters, charts (crisp, once per frame at f.ft)
    this.ctx.comp.draw(r, perFrame(this.mid, f, () => this.drawType(kind, f.ft, sh)), out);

    // foreground specks
    const c = this.front.ctx;
    this.front.clear();
    specks(c, d * (kind === 2 ? 2.4 : 1), { seed: sh.seed, n: kind === 2 ? 120 : 75 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    void llm;
    // the rush stages "128,256" itself (bigger than the slam): no slam over it or right after it
    const staged = kind === 2 || (f.ft >= this.tm.thousand && f.ft < this.tm.thousand + 0.7);
    return { reg: sh.seed, kinetic: staged ? 0 : 1 };
  }

  private drawType(kind: number, t: number, sh: { t0: number; t1: number }) {
    const c = this.mid.ctx, P = portrait(), RY = this.RY, tm = this.tm;
    this.applyCam(this.camAt(kind, t, sh.t0));
    const bs = this.barsAt(t);
    const bottomY = H - (P ? 420 : 160);

    // labels on the 3D bars: the token under each, its score or probability over it
    const barLabels = (n: number, mode: 'score' | 'p', size = 26) => {
      for (let i = 0; i < n; i++) {
        if (bs.h[i]! < 0.05) continue;
        const foot = proj(this.cam, new THREE.Vector3(this.xb(i), RY - 0.08, 0.45));
        const head = proj(this.cam, new THREE.Vector3(this.xb(i), RY + bs.h[i]! + 0.1, 0.2));
        if (!foot || !head) continue;
        label(c, showTok(this.top[i]!.t), foot[0], foot[1] + size * 0.9, { size, align: 'center', box: i === 0 ? C.INK : C.PAPER, color: i === 0 ? C.PAPER : C.INK });
        const v = mode === 'score' ? this.top[i]!.lp.toFixed(2).replace('-', '−') : pct(bs.pT[i]!, bs.pT[i]! < 0.1 ? 1 : 0);
        label(c, v, head[0], head[1] - size * 0.7, { size: size - 2, align: 'center', box: null });
      }
    };

    switch (kind) {
      case 0: {
        const st = this.ctx.story.at(t);
        const db = this.ctx.audio.downbeats.find((x) => x >= sh.t0 - 0.01) ?? sh.t0;
        if (t < db) {
          const p = proj(this.cam, new THREE.Vector3(0.9, this.tower.floorY(st.climb * (this.tower.layers + 1)) + 0.3, 0));
          if (p) label(c, `? · THE NEXT TOKEN · CLIMBING L${st.layer}`, clamp(p[0] + 60, 120, W - 700), p[1], { size: 28, box: C.INK, color: C.PAPER });
        } else {
          const p = proj(this.cam, new THREE.Vector3(-2.9, RY + 1.5, 0));
          const k = snapIn(t, db + 0.12);
          if (p && k > 0) {
            // ten empty slots waiting for their scores
            for (let i = 0; i < 10; i++) {
              const q = proj(this.cam, new THREE.Vector3(this.xb(i), RY + 0.35, 0.2));
              if (q) label(c, '?', q[0], q[1], { size: 30, align: 'center', box: null, color: C.INK });
            }
            label(c, 'THE ROOF: OUTPUT HEAD', clamp(p[0], 120, W - 520), p[1] - 30, { size: 30, box: C.INK, color: C.PAPER });
            label(c, `SCORES ALL ${fmtInt(this.vocab)} TOKENS`, clamp(p[0], 120, W - 520), p[1] + 22, { size: 26, box: C.PAPER });
          }
        }
        break;
      }
      case 1: {
        barLabels(P ? 5 : 6, 'score', P ? 28 : 26);
        label(c, 'SCORE = ln p · COMPUTED FROM THE TOP 10 AT T 1', W / 2, bottomY, { size: P ? 24 : 26, align: 'center', box: C.PAPER });
        break;
      }
      case 2: {
        // the count, staged as the slam: every slot passed, up to the whole vocabulary
        const s = this.slotAt(t, sh.t0);
        const n = Math.min(this.vocab, Math.round(s));
        const landed = t >= tm.thousand;
        const y = P ? H * 0.6 : H * 0.6;
        slab(c, fmtInt(n), W / 2, y, P ? 190 : 280, landed ? snapIn(t, tm.thousand, 0.08) : 1, 'center', W * 0.9);
        label(c, landed ? 'TOKENS IN THE VOCABULARY · EVERY ONE SCORED' : 'SLOTS PASSED', W / 2, y + (P ? 150 : 200), { size: P ? 26 : 30, align: 'center', box: C.PAPER });
        break;
      }
      case 3: {
        const top = this.top.reduce((a, b) => a + b.p, 0);
        const y0 = P ? H * 0.72 : H * 0.7;
        label(c, `TOP 10 · ${pct(top, 2)}`, W / 2, y0, { size: P ? 40 : 44, align: 'center', box: C.INK, color: C.PAPER });
        label(c, `THE OTHER ${fmtInt(this.vocab - 10)} · ${pct(this.rest, 2)}`, W / 2, y0 + (P ? 64 : 70), { size: P ? 32 : 36, align: 'center', box: C.PAPER, color: C.BLUE });
        label(c, 'P AT T 1 · COMPUTED', W / 2, y0 + (P ? 116 : 124), { size: 24, align: 'center', box: C.PAPER });
        break;
      }
      case 4: {
        const mode = bs.sq > 0.5 ? 'p' : 'score';
        barLabels(P ? 4 : 5, mode, P ? 28 : 26);
        label(c, 'SOFTMAX:  p = e^score ÷ Σ e^score', W / 2, P ? H * 0.2 : H * 0.215, { size: P ? 26 : 30, align: 'center', box: C.PAPER });
        const k = snapIn(t, tm.odds, 0.09);
        if (k > 0) this.sumBar(c, P ? W * 0.07 : W * 0.2, bottomY - (P ? 10 : 34), P ? W * 0.86 : W * 0.6, 54, k);
        break;
      }
      case 5: {
        barLabels(P ? 4 : 5, 'p', P ? 28 : 26);
        const kn = this.knobAt(kind, t);
        if (kn) this.tempReadout(c, kn, bs.T, 'THE REAL SETTING');
        const nuc = this.ctx.llm.steps[this.k]!.nucleus;
        label(c, `THE REAL DRAW (TOP-P 0.9): ${nuc.slice(0, 3).map(([tk, p]) => `${showTok(tk)} ${pct(p, 0)}`).join(' · ')}`, P ? W / 2 : W * 0.4, P ? H * 0.19 : bottomY, { size: P ? 24 : 26, align: 'center', box: C.PAPER });
        label(c, 'TOP 10 · COMPUTED', P ? W / 2 : W * 0.4, P ? H * 0.235 : H * 0.22, { size: 24, align: 'center', box: C.INK, color: C.PAPER });
        break;
      }
      case 6: {
        const kn = this.knobAt(kind, t)!;
        this.tempReadout(c, kn, bs.T, t >= tm.low ? 'ONE WINNER · SAFE' : '');
        const x0 = P ? W * 0.1 : W * 0.54, y0 = P ? H * 0.62 : H * 0.36, bw = P ? W * 0.56 : W * 0.3;
        this.chart(c, x0, y0, bw, bs.pT, P ? 58 : 66);
        label(c, 'TOP 10 · COMPUTED', x0, y0 - 52, { size: 24, box: C.INK, color: C.PAPER });
        break;
      }
      default: {
        barLabels(5, 'p', P ? 28 : 26);
        const kn = this.knobAt(kind, t)!;
        this.tempReadout(c, kn, bs.T, t >= tm.up ? 'FLAT · A BET' : '');
        label(c, 'TOP 10 · COMPUTED', P ? W / 2 : W * 0.4, P ? H * 0.24 : H * 0.25, { size: 24, align: 'center', box: C.INK, color: C.PAPER });
        break;
      }
    }
  }

  /** "T = 0.6" above the knob, and a short verdict under it. */
  private tempReadout(c: CanvasRenderingContext2D, kn: { x: number; y: number; r: number }, T: number, sub: string) {
    const P = portrait();
    const x = kn.x, y = kn.y - kn.r - (sub ? 118 : 70);
    const txt = `T = ${T.toFixed(1)}`;
    c.save();
    c.font = font(F.archivo(125, 900), P ? 96 : 100);
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillStyle = C.BLUE;
    c.fillText(txt, x + 4, y + 3);
    c.globalCompositeOperation = 'multiply';
    c.fillStyle = C.PINK;
    c.fillText(txt, x, y);
    c.restore();
    if (sub) label(c, sub, x, y + 74, { size: 28, align: 'center', box: C.INK, color: C.PAPER });
  }

  /** Horizontal bars of the top 5 (2D, big type). */
  private chart(c: CanvasRenderingContext2D, x0: number, y0: number, bw: number, pT: number[], rowH: number) {
    for (let i = 0; i < 5; i++) {
      const y = y0 + i * rowH;
      const tk = showTok(this.top[i]!.t);
      label(c, tk, x0, y, { size: 32, color: i === 0 ? C.PAPER : C.INK, box: i === 0 ? C.INK : C.PAPER });
      const bx = x0 + 150;
      c.fillStyle = C.INK;
      c.fillRect(bx, y + rowH * 0.28, bw, 3);
      c.fillStyle = i === 0 ? C.PINK : C.BLUE;
      c.fillRect(bx, y - rowH * 0.3, Math.max(4, bw * pT[i]!), rowH * 0.56);
      label(c, pT[i]! < 0.001 ? '<0.1%' : pct(pT[i]!, pT[i]! < 0.1 ? 1 : 0), bx + Math.max(4, bw * pT[i]!) + 14, y, { size: 30, box: C.PAPER });
    }
  }

  /** The probabilities stacked end to end: they sum to 1. */
  private sumBar(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, k: number) {
    let acc = 0;
    const segs = [...this.top.slice(0, 4).map((s) => ({ t: showTok(s.t), p: s.p })), { t: 'REST', p: 1 - this.top.slice(0, 4).reduce((a, b) => a + b.p, 0) }];
    const ww = w * ease.outExpo(k);
    segs.forEach((s, i) => {
      const sx = x + acc * ww, sw = s.p * ww;
      c.fillStyle = i === 0 ? C.PINK : i === segs.length - 1 ? C.INK : i % 2 ? C.BLUE : C.INK;
      c.fillRect(sx, y - h / 2, Math.max(2, sw - 3), h);
      if (sw > 120) label(c, `${s.t} ${pct(s.p, 0)}`, sx + 12, y, { size: 24, color: C.PAPER, box: null });
      acc += s.p;
    });
    label(c, 'Σ = 100%', x + ww + 20, y, { size: 30, box: C.INK, color: C.PAPER });
  }
}
