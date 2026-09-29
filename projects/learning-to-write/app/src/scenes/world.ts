// The whole episode is one continuous scene: a world of printed sheets the camera flies over, with no cuts.
// The page (the model's writing after "ROMEO:") sits in the middle and transforms as the training clock runs;
// the weights, the book, the guess, the surprise and the loss curve sit around it. Each lyric line moves the
// camera to what it is about; labels are written in when sung and left up. See SPEC.md "Shot plan".
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H, clearRT } from '../engine/gl';
import { LIN } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp } from '../engine/util';
import { Lyrics, type Line } from '../engine/lyrics';
import { rect, frame, camMatrix, union, portrait, type Cam, type PanelId, type Rect } from './world-layout';
import { drawPage, drawWeights, drawBook, drawGuess, drawSurprise, drawCurve, drawRing, drawLossGauge, excerptBox, label, sheet,
  writeText, inP, fmt, PINK, INK, BLUE, COLS, NUM } from './world-panels';

/** The character of the unseen line whose guess verse 2 shows: the one the training helped most (the right
 *  character's chance at the last checkpoint minus at step 0), a lowercase letter inside a word, at least 8 in. */
export function guessPos(run: import('../engine/run').Run): number {
  const line = run.p.line, P = run.p.steps, a = P[0]!.chars, b = P[P.length - 1]!.chars;
  if (line.startsWith("Good morrow, neighbour")) return 21; // the "r" of "neighbour": a word it completes (1.8% at step 0, 100.0% at the end)
  let best = 8, score = -1;
  for (let i = 8; i < line.length - 1; i++) {
    if (!/[a-z]/.test(line[i]!) || !/[a-z]/i.test(line[i - 1]!)) continue;
    const g = b[i]!.p - a[i]!.p;
    if (g > score) { score = g; best = i; }
  }
  return best;
}

interface Key { t: number; cam: Cam; dur: number; from?: Cam }

export default class World extends Scene {
  layer = new Layer2D();
  keys: Key[] = [];
  A: Record<string, number> = {};
  def: Line[][] = [];
  ex: Line[][] = [];
  bookText = '';

  override async init() {
    const ly = this.ctx.lyrics, au = this.ctx.audio;
    const L = (q: string, nth = 0): Line | undefined => { try { return ly.get(q, nth); } catch { return undefined; } };
    const w = (l: Line | undefined, word: string) => l?.words.find((x) => x.w.toLowerCase().replace(/[^a-z0-9]/g, '').startsWith(word))?.start ?? l?.start;
    const beat = (t: number) => au.timeOfBeat(Math.round(au.beatAt(t)));
    const A = this.A;
    const hook = L('minutes ago'), model = L('A model is numbers'), ten = L('Ten million'), rnd = L('All of them random');
    const b1 = L('Its only book'), b2 = L('It guesses the next character'), b3 = L('Then it sees the real'), b4 = L('That surprise');
    const g1 = L('Sixteen thousand'), g2 = L('Five thousand steps'), g3 = L('On the plays it practised'), g4 = L('On a play it never read'), g5 = L('learned the lines by heart');
    const o1 = L('on one graphics card'), o2 = L('The big ones learn'), o3 = L('Next time');
    for (let n = 0; n < 3; n++) {
      this.def.push(['Guess the next letter', 'See how wrong', 'Nudge every weight', 'And do it again'].map((q) => L(q, n)!).filter(Boolean));
      // the excerpt: the lines after "And do it again" up to the next section
      const d4 = this.def[n]![3];
      const stops = [b1, g1, o1].map((x) => x?.i ?? 1e9);
      const ex: Line[] = [];
      if (d4) for (let i = d4.i + 1; i < ly.lines.length && !stops.includes(i) && !/^Guess the next/.test(ly.lines[i]!.text); i++) ex.push(ly.lines[i]!);
      this.ex.push(ex);
    }
    const D = (n: number, q: number) => this.def[n]?.[q];
    const exS = (n: number) => this.ex[n]?.[0]?.start ?? (D(n, 3)?.end ?? 0) + 1;
    const exE = (n: number) => this.ex[n]?.[this.ex[n]!.length - 1]?.end ?? exS(n) + 4;
    A.hook = hook?.start ?? 12; A.model = model?.start ?? 16; A.million = w(ten, 'million') ?? 20; A.random = rnd?.start ?? 24;
    A.randomEnd = rnd?.end ?? 27;
    A.c1 = D(0, 0)?.start ?? 40; A.ex1 = exS(0); A.ex1e = exE(0);
    A.book = b1?.start ?? 60; A.guess = b2?.start ?? 64; A.real = w(b3, 'real') ?? 68; A.surprise = b4?.start ?? 72; A.lossW = w(b4, 'loss') ?? 73;
    A.b4e = b4?.end ?? 75;
    A.c2 = D(1, 0)?.start ?? 90; A.ex2 = exS(1); A.ex2e = exE(1);
    A.nudge = g1?.start ?? 110; A.steps = g2?.start ?? 114; A.practised = w(g3, 'six') ?? g3?.start ?? 118; A.unseen = g4?.start ?? 122;
    A.lowest = w(g4, 'five') ?? A.unseen + 2; A.heart = g5?.start ?? A.unseen + 5; A.heartEnd = g5?.end ?? A.heart + 4;
    A.c3 = D(2, 0)?.start ?? 140; A.ex3 = exS(2); A.ex3e = exE(2);
    A.out1 = o1?.start ?? 170; A.out2 = o2?.start ?? 175; A.next = o3?.start ?? 180;
    // the step-0 page types itself out between verse 1 and chorus 1
    // the step-0 page types itself out from "so this is what it wrote" through the instrumental
    A.type0 = w(rnd, 'so') ?? A.randomEnd + 0.3; A.type1 = Math.max(A.type0 + 4, A.c1 - 1.5);
    // the intro: the finished model writes (about 12 characters a second, 3.5 s in)
    A.intro0 = beat(Math.max(2, A.hook - 6)) + 1.0; A.introCps = 16;

    const page = rect('page'), ring = rect('ring');
    const pageTop: Rect = portrait() ? page : { x: page.x, y: page.y - page.h / 2 + 210, w: page.w - 60, h: 420 };
    const titleTop = union(['title']);
    const K: [number, Cam, number][] = [
      [0, frame(titleTop, 1.15), 1],
      [beat(Math.max(2, A.hook - 6)), frame(pageTop, 1.1), 3.5],
      [A.hook - 0.6, frame(page, 1.05), 2.2],
      [A.model - 0.6, frame(rect('weights'), 1.04), 2.4],
      [A.random - 0.6, frame(page, 1.05), 2.4],
      [A.c1 - 1.2, frame(ring, 1.02), 2.6],
      [A.ex1 - 0.8, frame(this.exBox(0), portrait() ? 1.02 : 1.35), 2.0],
      [A.book - 0.8, frame(rect('book'), 1.05), 2.4],
      [A.guess - 0.6, frame(rect('guess'), 1.04), 2.2],
      [A.surprise - 0.6, frame(rect('surprise'), 1.06), 2.0],
      [A.b4e + 0.8, frame(union(['guess', 'surprise']), 1.04), 3.5],
      [A.c2 - 1.2, frame(ring, 1.02), 2.8],
      [A.ex2 - 0.8, frame(this.exBox(1), portrait() ? 1.02 : 1.35), 2.0],
      [A.nudge - 0.8, frame(rect('weights'), 1.04), 2.6],
      [A.steps - 0.6, frame(rect('curve'), 1.04), 2.6],
      [A.heartEnd + 1.0, frame(rect('all'), 1.03), 5.0],
      [A.c3 - 0.4, frame(union(['page', 'ring', 'weights', 'guess']), 1.03), 3.0],
      [A.ex3 - 0.8, frame(this.exBox(2), portrait() ? 1.02 : 1.3), 2.4],
      [A.out1 - 0.8, frame(page, 1.05), 2.4],
      [A.out2 - 0.6, frame(rect('all'), 1.03), 3.5],
      [A.next - 0.6, frame(rect('next'), 1.1), 3.0],
      [A.next + 8, frame(union(['all', 'next']), 1.03), 8.0],
    ];
    K.sort((a, b) => a[0] - b[0]);
    this.keys = K.map(([t, cam, dur]) => ({ t, cam, dur }));
    // each move starts from wherever the previous one had got to
    for (let i = 0; i < this.keys.length; i++) this.keys[i]!.from = i === 0 ? this.keys[0]!.cam : this.camRaw(this.keys[i]!.t, i - 1);
    // a few real lines of the training text for the book
    this.bookText = BOOK;
  }

  /** Characters of the page's current sample shown: the finished model types in the intro, the hook's rewind
   *  erases it, the untrained model types after verse 1; then all of it. */
  typed(t: number): number {
    const A = this.A, [r0, r1] = this.ctx.clock.rewind, n = this.ctx.run.d.samples[this.ctx.run.d.samples.length - 1]!.text.length;
    if (t < A.intro0) return 0;
    if (t < r0) return Math.min(n, (t - A.intro0) * A.introCps);
    if (t < r1) return Math.min(n, (r0 - A.intro0) * A.introCps) * (1 - ease.inQuad(clamp((t - r0) / (r1 - r0))));
    if (t < A.type0) return 0;
    if (t < A.type1) return ((t - A.type0) / (A.type1 - A.type0)) * this.ctx.run.d.samples[0]!.text.length;
    return Infinity;
  }

  private exBox(n: number): Rect {
    const run = this.ctx.run, e = run.ex.excerpts[n]!, idx = run.ckptIdx(e.step);
    const a = e.a, b = e.b;
    return excerptBox(run, rect('page'), idx, a, b);
  }

  /** Camera of move i at time t (no drift). Moves ease in and out with a sine: the speed grows linearly from
   *  rest, so a move never reads as a cut (the gate flags a frame that changes 4x more than the second around it). */
  private camRaw(t: number, i: number): Cam {
    const k = this.keys[i]!;
    const u = 0.5 - 0.5 * Math.cos(Math.PI * clamp((t - k.t) / k.dur));
    const a = k.from ?? k.cam, b = k.cam;
    let z = Math.exp(lerp(Math.log(a.z), Math.log(b.z), u));
    // a long move pulls back on the way (both ends in view at the middle), so the camera flies instead of whipping
    const zmin = Math.min(a.z, b.z);
    const fit = Math.min(W / (Math.abs(b.x - a.x) + W / zmin), H / (Math.abs(b.y - a.y) + H / zmin));
    if (fit < zmin) z *= Math.pow(fit / zmin, Math.sin(Math.PI * u) * 0.85);
    return { x: lerp(a.x, b.x, u), y: lerp(a.y, b.y, u), z };
  }

  cam(t: number): Cam {
    let i = 0;
    while (i + 1 < this.keys.length && this.keys[i + 1]!.t <= t) i++;
    const c = this.camRaw(t, i);
    // always moving: a slow circle of 48 px on screen (about 27 px/s, never at rest) and a gentle zoom breath
    const w = (2 * Math.PI) / 11, z = c.z * (1 + 0.025 * Math.sin((2 * Math.PI * t) / 19));
    return { x: c.x + (48 * Math.cos(w * t)) / z, y: c.y + (48 * Math.sin(w * t)) / z, z };
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const { renderer, comp, run, clock } = this.ctx;
    const t = f.t, A = this.A;
    clearRT(renderer, out, LIN.paper);
    const L = this.layer, c = L.ctx;
    L.clear();
    const cam = this.cam(t);
    c.setTransform(...camMatrix(cam));
    // one number at a time: in the wide shots (the whole world in view) the panels' numbers fade out
    NUM.a = clamp((cam.z - (portrait() ? 0.42 : 0.42)) / 0.12);
    NUM.z = cam.z;
    const step = clock.step(t), ck = clock.ckpt(t);
    const P = run.p.steps, k = ck.i;
    // a new checkpoint blends in from the one before it over 0.6 s (pictures between checkpoints are blends)
    const k0 = ck.prev, uK = ease.inOutQuad(clamp(ck.since / 0.6));

    // title (it goes once the song starts: it is not a teaching label)
    const tr = rect('title');
    c.save();
    c.globalAlpha = 1 - clamp((t - (A.hook + 1)) / 1.5);
    c.textAlign = 'center';
    if (portrait()) {
      c.font = font(F.archivo(112, 900), 128);
      writeText(c, 'WATCH IT', tr.x, tr.y - 90, t, 0.4, { color: INK, cps: 10, fade: 0.5 });
      writeText(c, 'LEARN TO', tr.x, tr.y + 40, t, 1.2, { color: INK, cps: 10, fade: 0.5 });
      writeText(c, 'WRITE', tr.x, tr.y + 170, t, 2.0, { color: INK, cps: 10, fade: 0.5 });
    } else {
      c.font = font(F.archivo(112, 900), 104);
      writeText(c, 'WATCH IT LEARN TO WRITE', tr.x, tr.y + 20, t, 0.4, { color: INK, cps: 14, fade: 0.5 });
    }
    c.font = font(F.mono(600), 22);
    c.letterSpacing = '3px';
    writeText(c, 'ML, SLOWLY · 5 OF 5 · TRAINING', tr.x, tr.y + (portrait() ? 250 : 70), t, 2.4, { color: PINK, cps: 20 });
    c.restore();

    // the page
    const page = rect('page');
    const stagger = (ci: number) => (ci / (COLS * 16)) * 1.1;
    const hiN = [0, 1, 2].find((n) => t >= A[`ex${n + 1}`]! - 0.4 && t < (n < 2 ? A[`c${n + 2}`]! : 1e9));
    let hi;
    if (hiN !== undefined) {
      const ex = this.ex[hiN]!, e = run.ex.excerpts[hiN]!, a = e.a, b = e.b;
      const total = ex.reduce((s, l) => s + l.text.length, 0) || 1;
      const done = ex.reduce((s, l) => s + Lyrics.lineCharProgress(l, t), 0);
      hi = { a, b, sung: done / total, step: e.step, t0: A[`ex${hiN + 1}`]! - 0.4 };
    }
    drawPage(c, run, page, {
      ckptAt: (ci) => { const tc = t - stagger(ci); const q = clock.ckpt(tc); return { k: q.i, from: q.prev, u: clamp(q.since / 0.35) }; },
      typed: this.typed(t), hi, cursorOn: t < A.type1 + 0.3 && Math.floor(t * 1.6) % 2 === 0, stepLabel: `STEP ${fmt(step)}`, t,
    });

    // the ring (the chorus's picture): drawn in on chorus 1, left up
    const hot = this.def.findIndex(() => false);
    let hotQ = -1;
    for (const d of this.def) d.forEach((l, q) => { if (t >= l.start - 0.1 && t < l.end + 0.2) hotQ = q; });
    void hot;
    const bead = (Math.log1p(step) / Math.log1p(run.maxStep)) * 7 + t * 0.012;
    drawRing(c, rect('ring'), t, this.def[0]?.map((l) => l.start) ?? [1e9, 1e9, 1e9, 1e9], hotQ, bead, inP(t, A.c1 - 1.0, 1.0));

    // chorus 2 adds a layer: the loss under the page, falling as the clock runs
    drawLossGauge(c, run, page, t, step, A.c2 - 1.0);
    // the weights
    drawWeights(c, run, rect('weights'), t, k0, k, uK, { t0: 0, countT0: A.million - 0.3, oneT0: A.nudge + 0.8, oneHot: t >= A.nudge && t < A.steps });
    // the book
    drawBook(c, run, rect('book'), t, A.book, this.bookText);
    // the guess and the surprise: the checkpoint the clock is at, blended toward the next
    const Pk = P[k0]!, Pn = P[k]!;
    const gp = guessPos(run);
    drawGuess(c, run, rect('guess'), t, Pk, Pn, uK, gp, { t0: A.guess - 0.2, answerT0: A.real, pT0: A.real + 0.6 });
    drawSurprise(c, run, rect('surprise'), t, Pk, Pn, uK, gp, { t0: A.surprise - 0.2, allT0: A.lossW + 0.8, meanT0: A.b4e + 1.2 });
    // the curve
    drawCurve(c, run, rect('curve'), t, step, { t0: A.nudge + 1.5, trainT0: A.practised, valT0: A.unseen, minT0: A.lowest, memT0: A.heart });

    // the next episode
    const nx = rect('next');
    if (t >= A.out2) {
      sheet(c, nx, inP(t, A.out2, 0.8));
      label(c, nx.x - nx.w / 2 + 60, nx.y - 60, 'NEXT TIME, ML, SLOWLY #6', 'NEXT TOKEN', t, A.next, { size: portrait() ? 110 : 140, color: PINK, cps: 16 });
      const lines = portrait()
        ? ['Llama 3.1 8B: the same guessing game,', 'far more text. Watch it write.', '', `this one: ${fmt(run.d.n_params)} weights,`, `${(run.d.wall_s / 60).toFixed(1)} minutes on one RTX 3090`]
        : ['Llama 3.1 8B: the same guessing game, far more text. Watch it write.', '', `this one: ${fmt(run.d.n_params)} weights, ${(run.d.wall_s / 60).toFixed(1)} minutes on one RTX 3090`];
      c.font = font(F.mono(500), portrait() ? 30 : 24);
      lines.forEach((ln, i) => writeText(c, ln, nx.x - nx.w / 2 + 60, nx.y + 40 + i * (portrait() ? 44 : 36), t, A.next + 1.2 + i * 0.7, { color: i >= lines.indexOf('') ? BLUE : INK, cps: 40 }));
    }
    comp.draw(renderer, L.upload(), out);
    const end = this.ctx.audio.duration;
    return { reg: 0, kinetic: 0, fade: clamp((t - (end - 1.2)) / 1.0) };
  }
}

// The first lines of the training text (tiny Shakespeare opens with Coriolanus), shown in the book.
const BOOK = `First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?`;
