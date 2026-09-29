// The training clock: which training step the video is showing at song time t. The whole video agrees on it
// (the page of writing, the loss curve, the guess, the weights, the HUD). The intro shows the finished model
// (the last step) writing; on the hook's "ago" the clock rewinds to step 0 (the untrained model), holds through
// verse 1, runs through each chorus's definition to that chorus's checkpoint, holds while the model's sample
// from that checkpoint is sung, runs again through the instrumental before chorus 2 and through the bridge to
// the last step. Anchors are lyric lines, snapped to beats. Between anchors the step moves evenly in
// log(1 + step), so each checkpoint gets its share of screen time.
import type { AudioData } from './engine/audio';
import type { Lyrics } from './engine/lyrics';
import type { Run } from './engine/run';
import { clamp, ease } from './engine/util';

export class Clock {
  /** [song time, step] keys, times increasing. */
  keys: [number, number][] = [];
  /** Checkpoint changes: [song time, checkpoint index], scanned from the keys. */
  events: [number, number][] = [];
  /** When the hook rewinds (the intro's finished model back to step 0). */
  rewind: [number, number] = [0, 0];

  constructor(public run: Run, ly: Lyrics, au: AudioData) {
    const line = (q: string, nth = 0) => { try { return ly.get(q, nth); } catch { return undefined; } };
    const word = (q: string, w: string) => line(q)?.words.find((x) => x.w.toLowerCase().replace(/[^a-z]/g, '').startsWith(w))?.start;
    const snap = (t: number) => au.timeOfBeat(Math.round(au.beatAt(t)));
    const end = run.maxStep;
    const CHORUS_STEPS = run.ex.excerpts.map((e) => e.step);
    const k: [number, number][] = [[0, end]];
    const push = (t: number | undefined, step: number) => { if (t !== undefined && t > k[k.length - 1]![0] + 0.05) k.push([t, step]); };
    const ago = word('minutes ago', 'ago') ?? line('minutes ago')?.start;
    if (ago !== undefined) { push(ago, end); push(ago + 2.4, 0); this.rewind = [ago, ago + 2.4]; }
    for (let n = 0; n < 3; n++) {
      const d0 = line('Guess the next letter', n), d1 = line('And do it again', n);
      const prev = k[k.length - 1]![1];
      if (n === 1) {
        // the instrumental before chorus 2 already runs part of the way (the surprise bars keep shrinking)
        const v2 = line('That surprise');
        if (v2 && d0) { push(snap(v2.end + 1.5), prev); push(d0.start, Math.round(Math.sqrt(Math.max(1, prev) * CHORUS_STEPS[1]!))); }
      }
      if (n === 2) {
        // the bridge runs to the end of training
        const b0 = line('Sixteen thousand'), b1 = line('Five thousand steps');
        if (b0 && b1) { push(snap(b0.start), k[k.length - 1]![1]); push(b1.end, end); }
      }
      if (d0 && d1) { push(snap(d0.start), k[k.length - 1]![1]); push(d1.end, CHORUS_STEPS[n]!); }
    }
    this.keys = k;
    // checkpoint changes, scanned at 240 Hz
    const ev: [number, number][] = [[-1, run.ckptIdx(this.step(0))]];
    for (let t = 0; t < au.duration + 1; t += 1 / 240) {
      const i = run.ckptIdx(this.step(t));
      if (i !== ev[ev.length - 1]![1]) ev.push([t, i]);
    }
    this.events = ev;
  }

  /** The (fractional) training step shown at song time t. */
  step(t: number): number {
    const k = this.keys;
    if (t <= k[0]![0]) return k[0]![1];
    for (let i = 0; i + 1 < k.length; i++) {
      const [ta, sa] = k[i]!, [tb, sb] = k[i + 1]!;
      if (t <= tb) {
        const u = ease.inOutQuad(clamp((t - ta) / Math.max(1e-3, tb - ta)));
        return Math.expm1(Math.log1p(sa) + (Math.log1p(sb) - Math.log1p(sa)) * u);
      }
    }
    return k[k.length - 1]![1];
  }

  /** Checkpoint index shown at t and how long ago (s) it came up. */
  ckpt(t: number): { i: number; since: number; prev: number } {
    const ev = this.events;
    let lo = 0, hi = ev.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ev[m]![0] <= t) lo = m; else hi = m - 1; }
    return { i: ev[lo]![1], since: t - ev[lo]![0], prev: ev[Math.max(0, lo - 1)]![1] };
  }

  /** First song time at or after t0 the clock shows checkpoint i (Infinity if never). */
  reachAfter(i: number, t0 = 0): number {
    for (const [t, j] of this.events) if (t >= t0 && j === i) return t;
    return Infinity;
  }
}
