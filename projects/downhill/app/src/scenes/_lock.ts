// Beat-locked motion for v4 ("snappy and locky"): every change starts on the 8th-note grid, snaps in
// fast (outExpo, <= ~90 ms) and then HOLDS until the next change. No continuous drift. Use these
// instead of free-running tweens; keep ambient life to non-spatial texture (grain, flicker).
import type { AudioData } from '../engine/audio';
import { clamp, ease } from '../engine/util';

/** Nearest grid time to t, grid = beat / div (div 2 = 8ths, 4 = 16ths). */
export function q(au: AudioData, t: number, div = 2): number {
  const b = au.beatAt(t);
  return au.timeOfBeat(Math.round(b * div) / div);
}

/** 0 -> 1 snap that starts on the grid point nearest `at` and completes in `dur` seconds, then holds at 1. */
export function snap(au: AudioData, t: number, at: number, dur = 0.09, div = 2): number {
  const t0 = q(au, at, div);
  return t < t0 ? 0 : ease.outExpo(clamp((t - t0) / dur));
}

/** Snap a value through a list of [time, value] keys: each key lands on its grid point and holds. */
export function snapKeys(au: AudioData, t: number, keys: [number, number][], dur = 0.09, div = 2): number {
  let v = keys[0]?.[1] ?? 0;
  for (let i = 1; i < keys.length; i++) {
    const [at, to] = keys[i]!;
    const k = snap(au, t, at, dur, div);
    if (k <= 0) break;
    v = v + (to - v) * k;
  }
  return v;
}

/** Whole grid steps elapsed since `t0` (beat / div units), each step snapping in over `frac` of a step. */
export function steps(au: AudioData, t: number, t0: number, div = 1, frac = 0.18): number {
  const b0 = Math.round(au.beatAt(t0) * div), b = au.beatAt(t) * div - b0;
  if (b < 0) return 0;
  const k = Math.floor(b);
  return k + ease.outExpo(clamp((b - k) / frac));
}

/** Impact envelope for a hit on the grid point nearest `at` (1 at the hit, halving every `hl` s). */
export function hit(au: AudioData, t: number, at: number, hl = 0.08, div = 2): number {
  const t0 = q(au, at, div);
  return t < t0 ? 0 : Math.pow(0.5, (t - t0) / hl);
}

/** The downbeat at or before t (for cuts and scene-internal bar structure). */
export function barStart(au: AudioData, t: number): number {
  const d = au.downbeats;
  let s = d[0] ?? 0;
  for (const x of d) if (x <= t + 1e-6) s = x; else break;
  return s;
}
