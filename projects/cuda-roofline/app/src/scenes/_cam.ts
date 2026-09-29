// Beat-locked 3D camera helpers (v5): the camera is always moving, its speed breathes with the drums,
// and its big moves land on downbeats. Everything is a pure function of song time (no state), so
// motion blur, seeking and parallel segment renders stay exact.
import * as THREE from 'three';
import type { AudioData } from '../engine/audio';
import { clamp, ease } from '../engine/util';

const FPS = 100;
const cumCache = new WeakMap<AudioData, Map<string, Float32Array>>();

/** Integral of an envelope from 0 to t (seconds × envelope), from prefix sums at 100 fps. */
export function cumEnv(au: AudioData, name: string, t: number): number {
  let m = cumCache.get(au);
  if (!m) { m = new Map(); cumCache.set(au, m); }
  let c = m.get(name);
  if (!c) {
    const n = Math.ceil(au.duration * FPS) + 2;
    c = new Float32Array(n);
    for (let i = 1; i < n; i++) c[i] = c[i - 1]! + au.env(name, (i - 0.5) / FPS) / FPS;
    m.set(name, c);
  }
  const x = clamp(t * FPS, 0, c.length - 1.001);
  const i = Math.floor(x), f = x - i;
  return c[i]! * (1 - f) + c[i + 1]! * f;
}

/**
 * "Distance travelled" for continuous motion: `base` units per second plus `gain` units per second
 * of full drums, so the camera surges when the band hits and coasts in the breaks. Use it as the
 * phase of an orbit or the parameter of a dolly path (e.g. `angle = drive(au, t) * 0.15`).
 */
export function drive(au: AudioData, t: number, base = 1, gain = 2, t0 = 0): number {
  return base * (t - t0) + gain * (cumEnv(au, 'drums', t) - cumEnv(au, 'drums', t0));
}

/** Eased 0..1 progress between the grid points nearest t0 and t1 (a move that lands on the beat). */
export function beatEase(au: AudioData, t: number, t0: number, t1: number, fn: (x: number) => number = ease.inOutCubic): number {
  const a = au.timeOfBeat(Math.round(au.beatAt(t0))), b = au.timeOfBeat(Math.round(au.beatAt(t1)));
  return fn(clamp((t - a) / Math.max(1e-3, b - a)));
}

/** A decaying 0..1 push on each kick (for a small dolly-in that lands with the drum, not a shake). */
export function kickPush(au: AudioData, t: number, hl = 0.14): number {
  return au.hit('kick', t, hl);
}

/** Point on a horizontal orbit around `c`. */
export function orbit(c: THREE.Vector3, radius: number, height: number, angle: number): THREE.Vector3 {
  return new THREE.Vector3(c.x + Math.cos(angle) * radius, c.y + height, c.z + Math.sin(angle) * radius);
}

export interface Shot { pos: THREE.Vector3; target: THREE.Vector3; fov?: number; roll?: number }

/**
 * A camera path through keyed shots. Keys are times (snap them to downbeats with `au.downbeats`);
 * between keys the move eases in-out, and while a key holds the camera keeps drifting by `hold`.
 */
export class CamPath {
  constructor(public keys: { t: number; shot: Shot }[], public fn: (x: number) => number = ease.inOutCubic) {
    this.keys.sort((a, b) => a.t - b.t);
  }
  at(t: number): Shot {
    const k = this.keys;
    if (t <= k[0]!.t) return k[0]!.shot;
    for (let i = 0; i < k.length - 1; i++) {
      const a = k[i]!, b = k[i + 1]!;
      if (t < b.t) {
        const u = this.fn(clamp((t - a.t) / Math.max(1e-3, b.t - a.t)));
        return {
          pos: a.shot.pos.clone().lerp(b.shot.pos, u),
          target: a.shot.target.clone().lerp(b.shot.target, u),
          fov: (a.shot.fov ?? 40) + ((b.shot.fov ?? 40) - (a.shot.fov ?? 40)) * u,
          roll: (a.shot.roll ?? 0) + ((b.shot.roll ?? 0) - (a.shot.roll ?? 0)) * u,
        };
      }
    }
    return k[k.length - 1]!.shot;
  }
}

/** Apply a shot to a perspective camera (roll in radians around the view axis). */
export function applyShot(cam: THREE.PerspectiveCamera, s: Shot) {
  cam.position.copy(s.pos);
  cam.up.set(Math.sin(s.roll ?? 0), Math.cos(s.roll ?? 0), 0);
  cam.lookAt(s.target);
  if (s.fov !== undefined && cam.fov !== s.fov) { cam.fov = s.fov; cam.updateProjectionMatrix(); }
}

/**
 * A whip: 0 before `at`, then a fast eased 0..1 over `dur` (use it to swing the orbit angle or
 * the target by a large amount on a downbeat; motion blur turns it into a streak).
 */
export function whip(au: AudioData, t: number, at: number, dur = 0.18): number {
  const t0 = au.timeOfBeat(Math.round(au.beatAt(at) * 2) / 2);
  return t < t0 ? 0 : ease.inOutExpo(clamp((t - t0) / dur));
}
