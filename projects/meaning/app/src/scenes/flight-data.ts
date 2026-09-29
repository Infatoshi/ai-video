// data/emb.json (llm/embed.py): Llama 3.1 8B Instruct's input embedding table, what the video shows of it.
import * as THREE from 'three';

export interface Tok { id: number; t: string; cos?: number }
export interface Emb {
  source: { model: string; shard: string; tensor: string; shape: [number, number]; sha256_bytes: string };
  vocab: number;
  dim: number;
  rows: Record<string, { id: number; values: number[]; norm: number }>;
  neighbours: Record<string, Tok[]>;
  table: { first_id: number; rows: number; t: string[]; max_abs: number; i8: string };
  baseline: { mean_cos: number; p99_cos: number };
  analogies: Record<string, { expect: string; top5: Tok[]; rank_expect: number; cos_expect: number }>;
  pairs: { a: string; b: string; rank: number; top1: string }[];
  pairs_rank1: number;
  space: {
    method: string;
    var_kept: number[];
    var_kept_total: number;
    featured: (Tok & { p: [number, number, number] })[];
    land: Record<string, [number, number, number]>;
    arrow: [number, number, number];
    haze: { n: number; t: string[]; p: number[] };
  };
}

export async function loadEmb(): Promise<Emb> {
  const r = await fetch('data/emb.json');
  if (!r.ok) throw new Error('data/emb.json missing (run llm/embed.py)');
  return r.json();
}

/** World units per PCA display unit (the labelled words fill [-1, 1] in each axis). */
export const R = 12;
export const toWorld = (p: readonly number[]) => new THREE.Vector3(p[0]! * R, p[1]! * R, p[2]! * R);

/** Signed int8 table rows (base64) -> Float32Array in [-1, 1]. */
export function tableValues(tb: Emb['table']): Float32Array {
  const bin = atob(tb.i8);
  const out = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    const b = bin.charCodeAt(i);
    out[i] = (b > 127 ? b - 256 : b) / 127;
  }
  return out;
}

/** Cosine formatted for the page: 3 decimals, no leading zero dropped (0.297). */
export const fmtCos = (c: number) => c.toFixed(3);

/** A token as printed: a leading space shows as a middle dot. */
export const tokLabel = (t: string) => t.replace(/^ /, '·');

/**
 * Time-parameterised monotone cubic (Fritsch-Butland tangents per component): passes through every key, never
 * overshoots (a component that turns around stops there; one that keeps going keeps its speed through the key),
 * so the camera flows through keys instead of stopping at each one (3Blue1Brown-style fluid moves).
 */
export class Spline {
  private m: number[][];
  constructor(public ts: number[], public vs: number[][]) {
    const n = ts.length, dims = vs[0]!.length;
    this.m = ts.map(() => new Array(dims).fill(0));
    for (let k = 1; k < n - 1; k++) for (let d = 0; d < dims; d++) {
      const h0 = ts[k]! - ts[k - 1]!, h1 = ts[k + 1]! - ts[k]!;
      const d0 = (vs[k]![d]! - vs[k - 1]![d]!) / h0, d1 = (vs[k + 1]![d]! - vs[k]![d]!) / h1;
      this.m[k]![d] = d0 * d1 <= 0 ? 0 : (3 * (h0 + h1)) / ((2 * h1 + h0) / d0 + (h1 + 2 * h0) / d1);
    }
  }
  at(t: number): number[] {
    const ts = this.ts, vs = this.vs, n = ts.length;
    if (t <= ts[0]!) return vs[0]!.slice();
    if (t >= ts[n - 1]!) return vs[n - 1]!.slice();
    let i = 0;
    while (i < n - 2 && t >= ts[i + 1]!) i++;
    const t0 = ts[i]!, t1 = ts[i + 1]!, h = t1 - t0, u = (t - t0) / h;
    const h00 = 2 * u ** 3 - 3 * u ** 2 + 1, h10 = u ** 3 - 2 * u ** 2 + u, h01 = -2 * u ** 3 + 3 * u ** 2, h11 = u ** 3 - u ** 2;
    return vs[i]!.map((p0, d) => h00 * p0 + h10 * h * this.m[i]![d]! + h01 * vs[i + 1]![d]! + h11 * h * this.m[i + 1]![d]!);
  }
}
