// Shared parts for the memory (breakdown) and roofline (drop) scenes: the weight inventory of Llama 3.1 8B
// as printed sheets (every matrix at its real proportions), the VRAM pile, big two-ink type, labels
// projected from 3D, a type rain big enough to survive the halftone, and the RTX 3090 numbers.
import * as THREE from 'three';
import { W, H } from '../engine/gl';
import { HEX, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { hash } from '../engine/util';
import type { LlmData } from '../engine/llm';
import { flatMat, outlineInstanced, inkMat } from './_kit';
import { FLOOR_H } from './_tower';

export const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

// ------------------------------------------------------------------ numbers

/** RTX 3090 spec (NVIDIA Ampere GA102 whitepaper): memory bandwidth and dense bf16 tensor peak (FP32 accumulate). */
export const GPU = { name: 'RTX 3090', bw: 936e9, peak: 71e12 };
export const RIDGE = GPU.peak / GPU.bw; // FLOP per byte where the slope meets the ceiling (~75.9)

export const fmtInt = (n: number) => Math.round(n).toLocaleString('en-US');
export const fmt1 = (n: number) => n.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
export const fmt2 = (n: number) => n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/**
 * Bytes of weights one decode step reads: every matrix and norm, but of the embedding table (vocab x hidden,
 * untied from the output head) only the one row it looks up. 16.06 GB in memory, 15.01 GB read per token.
 */
export function readBytes(cfg: LlmData['config']) {
  const embed = cfg.vocab * cfg.hidden * 2, row = cfg.hidden * 2;
  return { weights: cfg.weight_bytes, embed, row, read: cfg.weight_bytes - embed + row };
}

/** Everything computed from the run's config and measured speeds (labelled as such on screen). */
export function runNumbers(d: LlmData) {
  const cfg = d.config, s1 = d.speed_tok_s['1'] ?? 0, s100 = d.speed_tok_s['100'] ?? 0;
  const flopsTok = 2 * cfg.params; // one multiply-add per weight per token
  const rb = readBytes(cfg);
  return {
    s1, s100, flopsTok, read: rb.read,
    effBw: rb.read * s1, // bytes/s actually streamed at batch 1 (measured tok/s x computed bytes per token)
    msTok: 1000 / s1, // measured time per token at batch 1
    tf1: (flopsTok * s1) / 1e12, // implied throughput at batch 1 (TFLOPS)
    tf100: (flopsTok * s100) / 1e12,
    util1: (flopsTok * s1) / GPU.peak, // fraction of the bf16 peak in use at batch 1
    mathMs: (flopsTok / GPU.peak) * 1000, // the token's math at peak speed
    memMs: (rb.read / GPU.bw) * 1000, // reading one token's weights at spec bandwidth
    speedup: s100 / s1,
    stepMs100: 1000 / (s100 / 100), // one batch step (100 tokens) at batch 100
  };
}

// ------------------------------------------------------------------ the weights as sheets

export type Kind = 'EMBED' | 'Q' | 'K' | 'V' | 'O' | 'GATE' | 'UP' | 'DOWN' | 'HEAD';
export const KINDS: Kind[] = ['EMBED', 'Q', 'K', 'V', 'O', 'GATE', 'UP', 'DOWN', 'HEAD'];
export const LAYER_KINDS: Kind[] = ['Q', 'K', 'V', 'O', 'GATE', 'UP', 'DOWN'];
/** Embedding and output head are 128,256 rows long: printed as fan-fold paper, 8 panels each. */
export const FOLD = 8;
/** World units per 4,096 weights of matrix side. */
export const SHEET_U = 1.3;

export interface Sheet {
  i: number;
  kind: Kind;
  layer: number; // 0 embed, 1..32, 33 head
  panel: number; // fan-fold panel (embed / head), else 0
  rows: number; // out features (as stored: rows x cols)
  cols: number; // in features
  bytes: number;
  /** read position in tower floors (0..33): the sheet is read when the token passes it */
  r: number;
  /** slot in the pile (y) */
  y: number;
}

export interface Inventory {
  sheets: Sheet[];
  byKind: Map<Kind, Sheet[]>;
  shape: Record<Kind, [number, number]>;
  bytes: Record<Kind, number>;
  layerBytes: number;
  total: number; // cfg.weight_bytes (matrices plus the norms): what memory holds
  read: number; // what one decode step reads (the embedding table contributes one row)
  /** cumulative bytes read at position f (floors 0..33), 0 -> read */
  bytesAt(f: number): number;
}

export function inventory(d: LlmData, floorY: (l: number) => number): Inventory {
  const c = d.config, kvd = c.kv_heads * c.head_dim;
  const shape: Record<Kind, [number, number]> = {
    EMBED: [c.vocab, c.hidden], Q: [c.hidden, c.hidden], K: [kvd, c.hidden], V: [kvd, c.hidden], O: [c.hidden, c.hidden],
    GATE: [c.ffn, c.hidden], UP: [c.ffn, c.hidden], DOWN: [c.hidden, c.ffn], HEAD: [c.vocab, c.hidden],
  };
  const bytes = Object.fromEntries(KINDS.map((k) => [k, shape[k][0] * shape[k][1] * 2])) as Record<Kind, number>;
  const sheets: Sheet[] = [];
  const push = (kind: Kind, layer: number, panel: number, r: number, y: number) => {
    const [rows, cols] = shape[kind];
    const fold = kind === 'EMBED' || kind === 'HEAD';
    sheets.push({ i: sheets.length, kind, layer, panel, rows: fold ? rows / FOLD : rows, cols, bytes: bytes[kind] / (fold ? FOLD : 1), r, y });
  };
  // read order: the embedding (floor 0..0.5), layer l's seven matrices while the token is on floor l, the head last
  for (let k = 0; k < FOLD; k++) push('EMBED', 0, k, (0.5 * k) / FOLD, floorY(0) - 0.42 + k * 0.05);
  for (let l = 1; l <= c.layers; l++) LAYER_KINDS.forEach((kind, j) => push(kind, l, 0, l - 0.5 + j / 7, floorY(l) + 0.07 + j * (FLOOR_H * 0.78) / 7));
  for (let k = 0; k < FOLD; k++) push('HEAD', c.layers + 1, k, c.layers + 0.5 + (0.5 * k) / FOLD, floorY(c.layers + 1) - 0.25 + k * 0.05);
  const layerBytes = LAYER_KINDS.reduce((a, k) => a + bytes[k], 0);
  const total = c.weight_bytes;
  const rb = readBytes(c), read = rb.read;
  // cumulative bytes read at each sheet's read position: the embedding gives its one looked-up row, every other
  // sheet its size, scaled so the last sheet lands on `read` (the norms spread in proportion)
  const rest = sheets.filter((s) => s.kind !== 'EMBED').reduce((a, s) => a + s.bytes, 0);
  const rs = sheets.map((s) => s.r).concat([c.layers + 1]);
  const cum: number[] = [0];
  for (const s of sheets) cum.push(cum[cum.length - 1]! + (s.kind === 'EMBED' ? (s.panel === 0 ? rb.row : 0) : (s.bytes * (read - rb.row)) / rest));
  const bytesAt = (f: number) => {
    if (f <= 0) return 0;
    if (f >= rs[rs.length - 1]!) return read;
    let i = 0;
    while (i < rs.length - 1 && rs[i + 1]! <= f) i++;
    const a = rs[i]!, b = rs[i + 1]!;
    return cum[i]! + ((f - a) / Math.max(1e-6, b - a)) * (cum[i + 1]! - cum[i]!);
  };
  const byKind = new Map<Kind, Sheet[]>();
  for (const k of KINDS) byKind.set(k, sheets.filter((s) => s.kind === k));
  return { sheets, byKind, shape, bytes, layerBytes, total, read, bytesAt };
}

/** Sheet size in world units: x = in features, z = out features. */
export const sheetSize = (s: { rows: number; cols: number }) => [(s.cols / 4096) * SHEET_U, (s.rows / 4096) * SHEET_U] as const;

const faceCache = new Map<string, THREE.CanvasTexture>();
/** A sheet's printed face: a field of blue weight cells, an ink border, and its name and shape. */
export function sheetFace(kind: Kind, rows: number, cols: number, fullRows: number): THREE.CanvasTexture {
  const key = `${kind}:${rows}:${cols}`;
  const hit = faceCache.get(key);
  if (hit) return hit;
  const s = 150;
  const cw = Math.max(96, Math.round((cols / 4096) * s)), ch = Math.max(48, Math.round((rows / 4096) * s));
  const cv = document.createElement('canvas');
  cv.width = cw * 2; cv.height = ch * 2;
  const c = cv.getContext('2d')!;
  c.scale(2, 2);
  c.fillStyle = HEX.paper; c.fillRect(0, 0, cw, ch);
  // weight cells: a deterministic field (texture, not data), banded like a real matrix heatmap
  const cell = 6;
  c.fillStyle = HEX.blue;
  const seed = KINDS.indexOf(kind) * 13 + 1;
  for (let y = 0; y < ch; y += cell) for (let x = 0; x < cw; x += cell) {
    const band = 0.5 + 0.5 * Math.sin((x / cw) * 9 + seed) * Math.cos((y / ch) * 7 - seed);
    if (hash(x, y, seed) < 0.22 + 0.4 * band) c.fillRect(x + 1, y + 1, cell - 2, cell - 2);
  }
  c.strokeStyle = HEX.ink; c.lineWidth = 5; c.strokeRect(2.5, 2.5, cw - 5, ch - 5);
  // label on a paper knock-out
  const short = Math.min(cw, ch);
  const fs = Math.max(18, Math.min(short * 0.34, 44));
  c.font = font(F.archivo(112, 900), fs);
  const tw = c.measureText(kind).width;
  const bx = 10, by = 10;
  c.fillStyle = HEX.paper; c.fillRect(bx, by, tw + 16, fs * 1.12);
  c.fillStyle = kind === 'EMBED' || kind === 'HEAD' ? HEX.pink : HEX.ink;
  c.textBaseline = 'top';
  c.fillText(kind, bx + 8, by + fs * 0.1);
  if (short > 90) {
    const dim = `${fmtInt(fullRows)}×${fmtInt(cols)}`;
    const ds = Math.min(20, fs * 0.5);
    c.font = font(F.mono(700), ds);
    const dw = c.measureText(dim).width;
    c.fillStyle = HEX.paper; c.fillRect(bx, by + fs * 1.12, dw + 16, ds * 1.35);
    c.fillStyle = HEX.ink; c.fillText(dim, bx + 8, by + fs * 1.12 + ds * 0.15);
  }
  const tx = new THREE.CanvasTexture(cv);
  tx.colorSpace = THREE.SRGBColorSpace;
  tx.anisotropy = 8;
  faceCache.set(key, tx);
  return tx;
}

/** Materials for a sheet box: printed face up (+y), blue back, ink edges. */
export function sheetMats(kind: Kind, rows: number, cols: number, fullRows: number): THREE.Material[] {
  const edge = flatMat('ink'), back = inkMat({ ink: 'blue', lit: 0.7, shade: 1 });
  const face = new THREE.MeshBasicMaterial({ map: sheetFace(kind, rows, cols, fullRows) });
  // BoxGeometry face order: +x -x +y -y +z -z
  return [edge, edge, face, back, edge, edge];
}

/** One InstancedMesh per sheet kind (count = n), outlined. */
export function sheetMeshes(inv: Inventory, n: (k: Kind) => number, thick = 0.035): Map<Kind, THREE.InstancedMesh> {
  const out = new Map<Kind, THREE.InstancedMesh>();
  for (const k of KINDS) {
    const s0 = inv.byKind.get(k)![0]!;
    const [w, d] = sheetSize(s0);
    const m = new THREE.InstancedMesh(new THREE.BoxGeometry(w, thick, d), sheetMats(k, s0.rows, s0.cols, inv.shape[k][0]), n(k));
    m.frustumCulled = false;
    outlineInstanced(m, 1.6);
    out.set(k, m);
  }
  return out;
}

// ------------------------------------------------------------------ 2D type

/** Big two-ink type: blue underprint, pink on top with multiply (prints purple where they overlap). */
export function slab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, k = 1, o: { align?: CanvasTextAlign; top?: string; under?: string; width?: number; maxW?: number; knock?: boolean } = {}) {
  if (k <= 0) return;
  c.save();
  let sz = size;
  c.font = font(F.archivo(o.width ?? 125, 900), sz);
  if (o.maxW) { const w = c.measureText(text).width; if (w > o.maxW) { sz *= o.maxW / w; c.font = font(F.archivo(o.width ?? 125, 900), sz); } }
  c.textAlign = o.align ?? 'center';
  c.textBaseline = 'middle';
  c.translate(x, y);
  const s = 1.25 - 0.25 * k;
  c.scale(s, s);
  if (o.knock) {
    // a paper knock-out under the word, so it prints clean over a busy 3D frame
    const w = c.measureText(text).width, al = o.align ?? 'center';
    const xl = al === 'center' ? -w / 2 : al === 'right' || al === 'end' ? -w : 0;
    c.fillStyle = PAPER;
    c.fillRect(xl - sz * 0.14, -sz * 0.5, w + sz * 0.32, sz * 0.98);
  }
  c.fillStyle = o.under ?? BLUE;
  c.fillText(text, sz * 0.04, sz * 0.035);
  c.globalCompositeOperation = 'multiply';
  c.fillStyle = o.top ?? PINK;
  c.fillText(text, 0, 0);
  c.restore();
}

/** A mono label (solid ink, >= 22 px, >= 600 weight so it survives the halftone). */
export function label(c: CanvasRenderingContext2D, text: string, x: number, y: number, o: { size?: number; color?: string; weight?: number; align?: CanvasTextAlign; base?: CanvasTextBaseline; bg?: string; pad?: number; rot?: number } = {}) {
  const size = Math.max(22, o.size ?? 24);
  c.save();
  c.translate(x, y);
  if (o.rot) c.rotate(o.rot);
  c.font = font(F.mono(Math.max(600, o.weight ?? 700)), size);
  c.textAlign = o.align ?? 'left';
  c.textBaseline = o.base ?? 'middle';
  c.letterSpacing = '1px';
  if (o.bg) {
    const w = c.measureText(text).width, p = o.pad ?? 8;
    const x0 = (o.align === 'center' ? -w / 2 : o.align === 'right' ? -w : 0) - p;
    const yc = o.base === 'top' ? size * 0.5 : o.base === 'bottom' || o.base === 'alphabetic' ? -size * 0.4 : 0;
    c.fillStyle = o.bg;
    c.fillRect(x0, yc - size * 0.72, w + p * 2, size * 1.44);
  }
  c.fillStyle = o.color ?? INK;
  c.fillText(text, 0, 0);
  c.restore();
}

/** Width of a mono label (same font as label()). */
export function labelW(c: CanvasRenderingContext2D, text: string, size = 24, weight = 700) {
  c.save(); c.font = font(F.mono(Math.max(600, weight)), Math.max(22, size)); c.letterSpacing = '1px';
  const w = c.measureText(text).width; c.restore(); return w;
}

/** Big mono number (a counter): ink digits with a pink misregistered underprint. */
export function bigNum(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, o: { align?: CanvasTextAlign; color?: string; under?: string } = {}) {
  c.save();
  c.font = font(F.mono(700), size);
  c.textAlign = o.align ?? 'center';
  c.textBaseline = 'middle';
  c.fillStyle = o.under ?? PINK;
  c.fillText(text, x + size * 0.035, y + size * 0.03);
  c.globalCompositeOperation = 'multiply';
  c.fillStyle = o.color ?? INK;
  c.fillText(text, x, y);
  c.restore();
}

/**
 * Type rain (background texture): columns of real strings (tensor names, sizes, tokens) scrolling at a
 * drums-driven speed. Big and bold enough to survive the halftone (the kit's tokenRain is 15 px).
 */
export function rain(c: CanvasRenderingContext2D, pool: string[], scroll: number, o: { cols?: number; color?: string; size?: number; seed?: number; density?: number } = {}) {
  const cols = o.cols ?? 9, size = o.size ?? 22, rowH = size * 2.2;
  c.save();
  c.font = font(F.mono(600), size);
  c.fillStyle = o.color ?? BLUE;
  const seed = o.seed ?? 0;
  for (let ci = 0; ci < cols; ci++) {
    const x = ((ci + 0.5) / cols) * W + (hash(ci, seed) - 0.5) * 60 - 60;
    const speed = 0.6 + hash(ci, seed + 3) * 0.9;
    const off = scroll * rowH * speed * 4 + hash(ci, seed + 7) * 1000;
    const r0 = Math.floor(off / rowH);
    for (let r = -1; r < H / rowH + 1; r++) {
      const k = r0 + r;
      if (hash(ci, k, seed) < (o.density ?? 0.7)) continue;
      const s = pool[Math.floor(hash(ci, k, seed + 11) * pool.length)]!;
      const y = H - (r * rowH - (off - r0 * rowH));
      if (hudZone(x, y, c.measureText(s).width)) continue;
      c.fillText(s, x, y);
    }
  }
  c.restore();
}

/** The overlay's corners and edges (hud.ts): the model tag, the readout, the floor ladder, the odds, the answer strip, the sung line. */
export function hudZone(x: number, y: number, w = 0): boolean {
  const P = H > W;
  if (y < (P ? 230 : 150)) return x < 470 || x + w > W - 330 || Math.abs(x + w / 2 - W / 2) < W * 0.32;
  if (x < 130) return true; // floor ladder
  if (y > H - (P ? 300 : 110)) return x < (P ? W : W * 0.6); // answer strip
  if (y > H - (P ? 390 : 260)) return x + w > W - (P ? 380 : 340); // odds
  return false;
}

/** Project a world point to frame px. Returns [x, y, inFront]. */
const _v = new THREE.Vector3();
export function toScreen(cam: THREE.Camera, p: THREE.Vector3 | [number, number, number]): [number, number, boolean] {
  if (Array.isArray(p)) _v.set(p[0], p[1], p[2]); else _v.copy(p);
  _v.project(cam);
  return [(_v.x + 1) * 0.5 * W, (1 - _v.y) * 0.5 * H, _v.z < 1];
}

/** A rubber stamp: a bordered word, rotated, pink over a misregistered blue. */
export function stamp(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, k: number, rot = -0.12) {
  if (k <= 0) return;
  c.save();
  c.translate(x, y);
  c.rotate(rot);
  const s = 1.6 - 0.6 * k;
  c.scale(s, s);
  c.font = font(F.archivo(125, 900), size);
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  const w = c.measureText(text).width + size * 0.5, h = size * 1.12;
  const lw = size * 0.09;
  for (const [col, dx, dy, mode] of [[BLUE, size * 0.03, size * 0.03, 'source-over'], [PINK, 0, 0, 'multiply']] as const) {
    c.globalCompositeOperation = mode;
    c.strokeStyle = col; c.fillStyle = col; c.lineWidth = lw;
    c.strokeRect(-w / 2 + dx, -h / 2 + dy, w, h);
    c.fillText(text, dx, dy + size * 0.04);
  }
  c.restore();
}

export const safeTop = () => (H > W ? 220 : 150);
