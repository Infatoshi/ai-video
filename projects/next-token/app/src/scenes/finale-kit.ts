// Shared parts for the finale and the outro (same author): print-safe type helpers, the odds bar with
// its needle at the real uniform draw, the die built from a step's real nucleus, context tile rows,
// attention arcs, the embedding bar code, and a clip that keeps scene type out of the HUD's corners.
// Everything draws solid ink at >= 22 px / weight >= 600 (smaller type halftones into mush).
import * as THREE from 'three';
import { W, H } from '../engine/gl';
import { HEX, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, hash } from '../engine/util';
import { showTok, type Llm, type LlmStep } from '../engine/llm';
import { outline, portrait, type InkName } from './_kit';

export const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

export const pct = (p: number, d = 1) => `${(p * 100).toFixed(d)}%`;

/**
 * Clip to where scene type may print: below the sung line and the model tag, off the floor ladder, and
 * clear of the HUD's bottom corners (answer strip bottom-left, odds bottom-right).
 */
export function safeClip(c: CanvasRenderingContext2D) {
  const P = portrait();
  const x0 = 116, x1 = W - (P ? 58 : 100), y0 = P ? 236 : 138, y1 = H - (P ? 150 : 112);
  const bx = W - (P ? 392 : 352), by = H - (P ? 424 : 300);
  c.beginPath();
  c.moveTo(x0, y0); c.lineTo(x1, y0); c.lineTo(x1, by); c.lineTo(bx, by); c.lineTo(bx, y1); c.lineTo(x0, y1); c.closePath();
  c.clip();
}

/** The safe box (without the odds notch) for layout: [x0, y0, x1, y1]. */
export function safeBox(): [number, number, number, number] {
  const P = portrait();
  return [P ? 124 : 130, P ? 250 : 150, W - (P ? 70 : 110), H - (P ? 160 : 120)];
}

/** Big two-ink type: an underprint offset from the top ink, the top ink multiplied over it (overprint). */
export function slab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number,
  o: { k?: number; align?: CanvasTextAlign; width?: number; weight?: number; top?: string; under?: string | null; off?: number; mono?: boolean } = {}) {
  const k = o.k ?? 1;
  c.save();
  c.font = font(o.mono ? F.mono(700) : F.archivo(o.width ?? 125, o.weight ?? 900), size);
  c.textAlign = o.align ?? 'center';
  c.textBaseline = 'middle';
  c.translate(x, y);
  const s = 1.25 - 0.25 * k;
  c.scale(s, s);
  const off = o.off ?? 0.035;
  if (o.under !== null) { c.fillStyle = o.under ?? BLUE; c.fillText(text, size * off, size * off * 0.85); }
  c.globalCompositeOperation = 'multiply';
  c.fillStyle = o.top ?? PINK;
  c.fillText(text, 0, 0);
  c.restore();
}

/** Plain solid label (mono 700 by default). */
export function label(c: CanvasRenderingContext2D, text: string, x: number, y: number, size = 24, color = INK, align: CanvasTextAlign = 'left', family?: string) {
  c.font = font(family ?? F.mono(700), Math.max(22, size));
  c.fillStyle = color;
  c.textAlign = align;
  c.textBaseline = 'middle';
  c.fillText(text, x, y);
  c.textAlign = 'left';
}

/** Token rain, print-safe: real token strings and ids, 22 px mono 600, sparse, scrolling with `scroll`. */
export function rain(c: CanvasRenderingContext2D, pool: string[], scroll: number, o: { cols?: number; seed?: number; size?: number; color?: string; density?: number } = {}) {
  const P = portrait();
  const cols = o.cols ?? (P ? 6 : 10), size = o.size ?? 22, rowH = size * 2.7, seed = o.seed ?? 0, dens = o.density ?? 0.24;
  c.save();
  safeClip(c);
  c.font = font(F.mono(600), size);
  c.fillStyle = o.color ?? BLUE;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  for (let ci = 0; ci < cols; ci++) {
    const x = ((ci + 0.5) / cols) * W + (hash(ci, seed) - 0.5) * 60;
    const speed = 0.6 + hash(ci, seed + 3) * 0.9;
    const off = scroll * rowH * speed * 4 + hash(ci, seed + 7) * 1000;
    const r0 = Math.floor(off / rowH);
    for (let r = -1; r < H / rowH + 1; r++) {
      const k = r0 + r;
      if (hash(ci, k, seed) > dens) continue;
      const lab = pool[Math.floor(hash(ci, k, seed + 11) * pool.length)]!;
      c.fillText(lab, x, H - (r * rowH - (off - r0 * rowH)));
    }
  }
  c.restore();
}

/** Rain pool: the prompt's and the answer's real tokens and ids (short ones only). */
export function rainPool(llm: Llm): string[] {
  const out: string[] = [];
  for (const x of [...llm.d.prompt, ...llm.steps.map((s) => s.tok)]) {
    const s = showTok(x.t);
    if (s.length <= 9) out.push(s);
    out.push(String(x.id));
  }
  return out;
}

// ------------------------------------------------------------------ the die

const faceCache = new Map<string, THREE.CanvasTexture>();
/** A die face: the token large, optionally a second line (its odds). 512 px. */
export function faceTex(tok: string, bg: InkName, fg: InkName, sub?: string): THREE.CanvasTexture {
  const key = `${tok}|${bg}|${fg}|${sub ?? ''}`;
  let tx = faceCache.get(key);
  if (tx) return tx;
  const n = 512;
  const cv = document.createElement('canvas');
  cv.width = n; cv.height = n;
  const c = cv.getContext('2d')!;
  c.fillStyle = HEX[bg]; c.fillRect(0, 0, n, n);
  c.strokeStyle = HEX[fg]; c.lineWidth = 26; c.strokeRect(13, 13, n - 26, n - 26);
  const lab = showTok(tok);
  let size = n * 0.46;
  c.font = font(F.mono(700), size);
  const w = c.measureText(lab).width;
  if (w > n * 0.8) { size *= (n * 0.8) / w; c.font = font(F.mono(700), size); }
  c.fillStyle = HEX[fg]; c.textAlign = 'center'; c.textBaseline = 'middle';
  c.fillText(lab, n / 2, n * (sub ? 0.42 : 0.52));
  if (sub) { c.font = font(F.mono(700), n * 0.14); c.fillText(sub, n / 2, n * 0.76); }
  tx = new THREE.CanvasTexture(cv);
  tx.colorSpace = THREE.SRGBColorSpace;
  tx.anisotropy = 8;
  faceCache.set(key, tx);
  return tx;
}

export interface Die { mesh: THREE.Mesh; land: THREE.Quaternion; step: number }
// the rotation that turns BoxGeometry face i (+x -x +y -y +z -z) to the camera (+z)
const TO_FRONT = [new THREE.Euler(0, -Math.PI / 2, 0), new THREE.Euler(0, Math.PI / 2, 0), new THREE.Euler(Math.PI / 2, 0, 0),
  new THREE.Euler(-Math.PI / 2, 0, 0), new THREE.Euler(0, 0, 0), new THREE.Euler(0, Math.PI, 0)];

/**
 * A step's die: its faces are the nucleus it sampled from (paper; the pick pink), then the next candidates
 * at temperature 1 that top-p cut (blue). A real choice prints each in-draw face's odds.
 */
export function makeDie(llm: Llm, s: number, size = 2.4): Die {
  const st = llm.steps[s]!;
  const inDraw = new Set(st.nucleus.map(([t]) => t));
  const cands = st.nucleus.slice(0, 6).map(([t, p]) => ({ t, p, inDraw: true }));
  for (const [t, p] of st.top10_t1) if (cands.length < 6 && !inDraw.has(t)) cands.push({ t, p, inDraw: false });
  const choice = st.nucleus.length > 1;
  const mats = cands.map((c, i) => new THREE.MeshBasicMaterial({
    map: faceTex(c.t, i === st.pick ? 'pink' : c.inDraw ? 'paper' : 'blue', i === st.pick ? 'paper' : c.inDraw ? 'ink' : 'paper', c.inDraw && choice ? pct(c.p) : undefined),
  }));
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size, size, size), mats);
  outline(mesh, 3);
  return { mesh, land: new THREE.Quaternion().setFromEuler(TO_FRONT[Math.min(st.pick, 5)]!), step: s };
}

const tq = new THREE.Quaternion(), te = new THREE.Euler();
/** Tumbling orientation for a drive distance `d`. */
export function tumble(d: number, seed = 0): THREE.Quaternion {
  return tq.setFromEuler(te.set(d * 1.9 + seed, d * 1.3 + seed * 2.1, d * 0.7 + seed * 0.5)).clone();
}

/** Put the die at `t`: tumbling until it settles onto its pick over the ~0.14 s before `land` (Infinity = never). */
export function rollDie(die: Die, d: number, t: number, land: number, seed = 0) {
  const k = land === Infinity ? 0 : ease.outCubic(clamp((t - (land - 0.14)) / 0.14));
  die.mesh.quaternion.copy(tumble(d, seed)).slerp(die.land, k);
}

// ------------------------------------------------------------------ the odds bar

export interface BarOpts {
  reveal?: number;
  /** 0..1 of the way from 0 to u. */
  needle?: number;
  landed?: boolean;
  fs?: number;
  ruler?: boolean;
  dist?: [string, number][];
  pick?: number;
  u?: number;
}

/** The nucleus as a stacked bar (what the dice sampled from), a 0..1 ruler and the needle at the real u. */
export function oddsBar(c: CanvasRenderingContext2D, st: LlmStep, x: number, y: number, w: number, h: number, o: BarOpts = {}) {
  const dist = o.dist ?? st.nucleus, pick = o.pick ?? st.pick, u = o.u ?? st.u;
  const reveal = o.reveal ?? 1, needle = o.needle ?? 0, landed = o.landed ?? false;
  const fs = o.fs ?? clamp(h * 0.34, 22, 46);
  c.save();
  c.textBaseline = 'middle';
  let acc = 0;
  const narrow: { x: number; w: number; lab: string; hot: boolean }[] = [];
  dist.forEach(([t, p], i) => {
    const x0 = x + acc * w * reveal, pw = p * w * reveal;
    const hot = landed && i === pick;
    c.fillStyle = hot ? PINK : i % 2 ? BLUE : INK;
    c.fillRect(x0, y, Math.max(3, pw - 4), h);
    const lab = showTok(t);
    c.font = font(F.mono(700), fs);
    const lw = Math.max(c.measureText(lab).width, fs * 3.4);
    if (reveal > 0.95) {
      if (pw > lw + 30 && h >= 70) {
        c.fillStyle = PAPER;
        c.fillText(lab, x0 + 14, y + h * 0.33);
        c.font = font(F.mono(700), Math.max(22, fs * 0.68));
        c.fillText(pct(p), x0 + 14, y + h * 0.72);
      } else if (pw > lw + 20) {
        c.fillStyle = PAPER;
        c.fillText(`${lab} ${pct(p, 0)}`, x0 + 12, y + h * 0.5);
      } else narrow.push({ x: x0, w: pw, lab: `${lab} ${pct(p)}`, hot });
    }
    acc += p;
  });
  // the 0..1 ruler
  const ry = y + h + 14;
  if (o.ruler ?? true) {
    c.fillStyle = INK;
    c.fillRect(x, ry, w, 4);
    for (let i = 0; i <= 10; i++) c.fillRect(x + (w * i) / 10 - 1.5, ry - 6, 4, i % 5 ? 12 : 22);
    c.font = font(F.mono(700), 22);
    c.textBaseline = 'top';
    c.fillText('0', x - 6, ry + 20);
    c.fillText('1', x + w - 8, ry + 20);
  }
  // segments too narrow for their label: labelled under the ruler, right-aligned to the segment
  c.textBaseline = 'top';
  narrow.forEach((n, k) => {
    c.fillStyle = n.hot ? PINK : INK;
    c.font = font(F.mono(700), 24);
    c.textAlign = 'right';
    const ly = ry + 56 + k * 34;
    c.fillRect(n.x + n.w - 6, ry + 6, 3, ly - ry - 6);
    c.fillText(n.lab, n.x + n.w - 14, ly);
    c.textAlign = 'left';
  });
  // the needle at the real uniform draw
  if (needle > 0) {
    const nx = x + w * u * clamp(needle);
    c.fillStyle = PINK;
    c.fillRect(nx - 4, y - 36, 8, h + 58);
    c.beginPath(); c.moveTo(nx - 18, y - 54); c.lineTo(nx + 18, y - 54); c.lineTo(nx, y - 30); c.fill();
    c.fillStyle = INK;
    c.font = font(F.mono(700), 26);
    c.textBaseline = 'bottom';
    const lab = `u = ${u.toFixed(3)}`;
    const lw = c.measureText(lab).width;
    c.fillText(lab, nx + 24 + lw > x + w + 60 ? nx - 24 - lw : nx + 24, y - 38);
  }
  c.restore();
}

// ------------------------------------------------------------------ tile rows, arcs, bar code

export interface Box { x: number; y: number; w: number; h: number; t: string }

/** Tokens as printed boxes (ink, paper type), wrapping inside [x, maxX]; an optional dashed empty slot at the end. */
export function tileRow(c: CanvasRenderingContext2D, toks: string[], x: number, y: number, size: number, maxX: number,
  o: { hot?: number[]; slot?: boolean; slotOn?: boolean; center?: boolean; fill?: string; hotFill?: string } = {}): Box[] {
  c.save();
  c.font = font(F.mono(700), size);
  c.textBaseline = 'middle';
  const padX = size * 0.34, bh = size * 1.5, gap = size * 0.24;
  const rows: Box[][] = [[]];
  let cx = x, cy = y;
  const items = [...toks, ...(o.slot ? [''] : [])];
  for (const t of items) {
    const lab = t === '' ? '' : showTok(t);
    const bw = t === '' ? size * 2.2 : c.measureText(lab).width + padX * 2;
    if (cx + bw > maxX && rows[rows.length - 1]!.length) { rows.push([]); cx = x; cy += bh + gap * 1.6; }
    rows[rows.length - 1]!.push({ x: cx, y: cy, w: bw, h: bh, t });
    cx += bw + gap;
  }
  if (o.center) for (const row of rows) {
    const last = row[row.length - 1]!;
    const dx = (maxX - (last.x + last.w)) / 2;
    for (const b of row) b.x += dx;
  }
  const boxes = rows.flat();
  boxes.forEach((b, i) => {
    if (b.t === '') {
      if (o.slotOn ?? true) {
        c.strokeStyle = PINK; c.lineWidth = 6; c.setLineDash([14, 9]);
        c.strokeRect(b.x + 3, b.y - b.h / 2 + 3, b.w - 6, b.h - 6);
        c.setLineDash([]);
      }
      return;
    }
    c.fillStyle = o.hot?.includes(i) ? o.hotFill ?? PINK : o.fill ?? INK;
    c.fillRect(b.x, b.y - b.h / 2, b.w, b.h);
    c.fillStyle = PAPER;
    c.fillText(showTok(b.t), b.x + padX, b.y + size * 0.04);
  });
  c.restore();
  return boxes;
}

/** Attention arcs from `from` back to each box, line width by the real weight (only weights >= `min`). */
export function arcs(c: CanvasRenderingContext2D, from: Box, boxes: Box[], w: number[], o: { min?: number; lift?: number; k?: number; hot?: number } = {}) {
  const min = o.min ?? 0.02, lift = o.lift ?? 1, k = o.k ?? 1;
  c.save();
  c.lineCap = 'round';
  const x0 = from.x + from.w / 2, y0 = from.y - from.h / 2;
  boxes.forEach((b, i) => {
    const wt = w[i] ?? 0;
    if (wt < min) return;
    const x1 = b.x + b.w / 2, y1 = b.y - b.h / 2;
    const span = Math.hypot(x1 - x0, y1 - y0);
    const cy = Math.min(y0, y1) - (80 + span * 0.42) * lift;
    c.strokeStyle = i === o.hot ? PINK : BLUE;
    c.lineWidth = 3 + 34 * wt;
    c.beginPath();
    c.moveTo(x0, y0);
    // grow along the curve with k
    const steps = 40, n = Math.max(1, Math.round(steps * clamp(k)));
    for (let s = 1; s <= n; s++) {
      const u = s / steps;
      const qx = (1 - u) * (1 - u) * x0 + 2 * (1 - u) * u * ((x0 + x1) / 2) + u * u * x1;
      const qy = (1 - u) * (1 - u) * y0 + 2 * (1 - u) * u * cy + u * u * y1;
      c.lineTo(qx, qy);
    }
    c.stroke();
  });
  c.restore();
}

/** An embedding as a bar code: one bar per value, pink up for positive, blue down for negative. */
export function barcode(c: CanvasRenderingContext2D, vals: number[], x: number, y: number, w: number, h: number, k = 1) {
  const m = Math.max(...vals.map(Math.abs), 1e-9);
  const bw = w / vals.length;
  const n = Math.round(vals.length * clamp(k));
  for (let i = 0; i < n; i++) {
    const v = vals[i]!, hh = (v / m) * h;
    c.fillStyle = v >= 0 ? PINK : BLUE;
    c.fillRect(x + i * bw + 1, v >= 0 ? y - hh : y, Math.max(2, bw - 3), Math.max(3, Math.abs(hh)));
  }
  c.fillStyle = INK;
  c.fillRect(x, y - 2, w, 4);
}

// ------------------------------------------------------------------ camera helpers

/** World point at frame fraction (fx, fy) on the plane z, for a camera looking straight down -z. */
export function worldAt(cam: THREE.PerspectiveCamera, fx: number, fy: number, z = 0): THREE.Vector3 {
  const d = cam.position.z - z;
  const hh = Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) * d, hw = hh * cam.aspect;
  return new THREE.Vector3(cam.position.x + (fx - 0.5) * 2 * hw, cam.position.y + (0.5 - fy) * 2 * hh, z);
}

/** Screen position (logical px) of a world point. */
export function toScreen(v: THREE.Vector3, cam: THREE.PerspectiveCamera): [number, number] {
  const p = v.clone().project(cam);
  return [((p.x + 1) / 2) * W, ((1 - p.y) / 2) * H];
}

/** A straight-on camera at distance D (for flat framings where 3D tiles sit next to 2D type). */
export function flatCam(cam: THREE.PerspectiveCamera, D = 10, fov = 36) {
  cam.position.set(0, 0, D);
  cam.up.set(0, 1, 0);
  if (cam.fov !== fov) { cam.fov = fov; cam.updateProjectionMatrix(); }
  cam.lookAt(0, 0, 0);
}
