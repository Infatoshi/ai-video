// Parts for the `spec` scene (verse 4): the roofline chart as a printed 3D object, the 10 x 10 grid of
// prompt cards, the stream of weight sheets, and the scene's type helpers. Every number the chart
// places comes from data/llm.json (measured speed at batch 1 and 100) or arithmetic on it; the compute
// roof is drawn as a shape only (no number), just above the measured batch-100 point.
import * as THREE from 'three';
import { W, H } from '../engine/gl';
import { rgba, HEX } from '../engine/palette';
import { F, font } from '../engine/type';
import { hash } from '../engine/util';
import { showTok } from '../engine/llm';
import { inked, inkMat, flatMat, outline, outlineInstanced } from './_kit';

export const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

// ------------------------------------------------------------------ the roofline chart (3D)
// x: arithmetic intensity, log10(FLOP per byte of weights read) from -0.6 to 3 (0.25 .. 1000)
// y: throughput, log10(tokens/s) from 0.8 to 3.9. In bf16 each 2-byte weight does one multiply-add
// (2 FLOPs) per token in the batch, so batch B = B FLOP/byte. The memory slope is the measured batch-1
// speed times B (every extra prompt rides the same read for free).
export const CH = { w: 12, h: 7, uMin: -0.6, uMax: 3.0, vMin: 0.8, vMax: 3.9, depth: 1.0 };
export const chX = (u: number) => ((u - CH.uMin) / (CH.uMax - CH.uMin)) * CH.w;
export const chY = (v: number) => ((v - CH.vMin) / (CH.vMax - CH.vMin)) * CH.h;

export interface Chart {
  group: THREE.Group;
  dot1: THREE.Mesh;
  dot100: THREE.Mesh;
  slopeEdge: THREE.Mesh;
  roofEdge: THREE.Mesh;
  pen: THREE.Mesh;
  /** log10 of the measured speeds and the roof, and helpers in chart space */
  v1: number;
  v100: number;
  roofV: number;
  ridgeU: number;
  /** Chart point (u = log10 FLOP/byte, v = log10 tok/s) to local 3D. */
  at: (u: number, v: number, z?: number) => THREE.Vector3;
  /** v on the roofline at u. */
  line: (u: number) => number;
}

export function buildChart(speed1: number, speed100: number): Chart {
  const g = new THREE.Group();
  const v1 = Math.log10(speed1), v100 = Math.log10(speed100);
  const roofV = v100 + 0.2; // schematic roof: just above the measured batch-100 point
  const ridgeU = roofV - v1;
  const line = (u: number) => Math.min(v1 + u, roofV);
  const at = (u: number, v: number, z = 0) => new THREE.Vector3(chX(u), chY(v), z);

  // the area under the roofline, extruded: memory-bound part (under the slope) and compute-bound part
  const mk = (u0: number, u1: number, o: Parameters<typeof inked>[1]) => {
    const s = new THREE.Shape();
    s.moveTo(chX(u0), 0);
    s.lineTo(chX(u0), chY(line(u0)));
    if (u0 < ridgeU && u1 > ridgeU) s.lineTo(chX(ridgeU), chY(roofV));
    s.lineTo(chX(u1), chY(line(u1)));
    s.lineTo(chX(u1), 0);
    s.closePath();
    const geo = new THREE.ExtrudeGeometry(s, { depth: CH.depth, bevelEnabled: false });
    geo.translate(0, 0, -CH.depth);
    const m = inked(geo, o, 2.2);
    g.add(m);
    return m;
  };
  mk(CH.uMin, ridgeU, { ink: 'blue', lit: 0.16, shade: 1 });
  mk(ridgeU, CH.uMax, { ink: 'blue', lit: 0.42, shade: 1, over: 'pink', overLit: 0, overShade: 0.4 });

  // the roofline itself: a thick ribbon along the top edge (slope in pink, roof in black)
  const ribbon = (a: THREE.Vector3, b: THREE.Vector3, ink: 'pink' | 'ink') => {
    const len = a.distanceTo(b);
    const geo = new THREE.BoxGeometry(len, 0.2, CH.depth + 0.16);
    geo.translate(len / 2, 0, 0);
    const m = new THREE.Mesh(geo, flatMat(ink));
    outline(m, 2);
    m.position.set(a.x, a.y, -CH.depth / 2);
    m.rotation.z = Math.atan2(b.y - a.y, b.x - a.x);
    g.add(m);
    return m;
  };
  const slopeEdge = ribbon(at(CH.uMin, line(CH.uMin)), at(ridgeU, roofV), 'pink');
  const roofEdge = ribbon(at(ridgeU, roofV), at(CH.uMax, roofV), 'ink');

  // axes and ticks (1, 10, 100 FLOP/byte)
  const axisMat = flatMat('ink');
  const xa = new THREE.Mesh(new THREE.BoxGeometry(CH.w + 0.9, 0.1, 0.1), axisMat);
  xa.position.set(CH.w / 2 + 0.3, -0.08, 0.05);
  g.add(xa);
  const ya = new THREE.Mesh(new THREE.BoxGeometry(0.1, CH.h + 0.6, 0.1), axisMat);
  ya.position.set(-0.08, CH.h / 2 + 0.2, 0.05);
  g.add(ya);
  for (const u of [0, 1, 2]) {
    const tk = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.4, 0.1), axisMat);
    tk.position.set(chX(u), -0.24, 0.05);
    g.add(tk);
  }
  const cone = new THREE.ConeGeometry(0.2, 0.5, 12);
  const ax = new THREE.Mesh(cone, axisMat); ax.rotation.z = -Math.PI / 2; ax.position.set(CH.w + 0.95, -0.08, 0.05); g.add(ax);
  const ay = new THREE.Mesh(cone, axisMat); ay.position.set(-0.08, CH.h + 0.7, 0.05); g.add(ay);

  // the operating points
  const sph = new THREE.SphereGeometry(0.38, 32, 20);
  const dot1 = inked(sph, { ink: 'pink', lit: 0.75, shade: 1 }, 3);
  dot1.position.copy(at(0, v1, 0.25));
  g.add(dot1);
  const dot100 = inked(sph, { ink: 'pink', lit: 0.75, shade: 1, over: 'blue', overLit: 0, overShade: 0.5 }, 3);
  dot100.position.copy(at(2, v100, 0.25));
  g.add(dot100);
  // the pen that traces the roofline
  const pen = inked(new THREE.SphereGeometry(0.26, 24, 16), { ink: 'ink', lit: 0.6, shade: 1 }, 2);
  g.add(pen);

  return { group: g, dot1, dot100, slopeEdge, roofEdge, pen, v1, v100, roofV, ridgeU, at, line };
}

// ------------------------------------------------------------------ 100 prompt cards (3D)

/** A chat card's printed top: paper, a blue title bar, three ink lines of "text" (a picture of a prompt). */
function cardTexture(): THREE.CanvasTexture {
  const cv = document.createElement('canvas');
  cv.width = 256; cv.height = 176;
  const c = cv.getContext('2d')!;
  c.fillStyle = HEX.paper; c.fillRect(0, 0, 256, 176);
  c.fillStyle = HEX.blue; c.fillRect(0, 0, 256, 38);
  c.fillStyle = HEX.ink;
  c.fillRect(22, 60, 200, 16); c.fillRect(22, 90, 150, 16);
  c.strokeStyle = HEX.ink; c.lineWidth = 10; c.strokeRect(5, 5, 246, 166);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

export interface Grid {
  group: THREE.Group;
  cards: THREE.InstancedMesh;
  tiles: THREE.InstancedMesh;
  rays: THREE.InstancedMesh;
  source: THREE.Mesh;
  /** Card centre (local). */
  cell: (i: number) => THREE.Vector3;
  pitchX: number;
  pitchZ: number;
}

export function buildGrid(tile: THREE.Mesh): Grid {
  const g = new THREE.Group();
  const pitchX = 1.3, pitchZ = 1.05;
  const cell = (i: number) => new THREE.Vector3(((i % 10) - 4.5) * pitchX, 0, (Math.floor(i / 10) - 4.5) * pitchZ);
  const side = inkMat({ ink: 'blue', lit: 0.5, shade: 1 });
  const top = new THREE.MeshBasicMaterial({ map: cardTexture() });
  const cardGeo = new THREE.BoxGeometry(1.1, 0.12, 0.8);
  const cards = new THREE.InstancedMesh(cardGeo, [side, side, top, side, side, side], 100);
  const tmp = new THREE.Object3D();
  for (let i = 0; i < 100; i++) { tmp.position.copy(cell(i)); tmp.updateMatrix(); cards.setMatrixAt(i, tmp.matrix); }
  cards.frustumCulled = false;
  outlineInstanced(cards, 1.6);
  g.add(cards);
  const tiles = new THREE.InstancedMesh(tile.geometry, tile.material as THREE.Material[], 100);
  tiles.frustumCulled = false;
  outlineInstanced(tiles, 1.4);
  g.add(tiles);
  const rayGeo = new THREE.BoxGeometry(1, 1, 1);
  rayGeo.translate(0, 0, 0.5);
  const rays = new THREE.InstancedMesh(rayGeo, flatMat('pink'), 100);
  rays.frustumCulled = false;
  g.add(rays);
  const source = inked(new THREE.BoxGeometry(5.6, 0.7, 3.2), { ink: 'blue', lit: 0.35, shade: 1, over: 'pink', overLit: 0, overShade: 0 }, 2.4);
  source.position.set(0, 6.2, -7.5);
  g.add(source);
  return { group: g, cards, tiles, rays, source, cell, pitchX, pitchZ };
}

// ------------------------------------------------------------------ weight sheets (3D)

export function buildSheets(n = 56): THREE.InstancedMesh {
  const m = new THREE.InstancedMesh(new THREE.BoxGeometry(1.5, 0.04, 1.05), inkMat({ ink: 'blue', lit: 0.3, shade: 1, over: 'pink', overLit: 0, overShade: 0.6 }), n);
  m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
  const col = new THREE.Color();
  for (let i = 0; i < n; i++) m.setColorAt(i, col.setRGB(hash(i, 5) < 0.18 ? 1 : 0, 1, 0)); // a few printed pink
  m.frustumCulled = false;
  outlineInstanced(m, 1.3);
  return m;
}

// ------------------------------------------------------------------ type (all solid ink, >= 22 px, >= 600)

export type Ink = string;
export const mono = (px: number, w = 700) => font(F.mono(w), px);
export const arch = (px: number, w = 900, wd = 125) => font(F.archivo(wd, w), px);

/** Plain type. */
export function txt(c: CanvasRenderingContext2D, s: string, x: number, y: number, f: string, col: Ink, align: CanvasTextAlign = 'left', base: CanvasTextBaseline = 'middle') {
  c.font = f; c.fillStyle = col; c.textAlign = align; c.textBaseline = base;
  c.fillText(s, x, y);
}

/** A knockout label: a solid box with the text in paper (or `fg`). Returns the box width. */
export function box(c: CanvasRenderingContext2D, s: string, x: number, y: number, f: string, bg: Ink, fg: Ink = PAPER, align: CanvasTextAlign = 'left', padX = 14, padY = 9): number {
  c.font = f;
  c.textBaseline = 'alphabetic'; // the box height comes from the glyph bounds, measured on the baseline we draw on
  const m = c.measureText(s);
  const w = m.width + padX * 2;
  const a = m.actualBoundingBoxAscent, d = m.actualBoundingBoxDescent;
  const hh = Math.max(a + d, 10) + padY * 2;
  const x0 = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  c.fillStyle = bg;
  c.fillRect(x0, y - hh / 2, w, hh);
  c.fillStyle = fg; c.textAlign = 'left'; c.textBaseline = 'alphabetic';
  c.fillText(s, x0 + padX, y + (a - d) / 2);
  return w;
}

/** Big two-ink type: blue underprint, pink on top with multiply (prints purple where they overlap). */
export function slab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, k = 1, align: CanvasTextAlign = 'center', top: Ink = PINK, under: Ink = BLUE) {
  if (k <= 0) return; // not snapped in yet
  c.save();
  c.font = arch(size);
  c.textAlign = align;
  c.textBaseline = 'middle';
  c.translate(x, y);
  const s = 1.25 - 0.25 * k;
  c.scale(s, s);
  c.fillStyle = under;
  c.fillText(text, size * 0.04, size * 0.035);
  c.globalCompositeOperation = 'multiply';
  c.fillStyle = top;
  c.fillText(text, 0, 0);
  c.restore();
}

/** A rubber stamp: a thick rotated frame with solid type inside. */
export function stamp(c: CanvasRenderingContext2D, s: string, x: number, y: number, size: number, col: Ink, rot = -0.1, k = 1) {
  if (k <= 0) return;
  c.save();
  c.translate(x, y);
  c.rotate(rot);
  const sc = 1.35 - 0.35 * k;
  c.scale(sc, sc);
  c.font = arch(size);
  const w = c.measureText(s).width + size * 0.6, h = size * 1.2;
  c.strokeStyle = col; c.lineWidth = Math.max(6, size * 0.09);
  c.strokeRect(-w / 2, -h / 2, w, h);
  c.fillStyle = col; c.textAlign = 'center'; c.textBaseline = 'middle';
  c.fillText(s, 0, size * 0.04);
  c.restore();
}

/** A token as a printed tile (Canvas2D): solid box, token text knocked out. Returns width. */
export function tile2(c: CanvasRenderingContext2D, tok: string, x: number, y: number, size: number, bg: Ink, fg: Ink = PAPER, minW = 0, align: CanvasTextAlign = 'left'): number {
  c.font = mono(size, 700);
  const lab = showTok(tok);
  const w = Math.max(minW, c.measureText(lab).width + size * 0.8), h = size * 1.55;
  const x0 = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  c.fillStyle = bg; c.fillRect(x0, y - h / 2, w, h);
  c.fillStyle = fg; c.textAlign = 'center'; c.textBaseline = 'middle';
  c.fillText(lab, x0 + w / 2, y + size * 0.05);
  return w;
}

/** A tick (check mark) and a cross, as thick strokes. */
export function tick(c: CanvasRenderingContext2D, x: number, y: number, s: number, col: Ink, k = 1) {
  if (k <= 0) return;
  c.save(); c.translate(x, y); c.scale(k, k);
  c.strokeStyle = col; c.lineWidth = s * 0.22; c.lineCap = 'square'; c.lineJoin = 'miter';
  c.beginPath(); c.moveTo(-s * 0.45, 0); c.lineTo(-s * 0.1, s * 0.35); c.lineTo(s * 0.5, -s * 0.4); c.stroke();
  c.restore();
}
export function cross(c: CanvasRenderingContext2D, x: number, y: number, s: number, col: Ink, k = 1) {
  if (k <= 0) return;
  c.save(); c.translate(x, y); c.scale(k, k);
  c.strokeStyle = col; c.lineWidth = s * 0.22; c.lineCap = 'square';
  c.beginPath(); c.moveTo(-s * 0.4, -s * 0.4); c.lineTo(s * 0.4, s * 0.4); c.moveTo(s * 0.4, -s * 0.4); c.lineTo(-s * 0.4, s * 0.4); c.stroke();
  c.restore();
}

/**
 * Background rain of real tokens (strings and ids), bold and big enough to print cleanly through the
 * halftone (the kit's tokenRain is 15 px / 500; the lead asked for >= 22 px / >= 600 in scenes).
 */
export function rain(c: CanvasRenderingContext2D, pool: { id: number; t: string }[], scroll: number, o: { cols?: number; seed?: number; color?: Ink; size?: number } = {}) {
  const cols = o.cols ?? 9, size = o.size ?? 22, rowH = size * 2.3, seed = o.seed ?? 0;
  c.save();
  c.font = mono(size, 600);
  c.fillStyle = o.color ?? BLUE;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  for (let ci = 0; ci < cols; ci++) {
    const x = ((ci + 0.5) / cols) * W + (hash(ci, seed) - 0.5) * 50;
    const speed = 0.6 + hash(ci, seed + 3) * 0.9;
    const off = scroll * rowH * speed * 4 + hash(ci, seed + 7) * 1000;
    const r0 = Math.floor(off / rowH);
    for (let r = -1; r < H / rowH + 1; r++) {
      const k = r0 + r;
      if (hash(ci, k, seed) < 0.72) continue;
      const tok = pool[Math.floor(hash(ci, k, seed + 11) * pool.length)]!;
      const y = H - (r * rowH - (off - r0 * rowH));
      c.fillText(hash(ci, k, seed + 5) < 0.5 ? String(tok.id) : showTok(tok.t), x, y);
    }
  }
  c.restore();
}
