// Canvas-generated textures for the H100 model (_h100.ts): the GH100 die surface (engraved block
// layout) and the SXM5 board's trace pattern. Pure functions of their inputs (seeded), built once.
//
// The die layout follows the logical arrangement of NVIDIA's GH100 block diagram (H100 whitepaper,
// Figure 6): host interface (PCIe Gen 5) along the north edge, NVLink along the south edge, memory
// controllers + HBM3 PHYs along the west and east edges (one side per three HBM stacks), two rows of
// four GPCs, and the L2 cache (two partitions) in the band between them. It is an illustration of that
// block diagram on silicon, not a die-shot trace.
import * as THREE from 'three';
import { mulberry32 } from '../engine/util';

/** Die-local layout in normalised coordinates (u across x, v across z; v = 0 is the north edge). */
export const DIE_LAYOUT = {
  host: { u0: 0.08, u1: 0.92, v0: 0.0, v1: 0.055 },     // PCIe Gen 5 host interface + GigaThread engine
  nvlink: { u0: 0.08, u1: 0.92, v0: 0.945, v1: 1.0 },   // 18 NVLink PHYs + high-speed hub
  mcW: { u0: 0.0, u1: 0.075, v0: 0.0, v1: 1.0 },         // memory controllers + HBM3 PHYs, west
  mcE: { u0: 0.925, u1: 1.0, v0: 0.0, v1: 1.0 },         // ... east
  core: { u0: 0.085, u1: 0.915 },
  gpcRow: [{ v0: 0.065, v1: 0.425 }, { v0: 0.575, v1: 0.935 }],
  l2: { v0: 0.44, v1: 0.56, gap: 0.03 },                 // two partitions, crossbar in the middle gap
};

/** GPC g (0..7): row = g >> 2 (0 north, 1 south), col = g & 3 (west → east). SMs 18 per GPC. */
export function gpcRect(g: number) {
  const { core, gpcRow } = DIE_LAYOUT;
  const col = g & 3, row = g >> 2;
  const w = (core.u1 - core.u0) / 4, gap = 0.006;
  return { u0: core.u0 + col * w + gap, u1: core.u0 + (col + 1) * w - gap, v0: gpcRow[row]!.v0, v1: gpcRow[row]!.v1 };
}

/**
 * SM site s (0..17) inside GPC g: 9 TPCs as 3 columns × 3 TPC rows, each TPC two SMs stacked (north,
 * south). A thin "GPC hub" strip sits on the GPC's edge that faces the L2 band.
 */
export function smRect(g: number, s: number) {
  const r = gpcRect(g);
  const row = g >> 2;
  const hub = 0.05 * (r.v1 - r.v0);
  // hub on the L2 side: south edge for the north row, north edge for the south row
  const v0 = row === 0 ? r.v0 : r.v0 + hub, v1 = row === 0 ? r.v1 - hub : r.v1;
  const tpc = s >> 1, sub = s & 1;
  const col = tpc % 3, trow = Math.floor(tpc / 3);
  const cw = (r.u1 - r.u0) / 3, rh = (v1 - v0) / 6;
  const gu = 0.1 * cw, gv = 0.09 * rh;
  const ri = trow * 2 + sub;
  return { u0: r.u0 + col * cw + gu * 0.5, u1: r.u0 + (col + 1) * cw - gu * 0.5, v0: v0 + ri * rh + gv * 0.5, v1: v0 + (ri + 1) * rh - gv * 0.5 };
}

type Ctx = CanvasRenderingContext2D;

function fineRows(c: Ctx, x0: number, y0: number, x1: number, y1: number, pitch: number, a: number, rnd: () => number) {
  // standard-cell rows: fine horizontal lines broken into cells of random length
  c.strokeStyle = `rgba(156,151,143,${a})`;
  c.lineWidth = 1;
  c.beginPath();
  for (let y = y0 + pitch * 0.5; y < y1; y += pitch) {
    let x = x0;
    while (x < x1) {
      const len = 6 + rnd() * 40;
      const e = Math.min(x1, x + len);
      if (rnd() > 0.12) { c.moveTo(x, y + 0.5); c.lineTo(e, y + 0.5); }
      x = e + 1.5 + rnd() * 3;
    }
  }
  c.stroke();
}

function sram(c: Ctx, x0: number, y0: number, x1: number, y1: number, cols: number, rows: number, a: number) {
  // SRAM macro grid: bank outlines + a dense bitcell texture
  const cw = (x1 - x0) / cols, rh = (y1 - y0) / rows;
  for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
    const bx = x0 + i * cw + 1.5, by = y0 + j * rh + 1.5, bw = cw - 3, bh = rh - 3;
    c.fillStyle = `rgba(94,91,87,${a * 0.35})`;
    c.fillRect(bx, by, bw, bh);
    c.strokeStyle = `rgba(156,151,143,${a * 0.9})`;
    c.lineWidth = 1;
    c.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
    // bitlines
    c.strokeStyle = `rgba(156,151,143,${a * 0.35})`;
    c.beginPath();
    for (let x = bx + 3; x < bx + bw - 1; x += 3) { c.moveTo(x + 0.5, by + 2); c.lineTo(x + 0.5, by + bh - 2); }
    c.stroke();
  }
}

function phyLanes(c: Ctx, x0: number, y0: number, x1: number, y1: number, vertical: boolean, a: number) {
  c.strokeStyle = `rgba(156,151,143,${a})`;
  c.lineWidth = 1;
  c.beginPath();
  if (vertical) for (let x = x0 + 2; x < x1 - 1; x += 4) { c.moveTo(x + 0.5, y0 + 2); c.lineTo(x + 0.5, y1 - 2); }
  else for (let y = y0 + 2; y < y1 - 1; y += 4) { c.moveTo(x0 + 2, y + 0.5); c.lineTo(x1 - 2, y + 0.5); }
  c.stroke();
}

/** One SM site at texture resolution: 4 partitions (2×2) + an L1/shared band + an L1-I strip. */
function smTile(c: Ctx, x0: number, y0: number, x1: number, y1: number, enabled: boolean, rnd: () => number) {
  const w = x1 - x0, h = y1 - y0;
  c.fillStyle = enabled ? 'rgba(30,30,33,1)' : 'rgba(20,20,22,1)';
  c.fillRect(x0, y0, w, h);
  c.strokeStyle = `rgba(238,233,223,${enabled ? 0.5 : 0.22})`;
  c.lineWidth = 1.5;
  c.strokeRect(x0 + 0.75, y0 + 0.75, w - 1.5, h - 1.5);
  if (!enabled) {
    // fused off: diagonal hatch, no detail
    c.save();
    c.beginPath(); c.rect(x0, y0, w, h); c.clip();
    c.strokeStyle = 'rgba(94,91,87,0.55)'; c.lineWidth = 1;
    c.beginPath();
    for (let k = -h; k < w; k += 7) { c.moveTo(x0 + k, y1); c.lineTo(x0 + k + h, y0); }
    c.stroke();
    c.restore();
    return;
  }
  const icache = y0 + h * 0.07, l1 = y0 + h * 0.8;
  sram(c, x0 + w * 0.06, y0 + h * 0.015, x1 - w * 0.06, icache - h * 0.01, 6, 1, 0.5);
  sram(c, x0 + w * 0.05, l1 + h * 0.02, x0 + w * 0.72, y1 - h * 0.03, 8, 2, 0.6);   // L1 / shared memory
  fineRows(c, x0 + w * 0.75, l1 + h * 0.02, x1 - w * 0.05, y1 - h * 0.03, 3, 0.35, rnd); // TMA / tex
  for (let p = 0; p < 4; p++) {
    const px0 = x0 + (p & 1 ? w * 0.51 : w * 0.04), px1 = px0 + w * 0.45;
    const py0 = icache + (p >> 1 ? (l1 - icache) * 0.51 : (l1 - icache) * 0.02), py1 = py0 + (l1 - icache) * 0.47;
    c.strokeStyle = 'rgba(156,151,143,0.55)';
    c.lineWidth = 1;
    c.strokeRect(px0 + 0.5, py0 + 0.5, px1 - px0 - 1, py1 - py0 - 1);
    const ph = py1 - py0, pw = px1 - px0;
    sram(c, px0 + pw * 0.05, py0 + ph * 0.18, px1 - pw * 0.05, py0 + ph * 0.42, 4, 1, 0.55);  // register file
    fineRows(c, px0 + pw * 0.05, py0 + ph * 0.04, px1 - pw * 0.05, py0 + ph * 0.16, 2.5, 0.4, rnd); // scheduler
    fineRows(c, px0 + pw * 0.05, py0 + ph * 0.45, px1 - pw * 0.05, py0 + ph * 0.72, 2.5, 0.45, rnd); // lanes
    // tensor core: a denser block
    c.fillStyle = 'rgba(94,91,87,0.32)';
    c.fillRect(px0 + pw * 0.05, py0 + ph * 0.75, pw * 0.5, ph * 0.21);
    fineRows(c, px0 + pw * 0.05, py0 + ph * 0.75, px0 + pw * 0.55, py0 + ph * 0.96, 2, 0.55, rnd);
    fineRows(c, px0 + pw * 0.6, py0 + ph * 0.75, px1 - pw * 0.05, py0 + ph * 0.96, 3, 0.35, rnd);
  }
}

export interface DieTex { map: THREE.CanvasTexture; emissive: THREE.CanvasTexture }

/** The die surface, `size` px square (4096 default ≈ 144 px/mm). */
export function makeDieTexture(size = 4096, enabled: (g: number, s: number) => boolean = () => true): DieTex {
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const c = cv.getContext('2d')!;
  const rnd = mulberry32(0x6100);
  const X = (u: number) => u * size, Y = (v: number) => v * size;
  // silicon: a dark field with faint large-scale variation
  c.fillStyle = '#121214';
  c.fillRect(0, 0, size, size);
  // global power grid: a very faint lattice everywhere
  c.strokeStyle = 'rgba(94,91,87,0.16)';
  c.lineWidth = 1;
  c.beginPath();
  for (let x = 0; x < size; x += 24) { c.moveTo(x + 0.5, 0); c.lineTo(x + 0.5, size); }
  for (let y = 0; y < size; y += 24) { c.moveTo(0, y + 0.5); c.lineTo(size, y + 0.5); }
  c.stroke();
  const L = DIE_LAYOUT;
  // memory controllers + HBM3 PHYs (west / east): 6 per side, two per HBM stack
  for (const side of [L.mcW, L.mcE]) {
    const x0 = X(side.u0) + 6, x1 = X(side.u1) - 6;
    for (let k = 0; k < 6; k++) {
      const y0 = Y(0.02 + k * (0.96 / 6)) + 5, y1 = Y(0.02 + (k + 1) * (0.96 / 6)) - 5;
      c.fillStyle = 'rgba(40,40,43,1)'; c.fillRect(x0, y0, x1 - x0, y1 - y0);
      c.strokeStyle = 'rgba(238,233,223,0.4)'; c.lineWidth = 1.5; c.strokeRect(x0 + 0.75, y0 + 0.75, x1 - x0 - 1.5, y1 - y0 - 1.5);
      // PHY: dense lanes on the outer half, controller logic on the inner half
      const mid = (x0 + x1) / 2;
      const outer0 = side === L.mcW ? x0 : mid, outer1 = side === L.mcW ? mid : x1;
      const inner0 = side === L.mcW ? mid : x0, inner1 = side === L.mcW ? x1 : mid;
      phyLanes(c, outer0, y0, outer1, y1, false, 0.5);
      fineRows(c, inner0 + 3, y0 + 3, inner1 - 3, y1 - 3, 3, 0.35, rnd);
    }
  }
  // host interface (north) and NVLink (south)
  {
    const r = L.host;
    const x0 = X(r.u0), x1 = X(r.u1), y0 = Y(r.v0) + 8, y1 = Y(r.v1) - 4;
    const pcie0 = x0, pcie1 = x0 + (x1 - x0) * 0.3;
    c.fillStyle = 'rgba(40,40,43,1)'; c.fillRect(x0, y0, x1 - x0, y1 - y0);
    c.strokeStyle = 'rgba(238,233,223,0.4)'; c.lineWidth = 1.5; c.strokeRect(x0 + 0.75, y0 + 0.75, x1 - x0 - 1.5, y1 - y0 - 1.5);
    phyLanes(c, pcie0 + 4, y0, pcie1, y1, true, 0.5);
    fineRows(c, pcie1 + 6, y0 + 4, x1 - 6, y1 - 4, 3, 0.35, rnd);
  }
  {
    const r = L.nvlink;
    const x0 = X(r.u0), x1 = X(r.u1), y0 = Y(r.v0) + 4, y1 = Y(r.v1) - 8;
    const n = 18, lw = (x1 - x0) / n;
    for (let k = 0; k < n; k++) {
      const bx0 = x0 + k * lw + 3, bx1 = x0 + (k + 1) * lw - 3;
      c.fillStyle = 'rgba(40,40,43,1)'; c.fillRect(bx0, y0, bx1 - bx0, y1 - y0);
      c.strokeStyle = 'rgba(238,233,223,0.4)'; c.lineWidth = 1.5; c.strokeRect(bx0 + 0.75, y0 + 0.75, bx1 - bx0 - 1.5, y1 - y0 - 1.5);
      phyLanes(c, bx0, y0 + (y1 - y0) * 0.35, bx1, y1, true, 0.5);
    }
  }
  // L2: two partitions of SRAM banks; the crossbar in the middle gap as dense vertical wiring
  {
    const { core, l2 } = L;
    const mid = (core.u0 + core.u1) / 2;
    for (const [u0, u1] of [[core.u0, mid - l2.gap / 2], [mid + l2.gap / 2, core.u1]] as const) {
      c.fillStyle = 'rgba(36,36,39,1)'; c.fillRect(X(u0), Y(l2.v0), X(u1) - X(u0), Y(l2.v1) - Y(l2.v0));
      sram(c, X(u0) + 6, Y(l2.v0) + 6, X(u1) - 6, Y(l2.v1) - 6, 24, 4, 0.6);
      c.strokeStyle = 'rgba(238,233,223,0.45)'; c.lineWidth = 2;
      c.strokeRect(X(u0) + 1, Y(l2.v0) + 1, X(u1) - X(u0) - 2, Y(l2.v1) - Y(l2.v0) - 2);
    }
    phyLanes(c, X(mid - l2.gap / 2) + 4, Y(l2.v0 - 0.01), X(mid + l2.gap / 2) - 4, Y(l2.v1 + 0.01), true, 0.4);
    // crossbar wiring fanning into both GPC rows (faint)
    c.strokeStyle = 'rgba(156,151,143,0.12)'; c.lineWidth = 1;
    c.beginPath();
    for (let x = X(core.u0); x < X(core.u1); x += 6) {
      c.moveTo(x + 0.5, Y(l2.v0 - 0.013)); c.lineTo(x + 0.5, Y(l2.v0));
      c.moveTo(x + 0.5, Y(l2.v1)); c.lineTo(x + 0.5, Y(l2.v1 + 0.013));
    }
    c.stroke();
  }
  // GPCs and their SM sites
  for (let g = 0; g < 8; g++) {
    const r = gpcRect(g);
    c.fillStyle = 'rgba(24,24,26,1)'; c.fillRect(X(r.u0), Y(r.v0), X(r.u1) - X(r.u0), Y(r.v1) - Y(r.v0));
    // the GPC hub strip (raster/shared logic) on the L2 side
    const row = g >> 2, hub = 0.05 * (r.v1 - r.v0);
    const hv0 = row === 0 ? r.v1 - hub : r.v0, hv1 = hv0 + hub;
    fineRows(c, X(r.u0) + 6, Y(hv0) + 3, X(r.u1) - 6, Y(hv1) - 3, 3, 0.4, rnd);
    for (let s = 0; s < 18; s++) {
      const q = smRect(g, s);
      smTile(c, X(q.u0), Y(q.v0), X(q.u1), Y(q.v1), enabled(g, s), rnd);
    }
    c.strokeStyle = 'rgba(238,233,223,0.6)'; c.lineWidth = 2.5;
    c.strokeRect(X(r.u0) + 1.25, Y(r.v0) + 1.25, X(r.u1) - X(r.u0) - 2.5, Y(r.v1) - Y(r.v0) - 2.5);
  }
  // die seal ring
  c.strokeStyle = 'rgba(238,233,223,0.55)'; c.lineWidth = 6;
  c.strokeRect(6, 6, size - 12, size - 12);
  c.strokeStyle = 'rgba(156,151,143,0.35)'; c.lineWidth = 2;
  c.strokeRect(16, 16, size - 32, size - 32);

  // emissive map: the hairlines only (so the engraving reads even in shadow)
  const ev = document.createElement('canvas');
  ev.width = ev.height = size;
  const e = ev.getContext('2d')!;
  e.filter = 'contrast(3) brightness(0.9)';
  e.drawImage(cv, 0, 0);
  const map = new THREE.CanvasTexture(cv);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;
  map.generateMipmaps = true;
  map.minFilter = THREE.LinearMipmapLinearFilter;
  const emissive = new THREE.CanvasTexture(ev);
  emissive.colorSpace = THREE.SRGBColorSpace;
  emissive.anisotropy = 8;
  return { map, emissive };
}

/** PCB traces for the SXM5 board (and a dimmer version for the package substrate). */
export function makeBoardTexture(w = 2048, h = 1152, seed = 7, density = 1): THREE.CanvasTexture {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const c = cv.getContext('2d')!;
  const rnd = mulberry32(seed);
  c.fillStyle = '#101012';
  c.fillRect(0, 0, w, h);
  // routed traces: 45°/90° polylines in bundles
  c.lineCap = 'round';
  for (let b = 0; b < 140 * density; b++) {
    const n = 3 + Math.floor(rnd() * 9);
    let x = rnd() * w, y = rnd() * h;
    const dir = Math.floor(rnd() * 4);
    const segs: [number, number][] = [[x, y]];
    let d = dir;
    for (let k = 0; k < 4; k++) {
      const len = 40 + rnd() * 260;
      const dx = [1, 0, -1, 0][d]!, dy = [0, 1, 0, -1][d]!;
      x += dx * len; y += dy * len;
      segs.push([x, y]);
      if (rnd() < 0.5) { x += (dx || (rnd() < 0.5 ? 1 : -1)) * 20; y += (dy || (rnd() < 0.5 ? 1 : -1)) * 20; segs.push([x, y]); }
      d = (d + (rnd() < 0.5 ? 1 : 3)) % 4;
    }
    for (let i = 0; i < n; i++) {
      const o = i * 5;
      c.strokeStyle = `rgba(94,91,87,${0.28 + rnd() * 0.18})`;
      c.lineWidth = 1.6;
      c.beginPath();
      segs.forEach(([px, py], k) => (k ? c.lineTo(px + o, py + o) : c.moveTo(px + o, py + o)));
      c.stroke();
    }
  }
  // vias
  c.fillStyle = 'rgba(156,151,143,0.35)';
  for (let k = 0; k < 2600 * density; k++) {
    const x = rnd() * w, y = rnd() * h;
    c.beginPath(); c.arc(x, y, 1.6, 0, Math.PI * 2); c.fill();
  }
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

/** SRAM-grid texture for the top of register files / L1 blocks in the SM close-up. */
export function makeSramTexture(size = 512): THREE.CanvasTexture {
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const c = cv.getContext('2d')!;
  c.fillStyle = '#18181a';
  c.fillRect(0, 0, size, size);
  sram(c, 4, 4, size - 4, size - 4, 8, 8, 0.9);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}
