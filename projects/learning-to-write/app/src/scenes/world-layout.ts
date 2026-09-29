// Where everything sits in the one world the camera flies over (logical px at zoom 1), per aspect.
// 16:9: the page in the middle, the weights to its left, the book / the guess / the surprise to its right,
// the loss curve beyond them. 9:16: the same pieces stacked top to bottom.
import { W, H } from '../engine/gl';

export interface Rect { x: number; y: number; w: number; h: number }
export type PanelId = 'title' | 'page' | 'weights' | 'book' | 'guess' | 'surprise' | 'curve' | 'ring' | 'next' | 'all';

const LAND: Record<Exclude<PanelId, 'all'>, Rect> = {
  title: { x: 0, y: -1180, w: 1800, h: 190 },
  page: { x: 0, y: 0, w: 940, h: 1000 },
  ring: { x: 0, y: 20, w: 1560, h: 1420 },
  weights: { x: -1520, y: 0, w: 720, h: 1000 },
  book: { x: 1620, y: -640, w: 1040, h: 360 },
  guess: { x: 1620, y: -50, w: 1040, h: 680 },
  surprise: { x: 1620, y: 560, w: 1040, h: 480 },
  curve: { x: 3230, y: -40, w: 1500, h: 1100 },
  next: { x: 0, y: 1500, w: 1400, h: 700 },
};

const PORT: Record<Exclude<PanelId, 'all'>, Rect> = {
  title: { x: 0, y: -3000, w: 960, h: 420 },
  page: { x: 0, y: 0, w: 940, h: 1000 },
  ring: { x: 0, y: 20, w: 1180, h: 1500 },
  weights: { x: 0, y: -1560, w: 760, h: 1000 },
  book: { x: 0, y: 1130, w: 900, h: 460 },
  guess: { x: 0, y: 1780, w: 920, h: 720 },
  surprise: { x: 0, y: 2440, w: 920, h: 520 },
  curve: { x: 0, y: 3330, w: 920, h: 1100 },
  next: { x: 0, y: 4330, w: 920, h: 900 },
};

export const portrait = () => H > W;
export function rect(id: PanelId): Rect {
  const L = portrait() ? PORT : LAND;
  if (id === 'all') return union(portrait() ? ['weights', 'page', 'book', 'guess'] : ['weights', 'page', 'book', 'guess', 'surprise', 'curve']);
  return L[id];
}
export function union(ids: PanelId[]): Rect {
  const rs = ids.map(rect);
  const x0 = Math.min(...rs.map((r) => r.x - r.w / 2)), x1 = Math.max(...rs.map((r) => r.x + r.w / 2));
  const y0 = Math.min(...rs.map((r) => r.y - r.h / 2)), y1 = Math.max(...rs.map((r) => r.y + r.h / 2));
  return { x: (x0 + x1) / 2, y: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0 };
}

export interface Cam { x: number; y: number; z: number }

/** The camera that fits rect `r` in the frame below the sung-line strip (with `pad` of air around it). */
export function frame(r: Rect, pad = 1.08): Cam {
  const top = portrait() ? 230 : 150, bottom = portrait() ? 60 : 40;
  const aw = W * 0.94, ah = H - top - bottom;
  const z = Math.min(aw / (r.w * pad), ah / (r.h * pad));
  // the rect's centre sits in the middle of the free area
  const cy = top + ah / 2;
  return { x: r.x, y: r.y - (cy - H / 2) / z, z };
}

/** World -> screen transform for a camera. */
export function camMatrix(c: Cam): [number, number, number, number, number, number] {
  return [c.z, 0, 0, c.z, W / 2 - c.x * c.z, H / 2 - c.y * c.z];
}
