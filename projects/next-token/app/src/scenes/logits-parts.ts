// Parts for the logits and append scenes: the vocabulary ribbon (a printed strip of 128,256 slots,
// drawn by a shader so every slot exists without 128k meshes), the spiral "draw", the temperature
// knob, temperature re-weighting of the real top 10, and the two-ink slab type / knockout labels.
import * as THREE from 'three';
import { HEX, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { W, H } from '../engine/gl';
import { INK, inked, flatMat, type InkName } from './_kit';

const PINK = rgba('pink'), BLUE = rgba('blue'), INKC = rgba('ink'), PAPER = rgba('paper');
export const C = { PINK, BLUE, INK: INKC, PAPER };

// ------------------------------------------------------------------ numbers

/** Probabilities re-weighted by temperature over a fixed candidate set: p_T ∝ p^(1/T), renormalised. */
export function tempProbs(p: number[], T: number): number[] {
  const lp = p.map((x) => Math.log(Math.max(x, 1e-30)));
  const m = Math.max(...lp);
  const w = lp.map((x) => Math.exp((x - m) / T));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((x) => x / s);
}

export const fmtInt = (n: number) => Math.round(n).toLocaleString('en-US');
export const pct = (p: number, d = 1) => `${(p * 100).toFixed(d)}%`;

// ------------------------------------------------------------------ the vocabulary ribbon

const RIBBON_VERT = /* glsl */ `
attribute float aSlot;
varying float vSlot;
varying float vY;
varying float vDist;
void main() {
  vSlot = aSlot;
  vY = uv.y;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vDist = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;
const RIBBON_FRAG = /* glsl */ `
uniform vec3 paper, blue, pink, ink;
uniform float tickH, nSlots, podium, fogNear, fogFar, markEvery, band, hot;
varying float vSlot;
varying float vY;
varying float vDist;
void main() {
  float s = vSlot;
  if (s > nSlots) discard;
  float fw = max(fwidth(s), 1e-5);
  float fy = max(fwidth(vY), 1e-5);
  float fr = fract(s);
  // how much of the slot pattern is below a pixel: fade the pattern into its average ink
  float dens = clamp(fw * 1.2 - 0.25, 0.0, 1.0);
  float sep = 1.0 - smoothstep(0.05, 0.05 + fw, min(fr, 1.0 - fr));
  sep = mix(sep, 0.1, dens);
  float tk = 1.0 - smoothstep(0.24, 0.24 + fw, abs(fr - 0.5));
  tk = mix(tk, 0.48, dens);
  float inTick = 1.0 - smoothstep(tickH, tickH + fy * 1.5, vY);
  float tick = s >= podium ? tk * inTick : 0.0;
  // a pink marker every markEvery slots
  float m = mod(s, markEvery);
  float wm = max(2.0, fw * 1.5);
  float mk = s > podium + 1.0 ? 1.0 - smoothstep(wm, wm + fw, min(m, markEvery - m)) : 0.0;
  mk *= step(vY, 0.8);
  // rails: black along the bottom, blue along the top
  float rail = 1.0 - smoothstep(0.045, 0.045 + fy, vY);
  float topr = smoothstep(0.955 - fy, 0.955, vY);
  // the last slots: a pink end cap
  float cap = step(nSlots - max(80.0, fw * 10.0), s);
  // the podium (top 10): hot pink wash behind the bars when lit
  float pod = s < podium ? hot * 0.35 : 0.0;
  float a = clamp(max(max(sep * 0.9, tick), band) + topr, 0.0, 1.0);
  float b = clamp(mk + cap + pod, 0.0, 1.0);
  vec3 col = paper * mix(vec3(1.0), blue / paper, a) * mix(vec3(1.0), pink / paper, b);
  col = mix(col, ink, rail);
  col = mix(col, paper, smoothstep(fogNear, fogFar, vDist));
  gl_FragColor = vec4(col, 1.0);
}`;

export function ribbonMat(o: { nSlots: number; podium?: number; tickH?: number; markEvery?: number; band?: number; fogNear?: number; fogFar?: number }): THREE.ShaderMaterial {
  const v = (k: InkName) => new THREE.Vector3(...INK[k].map((x) => Math.max(x, 0.03)));
  return new THREE.ShaderMaterial({
    vertexShader: RIBBON_VERT, fragmentShader: RIBBON_FRAG, side: THREE.DoubleSide,
    uniforms: {
      paper: { value: new THREE.Vector3(...INK.paper) }, blue: { value: v('blue') }, pink: { value: v('pink') }, ink: { value: v('ink') },
      tickH: { value: o.tickH ?? 0.18 }, nSlots: { value: o.nSlots }, podium: { value: o.podium ?? 10 },
      fogNear: { value: o.fogNear ?? 70 }, fogFar: { value: o.fogFar ?? 190 }, markEvery: { value: o.markEvery ?? 10000 },
      band: { value: o.band ?? 0.12 }, hot: { value: 0 },
    },
  });
}

/**
 * A standing strip (in the xy plane at z, from y0 to y0+h) made of pieces [x0, x1] carrying slots
 * [s0, s1] each (so the top 10 can be wide and the other 128,246 thin).
 */
export function stripGeo(pieces: { x0: number; x1: number; s0: number; s1: number }[], y0: number, h: number, z: number): THREE.BufferGeometry {
  const pos: number[] = [], uv: number[] = [], slot: number[] = [], idx: number[] = [];
  pieces.forEach((p, i) => {
    const b = i * 4;
    pos.push(p.x0, y0, z, p.x1, y0, z, p.x0, y0 + h, z, p.x1, y0 + h, z);
    uv.push(0, 0, 1, 0, 0, 1, 1, 1);
    slot.push(p.s0, p.s1, p.s0, p.s1);
    idx.push(b, b + 1, b + 2, b + 2, b + 1, b + 3);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('aSlot', new THREE.Float32BufferAttribute(slot, 1));
  g.setIndex(idx);
  return g;
}

/** A flat spiral band on the xz plane (y = 0), slots s0..s1 from the inside out. */
export function spiralGeo(r0: number, pitch: number, turns: number, w: number, s0: number, s1: number, n = 3000): THREE.BufferGeometry {
  const pos: number[] = [], uv: number[] = [], slot: number[] = [], idx: number[] = [];
  for (let i = 0; i <= n; i++) {
    const u = i / n, th = u * turns * Math.PI * 2, r = r0 + pitch * (th / (Math.PI * 2));
    const c = Math.cos(th), s = Math.sin(th);
    pos.push(c * (r - w / 2), 0, s * (r - w / 2), c * (r + w / 2), 0, s * (r + w / 2));
    uv.push(u, 0, u, 1);
    const k = s0 + u * (s1 - s0);
    slot.push(k, k);
    if (i < n) { const b = i * 2; idx.push(b, b + 1, b + 2, b + 2, b + 1, b + 3); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('aSlot', new THREE.Float32BufferAttribute(slot, 1));
  g.setIndex(idx);
  return g;
}

// ------------------------------------------------------------------ the temperature knob

export const KNOB_T0 = 0.2, KNOB_T1 = 1.5;
/** Clockwise angle from 12 o'clock for temperature T (0.2 at -135°, 1.5 at +135°). */
export const knobAngle = (T: number) => ((-135 + ((T - KNOB_T0) / (KNOB_T1 - KNOB_T0)) * 270) * Math.PI) / 180;

function dialTexture(): THREE.CanvasTexture {
  const S = 1024, cv = document.createElement('canvas');
  cv.width = S; cv.height = S;
  const c = cv.getContext('2d')!;
  const R = S / 2;
  c.fillStyle = HEX.paper;
  c.beginPath(); c.arc(R, R, R, 0, Math.PI * 2); c.fill();
  c.lineWidth = 16; c.strokeStyle = HEX.ink;
  c.beginPath(); c.arc(R, R, R - 10, 0, Math.PI * 2); c.stroke();
  // the arc of the scale: blue from SAFE to the real setting, pink beyond
  const toCanvas = (a: number) => a - Math.PI / 2;
  c.lineWidth = 26;
  c.strokeStyle = HEX.blue;
  c.beginPath(); c.arc(R, R, R - 62, toCanvas(knobAngle(0.2)), toCanvas(knobAngle(0.6))); c.stroke();
  c.strokeStyle = HEX.pink;
  c.beginPath(); c.arc(R, R, R - 62, toCanvas(knobAngle(0.6)), toCanvas(knobAngle(1.5))); c.stroke();
  // ticks every 0.1, majors labelled
  for (let i = 0; i <= 13; i++) {
    const T = 0.2 + i * 0.1, a = knobAngle(T);
    const major = [0.2, 0.6, 1.0, 1.5].some((x) => Math.abs(x - T) < 1e-6) || i === 13;
    const r0 = R - (major ? 120 : 96), r1 = R - 30;
    c.strokeStyle = HEX.ink; c.lineWidth = major ? 14 : 8;
    c.beginPath(); c.moveTo(R + Math.sin(a) * r0, R - Math.cos(a) * r0); c.lineTo(R + Math.sin(a) * r1, R - Math.cos(a) * r1); c.stroke();
  }
  c.textAlign = 'center'; c.textBaseline = 'middle';
  for (const T of [0.2, 0.6, 1.0, 1.5]) {
    const a = knobAngle(T), r = R - 175;
    c.fillStyle = T === 0.6 ? HEX.pink : HEX.ink;
    c.font = font(F.mono(700), 66);
    c.fillText(T.toFixed(1), R + Math.sin(a) * r, R - Math.cos(a) * r);
  }
  c.font = font(F.archivo(112, 900), 70);
  c.fillStyle = HEX.blue;
  c.fillText('SAFE', R - 205, R + 330);
  c.fillStyle = HEX.pink;
  c.fillText('BET', R + 215, R + 330);
  const tx = new THREE.CanvasTexture(cv);
  tx.colorSpace = THREE.SRGBColorSpace;
  tx.anisotropy = 8;
  return tx;
}

/** The temperature knob: a printed dial plate and a blue knob with a pink pointer. Face = +z. */
export class Knob {
  group = new THREE.Group();
  body = new THREE.Group();
  constructor() {
    const plate = new THREE.Mesh(new THREE.CircleGeometry(1.7, 96), new THREE.MeshBasicMaterial({ map: dialTexture() }));
    this.group.add(plate);
    const rim = inked(new THREE.CylinderGeometry(1.7, 1.7, 0.22, 96, 1, true), { ink: 'ink', lit: 0.4, shade: 1, side: THREE.DoubleSide }, 2.4);
    rim.rotation.x = Math.PI / 2; rim.position.z = -0.1;
    this.group.add(rim);
    const knob = inked(new THREE.CylinderGeometry(0.66, 0.76, 0.55, 48), { ink: 'blue', lit: 0.45, shade: 1, over: 'pink', overShade: 0.3 }, 2.6);
    knob.rotation.x = Math.PI / 2; knob.position.z = 0.3;
    this.body.add(knob);
    const ptr = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.62, 0.1), flatMat('pink'));
    ptr.position.set(0, 0.34, 0.58);
    this.body.add(ptr);
    const cap = new THREE.Mesh(new THREE.CircleGeometry(0.2, 32), flatMat('ink'));
    cap.position.z = 0.585;
    this.body.add(cap);
    this.group.add(this.body);
  }
  set(T: number) { this.body.rotation.z = -knobAngle(T); }
}

// ------------------------------------------------------------------ type

/** Big two-ink type: blue underprint, pink on top with multiply (prints purple where they overlap). */
export function slab(c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, k = 1, align: CanvasTextAlign = 'center', maxW = W * 0.9) {
  c.save();
  const fam = F.archivo(125, 900);
  c.font = font(fam, size);
  const w0 = c.measureText(text).width;
  if (w0 > maxW) { size *= maxW / w0; c.font = font(fam, size); }
  c.textAlign = align;
  c.textBaseline = 'middle';
  c.translate(x, y);
  const s = 1.25 - 0.25 * k;
  c.scale(s, s);
  c.fillStyle = BLUE;
  c.fillText(text, size * 0.035, size * 0.03);
  c.globalCompositeOperation = 'multiply';
  c.fillStyle = PINK;
  c.fillText(text, 0, 0);
  c.restore();
}

/**
 * A solid-ink label (mono, >= 22 px, weight >= 600), optionally on a knockout box so it reads over
 * anything. Returns its width.
 */
export function label(c: CanvasRenderingContext2D, text: string, x: number, y: number, o: { size?: number; color?: string; box?: string | null; align?: CanvasTextAlign; weight?: number; archivo?: boolean; pad?: number } = {}): number {
  const size = Math.max(22, o.size ?? 26);
  c.save();
  c.font = o.archivo ? font(F.archivo(112, 900), size) : font(F.mono(Math.max(600, o.weight ?? 700)), size);
  c.textBaseline = 'middle';
  const w = c.measureText(text).width;
  const align = o.align ?? 'left';
  const x0 = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  const pad = o.pad ?? Math.round(size * 0.32);
  if (o.box !== null && o.box !== undefined) {
    c.fillStyle = o.box;
    c.fillRect(x0 - pad, y - size * 0.62, w + pad * 2, size * 1.24);
  }
  c.fillStyle = o.color ?? INKC;
  c.textAlign = 'left';
  c.fillText(text, x0, y + size * 0.04);
  c.restore();
  return w;
}

/** Project a world point with a camera to logical px (null when behind the camera). */
export function proj(cam: THREE.Camera, v: THREE.Vector3): [number, number] | null {
  const p = v.clone().project(cam);
  if (p.z > 1) return null;
  return [(p.x + 1) / 2 * W, (1 - p.y) / 2 * H];
}

/** World position at z = 0 that a camera at (0, 0, dist) looking down -z (fov deg) shows at screen px (sx, sy). */
export function screenToPlane(sx: number, sy: number, dist: number, fov: number): [number, number] {
  const th = Math.tan((fov * Math.PI) / 360) * dist;
  return [((sx / W) * 2 - 1) * th * (W / H), (1 - (sy / H) * 2) * th];
}
