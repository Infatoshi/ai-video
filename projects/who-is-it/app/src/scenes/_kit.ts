// The density kit: what every scene uses, so the whole video holds one standard (SPEC.md "Rules").
//  - shots: beat-locked hard cuts inside a scene, at the section's rate (a new shot every bar in the
//    verses, every 2 beats in the choruses, every beat on the drop), chosen by the frame's own time
//  - ink materials for 3D: shading is ink coverage (lit = a light tint, shadow = full ink, an optional
//    second ink overprints the shadow side), plus constant-width black outlines
//  - background and foreground fields: halftone gradient washes, token rain, flying ink specks
//  - token tiles: printed slabs carrying a real token's text
// Everything is a pure function of time.
import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { AudioData } from '../engine/audio';
import type { Frame, SceneCtx } from '../engine/scene';
import { FSPass, W, H } from '../engine/gl';
import { HEX, LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, hash, mulberry32 } from '../engine/util';
import { showTok } from '../engine/attn';

export type InkName = 'pink' | 'blue' | 'ink' | 'paper';
export const INK = { pink: LIN.pink, blue: LIN.blue, ink: LIN.ink, paper: LIN.paper } as const;
export const ink3 = (k: InkName) => new THREE.Color().setRGB(...INK[k]);

// ------------------------------------------------------------------ shots

/** Beats per shot by section: the edit gets faster as the song does. */
export const CUT_EVERY: Record<string, number> = {
  intro: 4, verse1: 4, chorus1: 2, verse2: 4, chorus2: 2, verse3: 4, breakdown: 4, drop: 1, verse4: 2, chorus3: 2, outro: 1,
};

export interface Shot {
  /** Shot number within the scene (0, 1, 2 ...). */
  i: number;
  /** Shot window (song s) and 0..1 progress through it (from the sub-frame time: animate with it). */
  t0: number;
  t1: number;
  u: number;
  /** Seconds into the shot. */
  lt: number;
  /** A stable random seed for this shot; pass `reg: shot.seed` in the post overrides so each shot is a new print. */
  seed: number;
}

/**
 * The shot the frame is in. Cuts land on beats, every `every` beats (default: the section's rate from
 * CUT_EVERY), counted from `from` (default the scene's cut, i.e. its start without the transition overlap). The shot is picked with
 * the frame's own time (f.ft), so a cut never blends two shots in one motion-blurred frame.
 */
export function shot(ctx: SceneCtx, f: Frame, opts: { every?: number; from?: number } = {}): Shot {
  const au = ctx.audio;
  const b0 = Math.round(au.beatAt(opts.from ?? ctx.cut ?? ctx.start));
  // cut points (beat indices) from b0 on, each step the rate of the section it starts in, so a scene that
  // spans a section change keeps counting up without a jump
  const key = `${b0}:${opts.every ?? 'auto'}`;
  let cuts = cutCache.get(au)?.get(key);
  if (!cuts) {
    cuts = [b0];
    const last = au.beatAt(au.duration) + 8;
    while (cuts[cuts.length - 1]! < last) {
      const c = cuts[cuts.length - 1]!;
      cuts.push(c + (opts.every ?? CUT_EVERY[au.section(au.timeOfBeat(c) + 0.01)?.name ?? ''] ?? 4));
    }
    let m = cutCache.get(au);
    if (!m) { m = new Map(); cutCache.set(au, m); }
    m.set(key, cuts);
  }
  const bf = au.beatAt(f.ft) + 1e-3;
  let i = 0;
  while (i + 1 < cuts.length && cuts[i + 1]! <= bf) i++;
  const t0 = au.timeOfBeat(cuts[i]!), t1 = au.timeOfBeat(cuts[i + 1] ?? cuts[i]! + 4);
  return { i, t0, t1, u: clamp((f.t - t0) / Math.max(1e-3, t1 - t0)), lt: f.t - t0, seed: Math.floor(hash(ctx.start * 13.1, i) * 1000) };
}
const cutCache = new WeakMap<AudioData, Map<string, number[]>>();

/** Cuts at explicit times (song s, e.g. lyric words): index of the last one at or before the frame. */
export function shotAtTimes(f: Frame, times: number[]): { i: number; t0: number; t1: number; lt: number } {
  let i = -1;
  while (i + 1 < times.length && times[i + 1]! <= f.ft + 1e-4) i++;
  const t0 = times[i] ?? -Infinity, t1 = times[i + 1] ?? Infinity;
  return { i, t0, t1, lt: f.t - t0 };
}

/** A fast snap-in that lands on time `at` and holds: 0 before, eases to 1 over `dur` (~5 frames). */
export const snapIn = (t: number, at: number, dur = 0.08) => (t < at ? 0 : ease.outExpo(clamp((t - at) / dur)));

// ------------------------------------------------------------------ ink materials

const INK_VERT = /* glsl */ `
#include <clipping_planes_pars_vertex>
varying vec3 vN;
varying vec3 vV;
varying vec3 vIC;
void main() {
  vec4 p = vec4(position, 1.0);
  vec3 nrm = normal;
#ifdef USE_INSTANCING
  p = instanceMatrix * p;
  nrm = mat3(instanceMatrix) * nrm;
#endif
  vIC = vec3(0.0, 1.0, 0.0);
#ifdef USE_INSTANCING_COLOR
  vIC = instanceColor;
#endif
  vec4 mv = modelViewMatrix * p;
  vec4 mvPosition = mv;
  #include <clipping_planes_vertex>
  vN = normalize(normalMatrix * nrm);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;
const INK_FRAG = /* glsl */ `
#include <clipping_planes_pars_fragment>
uniform vec3 paper, inkA, inkB;
uniform float litA, shadeA, litB, shadeB, rim, hot;
varying vec3 vN;
varying vec3 vV;
varying vec3 vIC; // per instance: r = extra coverage of the second ink (a hit), g = main ink scale
void main() {
  #include <clipping_planes_fragment>
  vec3 n = normalize(vN);
  if (!gl_FrontFacing) n = -n;
  vec3 L = normalize(vec3(-0.45, 0.75, 0.55));           // key light fixed to the camera: every scene lit alike
  float ndl = dot(n, L) * 0.5 + 0.5;                        // half-lambert
  float sh = 1.0 - ndl;
  float a = mix(litA, shadeA, smoothstep(0.1, 0.85, sh));
  float b = mix(litB, shadeB, smoothstep(0.45, 0.95, sh));
  float r = pow(1.0 - abs(dot(n, normalize(vV))), 3.0) * rim; // rim of the second ink
  b = max(max(b, r), max(vIC.r, hot));
  a *= vIC.g;
  vec3 col = paper * mix(vec3(1.0), inkA / paper, clamp(a, 0.0, 1.0)) * mix(vec3(1.0), inkB / paper, clamp(b, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
}`;

export interface InkOpts {
  /** Main ink and its coverage on the lit side / in shadow (0..1). */
  ink?: InkName;
  lit?: number;
  shade?: number;
  /** Second ink overprinting the shadow side (and the rim). Set `uniforms.hot.value` (0..1) to flood a mesh with it. */
  over?: InkName;
  overLit?: number;
  overShade?: number;
  rim?: number;
  side?: THREE.Side;
}

/** A 3D surface printed in ink: coverage follows the light (tint when lit, solid in shadow). */
export function inkMat(o: InkOpts = {}): THREE.ShaderMaterial {
  const v = (k: InkName) => new THREE.Vector3(...INK[k].map((x) => Math.max(x, 0.03)));
  return new THREE.ShaderMaterial({
    vertexShader: INK_VERT, fragmentShader: INK_FRAG, side: o.side ?? THREE.FrontSide,
    uniforms: {
      paper: { value: new THREE.Vector3(...INK.paper) }, inkA: { value: v(o.ink ?? 'blue') }, inkB: { value: v(o.over ?? 'pink') },
      litA: { value: o.lit ?? 0.35 }, shadeA: { value: o.shade ?? 1 }, litB: { value: o.over ? o.overLit ?? 0 : 0 },
      shadeB: { value: o.over ? o.overShade ?? 0.7 : 0 }, rim: { value: o.rim ?? 0 }, hot: { value: 0 },
    },
  });
}

/** Flat solid ink (no shading): type, markers, the hero token. */
export const flatMat = (k: InkName, side: THREE.Side = THREE.FrontSide) => new THREE.MeshBasicMaterial({ color: ink3(k), side });

const OUTLINE_VERT = /* glsl */ `
uniform float width;
uniform vec2 res;
void main() {
  vec4 p = vec4(position, 1.0);
  vec3 nrm = normal;
#ifdef USE_INSTANCING
  p = instanceMatrix * p;
  nrm = mat3(instanceMatrix) * nrm;
#endif
  vec4 c = projectionMatrix * modelViewMatrix * p;
  vec3 nv = normalize(normalMatrix * nrm);
  vec2 nc = normalize((projectionMatrix * vec4(nv, 0.0)).xy + 1e-6);
  c.xy += nc * width * c.w * 2.0 / res;
  gl_Position = c;
}`;
const outlineMats = new Map<string, THREE.ShaderMaterial>();
/** Constant-width (logical px) black ink outline around a mesh (inverted hull on smoothed normals). */
export function outline(mesh: THREE.Mesh, width = 2.2, color: InkName = 'ink'): THREE.Mesh {
  const key = `${width}:${color}`;
  let m = outlineMats.get(key);
  if (!m) {
    m = new THREE.ShaderMaterial({
      vertexShader: OUTLINE_VERT, side: THREE.BackSide,
      fragmentShader: `uniform vec3 col; void main(){ gl_FragColor = vec4(col, 1.0); }`,
      uniforms: { width: { value: width }, res: { value: new THREE.Vector2(W, H) }, col: { value: new THREE.Vector3(...INK[color]) } },
    });
    outlineMats.set(key, m);
  }
  const g = mergeVertices(mesh.geometry.clone().deleteAttribute('uv').deleteAttribute('normal'), 1e-4);
  g.computeVertexNormals();
  const o = new THREE.Mesh(g, m);
  mesh.add(o);
  return o;
}

/** Outline for an InstancedMesh: a second instanced mesh sharing its instance matrices. */
export function outlineInstanced(inst: THREE.InstancedMesh, width = 2, color: InkName = 'ink'): THREE.InstancedMesh {
  const tmp = new THREE.Mesh(inst.geometry);
  const o = outline(tmp, width, color);
  const io = new THREE.InstancedMesh(o.geometry, o.material as THREE.Material, inst.count);
  io.instanceMatrix = inst.instanceMatrix;
  io.frustumCulled = false;
  inst.add(io);
  return io;
}

/** A mesh with the ink material and an outline. */
export function inked(geo: THREE.BufferGeometry, o: InkOpts = {}, line = 2.2): THREE.Mesh {
  const m = new THREE.Mesh(geo, inkMat(o));
  if (line > 0) outline(m, line);
  return m;
}

// ------------------------------------------------------------------ fields (background / foreground)

/**
 * Background wash: paper with two large soft blobs of ink that become halftone gradients in print. The
 * blobs move with `drift` (pass a drums-driven distance, see _cam.drive) and reseed per shot.
 */
export class Wash {
  pass = new FSPass(/* glsl */ `
    uniform vec3 c1, c2; uniform float seed, drift, amt, flood; uniform vec2 aspect;
    void main() {
      vec2 p = (vUv - 0.5) * aspect;
      vec2 a = (hash22(vec2(seed, 1.7)) - 0.5) * aspect * 0.9 + 0.12 * vec2(sin(drift * 0.7), cos(drift * 0.5));
      vec2 b = (hash22(vec2(seed, 5.3)) - 0.5) * aspect * 0.9 + 0.12 * vec2(cos(drift * 0.6), sin(drift * 0.8));
      float ka = exp(-dot(p - a, p - a) / 0.16) * amt, kb = exp(-dot(p - b, p - b) / 0.22) * amt * 0.8;
      vec3 col = C_PAPER * mix(vec3(1.0), c1 / C_PAPER, ka) * mix(vec3(1.0), c2 / C_PAPER, kb);
      col = mix(col, c1, flood);
      fragColor = vec4(col, 1.0);
    }`, {
    c1: { value: new THREE.Vector3(...INK.pink) }, c2: { value: new THREE.Vector3(...INK.blue) }, seed: { value: 0 }, drift: { value: 0 },
    amt: { value: 0.55 }, flood: { value: 0 }, aspect: { value: new THREE.Vector2(W / Math.min(W, H), H / Math.min(W, H)) },
  });
  render(r: THREE.WebGLRenderer, out: THREE.WebGLRenderTarget, o: { seed: number; drift: number; amt?: number; c1?: InkName; c2?: InkName; flood?: number }) {
    const u = this.pass.u;
    u.seed!.value = o.seed; u.drift!.value = o.drift; u.amt!.value = o.amt ?? 0.55; u.flood!.value = o.flood ?? 0;
    (u.c1!.value as THREE.Vector3).set(...INK[o.c1 ?? 'pink']);
    (u.c2!.value as THREE.Vector3).set(...INK[o.c2 ?? 'blue']);
    this.pass.render(r, out);
  }
}

/**
 * Token rain (Canvas2D): columns of real tokens (strings and ids) scrolling at a drums-driven speed.
 * Draw it first on a scene's 2D layer; solid blue, small, sparse so it reads as texture.
 */
export function tokenRain(c: CanvasRenderingContext2D, pool: { id: number; t: string }[], scroll: number, o: { cols?: number; color?: string; alpha?: number; size?: number; seed?: number } = {}) {
  const cols = o.cols ?? 11, size = o.size ?? 17, rowH = size * 2.1;
  c.save();
  c.font = font(F.mono(600), size);
  c.fillStyle = o.color ?? rgba('blue');
  c.globalAlpha = o.alpha ?? 1;
  const seed = o.seed ?? 0;
  for (let ci = 0; ci < cols; ci++) {
    const x = ((ci + 0.5) / cols) * W + (hash(ci, seed) - 0.5) * 40;
    const speed = 0.6 + hash(ci, seed + 3) * 0.9;
    const off = scroll * rowH * speed * 4 + hash(ci, seed + 7) * 1000;
    const r0 = Math.floor(off / rowH);
    for (let r = -1; r < H / rowH + 1; r++) {
      const k = r0 + r;
      if (hash(ci, k, seed) < 0.68) continue; // sparse
      const tok = pool[Math.floor(hash(ci, k, seed + 11) * pool.length)]!;
      const y = H - (r * rowH - (off - r0 * rowH));
      const label = hash(ci, k, seed + 5) < 0.5 ? String(tok.id) : showTok(tok.t);
      c.fillText(label, x, y);
    }
  }
  c.restore();
}

/**
 * Flying ink specks (foreground, Canvas2D): small squares, dashes and dots rushing past, their speed from
 * the drums. `travel` = distance so far (use _cam.drive).
 */
export function specks(c: CanvasRenderingContext2D, travel: number, o: { n?: number; seed?: number; colors?: string[]; size?: number } = {}) {
  const n = o.n ?? 70, seed = o.seed ?? 0;
  const cols = o.colors ?? [rgba('pink'), rgba('ink'), rgba('blue')];
  c.save();
  for (let i = 0; i < n; i++) {
    const r = mulberry32(i * 7919 + seed * 104729);
    const ax = r() * 2 - 1, ay = r() * 2 - 1, sp = 0.4 + r() * 1.2, kind = Math.floor(r() * 3), col = cols[Math.floor(r() * cols.length)]!;
    const z = ((travel * sp * 0.25 + r()) % 1 + 1) % 1; // 0 far .. 1 at the lens
    const pz = 0.08 + z * z * 1.6;
    const x = W / 2 + ax * W * 0.5 * pz, y = H / 2 + ay * H * 0.5 * pz;
    if (x < -40 || x > W + 40 || y < -40 || y > H + 40) continue;
    const s = (o.size ?? 7) * (0.4 + z * 2.2);
    c.fillStyle = col;
    if (kind === 0) c.fillRect(x - s / 2, y - s / 2, s, s);
    else if (kind === 1) { c.save(); c.translate(x, y); c.rotate(Math.atan2(ay, ax)); c.fillRect(-s * 1.6, -s * 0.18, s * 3.2, s * 0.36); c.restore(); }
    else { c.beginPath(); c.arc(x, y, s * 0.45, 0, Math.PI * 2); c.fill(); }
  }
  c.restore();
}

// ------------------------------------------------------------------ token tiles

const tileTex = new Map<string, THREE.CanvasTexture>();
/**
 * Texture for a token's printed face: the token text centred on a coloured card with a solid border.
 * Cached by (text, colours). `label` shows the token (leading space as a dot, see showTok).
 */
export function tokenFaceTexture(tok: string, bg: InkName = 'paper', fg: InkName = 'ink', opts: { id?: number; w?: number; h?: number } = {}): THREE.CanvasTexture {
  const key = `${tok}|${bg}|${fg}|${opts.id ?? ''}|${opts.w ?? 0}`;
  let tx = tileTex.get(key);
  if (tx) return tx;
  const cw = opts.w ?? 512, ch = opts.h ?? 256;
  const cv = document.createElement('canvas');
  cv.width = cw; cv.height = ch;
  const c = cv.getContext('2d')!;
  c.fillStyle = HEX[bg]; c.fillRect(0, 0, cw, ch);
  c.strokeStyle = HEX[fg]; c.lineWidth = 14; c.strokeRect(7, 7, cw - 14, ch - 14);
  const label = showTok(tok);
  let size = ch * 0.5;
  c.font = font(F.mono(700), size);
  const w = c.measureText(label).width;
  if (w > cw * 0.84) { size *= (cw * 0.84) / w; c.font = font(F.mono(700), size); }
  c.fillStyle = HEX[fg]; c.textAlign = 'center'; c.textBaseline = 'middle';
  c.fillText(label, cw / 2, ch * (opts.id !== undefined ? 0.4 : 0.52));
  if (opts.id !== undefined) { c.font = font(F.mono(700), ch * 0.2); c.fillText(String(opts.id), cw / 2, ch * 0.78); }
  tx = new THREE.CanvasTexture(cv);
  tx.colorSpace = THREE.SRGBColorSpace;
  tx.anisotropy = 8;
  tileTex.set(key, tx);
  return tx;
}

/**
 * A printed token tile: a slab whose front face carries the token (and optionally its id), sides in ink.
 * Width follows the text length unless given. Returns the mesh (front face = +z).
 */
export function tokenTile(tok: string, o: { face?: InkName; text?: InkName; side?: InkName; id?: number; w?: number; h?: number; d?: number; line?: number } = {}): THREE.Mesh {
  const len = showTok(tok).length;
  const h = o.h ?? 1, w = o.w ?? Math.max(1.2, 0.34 * len + 0.5) * h, d = o.d ?? 0.35 * h;
  const face = new THREE.MeshBasicMaterial({ map: tokenFaceTexture(tok, o.face ?? 'paper', o.text ?? 'ink', { id: o.id, w: Math.round(256 * w / h), h: 256 }) });
  const side = inkMat({ ink: o.side ?? 'blue', lit: 0.6, shade: 1 });
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), [side, side, side, side, face, side]);
  if ((o.line ?? 2) > 0) outline(m, o.line ?? 2);
  return m;
}

// ------------------------------------------------------------------ small helpers

const drawnAt = new WeakMap<object, number>();
/**
 * Redraw a Canvas2D layer once per output frame instead of once per motion-blur sub-frame (uploads cost
 * 2-4 ms each, and a frame can take 36 sub-frames). Draw with `f.ft` inside: the layer is then crisp
 * (no motion blur), which suits type, labels and print-like marks. Returns the uploaded texture.
 */
export function perFrame(layer: { clear(): void; upload(): THREE.Texture; texture: THREE.Texture }, f: Frame, draw: () => void): THREE.Texture {
  if (drawnAt.get(layer) !== f.ft) {
    layer.clear();
    draw();
    drawnAt.set(layer, f.ft);
    return layer.upload();
  }
  return layer.texture;
}

/** Distance travelled: `base` per second plus `gain` per second of full drums (re-exported from _cam). */
export { drive, beatEase, orbit, CamPath, applyShot, whip } from './_cam';

/** A camera for a scene, aspect from the frame (landscape or portrait). */
export const makeCam = (fov = 40) => new THREE.PerspectiveCamera(fov, W / H, 0.05, 200);

/** Portrait frame? Lay out for it (the 9:16 cut is a re-render, not a crop). */
export const portrait = () => H > W;

/** Map a landscape layout point to the current frame: x, y in 0..1 of the frame. */
export const px = (x: number, y: number) => [x * W, y * H] as const;

export type { AudioData };
