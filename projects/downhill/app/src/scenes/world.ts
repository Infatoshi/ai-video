// ML, slowly #1 "Downhill": the whole song is this one scene. One ball on one landscape (a real 2D slice of the
// network's loss landscape) while the network's boundary untangles beside it (the real network, evaluated per
// pixel from the recorded weights). The lesson moves through phases (story.ts) by camera moves and morphs, never
// cuts: the dots arrive, the knobs (weights), the loss, the map rises into a landscape, the loop (chorus x3), the
// slope and the step size, the too-big step (a real run, cut open), and the last chorus puts every piece back.
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { FSPass, Layer2D, W, H, clearRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp } from '../engine/util';
import { Model, NP, OFF, fmtLoss, fmtInt } from '../engine/model';
import { LineBatch } from '../engine/lines';
import { inkMat, outline, perFrame } from './_kit';
import { setOverlay } from '../engine/hud';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
const TAU = Math.PI * 2;
/** Height of the landscape: real loss, compressed above ~1 so the far walls don't dwarf the valley. */
const HS = 2.1;
const fH = (L: number) => 1.2 * Math.tanh(L / 1.2);
/** Slice units -> world units. */
const K = 0.26;

type Key = [number, number];
/** Keyed value: eased (in-out cubic) between keys, held outside. */
function kv(t: number, ks: Key[]): number {
  if (t <= ks[0]![0]) return ks[0]![1];
  for (let i = 1; i < ks.length; i++) {
    const [tb, vb] = ks[i]!;
    if (t <= tb) {
      const [ta, va] = ks[i - 1]!;
      return lerp(va, vb, ease.inOutCubic(clamp((t - ta) / Math.max(1e-3, tb - ta))));
    }
  }
  return ks[ks.length - 1]![1];
}
/** 0 -> 1 over [a, a + d] (eased), 1 -> 0 over [b, b + d]. */
const win = (t: number, a: number, b = Infinity, d = 0.6) => ease.inOutCubic(clamp((t - a) / d)) * (1 - ease.inOutCubic(clamp((t - b) / d)));
const on = (t: number, a: number, d = 0.6) => ease.inOutCubic(clamp((t - a) / d));

// ------------------------------------------------------------------ the network, per pixel
const PANEL_FRAG = /* glsl */ `
uniform vec4 th[85];
uniform vec4 th2[85];
uniform vec4 rect;       // x0, y0 (top-left, logical px, y down), size, box (data half-width)
uniform float alpha, region, line, wish, wishDraw, bigMix, frameA;
uniform vec2 res;
float net(vec2 p, bool second) {
  vec4 h1[4];
  for (int k = 0; k < 4; k++) {
    vec4 w0 = second ? th2[k] : th[k], w1 = second ? th2[4 + k] : th[4 + k], b = second ? th2[8 + k] : th[8 + k];
    h1[k] = tanh(p.x * w0 + p.y * w1 + b);
  }
  vec4 h2[4];
  for (int k = 0; k < 4; k++) h2[k] = second ? th2[76 + k] : th[76 + k];
  for (int j = 0; j < 16; j++) {
    float hj = h1[j / 4][j % 4];
    for (int k = 0; k < 4; k++) h2[k] += hj * (second ? th2[12 + j * 4 + k] : th[12 + j * 4 + k]);
  }
  float z = second ? th2[84].x : th[84].x;
  for (int k = 0; k < 4; k++) z += dot(tanh(h2[k]), second ? th2[80 + k] : th[80 + k]);
  return z;
}
vec3 shade(vec3 col, float z) {
  float pr = 1.0 / (1.0 + exp(-z)), conf = abs(2.0 * pr - 1.0);
  // class 1 (black dots) side: a light ink tint; class 0 (blue dots) side: a light blue tint
  vec3 tint = z > 0.0 ? mix(C_PAPER, C_INK, 0.13) : mix(C_PAPER, C_BLUE, 0.22);
  col = mix(col, tint, region * smoothstep(0.0, 0.6, conf));
  float fw = max(fwidth(z), 1e-4);
  float l = 1.0 - smoothstep(1.0, 2.2, abs(z) / fw);
  return mix(col, C_INK, l * line);
}
void main() {
  vec2 px = vec2(FRAG_PX.x, res.y - FRAG_PX.y);           // logical px, y down
  vec2 q = (px - rect.xy) / rect.z;                         // 0..1 in the panel
  if (q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0 || alpha <= 0.0) discard;
  vec2 p = vec2(q.x * 2.0 - 1.0, 1.0 - q.y * 2.0) * rect.w; // data coordinates (y up)
  vec3 col = C_PAPER;
  if (region > 0.0 || line > 0.0) {
    vec3 c1 = bigMix < 1.0 ? shade(col, net(p, false)) : col;
    vec3 c2 = bigMix > 0.0 ? shade(col, net(p, true)) : col;
    col = mix(c1, c2, bigMix);
  }
  if (wish > 0.0) {
    float z2 = net(p, true);
    float fw2 = max(fwidth(z2), 1e-4);
    // drawn in around the centre (angle sweep), like a pen going round the spiral
    float ang = fract(atan(p.y, p.x) / TAU + 0.25);
    float rr = length(p) / rect.w;
    float drawn = step(ang + rr * 0.0, wishDraw);
    float l2 = (1.0 - smoothstep(1.4, 2.8, abs(z2) / fw2)) * drawn;
    col = mix(col, C_PINK, l2 * wish);
  }
  // panel frame
  vec2 e = min(q, 1.0 - q) * rect.z;
  float fr = 1.0 - smoothstep(1.0, 2.2, min(e.x, e.y));
  col = mix(col, C_INK, fr * frameA);
  fragColor = vec4(col, 1.0);
}`;

function packTheta(th: Float32Array, out: THREE.Vector4[]) {
  // layout: [0..3] W1 row x, [4..7] W1 row y, [8..11] b1, [12..75] W2 (row j, block k), [76..79] b2, [80..83] W3, [84].x b3
  const v = (i: number, a: number) => out[i]!.set(th[a]!, th[a + 1]!, th[a + 2]!, th[a + 3]!);
  for (let k = 0; k < 4; k++) { v(k, OFF.W1 + 4 * k); v(4 + k, OFF.W1 + 16 + 4 * k); v(8 + k, OFF.b1 + 4 * k); }
  for (let j = 0; j < 16; j++) for (let k = 0; k < 4; k++) v(12 + j * 4 + k, OFF.W2 + j * 16 + 4 * k);
  for (let k = 0; k < 4; k++) { v(76 + k, OFF.b2 + 4 * k); v(80 + k, OFF.W3 + 4 * k); }
  out[84]!.set(th[OFF.b3]!, 0, 0, 0);
}

// ------------------------------------------------------------------ the landscape
const LAND_VERT = /* glsl */ `
attribute float aL;
attribute vec2 aAB;
attribute vec3 nUp;
uniform float rise, hs;
varying float vL;
varying vec2 vAB;
varying vec3 vN;
varying vec3 vW;
void main() {
  vL = aL; vAB = aAB;
  float h = hs * 1.2 * tanh(aL / 1.2) * rise;
  vec3 p = vec3(position.x, h, position.z);
  vN = normalize(mix(vec3(0.0, 1.0, 0.0), nUp, rise));
  vec4 w = modelMatrix * vec4(p, 1.0);
  vW = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const LAND_FRAG = /* glsl */ `
uniform vec3 paper, ink, blue, pink;
uniform vec2 ball;          // slice coords of the ball
uniform float reveal, fogR, fogA, dim, grid, contour, shade;
uniform vec3 camPos;
varying float vL;
varying vec2 vAB;
varying vec3 vN;
varying vec3 vW;
float g(float L) { return L >= 0.1 ? L * 10.0 : 1.0 + log2(max(L, 1e-4) / 0.1) * 1.5; }
void main() {
  float d = length(vAB - ball);
  if (d > reveal) discard;
  vec3 n = normalize(vN);
  vec3 Ld = normalize(vec3(-0.5, 0.8, 0.35));
  float sh = 1.0 - clamp(dot(n, Ld) * 0.5 + 0.5, 0.0, 1.0);
  float a = mix(0.03, 0.42, smoothstep(0.15, 0.85, sh)) * shade;
  vec3 col = paper * mix(vec3(1.0), ink / paper, a);
  // contours of the real loss (every 0.1 above 0.1, then halvings toward the valley floor)
  float gv = g(vL), fw = max(fwidth(gv), 1e-4);
  float c = abs(fract(gv + 0.5) - 0.5) / fw;
  float major = step(abs(mod(gv + 2.5, 5.0) - 2.5), 0.5);
  float lw = mix(0.9, 1.6, major);
  float cl = 1.0 - smoothstep(lw, lw + 1.0, c);
  col = mix(col, ink, cl * contour * mix(0.55, 1.0, major));
  // the slice's coordinate grid (every 2 units), faint
  vec2 gg = abs(fract(vAB / 2.0 + 0.5) - 0.5) / max(fwidth(vAB / 2.0), vec2(1e-4));
  float gl = 1.0 - smoothstep(0.6, 1.4, min(gg.x, gg.y));
  col = mix(col, mix(paper, blue, 0.5), gl * grid * 0.6);
  // the edge of what is revealed so far (the map being drawn): an ink rim
  float rim = 1.0 - smoothstep(0.0, 0.25, reveal - d);
  col = mix(col, ink, rim * step(reveal, 60.0));
  // it can't see far: fog around the ball
  float fog = smoothstep(fogR, fogR + 1.5, d) * fogA;
  col = mix(col, paper, max(fog, dim));
  gl_FragColor = vec4(col, 1.0);
}`;

interface Tm { [k: string]: number }

export default class World extends Scene {
  s3 = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(34, W / H, 0.05, 200);
  land!: THREE.Mesh;
  landMat!: THREE.ShaderMaterial;
  walls!: THREE.Mesh;
  wallMat = new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    vertexShader: /* glsl */ `varying vec3 vN; varying vec3 vW; void main(){ vN = normal; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: /* glsl */ `uniform vec3 paper, ink; uniform vec2 ballW; uniform float fogR, fogA, dim;
      varying vec3 vN; varying vec3 vW;
      void main(){
        float a = 0.1 + 0.12 * abs(vN.x);
        vec3 col = paper * mix(vec3(1.0), ink / paper, a);
        float d = length(vW.xz - ballW);
        col = mix(col, paper, max(smoothstep(fogR, fogR + 0.4, d) * fogA, dim));
        gl_FragColor = vec4(col, 1.0);
      }`,
    uniforms: { paper: { value: new THREE.Vector3(...LIN.paper) }, ink: { value: new THREE.Vector3(...LIN.ink) }, ballW: { value: new THREE.Vector2() }, fogR: { value: 99 }, fogA: { value: 0 }, dim: { value: 0 } },
  });
  ball!: THREE.Mesh;
  ballMat = inkMat({ ink: 'pink', lit: 0.55, shade: 1.0 });
  lines3 = new LineBatch(6000, { screen2D: false, blend: 'normal', depthTest: true });
  panel = new FSPass(PANEL_FRAG, {
    th: { value: Array.from({ length: 85 }, () => new THREE.Vector4()) },
    th2: { value: Array.from({ length: 85 }, () => new THREE.Vector4()) },
    rect: { value: new THREE.Vector4() }, alpha: { value: 1 }, region: { value: 0 }, line: { value: 0 }, wish: { value: 0 },
    wishDraw: { value: 0 }, bigMix: { value: 0 }, frameA: { value: 1 }, res: { value: new THREE.Vector2(W, H) },
  });
  ui = new Layer2D();
  /** Small type goes here: the HUD draws it after the print (crisp solid ink). */
  txt = new Layer2D();
  /** The ui context, with fillText of type under 40 px redirected to the txt layer (same state and transform). */
  cx!: CanvasRenderingContext2D;
  T: Tm = {};
  th = new Float32Array(NP);
  thB = new Float32Array(NP);
  dotL = new Float32Array(200);
  m!: Model;
  ac = 0; bc = 0; // slice centre
  P = H > W;
  /** Thumbnail mode: the picture without small type. */
  lite = false;

  override async init() {
    const m = (this.m = this.ctx.model);
    const S = m.surf;
    this.ac = (S.a0 + S.a1) / 2; this.bc = (S.b0 + S.b1) / 2;
    // landscape mesh on the recorded grid
    const na = S.na, nb = S.nb;
    const pos = new Float32Array(na * nb * 3), aL = new Float32Array(na * nb), aAB = new Float32Array(na * nb * 2), nUp = new Float32Array(na * nb * 3);
    const hAt = (i: number, j: number) => HS * fH(S.L[Math.min(nb - 1, Math.max(0, j)) * na + Math.min(na - 1, Math.max(0, i))]!);
    const da = ((S.a1 - S.a0) / (na - 1)) * K, db = ((S.b1 - S.b0) / (nb - 1)) * K;
    for (let j = 0; j < nb; j++) for (let i = 0; i < na; i++) {
      const k = j * na + i, a = S.a0 + (i / (na - 1)) * (S.a1 - S.a0), b = S.b0 + (j / (nb - 1)) * (S.b1 - S.b0);
      const [x, z] = this.xz(a, b);
      pos[3 * k] = x; pos[3 * k + 1] = 0; pos[3 * k + 2] = z;
      aL[k] = S.L[k]!; aAB[2 * k] = a; aAB[2 * k + 1] = b;
      const gx = (hAt(i + 1, j) - hAt(i - 1, j)) / (2 * da), gz = (hAt(i, j + 1) - hAt(i, j - 1)) / (2 * db);
      // z = -b*K: d/dz = -d/db
      const n = new THREE.Vector3(-gx, 1, gz).normalize();
      nUp[3 * k] = n.x; nUp[3 * k + 1] = n.y; nUp[3 * k + 2] = n.z;
    }
    const idx: number[] = [];
    for (let j = 0; j < nb - 1; j++) for (let i = 0; i < na - 1; i++) {
      const a = j * na + i, b = a + 1, c = a + na, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aL', new THREE.BufferAttribute(aL, 1));
    g.setAttribute('aAB', new THREE.BufferAttribute(aAB, 2));
    g.setAttribute('nUp', new THREE.BufferAttribute(nUp, 3));
    g.setIndex(idx);
    const v3 = (k: 'paper' | 'ink' | 'blue' | 'pink') => new THREE.Vector3(...LIN[k].map((x) => Math.max(x, 0.03)));
    this.landMat = new THREE.ShaderMaterial({
      vertexShader: LAND_VERT, fragmentShader: LAND_FRAG, side: THREE.DoubleSide, extensions: { derivatives: true } as any,
      uniforms: {
        paper: { value: new THREE.Vector3(...LIN.paper) }, ink: { value: v3('ink') }, blue: { value: v3('blue') }, pink: { value: v3('pink') },
        ball: { value: new THREE.Vector2() }, reveal: { value: 0 }, fogR: { value: 4.2 }, fogA: { value: 0 }, dim: { value: 0 }, grid: { value: 1 },
        contour: { value: 1 }, shade: { value: 1 }, camPos: { value: new THREE.Vector3() }, rise: { value: 0 }, hs: { value: HS },
      },
    });
    this.land = new THREE.Mesh(g, this.landMat);
    this.land.frustumCulled = false;
    this.s3.add(this.land);
    // walls: the landscape as a printed block (sides down to the floor)
    this.walls = new THREE.Mesh(new THREE.BufferGeometry(), this.wallMat);
    this.walls.frustumCulled = false;
    this.s3.add(this.walls);
    // the ball: the network's weights, at their place on the slice
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(0.19, 40, 24), this.ballMat);
    outline(this.ball, 2.4);
    this.s3.add(this.ball);
    this.times();
    const tl = this.txt, ui = this.ui.ctx;
    this.cx = new Proxy(ui, {
      get(target, prop) {
        if (prop === 'fillText') return (str: string, x: number, y: number) => {
          if (parseFloat(target.font) >= 40) return target.fillText(str, x, y);
          const tc = tl.ctx;
          tc.save();
          tc.setTransform(target.getTransform());
          tc.font = target.font; tc.fillStyle = target.fillStyle; tc.globalAlpha = target.globalAlpha;
          tc.textAlign = target.textAlign; tc.textBaseline = target.textBaseline; tc.letterSpacing = target.letterSpacing;
          tc.fillText(str, x, y);
          tc.restore();
        };
        const v = Reflect.get(target, prop, target);
        return typeof v === 'function' ? v.bind(target) : v;
      },
      set(target, prop, val) { return Reflect.set(target, prop, val, target); },
    });
    setOverlay(this.lite ? null : this.txt.canvas);
  }

  /** Slice (a, b) -> world (x, z). */
  xz(a: number, b: number): [number, number] { return [(a - this.ac) * K, -(b - this.bc) * K]; }
  /** Surface height (world) at slice (a, b) for the current rise. */
  yAt(a: number, b: number, rise: number) { return HS * fH(this.m.surfL(a, b)) * rise; }

  /** Lyric-anchored times (with fallbacks, so an early take or a missing line never breaks the scene). */
  private times() {
    const ly = this.ctx.lyrics, au = this.ctx.audio, T = this.T;
    const L = (q: string, n = 0, fb = NaN) => { try { return ly.get(q, n).start; } catch { return fb; } };
    const E = (q: string, n = 0, fb = NaN) => { try { return ly.get(q, n).end; } catch { return fb; } };
    const Wd = (q: string, w: string, n = 0, fb = NaN) => {
      try { const x = ly.get(q, n).words.find((y) => y.w.toLowerCase().replace(/[^a-z']/g, '').startsWith(w)); return x ? x.start : fb; } catch { return fb; }
    };
    T.v1 = L('Two spirals', 0, 16); T.v1b = L('Wound round', 0, T.v1 + 5); T.v1c = L('Draw me a line', 0, T.v1 + 10); T.v1d = L("The machine can't", 0, T.v1 + 15);
    T.v1e = E("The machine can't", 0, T.v1 + 19);
    T.v2 = L('Inside the machine', 0, T.v1 + 30); T.v2b = L('Three hundred thirty', 0, T.v2 + 5); T.v2c = L('Turn any knob', 0, T.v2 + 10);
    T.v2d = L('They start out random', 0, T.v2 + 15); T.v2e = E('They start out random', 0, T.v2 + 19);
    T.v3 = L('So measure how wrong', 0, T.v2 + 30); T.v3b = L('Add it up', 0, T.v3 + 5); T.v3c = L('Every setting of the knobs', 0, T.v3 + 10);
    T.v3d = L('The loss is how high', 0, T.v3 + 15); T.v3e = E('The loss is how high', 0, T.v3 + 19);
    for (let n = 0; n < 3; n++) {
      T[`c${n}`] = L('^Measure how wrong it is', n, NaN); T[`c${n}b`] = L('Find which way', n, NaN);
      T[`c${n}c`] = L('Take one small step', n, NaN); T[`c${n}d`] = L('And do it again', n, NaN); T[`c${n}e`] = E('And do it again', n, NaN);
    }
    T.v4 = L("It can't see the valley", 0, T.c0e + 10); T.v4b = L('It feels for the slope', 0, T.v4 + 5); T.v4c = L("Which way is down, that", 0, T.v4 + 10);
    T.v4d = L('How big a step', 0, T.v4 + 15); T.v4e = E('How big a step', 0, T.v4 + 19);
    T.br = L('Now make every step', 0, T.c1e + 2); T.brb = L('It jumps right over', 0, T.br + 5); T.brc = L('Lands on the far side', 0, T.br + 10);
    T.brd = L('It bounces and bounces', 0, T.br + 15); T.bre = E('It bounces and bounces', 0, T.br + 19);
    T.out = L('Three thousand one hundred', 0, T.c2e + 4); T.steps = Wd('Three thousand one hundred', 'steps', 0, T.out + 3);
    T.outb = L('And every dot', 0, T.out + 5); T.outc = L('Next time', 0, T.out + 10); T.oute = E('Next time', 0, T.out + 14);
    T.end = au.duration;
  }

  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides | void {
    const r = this.ctx.renderer, t = f.t, T = this.T, m = this.m, st = this.ctx.story, P = this.P;
    const step = st.step(t);
    // ------------------------------------------------ layout (all keyed, continuous)
    const landOn = on(t, T.v3c - 0.2, 1.4);                  // the map appears on "Every setting of the knobs"
    const rise = on(t, T.v3d, 3.2);                           // ... and rises on "The loss is how high"
    const bridge = win(t, T.br - 0.4, st.brx, 2.0);       // the too-big step: the cut chart takes the stage
    // panel: big and centred for the task; to the side once the knobs arrive
    const side = on(t, T.v2 - 1.2, 2.2);
    const pc = P
      ? { x: 540, y: lerp(1020, 1440, side), s: lerp(860, 720, side) }
      : { x: lerp(960, 1500, side), y: lerp(600, 560, side), s: lerp(740, 560, side) };
    // outro: the panel comes back to the middle, big
    const fin = on(t, T.outb - 0.6, 2.4);
    if (fin > 0) { pc.x = lerp(pc.x, P ? 540 : 1330, fin); pc.y = lerp(pc.y, P ? 1300 : 600, fin); pc.s = lerp(pc.s, P ? 900 : 700, fin); }
    const breathe = 1 + 0.012 * Math.sin(t * 0.37);
    const ps = pc.s * breathe, px0 = pc.x - ps / 2, py0 = pc.y - ps / 2;

    // ------------------------------------------------ which weights are shown
    const knobJ = OFF.W2 + 5 * 16 + 9;                       // the knob that turns in verse 2
    m.thetaAt(step, false, this.th);
    const turn = win(t, T.v2c, T.v2d, 0.8) * Math.sin(clamp((t - T.v2c) / Math.max(1, T.v2d - T.v2c)) * TAU) * 2.6;
    this.th[knobJ] = this.th[knobJ]! + turn;
    const bigOn = win(t, T.br + 0.2, st.brx - 0.5, 1.4);
    const bigStep = st.bigStep(t);

    // ------------------------------------------------ the 3D landscape
    clearRT(r, out, LIN.paper);
    const [ba, bb] = m.pathAt(step);
    if (landOn > 0.001) this.render3D(f, out, { landOn, rise, bridge, ba, bb, step });

    // ------------------------------------------------ the panel (the network itself)
    const u = this.panel.u;
    packTheta(this.th, u.th!.value as THREE.Vector4[]);
    if (bigOn > 0) m.thetaAt(bigStep, true, this.thB); else m.thetaAt(m.steps, false, this.thB);
    packTheta(this.thB, u.th2!.value as THREE.Vector4[]);
    u.bigMix!.value = bigOn;
    (u.rect!.value as THREE.Vector4).set(px0, py0, ps, m.box);
    const panelA = on(t, 0.4, 1.5);
    u.alpha!.value = panelA;
    const netOn = on(t, T.v1d - 0.1, 1.2);
    u.region!.value = netOn; u.line!.value = netOn;
    u.wish!.value = win(t, T.v1c, T.v1d - 0.2, 0.5);
    u.wishDraw!.value = clamp((t - T.v1c) / Math.max(1, (T.v1d - T.v1c) * 0.8));
    u.frameA!.value = 1;
    if (panelA > 0) this.panel.render(r, out);

    // ------------------------------------------------ 2D: dots, knobs, labels, charts, numbers
    const tex = perFrame(this.ui, f, () => { this.txt.clear(); this.draw2D(f, { px0, py0, ps, step, bigStep, bigOn, landOn, rise, bridge, turn, knobJ }); });
    this.ctx.comp.draw(r, tex, out);
    const endFade = clamp((t - (T.end - 2.2)) / 2.0);
    return { reg: 3, fade: endFade };
  }

  // ================================================================== 3D
  private render3D(f: Frame, out: THREE.WebGLRenderTarget, o: { landOn: number; rise: number; bridge: number; ba: number; bb: number; step: number }) {
    const r = this.ctx.renderer, t = f.t, T = this.T, m = this.m, P = this.P, cam = this.cam;
    const { rise, ba, bb } = o;
    // camera: top-down over the map, tilting into a 3/4 view as it rises; close on the ball for the slope;
    // up and back for the bridge; a slow drift always
    const az0 = -1.05 + 0.16 * Math.sin(t * 0.07) + t * 0.0025;
    const close = win(t, T.v4 - 0.8, T.c1 - 1.2, 2.2);
    const elev = kv(t, [[T.v3c, 1.45], [T.v3d, 1.45], [T.v3d + 3.6, 0.62]]) + close * -0.12 + o.bridge * 0.18 + win(t, this.ctx.story.brx + 0.5, T.c2 - 2.5, 4.0) * 0.06;
    const D0 = P ? 25 : 18.5;
    let dist = kv(t, [[T.v3c, D0 * 1.12], [T.v3d + 3.6, D0]]) * (1 - 0.42 * close) * (1 + 0.1 * o.bridge);
    const fin = on(t, T.outb - 0.6, 2.4);
    dist *= 1 + 0.12 * fin;
    const [bx, bz] = this.xz(ba, bb);
    const by = this.yAt(ba, bb, rise);
    const follow = clamp(0.35 + 0.55 * close);
    const tgt = new THREE.Vector3(lerp(0, bx, follow), lerp(0.55 * rise, by, follow), lerp(0, bz, follow));
    // the long break after the bridge: a slow look around the valley while the ball rolls on
    const solo = win(t, this.ctx.story.brx + 0.5, T.c2 - 2.5, 4.0);
    const az = az0 + close * 0.3 + solo * 0.35 + (P ? 0 : o.bridge * 0.35 * Math.sin((t - T.br) * 0.12));
    cam.position.set(tgt.x + Math.cos(az) * Math.cos(elev) * dist, tgt.y + Math.sin(elev) * dist, tgt.z + Math.sin(az) * Math.cos(elev) * dist);
    cam.up.set(0, 1, 0);
    cam.lookAt(tgt);
    // frame: the landscape takes the left of a 16:9 frame (the panel is on the right), the top of a 9:16 one
    const side = on(t, T.v2 - 1.2, 2.2);
    if (P) cam.setViewOffset(W, H, 0, H * 0.19 * side, W, H);
    else cam.setViewOffset(W, H, W * 0.155 * side, H * 0.02 * side, W, H);
    cam.fov = P ? 50 : 34;
    cam.updateProjectionMatrix();

    const U = this.landMat.uniforms;
    U.rise!.value = rise;
    (U.ball!.value as THREE.Vector2).set(ba, bb);
    // reveal: the map is drawn outward from the ball
    U.reveal!.value = o.landOn >= 0.999 ? 99 : 0.5 + 34 * ease.inOutCubic(o.landOn);
    U.fogR!.value = 4.2;
    U.fogA!.value = win(t, T.v4 - 0.4, T.v4c + 0.5, 2.6);
    U.dim!.value = (P ? 0.9 : 0.8) * o.bridge;
    this.land.visible = true;
    const WU = this.wallMat.uniforms, [bwx, bwz] = this.xz(ba, bb);
    (WU.ballW!.value as THREE.Vector2).set(bwx, bwz);
    WU.fogR!.value = (U.fogR!.value as number) * K;
    WU.fogA!.value = U.fogA!.value;
    WU.dim!.value = U.dim!.value;
    // walls follow the rise
    this.buildWalls(rise, o.landOn);
    // ball
    const bs = on(t, T.v3c + 0.4, 0.9) * (1 + 0.04 * Math.sin(t * 2.1)) * (1 - o.bridge);
    this.ball.position.set(bx, by + 0.19 * bs, bz);
    this.ball.scale.setScalar(Math.max(1e-3, bs));
    this.ball.visible = bs > 0.01;
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.s3, cam);

    // lines on the surface: the trail (from chorus 2), the downhill arrow (from chorus 1), the tangent disc
    const lb = this.lines3;
    lb.clear();
    const pink = LIN.pink, ink = LIN.ink;
    const trailOn = on(t, T.c1 - 0.5, 1.5) * (1 - o.bridge);
    if (trailOn > 0.01 && o.step > 1) {
      const n = Math.min(400, Math.max(2, Math.ceil(o.step / 8)));
      let prev: THREE.Vector3 | null = null;
      for (let i = 0; i <= n; i++) {
        const s = (i / n) * o.step, [a, b] = m.pathAt(s), [x, z] = this.xz(a, b);
        const p = new THREE.Vector3(x, this.yAt(a, b, rise) + 0.03, z);
        if (prev) lb.seg(prev.x, prev.y, prev.z, p.x, p.y, p.z, 3.2, pink[0], pink[1], pink[2], trailOn);
        prev = p;
      }
    }
    // the downhill arrow: the real -gradient, projected onto the slice (direction real; length log-compressed)
    const arrowOn = this.arrowAmt(t) * (1 - o.bridge);
    if (arrowOn > 0.01 && o.bridge < 0.99) {
      const [ga, gb] = m.downhillAt(o.step);
      const gn = Math.hypot(ga, gb) + 1e-9;
      const len = clamp(2.2 + 0.9 * Math.log10(gn / 0.004), 1.6, 3.6) * arrowOn;
      const ua = ga / gn, ub = gb / gn;
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= 14; i++) {
        const a = ba + ua * len * (i / 14), b = bb + ub * len * (i / 14), [x, z] = this.xz(a, b);
        pts.push(new THREE.Vector3(x, this.yAt(a, b, rise) + 0.05, z));
      }
      const aa = 1 - o.bridge;
      for (let i = 1; i < pts.length; i++) lb.seg(pts[i - 1]!.x, pts[i - 1]!.y, pts[i - 1]!.z, pts[i]!.x, pts[i]!.y, pts[i]!.z, 9, pink[0], pink[1], pink[2], aa);
      // head
      const tip = pts[pts.length - 1]!, back = pts[pts.length - 4]!;
      const dir = tip.clone().sub(back).normalize(), sideV = new THREE.Vector3(-dir.z, 0, dir.x).normalize().multiplyScalar(0.2);
      const hb = tip.clone().sub(dir.clone().multiplyScalar(0.34));
      lb.seg(tip.x, tip.y, tip.z, hb.x + sideV.x, hb.y, hb.z + sideV.z, 9, pink[0], pink[1], pink[2], aa);
      lb.seg(tip.x, tip.y, tip.z, hb.x - sideV.x, hb.y, hb.z - sideV.z, 9, pink[0], pink[1], pink[2], aa);
      // verse 4: the step is 0.3 x the slope: an ink tick at 30% of the arrow
      const lrOn = win(t, T.v4d - 0.2, T.c1 + 1.0, 0.8);
      if (lrOn > 0.01) {
        const k = Math.round(0.3 * 14), q = pts[k]!;
        lb.seg(q.x - sideV.x * 1.4, q.y, q.z - sideV.z * 1.4, q.x + sideV.x * 1.4, q.y, q.z + sideV.z * 1.4, 5, ink[0], ink[1], ink[2], lrOn);
        for (let i = 1; i <= k; i++) lb.seg(pts[i - 1]!.x, pts[i - 1]!.y + 0.02, pts[i - 1]!.z, pts[i]!.x, pts[i]!.y + 0.02, pts[i]!.z, 11, ink[0], ink[1], ink[2], lrOn);
      }
    }
    // verse 4: the slope under its feet (a small tangent ring)
    const feel = win(t, T.v4b - 0.2, T.c1 - 0.5, 0.8);
    if (feel > 0.01) {
      const R = 0.9 * feel;
      let prev: THREE.Vector3 | null = null;
      for (let i = 0; i <= 48; i++) {
        const an = (i / 48) * TAU, a = ba + Math.cos(an) * R, b = bb + Math.sin(an) * R, [x, z] = this.xz(a, b);
        const p = new THREE.Vector3(x, this.yAt(a, b, rise) + 0.04, z);
        if (prev) lb.seg(prev.x, prev.y, prev.z, p.x, p.y, p.z, 3, ink[0], ink[1], ink[2], feel);
        prev = p;
      }
    }
    if (lb.count) lb.render(r, out, cam);
  }

  /** How much of the downhill arrow shows: from chorus 1's "Find which way is down", then always (it re-aims each step). */
  private arrowAmt(t: number) {
    const T = this.T;
    return on(t, (T.c0b ?? T.v4) - 0.2, 0.9);
  }

  private wallGeoRise = -1;
  private buildWalls(rise: number, landOn: number) {
    this.walls.visible = landOn > 0.98;
    if (!this.walls.visible || Math.abs(rise - this.wallGeoRise) < 1e-4) return;
    this.wallGeoRise = rise;
    const S = this.m.surf, pos: number[] = [], nrm: number[] = [];
    const edge = (pts: [number, number][], nx: number, nz: number) => {
      for (let i = 0; i < pts.length - 1; i++) {
        const [a0, b0] = pts[i]!, [a1, b1] = pts[i + 1]!;
        const [x0, z0] = this.xz(a0, b0), [x1, z1] = this.xz(a1, b1);
        const y0 = this.yAt(a0, b0, rise), y1 = this.yAt(a1, b1, rise), f = -0.35;
        pos.push(x0, f, z0, x1, f, z1, x1, y1, z1, x0, f, z0, x1, y1, z1, x0, y0, z0);
        for (let k = 0; k < 6; k++) nrm.push(nx, 0, nz);
      }
    };
    const n = 80, A = (i: number) => S.a0 + (i / n) * (S.a1 - S.a0), B = (i: number) => S.b0 + (i / n) * (S.b1 - S.b0);
    edge(Array.from({ length: n + 1 }, (_, i) => [A(i), S.b0] as [number, number]), 0, 1);
    edge(Array.from({ length: n + 1 }, (_, i) => [A(n - i), S.b1] as [number, number]), 0, -1);
    edge(Array.from({ length: n + 1 }, (_, i) => [S.a1, B(i)] as [number, number]), 1, 0);
    edge(Array.from({ length: n + 1 }, (_, i) => [S.a0, B(n - i)] as [number, number]), -1, 0);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
    this.walls.geometry.dispose();
    this.walls.geometry = g;
    (this.walls.material as THREE.Material).side = THREE.DoubleSide;
  }

  /** Project a world point to logical screen px (with the current camera and its view offset). */
  private proj(x: number, y: number, z: number): [number, number, boolean] {
    const v = new THREE.Vector3(x, y, z).project(this.cam);
    return [(v.x * 0.5 + 0.5) * W, (0.5 - v.y * 0.5) * H, v.z < 1];
  }

  // ================================================================== 2D
  private draw2D(f: Frame, o: { px0: number; py0: number; ps: number; step: number; bigStep: number; bigOn: number; landOn: number; rise: number; bridge: number; turn: number; knobJ: number }) {
    const c = this.cx, t = f.ft, T = this.T, m = this.m, P = this.P;
    const { px0, py0, ps } = o;
    const toPx = (x: number, y: number): [number, number] => [px0 + ((x / m.box) * 0.5 + 0.5) * ps, py0 + (0.5 - (y / m.box) * 0.5) * ps];
    const N = m.y.length;

    // ---- the dots: they arrive along their arms in the intro
    const arrive = (i: number) => {
      const k = i % (N / 2), cls = Math.floor(i / (N / 2));
      const r = Math.hypot(m.X[2 * i]!, m.X[2 * i + 1]!);
      const t0 = 0.6 + r * Math.min(3.6, T.v1 - 4.5) + cls * 0.3 + (k % 7) * 0.02;
      return ease.outCubic(clamp((t - t0) / 0.8));
    };
    // per-dot error (for rings and bars) at the weights on screen
    const dl = m.dotLoss(o.bigOn > 0.5 ? this.thB : this.th, this.dotL);
    const ringA = Math.max(win(t, T.v1d + 0.4, T.v2 - 0.3, 0.6), win(t, T.v2d, T.v3 - 0.2, 0.6), this.chorusLine(t, 0),
      win(t, T.brd, this.ctx.story.brx - 0.5, 0.8));
    const barsA = win(t, T.v3 - 0.1, T.v3b + 1.2, 0.5);
    const sumK = on(t, T.v3b, 1.8);
    const dotR = (P ? 7 : 6) * (ps / 600);
    // the waltz's one: every dot swells a little on each downbeat and settles over the bar (the picture breathes
    // with the music; also what keeps the 2D phases from ever standing still)
    const dbs = this.ctx.audio.downbeats;
    let db = -1e9;
    for (const x of dbs) { if (x <= t + 1e-4) db = x; else break; }
    const since = t - db, pulse = since < 0.24 ? ease.inOutCubic(since / 0.24) : Math.exp(-(since - 0.24) / 0.5);
    for (let i = 0; i < N; i++) {
      const a = arrive(i);
      if (a <= 0) continue;
      const [x, y] = toPx(m.X[2 * i]!, m.X[2 * i + 1]!);
      const rr = dotR * (0.4 + 0.6 * a) * (1 + (P ? 0.17 : 0.24 + 0.08 * clamp((740 - ps) / 180)) * pulse);
      c.globalAlpha = a;
      c.fillStyle = m.y[i] ? INK : BLUE;
      c.beginPath(); c.arc(x, y, rr, 0, TAU); c.fill();
      c.globalAlpha = 1;
      const wrong = (dl[i]! > Math.LN2);
      if (wrong && ringA > 0.01) {
        c.strokeStyle = PINK; c.lineWidth = 2.6; c.globalAlpha = ringA;
        c.beginPath(); c.arc(x, y, rr + 5, 0, TAU); c.stroke(); c.globalAlpha = 1;
      }
      // verse 3: each dot's error as a pink bar, one after another along the arms, then they add up
      if (barsA > 0.01) {
        const k = i % (N / 2), cls = Math.floor(i / (N / 2));
        const order = (k + cls * 0.5) / (N / 2);
        const grow = ease.outCubic(clamp((t - T.v3 - order * Math.max(1, (T.v3b - T.v3) * 0.8)) / 0.4));
        const h = Math.min(60, dl[i]! * 30) * grow * (1 - sumK);
        if (h > 0.3) { c.fillStyle = PINK; c.globalAlpha = barsA; c.fillRect(x - 2, y - rr - 2 - h, 4, h); c.globalAlpha = 1; }
      }
    }

    // verse 1, "wound round each other": a pink tracer runs out along each arm, centre to edge
    const tr = win(t, T.v1b - 0.1, T.v1c - 0.3, 0.4);
    if (tr > 0.01) {
      const u = clamp((t - T.v1b) / Math.max(1, (T.v1c - T.v1b) * 0.85));
      c.strokeStyle = PINK; c.lineWidth = 3;
      for (let i = 0; i < N; i++) {
        const r0 = Math.hypot(m.X[2 * i]!, m.X[2 * i + 1]!) / 1.05;
        const k = Math.exp(-Math.pow((r0 - u) / 0.07, 2));
        if (k < 0.05) continue;
        const [x, y] = toPx(m.X[2 * i]!, m.X[2 * i + 1]!);
        c.globalAlpha = tr * k;
        c.beginPath(); c.arc(x, y, dotR + 6, 0, TAU); c.stroke();
      }
      c.globalAlpha = 1;
    }

    // ---- the title (intro)
    const ti = win(t, 1.0, T.v1 - 1.6, 1.4);
    if (ti > 0.01) {
      c.save();
      c.globalAlpha = ti;
      c.textAlign = 'center';
      c.fillStyle = INK;
      c.font = font(F.archivo(125, 900), P ? 120 : 110);
      c.fillText('DOWNHILL', W / 2, P ? 470 : 150);
      c.font = font(F.mono(600), P ? 22 : 20); c.letterSpacing = '3px';
      c.fillText('HOW A MODEL LEARNS', W / 2, P ? 520 : 196);
      c.restore();
    }

    // ---- the knobs (verse 2): 337 dials, each at its real weight
    const kn = win(t, T.v2 - 0.2, T.v3c + 0.6, 1.0);
    if (kn > 0.01) this.drawKnobs(c, t, kn, { ...o, pulse });

    // ---- landscape labels (projected from 3D)
    if (o.landOn > 0.5 && !this.lite) this.landLabels(c, t, o);

    // ---- the bridge: the too-big step, cut open, and its run
    if (o.bridge > 0.01) this.drawBridge(c, t, o);

    // ---- the loss curve (chorus 3's layer) under the panel
    const curveA = win(t, T.c2 - 0.5, T.outb - 0.5, 1.2);
    if (curveA > 0.01 && !this.lite) this.drawCurve(c, t, curveA, o, false);

    if (this.lite) { c.letterSpacing = '0px'; return; }
    // ---- the one number
    this.drawFeature(c, t, o);

    // ---- panel captions
    c.font = font(F.mono(500), P ? 16 : 14);
    c.fillStyle = INK; c.letterSpacing = '1px';
    const capA = on(t, T.v1 + 1.5, 1.2) * (1 - win(t, T.br, this.ctx.story.brx - 0.5, 0.8));
    if (capA > 0.01) {
      c.globalAlpha = capA;
      c.fillText('200 DOTS · 100 BLUE · 100 BLACK', px0, py0 + ps + 26);
      const net = on(t, T.v1d, 1.0);
      if (net > 0.01) { c.globalAlpha = capA * net; c.textAlign = 'right'; c.fillText('THE NETWORK’S LINE', px0 + ps, py0 - 12); c.textAlign = 'left'; }
      c.globalAlpha = 1;
    }
    const bigCap = win(t, T.br + 0.5, this.ctx.story.brx - 0.5, 0.8);
    if (bigCap > 0.01) {
      c.globalAlpha = bigCap; c.fillStyle = PINK; c.textAlign = 'right';
      c.fillText(`THE TOO-BIG RUN (LEARNING RATE 3.0) · STEP ${fmtInt(Math.floor(o.bigStep))}`, px0 + ps, py0 - 12); c.textAlign = 'left'; c.globalAlpha = 1;
    }
    // the wish line's label
    const wishA = win(t, T.v1c + 0.5, T.v1d - 0.2, 0.5);
    if (wishA > 0.01) { c.globalAlpha = wishA; c.fillStyle = PINK; c.fillText('A LINE THAT KEEPS THEM APART', px0, py0 - 12); c.globalAlpha = 1; }
    // next episode
    const nx = on(t, T.outc - 0.1, 1.0);
    if (nx > 0.01) {
      c.save();
      c.globalAlpha = nx;
      c.fillStyle = INK;
      const x = P ? W / 2 : 330, y = P ? 1850 - 80 : H - 110;
      c.textAlign = P ? 'center' : 'left';
      c.font = font(F.mono(700), P ? 22 : 20); c.letterSpacing = '3px';
      c.fillText('NEXT · ML, SLOWLY 2/5', x, y - 44);
      c.font = font(F.archivo(125, 900), P ? 64 : 58); c.letterSpacing = '0px';
      c.fillText('TOKENS', x, y + 12);
      c.restore();
    }
    c.letterSpacing = '0px';
  }

  /** Chorus n's "Measure how wrong it is" .. "Find which way": the wrong dots ring up. Later choruses ring always. */
  private chorusLine(t: number, _n: number) {
    const T = this.T;
    let a = 0;
    for (let n = 0; n < 3; n++) {
      const c0 = T[`c${n}`], c1 = T[`c${n}e`];
      if (!Number.isFinite(c0!)) continue;
      a = Math.max(a, win(t, c0! - 0.2, c1! + 1.0, 0.6));
    }
    return a;
  }

  private drawKnobs(c: CanvasRenderingContext2D, t: number, kn: number, o: { turn: number; knobJ: number; landOn: number; rise: number; px0: number; pulse: number }) {
    const T = this.T, P = this.P, th = this.th;
    const cols = P ? 24 : 22, rows = Math.ceil(NP / cols);
    const x0 = P ? 90 : 150, y0 = P ? 300 : 330, gx = P ? 38 : 32, gy = P ? 38 : 34, R = P ? 13 : 11.5;
    // collapse into the ball's place on the map (verse 3's "Every setting of the knobs is a place")
    const col = ease.inOutCubic(clamp((t - T.v3c) / 1.6));
    let bxs = x0 + (cols * gx) / 2, bys = y0 + (rows * gy) / 2;
    if (col > 0) {
      const [ba, bb] = this.m.pathAt(0), [x, z] = this.xz(ba, bb);
      const p = this.proj(x, this.yAt(ba, bb, o.rise), z);
      bxs = p[0]; bys = p[1];
    }
    const drawIn = (i: number) => ease.outCubic(clamp((t - T.v2 - (i / NP) * 2.4) / 0.5));
    c.save();
    for (let i = 0; i < NP; i++) {
      const a = drawIn(i) * kn;
      if (a <= 0.01) continue;
      const cx0 = x0 + (i % cols) * gx + gx / 2, cy0 = y0 + Math.floor(i / cols) * gy + gy / 2;
      const cx = lerp(cx0, bxs, col), cy = lerp(cy0, bys, col), rr = R * (1 - 0.9 * col) * (1 + (P ? 0 : 0.035) * o.pulse);
      const hot = i === o.knobJ && Math.abs(o.turn) > 0.01;
      c.globalAlpha = a * Math.pow(1 - col, 1.5);
      c.strokeStyle = hot ? PINK : INK; c.lineWidth = (hot ? 3 : 1.6) * (1 - 0.7 * col);
      c.beginPath(); c.arc(cx, cy, rr, 0, TAU); c.stroke();
      const ang = -Math.PI / 2 + clamp(th[i]! * 1.1, -2.6, 2.6);
      c.beginPath(); c.moveTo(cx, cy); c.lineTo(cx + Math.cos(ang) * rr * 0.9, cy + Math.sin(ang) * rr * 0.9); c.stroke();
    }
    c.globalAlpha = kn * (1 - col);
    c.fillStyle = INK;
    c.font = font(F.mono(500), P ? 16 : 14); c.letterSpacing = '1px';
    const lab = on(t, T.v2b, 0.8);
    c.globalAlpha = kn * (1 - col) * lab;
    c.fillText('EACH KNOB IS ONE WEIGHT: ONE NUMBER THE NETWORK MULTIPLIES BY', x0, y0 - 18);
    const rnd = on(t, T.v2d, 0.8);
    if (rnd > 0) { c.globalAlpha = kn * (1 - col) * rnd; c.fillText('ALL SET AT RANDOM TO START', x0, y0 + rows * gy + 30); }
    c.restore();
  }

  private landLabels(c: CanvasRenderingContext2D, t: number, o: { rise: number; step: number; bridge: number; landOn: number; py0: number }) {
    const T = this.T, m = this.m, P = this.P, S = m.surf;
    c.save();
    c.font = font(F.mono(500), P ? 16 : 14); c.letterSpacing = '1px';
    const A = (1 - o.bridge) * on(t, T.v3d + 1.5, 1.2);
    // the honest caption: what this picture is
    if (A > 0.01) {
      const cs = [[S.a0, S.b0], [S.a1, S.b0], [S.a0, S.b1], [S.a1, S.b1]].map(([a, b]) => { const [x, z] = this.xz(a!, b!); return this.proj(x, -0.35, z); });
      const low = Math.max(...cs.map((q) => q[1])), cx = cs.reduce((s2, q) => s2 + q[0], 0) / 4;
      c.globalAlpha = A; c.fillStyle = INK; c.textAlign = 'center';
      const yMax = P ? o.py0 - 64 : H - 70;
      c.fillText('A 2D SLICE OF A 337-DIMENSIONAL LANDSCAPE', cx, Math.min(yMax, low + 30));
      c.fillText('(THE TWO DIRECTIONS THE WEIGHTS MOVED MOST)', cx, Math.min(yMax + 20, low + 50));
      c.textAlign = 'left';
    }
    // height = loss (verse 3's last line), then a small ruler
    const hA = win(t, T.v3d + 0.5, T.c0 + 2.0, 0.8) * (1 - o.bridge);
    if (hA > 0.01) {
      const [ba, bb] = m.pathAt(o.step), [x, z] = this.xz(ba, bb), y = this.yAt(ba, bb, o.rise);
      const p = this.proj(x, y + 0.5, z), q = this.proj(x, 0, z);
      c.globalAlpha = hA; c.strokeStyle = PINK; c.lineWidth = 3;
      c.beginPath(); c.moveTo(q[0], q[1]); c.lineTo(p[0], p[1]); c.stroke();
      c.fillStyle = PINK; c.font = font(F.archivo(100, 800), P ? 30 : 26);
      c.fillText('HEIGHT = LOSS', p[0] + 16, p[1]);
    }
    // verse 4: the words on the arrow
    const gA = win(t, T.v4c, T.c1 + 0.5, 0.6), lA = win(t, T.v4d, T.c1 + 1.0, 0.6);
    if (gA > 0.01 || lA > 0.01) {
      const [ba, bb] = m.pathAt(o.step), [ga, gb] = m.downhillAt(o.step), gn = Math.hypot(ga, gb) + 1e-9;
      const a = ba + (ga / gn) * 1.9, b = bb + (gb / gn) * 1.9, [x, z] = this.xz(a, b);
      const p = this.proj(x, this.yAt(a, b, o.rise) + 0.2, z);
      c.font = font(F.archivo(100, 800), P ? 30 : 26);
      if (gA > 0.01) { c.globalAlpha = gA; c.fillStyle = PINK; c.fillText('GRADIENT', p[0] + 18, p[1] - 8); }
      if (lA > 0.01) {
        c.globalAlpha = lA; c.fillStyle = INK; c.font = font(F.mono(600), P ? 20 : 18);
        c.fillText('ONE STEP = 0.3 × THE SLOPE', p[0] + 18, p[1] + 22);
      }
    }
    // the start marker (on the map)
    const sA = win(t, T.v3c + 1.0, T.v4 - 1.0, 0.8) * (1 - o.bridge);
    if (sA > 0.01) {
      const [a, b] = m.pathAt(0), [x, z] = this.xz(a, b);
      const p = this.proj(x, this.yAt(a, b, o.rise), z);
      c.globalAlpha = sA; c.fillStyle = INK; c.font = font(F.mono(600), P ? 16 : 14);
      c.textAlign = 'right'; c.fillText('START: RANDOM WEIGHTS', p[0] - 26, p[1] + 40); c.textAlign = 'left';
    }
    c.restore();
  }

  /** The loss curve: loss vs step, the real run (and in the bridge, the too-big run beside it). */
  private drawCurve(c: CanvasRenderingContext2D, t: number, A: number, o: { px0: number; py0: number; ps: number; step: number; bigStep: number }, big: boolean) {
    const m = this.m, P = this.P;
    const x0 = big ? (P ? 110 : 150) : o.px0, w = big ? (P ? 860 : 900) : o.ps;
    const y1 = big ? (P ? 900 : 860) : o.py0 + o.ps + (P ? 170 : 150), h = big ? (P ? 420 : 380) : (P ? 120 : 100);
    const y0 = y1 - h, Lmax = big ? 0.9 : 0.75;
    const X = (s: number) => x0 + (s / 5000) * w, Y = (L: number) => y1 - (Math.min(L, Lmax) / Lmax) * h;
    c.save();
    c.globalAlpha = A;
    c.strokeStyle = INK; c.lineWidth = 1.5;
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x0, y1); c.lineTo(x0 + w, y1); c.stroke();
    c.font = font(F.mono(500), big ? (P ? 18 : 17) : (P ? 15 : 13)); c.fillStyle = INK; c.letterSpacing = '1px';
    c.fillText('LOSS', x0 + 6, y0 + 4);
    c.textAlign = 'right'; c.fillText('STEP 5,000', x0 + w, y1 + 20); c.textAlign = 'left'; c.fillText('0', x0 - 2, y1 + 20);
    // the real run (lr 0.3), drawn up to the current step
    const upto = big ? 5000 : Math.min(5000, o.step);
    c.strokeStyle = big ? INK : PINK; c.lineWidth = big ? 2 : 3;
    c.beginPath();
    for (let s = 0; s <= upto; s += 10) { const x = X(s), y = Y(m.loss[s]!); if (s === 0) c.moveTo(x, y); else c.lineTo(x, y); }
    c.stroke();
    if (big) {
      // the too-big run (lr 3.0), drawn as its clock runs
      c.strokeStyle = PINK; c.lineWidth = 1.6;
      c.beginPath();
      const ub = Math.min(5000, o.bigStep);
      for (let s = 0; s <= ub; s += 2) { const x = X(s), y = Y(m.lossBig[s]!); if (s === 0) c.moveTo(x, y); else c.lineTo(x, y); }
      c.stroke();
      c.font = font(F.mono(700), P ? 20 : 19);
      c.fillStyle = INK; c.fillText('LEARNING RATE 0.3', X(2900), Y(0.14));
      c.fillStyle = PINK; c.fillText('LEARNING RATE 3.0', X(2900), Y(0.9) - 14);
    }
    c.restore();
  }

  private drawBridge(c: CanvasRenderingContext2D, t: number, o: { bridge: number; bigStep: number; px0: number; py0: number; ps: number; step: number }) {
    const T = this.T, m = this.m, P = this.P, f = m.f;
    const A = o.bridge;
    // the cut: loss along the too-big run's own downhill direction at its step 9 (real), step length 0..4
    const cutA = A * (1 - on(t, T.brd - 0.2, 1.0));
    const x0 = P ? 110 : 150, w = P ? 860 : 900, y1 = P ? 900 : 860, h = P ? 420 : 380, y0 = y1 - h;
    const Lmin = 0.6, Lmax = 1.02;
    const X = (e: number) => x0 + (e / 4) * w, Y = (L: number) => y1 - clamp((L - Lmin) / (Lmax - Lmin), 0, 1.05) * h;
    const cutL = (e: number) => { const k = (e / 4) * (m.cut.L.length - 1), i = Math.floor(k); return m.cut.L[Math.min(i, m.cut.L.length - 1)]! * (1 - (k - i)) + (m.cut.L[Math.min(i + 1, m.cut.L.length - 1)] ?? 0) * (k - i); };
    if (cutA > 0.01) {
      c.save();
      c.globalAlpha = cutA;
      c.font = font(F.mono(600), P ? 20 : 19); c.fillStyle = INK; c.letterSpacing = '1px';
      c.fillText('THE LANDSCAPE CUT OPEN ALONG THE DOWNHILL DIRECTION', x0, y0 - 46);
      c.font = font(F.mono(500), P ? 17 : 16);
      c.fillText(`(THE TOO-BIG RUN, AT ITS STEP ${f.cut_step})`, x0, y0 - 20);
      c.save(); c.translate(x0 - 22, y0 + h * 0.5); c.rotate(-Math.PI / 2); c.textAlign = 'center'; c.fillText('LOSS', 0, 0); c.restore();
      c.beginPath(); c.moveTo(x0 - 8, y0); c.lineTo(x0 - 8, y1 + 8); c.lineWidth = 1.5; c.strokeStyle = INK; c.stroke();
      // the valley profile, drawn in
      const drawn = on(t, T.br - 0.2, 2.0);
      c.strokeStyle = INK; c.lineWidth = 3;
      c.beginPath();
      for (let i = 0; i <= 200 * drawn; i++) { const e = (i / 200) * 4, x = X(e), y = Y(cutL(e)); if (i === 0) c.moveTo(x, y); else c.lineTo(x, y); }
      c.stroke();
      c.lineWidth = 1.5; c.beginPath(); c.moveTo(x0, y1 + 8); c.lineTo(x0 + w, y1 + 8); c.stroke();
      c.font = font(F.mono(500), P ? 17 : 16);
      c.textAlign = 'right'; c.fillText('HOW BIG A STEP →', x0 + w, y1 + 34); c.textAlign = 'left';
      // where it stands, the small step, the big step
      const bx = X(0), by = Y(f.cut_loss);
      c.fillStyle = PINK; c.beginPath(); c.arc(bx, by - 9, 9, 0, TAU); c.fill();
      const sA = on(t, T.br + 1.0, 0.8), jump = ease.inOutCubic(clamp((t - T.brb) / 1.6)), land = on(t, T.brc, 0.8);
      if (sA > 0.01) {
        // the 0.3 step: a short ink hop
        const e = f.lr, x = X(e), y = Y(f.cut_small);
        c.globalAlpha = cutA * sA; c.strokeStyle = INK; c.lineWidth = 2; c.setLineDash([6, 6]);
        c.beginPath(); c.moveTo(bx, by - 9); c.quadraticCurveTo((bx + x) / 2, by - 50, x, y - 9); c.stroke(); c.setLineDash([]);
        c.fillStyle = INK; c.beginPath(); c.arc(x, y - 9, 7, 0, TAU); c.fill();
        c.font = font(F.mono(700), P ? 20 : 19); c.fillText('0.3', x + 12, y - 24);
      }
      if (jump > 0.01) {
        // the 3.0 step: over the valley floor to the far wall
        const ex = X(f.lr_big), ey = Y(f.cut_big);
        c.globalAlpha = cutA; c.strokeStyle = PINK; c.lineWidth = 4;
        c.beginPath();
        const n = 60;
        for (let i = 0; i <= n * jump; i++) {
          const u = i / n, x = lerp(bx, ex, u), y = lerp(by - 9, ey - 9, u) - Math.sin(Math.PI * u) * h * 0.32;
          if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
        }
        c.stroke();
        c.font = font(F.mono(700), P ? 20 : 19); c.fillStyle = PINK; c.fillText('3.0', ex + 16, ey - 20);
        if (land > 0.01) { c.globalAlpha = cutA * land; c.beginPath(); c.arc(ex, ey - 9, 9, 0, TAU); c.fill(); }
      }
      c.restore();
    }
    // then the two runs, side by side: the loss of each over 5,000 steps
    const runA = A * on(t, T.brd - 0.2, 1.0);
    if (runA > 0.01) this.drawCurve(c, t, runA, o, true);
  }

  /** The one featured number (big, top of the frame's free side), with its caption, crossfading as the lyric moves. */
  private drawFeature(c: CanvasRenderingContext2D, t: number, o: { step: number; bigStep: number; px0: number; py0: number; ps: number }) {
    const T = this.T, m = this.m, P = this.P, f = m.f;
    type Ft = { a: number; b: number; big: string; cap: string; pink?: boolean; roll?: number };
    const L = (s: number) => fmtLoss(m.lossAt(s));
    const list: Ft[] = [
      { a: T.v1d, b: T.v2 - 0.4, big: `${f.wrong0}`, cap: 'DOTS ON THE WRONG SIDE', pink: true },
      { a: T.v2b, b: T.v3 - 0.4, big: `${f.params}`, cap: 'WEIGHTS' },
      { a: T.v3b, b: T.c0c, big: L(o.step), cap: 'LOSS: HOW WRONG, IN ONE NUMBER', pink: true },
    ];
    for (let n = 0; n < 3; n++) {
      const c0 = T[`c${n}`]!, cc = T[`c${n}c`]!, ce = T[`c${n}e`]!;
      if (!Number.isFinite(c0)) continue;
      if (n > 0) list.push({ a: c0 - 0.3, b: cc, big: L(o.step), cap: 'LOSS', pink: true });
      list.push({ a: cc, b: n === 0 ? T.v4 - 0.3 : n === 1 ? T.br - 0.4 : T.out - 0.3, big: fmtInt(Math.floor(o.step)), cap: Math.round(o.step) === 1 ? 'STEP' : 'STEPS', roll: o.step });
      void ce;
    }
    list.push({ a: T.v4d, b: T.c1 - 0.3, big: f.lr.toFixed(1), cap: 'LEARNING RATE', pink: true });
    list.push({ a: T.br + 0.3, b: T.brc, big: f.lr_big.toFixed(1), cap: 'LEARNING RATE: TEN TIMES AS BIG', pink: true });
    list.push({ a: T.brc, b: T.brd, big: fmtLoss(f.cut_big), cap: `LOSS AFTER THE JUMP (IT WAS ${fmtLoss(f.cut_loss)})`, pink: true });
    list.push({ a: T.brd, b: this.ctx.story.brx, big: `${Math.round(f.big_final_acc * 100)}%`, cap: 'RIGHT AFTER 5,000 TOO-BIG STEPS', pink: true });
    list.push({ a: T.steps - 0.2, b: T.outb, big: fmtInt(f.first_all_right), cap: 'STEPS', pink: true });
    list.push({ a: T.outb, b: T.end + 9, big: `${f.points - m.wrongAt(o.step)} / ${f.points}`, cap: 'DOTS ON THEIR OWN SIDE', pink: true });
    const side = on(t, T.v2 - 1.2, 2.2);
    const x = P ? 90 : lerp(o.px0 - 420, o.px0, side), y = P ? 250 : lerp(o.py0 + 120, o.py0 - 70, side);
    c.save();
    for (const it of list) {
      const a = win(t, it.a, it.b, P ? 0.6 : 0.4);
      if (a <= 0.01) continue;
      c.globalAlpha = a;
      const dy = (1 - a) * 14;
      const yy = y;
      c.fillStyle = it.pink ? PINK : INK;
      c.font = font(F.archivo(112, 900), P ? 76 : 64);
      let w = c.measureText(it.big).width;
      if (it.roll !== undefined && it.roll < 20 && it.roll % 1 > 1e-3) {
        // the first few steps tick over (roll), not pop
        const k = ease.inOutCubic(it.roll % 1), n0 = Math.floor(it.roll);
        c.globalAlpha = a * (1 - k); c.fillText(fmtInt(n0), x, yy + dy - 18 * k);
        c.globalAlpha = a * k; c.fillText(fmtInt(n0 + 1), x, yy + dy + 18 * (1 - k));
        c.globalAlpha = a;
        w = Math.max(w, c.measureText(fmtInt(n0 + 1)).width);
      } else c.fillText(it.big, x, yy + dy);
      c.fillStyle = INK; c.font = font(F.mono(600), P ? 18 : 16); c.letterSpacing = '1.5px';
      c.fillText(it.cap, x + w + 18, yy + dy - 4);
      c.letterSpacing = '0px';
    }
    c.restore();
  }
}
