// Transitions between overlapping timeline entries (v5). The engine renders both scenes over the
// overlap and composites them here with `k` = 0..1 progress through the overlap. Quick kinds switch
// on the beat at k = 0.5 (whip, glitch, flash, pixel); the big ones travel (dive, streak, shatter,
// iris). All inputs and outputs are linear HDR.
import * as THREE from 'three';
import { FSPass } from './gl';

export type TransitionKind = 'xfade' | 'dive' | 'streak' | 'shatter' | 'glitch' | 'whip' | 'flash' | 'iris' | 'pixel';
const KIND: Record<TransitionKind, number> = { xfade: 0, dive: 1, streak: 2, shatter: 3, glitch: 4, whip: 5, flash: 6, iris: 7, pixel: 8 };

export interface TransitionSpec {
  kind: TransitionKind;
  /** Overlap length in seconds (the timeline centres it on the cut; see `pre`). */
  dur?: number;
  /** Fraction of the overlap before the cut (default 0.5). */
  pre?: number;
  /** Focus point in uv (0..1, y up) for dive / streak / iris / shatter. */
  center?: [number, number];
  /** Direction for whip (uv units per unit, default left-to-right). */
  dir?: [number, number];
}

const FRAG = /* glsl */ `
uniform sampler2D a;      // outgoing scene
uniform sampler2D b;      // incoming scene
uniform float k;          // 0..1 through the overlap
uniform int kind;
uniform vec2 center;
uniform vec2 dir;
uniform float aspect;     // W / H
uniform float seed;

vec3 A(vec2 uv) { return texture(a, clamp(uv, 0.0, 1.0)).rgb; }
vec3 B(vec2 uv) { return texture(b, clamp(uv, 0.0, 1.0)).rgb; }

// radial blur toward/away from c: n taps spread over [s0, s1] scales
vec3 radialA(vec2 uv, vec2 c, float amt) {
  vec3 s = vec3(0.0);
  for (int i = 0; i < 16; i++) { float f = float(i) / 15.0; s += A(c + (uv - c) * (1.0 - amt * f)); }
  return s / 16.0;
}
vec3 radialB(vec2 uv, vec2 c, float amt) {
  vec3 s = vec3(0.0);
  for (int i = 0; i < 16; i++) { float f = float(i) / 15.0; s += B(c + (uv - c) * (1.0 - amt * f)); }
  return s / 16.0;
}
float ease3(float x) { return x < 0.5 ? 4.0 * x * x * x : 1.0 - pow(-2.0 * x + 2.0, 3.0) / 2.0; }

void main() {
  vec2 uv = vUv;
  vec3 col;
  if (kind == 0) {                                   // xfade
    col = mix(A(uv), B(uv), k);
  } else if (kind == 1) {                            // dive: fly into the focus point, the next scene opens out of it
    float e = ease3(k);
    vec2 c = center;
    if (k < 0.5) {
      float z = 1.0 + 9.0 * pow(e * 2.0, 2.2);       // outgoing: scale up around c
      vec2 u = c + (uv - c) / z;
      col = radialA(u, c, 0.35 * e * 2.0);
    } else {
      float z = mix(0.15, 1.0, pow((e - 0.5) * 2.0, 0.6));
      vec2 u = c + (uv - c) / z;
      bool inside = all(greaterThanEqual(u, vec2(0.0))) && all(lessThanEqual(u, vec2(1.0)));
      col = inside ? radialB(u, c, 0.25 * (1.0 - (e - 0.5) * 2.0)) : vec3(0.0);
    }
    float flash = exp(-pow((k - 0.5) / 0.03, 2.0));   // a warm hit on the cut, not a white frame
    col += C_EMBER * flash * 1.1 + vec3(1.0) * flash * 0.12;
  } else if (kind == 2) {                            // streak: light-speed radial smear, hard switch at 0.5
    float amt = 0.85 * exp(-pow((k - 0.5) / 0.22, 2.0));
    col = k < 0.5 ? radialA(uv, center, amt) : radialB(uv, center, amt);
    vec2 d = (uv - center) * vec2(aspect, 1.0);
    float ang = atan(d.y, d.x);
    float lines = pow(hash11(floor(ang * 240.0 / 6.2832) + seed), 18.0);
    float r = length(d);
    col += C_SIGNAL * lines * amt * 3.0 * smoothstep(0.05, 0.6, r);
    col *= 1.0 + 1.6 * amt;
  } else if (kind == 3) {                            // shatter: the outgoing frame breaks into shards that fall away
    vec2 grid = vec2(16.0, 9.0);
    vec2 p = uv * grid;
    vec2 cell = floor(p);
    vec2 cc = (cell + 0.5) / grid;
    float delay = 0.45 * length((cc - center) * vec2(aspect, 1.0)) / 1.2 + 0.08 * hash12(cell + seed);
    float t = clamp((k - delay) / 0.5, 0.0, 1.0);
    t = t * t;
    float s = 1.0 - t;                               // shard scale
    float rot = (hash12(cell + 7.0 + seed) - 0.5) * 3.0 * t;
    vec2 fall = vec2((hash12(cell + 3.0) - 0.5) * 0.3, -0.9) * t * t;
    vec2 q = (uv - cc - fall) * vec2(aspect, 1.0);
    q = mat2(cos(rot), -sin(rot), sin(rot), cos(rot)) * q / max(s, 1e-3);
    q /= vec2(aspect, 1.0);
    vec2 half_ = 0.5 / grid;
    bool inShard = s > 0.001 && abs(q.x) < half_.x && abs(q.y) < half_.y;
    col = inShard ? A(cc + q) : B(uv);
    float edge = inShard ? smoothstep(0.85, 1.0, max(abs(q.x) / half_.x, abs(q.y) / half_.y)) : 0.0;
    col += C_SIGNAL * edge * 1.2 * t * (1.0 - t) * 4.0;
  } else if (kind == 4) {                            // glitch: RGB split and slice displacement, switch at 0.5
    float g = exp(-pow((k - 0.5) / 0.18, 2.0));
    float sl = floor(uv.y * 36.0);
    float jit = (hash11(sl + floor(k * 24.0) * 13.0 + seed) - 0.5) * 0.25 * g * step(0.55, hash11(sl * 3.1 + floor(k * 24.0)));
    vec2 u = vec2(uv.x + jit, uv.y);
    vec2 o = vec2(0.012 * g, 0.0);
    bool useB = k >= 0.5 ? hash12(floor(uv * vec2(24.0, 14.0)) + floor(k * 30.0)) > 0.15 * g : hash12(floor(uv * vec2(24.0, 14.0)) + floor(k * 30.0)) < 0.15 * g;
    col = useB ? vec3(B(u + o).r, B(u).g, B(u - o).b) : vec3(A(u + o).r, A(u).g, A(u - o).b);
  } else if (kind == 5) {                            // whip: directional smear, switch at 0.5
    float amt = 0.22 * exp(-pow((k - 0.5) / 0.2, 2.0));
    vec3 s = vec3(0.0);
    for (int i = 0; i < 24; i++) {
      float f = (float(i) / 23.0 - 0.5) * amt;
      vec2 u = uv + dir * f + dir * (k < 0.5 ? k : k - 1.0) * 0.6;   // out slides away, in slides home: no jump at either end
      s += k < 0.5 ? A(u) : B(u);
    }
    col = s / 24.0;
  } else if (kind == 6) {                            // flash: bloom to hot white/ember at the cut
    float f = exp(-pow((k - 0.5) / 0.12, 2.0));
    col = k < 0.5 ? A(uv) : B(uv);
    col = mix(col, C_EMBER * 3.0 + vec3(2.0), f * 0.85);
  } else if (kind == 7) {                            // iris: a hot ring opens from the focus point
    vec2 d = (uv - center) * vec2(aspect, 1.0);
    float r = length(d);
    float R = ease3(k) * 2.2;
    col = r < R ? B(uv) : A(uv);
    col += C_SIGNAL * 3.0 * exp(-pow((r - R) / 0.012, 2.0)) * (1.0 - k);
  } else {                                           // pixel: mosaic up to the cut and back down
    float m = exp(-pow((k - 0.5) / 0.2, 2.0));
    float cells = mix(1080.0, 18.0, m);
    vec2 g = vec2(cells * aspect, cells);
    vec2 u = (floor(uv * g) + 0.5) / g;
    col = k < 0.5 ? A(u) : B(u);
  }
  fragColor = vec4(col, 1.0);
}`;

export class TransitionPass {
  pass: FSPass;
  constructor(aspect: number) {
    this.pass = new FSPass(FRAG, {
      a: { value: null }, b: { value: null }, k: { value: 0 }, kind: { value: 0 },
      center: { value: new THREE.Vector2(0.5, 0.5) }, dir: { value: new THREE.Vector2(1, 0) },
      aspect: { value: aspect }, seed: { value: 0 },
    });
  }
  render(renderer: THREE.WebGLRenderer, a: THREE.Texture, b: THREE.Texture, k: number, spec: TransitionSpec, seed: number, out: THREE.WebGLRenderTarget) {
    const u = this.pass.u;
    u.a!.value = a; u.b!.value = b; u.k!.value = k;
    u.kind!.value = KIND[spec.kind] ?? 0;
    (u.center!.value as THREE.Vector2).set(...(spec.center ?? [0.5, 0.5]));
    (u.dir!.value as THREE.Vector2).set(...(spec.dir ?? [1, 0]));
    u.seed!.value = seed;
    this.pass.render(renderer, out);
  }
}
