// Global post-processing: the risograph print. The composited linear frame is
// separated into three inks (fluorescent pink, riso blue, black), each ink is halftoned on its own
// screen angle and printed slightly out of register, onto off-white paper with fibre and ink mottle.
// No bloom: brightness tops out at the paper. The overlay (hud.ts) goes on after the print, crisp.
import * as THREE from 'three';
import { FSPass, W, H } from './gl';
import { LIN } from './palette';

/** The tone shoulder (linear HDR -> 0..1 linear), shared with the engine's sampling error estimate. */
export const SHOULDER_GLSL = /* glsl */ `
vec3 shoulder(vec3 x) { return clamp(x, 0.0, 1.0); }`;

export interface PostParams {
  exposure: number;
  /** HUD opacity (0 hides the model HUD and registration marks). */
  hud: number;
  /** Halftone cell in logical px (8 at 1080p). */
  cell: number;
  /** Plate misregistration in logical px. */
  misreg: number;
  /** Registration / paper seed: the kit's shot cutter changes it on every cut, so each shot is a new print. */
  reg: number;
  /** Paper fibre and ink mottle strength (0..1). */
  grain: number;
  /** 0..1: skip the halftone (solid inks; for a clean readable moment). */
  solid: number;
  /** Fade to blank paper 0..1. */
  fade: number;
  /** Flash to paper white 0..1 (a white-out hit). */
  flash: number;
  /** 0..1: flood the sheet with pink ink (a hit). */
  flood: number;
  shake: [number, number];
  zoom: number;
  /** 0..1: print the negative (paper <-> black). */
  invert: number;
  /** Overlay switches (1 on, 0 off): the concept-word slams and the sung lyric line (hud.ts). A scene
   * that stages the word itself turns `kinetic` off for that moment. */
  kinetic: number;
  lyric: number;
}

export const DEFAULT_POST: PostParams = {
  exposure: 1,
  hud: 1,
  cell: 8,
  misreg: 1.6,
  reg: 0,
  grain: 1,
  solid: 0,
  fade: 0,
  flash: 0,
  flood: 0,
  shake: [0, 0],
  zoom: 1,
  invert: 0,
  kinetic: 1,
  lyric: 1,
};

/** Ink colours clamped away from 0 so their optical density is finite (a printed ink never reaches 0). */
const inkLin = (k: 'pink' | 'blue' | 'ink') => LIN[k].map((x) => Math.max(x, 0.03)) as [number, number, number];

export class Post {
  private final: FSPass;

  constructor() {
    const paper = LIN.paper, inks = [inkLin('pink'), inkLin('blue'), inkLin('ink')];
    // separation: optical density of a colour relative to paper = sum_i d_i * OD(ink_i); solve for d
    const M = new THREE.Matrix3().set(
      ...([0, 1, 2].flatMap((row) => inks.map((k) => -Math.log(k[row]! / paper[row]!))) as [number, number, number, number, number, number, number, number, number]),
    );
    const Mi = M.clone().invert();
    const lum = (c: number[]) => 0.2126 * c[0]! + 0.7152 * c[1]! + 0.0722 * c[2]!;
    const kl = inks.map((k) => lum(k.map((x, i) => x / paper[i]!)));
    const v3 = (c: number[]) => new THREE.Vector3(c[0], c[1], c[2]);
    this.final = new FSPass(/* glsl */ `
      uniform sampler2D src; uniform sampler2D hudTex;
      uniform mat3 sepM; uniform vec3 paper, inkP, inkB, inkK, kLum;
      uniform float exposure, hud, cell, misreg, reg, grain, solid, fade, flash, flood, zoom, invert;
      uniform vec2 shake; uniform vec2 res;

      vec3 frameAt(vec2 uv) {
        vec3 c = texture(src, uv).rgb * exposure;
        c = mix(c, max(paper + inkK - c, inkK), invert);
        return c;
      }
      // ink amounts (pink, blue, black) as area coverage 0..1
      vec3 coverage(vec3 c) {
        vec3 od = -log(clamp(min(c, paper), 1e-4, 1.0) / paper);
        vec3 d = clamp(sepM * od, 0.0, 1.5);
        // density -> halftone area (a dot of coverage a reflects paper*(1-a) + ink*a)
        return clamp((1.0 - pow(kLum, d)) / (1.0 - kLum), 0.0, 1.0);
      }
      float screen(vec2 px, float ang, float a) {
        vec2 q = rot2(ang) * px / cell;
        float spot = 0.5 + 0.25 * (cos(TAU * q.x) + cos(TAU * q.y));
        float e = fwidth(spot) * 0.75 + 1e-4;
        float h = smoothstep(1.0 - a - e, 1.0 - a + e, spot);
        h = mix(h, 1.0, smoothstep(0.93, 0.99, a));
        return mix(h, a, solid) * step(0.004, a);
      }
      vec2 regOff(float k) { return (hash22(vec2(reg * 7.13 + k * 3.7, k + 1.3)) - 0.5) * 2.0 * misreg; }

      void main() {
        vec2 uv = (vUv - 0.5) / zoom + 0.5 - shake / res;
        vec2 px = FRAG_PX;
        vec2 oP = regOff(1.0), oB = regOff(2.0), oK = regOff(3.0) * 0.35;
        float aP = coverage(frameAt(uv + oP / res)).x;
        float aB = coverage(frameAt(uv + oB / res)).y;
        float aK = coverage(frameAt(uv + oK / res)).z;
        aP = min(1.0, aP + flood);
        // ink mottle: coverage breathes a little, fixed to the sheet (reseeded per print)
        vec2 sp = px + reg * 97.0;
        float mot = snoise(sp / 55.0) * 0.6 + snoise(sp / 13.0) * 0.4;
        aP = clamp(aP * (1.0 + 0.10 * grain * mot), 0.0, 1.0);
        aB = clamp(aB * (1.0 - 0.10 * grain * mot), 0.0, 1.0);
        float hP = screen(px + oP, 0.2618, aP);   // 15 deg
        float hB = screen(px + oB, 1.3090, aB);   // 75 deg
        float hK = screen(px + oK, 0.7854, aK);   // 45 deg
        vec3 col = paper;
        col *= mix(vec3(1.0), inkP / paper, hP);
        col *= mix(vec3(1.0), inkB / paper, hB);
        col *= mix(vec3(1.0), inkK / paper, hK);
        // paper: long fibres and a faint tooth
        float fib = snoise(vec2(sp.x / 240.0, sp.y / 7.0)) * 0.5 + snoise(sp / 3.0) * 0.5;
        col *= 1.0 - grain * (0.022 * fib + 0.012);
        // the overlay (HUD, sung line, slams) prints last and crisp: solid ink, never halftoned or misregistered
        vec4 h = texture(hudTex, vUv);
        col = mix(col, h.rgb, h.a * hud); // straight-alpha CanvasTexture: rgb is not premultiplied, so no divide
        col = mix(col, paper, max(fade, flash));
        vec3 s = toSRGB(sat(col));
        s += (hash12(gl_FragCoord.xy * 1.37) - 0.5) / 255.0; // dither
        fragColor = vec4(sat(s), 1.0);
      }`, {
      src: { value: null }, hudTex: { value: null },
      sepM: { value: Mi }, paper: { value: v3(paper) }, inkP: { value: v3(inks[0]!) }, inkB: { value: v3(inks[1]!) }, inkK: { value: v3(inks[2]!) },
      kLum: { value: v3(kl) },
      exposure: { value: 1 }, hud: { value: 1 }, cell: { value: 8 }, misreg: { value: 1.6 }, reg: { value: 0 }, grain: { value: 1 },
      solid: { value: 0 }, fade: { value: 0 }, flash: { value: 0 }, flood: { value: 0 }, zoom: { value: 1 }, invert: { value: 0 },
      shake: { value: new THREE.Vector2() }, res: { value: new THREE.Vector2(W, H) },
    });
  }

  /** Apply the print: src (linear) + HUD -> out (sRGB 8-bit target or screen). */
  render(renderer: THREE.WebGLRenderer, src: THREE.Texture, hud: THREE.Texture, out: THREE.WebGLRenderTarget | null, p: PostParams, _time: number) {
    const f = this.final.u;
    f.src!.value = src;
    f.hudTex!.value = hud;
    f.exposure!.value = p.exposure;
    f.hud!.value = p.hud;
    f.cell!.value = p.cell;
    f.misreg!.value = p.misreg;
    f.reg!.value = p.reg;
    f.grain!.value = p.grain;
    f.solid!.value = p.solid;
    f.fade!.value = p.fade;
    f.flash!.value = p.flash;
    f.flood!.value = p.flood;
    f.zoom!.value = p.zoom;
    f.invert!.value = p.invert;
    (f.shake!.value as THREE.Vector2).set(p.shake[0], p.shake[1]);
    this.final.render(renderer, out);
  }
}
