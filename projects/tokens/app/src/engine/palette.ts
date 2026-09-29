import { hexToLinear } from './util';

// Risograph print: three inks on off-white paper. Everything on screen is one of these inks (solid or
// as a tint), or two of them overprinted. The post pass separates each frame into the three inks and
// halftones them (post.ts), so colours outside this set print as muddy black: stay inside it.
export const HEX = {
  paper: '#F2EFE6', // the sheet: every frame's white
  pink: '#FF48B0', // fluorescent pink: the hero ink (the token, the hook, what is being sung)
  blue: '#0078BF', // riso blue: structure (the tower, the grid, the model)
  ink: '#231F20', // black: type, outlines, numbers
  purple: '#3E2D8E', // pink over blue (overprint, for reference; print it by drawing both)
  tint: '#C9C4BC', // black at ~20% (rules, secondary lines)
  // v5 names, kept as aliases so shared helpers keep compiling
  bone: '#F2EFE6',
  ink2: '#E4DFD4',
  graphite: '#8F8A83',
  ash: '#C9C4BC',
  signal: '#FF48B0',
  ember: '#FF48B0',
  blood: '#3E2D8E',
  acid: '#0078BF',
} as const;

export type PaletteKey = keyof typeof HEX;

/** Linear RGB triplets for GL uniforms. */
export const LIN: Record<PaletteKey, [number, number, number]> = Object.fromEntries(
  Object.entries(HEX).map(([k, v]) => [k, hexToLinear(v)]),
) as Record<PaletteKey, [number, number, number]>;

/** CSS rgba() for Canvas2D. */
export function rgba(key: PaletteKey | string, a = 1): string {
  const hex = (HEX as Record<string, string>)[key] ?? key;
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
