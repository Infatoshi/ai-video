// An illustrated strawberry for the thumbnails, drawn flat in the riso inks (it reads at phone size and prints
// like poster art): a pink body with a blue shadow overprinted, paper-coloured seed knockouts with a black tick,
// a blue calyx and stem. `drawSliced` cuts it across its length into chunks with pale flesh at each cut
// ("str | aw | berry"). Local units: the body spans about -1..1; +y is the tip.
import { rgba } from '../engine/palette';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

function bodyPath(c: CanvasRenderingContext2D) {
  c.beginPath();
  c.moveTo(0, -0.62);
  c.bezierCurveTo(0.42, -0.78, 0.86, -0.62, 0.9, -0.2);
  c.bezierCurveTo(0.94, 0.22, 0.52, 0.66, 0.1, 0.94);
  c.quadraticCurveTo(0, 1.0, -0.1, 0.94);
  c.bezierCurveTo(-0.52, 0.66, -0.94, 0.22, -0.9, -0.2);
  c.bezierCurveTo(-0.86, -0.62, -0.42, -0.78, 0, -0.62);
  c.closePath();
}

function body(c: CanvasRenderingContext2D) {
  bodyPath(c);
  c.fillStyle = PINK; c.fill();
  // volume: blue overprinted in a crescent on the shadow side (prints purple)
  c.save(); bodyPath(c); c.clip();
  c.globalCompositeOperation = 'multiply';
  const g = c.createRadialGradient(-0.3, -0.25, 0.2, -0.1, 0.05, 1.25);
  g.addColorStop(0, 'rgba(0,120,191,0)'); g.addColorStop(0.62, 'rgba(0,120,191,0.12)'); g.addColorStop(1, 'rgba(0,120,191,0.85)');
  c.fillStyle = g; c.fillRect(-1.2, -1.2, 2.4, 2.4);
  c.globalCompositeOperation = 'source-over';
  // a paper highlight on the lit shoulder
  const h = c.createRadialGradient(-0.42, -0.3, 0, -0.42, -0.3, 0.42);
  h.addColorStop(0, 'rgba(242,239,230,0.75)'); h.addColorStop(1, 'rgba(242,239,230,0)');
  c.fillStyle = h; c.fillRect(-1.2, -1.2, 2.4, 2.4);
  // seeds: offset rows, each a paper teardrop knocked out of the pink with a black tick
  for (let row = 0; row < 9; row++) {
    const y = -0.46 + row * 0.16, n = 7 - Math.abs(row - 2) * 0.5;
    for (let k = 0; k < 8; k++) {
      const x = -0.84 + (k + (row % 2) * 0.5) * 0.23;
      const s = 0.05 * (1 - Math.max(0, y) * 0.35);
      c.save(); c.translate(x, y); c.rotate(x * 0.35);
      c.beginPath(); c.ellipse(0, 0, s * 0.62, s, 0, 0, Math.PI * 2); c.fillStyle = PAPER; c.fill();
      c.beginPath(); c.ellipse(0, s * 0.25, s * 0.28, s * 0.5, 0, 0, Math.PI * 2); c.fillStyle = INK; c.fill();
      c.restore();
      void n;
    }
  }
  c.restore();
  bodyPath(c); c.lineWidth = 0.035; c.strokeStyle = INK; c.stroke();
}

function calyx(c: CanvasRenderingContext2D) {
  c.save(); c.translate(0, -0.66);
  c.fillStyle = BLUE; c.strokeStyle = INK; c.lineWidth = 0.03;
  for (let i = 0; i < 7; i++) {
    const a = -Math.PI / 2 + (i - 3) * 0.42;
    c.save(); c.rotate(a + Math.PI / 2);
    c.beginPath(); c.moveTo(0, 0); c.quadraticCurveTo(0.14, 0.22, 0, 0.5 - Math.abs(i - 3) * 0.05); c.quadraticCurveTo(-0.14, 0.22, 0, 0);
    c.fill(); c.stroke(); c.restore();
  }
  c.beginPath(); c.moveTo(-0.05, 0.02); c.quadraticCurveTo(-0.08, -0.25, 0.06, -0.38); c.lineTo(0.13, -0.33); c.quadraticCurveTo(0.02, -0.2, 0.06, 0.02); c.closePath();
  c.fill(); c.stroke();
  c.restore();
}

/** The whole strawberry at (x, y), `size` px tall-ish, rotated `rot` radians. */
export function drawStrawberry(c: CanvasRenderingContext2D, x: number, y: number, size: number, rot = 0) {
  c.save(); c.translate(x, y); c.rotate(rot); c.scale(size / 2, size / 2);
  body(c); calyx(c);
  c.restore();
}

/**
 * The strawberry lying on its side (calyx left, tip right), cut across its length into chunks of the given
 * fractions, spread apart by `gap` px; pale flesh at every cut. Returns each chunk's centre x in px.
 */
export function drawSliced(c: CanvasRenderingContext2D, x: number, y: number, size: number, fractions: number[], gap: number): number[] {
  const s = size / 2;
  // along the lying berry, local u runs from the calyx (-0.95) to the tip (+1.0)
  const u0 = -0.98, u1 = 1.0;
  const cuts = [u0];
  fractions.forEach((f) => cuts.push(cuts[cuts.length - 1]! + f * (u1 - u0)));
  const centres: number[] = [];
  fractions.forEach((_, i) => {
    const a = cuts[i]!, b = cuts[i + 1]!;
    const off = (i - (fractions.length - 1) / 2) * gap;
    c.save();
    c.translate(x + off, y);
    const ca = i === 0 ? -1.6 : a, cb = i === fractions.length - 1 ? 1.3 : b;
    c.beginPath(); c.rect(ca * s, -1.3 * s, (cb - ca) * s, 2.6 * s); c.clip();
    // rotate(-90deg): the calyx (local -y) to the left, the tip (local +y) to the right
    c.save(); c.rotate(-Math.PI / 2); c.scale(s, s); body(c); if (i === 0) calyx(c); c.restore();
    // flesh at the cut edges: a pale band with a paler core line
    for (const e of [i > 0 ? a : null, i < fractions.length - 1 ? b : null]) {
      if (e === null) continue;
      c.save();
      c.save(); c.rotate(-Math.PI / 2); c.scale(s, s); bodyPath(c); c.restore(); c.clip();
      const w = 0.07 * s, x0 = e * s - (e === a ? 0 : w);
      c.fillStyle = 'rgba(255,196,222,1)'; c.fillRect(x0, -1.3 * s, w, 2.6 * s);
      c.fillStyle = PAPER; c.fillRect(x0 + w * 0.35, -0.35 * s, w * 0.3, 0.7 * s);
      c.restore();
    }
    c.restore();
    centres.push(x + off + ((a + b) / 2) * s);
  });
  return centres;
}
