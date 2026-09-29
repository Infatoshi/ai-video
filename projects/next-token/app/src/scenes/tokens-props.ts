// 3D props for the `tokens` scene: a keyboard (the question is typed on it, ENTER is its hero key),
// printed tiles with a front and a back face (the flip from letters to ids), and a conveyor belt that
// feeds the prompt's tiles into the tower's lobby. All printed with the kit's inks and outlines.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { HEX } from '../engine/palette';
import { F, font } from '../engine/type';
import { inkMat, inked, outline, flatMat, type InkName } from './_kit';

/** A canvas texture (sRGB, anisotropic) from a draw callback on a w x h canvas. */
export function canvasTex(w: number, h: number, draw: (c: CanvasRenderingContext2D) => void): THREE.CanvasTexture {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const c = cv.getContext('2d')!;
  draw(c);
  const tx = new THREE.CanvasTexture(cv);
  tx.colorSpace = THREE.SRGBColorSpace;
  tx.anisotropy = 8;
  return tx;
}

/**
 * A printed slab: `front` on +z, `back` on -z (turned 180 degrees so it reads upright after a flip about
 * the x axis), sides in ink with an outline.
 */
export function faceTile(front: THREE.Texture, back: THREE.Texture | null, w: number, h: number, d: number, side: InkName = 'blue', line = 2): THREE.Mesh {
  const sideM = inkMat({ ink: side, lit: 0.6, shade: 1 });
  let backM: THREE.Material = sideM;
  if (back) {
    const b = back.clone();
    b.center.set(0.5, 0.5);
    b.rotation = Math.PI;
    b.needsUpdate = true;
    backM = new THREE.MeshBasicMaterial({ map: b });
  }
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), [sideM, sideM, sideM, sideM, new THREE.MeshBasicMaterial({ map: front }), backM]);
  if (line > 0) outline(m, line);
  return m;
}

/** The face of a tile: a card in `bg` with a solid border and `label` in `fg` (mono, fitted). Optional small second line. */
export function faceTexture(label: string, bg: InkName, fg: InkName, aspect: number, sub?: string): THREE.CanvasTexture {
  const ch = 256, cw = Math.round(ch * aspect);
  return canvasTex(cw, ch, (c) => {
    c.fillStyle = HEX[bg]; c.fillRect(0, 0, cw, ch);
    c.strokeStyle = HEX[fg]; c.lineWidth = 14; c.strokeRect(7, 7, cw - 14, ch - 14);
    let size = ch * (sub ? 0.46 : 0.56);
    c.font = font(F.mono(700), size);
    const w = c.measureText(label).width;
    if (w > cw * 0.84) { size *= (cw * 0.84) / w; c.font = font(F.mono(700), size); }
    c.fillStyle = HEX[fg]; c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(label, cw / 2, ch * (sub ? 0.4 : 0.53));
    if (sub) { c.font = font(F.mono(700), ch * 0.2); c.fillText(sub, cw / 2, ch * 0.78); }
  });
}

// ------------------------------------------------------------------ keyboard

export interface Key { mesh: THREE.Mesh; mat: THREE.ShaderMaterial; label: string; x: number; z: number; w: number }

const CAP_H = 0.42;

function legend(label: string, w: number): THREE.Mesh {
  const big = label.length === 1;
  const tex = canvasTex(Math.round(160 * w), 160, (c) => {
    c.fillStyle = HEX.ink;
    c.font = font(F.archivo(100, 900), big ? 104 : 58);
    c.textAlign = big ? 'left' : 'center';
    c.textBaseline = big ? 'top' : 'middle';
    if (big) c.fillText(label, 22, 16);
    else c.fillText(label, 80 * w, 80);
  });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(0.8 * w, 0.8), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
  m.rotation.x = -Math.PI / 2;
  m.position.y = CAP_H / 2 + 0.004;
  return m;
}

/**
 * A keyboard (US layout, trimmed), centred on the origin, keys on top of a black base plate.
 * Units: one key pitch = 1. Rows run along x, front row at +z.
 */
export function buildKeyboard(): { group: THREE.Group; keys: Key[]; enter: Key } {
  const group = new THREE.Group();
  const keys: Key[] = [];
  const rows: { z: number; x0: number; keys: [string, number][] }[] = [
    { z: -1.5, x0: -7.5, keys: [['TAB', 1.5], ...'QWERTYUIOP'.split('').map((k) => [k, 1] as [string, number]), ['DEL', 2.25]] },
    { z: -0.5, x0: -7.5, keys: [['CAPS', 1.75], ...'ASDFGHJKL\''.split('').map((k) => [k, 1] as [string, number]), ['ENTER', 2]] },
    { z: 0.5, x0: -7.5, keys: [['SHIFT', 2.25], ...'ZXCVBNM,.?'.split('').map((k) => [k, 1] as [string, number]), ['SHIFT', 1.5]] },
    { z: 1.5, x0: -7.5, keys: [['FN', 1.5], ['ALT', 1.5], ['CMD', 1.75], ['SPACE', 6], ['CMD', 1.75], ['ALT', 1.25]] },
  ];
  const geos = new Map<number, THREE.BufferGeometry>();
  let enter: Key | null = null;
  for (const row of rows) {
    let x = row.x0;
    for (const [label, w] of row.keys) {
      let g = geos.get(w);
      if (!g) { g = new RoundedBoxGeometry(w - 0.14, CAP_H, 0.86, 2, 0.09); geos.set(w, g); }
      const isEnter = label === 'ENTER';
      const mat = isEnter ? inkMat({ ink: 'pink', lit: 0.3, shade: 1, over: 'pink', overLit: 0, overShade: 0 })
        : inkMat({ ink: 'blue', lit: 0.12, shade: 0.75, over: 'pink', overLit: 0, overShade: 0 });
      const mesh = new THREE.Mesh(g, mat);
      outline(mesh, isEnter ? 3 : 1.8);
      const cx = x + w / 2;
      mesh.position.set(cx, CAP_H / 2, row.z);
      mesh.add(legend(label, w));
      group.add(mesh);
      const k: Key = { mesh, mat, label, x: cx, z: row.z, w };
      keys.push(k);
      if (isEnter) enter = k;
      x += w;
    }
  }
  const base = inked(new RoundedBoxGeometry(15.4, 0.4, 4.7, 2, 0.14), { ink: 'ink', lit: 0.2, shade: 0.8 }, 2.4);
  base.position.set(0.2, -0.2, 0);
  group.add(base);
  return { group, keys, enter: enter! };
}

/** The key a typed character is on (letters case-folded; space = SPACE). */
export function keyFor(keys: Key[], ch: string): Key | undefined {
  const l = ch === ' ' ? 'SPACE' : ch.toUpperCase();
  return keys.find((k) => k.label === l);
}

// ------------------------------------------------------------------ conveyor

/**
 * A conveyor belt along +x starting at x = x0 (the lobby), `len` long, top surface at y = top. The slats
 * are instanced so the belt can visibly step (`setOffset`).
 */
export class Belt {
  group = new THREE.Group();
  slats: THREE.InstancedMesh;
  private tmp = new THREE.Object3D();
  constructor(public x0: number, public len: number, public top: number, public width = 2.6, public pitch = 0.9) {
    const body = inked(new THREE.BoxGeometry(len, 0.34, width), { ink: 'blue', lit: 0.25, shade: 1 }, 2.2);
    body.position.set(x0 + len / 2, top - 0.17, 0);
    this.group.add(body);
    for (const s of [-1, 1]) {
      const rail = inked(new THREE.BoxGeometry(len, 0.16, 0.14), { ink: 'ink', lit: 0.3, shade: 1 }, 1.6);
      rail.position.set(x0 + len / 2, top + 0.04, s * (width / 2 + 0.07));
      this.group.add(rail);
    }
    for (let x = x0 + 3; x < x0 + len; x += 5) {
      const leg = inked(new THREE.BoxGeometry(0.3, 6, 0.3), { ink: 'ink', lit: 0.3, shade: 1 }, 1.6);
      leg.position.set(x, top - 3.3, width / 2 - 0.3);
      this.group.add(leg);
    }
    const n = Math.ceil(len / pitch);
    this.slats = new THREE.InstancedMesh(new THREE.BoxGeometry(0.1, 0.03, width - 0.1), flatMat('ink'), n);
    this.slats.frustumCulled = false;
    this.group.add(this.slats);
    this.setOffset(0);
  }
  /** Belt travel so far (units toward -x). */
  setOffset(off: number) {
    const n = this.slats.count;
    for (let i = 0; i < n; i++) {
      const u = (((i * this.pitch - off) % this.len) + this.len) % this.len;
      this.tmp.position.set(this.x0 + u, this.top + 0.012, 0);
      this.tmp.updateMatrix();
      this.slats.setMatrixAt(i, this.tmp.matrix);
    }
    this.slats.instanceMatrix.needsUpdate = true;
  }
}
