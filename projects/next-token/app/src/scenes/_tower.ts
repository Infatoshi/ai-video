// The model as a building (shared by every scene that shows Llama 3.1 8B from outside). Floor 0 is the
// embedding lobby, floors 1..32 are the transformer layers, the roof is the output head. Each layer
// floor has a slab, a ring of 32 attention-head pillars and a feed-forward block, and a KV shelf on its
// flank that grows one card pair per context position. The residual stream is a pink shaft up the middle
// and the token in flight rides it. Dimensions from data/llm.json's config.
import * as THREE from 'three';
import { inked, inkMat, flatMat, outline, outlineInstanced, tokenFaceTexture, type InkName } from './_kit';

export const FLOOR_H = 0.62;
export const SLAB_W = 5.2;

export interface TowerOpts {
  layers?: number;
  heads?: number;
  /** Max KV card pairs shown per floor (the shelf's capacity). */
  kvMax?: number;
}

export class Tower {
  group = new THREE.Group();
  layers: number;
  slabs: THREE.Mesh[] = [];
  heads: THREE.InstancedMesh;
  kv: THREE.InstancedMesh;
  ff: THREE.Mesh[] = [];
  shaft: THREE.Mesh;
  rider: THREE.Group;
  roof: THREE.Mesh;
  labels: THREE.Mesh[] = [];
  private headsPer: number;
  private kvMax: number;
  private tmp = new THREE.Object3D();
  private col = new THREE.Color();

  constructor(o: TowerOpts = {}) {
    this.layers = o.layers ?? 32;
    this.headsPer = o.heads ?? 32;
    this.kvMax = o.kvMax ?? 96;
    const L = this.layers;

    // lobby (embedding) and roof (output head): wider, black-inked
    const lobby = inked(new THREE.BoxGeometry(SLAB_W * 1.25, 0.34, SLAB_W * 1.25), { ink: 'ink', lit: 0.12, shade: 0.75 });
    lobby.position.y = -0.17;
    this.group.add(lobby);
    this.slabs.push(lobby);

    const slabGeo = new THREE.BoxGeometry(SLAB_W, 0.1, SLAB_W);
    const ffGeo = new THREE.BoxGeometry(SLAB_W * 0.36, FLOOR_H * 0.46, SLAB_W * 0.36);
    for (let l = 1; l <= L; l++) {
      const y = this.floorY(l);
      const s = inked(slabGeo, { ink: 'blue', lit: 0.3, shade: 1, over: 'pink', overLit: 0, overShade: 0 });
      s.position.y = y;
      this.group.add(s);
      this.slabs.push(s);
      const f = inked(ffGeo, { ink: 'blue', lit: 0.55, shade: 1, over: 'pink', overLit: 0, overShade: 0 }, 1.6);
      f.position.set(SLAB_W * 0.22, y + 0.05 + FLOOR_H * 0.23, -SLAB_W * 0.22);
      this.group.add(f);
      this.ff.push(f);
    }
    this.roof = inked(new THREE.BoxGeometry(SLAB_W * 1.1, 0.3, SLAB_W * 1.1), { ink: 'ink', lit: 0.1, shade: 0.8, over: 'pink', overLit: 0, overShade: 0 });
    this.roof.position.y = this.floorY(L + 1) - 0.1;
    this.group.add(this.roof);

    // attention heads: a ring of pillars on every layer floor
    const H = this.headsPer;
    const hGeo = new THREE.BoxGeometry(0.12, FLOOR_H * 0.5, 0.12);
    this.heads = new THREE.InstancedMesh(hGeo, inkMat({ ink: 'blue', lit: 0.45, shade: 1, over: 'pink' }), L * H);
    this.heads.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(L * H * 3), 3);
    for (let l = 1; l <= L; l++) for (let h = 0; h < H; h++) {
      const a = (h / H) * Math.PI * 2, r = SLAB_W * 0.36;
      this.tmp.position.set(Math.cos(a) * r - SLAB_W * 0.08, this.floorY(l) + 0.05 + FLOOR_H * 0.25, Math.sin(a) * r + SLAB_W * 0.08);
      this.tmp.rotation.set(0, -a, 0);
      this.tmp.updateMatrix();
      const i = (l - 1) * H + h;
      this.heads.setMatrixAt(i, this.tmp.matrix);
      this.heads.setColorAt(i, this.col.setRGB(0, 1, 0));
    }
    this.heads.frustumCulled = false;
    this.group.add(this.heads);

    // KV shelf on the +x flank: per floor a row of K (blue) / V (pink) cards, one pair per position
    const kGeo = new THREE.BoxGeometry(0.05, FLOOR_H * 0.34, 0.05);
    this.kv = new THREE.InstancedMesh(kGeo, inkMat({ ink: 'blue', lit: 0.8, shade: 1, over: 'pink' }), L * this.kvMax * 2);
    this.kv.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(L * this.kvMax * 2 * 3), 3);
    this.kv.frustumCulled = false;
    this.group.add(this.kv);
    this.setKV(0);

    // residual stream shaft and the token riding it
    this.shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, this.floorY(L + 1) + 0.4, 24, 1, true), flatMat('pink', THREE.DoubleSide));
    this.shaft.position.y = (this.floorY(L + 1) + 0.4) / 2 - 0.3;
    this.group.add(this.shaft);
    this.rider = new THREE.Group();
    this.group.add(this.rider);

    // floor numbers on the front edge
    for (let l = 1; l <= L; l++) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.22), new THREE.MeshBasicMaterial({ map: tokenFaceTexture(`L${l}`, 'paper', 'ink', { w: 256, h: 90 }), transparent: false }));
      m.position.set(-SLAB_W / 2 + 0.36, this.floorY(l) + 0.2, SLAB_W / 2 + 0.01);
      this.group.add(m);
      this.labels.push(m);
    }
  }

  /** y of floor l (0 = lobby top, 1..32 layers, 33 = roof). */
  floorY(l: number) { return 0.15 + l * FLOOR_H; }
  get height() { return this.floorY(this.layers + 1); }

  /** Put a mesh (e.g. a tokenTile) on the shaft at climb 0..1 (lobby -> roof). */
  ride(obj: THREE.Object3D, climb: number) {
    if (obj.parent !== this.rider) { this.rider.clear(); this.rider.add(obj); }
    obj.position.set(0, this.floorY(climb * (this.layers + 1)) + 0.25, 0);
  }

  /**
   * Light the tower for a token at `climb` (0..1): the floor it is on and the one just passed flood with
   * pink; the rest stay blue.
   */
  light(climb: number, glow = 1) {
    const at = climb * (this.layers + 1);
    for (let l = 1; l <= this.layers; l++) {
      const d = at - l;
      const hit = d >= 0 && d < 1.2 ? 1 - d / 1.2 : 0;
      const u = (this.slabs[l]!.material as THREE.ShaderMaterial).uniforms;
      u.hot!.value = hit * glow;
      (this.ff[l - 1]!.material as THREE.ShaderMaterial).uniforms.hot!.value = hit * glow * 0.9;
    }
  }

  /** Per-head pink hits for floor l (weights 0..1, length = heads), e.g. from real attention mass. */
  setHeads(l: number, w: ArrayLike<number>) {
    const H = this.headsPer;
    for (let h = 0; h < H; h++) this.heads.setColorAt((l - 1) * H + h, this.col.setRGB(Math.min(1, w[h] ?? 0), 1, 0));
    this.heads.instanceColor!.needsUpdate = true;
  }
  clearHeads() {
    for (let i = 0; i < this.layers * this.headsPer; i++) this.heads.setColorAt(i, this.col.setRGB(0, 1, 0));
    this.heads.instanceColor!.needsUpdate = true;
  }

  /** Show n KV card pairs per floor (the context length so far); the newest pair can be flashed pink. */
  setKV(n: number, newest = 0) {
    const L = this.layers, M = this.kvMax, shown = Math.min(M, Math.max(0, Math.floor(n)));
    let i = 0;
    for (let l = 1; l <= L; l++) for (let k = 0; k < M; k++) for (let kv = 0; kv < 2; kv++) {
      const on = k < shown;
      const x = SLAB_W / 2 + 0.12 + (k % 48) * 0.07, z = -SLAB_W / 2 + 0.3 + kv * 0.16 + Math.floor(k / 48) * 0.36;
      this.tmp.position.set(x, this.floorY(l) + 0.05 + FLOOR_H * 0.17, z);
      this.tmp.rotation.set(0, 0, 0);
      this.tmp.scale.setScalar(on ? 1 : 0);
      this.tmp.updateMatrix();
      this.kv.setMatrixAt(i, this.tmp.matrix);
      this.kv.setColorAt(i, this.col.setRGB(kv ? 1 : on && k === shown - 1 ? newest : 0, kv ? 0 : 1, 0)); // K blue, V pink
      i++;
    }
    this.kv.instanceMatrix.needsUpdate = true;
    this.kv.instanceColor!.needsUpdate = true;
  }

  /** Outline the instanced parts too (call once; costs two extra instanced draws). */
  outlineAll() {
    outlineInstanced(this.heads, 1.4);
    outlineInstanced(this.kv, 1.2);
  }
}

/** A tile riding the tower: the in-flight token, as a pink slab with its text. */
export function riderTile(tok: string, ink: InkName = 'pink'): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.5, 0.5), [flatMat(ink), flatMat(ink), flatMat(ink), flatMat(ink),
    new THREE.MeshBasicMaterial({ map: tokenFaceTexture(tok, ink, 'paper') }), flatMat(ink)]);
  outline(m, 2.4);
  return m;
}
