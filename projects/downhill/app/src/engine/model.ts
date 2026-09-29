// The run the video explains (data/model.json, written by model/train.py): a 2-16-16-1 tanh network (337
// weights) learning two spirals by plain full-batch gradient descent, every step recorded, plus the too-big-step
// run and a 2D slice of the loss landscape. Every number on screen comes from here.

export interface ModelFacts {
  params: number; hidden: number[]; points: number; per_class: number; lr: number; lr_big: number; steps: number;
  loss0: number; acc0: number; wrong0: number; first_all_right: number; loss_at_first_all_right: number;
  loss_final: number; acc_final: number; wrong_at_3000: number; gnorm0: number; step0_len: number;
  big_min_loss: number; big_min_loss_step: number; big_best_acc: number; big_best_acc_step: number;
  big_final_loss: number; big_final_acc: number; big_max_loss: number;
  cut_step: number; cut_loss: number; cut_small: number; cut_big: number; cut_floor: number; cut_floor_eta: number;
  pca_var: number[]; pca_captured: number; random_plane_captured: number; random_plane_surface_at_step0: number;
  slice_vs_real_step0: [number, number]; slice_vs_real_max_gap: number;
}

interface ModelJSON {
  facts: ModelFacts;
  X: number[]; y: number[]; box: number;
  loss: number[]; acc: number[]; gnorm: number[]; wrong: number[];
  every: number; theta: number[][];
  path: number[]; downhill: number[];
  surface: { a0: number; a1: number; b0: number; b1: number; na: number; nb: number; loss: number[] };
  big: { loss: number[]; acc: number[]; theta: number[][]; cut: { step: number; eta: number[]; loss: number[] } };
  per_dot_loss0: number[];
}

export const NP = 337;
const H1 = 16, H2 = 16;

/** Offsets of each block in the flat weight vector (model/spiral.py unflat order). */
export const OFF = { W1: 0, b1: 32, W2: 48, b2: 304, W3: 320, b3: 336 } as const;

export class Model {
  f: ModelFacts;
  X: Float32Array; y: Uint8Array; box: number;
  loss: Float32Array; acc: Float32Array; gnorm: Float32Array; wrong: Int16Array;
  every: number; theta: Float32Array[]; thetaBig: Float32Array[];
  path: Float32Array; downhill: Float32Array;
  surf: { a0: number; a1: number; b0: number; b1: number; na: number; nb: number; L: Float32Array };
  lossBig: Float32Array; accBig: Float32Array;
  cut: { step: number; eta: Float32Array; L: Float32Array };
  perDot0: Float32Array;
  steps: number;

  constructor(j: ModelJSON) {
    this.f = j.facts;
    this.X = Float32Array.from(j.X); this.y = Uint8Array.from(j.y); this.box = j.box;
    this.loss = Float32Array.from(j.loss); this.acc = Float32Array.from(j.acc); this.gnorm = Float32Array.from(j.gnorm);
    this.wrong = Int16Array.from(j.wrong);
    this.every = j.every;
    this.theta = j.theta.map((r) => Float32Array.from(r));
    this.thetaBig = j.big.theta.map((r) => Float32Array.from(r));
    this.path = Float32Array.from(j.path); this.downhill = Float32Array.from(j.downhill);
    const s = j.surface;
    this.surf = { a0: s.a0, a1: s.a1, b0: s.b0, b1: s.b1, na: s.na, nb: s.nb, L: Float32Array.from(s.loss) };
    this.lossBig = Float32Array.from(j.big.loss); this.accBig = Float32Array.from(j.big.acc);
    this.cut = { step: j.big.cut.step, eta: Float32Array.from(j.big.cut.eta), L: Float32Array.from(j.big.cut.loss) };
    this.perDot0 = Float32Array.from(j.per_dot_loss0);
    this.steps = this.loss.length - 1;
  }

  static async load(): Promise<Model> {
    const r = await fetch('data/model.json');
    if (!r.ok) throw new Error('data/model.json missing (run model/train.py)');
    return new Model(await r.json());
  }

  private lerpArr(a: Float32Array, s: number) {
    const x = Math.min(Math.max(s, 0), a.length - 1), i = Math.floor(x), k = x - i;
    return i + 1 < a.length ? a[i]! * (1 - k) + a[i + 1]! * k : a[i]!;
  }
  /** Loss / accuracy at a (fractional) step of the run. */
  lossAt(s: number) { return this.lerpArr(this.loss, s); }
  accAt(s: number) { return this.acc[Math.round(Math.min(Math.max(s, 0), this.steps))]!; }
  wrongAt(s: number) { return this.wrong[Math.round(Math.min(Math.max(s, 0), this.steps))]!; }
  lossBigAt(s: number) { return this.lerpArr(this.lossBig, s); }

  /** The weights at a fractional step (recorded every `every` steps; linear in between: steps are tiny). */
  thetaAt(s: number, big = false, out = new Float32Array(NP)): Float32Array {
    const T = big ? this.thetaBig : this.theta;
    const x = Math.min(Math.max(s / this.every, 0), T.length - 1), i = Math.floor(x), k = x - i;
    const a = T[i]!, b = T[Math.min(i + 1, T.length - 1)]!;
    for (let j = 0; j < NP; j++) out[j] = a[j]! * (1 - k) + b[j]! * k;
    return out;
  }

  /** Where the weights are on the slice (PCA coordinates) at a fractional step. */
  pathAt(s: number): [number, number] {
    const n = this.steps, x = Math.min(Math.max(s, 0), n), i = Math.floor(x), k = x - i, j = Math.min(i + 1, n);
    const P = this.path;
    return [P[2 * i]! * (1 - k) + P[2 * j]! * k, P[2 * i + 1]! * (1 - k) + P[2 * j + 1]! * k];
  }
  /** The real downhill direction (-gradient) projected onto the slice, at step s. */
  downhillAt(s: number): [number, number] {
    const i = Math.round(Math.min(Math.max(s, 0), this.steps));
    return [this.downhill[2 * i]!, this.downhill[2 * i + 1]!];
  }
  /** The slice's loss at slice coordinates (a, b), bilinear on the recorded grid. */
  surfL(a: number, b: number): number {
    const S = this.surf;
    const x = ((a - S.a0) / (S.a1 - S.a0)) * (S.na - 1), y = ((b - S.b0) / (S.b1 - S.b0)) * (S.nb - 1);
    const i = Math.min(Math.max(Math.floor(x), 0), S.na - 2), j = Math.min(Math.max(Math.floor(y), 0), S.nb - 2);
    const u = Math.min(Math.max(x - i, 0), 1), v = Math.min(Math.max(y - j, 0), 1);
    const L = S.L, w = S.na;
    return (L[j * w + i]! * (1 - u) + L[j * w + i + 1]! * u) * (1 - v) + (L[(j + 1) * w + i]! * (1 - u) + L[(j + 1) * w + i + 1]! * u) * v;
  }

  /** The network's logit at (x, y) for weights th. */
  static logit(th: Float32Array, x: number, y: number): number {
    const h1 = new Float32Array(H1);
    for (let j = 0; j < H1; j++) h1[j] = Math.tanh(x * th[OFF.W1 + j]! + y * th[OFF.W1 + H1 + j]! + th[OFF.b1 + j]!);
    let z = th[OFF.b3]!;
    for (let k = 0; k < H2; k++) {
      let a = th[OFF.b2 + k]!;
      for (let j = 0; j < H1; j++) a += h1[j]! * th[OFF.W2 + j * H2 + k]!;
      z += Math.tanh(a) * th[OFF.W3 + k]!;
    }
    return z;
  }
  /** Each dot's error (cross-entropy) for weights th. */
  dotLoss(th: Float32Array, out = new Float32Array(this.y.length)) {
    for (let i = 0; i < this.y.length; i++) {
      const z = Model.logit(th, this.X[2 * i]!, this.X[2 * i + 1]!), y = this.y[i]!;
      out[i] = Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
    }
    return out;
  }
}

/** Numbers as printed: 0.696, 0.0049, 3,121. */
export const fmtLoss = (x: number) => (x >= 0.1 ? x.toFixed(3) : x >= 0.01 ? x.toFixed(4) : x.toFixed(5));
export const fmtInt = (n: number) => Math.round(n).toLocaleString('en-US');
