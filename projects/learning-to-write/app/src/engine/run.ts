// The training run the video is about (data/run.json, recorded on the GPU host's 3090 by train/train.py, trimmed by
// train/export.py) and the probe of its checkpoints (data/probe.json, train/probe.py). Every number the video
// shows about the model comes from here, or from arithmetic on it (labelled as computed where it is shown).

export interface Sample { step: number; t: number; text: string }
export interface Eval { step: number; train: number; val: number; t: number }
export interface ProbeChar { ch: string; p: number; loss: number; top: [string, number][]; dist: number[] }
export interface ProbeStep { step: number; line_loss: number; chars: ProbeChar[]; after_prompt: number[]; emb: number[][]; one: number }

export interface RunData {
  n_params: number;
  n_params_without_position: number;
  vocab: string;
  vocab_size: number;
  dataset: { name: string; url: string; chars: number; sha256: string; train_chars: number; val_chars: number };
  device: string;
  torch: string;
  wall_s: number;
  overhead_s: number;
  chars_per_step: number;
  started: string;
  config: Record<string, any>;
  /** Every step: the batch loss it trained on (dropout on) and the wall clock (s) when it finished. */
  steps: { step: number[]; loss: number[]; t: number[] };
  evals: Eval[];
  samples: Sample[];
}

export interface ProbeData {
  line_ctx: string;
  line: string;
  prompt: string;
  vocab: string;
  steps: ProbeStep[];
  moves: { step: number; median_abs: number; mean_abs: number; changed: number; median_abs_weight: number }[];
  one: { char: string; dim: number };
}

/** What each chorus sings (song/make_sing.py): the checkpoint and the span of its sample, verbatim. */
export interface Excerpts {
  minutes: number;
  wall_s: number;
  loss0: number;
  loss1: number;
  excerpts: { step: number; a: number; b: number; text: string; lines: string[] }[];
}

export class Run {
  constructor(public d: RunData, public p: ProbeData, public ex: Excerpts) {}

  static async load(): Promise<Run> {
    const [a, b, e] = await Promise.all([fetch('data/run.json'), fetch('data/probe.json'), fetch('data/excerpts.json')]);
    if (!a.ok || !b.ok || !e.ok) throw new Error('data/run.json, probe.json or excerpts.json missing (train/export.py)');
    return new Run(await a.json(), await b.json(), await e.json());
  }

  get maxStep() { return this.d.config.max_iters as number; }
  /** Checkpoint steps (where samples, probes and weights were recorded). */
  get ckpts() { return this.d.samples.map((s) => s.step); }

  /** Index of the last checkpoint at or before `step`. */
  ckptIdx(step: number) {
    const c = this.d.samples;
    let i = 0;
    while (i + 1 < c.length && c[i + 1]!.step <= step + 1e-6) i++;
    return i;
  }

  /** Wall-clock seconds into the run when `step` (fractional) finished. */
  wallAt(step: number) {
    const s = this.d.steps;
    if (step <= 0) return 0;
    const i = Math.min(s.t.length - 1, Math.max(0, Math.floor(step) - 1));
    const a = s.t[i]!, b = s.t[Math.min(s.t.length - 1, i + 1)]!;
    return a + (b - a) * (step - Math.floor(step));
  }

  /** The evaluated loss (20 fixed batches, no dropout) at `step`, linearly between evaluations. */
  evalAt(step: number, which: 'train' | 'val' = 'val') {
    const e = this.d.evals;
    if (step <= e[0]!.step) return e[0]![which];
    for (let i = 0; i + 1 < e.length; i++) {
      const a = e[i]!, b = e[i + 1]!;
      if (step <= b.step) return a[which] + (b[which] - a[which]) * ((step - a.step) / Math.max(1e-6, b.step - a.step));
    }
    return e[e.length - 1]![which];
  }

  /** The batch loss of step `step` (1-based; step 0 has none, returns the step-0 evaluation). */
  stepLoss(step: number) {
    if (step < 1) return this.d.evals[0]!.train;
    return this.d.steps.loss[Math.min(this.d.steps.loss.length - 1, Math.round(step) - 1)]!;
  }

  finalEval() { return this.d.evals[this.d.evals.length - 1]!; }
  sampleAt(step: number) { return this.d.samples[this.ckptIdx(step)]!; }
  probeAt(step: number) { return this.p.steps[this.ckptIdx(step)]!; }
}

/** Display form of a character: newline as ↵, space as a middle dot. */
export function showChar(c: string): string {
  if (c === '\n') return '↵';
  if (c === ' ') return '·';
  return c;
}
