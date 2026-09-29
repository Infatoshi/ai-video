// What Llama 3.1 8B Instruct's attention heads actually did with the video's two sentences
// (data/attn.json: recorded on the 3090 by llm/attn.py, analysed by llm/flip.py). Every number the
// video shows about the model comes from here.

export interface Census {
  /** Asking token index in sentence A and B. */
  q: [number, number];
  /** [layer][head] -> [w(animal) in A, w(street) in A, w(animal) in B, w(street) in B]. */
  w: number[][][];
  /** [layer][head] -> [w(start) in A, w(start) in B]. */
  bos: number[][][];
  /** [layer][head] -> [row in A, row in B] (the asking token's weights over tokens 0..q), or null. */
  rows: number[][][][] | null;
  same_prefix: boolean;
  max_abs_diff: number | null;
  heads: number;
  right: number;
  clean: number;
  wrong: number;
  clean_wrong: number;
  no_flip: number;
  quiet: number;
  clean_by_layer: number[];
  mean_bos: [number, number];
  top: { layer: number; head: number; margin: number; A: [number, number]; B: [number, number] }[];
  /** [layer][head]: 2 clean flip, 1 leans right, 0 no switch, -1 leans wrong (from the unrounded weights). */
  kind: number[][];
}

export interface AttnData {
  model: string;
  gpu: string;
  dtype: string;
  config: { layers: number; heads: number; kv_heads: number; hidden: number };
  /** A clean flip moves at least this much weight each way; heads under `quiet` on both words are "elsewhere". */
  clean: number;
  quiet: number;
  featured: { pair: string; layer: number; head: number };
  /** The two sentences, token by token (token 0 is <|begin_of_text|>). */
  A: string[];
  B: string[];
  ref: [string, string];
  refA: [number, number];
  refB: [number, number];
  /** The featured head's full attention (row = asking token, T x T) in A and B. */
  head: { A: number[][]; B: number[][] };
  grid: { who: Census; change: Census; end: Census };
  probe: Record<'A' | 'B', { text: string; top10: [string, number][]; words: Record<string, number> }>;
  chat: Record<'A' | 'B', { question: string; answer: string; top10: [string, number][] }>;
  pairs: { id: string; note: string; A: string[]; B: string[]; ref: string[] }[];
}

export class Attn {
  constructor(public d: AttnData) {}

  static async load(): Promise<Attn> {
    const r = await fetch('data/attn.json');
    if (!r.ok) throw new Error('data/attn.json missing (run llm/attn.py on the GPU host, then llm/flip.py)');
    return new Attn(await r.json());
  }

  get T() { return this.d.A.length; }
  /** The featured head's weights from token q in sentence A (s = 0) or B (s = 1). */
  row(s: 0 | 1, q: number): number[] { return (s ? this.d.head.B : this.d.head.A)[q]!.slice(0, q + 1); }
  /** Every head's row from the changed word (tired / wide), for the grid. */
  gridRow(layer: number, head: number, s: 0 | 1): number[] { return this.d.grid.change.rows![layer]![head]![s]!; }
  /** How head (l, h) behaves when "tired" becomes "wide": 2 clean flip, 1 leans right, -1 leans wrong, 0 no switch. */
  kind(layer: number, head: number): 2 | 1 | 0 | -1 {
    return this.d.grid.change.kind[layer]![head]! as 2 | 1 | 0 | -1;
  }
}

/** Display form of a token: the start token as "start", leading spaces dropped. */
export function showTok(t: string): string {
  if (t === '<|begin_of_text|>') return 'start';
  if (t === '<|eot_id|>') return '<END>';
  if (t.startsWith('<|')) return t.replace('<|', '<').replace('|>', '>');
  return t.replace(/^ /, '·').replace(/\n/g, '↵');
}
