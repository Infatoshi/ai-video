// What Llama 3.1 8B Instruct actually did with the video's question (data/llm.json, recorded on the
// 3090 by llm/dump.py). Every number the video shows about the model comes from here.

export interface LlmStep {
  /** Position of the new token in the context. */
  pos: number;
  /** The sampled token. */
  tok: { id: number; t: string };
  /** Top 10 at temperature 1 [token, p]. */
  top10_t1: [string, number][];
  /** The nucleus actually sampled from (temperature 0.6, top-p 0.9), renormalised, first 10. */
  nucleus: [string, number][];
  nucleus_size: number;
  /** The uniform draw and the index in the nucleus it landed on. */
  u: number;
  pick: number;
  entropy_t1: number;
  /** Logit lens: for the embedding and each of the 32 layers, the residual norm and the top 3 guesses. */
  lens: { norm: number; top: [string, number][] }[];
  /** Attention of the new token over the context: [layers, heads, ctx] as uint8 of max. */
  attn: { shape: [number, number, number]; max: number; u8: string };
  /** Feed-forward activations (|act| as uint8 of their max) at layers 0, 15, 31. */
  ff: Record<string, { n: number; u8: string; active: number }>;
}

export interface LlmData {
  model: string;
  gpu: string;
  dtype: string;
  config: { layers: number; heads: number; kv_heads: number; hidden: number; head_dim: number; ffn: number; vocab: number; params: number; weight_bytes: number; kv_bytes_per_token: number };
  question: string;
  question_tokens: string[];
  strawberry: Record<string, string[]>;
  /** The whole chat-templated prompt, token by token. */
  prompt: { id: number; t: string }[];
  sampling: { temperature: number; top_p: number; seed: number };
  answer: string;
  steps: LlmStep[];
  /** Input embeddings of a word list: PCA to 3D (normalised to +-1) and the cosine matrix. */
  embed: { words: string[]; pca3: [number, number, number][]; cos: number[][]; first64: Record<string, number[]> };
  /** Speculative decoding: a round at every answer position (the 1B draft's 4 greedy guesses, the 8B's own
   * greedy picks for those 4 slots plus a bonus 5th, how many matched), and the round the video shows. */
  spec: { at: number; context_answer_tokens: number; draft: string[]; big: string[]; accepted: number; rounds: { at: number; draft: string[]; big: string[]; accepted: number }[] };
  /** Measured decode speed on the 3090 (HF transformers, bf16), by batch size. */
  speed_tok_s: Record<string, number>;
}

const b64 = (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

export class Llm {
  private attnCache = new Map<number, Uint8Array>();
  private ffCache = new Map<string, Uint8Array>();
  constructor(public d: LlmData) {}

  static async load(): Promise<Llm> {
    const r = await fetch('data/llm.json');
    if (!r.ok) throw new Error('data/llm.json missing (run llm/dump.py on the GPU host)');
    return new Llm(await r.json());
  }

  get steps() { return this.d.steps; }
  get cfg() { return this.d.config; }
  /** Index where the user's question starts in the prompt (after the chat header). */
  get questionStart() {
    const q = this.d.question_tokens;
    const p = this.d.prompt.map((x) => x.t);
    for (let i = 0; i + q.length <= p.length; i++) if (q.every((t, k) => p[i + k] === t)) return i;
    return 0;
  }

  /** Attention weights (0..1) of step `s`'s new token over its context, for one layer and head. */
  attn(s: number, layer: number, head: number): Float32Array {
    const st = this.d.steps[s]!;
    let u = this.attnCache.get(s);
    if (!u) { u = b64(st.attn.u8); this.attnCache.set(s, u); }
    const [, H, C] = st.attn.shape;
    const o = (layer * H + head) * C, out = new Float32Array(C), k = st.attn.max / 255;
    for (let i = 0; i < C; i++) out[i] = u[o + i]! * k;
    return out;
  }

  /** Feed-forward |activations| (0..1 of their max) of step `s` at layer 0, 15 or 31. */
  ff(s: number, layer: 0 | 15 | 31): Uint8Array {
    const key = `${s}:${layer}`;
    let u = this.ffCache.get(key);
    if (!u) { u = b64(this.d.steps[s]!.ff[String(layer)]!.u8); this.ffCache.set(key, u); }
    return u;
  }

  /** Tokens of the answer, as shown (the end-of-turn token as ⏎). */
  answerTokens() { return this.d.steps.map((s) => s.tok.t); }
}

/** Display form of a token: leading space as a visible dot, specials short. */
export function showTok(t: string): string {
  if (t === '<|eot_id|>') return '<END>';
  if (t.startsWith('<|')) return t.replace('<|', '<').replace('|>', '>');
  return t.replace(/^ /, '·').replace(/\n/g, '↵');
}
