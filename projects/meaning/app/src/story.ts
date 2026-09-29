// The story clock: which of the model's answer tokens is in flight at song time t, how far up the
// tower it has climbed, and when each one is emitted. The HUD and every scene read this, so the whole
// video agrees on "where the model is". Emissions are anchored to lyric words and snapped to beats.
import type { AudioData } from './engine/audio';
import type { Lyrics } from './engine/lyrics';
import type { Llm } from './engine/llm';

export type Mode = 'prefill' | 'decode' | 'batch' | 'spec';

export interface StoryState {
  /** Answer tokens emitted so far (0..N). */
  emitted: number;
  /** Step in flight (the next token to be emitted), clamped to N-1. */
  step: number;
  /** 0..1 progress of the token in flight up the tower (embedding -> top of layer 32). */
  climb: number;
  /** Floor the token in flight is on (0..32). */
  layer: number;
  /** Context position of the token in flight. */
  pos: number;
  mode: Mode;
  /** Seconds since the last emission (Infinity before the first). */
  since: number;
  /** Seconds until the next emission. */
  until: number;
}

export class Story {
  /** Song time each answer token is emitted. */
  emits: number[] = [];
  /** When the question is sent into the model (the prefill starts climbing). */
  sent = 0;
  modes: { t0: number; t1: number; mode: Mode }[] = [];
  N: number;

  constructor(public llm: Llm, ly: Lyrics, au: AudioData) {
    this.N = llm.steps.length;
    const word = (line: string, w: string, nth = 0): number | undefined => {
      try {
        const l = ly.get(line, nth);
        const x = l.words.find((y) => y.w.toLowerCase().replace(/[^a-z']/g, '').startsWith(w));
        return x ? x.start : undefined;
      } catch { return undefined; }
    };
    const snap = (t: number) => au.timeOfBeat(Math.round(au.beatAt(t)));
    const end = au.duration;
    this.sent = snap(word('Type a question', 'enter') ?? 2);
    // the run (data/llm.json): "There are 2 r's in the word 'strawberry'." + <|eot_id|>, 15 tokens.
    // Chorus 2's "pick" is step 3, where the dice chose "2" (69.7%) over "3" (30.3%).
    const fixed: [number, number | undefined][] = [
      [0, word('Guess it, pick it', 'pick', 0)],
      [1, word('Stack it thirty-two high', 'power')],
      [2, word('Just one more token', 'token', 1)],
      [3, word('Guess it, pick it', 'pick', 1)],
      [4, word('Roll the dice', 'token')],
      [5, word('Feed the whole thing back', 'again')],
      [6, word('Every single token, it reads', 'weight')],
      [7, word('Sixteen gigs', 'waits')],
      [this.N - 1, word('says it', 'done')],
    ];
    const at = new Map<number, number>();
    for (const [k, t] of fixed) if (t !== undefined && k >= 0 && k < this.N && !at.has(k)) at.set(k, snap(t));
    // the speculative round (llm.d.spec: the 1B draft's accepted tokens are the answer's own next tokens):
    // they land together on "go"
    const go = word('checks all four', 'go');
    const drop = au.sections.find((s) => s.name === 'drop');
    const spec = llm.d.spec;
    if (go !== undefined && spec) for (let i = 0; i < spec.accepted; i++) if (!at.has(spec.at + i)) at.set(spec.at + i, snap(go));
    // everything not anchored is spread evenly between its anchored neighbours (the drop takes the rest, fast)
    const ks = [...at.keys()].sort((a, b) => a - b);
    const e: number[] = new Array(this.N);
    for (let k = 0; k < this.N; k++) {
      if (at.has(k)) { e[k] = at.get(k)!; continue; }
      const lo = [...ks].reverse().find((x) => x < k), hi = ks.find((x) => x > k);
      const t0 = lo !== undefined ? at.get(lo)! : this.sent, t1 = hi !== undefined ? at.get(hi)! : end;
      const k0 = lo ?? -1, k1 = hi ?? this.N;
      // when the gap spans the drop, pack these tokens into it
      let a = t0, b = t1;
      if (drop && drop.start > t0 && drop.start < t1) { a = drop.start; b = Math.min(t1, drop.end); }
      e[k] = snap(a + ((k - k0) / (k1 - k0)) * (b - a));
    }
    for (let k = 1; k < this.N; k++) e[k] = Math.max(e[k]!, e[k - 1]!);
    this.emits = e;
    const sec = (n: string) => au.sections.find((s) => s.name === n);
    const v4 = sec('verse4');
    if (drop) this.modes.push({ t0: drop.start, t1: drop.end, mode: 'batch' });
    if (v4) this.modes.push({ t0: v4.start, t1: v4.end, mode: 'spec' });
  }

  at(t: number): StoryState {
    const N = this.N, e = this.emits;
    let emitted = 0;
    while (emitted < N && e[emitted]! <= t) emitted++;
    const step = Math.min(emitted, N - 1);
    const prev = emitted === 0 ? this.sent : e[emitted - 1]!;
    const next = emitted < N ? e[emitted]! : Infinity;
    let climb = emitted >= N ? 1 : Math.min(1, Math.max(0, (t - prev) / Math.max(1e-3, next - prev)));
    if (t < this.sent) climb = 0;
    const mode: Mode = t < (e[0] ?? Infinity) ? 'prefill' : this.modes.find((m) => t >= m.t0 && t < m.t1)?.mode ?? 'decode';
    const L = this.llm.cfg.layers;
    const pos = this.llm.d.prompt.length + step;
    return { emitted, step, climb, layer: Math.min(L, Math.floor(climb * (L + 1))), pos, mode,
      since: emitted > 0 ? t - e[emitted - 1]! : Infinity, until: next - t };
  }

  /** The answer text emitted by t. */
  textAt(t: number): string[] {
    const n = this.at(t).emitted;
    return this.llm.steps.slice(0, n).map((s) => s.tok.t);
  }
}
