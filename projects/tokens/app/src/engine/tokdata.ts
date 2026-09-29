// The episode's real data (data/tok.json, built by tok/build_data.py): the Llama 3.1 tokenizer's splits and
// ids, the spelled-letters test on Llama 3.1 8B Instruct, and our BPE trained on this song's lyrics. Every
// number the video shows comes from here.

export interface Tok { id: number; t: string }

export interface TokData {
  llama: {
    tokenizer: string;
    vocab_total: number;
    vocab_regular: number;
    special: number;
    question: string;
    question_tokens: Tok[];
    chat_prompt_tokens: number;
    answer: string;
    answer_tokens: Tok[];
    words: Record<string, Tok[]>;
    field: { cols: number; strawberry: [number, number]; patch: Record<string, string> };
  };
  spell_test: {
    model: string; gpu: string; n: number; temperature: number; top_p: number; seed: number; max_new_tokens: number;
    word: { question: string; tokens: string[]; greedy: string; said: Record<string, number> };
    spaced: { question: string; tokens: string[]; greedy: string; said: Record<string, number>; ids: number[] };
  };
  bpe: {
    corpus: { lines: number; words: number; unique_words: number; chars: number; alphabet: string };
    merges: { k: number; a: string; b: string; ab: string; count: number }[];
    vocab: number;
    ids: Record<string, number>;
    line: { text: string; words: string[]; steps: { k: number; words: string[][] }[] };
    strawberry: { text: string; words: string[]; steps: { k: number; words: string[][] }[] };
  };
}

export async function loadTok(): Promise<TokData> {
  const r = await fetch('data/tok.json');
  if (!r.ok) throw new Error('data/tok.json missing (python3 tok/build_data.py)');
  return r.json();
}
