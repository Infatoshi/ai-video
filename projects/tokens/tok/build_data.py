"""Assemble data/tok.json (what the video shows) from the three real runs:

  python3 tok/build_data.py      # from projects/tokens/

  out/llama_tok.json   the Llama 3.1 tokenizer's splits and ids (tok/llama_tok.py on the GPU host, CPU)
  out/spell_test.json  Llama 3.1 8B Instruct asked 100 times per spelling (tok/spell_test.py, 3090)
  song/lyrics.sing.txt BPE trained on the song's own lyrics (tok/bpe.py, here)

Only derived numbers leave out/: splits, ids, tallies, merges (the tokenizer files stay in the HF cache).
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE.parent
sys.path.insert(0, str(HERE))
import bpe  # noqa: E402

lt = json.loads((P / "out/llama_tok.json").read_text())
st = json.loads((P / "out/spell_test.json").read_text())
lines, words = bpe.corpus(P / "song/lyrics.sing.txt")
merges, final = bpe.train(words)
letters = sorted(set("".join(words)))


def said(v):
    """What each answer said the count is: the total it states ("there are N", "has N", "N r's"), read by
    pattern; 'none' = it began counting out loud and hit the 40-token cap before stating a total."""
    out = {"2": 0, "3": 0, "other": 0, "none": 0}
    pat = re.compile(r"(?:there (?:are|is)|has|contains) (\d+|two|three)\b|\b(\d+|two|three) '?r'?'?s\b")
    for a in v["answers"]:
        m = pat.search(a.lower().replace("’", "'"))
        if not m:
            out["none"] += 1
            continue
        x = m[1] or m[2]
        x = {"two": "2", "three": "3"}.get(x, x)
        out[x if x in ("2", "3") else "other"] += 1
    return out


def bpe_line(text):
    """A display line under our BPE: per word, the chunks after each merge (only the steps that change it)."""
    ws = re.findall(r"[a-z']+", text.lower())
    steps = []
    prev = None
    for k in range(len(merges) + 1):
        seg = [bpe.apply(w, merges, k) for w in ws]
        if seg != prev:
            steps.append({"k": k, "words": seg})
            prev = seg
    return {"text": text, "words": ws, "steps": steps}


vocab_ids = {c: i for i, c in enumerate(letters)}
for m in merges:
    vocab_ids[m["ab"]] = len(letters) + m["k"] - 1

def field_patch(half=9):
    """The readable neighbours of " strawberry" in id order (a 19 x 19 window); anything profane is left
    blank (the cell still shows, its text does not)."""
    cols = lt["field"]["cols"]
    r0, c0 = lt["field"]["strawberry"]
    rude = re.compile(r"fuck|shit|cunt|nigg|porn|slut|whore|bitch", re.I)
    out = {}
    for k, v in lt["field"]["patch"].items():
        r, c = divmod(int(k), cols)
        if abs(r - r0) <= half and abs(c - c0) <= half:
            out[k] = "" if rude.search(v) else v
    return out


words_llama = lt["words"]
pick = [" strawberry", "strawberry", " Strawberry", " STRAWBERRY", " strawberries", " s t r a w b e r r y",
        " Motown", " ChatGPT", " Llama", " llama", " tokenization", " hello", " HELLO", " unhappiness", " 1234567"]
d = {
    "llama": {
        "tokenizer": lt["tokenizer"], "vocab_total": lt["vocab_total"], "vocab_regular": lt["vocab_regular"],
        "special": lt["special"], "question": lt["question"], "question_tokens": lt["question_tokens"],
        "chat_prompt_tokens": lt["chat_prompt_tokens"],
        # Next Token's recorded answer (projects/next-token/data/llm.json, seed 7): its own tokens
        "answer": "There are 2 r's in the word 'strawberry'.", "answer_tokens": lt["answer_tokens"],
        "words": {w: words_llama[w] for w in pick},
        "field": {"cols": lt["field"]["cols"], "strawberry": lt["field"]["strawberry"], "patch": field_patch()},
    },
    "spell_test": {
        "model": st["model"], "gpu": st["gpu"], "n": st["n"], "temperature": st["temperature"], "top_p": st["top_p"],
        "seed": st["seed"], "max_new_tokens": 40,
        "word": {"question": st["variants"]["word"]["question"], "tokens": st["variants"]["word"]["question_tokens"],
                 "greedy": st["variants"]["word"]["greedy"], "said": said(st["variants"]["word"])},
        "spaced": {"question": st["variants"]["spaced"]["question"],
                   "tokens": st["variants"]["spaced"]["question_tokens"],
                   "greedy": st["variants"]["spaced"]["greedy"], "said": said(st["variants"]["spaced"])},
    },
    "bpe": {
        "corpus": {"lines": len(lines), "words": len(words), "unique_words": len(set(words)),
                   "chars": sum(len(w) for w in words), "alphabet": "".join(letters)},
        "merges": merges, "vocab": len(letters) + len(merges), "ids": vocab_ids,
        "line": bpe_line("the model reads the numbers never the letters"),
        "strawberry": bpe_line("strawberry"),
    },
}
# the spaced-out question's ids, for the input row
tok_ids = {x["t"]: x["id"] for x in lt["words"][" s t r a w b e r r y"]}
d["spell_test"]["spaced"]["ids"] = [x["id"] for x in lt["words"][" s t r a w b e r r y"]]
(P / "data/tok.json").write_text(json.dumps(d, ensure_ascii=False))
print("word said", d["spell_test"]["word"]["said"], "| spaced said", d["spell_test"]["spaced"]["said"])
print("bpe", len(merges), "merges, vocab", d["bpe"]["vocab"], "| line steps", len(d["bpe"]["line"]["steps"]))
print("size", (P / "data/tok.json").stat().st_size, "bytes")
