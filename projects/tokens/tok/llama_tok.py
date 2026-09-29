"""The real Llama 3.1 tokenizer on the video's words (the GPU host, CPU only; no model weights, no GPU).

  uv run --project ~/dev/cuda-song/score python llama_tok.py out.json

Reads the tokenizer from the local HF cache (meta-llama/Llama-3.1-8B-Instruct; the gated files stay in the
cache, only the derived splits and ids are written). Records: vocabulary size (regular + special), the
number of BPE merges, Next Token's question and answer (split + ids), "strawberry" with and without a
leading space and in other spellings, and a list of other words with their splits, so the video can pick
the surprising ones.
"""
import json, sys

from transformers import AutoTokenizer

REPO = "meta-llama/Llama-3.1-8B-Instruct"
FIELD_COLS = 358  # 358 x 359 >= 128,256
QUESTION = "How many r's are in strawberry?"
ANSWER = "There are 2 r's in the word 'strawberry'."
WORDS = [
    "strawberry", " strawberry", "Strawberry", " Strawberry", " STRAWBERRY", "STRAWBERRY", " strawberries",
    " straw", " berry", " raspberry", "raspberry", " blueberry", " r", "r", " rr", " rrr",
    " s t r a w b e r r y", " s-t-r-a-w-b-e-r-r-y", " S T R A W B E R R Y",
    " the", " token", " tokens", " tokenizer", " tokenization", " letters", " letter", " number", " numbers",
    " chunk", " chunks", " model", " Motown", " doo", " wop", " Supremes",
    " unbelievable", " antidisestablishmentarianism", " Mississippi", " bookkeeper", " lollipop",
    " banana", " bananas", " occurrence", " necessary", " embarrassment", " onomatopoeia",
    " 2026", " 1234567", " 73700", " 128256", " hello", " Hello", " HELLO", " helloooo",
    " ChatGPT", " Llama", " llama", " anvil", " Elliot",
    " cat", " cats", " dog", " running", " unhappiness", " happiness",
]


def main(out):
    tok = AutoTokenizer.from_pretrained(REPO)
    back = tok.backend_tokenizer
    model = json.loads(back.to_str())["model"]
    n_regular = len(model["vocab"])
    n_merges = len(model["merges"])
    n_total = len(tok)
    specials = n_total - n_regular

    def split(s):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        return [{"id": int(i), "t": tok.decode([i])} for i in ids]

    chat = tok.apply_chat_template([{"role": "user", "content": QUESTION}], add_generation_prompt=True,
                                   return_dict=True)["input_ids"]
    # first 256 regular ids are single bytes: how many are there, and which id is "r"
    r_id = tok.convert_tokens_to_ids("r")
    d = {
        "tokenizer": REPO,
        "vocab_total": n_total,
        "vocab_regular": n_regular,
        "special": specials,
        "merges": n_merges,
        "question": QUESTION,
        "question_tokens": split(QUESTION),
        "chat_prompt_tokens": len(chat),
        "answer": ANSWER,
        "answer_tokens": split(ANSWER),
        "r_id": int(r_id),
        "words": {w: split(w) for w in WORDS},
    }
    # every regular token that is only the letter r repeated, and every token containing "berry"
    vocab = model["vocab"]
    dec = lambda s: tok.convert_tokens_to_string([s])
    d["r_only_tokens"] = sorted({dec(k): v for k, v in vocab.items() if dec(k).strip() and set(dec(k).strip()) == {"r"}}.items(), key=lambda x: x[1])
    d["berry_tokens"] = sorted([(dec(k), v) for k, v in vocab.items() if "berry" in dec(k).lower()], key=lambda x: x[1])
    d["strawberry_rank"] = vocab.get("Ġstrawberry")
    # the vocabulary laid out as a grid in id order (FIELD_COLS wide): the real tokens in a patch around
    # " strawberry", for the video's zoom from readable tiles out to all 128,256 as dots
    r0, c0 = divmod(73700, FIELD_COLS)
    patch = {}
    for r in range(r0 - 12, r0 + 13):
        for c in range(c0 - 26, c0 + 27):
            i = r * FIELD_COLS + c
            if 0 <= c < FIELD_COLS and 0 <= i < n_total:
                patch[i] = tok.decode([i])
    d["field"] = {"cols": FIELD_COLS, "strawberry": [r0, c0], "patch": patch}
    open(out, "w").write(json.dumps(d, indent=1, ensure_ascii=False))
    print(f"vocab {n_total} = {n_regular} regular + {specials} special; merges {n_merges}")
    print("question:", [(x["t"], x["id"]) for x in d["question_tokens"]], "chat prompt", len(chat))
    print("answer:", [(x["t"], x["id"]) for x in d["answer_tokens"]])
    for w, s in d["words"].items():
        print(f"{w!r:34} {len(s):2} {[x['t'] for x in s]} {[x['id'] for x in s]}")
    print("r-only:", d["r_only_tokens"])
    print("berry:", d["berry_tokens"][:40])


if __name__ == "__main__":
    main(sys.argv[1])
