"""Record where Llama 3.1 8B Instruct's attention heads look, for "Who's It?" (run on the GPU host from
~/dev/cuda-song with its score/ venv; bf16 on the 3090, eager attention so the weights come back).

  cd ~/dev/cuda-song && uv run --project score python ~/dev/ai-video/projects/who-is-it/llm/attn.py \
      ~/dev/ai-video/projects/who-is-it/out/llm

Every sentence of every pair is run once as plain text (with the <|begin_of_text|> token the model
always sees first). Saved per sentence: the tokens and the full attention of every layer x head
(32 x 32 x T x T, float16) in attn.npz; attn_meta.json has the tokens, the config and two behaviour
probes per sentence (what the model predicts when asked which one was tired/wide; the chat answer to
"what does 'it' refer to?"). The analysis (which heads flip between the two sentences of a pair, at
which word) runs on the Mac: llm/flip.py.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Every pair tried is recorded, not only the one the video uses. A/B differ by one word (or one
# pronoun); "who" is the word asking, "ref" the two candidate answers, "probe" a continuation whose
# next word says which one the model thinks it was.
PAIRS = [
    dict(id="brief", note="the brief's pair: the deciding word comes after 'it'",
         A="The animal didn't cross the street because it was too tired.",
         B="The animal didn't cross the street because it was too wide.",
         who="it", ref=["animal", "street"], change=["tired", "wide"],
         probeA=" The one that was too tired was the", probeB=" The one that was too wide was the"),
    dict(id="trophy", note="Winograd's trophy/suitcase: the deciding word comes after 'it'",
         A="The trophy didn't fit in the suitcase because it was too big.",
         B="The trophy didn't fit in the suitcase because it was too small.",
         who="it", ref=["trophy", "suitcase"], change=["big", "small"],
         probeA=" The one that was too big was the", probeB=" The one that was too small was the"),
    dict(id="before", note="the deciding word moved before 'it'",
         A="The animal stopped at the street. Too tired, it lay down.",
         B="The animal stopped at the street. Too wide, it stretched on.",
         who="it", ref=["animal", "street"], change=["tired", "wide"],
         probeA=" The one that was too tired was the", probeB=" The one that was too wide was the"),
    dict(id="plural", note="number decides: 'it' can't be the plural one",
         A="The animal didn't cross the streets because it was too tired.",
         B="The animals didn't cross the street because it was too wide.",
         who="it", ref=["animal", "street"], change=["streets", "animals"],
         probeA=" The one that was too tired was the", probeB=" The one that was too wide was the"),
    dict(id="pronoun", note="the asking word itself changes: they vs it",
         A="The animals didn't cross the street because they were too tired.",
         B="The animals didn't cross the street because it was too wide.",
         who=["they", "it"], ref=["animals", "street"], change=["they", "it"],
         probeA=" The ones that were too tired were the", probeB=" The one that was too wide was the"),
]


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").cuda()
    model.eval()
    cfg = model.config
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    text = lambda i: tok.decode([int(i)])
    arrays, meta = {}, {"model": MODEL, "gpu": torch.cuda.get_device_name(0), "dtype": "bf16",
                        "attn_implementation": "eager",
                        "config": {"layers": cfg.num_hidden_layers, "heads": cfg.num_attention_heads,
                                   "kv_heads": cfg.num_key_value_heads, "hidden": cfg.hidden_size},
                        "pairs": []}

    def next_probs(ids, words):
        with torch.no_grad():
            p = torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1)
        top = torch.topk(p, 10)
        want = {w: round(float(p[tok(" " + w, add_special_tokens=False)["input_ids"][0]]), 5) for w in words}
        return {"top10": [[text(i), round(float(v), 5)] for v, i in zip(top.values, top.indices)], "words": want}

    for pr in PAIRS:
        rec = {k: v for k, v in pr.items()}
        rec["runs"] = {}
        for side in "AB":
            s = pr[side]
            ids = tok(s, return_tensors="pt")["input_ids"].cuda()  # adds <|begin_of_text|>
            with torch.no_grad():
                o = model(input_ids=ids, output_attentions=True)
            att = torch.stack([a[0] for a in o.attentions]).float()  # L x H x T x T
            rowsum = att.sum(-1)
            arrays[f"{pr['id']}{side}"] = att.cpu().numpy().astype(np.float16)
            # behaviour: which one does the model say was tired / wide?
            probe = tok(s + pr["probe" + side], return_tensors="pt")["input_ids"].cuda()
            p_cont = next_probs(probe, sorted({*pr["ref"], "animal", "animals", "street", "streets",
                                                *(["trophy", "suitcase"] if pr["id"] == "trophy" else [])}))
            who = pr["who"] if isinstance(pr["who"], str) else pr["who"]["AB".index(side)]
            q = f'In the sentence "{s}", what does "{who}" refer to? Answer with one word.'
            chat = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)["input_ids"].cuda()
            p_chat = next_probs(chat, [w.capitalize() for w in pr["ref"]] + pr["ref"])
            with torch.no_grad():
                g = model.generate(chat, max_new_tokens=8, do_sample=False)
            ans = tok.decode(g[0, chat.shape[1]:], skip_special_tokens=True)
            rec["runs"][side] = {"text": s, "tokens": [text(i) for i in ids[0]], "ids": ids[0].tolist(),
                                 "rowsum_err": round(float((rowsum - 1).abs().max()), 6),
                                 "probe": {"text": s + pr["probe" + side], **p_cont},
                                 "chat": {"question": q, "answer": ans, **p_chat}}
            print(pr["id"], side, rec["runs"][side]["tokens"], "|", p_cont["words"], "| chat:", repr(ans), flush=True)
        meta["pairs"].append(rec)
    np.savez_compressed(out / "attn.npz", **arrays)
    meta["wall_s"] = round(time.time() - t0, 1)
    (out / "attn_meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, f"{meta['wall_s']}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
    sys.stdout.flush()
