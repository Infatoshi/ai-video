"""Does Llama 3.1 8B count the r's better when it gets the letters? (the GPU host's 3090, bf16, under the GPU lease)

  uv run --project ~/dev/cuda-song/score python spell_test.py out.json

The same question in three spellings: the word as it is (" strawberry" is one token), and the letters
separated by spaces or hyphens (every letter its own token). Each is asked N times with the model's own
default sampling (temperature 0.6, top-p 0.9, seeded), plus once greedy; the first number in each answer is
tallied. Nothing else is claimed from it: it measures what this model answers for each input.
"""
import json, re, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "meta-llama/Llama-3.1-8B-Instruct"
N, TEMP, TOP_P, SEED, MAX_NEW = 100, 0.6, 0.9, 7, 40
VARIANTS = {
    "word": "How many r's are in strawberry?",
    "spaced": "How many r's are in s t r a w b e r r y?",
    "hyphens": "How many r's are in s-t-r-a-w-b-e-r-r-y?",
}
NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def first_number(s):
    m = re.search(r"\b(\d+|zero|one|two|three|four|five|six)\b", s.lower())
    if not m:
        return None
    x = m[1]
    return int(x) if x.isdigit() else NUM[x]


def main(out):
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).cuda().eval()
    res = {"model": REPO, "gpu": torch.cuda.get_device_name(0), "n": N, "temperature": TEMP, "top_p": TOP_P,
           "seed": SEED, "variants": {}}
    for name, q in VARIANTS.items():
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"].cuda()
        q_tok = [tok.decode([i]) for i in tok(q, add_special_tokens=False)["input_ids"]]
        t0 = time.time()
        with torch.no_grad():
            g = model.generate(ids, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
            greedy = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
            torch.manual_seed(SEED)
            s = model.generate(ids.repeat(N, 1), do_sample=True, temperature=TEMP, top_p=TOP_P,
                               max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
        answers = [tok.decode(x[ids.shape[1]:], skip_special_tokens=True) for x in s]
        nums = [first_number(a) for a in answers]
        tally = {}
        for n in nums:
            tally[str(n)] = tally.get(str(n), 0) + 1
        res["variants"][name] = {"question": q, "question_tokens": q_tok, "n_question_tokens": len(q_tok),
                                 "greedy": greedy, "greedy_number": first_number(greedy), "tally": tally,
                                 "answers": answers, "secs": round(time.time() - t0, 1)}
        print(f"{name:8} {len(q_tok):2} tokens  greedy {first_number(greedy)} {greedy!r}\n         tally {tally}", flush=True)
    open(out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
