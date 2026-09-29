"""Record what Llama 3.1 8B Instruct actually does with one question, for the video (run on the GPU host,
from ~/dev/cuda-song with its score/ venv; bf16 on the 3090).

  uv run --project score python dump.py out/llm.json

Everything the video shows about the model comes from here: the chat tokens, the sampled answer (the
model's own default sampling: temperature 0.6, top-p 0.9, fixed seed) with each step's top-10 and the
uniform draw that picked the token, the logit lens up the 32 layers, the residual norm per layer,
every head's attention row for the new token (uint8, base64), feed-forward activations (|act| as uint8)
at three layers, input embeddings of a word list projected to 3D (PCA), a speculative-decoding round
with Llama 3.2 1B as the draft, and measured decode speed at batch 1 and batch 100.
"""
import base64, json, math, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BIG, DRAFT = "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"
QUESTION = "How many r's are in strawberry?"
TEMP, TOP_P, SEED, MAX_NEW = 0.6, 0.9, 7, 48
FF_LAYERS = (0, 15, 31)
WORDS = ["king", "queen", "prince", "man", "woman", "cat", "kitten", "dog", "puppy", "apple", "banana",
         "cherry", "grape", "Paris", "France", "London", "England", "Tokyo", "Japan", "GPU", "CPU", "memory",
         "kernel", "red", "blue", "green", "one", "two", "three", "run", "walk", "jump", "happy", "sad", "angry"]


def u8(x):
    x = x.float().abs()
    x = (x / x.max().clamp(min=1e-9) * 255).round().to(torch.uint8).cpu().numpy()
    return base64.b64encode(x.tobytes()).decode()


def main(out):
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BIG)
    model = AutoModelForCausalLM.from_pretrained(BIG, dtype=torch.bfloat16, attn_implementation="eager").cuda()
    model.eval()
    cfg = model.config
    L, H, KVH, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size
    hd = D // H
    norm, head = model.model.norm, model.lm_head

    ids = tok.apply_chat_template([{"role": "user", "content": QUESTION}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].cuda()
    text = lambda i: tok.decode([int(i)])
    prompt = [{"id": int(i), "t": text(i)} for i in ids[0]]
    q_ids = tok(QUESTION, add_special_tokens=False)["input_ids"]

    ff = {}
    hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda m, a, l=l: ff.__setitem__(l, a[0][0, -1].detach())) for l in FF_LAYERS]

    steps, past, cur = [], None, ids
    t0 = time.time()
    with torch.no_grad():
        for s in range(MAX_NEW):
            o = model(input_ids=cur, past_key_values=past, use_cache=True, output_attentions=True, output_hidden_states=True)
            past = o.past_key_values
            logits = o.logits[0, -1].float()
            p1 = torch.softmax(logits, -1)
            pt = torch.softmax(logits / TEMP, -1)
            sp, si = pt.sort(descending=True)
            keep = (sp.cumsum(0) - sp) < TOP_P
            sp, si = sp[keep] / sp[keep].sum(), si[keep]
            u = float(torch.rand(()))
            pick = int((sp.cumsum(0) < u).sum().clamp(max=len(sp) - 1))
            nxt = int(si[pick])
            top = torch.topk(p1, 10)
            lens = []
            for l, h in enumerate(o.hidden_states):
                # the last hidden state is already after the final norm; the others need it
                pl = torch.softmax(head(h[0, -1] if l == len(o.hidden_states) - 1 else norm(h[0, -1])).float(), -1)
                tp = torch.topk(pl, 3)
                lens.append({"norm": round(float(h[0, -1].float().norm()), 2),
                             "top": [[text(i), round(float(v), 4)] for v, i in zip(tp.values, tp.indices)]})
            att = torch.stack([a[0, :, -1] for a in o.attentions])  # L x H x ctx
            steps.append({
                "pos": int(ids.shape[1] + s), "tok": {"id": nxt, "t": text(nxt)},
                "top10_t1": [[text(i), round(float(v), 4)] for v, i in zip(top.values, top.indices)],
                "nucleus": [[text(i), round(float(v), 4)] for v, i in zip(sp[:10], si[:10])],
                "nucleus_size": len(sp), "u": round(u, 4), "pick": pick,
                "entropy_t1": round(float(-(p1 * p1.clamp(min=1e-12).log()).sum()), 3),
                "lens": lens, "attn": {"shape": list(att.shape), "max": round(float(att.max()), 4), "u8": u8(att.reshape(-1, att.shape[-1]))},
                "ff": {str(l): {"n": ff[l].numel(), "u8": u8(ff[l]), "active": int((ff[l].abs() > ff[l].abs().max() * 0.1).sum())} for l in FF_LAYERS},
            })
            print(s, repr(text(nxt)), f"u={u:.3f}", flush=True)
            cur = torch.tensor([[nxt]], device="cuda")
            if nxt in (tok.convert_tokens_to_ids("<|eot_id|>"), tok.eos_token_id):
                break
    for h in hooks:
        h.remove()
    answer_ids = [s["tok"]["id"] for s in steps]

    emb = model.model.embed_tokens.weight.detach()
    wl = []
    for w in WORDS:
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            wl.append((w, t[0]))
    E = emb[[i for _, i in wl]].float()
    Ec = E - E.mean(0)
    _, _, V = torch.linalg.svd(Ec, full_matrices=False)
    P = (Ec @ V[:3].T).cpu().numpy()
    P = P / np.abs(P).max()
    cos = torch.nn.functional.normalize(E, dim=-1) @ torch.nn.functional.normalize(E, dim=-1).T
    emb_first = {prompt[i]["t"]: [round(float(x), 4) for x in emb[prompt[i]["id"]][:64].float()] for i in range(len(prompt))}

    # speculative decoding, one round at every point of the answer: the draft proposes 4 greedy tokens,
    # the big model scores all 4 in one pass, and the prefix where its own top pick agrees is kept
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16).cuda()
    rounds = []
    with torch.no_grad():
        for k0 in range(len(answer_ids) - 1):
            ctx = torch.cat([ids, torch.tensor([answer_ids[:k0]], device="cuda", dtype=ids.dtype)], 1) if k0 else ids
            prop, c = [], ctx
            for _ in range(4):
                n = int(draft(input_ids=c).logits[0, -1].argmax())
                prop.append(n)
                c = torch.cat([c, torch.tensor([[n]], device="cuda", dtype=ids.dtype)], 1)
            big = model(input_ids=c).logits[0, ctx.shape[1] - 1:].float().argmax(-1).tolist()  # 5: one bonus pick
            acc = 0
            for a, b in zip(prop, big):
                if a != b:
                    break
                acc += 1
            rounds.append({"at": k0, "draft": [text(i) for i in prop], "big": [text(i) for i in big], "accepted": acc})
    # the round the video shows: its accepted draft is the answer's own next tokens (the answer was sampled,
    # the check is greedy) and it stops before the end-of-turn token; most accepted, then latest
    N = len(answer_ids)
    atext = [text(i) for i in answer_ids]
    ok = [r for r in rounds if r["draft"][:r["accepted"]] == atext[r["at"]:r["at"] + r["accepted"]] and r["at"] + r["accepted"] <= N - 1]
    best = max(ok or rounds, key=lambda r: (r["accepted"], r["at"]))
    spec = {**best, "context_answer_tokens": best["at"], "rounds": rounds}
    del draft
    torch.cuda.empty_cache()

    # measured decode speed (tokens/s) at batch 1 and batch 100, 64 new tokens, greedy
    speed = {}
    for B in (1, 100):
        x = ids.repeat(B, 1)
        torch.cuda.synchronize()
        with torch.no_grad():
            o = model(input_ids=x, use_cache=True)
            past, nx = o.past_key_values, o.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(64):
                o = model(input_ids=nx, past_key_values=past, use_cache=True)
                past, nx = o.past_key_values, o.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
        speed[B] = round(64 * B / (time.time() - t), 1)
        print("batch", B, speed[B], "tok/s", flush=True)

    params = sum(p.numel() for p in model.parameters())
    res = {
        "model": BIG, "gpu": torch.cuda.get_device_name(0), "dtype": "bf16",
        "config": {"layers": L, "heads": H, "kv_heads": KVH, "hidden": D, "head_dim": hd,
                   "ffn": cfg.intermediate_size, "vocab": cfg.vocab_size, "params": params,
                   "weight_bytes": params * 2, "kv_bytes_per_token": 2 * L * KVH * hd * 2},
        "question": QUESTION,
        "question_tokens": [text(i) for i in q_ids],
        "strawberry": {w: [text(i) for i in tok(w, add_special_tokens=False)["input_ids"]]
                       for w in ("strawberry", " strawberry", " Strawberry", "raspberry")},
        "prompt": prompt, "sampling": {"temperature": TEMP, "top_p": TOP_P, "seed": SEED},
        "answer": tok.decode(answer_ids), "steps": steps,
        "embed": {"words": [w for w, _ in wl], "pca3": P.round(4).tolist(),
                  "cos": cos.cpu().numpy().round(3).tolist(), "first64": emb_first},
        "spec": spec, "speed_tok_s": speed, "wall_s": round(time.time() - t0, 1),
    }
    json.dump(res, open(out, "w"))
    print("answer:", repr(res["answer"]))
    print("strawberry:", res["strawberry"])
    print("spec:", spec)


if __name__ == "__main__":
    main(sys.argv[1])
    sys.stdout.flush()
