"""Ask each saved checkpoint of the run how surprised it is by a line it never trained on (CPU only).

  python probe.py [run]

The line is the first speech of the validation text (The Taming of the Shrew, which is not in the training
split): "GREMIO:\\nGood morrow, neighbour Baptista.\\n". For every checkpoint and every character after the
speaker's name: the probability the model gave the right next character (p), the loss for that guess
(-ln p, "how surprised it was"), its top 5 guesses and its whole 65-way guess. Plus the full 65-way guess for the character after
the prompt "ROMEO:\\n" at every checkpoint. And the numbers themselves: a 65 x 48 corner of the character
embedding table (one row per character) at every checkpoint, one weight's value at every checkpoint, and how far
the weights moved in single steps (steps 0->1, 1->2, 2->3 are consecutive checkpoints). Writes <run>/probe.json.
"""
import json, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F

from train import GPT, HERE, PROMPT

LINE_CTX = "\n\nGREMIO:\n"
LINE = "Good morrow, neighbour Baptista.\n"


def main(run="run"):
    out = HERE / run
    r = json.loads((out / "run.json").read_text())
    c = r["config"]
    chars = r["vocab"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    text = (HERE / "input.txt").read_text()
    assert LINE_CTX.lstrip("\n") + LINE in text[r["dataset"]["train_chars"]:][:200], "probe line must be validation text"
    model = GPT(len(chars), c["block_size"], c["n_layer"], c["n_head"], c["n_embd"], 0.0)
    model.eval()
    res = dict(line_ctx=LINE_CTX, line=LINE, prompt=PROMPT, vocab=chars, steps=[])
    ids = torch.tensor([[stoi[ch] for ch in LINE_CTX + LINE]])
    k0 = len(LINE_CTX)
    pids = torch.tensor([[stoi[ch] for ch in PROMPT]])
    prev = None
    res["moves"] = []
    ONE = (stoi["e"], 0)  # one number to follow: the character "e"'s embedding, first coordinate
    for s in r["samples"]:
        sd = torch.load(out / "ckpt" / f"step_{s['step']}.pt", map_location="cpu")
        model.load_state_dict(sd)
        flat = torch.cat([v.float().flatten() for k, v in sorted(sd.items()) if k != "head.weight"])
        if prev is not None and s["step"] - prev[0] == 1:
            d = (flat - prev[1]).abs()
            res["moves"].append(dict(step=s["step"], median_abs=float(d.median()), mean_abs=float(d.mean()),
                                     changed=float((d > 0).float().mean()), median_abs_weight=float(prev[1].abs().median())))
        prev = (s["step"], flat)
        with torch.no_grad():
            lp = F.log_softmax(model(ids)[0][0].float(), -1)       # position i predicts character i + 1
            pp = F.softmax(model(pids)[0][0, -1].float(), -1)
        per = []
        for i in range(k0 - 1, ids.shape[1] - 1):
            right = ids[0, i + 1].item()
            top = torch.topk(lp[i].exp(), 5)
            per.append(dict(ch=chars[right], p=round(lp[i, right].exp().item(), 5), loss=round(-lp[i, right].item(), 4),
                            top=[[chars[j], round(v, 4)] for v, j in zip(top.values.tolist(), top.indices.tolist())],
                            dist=[round(x, 4) for x in lp[i].exp().tolist()]))
        mean = sum(x["loss"] for x in per) / len(per)
        emb = sd["wte.weight"].float()
        res["steps"].append(dict(step=s["step"], line_loss=round(mean, 4), chars=per,
                                 after_prompt=[round(x, 5) for x in pp.tolist()],
                                 emb=[[round(x, 3) for x in row] for row in emb[:, :48].tolist()],
                                 one=round(emb[ONE].item(), 5)))
        print(f"step {s['step']:>5}: mean loss on the line {mean:.3f} (1 in {math.exp(mean):.1f}); "
              f"after prompt top {chars[int(pp.argmax())]!r} {pp.max().item():.3f}", flush=True)
    res["one"] = dict(char="e", dim=0)
    for m in res["moves"]:
        print("single step", m, flush=True)
    (out / "probe.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(*sys.argv[1:])
