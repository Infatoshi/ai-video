"""Train a small character-level GPT from scratch on tiny Shakespeare and record how it learns to write.

  python train.py [--device cuda|cpu] [--max_iters 5000] [--out run]

The model and recipe follow nanoGPT's shakespeare_char config (Karpathy, MIT): 6 layers, 6 heads, 384 wide,
256-character context, dropout 0.2, AdamW lr 1e-3 (warmup 100, cosine to 1e-4), batch 64, 5000 steps, bf16
autocast on the GPU. The data is tiny Shakespeare (karpathy/char-rnn, 1,115,394 characters of the plays),
split 90/10 into train and validation text; the model never trains on the validation text.

Writes <out>/run.json:
  config, parameter count, vocabulary, dataset sha256, device, torch version
  steps[]      every step: the loss it was trained on (one batch, dropout on), learning rate, elapsed seconds
  evals[]      every 250 steps and at each sample step: train and validation loss on 20 fixed batches (no dropout)
  samples[]    at steps 0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000:
               400 characters after the fixed prompt, temperature 1.0, the same seed every time (so the only
               thing that changes between samples is the weights), plus the elapsed wall-clock seconds
and <out>/ckpt/step_<N>.pt (the weights at each sample step) for probe.py.
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROMPT = "ROMEO:\n"
SEED = 1337
SAMPLE_SEED = 42
SAMPLE_STEPS = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]


class Block(nn.Module):
    def __init__(s, d, h, p):
        super().__init__()
        s.h, s.p = h, p
        s.ln1, s.ln2 = nn.LayerNorm(d, bias=False), nn.LayerNorm(d, bias=False)
        s.qkv, s.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        s.fc, s.out = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        s.drop = nn.Dropout(p)

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, 2)
        q, k, v = (t.view(B, T, s.h, C // s.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=s.p if s.training else 0.0, is_causal=True)
        x = x + s.drop(s.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + s.drop(s.out(F.gelu(s.fc(s.ln2(x)))))


class GPT(nn.Module):
    def __init__(s, vocab, block, n_layer, n_head, d, p):
        super().__init__()
        s.block = block
        s.wte, s.wpe = nn.Embedding(vocab, d), nn.Embedding(block, d)
        s.drop = nn.Dropout(p)
        s.blocks = nn.ModuleList(Block(d, n_head, p) for _ in range(n_layer))
        s.ln_f = nn.LayerNorm(d, bias=False)
        s.head = nn.Linear(d, vocab, bias=False)
        s.head.weight = s.wte.weight  # weight tying, as in GPT-2 / nanoGPT
        for n, m in s.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        for n, p_ in s.named_parameters():
            if n.endswith("proj.weight") or n.endswith("out.weight"):
                nn.init.normal_(p_, 0.0, 0.02 / math.sqrt(2 * n_layer))

    def forward(s, idx, targets=None):
        T = idx.shape[1]
        x = s.drop(s.wte(idx) + s.wpe(torch.arange(T, device=idx.device)))
        for b in s.blocks:
            x = b(x)
        logits = s.head(s.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--out", default="run")
    ap.add_argument("--sample_chars", type=int, default=400)
    ap.add_argument("--eval_iters", type=int, default=20)
    a = ap.parse_args()
    cfg = dict(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2, batch_size=64, lr=1e-3, min_lr=1e-4,
               warmup=100, max_iters=a.max_iters, lr_decay_iters=a.max_iters, beta1=0.9, beta2=0.99,
               weight_decay=0.1, grad_clip=1.0, eval_every=250, eval_iters=a.eval_iters, prompt=PROMPT, seed=SEED,
               sample_seed=SAMPLE_SEED, sample_temperature=1.0, sample_chars=a.sample_chars)
    out = HERE / a.out
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    raw = (HERE / "input.txt").read_bytes()
    text = raw.decode()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    split = {"train": data[:n], "val": data[n:]}
    dev = a.device
    torch.manual_seed(SEED)
    B, T = cfg["batch_size"], cfg["block_size"]

    def batch(name, g):
        d = split[name]
        ix = torch.randint(len(d) - T - 1, (B,), generator=g)
        x = torch.stack([d[i:i + T] for i in ix])
        y = torch.stack([d[i + 1:i + T + 1] for i in ix])
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

    # the same 20 batches every evaluation, so the curve moves only because the model does
    fixed = {k: [batch(k, torch.Generator().manual_seed(1000 + i + (500 if k == "val" else 0)))
                 for i in range(cfg["eval_iters"])] for k in split}
    model = GPT(len(chars), T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["dropout"]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_nonpos = n_params - model.wpe.weight.numel()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": nodecay, "weight_decay": 0.0}], lr=cfg["lr"],
                            betas=(cfg["beta1"], cfg["beta2"]), fused=dev == "cuda")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)

    def lr_at(it):
        if it < cfg["warmup"]:
            return cfg["lr"] * (it + 1) / (cfg["warmup"] + 1)
        r = min(1.0, (it - cfg["warmup"]) / (cfg["lr_decay_iters"] - cfg["warmup"]))
        return cfg["min_lr"] + 0.5 * (1 + math.cos(math.pi * r)) * (cfg["lr"] - cfg["min_lr"])

    @torch.no_grad()
    def evaluate():
        model.eval()
        r = {}
        for k, bs in fixed.items():
            ls = []
            for x, y in bs:
                with amp:
                    ls.append(model(x, y)[1].item())
            r[k] = sum(ls) / len(ls)
        model.train()
        return r

    @torch.no_grad()
    def sample():
        model.eval()
        g = torch.Generator(device=dev).manual_seed(SAMPLE_SEED)
        idx = torch.tensor([[stoi[c] for c in PROMPT]], device=dev)
        for _ in range(cfg["sample_chars"]):
            with amp:
                logits = model(idx[:, -T:])[0][:, -1, :].float()
            nxt = torch.multinomial(F.softmax(logits / cfg["sample_temperature"], -1), 1, generator=g)
            idx = torch.cat([idx, nxt], 1)
        model.train()
        return "".join(chars[i] for i in idx[0].tolist()[len(PROMPT):])

    steps, evals, samples = [], [], []
    g = torch.Generator().manual_seed(SEED)
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    sync()
    t0 = time.time()
    overhead = 0.0  # seconds spent on evaluation, sampling and checkpoints (inside the wall clock)
    for it in range(cfg["max_iters"] + 1):
        if it % cfg["eval_every"] == 0 or it in SAMPLE_STEPS:
            sync(); ta = time.time()
            e = evaluate()
            evals.append(dict(step=it, train=round(e["train"], 4), val=round(e["val"], 4), t=round(ta - t0, 2)))
            if it in SAMPLE_STEPS:
                s = sample()
                samples.append(dict(step=it, t=round(ta - t0, 2), text=s))
                torch.save(model.state_dict(), out / "ckpt" / f"step_{it}.pt")
                print(f"step {it} t {ta - t0:.1f}s train {e['train']:.4f} val {e['val']:.4f}\n  {s[:120]!r}", flush=True)
            sync(); overhead += time.time() - ta
        if it == cfg["max_iters"]:
            break
        lr = lr_at(it)
        for gr in opt.param_groups:
            gr["lr"] = lr
        x, y = batch("train", g)
        with amp:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        # the loss of the batch this step learned from, and the wall clock when the step finished
        steps.append(dict(step=it + 1, loss=round(loss.item(), 4), lr=round(lr, 6), t=round(time.time() - t0, 3)))
    sync()
    wall = time.time() - t0
    res = dict(config=cfg, n_params=n_params, n_params_without_position=n_params_nonpos, vocab="".join(chars),
               vocab_size=len(chars), dataset=dict(name="tiny Shakespeare", url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                                                   chars=len(text), sha256=hashlib.sha256(raw).hexdigest(),
                                                   train_chars=n, val_chars=len(data) - n),
               device=torch.cuda.get_device_name() if dev == "cuda" else "cpu", torch=torch.__version__,
               wall_s=round(wall, 2), overhead_s=round(overhead, 2), chars_per_step=B * T,
               started=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
               steps=steps, evals=evals, samples=samples)
    (out / "run.json").write_text(json.dumps(res, indent=1))
    print(f"done: {n_params:,} params, {wall / 60:.2f} min wall ({overhead:.1f} s eval/sample), "
          f"final train {evals[-1]['train']:.4f} val {evals[-1]['val']:.4f}", flush=True)
    # the full GPU run hands its lease straight on to the song (after_train.sh: lyrics from these samples, then
    # generate and score the takes), so the GPU never sits idle between the two. exec, not a child process: the
    # lease stays with this pid while this process's CUDA context (about 2 GB) goes away. (2026-09-28's run used
    # subprocess.run: the training process kept 2 GB of the 3090 and half the song batches ran out of memory.)
    after = HERE / "after_train.sh"
    if dev == "cuda" and a.out == "run" and after.exists():
        sys.stdout.flush()
        os.execvp("bash", ["bash", str(after)])


if __name__ == "__main__":
    main()
