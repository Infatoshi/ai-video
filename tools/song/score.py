"""Rank song candidates: lyric accuracy (Demucs vocals -> Whisper large-v3 -> WER against the sung
lyrics) and Audiobox Aesthetics (CE/CU/PC/PQ). Run on the other GPU host from score/'s venv.

  uv run --project score python score.py out/<name> [out/<name2> ...]

Writes <dir>/scores.json and prints a table sorted by word recall. Stems and transcripts are cached next
to each wav (<seed>.vocals.wav, <seed>.whisper.json), so reruns only score new files.
"""
import json, os, re, sys
from pathlib import Path

_HF = Path(__file__).resolve().parent / ".hf"  # the other GPU host's shared /data/hf is not writable
os.environ["HF_HOME"] = str(_HF)
os.environ["HF_HUB_CACHE"] = str(_HF / "hub")

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
LYRICS = (HERE / "lyrics.sing.txt").read_text()

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n):
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return str(n)


# Applied to both reference and hypothesis after lowercasing and punctuation removal.
CANON = [
    (r"\bg p us\b|\bgpus\b", "gpus"), (r"\bg p u\b|\bgpu\b", "gpu"), (r"\bk v\b|\bkv\b|\bkay vee\b", "kv"),
    (r"\bm m a\b|\bmma\b", "mma"), (r"\bn v link\b|\bnv link\b|\bnvlink\b|\benvy link\b", "nvlink"),
    (r"\b(koo|ku|coo|cu|q|cue|kou) ?blas\b|\bcublas\b|\bqblas\b", "cublas"), (r"\bem ?nist\b|\bm nist\b|\bmnist\b", "mnist"),
    (r"\bmem ?copy\b|\bmemcpy\b|\bmem cpy\b", "memcpy"), (r"\bfloat ?(four|4)\b", "float4"),
    (r"\bnormal ?float ?(four|4)\b", "normalfloat4"), (r"\bpy ?torch\b|\bpie torch\b", "pytorch"),
    (r"\bcuda\b|\bkuda\b|\bkoodah\b|\bcooda\b", "cuda"), (r"\bcut ?lass\b", "cutlass"),
    (r"\ball ?re ?duce\b", "allreduce"), (r"\bmicro ?batch\b", "microbatch"), (r"\btera ?flops\b", "teraflops"),
    (r"\b(\d+) ?x ?(\d+)\b", lambda m: f"{num_words(m[1])} by {num_words(m[2])}"),
    (r"\bn ?x ?n\b", "n by n"),
    (r"\bsoft ?max\b", "softmax"), (r"\bhundred and\b", "hundred"), (r"\b(\w+) 000\b", r"\1 thousand"),
    (r"\b128 thousand\b", "hundred twenty eight thousand"), (r"\bfeed forward\b", "feedforward"),
    (r"\bstraw ?berry\b", "strawberry"), (r"\bkv cache\b|\bk v cache\b|\bkay vee cache\b", "kv cache"), (r"\bpipe ?line\b", "pipeline"), (r"\broof ?line\b", "roofline"),
    # tokens (ML, slowly #2): "A-I", the chunks of strawberry spelled out, its token id sung digit by digit
    (r"\ba i\b", "ai"), (r"\bstr aw berry\b|\bs t r aw berry\b", "s t r a w berry"),
    (r"\bseven three seven (?:zero|oh|o) (?:zero|oh|o)\b|\b73700\b|\bseventy three 700\b|\b737 zero\b", "seven three seven oh oh"),
    # meaning (ML, slowly #3): 4,096 numbers per row ("4,096" -> "four 096" after the digit step), "chat bot"
    (r"\bfour 096\b|\b4096\b|\bfour thousand and ninety six\b", "four thousand ninety six"), (r"\bchat ?bot\b", "chatbot"),
]


def norm(text):
    t = re.sub(r"\[[^\]]*\]", " ", text.lower())
    t = t.replace("’", "'").replace("-", " ")
    t = re.sub(r"(\d+)", lambda m: " " + num_words(m[1]) + " " if len(m[1]) <= 2 else m[1], t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    t = re.sub(r"'", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    for pat, rep in CANON:
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


REF = norm(LYRICS)
REF_LINES = [norm(l) for l in LYRICS.splitlines() if l.strip() and not l.startswith("[")]


def separate(wavs):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torchaudio
    model = None
    for w in wavs:
        dst = w.with_suffix(".vocals.wav")
        if dst.exists():
            continue
        if model is None:
            model = get_model("htdemucs_ft").cuda().eval()
        a, sr = sf.read(str(w), dtype="float32", always_2d=True)
        y = torchaudio.functional.resample(torch.from_numpy(a.T.copy()), sr, model.samplerate)
        if y.shape[0] == 1:
            y = y.repeat(2, 1)
        with torch.no_grad():
            s = apply_model(model, y[None].cuda(), split=True, overlap=0.25, progress=False)[0]
        v = s[model.sources.index("vocals")].mean(0).cpu().numpy()
        sf.write(str(dst), v, model.samplerate)


def transcribe(wavs):
    from transformers import pipeline
    asr = None
    out = {}
    for w in wavs:
        j = w.with_suffix(".whisper.json")
        if not j.exists():
            if asr is None:
                asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                               torch_dtype=torch.float16, device="cuda:0")
            import librosa
            y, _ = librosa.load(str(w.with_suffix(".vocals.wav")), sr=16000)
            # sequential long-form decoding with Whisper's own repetition/fallback guards
            r = asr(y, return_timestamps=True, generate_kwargs={
                "language": "en", "task": "transcribe", "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.35, "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),
                "logprob_threshold": -1.0, "no_speech_threshold": 0.6})
            j.write_text(json.dumps({"text": r["text"], "chunks": [
                {"t": list(c["timestamp"]), "text": c["text"]} for c in r.get("chunks", [])]}, indent=1))
        out[w] = json.loads(j.read_text())
    return out


def aesthetics(wavs):
    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    res = []
    for w in wavs:  # tensors, not paths: torchaudio.load needs torchcodec here
        a, sr = sf.read(str(w), dtype="float32", always_2d=True)
        res += pred.forward([{"path": torch.from_numpy(a.T.copy()), "sample_rate": sr}])
    return dict(zip(wavs, res))


def main(dirs):
    import jiwer
    global REF, REF_LINES
    for d in map(Path, dirs):
        if (d / "lyrics.txt").exists():  # per-directory reference lyrics (e.g. the p(doom) baseline)
            txt = (d / "lyrics.txt").read_text()
            REF = norm(txt)
            REF_LINES = [norm(l) for l in txt.splitlines() if l.strip() and not l.startswith("[")]
        wavs = sorted(p for p in d.glob("*.wav") if p.stem.isdigit())
        separate(wavs)
        tr = transcribe(wavs)
        ae = aesthetics(wavs)
        rows = []
        for w in wavs:
            ref, ref_lines = REF, REF_LINES
            meta = w.with_suffix(".json")
            if meta.exists() and not (d / "lyrics.txt").exists():  # score against what this take was asked to sing
                txt = json.loads(meta.read_text()).get("lyrics") or LYRICS
                ref = norm(txt)
                ref_lines = [norm(l) for l in txt.splitlines() if l.strip() and not l.startswith("[")]
            hyp = norm(tr[w]["text"])
            m = jiwer.process_words(ref, hyp)
            # lines whose words mostly failed to come through
            bad = []
            for line in ref_lines:
                hits = sum(1 for wd in line.split() if f" {wd} " in f" {hyp} ")
                if hits < 0.6 * len(line.split()):
                    bad.append(line)
            info = sf.info(str(w))
            v, sr = sf.read(str(w.with_suffix(".vocals.wav")))
            rms = np.sqrt(np.convolve(v ** 2, np.ones(sr // 10) / (sr // 10), "same"))
            vocal_frac = float((rms > 0.02).mean())
            recall = m.hits / max(1, len(ref.split()))  # share of lyric words heard, in order
            rows.append({"seed": w.stem, "recall": round(recall, 3), "wer": round(m.wer, 3), "dur": round(info.duration, 1),
                         "vocal_frac": round(vocal_frac, 2),
                         **{k: round(float(x), 2) for k, x in ae[w].items()},
                         "bad_lines": bad, "hyp": tr[w]["text"]})
        rows.sort(key=lambda r: -r["recall"])
        (d / "scores.json").write_text(json.dumps(rows, indent=1))
        print(f"== {d}")
        for r in rows:
            print(f"{r['seed']:>12} recall {r['recall']:.3f} wer {r['wer']:.3f} dur {r['dur']:6.1f} voc {r['vocal_frac']:.2f} "
                  f"CE {r.get('CE', 0):.2f} CU {r.get('CU', 0):.2f} PC {r.get('PC', 0):.2f} PQ {r.get('PQ', 0):.2f}"
                  f"  bad {len(r['bad_lines'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
