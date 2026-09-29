"""downhill: score song takes on the Mac (Apple GPU) instead of queueing behind four other agents on the GPU host's 3090.

  cd projects/downhill && uv run --project analysis --with jiwer --with mlx-whisper python song/score_mac.py out/takes/*.wav

Same measures as tools/song/score.py + rhythm.py, adapted:
  recall    Demucs htdemucs_ft vocals (MPS) -> Whisper large-v3 (mlx-whisper) -> share of lyric words heard, in order
            (text normalised exactly as score.py: numbers to words, CANON rewrites)
  words     Whisper large-v3-turbo word timestamps -> <seed>.words.json (rhythm.py's format, for pick + structure)
  tempo     constant grid fitted to the mix's onset strength within 5% of the requested 90 BPM (rhythm.py's method)
  meter     3 vs 4: onset autocorrelation at 3 and 6 beats vs 4 and 8 beats (> 0 = the take is in 3)
  rhyme     share of line-final words within 60 ms of the 8th grid; on beat 1 of a 3-beat bar
Audiobox aesthetics are not computed here (the ear and structure.py decide the shape).
Writes out/takes/scores.json and prints a table sorted by recall.
"""
import json, re, sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
LYR = (HERE / "lyrics.sing.txt").read_text()
BPM0 = 90.0

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n):
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return str(n)


def norm(text):  # tools/song/score.py's normalisation (its CANON has nothing this song needs beyond numbers)
    t = re.sub(r"\[[^\]]*\]", " ", text.lower())
    t = t.replace("’", "'").replace("-", " ")
    t = re.sub(r"(\d+)", lambda m: " " + num_words(m[1]) + " " if len(m[1]) <= 2 else m[1], t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    t = re.sub(r"'", "", t)
    t = re.sub(r"\bhundred and\b", "hundred", t)
    t = re.sub(r"\b(\w+) 000\b", r"\1 thousand", t)
    t = re.sub(r"\b337\b", "three hundred thirty seven", t)
    t = re.sub(r"\b3121\b|\b3 121\b", "three thousand one hundred twenty one", t)
    return re.sub(r"\s+", " ", t).strip()


REF = norm(LYR)
LINES = [l.strip() for l in LYR.splitlines() if l.strip() and not l.startswith("[")]


def separate(wavs):
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torchaudio
    model = None
    for w in wavs:
        dst = w.with_suffix(".vocals.wav")
        if dst.exists():
            continue
        if model is None:
            model = get_model("htdemucs_ft").to("mps").eval()
        a, sr = sf.read(str(w), dtype="float32", always_2d=True)
        y = torchaudio.functional.resample(torch.from_numpy(a.T.copy()), sr, model.samplerate)
        with torch.no_grad():
            s = apply_model(model, y[None].to("mps"), split=True, overlap=0.25, progress=False)[0]
        v = s[model.sources.index("vocals")].mean(0).cpu().numpy()
        sf.write(str(dst), v, model.samplerate)
        print("separated", w.name, flush=True)


def transcribe(w):
    import mlx_whisper
    j = w.with_suffix(".whisper.json")
    if not j.exists():
        r = mlx_whisper.transcribe(str(w.with_suffix(".vocals.wav")), path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                                   language="en", condition_on_previous_text=False, compression_ratio_threshold=1.35,
                                   temperature=(0.0, 0.2, 0.4, 0.6, 0.8), logprob_threshold=-1.0, no_speech_threshold=0.6)
        j.write_text(json.dumps({"text": r["text"]}))
    return json.loads(j.read_text())["text"]


def words_of(w):
    import mlx_whisper
    j = w.with_suffix(".words.json")
    if not j.exists():
        r = mlx_whisper.transcribe(str(w.with_suffix(".vocals.wav")), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                   language="en", word_timestamps=True, condition_on_previous_text=False)
        ws = [{"w": x["word"], "t": [x["start"], x["end"]]} for s in r["segments"] for x in s.get("words", [])]
        j.write_text(json.dumps(ws))
    return json.loads(j.read_text())


def grid(wav):
    import librosa
    y, sr = librosa.load(str(wav), sr=22050)
    hop = 32
    o = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, lag=1, max_size=1)
    o = o / (np.percentile(o, 99) + 1e-9)
    fps, dur = sr / hop, len(y) / sr

    def score(P, off):
        idx = np.round((off + P * np.arange(int(dur / P) + 1)) * fps).astype(int)
        idx = idx[(idx > 2) & (idx < len(o) - 2)]
        return np.maximum.reduce([o[idx - 1], o[idx], o[idx + 1]]).mean()

    best = (0.0, BPM0, 0.0)
    for bpm in np.arange(BPM0 * 0.95, BPM0 * 1.05, 0.05):
        P = 60 / bpm
        for off in np.arange(0, P, 0.005):
            v = score(P, off)
            if v > best[0]:
                best = (v, bpm, off)
    _, bpm, off = best
    P = 60 / bpm
    beats = np.arange(off % P, dur, P)
    # meter: onset autocorrelation at 3/6 vs 4/8 beats; bar phase: the strongest low-band beat position mod 3
    hop2 = 256
    oo = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop2); oo = oo - oo.mean()
    ac = np.correlate(oo, oo, "full")[len(oo) - 1:]; ac /= ac[0]
    f2 = sr / hop2
    at = lambda nb: ac[int(round(nb * P * f2)) - 2:int(round(nb * P * f2)) + 3].max()
    meter = float(at(3) + at(6) - at(4) - at(8))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop2))
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[fr < 200]), sr=sr, hop_length=hop2)
    idx = np.clip(np.round(beats * f2).astype(int), 1, len(low) - 2)
    acc = np.array([low[i - 1:i + 2].max() for i in idx])
    ph = int(np.argmax([acc[k::3].mean() for k in range(3)]))
    return beats, beats[ph::3], float(bpm), meter


def main(paths):
    import jiwer
    wavs = [Path(p) for p in paths if re.fullmatch(r"\d+\.wav", Path(p).name)]
    separate(wavs)
    rows = []
    for w in wavs:
        hyp = norm(transcribe(w))
        m = jiwer.process_words(REF, hyp)
        recall = m.hits / max(1, len(REF.split()))
        bad = [l for l in LINES if sum(1 for wd in norm(l).split() if f" {wd} " in f" {hyp} ") < 0.6 * len(norm(l).split())]
        ww = [x for x in words_of(w) if re.sub(r"[^a-z0-9]", "", x["w"].lower())]
        beats, downs, bpm, meter = grid(w)
        P = 60 / bpm
        toks = [(li, re.sub(r"[^a-z0-9]", "", t.lower())) for li, l in enumerate(LINES) for t in re.split(r"[\s-]+", l) if re.sub(r"[^a-z0-9]", "", t.lower())]
        sm = SequenceMatcher(a=[t for _, t in toks], b=[re.sub(r"[^a-z0-9]", "", x["w"].lower()) for x in ww], autojunk=False)
        hit = {}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
                for k in range(i2 - i1):
                    hit[i1 + k] = ww[j1 + k]
        ends = {}
        for li in range(len(LINES)):
            idx = [i for i, (l, _) in enumerate(toks) if l == li]
            if hit.get(idx[-1]) and hit[idx[-1]]["t"][0] is not None:
                ends[li] = hit[idx[-1]]["t"][0]
        g8 = np.concatenate([beats, beats[:-1] + P / 2])
        on_grid = sum(np.min(np.abs(g8 - t)) <= 0.06 for t in ends.values())
        on_one = sum(np.min(np.abs(downs - t)) <= 0.06 for t in ends.values())
        n = max(1, len(ends))
        # sung words per minute and in the busiest 30 s (the course gate's ceilings: 95 and 120)
        wt = sorted(x["t"][0] for x in ww if x["t"][0] is not None)
        span = (wt[-1] - wt[0]) if len(wt) > 1 else 1
        wpm = len(wt) / span * 60
        wpm30 = max((sum(1 for x in wt if a <= x < a + 30) * 2 for a in wt), default=0)
        info = sf.info(str(w))
        rows.append(dict(seed=w.stem, recall=round(recall, 3), wer=round(m.wer, 3), dur=round(info.duration, 1), bpm=round(bpm, 1),
                         meter3=round(meter, 3), lines_heard=f"{len(ends)}/{len(LINES)}", rhyme_grid=round(on_grid / n, 2),
                         rhyme_one=round(on_one / n, 2), wpm=round(wpm), wpm30=wpm30, first_word=round(wt[0], 1) if wt else None,
                         last_word=round(wt[-1], 1) if wt else None, bad_lines=bad))
        r = rows[-1]
        print(f"{r['seed']:>6} recall {r['recall']:.3f} wer {r['wer']:.3f} bpm {r['bpm']:5.1f} meter3 {r['meter3']:+.3f} "
              f"lines {r['lines_heard']} grid {r['rhyme_grid']:.2f} one {r['rhyme_one']:.2f} wpm {r['wpm']} /30s {r['wpm30']} "
              f"words {r['first_word']}-{r['last_word']} bad {len(bad)}", flush=True)
    rows.sort(key=lambda r: -r["recall"])
    out = Path(paths[0]).parent / "scores.json"
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
