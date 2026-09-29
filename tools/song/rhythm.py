"""Does the singing sit on the beat? Rank takes by where the words land (run on the GPU host, score/ venv).

  uv run --project score python rhythm.py out/<run> [seed ...]

Beats: a constant-tempo grid fitted to the mix's onset strength; downbeat phase from beat-this (CPJKU); word times from Whisper large-v3-turbo
(word timestamps) on the Demucs vocal stem that score.py cached (<seed>.vocals.wav). Lyric lines
are matched to Whisper words with a sequence alignment; each matched word start is snapped to the
nearest vocal onset (backtracked, ±150 ms) because Whisper's word times are ±100-200 ms. Per take:
  rhyme_grid     share of line-final words whose onset is within 60 ms of the 8th-note grid
  rhyme_strong   share of line-final words on beat 1 or 3 of their bar (±60 ms)
  couplet_par    mean |difference| (beats) between the rhyme onsets of each AABB couplet, measured
                 inside their own 2-bar phrases; 0 = both rhymes land on the same spot
  start_spread   spread (beats) of line starts within the 2-bar phrase; 0 = every line starts alike
  stretched      acronyms (CPU, GPU) sung longer than 1.2 beats
Writes <dir>/rhythm.json.
"""
import json, os, re, sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(HERE / ".hf")
os.environ["HF_HUB_CACHE"] = str(HERE / ".hf" / "hub")


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def lyric_lines(txt):
    return [l.strip() for l in txt.splitlines() if l.strip() and not l.startswith("[")]


def beats_of(wav, bpm0=None):
    """Constant-tempo grid (these songs never change tempo): tempo and phase by maximizing the mix's
    onset strength on the grid (analysis/analyze.py's method; within 5% of the requested BPM when the
    take's json has one, else 88-150, so a 174 BPM take can't lock to its half-time), then the bar phase is the
    beat index (mod 4) that most of beat-this's downbeats fall on."""
    import librosa
    from beat_this.inference import File2Beats
    y, sr = librosa.load(str(wav), sr=22050)
    hop = 32
    o = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, lag=1, max_size=1)
    o = o / (np.percentile(o, 99) + 1e-9)
    fps, dur = sr / hop, len(y) / sr

    def score(P, off):
        idx = np.round((off + P * np.arange(int(dur / P) + 1)) * fps).astype(int)
        idx = idx[(idx > 2) & (idx < len(o) - 2)]
        return np.maximum.reduce([o[idx - 1], o[idx], o[idx + 1]]).mean()

    best = (0.0, 132.0, 0.0)
    lo, hi = (bpm0 * 0.95, bpm0 * 1.05) if bpm0 else (88.0, 150.0)
    for bpm in np.arange(lo, hi, 0.05):
        P = 60 / bpm
        for off in np.arange(0, P, 0.005):
            v = score(P, off)
            if v > best[0]:
                best = (v, bpm, off)
    _, bpm, off = best
    for b2 in np.arange(bpm - 0.05, bpm + 0.05, 0.002):
        for o2 in np.arange(off - 0.01, off + 0.01, 0.001):
            v = score(60 / b2, o2)
            if v > best[0]:
                best = (v, b2, o2)
    _, bpm, off = best
    P = 60 / bpm
    grid = np.arange(off % P, dur, P)
    f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    _, dn = (np.asarray(x) for x in f2b(str(wav)))
    k = np.round((dn - grid[0]) / P).astype(int) % 4
    bar = int(np.bincount(k, minlength=4).argmax())
    return grid, grid[bar::4]


_asr = None


def words_of(voc):
    global _asr
    j = voc.with_name(voc.name.replace(".vocals.wav", ".words.json"))
    if not j.exists():
        import librosa
        from transformers import pipeline
        if _asr is None:
            _asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                            dtype=torch.float16, device="cuda:0")
        y, _ = librosa.load(str(voc), sr=16000)
        r = _asr(y, return_timestamps="word", chunk_length_s=30, batch_size=1,  # word DTW keeps attentions: 8 OOMs a 24 GB card
                 generate_kwargs={"language": "en", "task": "transcribe"})
        j.write_text(json.dumps([{"w": c["text"], "t": list(c["timestamp"])} for c in r["chunks"]]))
        torch.cuda.empty_cache()
    return json.loads(j.read_text())


def onsets_of(voc):
    import librosa
    y, sr = librosa.load(str(voc), sr=22050)
    return librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, backtrack=True, units="time")


def snap(t, ons, win=0.15):
    k = np.searchsorted(ons, t)
    cand = [ons[i] for i in (k - 1, k) if 0 <= i < len(ons) and abs(ons[i] - t) <= win]
    return min(cand, key=lambda o: abs(o - t)) if cand else t


def main(d, seeds):
    d = Path(d)
    wavs = sorted(p for p in d.glob("*.wav") if p.stem.isdigit() and (not seeds or p.stem in seeds))
    rows = []
    for w in wavs:
        meta = json.loads(w.with_suffix(".json").read_text())
        lines = lyric_lines(meta["lyrics"])
        toks = [(li, norm(t)) for li, l in enumerate(lines) for t in l.split() if norm(t)]
        voc = w.with_suffix(".vocals.wav")
        ww = [x for x in words_of(voc) if norm(x["w"])]
        sm = SequenceMatcher(a=[t for _, t in toks], b=[norm(x["w"]) for x in ww], autojunk=False)
        hit = {}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
                for k in range(i2 - i1):
                    hit[i1 + k] = ww[j1 + k]
        ons = onsets_of(voc)
        meta = w.with_suffix(".json")
        beats, downs = beats_of(w, json.loads(meta.read_text()).get("bpm") if meta.exists() else None)
        P = float(np.median(np.diff(beats)))
        bar0 = lambda t: downs[max(0, np.searchsorted(downs, t, side="right") - 1)]
        pos_bar = lambda t: (t - bar0(t)) / P                      # beats into the bar
        grid8 = lambda t: np.min(np.abs(np.concatenate([beats, beats[:-1] + np.diff(beats) / 2]) - t))
        ends, starts = {}, {}
        for li in range(len(lines)):
            idx = [i for i, (l, _) in enumerate(toks) if l == li]
            if hit.get(idx[0]) and hit[idx[0]]["t"][0] is not None:
                starts[li] = snap(hit[idx[0]]["t"][0], ons)
            if hit.get(idx[-1]) and hit[idx[-1]]["t"][0] is not None:
                ends[li] = snap(hit[idx[-1]]["t"][0], ons)
        n = len(ends)
        on_grid = sum(grid8(t) <= 0.06 for t in ends.values())
        strong = sum(np.min(np.abs(beats - t)) <= 0.06 and round(pos_bar(t)) % 4 in (0, 2) for t in ends.values())
        # couplets: lines (0,1), (2,3), ... inside each 4-line block; the 2-line pre-choruses are couplets too
        par = []
        for a in range(0, len(lines) - 1, 2):
            if a in ends and a + 1 in ends and a in starts and a + 1 in starts:
                pa = (ends[a] - starts[a]) / P
                pb = (ends[a + 1] - starts[a + 1]) / P
                par.append(abs(pa - pb))
        sph = np.array([pos_bar(t) % 4 for t in starts.values()])
        spread = float(np.std(np.unwrap(sph * np.pi / 2) * 2 / np.pi)) if len(sph) else 9.0
        stretched = []
        for i, (li, tk) in enumerate(toks):
            x, nx = hit.get(i), hit.get(i + 1)
            if not (x and tk in ("gpu", "cpu") and x["t"][0] is not None):
                continue
            t0 = snap(x["t"][0], ons)
            t1 = min(v for v in (x["t"][1], snap(nx["t"][0], ons) if nx and nx["t"][0] else None) if v)
            if t1 - t0 > 1.2 * P:  # bounded by the next sung word, so intro/whisper drift can't inflate it
                stretched.append(f"{tk}@{t0:.1f}:{(t1 - t0) / P:.1f}b")
        rows.append(dict(seed=w.stem, bpm=round(60 / P, 1), matched=f"{n}/{len(lines)}",
                         rhyme_grid=round(float(on_grid) / max(1, n), 2), rhyme_strong=round(float(strong) / max(1, n), 2),
                         couplet_par=round(float(np.mean(par)) if par else 9.0, 2),
                         start_spread=round(spread, 2), stretched=stretched))
        print(rows[-1], flush=True)
    rows.sort(key=lambda r: (r["couplet_par"], -r["rhyme_grid"]))
    (d / "rhythm.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
