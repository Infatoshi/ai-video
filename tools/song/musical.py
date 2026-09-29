# /// script
# dependencies = ["librosa", "numpy", "soundfile", "matplotlib"]
# ///
"""Musical checks on shortlisted takes (what the lyric/aesthetics scores miss).

  uv run song/musical.py song/takes/<take>.wav ...

Per take: tempo and its stability, key, loudness, how the song ends (fade vs cut), vocal pitch
range and intonation (cents off the nearest semitone), and hook consistency: the pitch contour of
each sung "I'm chasing the roofline" (located from the Whisper chunks) compared pairwise with DTW,
in semitones. Writes <take>.png (log-mel spectrogram with the hook windows marked).
"""
import json, subprocess, sys
from pathlib import Path

import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = "C C# D D# E F F# G G# A A# B".split()
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_of(y, sr):
    c = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr).mean(1)
    sc = [(np.corrcoef(np.roll(MAJ, k), c)[0, 1], f"{NAMES[k]} major") for k in range(12)]
    sc += [(np.corrcoef(np.roll(MIN, k), c)[0, 1], f"{NAMES[k]} minor") for k in range(12)]
    return max(sc)[1]


def lufs(path):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    for line in out.splitlines()[::-1]:
        if line.strip().startswith("I:"):
            return float(line.split()[1])


def hooks(wj):
    """(start, end) of chunks containing the hook, from Whisper's chunk timestamps."""
    ch = json.loads(wj.read_text())["chunks"]
    out = []
    for c in ch:
        t = c["text"].lower().replace("-", " ")
        if "chasin" in t and ("roof" in t or "root" in t or "route" in t):
            s, e = c["t"]
            if e is None:
                e = s + 2.5
            out.append((s, min(e, s + 3.0)))
    return out


def contour(f0, t, s, e):
    m = (t >= s) & (t <= e) & np.isfinite(f0)
    x = librosa.hz_to_midi(f0[m])
    return x - np.median(x) if len(x) > 10 else None


def main(paths):
    for p in map(Path, paths):
        y, sr = librosa.load(p, sr=22050, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        ibi = np.diff(beats)
        v, _ = librosa.load(p.with_suffix(".vocals.wav"), sr=22050, mono=True)
        f0, vflag, _ = librosa.pyin(v, fmin=80, fmax=1000, sr=sr, frame_length=2048, hop_length=256)
        t = librosa.times_like(f0, sr=sr, hop_length=256)
        voiced = f0[np.isfinite(f0)]
        midi = librosa.hz_to_midi(voiced)
        cents = np.abs(100 * (midi - np.round(midi)))
        rms = librosa.feature.rms(y=y)[0]
        rt = librosa.times_like(rms, sr=sr)
        tail = rms[rt > rt[-1] - 1.0].mean() / (np.percentile(rms, 95) + 1e-9)
        hk = hooks(p.with_suffix(".whisper.json"))
        cons = []
        cs = [contour(f0, t, s, e) for s, e in hk]
        cs = [c for c in cs if c is not None]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                D, wp = librosa.sequence.dtw(cs[i][None], cs[j][None], metric="euclidean")
                cons.append(D[-1, -1] / len(wp))
        print(f"{p.stem:>16}  tempo {float(np.atleast_1d(tempo)[0]):6.1f}  beat-jitter {np.std(ibi) * 1000:5.1f} ms  "
              f"key {key_of(y, sr):9}  {lufs(p):6.1f} LUFS  end-level {tail:.2f}  "
              f"vocal {librosa.midi_to_note(np.percentile(midi, 5))}-{librosa.midi_to_note(np.percentile(midi, 95))}  "
              f"off-pitch {np.median(cents):4.1f} c  hooks {len(hk)}  hook-dtw {np.mean(cons) if cons else float('nan'):.2f} st")
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96), ref=np.max)
        fig, ax = plt.subplots(figsize=(18, 3), dpi=80)
        ax.imshow(S, origin="lower", aspect="auto", extent=[0, len(y) / sr, 0, 96], cmap="magma")
        for s, e in hk:
            ax.axvspan(s, e, color="cyan", alpha=0.25)
        ax.set_title(p.stem)
        fig.tight_layout()
        fig.savefig(p.with_suffix(".png"))
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
