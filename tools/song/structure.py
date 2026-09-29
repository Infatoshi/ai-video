# /// script
# dependencies = ["librosa", "numpy", "matplotlib", "soundfile"]
# ///
"""Look at a take's structure (what the ear would check first): log-mel spectrogram, drums/bass/vocal
energy, and where Whisper heard each lyric line, so a missing drop, a mumbled verse or a cut-off ending
shows at a glance.

  uv run song/structure.py <take.wav> <take.words.json> song/lyrics.sing.txt <out.png>
"""
import json, re, sys
from difflib import SequenceMatcher

import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())


def main(wav, words_json, lyrics_txt, out):
    y, sr = librosa.load(wav, sr=22050, mono=True)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, hop_length=512), ref=np.max)
    lines, sect = [], []
    cur = ""
    for l in open(lyrics_txt).read().splitlines():
        if l.startswith("["):
            cur = l.strip("[]").split(" -")[0]
        elif l.strip():
            lines.append(l.strip()); sect.append(cur)
    words = json.load(open(words_json))
    toks = [(li, norm(w)) for li, l in enumerate(lines) for w in re.split(r"[\s-]+", l) if norm(w)]
    sm = SequenceMatcher(a=[t for _, t in toks], b=[norm(x["w"]) for x in words], autojunk=False)
    hit = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                hit[i1 + k] = words[j1 + k]
    fig, ax = plt.subplots(2, 1, figsize=(20, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    dur = len(y) / sr
    ax[0].imshow(S, origin="lower", aspect="auto", extent=[0, dur, 0, 96], cmap="magma")
    for li, l in enumerate(lines):
        ts = [hit[i]["t"][0] for i, (x, _) in enumerate(toks) if x == li and i in hit and hit[i]["t"][0] is not None]
        n = sum(1 for x, _ in toks if x == li)
        if ts:
            ax[0].axvline(min(ts), color="w", lw=0.6, alpha=0.7)
            ax[0].text(min(ts), 90 - (li % 6) * 7, f"{li}:{sect[li][:6]} {len(ts)}/{n}", color="w", fontsize=7)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    t = np.arange(len(rms)) * 512 / sr
    ax[1].plot(t, 20 * np.log10(rms + 1e-6), lw=0.8)
    ax[1].set_ylabel("dB")
    ax[1].set_xlabel("s")
    fig.suptitle(wav)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    heard = sum(1 for li in range(len(lines)) if any(x == li and i in hit for i, (x, _) in enumerate(toks)))
    print(f"{wav}: {heard}/{len(lines)} lines heard, {len(hit)}/{len(toks)} words matched in order")


if __name__ == "__main__":
    main(*sys.argv[1:5])
