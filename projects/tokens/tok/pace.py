"""Projected course-gate pace for song takes, from Whisper's word times (rhythm.py's <seed>.words.json): sung
words per minute over the vocal span and in the busiest 30 s (tools/gate/gate.py's definitions), plus the
longest vocal gaps (where the instrumental bars are).

  python3 tok/pace.py out/takes/<run>/*.words.json
"""
import json, re, sys
from difflib import SequenceMatcher
from pathlib import Path

norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())


def lyric_times(path, lyrics=Path(__file__).resolve().parent.parent / "song/lyrics.sing.txt"):
    """Start time of every lyric word (the aligner's word list: lines split on whitespace), from Whisper's
    words by sequence alignment; words Whisper missed are interpolated. This is what data/lyrics.json will
    count, unlike Whisper's own word list (hallucinated repeats, split words)."""
    ws = [w for w in json.load(open(path)) if w["t"][0] is not None]
    lines = [l.strip() for l in open(lyrics).read().splitlines() if l.strip() and not l.startswith("[")]
    toks = [norm(w) for l in lines for w in l.split() if norm(w)]
    hyp = [norm(w["w"]) for w in ws]
    sm = SequenceMatcher(a=toks, b=hyp, autojunk=False)
    t = [None] * len(toks)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                t[i1 + k] = ws[j1 + k]["t"][0]
    hit = sum(x is not None for x in t)
    if t[0] == 0.0:
        t[0] = None
    known = [i for i, x in enumerate(t) if x is not None]
    for i in range(len(t)):
        if t[i] is None:
            lo = max((k for k in known if k < i), default=None); hi = min((k for k in known if k > i), default=None)
            if lo is None: t[i] = t[hi] - 0.3 * (hi - i)
            elif hi is None: t[i] = t[lo] + 0.3 * (i - lo)
            else: t[i] = t[lo] + (t[hi] - t[lo]) * (i - lo) / (hi - lo)
    return sorted(t), hit / len(toks)


def pace_lyrics(path):
    st, cov = lyric_times(path)
    span = st[-1] + 0.4 - st[0]
    return len(st), span, len(st) / span * 60, max(sum(1 for x in st if a <= x < a + 30) * 2 for a in st), cov


def pace(path):
    ws = [w for w in json.load(open(path)) if w["t"][0] is not None]
    st = sorted(w["t"][0] for w in ws)
    # Whisper parks the first word at 0.0 over an instrumental intro: use its end minus 0.4 s instead
    if ws and ws[0]["t"][0] == 0.0 and ws[0]["t"][1]:
        st[0] = max(0.0, ws[0]["t"][1] - 0.4)
    ends = [w["t"][1] or w["t"][0] for w in ws]
    span = max(ends) - st[0]
    wpm = len(st) / span * 60
    wpm30 = max(sum(1 for x in st if a <= x < a + 30) * 2 for a in st)
    gaps = sorted(((b - a, a) for a, b in zip(st, st[1:])), reverse=True)[:6]
    return len(st), span, wpm, wpm30, gaps


if __name__ == "__main__":
    for p in sys.argv[1:]:
        n, span, wpm, wpm30, cov = pace_lyrics(p)
        print(f"{p.split('/')[-1]:>20} lyric words {n} over {span:5.1f} s: {wpm:5.1f} wpm, busiest 30 s {wpm30:4d} "
              f"(Whisper matched {cov:.0%} of them)")
        n, span, wpm, wpm30, gaps = pace(p)
        print(f"{p.split('/')[-1]:>20} {n:4d} words over {span:5.1f} s: {wpm:5.1f} wpm, busiest 30 s {wpm30:4d}; "
              f"gaps {' '.join(f'{g:.1f}@{a:.0f}' for g, a in gaps)}")
