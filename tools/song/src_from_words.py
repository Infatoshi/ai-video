"""Write the aligner's lyric source (lyrics/<song>.src.js: [start, end, text] per sung line) for a take,
from its Whisper word times (<seed>.words.json, written by rhythm.py) and the lyrics it was asked to sing.

  python3 song/src_from_words.py <seed>.words.json song/lyrics.sing.txt lyrics/token.src.js "Next Token"

Lines are matched to the words by a sequence alignment; a line Whisper missed gets times interpolated
from its neighbours. The times only seed the aligner's per-line windows (analysis/align.py refines them).
"""
import json, re, sys
from difflib import SequenceMatcher

norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())


def main(words_json, lyrics_txt, out_js, title):
    words = json.load(open(words_json))
    lines = [l.strip() for l in open(lyrics_txt).read().splitlines() if l.strip() and not l.startswith("[")]
    toks = [(li, norm(w)) for li, l in enumerate(lines) for w in re.split(r"[\s-]+", l) if norm(w)]
    hyp = [norm(x["w"]) for x in words]
    sm = SequenceMatcher(a=[t for _, t in toks], b=hyp, autojunk=False)
    hit = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                hit[i1 + k] = words[j1 + k]
    span = []
    for li in range(len(lines)):
        ts = [hit[i]["t"] for i, (l, _) in enumerate(toks) if l == li and i in hit and hit[i]["t"][0] is not None]
        span.append((min(t[0] for t in ts), max(t[1] or t[0] for t in ts)) if len(ts) >= 2 else None)
    # interpolate the lines Whisper missed
    for li, s in enumerate(span):
        if s is None:
            prev = next((span[k] for k in range(li - 1, -1, -1) if span[k]), None)
            nxt = next((span[k] for k in range(li + 1, len(span)) if span[k]), None)
            a = prev[1] if prev else 0.0
            b = nxt[0] if nxt else a + 3.0
            span[li] = (a, b)
    rows = ",\n".join(f"  [{s:.1f}, {e:.1f}, {json.dumps(t)}]" for (s, e), t in zip(span, lines))
    missed = sum(1 for li in range(len(lines)) if not any(l == li and i in hit for i, (l, _) in enumerate(toks)))
    open(out_js, "w").write(f"// \"{title}\": sung lines with approximate start/end seconds from Whisper words.\n"
                            f"// Word timings come from analysis/align.py -> data/lyrics.json.\nconst LY = [\n{rows}\n];\n")
    print(f"{len(lines)} lines, {missed} interpolated")


if __name__ == "__main__":
    main(*sys.argv[1:5])
