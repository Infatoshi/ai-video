"""Byte-pair encoding, from scratch, trained on this song's own lyrics (runs anywhere, stdlib only).

  python3 tok/bpe.py song/lyrics.sing.txt data/bpe.json

The corpus is every sung line in order (the chorus as often as it is sung), lowercased, split into words
(letters and apostrophes; hyphens and punctuation split). Training: every word starts as single letters;
count every pair of neighbours inside a word, over the whole song; glue the most frequent pair into one
chunk everywhere; repeat until no pair occurs twice. Ties go to the pair seen first in the song. Every merge
is recorded in order with its count, so the video can replay them one at a time.
"""
import json, re, sys
from collections import Counter


def corpus(path):
    lines = [l.strip() for l in open(path) if l.strip() and not l.startswith("[")]
    lines = [re.sub(r"[()]", "", l) for l in lines]
    words = [w for l in lines for w in re.findall(r"[a-z']+", l.lower().replace("-", " ")) if w.strip("'")]
    return lines, words


def train(words):
    seqs = {w: list(w) for w in dict.fromkeys(words)}   # unique words in first-seen order
    freq = Counter(words)
    merges = []
    while True:
        pairs, first = Counter(), {}
        for w, s in seqs.items():
            for a, b in zip(s, s[1:]):
                pairs[a, b] += freq[w]
                first.setdefault((a, b), len(first))
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], -first[p]))
        if pairs[best] < 2:
            break
        a, b = best
        for w, s in seqs.items():
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(s[i]); i += 1
            seqs[w] = out
        merges.append({"k": len(merges) + 1, "a": a, "b": b, "ab": a + b, "count": pairs[best]})
    return merges, seqs


def apply(word, merges, k):
    s = list(word)
    for m in merges[:k]:
        out, i = [], 0
        while i < len(s):
            if i + 1 < len(s) and s[i] == m["a"] and s[i + 1] == m["b"]:
                out.append(m["ab"]); i += 2
            else:
                out.append(s[i]); i += 1
        s = out
    return s


def main(src, dst):
    lines, words = corpus(src)
    merges, final = train(words)
    letters = sorted(set("".join(words)))
    freq = Counter(words)
    # the step at which each word first became a single chunk
    whole = {}
    for w in freq:
        for k in range(len(merges) + 1):
            if len(apply(w, merges, k)) == 1:
                whole[w] = k
                break
    d = {
        "corpus": {"lines": len(lines), "words": len(words), "unique_words": len(freq), "letters": len(letters),
                   "chars": sum(len(w) for w in words), "alphabet": "".join(letters)},
        "merges": merges,
        "vocab_final": len(letters) + len(merges),
        "chunks_final": sum(len(final[w]) * n for w, n in freq.items()),
        "whole_at": dict(sorted(whole.items(), key=lambda x: x[1])),
        "top_words": freq.most_common(20),
        "final": {w: final[w] for w in freq},
    }
    json.dump(d, open(dst, "w"), indent=1)
    print(f"{len(words)} words ({len(freq)} unique, {d['corpus']['chars']} letters, alphabet of {len(letters)}), "
          f"{len(merges)} merges -> {d['chunks_final']} chunks")
    for m in merges[:30]:
        print(f"  {m['k']:3} {m['a']!r:>8} + {m['b']!r:<8} -> {m['ab']!r:10} x{m['count']}")
    print("whole at:", list(d["whole_at"].items())[:40])
    print("strawberry:", [(k, apply("strawberry", merges, k)) for k in (0, 10, 25, 50, 100, len(merges))])


if __name__ == "__main__":
    main(*sys.argv[1:3])
