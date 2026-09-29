"""Everything the video shows about Llama 3.1 8B Instruct's input embedding table (CPU only, numpy):

  uv run --with numpy python llm/embed.py out/data_src data/emb.json

<src> holds extract.py's output (emb_bf16.npy: the table's exact bf16 bits, vocab.json, emb_info.json with the
sha256 of the bytes as read from the safetensors shard on the GPU host). Records:
  rows        the full 4,096 numbers of the hero word " rain" (and " three", " king")
  table       the 48 real rows around " rain" (int8 of their max |value|, for the table wall; ink density only)
  neighbours  the nearest tokens of the whole 128,256-token vocabulary by cosine similarity (raw: every token,
              spellings included), for rain, three, king
  baseline    mean cosine of random pairs of common words (what "unrelated" looks like here)
  analogies   X - man + woman for boy, father, king (3CosAdd on unit vectors, the inputs excluded), top 5 with
              cosines and the rank of the expected word; plus the same for 23 male/female pairs (how often it lands)
  space       a 3D PCA fitted to the featured words (every labelled point), the share of their variance the 3
              axes keep, those points' coordinates, the analogy landing points, and 8,000 other common words
              projected the same way (the haze)
Tokens keep their leading space (" rain" is the word as it appears mid-sentence; "Rain" without it is another row).
"""
import json, re, sys
from pathlib import Path

import numpy as np

HERO = [" rain", " three", " king"]
# every labelled point in the flight (the PCA is fitted to exactly these)
FEATURED = [
    # rain and its raw neighbours (spellings included, as the table has them)
    " rain", " Rain", "Rain", " rains", " snow", " raining", " rainfall", "rain", " wind", " sun", " weather", " storm",
    # numbers
    " one", " two", " three", " four", " five", " six", " seven", " Three",
    # people, and king's spellings (the analogy's answers)
    " man", " woman", " boy", " girl", " father", " mother", " king", " queen", " King", " KING", "King", " kings",
    # scenery islands
    " red", " blue", " green", " yellow", " ocean", " sea", " river", " lake", " cat", " dog", " horse",
    " coffee", " tea", " wine",
]
PAIRS = [("king", "queen"), ("boy", "girl"), ("father", "mother"), ("brother", "sister"), ("he", "she"), ("his", "her"),
         ("husband", "wife"), ("son", "daughter"), ("uncle", "aunt"), ("prince", "princess"), ("actor", "actress"),
         ("male", "female"), ("gentleman", "lady"), ("boys", "girls"), ("men", "women"), ("dad", "mom"),
         ("grandfather", "grandmother"), ("nephew", "niece"), ("lord", "lady"), ("groom", "bride"), ("monk", "nun"),
         ("waiter", "waitress"), ("hero", "heroine")]


def main(src, out):
    src = Path(src)
    bits = np.load(src / "emb_bf16.npy")
    E = (bits.astype(np.uint32) << 16).view(np.float32)  # bf16 -> fp32, exact
    V = json.loads((src / "vocab.json").read_text())
    info = json.loads((src / "emb_info.json").read_text())
    import hashlib
    assert hashlib.sha256(bits.tobytes()).hexdigest() == info["sha256_bytes"], "copy differs from the shard's bytes"
    ids = {}
    for i, s in enumerate(V):
        ids.setdefault(s, i)
    tid = lambda s: ids[s]
    norms = np.linalg.norm(E, axis=1)
    N = E / norms[:, None]
    tok = lambda j: {"id": int(j), "t": V[j]}

    def nearest(s, k):
        i = tid(s)
        c = N @ N[i]
        o = [j for j in np.argsort(-c)[:k + 1] if j != i][:k]
        return [{**tok(j), "cos": round(float(c[j]), 4)} for j in o]

    common = [i for i, s in enumerate(V[:50000]) if s.startswith(" ") and s[1:].isalpha() and s[1:].isascii()
              and s[1:].islower() and len(s) > 3]
    rng = np.random.default_rng(0)
    a, b = rng.choice(common, 20000), rng.choice(common, 20000)
    keep = a != b
    rc = np.einsum("ij,ij->i", N[a[keep]], N[b[keep]])

    def analogy(x, k=5):
        v = N[tid(" " + x)] - N[tid(" man")] + N[tid(" woman")]
        c = N @ (v / np.linalg.norm(v))
        ex = {tid(" " + x), tid(" man"), tid(" woman")}
        order = [j for j in np.argsort(-c)[:k + 10] if j not in ex]
        return v, c, ex, [{**tok(j), "cos": round(float(c[j]), 4)} for j in order[:k]]

    def rank_excl(c, ex, j):
        return int(sum(1 for m in np.nonzero(c > c[j])[0] if m not in ex)) + 1

    analogies = {}
    for x, y in [("boy", "girl"), ("father", "mother"), ("king", "queen")]:
        v, c, ex, top = analogy(x)
        analogies[x] = {"expect": y, "top5": top, "rank_expect": rank_excl(c, ex, tid(" " + y)),
                        "cos_expect": round(float(c[tid(" " + y)]), 4)}
    pairs = []
    for x, y in PAIRS:
        v, c, ex, top = analogy(x, 1)
        pairs.append({"a": x, "b": y, "rank": rank_excl(c, ex, tid(" " + y)), "top1": top[0]["t"]})
    king_own = N @ N[tid(" king")]
    queen_rank_before = int((king_own > king_own[tid(" queen")]).sum())  # minus king itself, +1 for the rank

    # the 3D space: PCA fitted to the featured words
    fi = [tid(s) for s in FEATURED]
    X = N[fi]
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    P = Vt[:3]
    proj = lambda M: (M - mu) @ P.T
    Y = proj(X)
    scale = 1.0 / np.abs(Y).max()  # display units: the featured points fill [-1, 1]
    haze = common[:8000]
    Yh = proj(N[haze]) * scale
    land = {x: [round(float(q), 4) for q in proj(analogy(x)[0][None])[0] * scale] for x in analogies}
    arrow = (proj(N[[tid(" woman")]]) - proj(N[[tid(" man")]]))[0] * scale

    # the table wall: the 48 real rows around " rain" (ids 11,398..11,445), each value as int8 of the rows' max |value|
    r0 = tid(" rain") - 24
    T = E[r0:r0 + 48]
    tmax = float(np.abs(T).max())
    import base64
    table = {"first_id": r0, "rows": 48, "t": V[r0:r0 + 48], "max_abs": round(tmax, 6),
             "i8": base64.b64encode(np.round(T / tmax * 127).astype(np.int8).tobytes()).decode()}
    out_d = {
        "source": {**info, "note": "input embedding table (model.embed_tokens.weight), bf16 bits read from the shard"},
        "vocab": len(V), "dim": int(E.shape[1]),
        "rows": {s: {"id": tid(s), "values": [float(f"{x:.6g}") for x in E[tid(s)]], "norm": round(float(norms[tid(s)]), 4)}
                 for s in HERO},
        "neighbours": {s: nearest(s, 12) for s in HERO},
        "table": table,
        "baseline": {"pairs": int(keep.sum()), "mean_cos": round(float(rc.mean()), 4), "p99_cos": round(float(np.percentile(rc, 99)), 4),
                     "of": f"random pairs of the {len(common)} lowercase whole-word tokens among the first 50,000 ids"},
        "analogies": analogies,
        "pairs": pairs, "pairs_rank1": sum(p["rank"] == 1 for p in pairs),
        "queen_rank_in_king_neighbours": queen_rank_before,
        "space": {
            "method": f"PCA fitted to the {len(FEATURED)} featured tokens' unit vectors (4,096 -> 3)",
            "var_kept": [round(float(x), 4) for x in var[:3]], "var_kept_total": round(float(var[:3].sum()), 4),
            "featured": [{**tok(i), "p": [round(float(q), 4) for q in y]} for i, y in zip(fi, Y * scale)],
            "land": land, "arrow": [round(float(q), 4) for q in arrow],
            "haze": {"n": len(haze), "t": [V[j] for j in haze],
                     "p": [round(float(q), 3) for q in Yh.reshape(-1)]},
        },
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_d, ensure_ascii=False, separators=(",", ":")))
    s = {k: v for k, v in out_d.items() if k not in ("rows", "space", "table")}
    print(json.dumps(s, ensure_ascii=False, indent=1)[:6000])
    print("var kept", out_d["space"]["var_kept"], out_d["space"]["var_kept_total"])
    print("rain row: first 8", out_d["rows"][" rain"]["values"][:8], "min", E[tid(" rain")].min(), "max", E[tid(" rain")].max())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
