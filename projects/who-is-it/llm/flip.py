# /// script
# dependencies = ["numpy"]
# ///
"""Which attention heads flip? Reads out/llm/attn.npz + attn_meta.json (written by llm/attn.py on the GPU host)
and, for every pair, every asking position and every layer x head, compares the weight on the two
candidate answers in sentence A (answer: ref[0]) and sentence B (answer: ref[1]).

  uv run llm/flip.py out/llm [--json data/attn.json]

Per head: d = w(ref0) - w(ref1) in each sentence. It flips the right way when d_A > 0 and d_B < 0;
the flip margin is min(d_A, -d_B) (both sentences have to agree); a clean flip has a margin of at
least CLEAN. The wrong way: d_A < 0 and d_B > 0. Heads that put under QUIET on both words in both
sentences are "looking elsewhere". Prints the counts, where the flipping heads are (layers) and the
top heads; --json writes what the video shows.
"""
import json, sys
from pathlib import Path

import numpy as np

CLEAN = 0.05   # a clean flip moves at least 5 points of weight each way
QUIET = 0.02   # a head that gives both words under 2% in both sentences is looking elsewhere


def find(tokens, word, after=0):
    for i, t in enumerate(tokens):
        if i >= after and t.strip().lower().strip(".,") == word:
            return i
    raise KeyError(f"{word!r} not in {tokens}")


def analyse(att, meta):
    out = []
    for pr in meta["pairs"]:
        pid = pr["id"]
        A, B = att[pid + "A"].astype(np.float32), att[pid + "B"].astype(np.float32)
        tA, tB = pr["runs"]["A"]["tokens"], pr["runs"]["B"]["tokens"]
        who = pr["who"] if isinstance(pr["who"], list) else [pr["who"], pr["who"]]
        # candidate positions (same index in both sentences when the sentences line up)
        refA = [find(tA, w) if w in [t.strip().lower() for t in tA] else find(tA, w + "s") for w in pr["ref"]]
        refB = [find(tB, w) if w in [t.strip().lower() for t in tB] else find(tB, w + "s") for w in pr["ref"]]
        qs = {"who": (find(tA, who[0], refA[0] + 1), find(tB, who[1], refB[0] + 1))}
        cA = [i for i, t in enumerate(tA) if t.strip().lower().strip(".,") == pr["change"][0]]
        cB = [i for i, t in enumerate(tB) if t.strip().lower().strip(".,") == pr["change"][1]]
        if cA and cB and cA[-1] > qs["who"][0]:
            qs["change"] = (cA[-1], cB[-1])
        qs["end"] = (len(tA) - 1, len(tB) - 1)
        res = {"id": pid, "note": pr["note"], "A": tA, "B": tB, "ref": pr["ref"], "refA": refA, "refB": refB,
               "probe": {s: pr["runs"][s]["probe"]["words"] for s in "AB"},
               "chat": {s: pr["runs"][s]["chat"]["answer"] for s in "AB"}, "at": {}}
        L, H = A.shape[:2]
        for name, (qa, qb) in qs.items():
            wa = A[:, :, qa, :]  # L x H x T
            wb = B[:, :, qb, :]
            a0, a1 = wa[..., refA[0]], wa[..., refA[1]]
            b0, b1 = wb[..., refB[0]], wb[..., refB[1]]
            dA, dB = a0 - a1, b0 - b1
            right = (dA > 0) & (dB < 0)
            wrong = (dA < 0) & (dB > 0)
            margin = np.minimum(dA, -dB)
            quiet = (np.maximum.reduce([a0, a1, b0, b1]) < QUIET)
            clean = right & (margin >= CLEAN)
            clean_wrong = wrong & (np.minimum(-dA, dB) >= CLEAN)
            same_prefix = tA[:qa + 1] == tB[:qb + 1]
            order = np.argsort(-margin, axis=None)
            top = []
            for k in order[:12]:
                l, h = divmod(int(k), H)
                top.append({"layer": l, "head": h, "margin": round(float(margin[l, h]), 4),
                            "A": [round(float(a0[l, h]), 4), round(float(a1[l, h]), 4)],
                            "B": [round(float(b0[l, h]), 4), round(float(b1[l, h]), 4)],
                            "bosA": round(float(wa[l, h, 0]), 4), "bosB": round(float(wb[l, h, 0]), 4),
                            "selfA": round(float(wa[l, h, qa]), 4), "selfB": round(float(wb[l, h, qb]), 4)})
            maxdiff = float(np.abs(wa[..., :qa + 1] - wb[..., :qb + 1]).max()) if same_prefix else None
            # per head: 2 clean flip, 1 leans right, -1 leans wrong, 0 no switch (what the video's grid colours)
            kind = np.where(clean, 2, np.where(right, 1, np.where(wrong, -1, 0)))
            res["at"][name] = {
                "q": [qa, qb], "tok": [tA[qa], tB[qb]], "same_prefix": same_prefix, "max_abs_diff": maxdiff,
                "heads": L * H, "right": int(right.sum()), "clean": int(clean.sum()),
                "wrong": int(wrong.sum()), "clean_wrong": int(clean_wrong.sum()),
                "no_flip": int((~right & ~wrong).sum()), "quiet": int(quiet.sum()),
                "clean_by_layer": clean.sum(1).tolist(), "top": top, "kind": kind.tolist(),
                "mean_bos": [round(float(wa[..., 0].mean()), 4), round(float(wb[..., 0].mean()), 4)],
            }
        out.append(res)
    return out


def main(d, js=None):
    d = Path(d)
    att = np.load(d / "attn.npz")
    meta = json.loads((d / "attn_meta.json").read_text())
    res = analyse(att, meta)
    for r in res:
        print(f"\n== {r['id']}: {r['note']}")
        print("   A:", "|".join(r["A"]))
        print("   B:", "|".join(r["B"]))
        print("   probe A", r["probe"]["A"], " B", r["probe"]["B"])
        print("   chat  A", repr(r["chat"]["A"]), " B", repr(r["chat"]["B"]))
        for name, a in r["at"].items():
            print(f"   @{name} {a['tok']} same_prefix={a['same_prefix']} maxdiff={a['max_abs_diff']}: "
                  f"right {a['right']} (clean {a['clean']}), wrong {a['wrong']} (clean {a['clean_wrong']}), "
                  f"no flip {a['no_flip']}, quiet {a['quiet']}, mean BOS {a['mean_bos']}")
            print("      clean by layer:", a["clean_by_layer"])
            for t in a["top"][:6]:
                print(f"      L{t['layer']:>2} H{t['head']:>2} margin {t['margin']:.3f}  A {t['A']}  B {t['B']}  "
                      f"bos {t['bosA']}/{t['bosB']} self {t['selfA']}/{t['selfB']}")
    if js:
        Path(js).write_text(json.dumps(app_data(att, meta, res), separators=(",", ":")))
        print("wrote", js)


FEATURED = ("brief", 8, 29)   # the pair and head the video shows (the top margin at "tired"/"wide", see DEVLOG)


def app_data(att, meta, res):
    """What the video reads (data/attn.json): the featured pair's tokens, the featured head's full attention
    matrices for both sentences, every head's weights on the two answers at the asking positions (the grid),
    the census counts, the behaviour probes, and a summary of every pair tried."""
    pid, L0, H0 = FEATURED
    r = next(x for x in res if x["id"] == pid)
    pr = next(x for x in meta["pairs"] if x["id"] == pid)
    A, B = att[pid + "A"].astype(np.float32), att[pid + "B"].astype(np.float32)
    rnd = lambda x: np.round(np.asarray(x, dtype=np.float64), 5).tolist()
    grid = {}
    for name, a in r["at"].items():
        qa, qb = a["q"]
        grid[name] = {"q": [qa, qb],
                      # [layer][head] -> [w(ref0) in A, w(ref1) in A, w(ref0) in B, w(ref1) in B]
                      "w": rnd(np.stack([A[:, :, qa, r["refA"][0]], A[:, :, qa, r["refA"][1]],
                                         B[:, :, qb, r["refB"][0]], B[:, :, qb, r["refB"][1]]], -1)),
                      "bos": rnd(np.stack([A[:, :, qa, 0], B[:, :, qb, 0]], -1)),
                      # [layer][head] -> [row in A, row in B]: every head's full weights from the asking token
                      "rows": np.round(np.stack([A[:, :, qa, :qa + 1], B[:, :, qb, :qb + 1]], 2), 4).tolist()
                      if qa == qb and name != "end" else None,
                      **{k: a[k] for k in ("same_prefix", "max_abs_diff", "heads", "right", "clean", "wrong",
                                           "clean_wrong", "no_flip", "quiet", "clean_by_layer", "mean_bos", "top", "kind")}}
    return {
        "model": meta["model"], "gpu": meta["gpu"], "dtype": meta["dtype"], "config": meta["config"],
        "clean": CLEAN, "quiet": QUIET, "featured": {"pair": pid, "layer": L0, "head": H0},
        "A": r["A"], "B": r["B"], "ref": r["ref"], "refA": r["refA"], "refB": r["refB"],
        "head": {"A": rnd(A[L0, H0]), "B": rnd(B[L0, H0])},   # T x T, row = asking token
        "grid": grid,
        "probe": {s: pr["runs"][s]["probe"] for s in "AB"},
        "chat": {s: {k: pr["runs"][s]["chat"][k] for k in ("question", "answer", "top10")} for s in "AB"},
        "pairs": [{"id": x["id"], "note": x["note"], "A": x["A"], "B": x["B"], "ref": x["ref"],
                   "probe": x["probe"], "chat": x["chat"],
                   "at": {n: {k: v for k, v in a.items() if k not in ("top", "kind")} | {"top": a["top"][:3]}
                          for n, a in x["at"].items()}} for x in res],
    }


if __name__ == "__main__":
    a = sys.argv[1:]
    js = a[a.index("--json") + 1] if "--json" in a else None
    main(a[0], js)
