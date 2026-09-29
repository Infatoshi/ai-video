"""Rank this song's takes on what score.py can't see: the choruses end on the model's gibberish, which Whisper
can't recall as words. Per take (from score.py's scores.json + rhythm.py's rhythm.json, copied from the GPU host):
  recall_words   word recall on the sung lines that are not the model's excerpts (the lyrics a beginner reads)
  ex1..ex3       letter similarity (difflib ratio, letters only, lowercase) between each excerpt and the best
                 window of the Whisper transcript around where it should be: did the singer sing those letters?
  bpm, rhyme_grid, couplet_par from rhythm.json
  build          loudness (dB) of the last third minus the first third (the choir should grow)

  python3 song/judge.py <run dir> [<run dir> ...] --excerpts song/excerpts.json [--pick out.txt]

Each run dir holds scores.json and rhythm.json. --pick writes "<run> <seed>" of the best take (the sort key below)
among takes at least 210 s long whose tempo is within 5% of 80 BPM.
"""
import json, re, sys
from difflib import SequenceMatcher
from pathlib import Path

args = sys.argv[1:]
ex = json.load(open(args[args.index("--excerpts") + 1]))["excerpts"]
pick = args[args.index("--pick") + 1] if "--pick" in args else None
skip = {args[args.index(k) + 1] for k in ("--excerpts", "--pick") if k in args}
dirs = [a for a in args if not a.startswith("--") and a not in skip]
norm = lambda s: re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
letters = lambda s: re.sub(r"[^a-z]", "", s.lower())

rows = []
for d in map(Path, dirs):
    sc = json.load(open(d / "scores.json"))
    rh = {r["seed"]: r for r in json.load(open(d / "rhythm.json"))} if (d / "rhythm.json").exists() else {}
    tmpl = (Path(__file__).resolve().parent / "lyrics.sing.tmpl.txt").read_text()
    fixed = [l for l in tmpl.splitlines() if l.strip() and not l.startswith("[") and "{EX" not in l]
    for r in sc:
        hyp = norm(r["hyp"])
        hw = hyp.split()
        ref = [w for l in fixed for w in norm(re.sub(r"\{\w+\}", "", l)).split()]
        sm = SequenceMatcher(a=ref, b=hw, autojunk=False)
        hits = sum(b.size for b in sm.get_matching_blocks())
        hl = letters(r["hyp"])
        sims = []
        for e in ex:
            tgt = letters(e["text"])
            n = len(tgt)
            best = 0.0
            for s in range(0, max(1, len(hl) - n + 1), 3):
                best = max(best, SequenceMatcher(a=tgt, b=hl[s:s + n], autojunk=False).ratio())
            sims.append(round(best, 2))
        x = rh.get(r["seed"], {})
        rows.append(dict(run=d.name, seed=r["seed"], recall=r["recall"], recall_words=round(hits / max(1, len(ref)), 3),
                         ex=sims, dur=r["dur"], CE=r.get("CE"), PQ=r.get("PQ"), bpm=x.get("bpm"),
                         rhyme_grid=x.get("rhyme_grid"), couplet_par=x.get("couplet_par"), bad=len(r["bad_lines"])))
rows.sort(key=lambda r: -(r["recall_words"] + 0.5 * sum(r["ex"]) / 3))
for r in rows:
    print(r)
if pick:
    ok = [r for r in rows if r["dur"] >= 210 and (r["bpm"] is None or 76 <= r["bpm"] <= 84)] or rows
    open(pick, "w").write(f"{ok[0]['run']} {ok[0]['seed']}\n")
    print("pick", ok[0]["run"], ok[0]["seed"])
