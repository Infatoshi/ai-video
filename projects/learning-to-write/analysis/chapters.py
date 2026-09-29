"""Chapter marks for tools/publish/yt.py: the episode is one continuous scene, so data/timeline.json (which yt.py
reads for chapters) lists the song's sections, each starting a beat before its first sung line.

  python3 analysis/chapters.py
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1]
ly = json.loads((P / "data" / "lyrics.json").read_text())["lines"]
au = json.loads((P / "data" / "audio.json").read_text())
beat = au["beat_period"] if "beat_period" in au else 60 / au["bpm"]


def first(q, nth=0):
    hits = [l for l in ly if q.lower() in l["text"].lower()]
    return hits[nth]["start"] if len(hits) > nth else None


marks = [("hook", 0.0), ("model", first("A model is numbers")), ("chorus1", first("Guess the next letter", 0)),
         ("loss", first("Its only book")), ("chorus2", first("Guess the next letter", 1)),
         ("steps", first("Sixteen thousand")), ("chorus3", first("Guess the next letter", 2)),
         ("outro", first("minutes on one graphics card"))]
tl = [{"id": k, "start": round(max(0.0, t - beat), 2) if k != "hook" else 0.0} for k, t in marks if t is not None]
(P / "data" / "timeline.json").write_text(json.dumps(tl, indent=1))
print(json.dumps(tl))
