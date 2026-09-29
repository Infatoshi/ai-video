"""Chapters for YouTube (tools/publish/yt.py reads data/timeline.json: [{id, start, end}] with ids that
publish.json's "chapters" names). The video is one continuous scene, so the chapters are anchored to sung lines
instead of timeline entries. Chapters under 10 s (YouTube's minimum) fold into the one before.

  python3 tok/chapters.py      # from projects/tokens/, after the alignment (data/lyrics.json)
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parent.parent
ly = json.loads((P / "data/lyrics.json").read_text())["lines"]
dur = json.loads((P / "data/audio.json").read_text())["duration"]


def start(q, nth=0):
    hits = [l for l in ly if q.lower() in l["text"].lower()]
    return hits[nth]["start"] if len(hits) > nth else None


marks = [("intro", 0.0), ("cut", start("Before it reads")), ("chorus", start("A token is a chunk")),
         ("bpe", start("Where do the chunks")), ("vocab", start("chunks were made this way")),
         ("chorus2", start("A token is a chunk", 1)),
         ("strawberry", start("Look at strawberry")), ("test", start("Give it one chunk")),
         ("bridge", start("When it wrote the word")), ("finale", start("A token is a chunk", 2)),
         ("outro", start("What does a number mean"))]
marks = [(k, 0.0 if k == "intro" else max(0.0, t - (1.6 if k == "vocab" else 1.0))) for k, t in marks if t is not None]
out = []
for k, t in marks:
    if out and t - out[-1]["start"] < 10:
        continue
    out.append({"id": k, "start": round(t, 2)})
for a, b in zip(out, out[1:] + [{"start": dur}]):
    a["end"] = b["start"]
(P / "data/timeline.json").write_text(json.dumps(out, indent=1))
for e in out:
    print(f"{int(e['start']) // 60}:{int(e['start']) % 60:02d}  {e['id']}  ({e['end'] - e['start']:.0f} s)")
