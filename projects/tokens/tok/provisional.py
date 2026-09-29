"""Provisional timing (data/*.approx.json) from the planned bar structure at 100 BPM, so the scene can be
built before the take exists. The app loads data/lyrics.json / audio.json first; the aligned take's files
replace these. Never used in a render."""
import json, re
from pathlib import Path

P = Path(__file__).resolve().parent.parent
BEAT = 0.6
plan = [("intro", 8), ("verse1", 16), ("chorus1", 8), ("inst1", 4), ("verse2", 16), ("chorus2", 8), ("inst2", 4),
        ("verse3", 16), ("bridge", 8), ("chorus3", 10), ("outro", 6)]
sing = [l.strip() for l in (P / "song/lyrics.sing.txt").read_text().splitlines()]
blocks, cur = {}, None
names = iter(n for n, _ in plan if not n.startswith(("intro", "inst")))
for l in sing:
    if l.startswith("["):
        cur = None if ("Intro" in l or "Instrumental" in l) else next(names)
        continue
    if l and cur:
        blocks.setdefault(cur, []).append(l)
sections, lines, bar = [], [], 0
for name, bars in plan:
    t0 = bar * 4 * BEAT
    for k, text in enumerate(blocks.get(name, [])):
        ls = t0 + k * 8 * BEAT + 0.5 * BEAT
        ws = text.split()
        main = [w for w in ws if not w.startswith("(")]
        words, i = [], 0
        echo = False
        for w in ws:
            if w.startswith("("):
                echo = True
            if not echo:
                s = ls + i * (5.0 * BEAT / max(1, len(main))); i += 1
                e = s + 5.0 * BEAT / max(1, len(main)) * 0.9
            else:
                j = len(words) - len(main)
                s = ls + 5.5 * BEAT + j * 0.4; e = s + 0.36
            words.append({"w": w, "start": round(s, 3), "end": round(e, 3)})
        lines.append({"i": len(lines), "text": text, "start": words[0]["start"], "end": words[-1]["end"], "words": words})
    sections.append({"name": name, "start": round(t0, 3), "end": round((bar + bars) * 4 * BEAT, 3)})
    bar += bars
dur = bar * 4 * BEAT
beats = [round(i * BEAT, 3) for i in range(int(dur / BEAT) + 1)]
(P / "data/lyrics.approx.json").write_text(json.dumps({"lines": lines, "extras": [], "notes": "provisional"}))
(P / "data/audio.approx.json").write_text(json.dumps({"duration": dur, "bpm": 100, "beat_period": BEAT, "time_signature": 4,
    "beats": beats, "downbeats": beats[::4], "sections": sections, "fps": 100, "onsets": {}}))
print(len(lines), "lines", dur, "s", [(s["name"], s["start"]) for s in sections])
