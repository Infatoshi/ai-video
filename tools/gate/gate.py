# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Activity gate: how much is happening on screen, second by second (SPEC.md "Gates").

  uv run tools/gate/gate.py projects/next-token/out/final/next_token.mp4 [--vs other.mp4] [--floor 0.55] [--project DIR]

Decodes the video at 30 fps, 320 px wide, greyscale, and per second measures
  motion  mean |frame difference| (how much changes)
  detail  mean gradient magnitude (how much is there)
  cuts    frames whose difference jumps above 4x the local median (hard cuts)
activity = motion and detail each normalised by the video's median, averaged. Seconds below
`floor` x median are flagged.

A project with a gate.json ({"profile": "course", ...}, the slow series in AGENTS.md) is gated on pace instead:
no calm-seconds floor; ceilings on sung words per minute (whole song and the busiest 30 s, from data/lyrics.json)
and hard changes per minute (cuts plus anything that pops in within a frame; fades do not count); a floor on the
median shot length; frozen stretches (nothing moves) capped outside the last 5 s. Exits 1 if any check fails. Writes <video>.gate.json and <video>.gate.png (activity under the song's
energy from the project's data/audio.json (the project is inferred from projects/<p>/out/final/<video>), flagged seconds in pink; with --vs, the other video's curve in grey).
"""
import json, subprocess, sys
from pathlib import Path

import numpy as np

FPS, WD = 30, 320


def frames(path):
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(info[0]), int(info[1])
    hh = int(round(WD * h / w / 2) * 2)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={FPS},scale={WD}:{hh}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    n = WD * hh
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(hh, WD).astype(np.float32) / 255


def measure(path):
    diffs, dets = [], []
    prev = None
    for f in frames(path):
        gy, gx = np.gradient(f)
        dets.append(float(np.mean(np.hypot(gx, gy))))
        diffs.append(0.0 if prev is None else float(np.mean(np.abs(f - prev))))
        prev = f
    d, g = np.array(diffs), np.array(dets)
    med = np.array([np.median(d[max(0, i - 15):i + 15]) for i in range(len(d))])
    cut = d > 4 * np.maximum(med, 1e-3)
    secs = len(d) // FPS
    per = lambda x, fn=np.mean: np.array([fn(x[s * FPS:(s + 1) * FPS]) for s in range(secs)])
    motion, detail, cuts = per(d), per(g), per(cut.astype(float), np.sum)
    act = 0.5 * motion / np.median(motion) + 0.5 * detail / np.median(detail)
    return dict(motion=motion.round(5).tolist(), detail=detail.round(5).tolist(), cuts=cuts.astype(int).tolist(),
                activity=act.round(3).tolist(), cut_t=(np.nonzero(cut)[0] / FPS).round(2).tolist(),
                still=(d < 1e-3).astype(int).tolist())


COURSE = dict(max_words_per_min=95, max_words_per_min_30s=120, max_hard_changes_per_min=6, min_median_shot_s=8,
              max_frozen_s=3)


def pace(m, root, dur, cfg):
    """The course's pace checks; returns (numbers, failures)."""
    lines = json.loads((root / "data" / "lyrics.json").read_text())["lines"]
    wt = sorted(w["start"] for l in lines for w in l.get("words", []))
    span = (max(w["end"] for l in lines for w in l.get("words", [])) - wt[0]) if wt else 0
    wpm = len(wt) / span * 60 if span else 0
    wpm30 = max((sum(1 for x in wt if a <= x < a + 30) * 2 for a in wt), default=0)
    ct = m["cut_t"]
    shots = np.diff([0.0] + ct + [dur])
    runs, a = [], None
    for i, st in enumerate(m["still"] + [0]):
        if st and a is None:
            a = i
        elif not st and a is not None:
            runs.append((a / FPS, (i - a) / FPS)); a = None
    frozen = [(round(a, 1), round(b, 1)) for a, b in runs if b > cfg["max_frozen_s"] and a < dur - 5]
    got = dict(words_per_min=round(wpm), words_per_min_30s=wpm30, hard_changes_per_min=round(len(ct) / dur * 60, 1),
               median_shot_s=round(float(np.median(shots)), 1), frozen=frozen)
    fails = [f"{k} {got[k]} > {cfg['max_' + k]}" for k in ("words_per_min", "words_per_min_30s", "hard_changes_per_min")
             if got[k] > cfg["max_" + k]]
    if got["median_shot_s"] < cfg["min_median_shot_s"]:
        fails.append(f"median_shot_s {got['median_shot_s']} < {cfg['min_median_shot_s']}")
    fails += [f"frozen {b} s at {a} s" for a, b in frozen]
    return got, fails


def main():
    a = sys.argv[1:]
    vs = a[a.index("--vs") + 1] if "--vs" in a else None
    floor = float(a[a.index("--floor") + 1]) if "--floor" in a else 0.55
    path = Path(a[0]).resolve()
    root = Path(a[a.index("--project") + 1]).resolve() if "--project" in a else path.parents[2]
    m = measure(path)
    act = np.array(m["activity"])
    med = float(np.median(act))
    gj = root / "gate.json"
    cfg = {**COURSE, **json.loads(gj.read_text())} if gj.exists() else None
    flagged = [] if cfg else [i for i, x in enumerate(act) if x < floor * med]
    m.update(median=round(med, 3), p10=round(float(np.percentile(act, 10)), 3), spread=round(float(np.percentile(act, 10) / med), 3),
             flagged=flagged, cuts_total=int(sum(m["cuts"])))
    fails = []
    if cfg:
        m["pace"], fails = pace(m, root, len(m["still"]) / FPS, cfg)
    path.with_suffix(".gate.json").write_text(json.dumps(m))
    print(f"{path.name}: {len(act)} s, median activity {med:.3f}, p10/median {m['spread']:.2f}, "
          f"{len(flagged)} s under {floor} x median: {flagged}, {m['cuts_total']} cuts")
    if cfg:
        print("pace:", m["pace"], "\n" + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 4.5), facecolor="#F2EFE6")
    ax.set_facecolor("#F2EFE6")
    au = json.loads((root / "data" / "audio.json").read_text())
    if "rms" in au:
        e = np.array(au["rms"], dtype=float)
        tt = np.arange(len(e)) / au.get("fps", 100)
        ax.fill_between(tt, 0, e / e.max() * act.max(), color="#0078BF", alpha=0.18, lw=0, label="song energy")
    if vs:
        o = measure(vs)
        ax.plot(np.arange(len(o["activity"])) + 0.5, o["activity"], color="#8F8A83", lw=1.2, label=Path(vs).name)
    x = np.arange(len(act)) + 0.5
    ax.plot(x, act, color="#231F20", lw=1.8, label=path.name)
    if not cfg:
        ax.axhline(floor * med, color="#FF48B0", lw=1, ls="--", label=f"floor ({floor} x median)")
    for t in m["cut_t"] if cfg else []:
        ax.axvline(t, color="#FF48B0", lw=0.8, alpha=0.7)
    ax.scatter([i + 0.5 for i in flagged], act[flagged], color="#FF48B0", s=26, zorder=3)
    for s in au.get("sections", []):
        ax.axvline(s["start"], color="#231F20", lw=0.5, alpha=0.4)
        ax.text(s["start"] + 0.3, act.max() * 1.02, s["name"], fontsize=8, color="#231F20")
    ax.set_xlim(0, len(act))
    ax.set_xlabel("seconds")
    ax.set_ylabel("activity (1 = median)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".gate.png"), dpi=110)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
