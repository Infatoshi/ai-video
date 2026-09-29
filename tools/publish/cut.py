#!/usr/bin/env python3
"""cut: build a short from beat-aligned segments of a finished render (no re-render).

  tools/publish/cut.py <project> <variant>

The variant in publish.json has "source" (a rendered video, e.g. out/final/<p>_vertical.mp4), "cut" (a list of
[t0, t1] song times) and "video" (the output). Each time snaps to the nearest beat of data/audio.json, so pick
segments that start on the same beat of the bar and the music joins in time. Video is cut frame-accurately and
re-encoded; audio comes from the master (audio/<song>.wav) with 12 ms crossfades at the joins. Captions: the lyric
lines that fall inside the segments, re-timed, go to out/publish/<variant>/captions.srt via `yt.py package`.
"""
import json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(proj, var):
    p = REPO / "projects" / proj
    pub = json.loads((p / "publish.json").read_text())
    v = pub["variants"][var]
    au = json.loads((p / "data" / "audio.json").read_text())
    beats = au["beats"]
    snap = lambda t: min(beats, key=lambda b: abs(b - t))
    segs = [(snap(a), snap(b)) for a, b in v["cut"]]
    song = next((p / "audio").glob("*.wav"))
    fps = 60
    vf, af = [], []
    for i, (a, b) in enumerate(segs):
        fa, fb = round(a * fps), round(b * fps)
        vf.append(f"[0:v]trim=start_frame={fa}:end_frame={fb},setpts=PTS-STARTPTS[v{i}]")
        af.append(f"[1:a]atrim=start={fa / fps}:end={fb / fps},asetpts=PTS-STARTPTS[a{i}]")
    n = len(segs)
    chain = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    ac = "[a0]"
    for i in range(1, n):
        ac_out = f"[x{i}]" if i < n - 1 else "[a]"
        af.append(f"{ac}[a{i}]acrossfade=d=0.012:c1=tri:c2=tri{ac_out}")
        ac = ac_out
    if n == 1:
        af.append("[a0]anull[a]")
    out = p / v["video"]
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(p / v["source"]), "-i", str(song), "-filter_complex",
           ";".join(vf + af + [chain]), "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    # the kept song time of each output second, for re-timed captions
    v["segments_snapped"] = [[round(a, 3), round(b, 3)] for a, b in segs]
    (p / "publish.json").write_text(json.dumps(pub, indent=1))
    dur = sum(b - a for a, b in segs)
    print(out, f"{dur:.2f} s", "segments", v["segments_snapped"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
