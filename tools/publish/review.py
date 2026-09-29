#!/usr/bin/env python3
"""review: the launch review page, what Elliot looks at before a video goes public.

  tools/publish/review.py <project> [<project> ...]    -> out/review/launch.html (a standalone local page; opens in the browser)

Per project and variant: the thumbnail as the YouTube feed shows it (with the title as it truncates), the
phone-grid size (168x94), the Test & Compare candidates side by side, the description's first two lines (all a
viewer sees before "more"), chapters, links to the private uploads with live status, and the launch decisions
(pinned comment, X post text, facts to confirm) with copy buttons. Reads publish.json, out/publish/<variant>/,
and the YouTube API (tools/publish/yt.py's token).
"""
import base64, html, io, json, subprocess, sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E = html.escape


def img64(path, width):
    """JPEG data URI of an image resized to `width` px (ffmpeg, so no Python imaging dependency)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={width}:-2", "-q:v", "3", "-f", "image2", "-c:v", "mjpeg", "-"],
                         capture_output=True, check=True).stdout
    return "data:image/jpeg;base64," + base64.b64encode(out).decode()


def frame64(video, t, width):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2",
                          "-q:v", "3", "-f", "image2", "-c:v", "mjpeg", "-"], capture_output=True, check=True).stdout
    return "data:image/jpeg;base64," + base64.b64encode(out).decode()


def duration(video):
    s = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True).stdout.strip()
    d = float(s or 0)
    return f"{int(d) // 60}:{int(d) % 60:02d}"


def status(ids):
    if not ids:
        return {}
    r = yt.call(f"{yt.API}/videos?part=status,processingDetails,statistics&id=" + ",".join(ids))
    return {v["id"]: v for v in r["items"]}


def copy_block(label, text):
    return (f'<div class="copy"><div class="copy-head"><span class="label">{E(label)}</span>'
            f'<button type="button" data-copy>Copy</button></div><p class="copytext">{E(text)}</p></div>')


def variant_html(p, pub, name, v, st):
    d = p / "out" / "publish" / name
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else yt.meta_for(pub, name, p)
    video = p / v["video"]
    dur = duration(video)
    vid = v.get("youtube_id")
    s = st.get(vid, {})
    priv = s.get("status", {}).get("privacyStatus", "not uploaded")
    if priv == "private" and s.get("status", {}).get("publishAt"):
        at = datetime.fromisoformat(s["status"]["publishAt"].replace("Z", "+00:00")).astimezone(ZoneInfo("America/Edmonton"))
        priv = "scheduled " + at.strftime("%a %b %-d, %-I:%M %p") + " MT"
    proc = s.get("processingDetails", {}).get("processingStatus", "")
    vertical = "vertical" in name or "hook" in name
    secs = int(dur.split(":")[0]) * 60 + int(dur.split(":")[1])
    desc = meta["description"].strip().split("\n")
    fold = "\n".join(desc[:2]) if len(desc[0]) < 120 else desc[0]
    chapters = [l for l in desc if len(l) > 4 and l[0].isdigit() and ":" in l[:5]]
    title = meta["title"]
    parts = []
    parts.append(f'<article class="variant" id="{E(name)}"><header class="vhead"><h3>{E(name)}</h3>'
                 f'<span class="pill {"ok" if priv == "public" or priv.startswith("scheduled") else "hold"}">{E(priv)}</span>'
                 + (f'<span class="pill quiet">{E(proc)}</span>' if proc and proc != "succeeded" else "")
                 + f'<span class="meta">{dur} · {("9:16 Short" if secs <= 180 else "9:16 (over 3:00, not a Short)") if vertical else "16:9"}</span>'
                 + (f'<a class="meta" href="https://youtu.be/{vid}">youtu.be/{vid}</a>' if vid else "") + "</header>")
    parts.append(f'<video class="player{" portrait" if vertical else ""}" controls preload="metadata" src="{E(video.resolve().as_uri())}"></video>')
    if vertical:
        frames = [frame64(video, secs * f, 220) for f in (0.06, 0.42, 0.93)]  # setup, middle, ending
        parts.append('<div class="shorts">' + "".join(f'<img class="short" src="{f}" alt="frame of {E(name)}">' for f in frames)
                     + '<p class="note">Shorts use a frame as the cover (chosen in the YouTube app, not the API). Title as it shows on the Shorts shelf:</p>'
                     + f'<p class="shorttitle">{E(title)}</p></div>')
    else:
        chosen = p / v["thumbnail"]
        tests = [p / t for t in v.get("thumbnails_test", [])]
        parts.append('<div class="feed"><div class="card"><img src="' + img64(chosen, 640) + '" alt="thumbnail">'
                     f'<span class="dur">{dur}</span><div class="ctitle">{E(title)}</div><div class="cchan">Elliotcodes</div></div>'
                     '<div class="phone"><span class="label">Phone grid, 168 x 94</span><div class="phonerow">'
                     + "".join(f'<figure><img src="{img64(t, 168)}" width="168" height="94" alt="variant {t.stem[-1]}"><figcaption>{t.stem[-1]}{" · proposed" if t == chosen else ""}</figcaption></figure>' for t in (tests or [chosen]))
                     + '</div></div></div>')
        if tests:
            parts.append('<div class="tests"><span class="label">Test &amp; Compare candidates</span><div class="testrow">'
                         + "".join(f'<figure><img src="{img64(t, 420)}" alt="thumbnail {t.stem[-1]}"><figcaption>{t.stem[-1]}{" (proposed first)" if t == chosen else ""}</figcaption></figure>' for t in tests)
                         + "</div></div>")
    parts.append(f'<div class="desc"><span class="label">Above the fold</span><p class="fold">{E(fold)}</p>'
                 + (f'<details><summary>{f"{len(chapters)} chapters, full description" if chapters else "Full description"}</summary><pre>{E(meta["description"])}</pre></details>' if chapters or len(desc) > 2 else "")
                 + (copy_block("Pinned comment draft (posts only once you approve it)", v["pinned_comment"]) if v.get("pinned_comment") else "")
                 + "</div></article>")
    return "".join(parts)


def main(projects):
    blocks, todo, uploaded = [], [], 0
    for name in projects:
        p = REPO / "projects" / name
        pub = json.loads((p / "publish.json").read_text())
        st = status([v["youtube_id"] for v in pub["variants"].values() if v.get("youtube_id")])
        uploaded += sum(1 for v in pub["variants"].values() if v.get("youtube_id"))
        launch = pub.get("launch", {})
        body = "".join(variant_html(p, pub, vn, v, st) for vn, v in pub["variants"].items())
        dec = ""
        if launch.get("already_posted"):
            dec += f'<p class="posted"><span class="label">Already out</span> {E(launch["already_posted"])}</p>'
        if launch.get("pinned_comment"):
            dec += copy_block("Pinned comment (posts once you approve it)", launch["pinned_comment"])
        if launch.get("x_post"):
            dec += copy_block(f'X post (attach {launch.get("x_media", "")})', launch["x_post"])
        for c in launch.get("checks", []):
            dec += f'<p class="check">{E(c)}</p>'
        blocks.append(f'<section class="project"><h2>{E(name)}</h2>{body}' + (f'<div class="decide"><h4>Decide</h4>{dec}</div>' if dec else "") + "</section>")
    studio = ["Make each video and the playlist Public (YouTube locks API uploads private)",
              "Then, because YouTube only allows these on public videos: Test & Compare with A, B, C on each 16:9 video",
              "End screens: each 16:9 video points at the other, plus Subscribe",
              "Link each Short to its full video (Related video)",
              "Pin the comment once you approve it and it is posted"]
    page = TEMPLATE.replace("{{BODY}}", "".join(blocks)).replace("{{STUDIO}}", "".join(f"<li>{E(s)}</li>" for s in studio)) \
        .replace("{{DATE}}", datetime.now().strftime("%b %-d, %Y %-I:%M %p")).replace("{{PLAYLIST}}", "Music videos that teach ML (private)") \
        .replace("{{STATE}}", "Everything below is uploaded and private." if uploaded else "Nothing below is uploaded yet.")
    out = REPO / "out" / "review" / "launch.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
    subprocess.run(["open", str(out)], check=False)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Launch Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@112,500;112,800;125,900&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{--paper:#F2EFE6;--card:#FBF9F4;--ink:#231F20;--muted:#6F6A63;--rule:#DAD5CB;--pink:#E0348F;--blue:#0068A8;--hold:#B8691B;--ok:#2E7D4F;
  --display:"Archivo",ui-sans-serif,system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#17161A;--card:#211F25;--ink:#EFEBE3;--muted:#A39E96;--rule:#34313A;--pink:#FF5CB0;--blue:#4AA8E8;--hold:#E0A15A;--ok:#6FC38F;color-scheme:dark}}
:root[data-theme="dark"]{--paper:#17161A;--card:#211F25;--ink:#EFEBE3;--muted:#A39E96;--rule:#34313A;--pink:#FF5CB0;--blue:#4AA8E8;--hold:#E0A15A;--ok:#6FC38F;color-scheme:dark}
body{background:var(--paper);color:var(--ink);font:15px/1.55 var(--display);font-stretch:112%;padding-inline:clamp(16px,4vw,48px);padding-block:32px 64px}
.wrap{max-width:1100px;margin:0 auto;display:grid;gap:40px}
h1{font-weight:900;font-stretch:125%;font-size:clamp(30px,5vw,48px);line-height:1;margin:0;text-wrap:balance}
h2{font-weight:900;font-stretch:125%;font-size:26px;margin:0;letter-spacing:.01em}
h3{font-family:var(--mono);font-size:14px;font-weight:600;margin:0}
h4{margin:0;font-size:13px;text-transform:uppercase;letter-spacing:.08em}
.label{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.lede{max-width:65ch;color:var(--muted);margin:10px 0 0}
.top{display:grid;gap:14px}
.studio{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:16px 20px}
.studio ol{margin:8px 0 0;padding-left:20px;display:grid;gap:4px}
.project{display:grid;gap:22px;border-top:3px solid var(--ink);padding-top:18px}
.variant{display:grid;gap:14px;padding-bottom:22px;border-bottom:1px solid var(--rule)}
.vhead{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted)}
a.meta{color:var(--blue)}
.pill{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.06em;padding:2px 8px;border-radius:99px;border:1px solid currentColor}
.pill.hold{color:var(--hold)}.pill.ok{color:var(--ok)}.pill.quiet{color:var(--muted)}
.feed{display:grid;grid-template-columns:minmax(0,360px) 1fr;gap:24px;align-items:start}
.card{position:relative;max-width:360px}
.player{width:100%;max-width:960px;border-radius:10px;background:#000;display:block}
.player.portrait{max-width:340px}
.card img{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:10px;display:block}
.dur{position:absolute;right:8px;top:calc((100% - 58px) - 30px);background:rgba(0,0,0,.8);color:#fff;font:600 12px var(--mono);padding:1px 5px;border-radius:4px}
.ctitle{font-weight:800;font-size:15px;line-height:1.35;margin-top:10px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.cchan{font-size:13px;color:var(--muted)}
.phone{display:grid;gap:8px}
.phonerow,.testrow{display:flex;flex-wrap:wrap;gap:14px}
figure{margin:0;display:grid;gap:4px}
figcaption{font-family:var(--mono);font-size:11px;color:var(--muted)}
.phonerow img{border-radius:6px;display:block;max-width:100%;height:auto}
.tests{display:grid;gap:8px}
.testrow figure{flex:1 1 240px;max-width:340px}
.testrow img{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:8px;display:block}
.shorts{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}
.short{width:150px;max-width:30%;aspect-ratio:9/16;object-fit:cover;border-radius:10px}
.note{flex-basis:100%;margin:0;color:var(--muted);font-size:13px}
.shorttitle{margin:0;font-weight:800}
.desc{display:grid;gap:6px;max-width:72ch}
.fold{margin:0;white-space:pre-line}
details summary{cursor:pointer;color:var(--blue);font-size:13px}
pre{white-space:pre-wrap;font:12px/1.5 var(--mono);background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:12px;overflow-x:auto}
.decide{display:grid;gap:12px;background:var(--card);border:1px solid var(--rule);border-left:4px solid var(--pink);border-radius:10px;padding:16px 20px}
.copy{display:grid;gap:6px}.copy-head{display:flex;justify-content:space-between;align-items:center;gap:12px}
.copytext{margin:0;font-family:var(--mono);font-size:13px;max-width:72ch}
button{font:600 12px var(--mono);color:var(--ink);background:transparent;border:1px solid var(--rule);border-radius:6px;padding:4px 10px;cursor:pointer}
button:focus-visible,a:focus-visible,summary:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
.check{margin:0;padding-left:14px;border-left:2px solid var(--hold)}
.posted{margin:0}
@media (max-width:720px){.feed{grid-template-columns:1fr}}
body{margin:0}img{max-width:100%}
</style></head><body>
<div class="wrap">
<div class="top">
<span class="label">ai-video · launch review · {{DATE}}</span>
<h1>Launch Review</h1>
<p class="lede">{{STATE}} Look at the thumbnails at the sizes viewers see them, read the two lines above the fold, then say which items to change, which texts to post, and when to go public. Playlist: {{PLAYLIST}}.</p>
<div class="studio"><span class="label">One Codex computer-use pass after your go (YouTube refuses these while private)</span><ol>{{STUDIO}}</ol></div>
</div>
{{BODY}}
</div>
<script>
document.querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',()=>{
  const t=b.closest('.copy').querySelector('.copytext');
  navigator.clipboard.writeText(t.textContent).then(()=>{b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1500)})
  .catch(()=>{const r=document.createRange();r.selectNodeContents(t);const s=getSelection();s.removeAllRanges();s.addRange(r);});
}));
</script>
</body></html>
"""

if __name__ == "__main__":
    main(sys.argv[1:])
