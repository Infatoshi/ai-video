#!/usr/bin/env python3
"""yt: the YouTube side of the workflow, all from the CLI (stdlib + youtubeuploader for the upload itself).

  tools/publish/yt.py auth                                   one-time consent (opens the browser; Elliot clicks Allow)
  tools/publish/yt.py package  <project> [variant ...]       projects/<p>/publish.json -> out/publish/<variant>/
  tools/publish/cut.py <project> <variant>                   a short from beat-aligned segments of a render
  tools/publish/yt.py upload   <project> <variant>           upload PRIVATE (+ captions, thumbnail, AI disclosure)
  tools/publish/yt.py update   <project> [variant ...]       push title/description/tags/thumbnail to uploaded videos
  tools/publish/yt.py status   <project>                     privacy, processing, views of the project's videos
  tools/publish/yt.py playlist create "<title>" [--description ...]      (private)
  tools/publish/yt.py playlist add <playlist_id> <video_id> [...]
  tools/publish/yt.py comment  <video_id> "<text>" --yes     posts as Elliot: only with his go on that exact text
  tools/publish/yt.py analytics <video_id> [--days 28]       views, watch time, retention (YouTube Analytics API)

Not possible over the API (Studio only): end screens, cards, Test & Compare thumbnails, a Short's related
video, pinning a comment, and making a video public while the Google project is unaudited (API uploads are
forced private). Any Google OAuth desktop client with the YouTube Data and Analytics APIs enabled.

publish.json (per project): chapters {timeline id: title}, tags, categoryId, language, containsSyntheticMedia,
description (with {chapters}), variants {name: {video, title, thumbnail?, youtube_id?}}. Chapters come from
<project>/data/timeline.json (app: `bun scripts/render.ts timeline`), captions from <project>/data/lyrics.json.
"""
import argparse, http.server, json, os, shutil, subprocess, sys, threading, time, urllib.parse, urllib.request, webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CFG = Path.home() / ".config" / "youtubeuploader"   # shared with youtubeuploader (same client, same token)
SECRETS, TOKEN = CFG / "client_secrets.json", CFG / "request.token"
SCOPES = ["https://www.googleapis.com/auth/youtube", "https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube.force-ssl", "https://www.googleapis.com/auth/youtubepartner",
          "https://www.googleapis.com/auth/yt-analytics.readonly"]
API = "https://www.googleapis.com/youtube/v3"


# ------------------------------------------------------------------ auth

def client():
    return json.loads(SECRETS.read_text())["installed"]


def cmd_auth(_a):
    c = client()
    redirect = "http://localhost:8080/oauth2callback"
    url = "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode({
        "client_id": c["client_id"], "redirect_uri": redirect, "response_type": "code", "scope": " ".join(SCOPES),
        "access_type": "offline", "prompt": "consent", "include_granted_scopes": "true"})
    got = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got.update({k: v[0] for k, v in q.items()})
            self.send_response(200); self.end_headers()
            self.wfile.write(b"yt: authorized, you can close this tab.")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("localhost", 8080), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    print("opening the consent page (pick the Elliotcodes account, then Allow)")
    webbrowser.open(url)
    for _ in range(600):
        if got:
            break
        time.sleep(0.5)
    if "code" not in got:
        sys.exit(f"no authorization code: {got or 'timed out'}")
    body = urllib.parse.urlencode({"code": got["code"], "client_id": c["client_id"], "client_secret": c["client_secret"],
                                   "redirect_uri": redirect, "grant_type": "authorization_code"}).encode()
    t = json.load(urllib.request.urlopen(urllib.request.Request(c["token_uri"], data=body)))
    save_token(t)
    print("scopes:", t.get("scope"))


def save_token(t, old=None):
    # the same JSON shape as Go's oauth2.Token, so youtubeuploader reads it too
    exp = datetime.now(timezone.utc) + timedelta(seconds=int(t.get("expires_in", 3600)))
    tok = {"access_token": t["access_token"], "token_type": "Bearer",
           "refresh_token": t.get("refresh_token") or (old or {}).get("refresh_token"), "expiry": exp.isoformat()}
    CFG.mkdir(parents=True, exist_ok=True)
    fd = os.open(TOKEN, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(tok, f)


def access_token():
    t = json.loads(TOKEN.read_text())
    try:
        if datetime.fromisoformat(t["expiry"].replace("Z", "+00:00")).timestamp() > time.time() + 120:
            return t["access_token"]
    except (KeyError, ValueError):
        pass
    c = client()
    body = urllib.parse.urlencode({"client_id": c["client_id"], "client_secret": c["client_secret"],
                                   "refresh_token": t["refresh_token"], "grant_type": "refresh_token"}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(c["token_uri"], data=body)))
    save_token(r, t)
    return r["access_token"]


def call(url, data=None, method="GET", ctype="application/json"):
    if isinstance(data, (dict, list)):
        data = json.dumps(data).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": "Bearer " + access_token(), "Content-Type": ctype})
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url.split('?')[0]} -> {e.code}: {e.read().decode()[:500]}")


# ------------------------------------------------------------------ package

def project(name):
    p = REPO / "projects" / name
    if not (p / "publish.json").exists():
        sys.exit(f"{p}/publish.json missing")
    return p, json.loads((p / "publish.json").read_text())


def chapters(p, pub):
    tl = json.loads((p / "data" / "timeline.json").read_text())
    rows = [(0 if i == 0 else e["start"] + 0.5, pub["chapters"][e["id"]]) for i, e in enumerate(tl) if e["id"] in pub["chapters"]]
    return "\n".join(f"{int(t) // 60}:{int(t) % 60:02d} {n}" for t, n in rows)


def srt(p, segments=None):
    """Lyric lines as SRT; with segments ([[t0, t1], ...] of song time kept in a cut), only the lines inside them,
    re-timed onto the cut's clock."""
    ly = json.loads((p / "data" / "lyrics.json").read_text())
    ts = lambda t: f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(round(t * 1000)) % 1000:03d}"
    cues = []
    for l in ly["lines"]:
        a, b = l["words"][0]["start"], l["words"][-1]["end"]
        if segments:
            off = 0.0
            for s0, s1 in segments:
                if a >= s0 - 0.3 and a < s1:
                    if min(b, s1) - max(a, s0) >= 0.6:  # skip a line the cut only grazes
                        cues.append((max(a, s0) - s0 + off, min(b, s1) - s0 + off, l["text"]))
                    break
                off += s1 - s0
        else:
            cues.append((a, b, l["text"]))
    return "\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n" for i, (a, b, t) in enumerate(cues, 1))


def meta_for(pub, var, p):
    v = pub["variants"][var]
    desc = v["description"] if "description" in v else pub["description"].replace("{chapters}", chapters(p, pub))
    return {"title": v["title"], "description": desc, "tags": pub["tags"],
            "privacyStatus": "private", "madeForKids": False, "embeddable": True, "license": "youtube",
            "publicStatsViewable": True, "categoryId": pub.get("categoryId", "27"), "language": pub.get("language", "en"),
            "containsSyntheticMedia": pub.get("containsSyntheticMedia", True)}


def cmd_package(a):
    p, pub = project(a.project)
    for var in a.variants or list(pub["variants"]):
        d = p / "out" / "publish" / var
        d.mkdir(parents=True, exist_ok=True)
        m = meta_for(pub, var, p)
        (d / "meta.json").write_text(json.dumps(m, indent=1))
        (d / "captions.srt").write_text(srt(p, pub["variants"][var].get("segments_snapped")))
        th = pub["variants"][var].get("thumbnail")
        if th:
            shutil.copy(p / th, d / "thumbnail.jpg")
        print(var, "|", m["title"], "|", sorted(x.name for x in d.iterdir()))


# ------------------------------------------------------------------ upload / update / status

def set_status(vid, pub):
    body = {"id": vid, "status": {"privacyStatus": "private", "embeddable": True, "license": "youtube", "publicStatsViewable": True,
                                  "selfDeclaredMadeForKids": False, "containsSyntheticMedia": pub.get("containsSyntheticMedia", True)}}
    return call(f"{API}/videos?part=status", body, "PUT")["status"]


def cmd_upload(a):
    p, pub = project(a.project)
    v = pub["variants"][a.variant]
    if v.get("youtube_id") and not a.again:
        sys.exit(f"{a.variant} is already uploaded: https://youtu.be/{v['youtube_id']} (use update, or --again)")
    d = p / "out" / "publish" / a.variant
    if not (d / "meta.json").exists():
        sys.exit("run package first")
    access_token()  # refresh the shared cache before youtubeuploader reads it
    args = ["youtubeuploader", "-secrets", str(SECRETS), "-cache", str(TOKEN), "-notify=false", "-quiet",
            "-filename", str(p / v["video"]), "-metaJSON", str(d / "meta.json"), "-caption", str(d / "captions.srt"),
            "-metaJSONout", str(d / "uploaded.json")]
    if (d / "thumbnail.jpg").exists():
        args += ["-thumbnail", str(d / "thumbnail.jpg")]
    subprocess.run(args, check=True)
    vid = json.loads((d / "uploaded.json").read_text())["id"]
    st = set_status(vid, pub)  # youtubeuploader drops containsSyntheticMedia
    pub["variants"][a.variant]["youtube_id"] = vid
    (p / "publish.json").write_text(json.dumps(pub, indent=1))
    print(f"https://youtu.be/{vid} {st['privacyStatus']} ai-disclosure={st.get('containsSyntheticMedia')}")


def cmd_update(a):
    p, pub = project(a.project)
    for var in a.variants or [k for k, v in pub["variants"].items() if v.get("youtube_id")]:
        v = pub["variants"][var]
        m = meta_for(pub, var, p)
        snip = {"title": m["title"], "description": m["description"], "tags": m["tags"], "categoryId": m["categoryId"],
                "defaultLanguage": m["language"], "defaultAudioLanguage": m["language"]}
        r = call(f"{API}/videos?part=snippet", {"id": v["youtube_id"], "snippet": snip}, "PUT")
        print(var, v["youtube_id"], "|", r["snippet"]["title"])
        if v.get("thumbnail"):
            data = (p / v["thumbnail"]).read_bytes()
            call(f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={v['youtube_id']}", data, "POST", "image/jpeg")
            print("  thumbnail set")


def cmd_status(a):
    p, pub = project(a.project)
    ids = {v["youtube_id"]: k for k, v in pub["variants"].items() if v.get("youtube_id")}
    if not ids:
        sys.exit("nothing uploaded")
    r = call(f"{API}/videos?part=snippet,status,statistics,processingDetails&id=" + ",".join(ids))
    for v in r["items"]:
        s, st = v["statistics"], v["status"]
        print(f"{ids[v['id']]:<22} https://youtu.be/{v['id']}  {st['privacyStatus']:<8} {v.get('processingDetails', {}).get('processingStatus', ''):<10}"
              f" views {s.get('viewCount', 0):>7} likes {s.get('likeCount', 0):>5} comments {s.get('commentCount', 0):>4} | {v['snippet']['title']}")


# ------------------------------------------------------------------ playlists, comments, analytics

def cmd_playlist(a):
    if a.op == "create":
        r = call(f"{API}/playlists?part=snippet,status", {"snippet": {"title": a.args[0], "description": a.description or ""},
                                                           "status": {"privacyStatus": "private"}}, "POST")
        print(r["id"], "private |", r["snippet"]["title"])
    elif a.op == "add":
        pl, vids = a.args[0], a.args[1:]
        for vid in vids:
            call(f"{API}/playlistItems?part=snippet", {"snippet": {"playlistId": pl, "resourceId": {"kind": "youtube#video", "videoId": vid}}}, "POST")
            print("added", vid)
    else:
        sys.exit("playlist create|add")


def cmd_comment(a):
    if not a.yes:
        sys.exit(f"not posted. This comments as Elliot; rerun with --yes once he has approved this exact text:\n{a.text}")
    r = call(f"{API}/commentThreads?part=snippet", {"snippet": {"videoId": a.video_id, "topLevelComment": {"snippet": {"textOriginal": a.text}}}}, "POST")
    print("posted", r["id"], "(pin it in Studio: the API cannot pin)")


def cmd_analytics(a):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=a.days)
    q = urllib.parse.urlencode({"ids": "channel==MINE", "startDate": start.isoformat(), "endDate": end.isoformat(),
                                "metrics": "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,likes,comments,shares,subscribersGained",
                                "filters": f"video=={a.video_id}"})
    r = call(f"https://youtubeanalytics.googleapis.com/v2/reports?{q}")
    for h, v in zip([c["name"] for c in r.get("columnHeaders", [])], (r.get("rows") or [[]])[0]):
        print(f"{h:<26} {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("auth").set_defaults(fn=cmd_auth)
    x = sp.add_parser("package"); x.add_argument("project"); x.add_argument("variants", nargs="*"); x.set_defaults(fn=cmd_package)
    x = sp.add_parser("upload"); x.add_argument("project"); x.add_argument("variant"); x.add_argument("--again", action="store_true"); x.set_defaults(fn=cmd_upload)
    x = sp.add_parser("update"); x.add_argument("project"); x.add_argument("variants", nargs="*"); x.set_defaults(fn=cmd_update)
    x = sp.add_parser("status"); x.add_argument("project"); x.set_defaults(fn=cmd_status)
    x = sp.add_parser("playlist"); x.add_argument("op"); x.add_argument("args", nargs="+"); x.add_argument("--description"); x.set_defaults(fn=cmd_playlist)
    x = sp.add_parser("comment"); x.add_argument("video_id"); x.add_argument("text"); x.add_argument("--yes", action="store_true"); x.set_defaults(fn=cmd_comment)
    x = sp.add_parser("analytics"); x.add_argument("video_id"); x.add_argument("--days", type=int, default=28); x.set_defaults(fn=cmd_analytics)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
