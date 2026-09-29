#!/usr/bin/env python3
"""tt: TikTok from the CLI (stdlib only). Uploads go to the authorized account's TikTok drafts (inbox); you open the
TikTok app, edit the caption and post. Nothing is published by this tool.

  tools/publish/tt.py auth   [--env sandbox|production]   one-time: opens TikTok's consent page, you click Authorize
  tools/publish/tt.py whoami [--env ...]                  the authorized account (user.info.basic)
  tools/publish/tt.py upload <video.mp4> [--env ...]      chunked upload to the drafts inbox (video.upload)
  tools/publish/tt.py status <publish_id> [--env ...]     processing state of an upload

A TikTok for Developers app (and its sandbox): Desktop, Login Kit + Content Posting API, scopes user.info.basic + video.upload, redirect http://localhost:8765/callback. Keys in
~/.config/tiktok/client.json ({"sandbox": {client_key, client_secret}, "production": {...}}, mode 600); tokens in
~/.config/tiktok/token_<env>.json. Default env: sandbox until the production app passes review.
"""
import argparse, base64, hashlib, http.server, json, os, secrets, sys, threading, time, urllib.error, urllib.parse, urllib.request, webbrowser
from pathlib import Path

CFG = Path.home() / ".config" / "tiktok"
REDIRECT = "http://localhost:8765/callback"
SCOPES = "user.info.basic,video.upload"
API = "https://open.tiktokapis.com/v2"
MB = 1024 * 1024


def client(env):
    c = json.loads((CFG / "client.json").read_text())[env]
    return c["client_key"], c["client_secret"]


def write_private(path, obj):
    CFG.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)


def post_form(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(),
                                 headers={"Content-Type": "application/x-www-form-urlencoded", "Cache-Control": "no-cache"})
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        sys.exit(f"{url} -> {e.code}: {e.read().decode()[:400]}")


def save_token(env, t):
    if "access_token" not in t:
        sys.exit(f"token error: { {k: v for k, v in t.items() if 'token' not in k} }")
    t["expires_at"] = time.time() + int(t.get("expires_in", 86400))
    write_private(CFG / f"token_{env}.json", t)


def access_token(env):
    t = json.loads((CFG / f"token_{env}.json").read_text())
    if t.get("expires_at", 0) > time.time() + 120:
        return t["access_token"]
    key, sec = client(env)
    r = post_form(f"{API}/oauth/token/", {"client_key": key, "client_secret": sec, "grant_type": "refresh_token", "refresh_token": t["refresh_token"]})
    save_token(env, r)
    return r["access_token"]


def api(env, method, path, body=None, query=""):
    req = urllib.request.Request(f"{API}{path}{query}", method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Bearer " + access_token(env), "Content-Type": "application/json; charset=UTF-8"})
    try:
        r = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> {e.code}: {e.read().decode()[:500]}")
    err = r.get("error", {})
    if err.get("code") not in (None, "ok"):
        sys.exit(f"{path}: {err}")
    return r.get("data", r)


def cmd_auth(a):
    key, sec = client(a.env)
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).decode().rstrip("=")
    challenge = hashlib.sha256(verifier.encode()).hexdigest()  # TikTok desktop PKCE: hex, not base64url
    state = secrets.token_urlsafe(16)
    url = "https://www.tiktok.com/v2/auth/authorize/?" + urllib.parse.urlencode({
        "client_key": key, "response_type": "code", "scope": SCOPES, "redirect_uri": REDIRECT, "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256"})
    got = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path.rstrip("/") != "/callback":
                self.send_response(404); self.end_headers(); return
            got.update({k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()})
            self.send_response(200); self.end_headers()
            self.wfile.write(b"tt: authorized, you can close this tab.")

        def log_message(self, *x):
            pass

    srv = http.server.HTTPServer(("localhost", 8765), H)
    threading.Thread(target=lambda: [srv.handle_request() for _ in range(5)], daemon=True).start()
    print(f"opening TikTok's consent page ({a.env}); sign in if asked, then Authorize")
    webbrowser.open(url)
    for _ in range(1200):
        if "code" in got or "error" in got:
            break
        time.sleep(0.5)
    if got.get("state") != state or "code" not in got:
        sys.exit(f"no code: { {k: v for k, v in got.items() if k != 'code'} or 'timed out'}")
    r = post_form(f"{API}/oauth/token/", {"client_key": key, "client_secret": sec, "code": got["code"], "grant_type": "authorization_code",
                                          "redirect_uri": REDIRECT, "code_verifier": verifier})
    save_token(a.env, r)
    print("authorized; scopes:", r.get("scope"))


def cmd_whoami(a):
    d = api(a.env, "GET", "/user/info/", query="?fields=open_id,display_name,avatar_url")
    u = d.get("user", d)
    print(u.get("display_name"), u.get("open_id"))


def cmd_upload(a):
    path = Path(a.file)
    size = path.stat().st_size
    if size <= 64 * MB:
        chunk, count = size, 1  # whole file in one request
    else:
        # at least 2 chunks of <= 64 MB; total_chunk_count = floor(size / chunk_size) and the last chunk takes the
        # remainder (TikTok allows up to 128 MB for it)
        count = max(2, -(-size // (64 * MB)))  # ceil, so size // count stays <= 64 MB
        chunk = size // count
        assert size // chunk == count and 5 * MB <= chunk <= 64 * MB, (size, chunk, count)
    d = api(a.env, "POST", "/post/publish/inbox/video/init/",
            {"source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk, "total_chunk_count": count}})
    pid, up = d["publish_id"], d["upload_url"]
    print(f"publish_id {pid}: {size / MB:.1f} MB in {count} chunk(s)")
    ctype = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}[path.suffix.lower()]
    with open(path, "rb") as f:
        for i in range(count):
            start = i * chunk
            end = size - 1 if i == count - 1 else start + chunk - 1
            f.seek(start)
            body = f.read(end - start + 1)
            req = urllib.request.Request(up, data=body, method="PUT", headers={
                "Content-Type": ctype, "Content-Length": str(len(body)), "Content-Range": f"bytes {start}-{end}/{size}"})
            try:
                code = urllib.request.urlopen(req).status
            except urllib.error.HTTPError as e:
                code = e.code
                if code not in (201, 206):
                    sys.exit(f"chunk {i + 1}/{count} -> {code}: {e.read().decode()[:300]}")
            print(f"  chunk {i + 1}/{count} -> {code}")
    for _ in range(60):
        s = api(a.env, "POST", "/post/publish/status/fetch/", {"publish_id": pid})
        print("  status:", s.get("status"), s.get("fail_reason") or "")
        if s.get("status") in ("SEND_TO_USER_INBOX", "PUBLISH_COMPLETE", "FAILED"):
            break
        time.sleep(5)
    print("in the TikTok app: Inbox -> the upload notification -> edit the caption -> Post")


def cmd_status(a):
    print(api(a.env, "POST", "/post/publish/status/fetch/", {"publish_id": a.publish_id}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sandbox", choices=["sandbox", "production"])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("auth").set_defaults(fn=cmd_auth)
    sp.add_parser("whoami").set_defaults(fn=cmd_whoami)
    x = sp.add_parser("upload"); x.add_argument("file"); x.set_defaults(fn=cmd_upload)
    x = sp.add_parser("status"); x.add_argument("publish_id"); x.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
