#!/usr/bin/env python3
"""schedule_comments: one-shot launchd jobs that post each variant's approved pinned_comment 3 minutes after its
scheduled release (publish.json publish_at) and pin it (tools/publish/release_comment.sh). Only for comments Elliot
approved word for word. Each job removes itself after it runs.

  tools/publish/schedule_comments.py <project>:<variant> [...]
  tools/publish/schedule_comments.py --list            (pending jobs)
"""
import json, os, plistlib, subprocess, sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
LA = Path.home() / "Library" / "LaunchAgents"


def main(args):
    if args == ["--list"]:
        for p in sorted(LA.glob("com.aivideo.comment.*.plist")):
            d = plistlib.loads(p.read_bytes())
            print(p.name, d["StartCalendarInterval"], " ".join(d["ProgramArguments"][-2:]))
        return
    for a in args:
        proj, var = a.split(":")
        v = json.loads((REPO / "projects" / proj / "publish.json").read_text())["variants"][var]
        at = datetime.fromisoformat(v["publish_at"].replace("Z", "+00:00")).astimezone(ZoneInfo("America/Edmonton")) + timedelta(minutes=3)
        label = f"com.aivideo.comment.{var}"
        pl = LA / f"{label}.plist"
        pl.write_bytes(plistlib.dumps({
            "Label": label,
            "ProgramArguments": ["/bin/bash", str(REPO / "tools/publish/release_comment.sh"), proj, var],
            "StartCalendarInterval": {"Month": at.month, "Day": at.day, "Hour": at.hour, "Minute": at.minute},
            "EnvironmentVariables": {"PATH": f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"},
            "StandardOutPath": str(REPO / "out/review" / f"comment_{var}.launchd.log"),
            "StandardErrorPath": str(REPO / "out/review" / f"comment_{var}.launchd.log"),
        }))
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(pl)], capture_output=True)
        r = subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(pl)], capture_output=True, text=True)
        print(f"{var}: {at:%a %b %-d %-I:%M %p} MT", "loaded" if r.returncode == 0 else f"FAILED {r.stderr.strip()}")


if __name__ == "__main__":
    main(sys.argv[1:])
