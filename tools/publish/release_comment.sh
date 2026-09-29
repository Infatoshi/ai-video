#!/usr/bin/env bash
# tools/publish/release_comment.sh <project> <variant>: after a scheduled release, post the variant's approved
# pinned_comment (publish.json) as Elliotcodes and have Codex pin it. Run once per release by a one-shot launchd job
# (tools/publish/schedule_comments.py), which this script removes when it is done. Log: out/review/comment_<variant>.log
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJ=$1; VAR=$2
LOG="$ROOT/out/review/comment_$VAR.log"; mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "== $(date '+%F %T %Z') $PROJ/$VAR"
cd "$ROOT"
read -r VID TEXT < <(python3 -c "
import json; v=json.load(open('projects/$PROJ/publish.json'))['variants']['$VAR']; print(v['youtube_id'], v['pinned_comment'])")
# wait (up to 90 min) for the release to go public
for i in $(seq 1 90); do
  st=$(python3 -c "import sys; sys.path.insert(0,'tools/publish'); import yt; print(yt.call(yt.API + '/videos?part=status&id=$VID')['items'][0]['status']['privacyStatus'])")
  [ "$st" = public ] && break
  sleep 60
done
echo "status: $st"
if [ "$st" != public ]; then echo "not public, giving up (rerun: $0 $PROJ $VAR)"; exit 1; fi
# post once: skip if the channel already has this exact comment on the video
exists=$(python3 -c "
import sys; sys.path.insert(0,'tools/publish'); import yt
r = yt.call(yt.API + '/commentThreads?part=snippet&maxResults=100&videoId=$VID')
print(any(t['snippet']['topLevelComment']['snippet']['textOriginal'] == sys.argv[1] for t in r.get('items', [])))" "$TEXT")
[ "$exists" = True ] && echo "comment already there" || python3 tools/publish/yt.py comment "$VID" "$TEXT" --yes
# pin (UI only): Codex computer use
P=$(python3 - "$VID" "$TEXT" <<'PY'
import sys
t = open("tools/publish/studio_prompts/pin_comment.md").read()
print(t.replace("{VIDEO_ID}", sys.argv[1]).replace("{TEXT}", sys.argv[2]))
PY
)
"$HOME/.local/bin/codex" exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C "$ROOT" "$P" 2>&1 | awk '/^codex$/{f=1;next} f' | tail -4
# one-shot: remove this job
PL="$HOME/Library/LaunchAgents/com.aivideo.comment.$VAR.plist"
[ -f "$PL" ] && { launchctl bootout "gui/$(id -u)" "$PL" 2>/dev/null; rm -f "$PL"; echo "removed $PL"; }
