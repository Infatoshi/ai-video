#!/usr/bin/env bash
# tools/new_project.sh <name> [template=next-token]: start a new video from the latest project's engine and kit.
# Copies the template's tracked files minus its song, data, scenes and story, then lists what to write next.
set -euo pipefail
NAME=$1; FROM=${2:-next-token}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="projects/$FROM"; DST="$ROOT/projects/$NAME"
[ -e "$DST" ] && { echo "$DST exists"; exit 1; }
cd "$ROOT"
git ls-files "$SRC" | grep -v -E "^$SRC/(data/|audio/|lyrics/|llm/|song/lyrics|song/runs/|app/public/plates|publish\.json|README\.md|SPEC\.md|DEVLOG\.md|AGENTS\.md)" \
  | grep -v -E "^$SRC/app/src/scenes/[^_]" | grep -v -E "^$SRC/app/src/scenes/_tower\.ts" | while read -r f; do
  mkdir -p "$DST/$(dirname "${f#$SRC/}")"; cp -P "$f" "$DST/${f#$SRC/}"
done
mkdir -p "$DST"/{data,audio,lyrics,song/runs}
ln -sfn ../../audio "$DST/app/public/audio"; ln -sfn ../../data "$DST/app/public/data"
cat > "$DST/SPEC.md" <<EOF
# $NAME

The brief (Elliot's approved prompt, verbatim), then the topic arc, song, look, rules and shot plan.
See ../$FROM/SPEC.md for the shape. Real numbers only: name the source of every number on screen.

>>> NEXT
Brief not written yet.
EOF
printf "# DEVLOG (times Mountain)\n\n## %s: %s\n\n- Forked from %s (engine + kit).\n" "$(date +%F)" "$NAME" "$FROM" > "$DST/DEVLOG.md"
printf "# %s agent notes\n\nRead ../../AGENTS.md (the workflow) first, then SPEC.md.\n" "$NAME" > "$DST/AGENTS.md"
printf "# %s\n\nNot written yet: the pitch, the real data, the song, preview and render (see ../%s/README.md).\n" "$NAME" "$FROM" > "$DST/README.md"
echo "created $DST. Still project-specific, rewrite before rendering:"
echo "  analysis/common.py (SONG entry), analysis/analyze.py (SECTION_BARS for the new take), app/src/timeline.ts,"
echo "  app/src/words.ts (slams), app/src/story.ts + engine/llm.ts + engine/hud.ts readout (the data the overlay shows),"
echo "  song/lyrics.txt + lyrics.sing.txt + runs/*.json, publish.json (tools/publish/yt.py), the song file name"
echo "  (audio/token.mp3 in app/src/main.ts, app/scripts/render.ts, app/scripts/render-parallel.sh + its OUTNAME default),"
echo "  app/index.html <title> and render.ts's page-title check + default port (forks of the course episodes)."
echo "  Then: cd app && bun install."
