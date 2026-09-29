#!/usr/bin/env bash
# Per-beat contact sheets (SPEC.md "Gates"): one frame on every beat of the song, 48 per sheet
# (8 x 6), at 1 sample, into out/qa/beats/. Empty or weak moments can't hide between scene samples.
#   scripts/beatsheet.sh [--portrait]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=../out/qa/beats; if [ "${1:-}" = "--portrait" ]; then OUT=../out/qa/beats_portrait; fi
mkdir -p "$OUT"
python3 -c "
import json; b = json.load(open('../data/audio.json'))['beats']
b = [round(x + 0.02, 3) for x in b]  # just after the beat: the hit has landed
for i in range(0, len(b), 48): print(','.join(map(str, b[i:i + 48])))" | {
  n=0
  while read -r times; do
    bun scripts/render.ts sheet --times "$times" --cols 8 --out "$OUT/beats_$(printf %02d $n).png" "$@" > /dev/null
    echo "$OUT/beats_$(printf %02d $n).png"
    n=$((n + 1))
  done
}
