#!/usr/bin/env bash
# Full-song export as parallel segments, then a lossless concat and one audio mux.
#   scripts/render-parallel.sh [jobs=4] [segment_seconds=12] [extra render.ts args...]
# e.g. scripts/render-parallel.sh 4 12 --samples auto --max-samples 108 --shutter 0.2
# Segment boundaries are whole seconds, so frame indices (round(t*60)) line up exactly.
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${1:-4}; SEG=${2:-12}; shift 2 || true
EXTRA=("$@")
DUR=$(python3 -c "import json;print(json.load(open('../data/audio.json'))['duration'])")
NSEC=$(python3 -c "import math;print(math.ceil($DUR))")
OUT=../out/final; mkdir -p "$OUT/seg"; rm -f "$OUT"/seg/*.mp4 "$OUT"/seg/*.log
ranges=()
for ((s = 0; s < NSEC; s += SEG)); do
  e=$((s + SEG)); to=$e; [ "$e" -ge "$NSEC" ] && to=$DUR
  ranges+=("$s $to")
done
printf '%s\n' "${ranges[@]}" | xargs -P "$JOBS" -L 1 bash -c '
  f=$(printf "%06.2f" "$0")
  bun scripts/render.ts video --from "$0" --to "$1" --noaudio --out "'"$OUT"'/seg/seg_$f.mp4" '"${EXTRA[*]:-}"' \
    > "'"$OUT"'/seg/seg_$f.log" 2>&1 && echo "done $0-$1" || { echo "FAILED $0-$1"; exit 255; }'
ls "$OUT"/seg/seg_*.mp4 | sort | sed "s|^|file '$(pwd)/|; s|$|'|" > "$OUT/list.txt"
ffmpeg -y -loglevel error -f concat -safe 0 -i "$OUT/list.txt" -i ../audio/roofline.mp3 \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 320k -shortest -movflags +faststart "$OUT/cuda_roofline.mp4"
echo "wrote $OUT/cuda_roofline.mp4"
