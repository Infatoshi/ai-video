#!/usr/bin/env bash
# tools/song/pick.sh <project> <run> <seed>: master a take and seed its lyric times, then stage it for alignment.
#   tools/song/pick.sh next-token n1_capA 803
# Masters to -14 LUFS integrated / -1 dBTP (two-pass loudnorm) -> <project>/audio/<song>.wav (24-bit) + .mp3, where
# <song> is SONG in <project>/analysis/common.py. Then run the alignment on the GPU host (AGENTS.md, stage 3).
# Check line 0's seed start afterwards: Whisper puts the first word at 0.0 over an instrumental intro.
set -euo pipefail
PROJ=$1; RUN=$2; SEED=$3
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
P="$ROOT/projects/$PROJ"
source "$ROOT/tools/hosts.sh"
SONG=$(python3 -c "import re;print(re.search(r'^SONG = \"(\w+)\"', open('$P/analysis/common.py').read(), re.M)[1])")
cd "$P"
mkdir -p song/takes audio lyrics
scp -q "$GPU:$SONGDIR/out/$RUN/$SEED.wav" song/takes/pick_raw.wav
scp -q "$GPU:$SONGDIR/out/$RUN/$SEED.words.json" song/takes/pick.words.json
M=$(ffmpeg -hide_banner -i song/takes/pick_raw.wav -af loudnorm=I=-14:TP=-1:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
mi=$(echo "$M" | python3 -c "import json,sys;d=json.load(sys.stdin);print(f\"measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:offset={d['target_offset']}\")")
ffmpeg -hide_banner -loglevel error -y -i song/takes/pick_raw.wav -af "loudnorm=I=-14:TP=-1:LRA=11:$mi:linear=true" -ar 44100 -c:a pcm_s24le "audio/$SONG.wav"
ffmpeg -hide_banner -loglevel error -y -i "audio/$SONG.wav" -c:a libmp3lame -b:a 320k "audio/$SONG.mp3"
ffmpeg -hide_banner -i "audio/$SONG.wav" -af loudnorm=I=-14:TP=-1:print_format=summary -f null - 2>&1 | grep -E "Input Integrated|Input True Peak"
python3 "$ROOT/tools/song/src_from_words.py" song/takes/pick.words.json song/lyrics.sing.txt "lyrics/$SONG.src.js" "$PROJ"
ssh "$GPU" "mkdir -p $REPODIR/projects/$PROJ/audio $REPODIR/projects/$PROJ/lyrics"
rsync -a "audio/$SONG.wav" "$GPU:$REPODIR/projects/$PROJ/audio/"
rsync -a "lyrics/$SONG.src.js" "$GPU:$REPODIR/projects/$PROJ/lyrics/"
rsync -a --exclude stems --exclude work --exclude qa --exclude .cache --exclude __pycache__ --exclude .venv analysis/ "$GPU:$REPODIR/projects/$PROJ/analysis/"
echo "picked $RUN/$SEED for $PROJ ($SONG)"
