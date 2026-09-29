#!/usr/bin/env bash
# downhill: the same alignment + analysis as run_linux.sh, on the Mac (Apple GPU for Demucs, mlx-whisper for Whisper,
# CPU for the CTC models), used while the GPU host's single GPU was queued four agents deep. SONG is set in common.py.
#   analysis/run_mac.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=(uv run --project . python)
SONG=$(python3 -c "import re;print(re.search(r'^SONG = \"(\w+)\"', open('common.py').read(), re.M)[1])")
AUDIO=$(python3 -c "import re;s=open('common.py').read();print(re.search(r'\"$SONG\": dict\(audio=\"([^\"]+)\"', s)[1])")
STEM=$(basename "$AUDIO" .wav)
mkdir -p stems/htdemucs_ft work/"$SONG"
if [ ! -f "stems/htdemucs_ft/$STEM/vocals.wav" ]; then
  TORCH_HOME="$PWD/.cache/torch" "${PY[@]}" -m demucs -n htdemucs_ft -d mps -o stems "../$AUDIO"
fi
ffmpeg -loglevel error -y -i "stems/htdemucs_ft/$STEM/vocals.wav" -ac 1 -ar 16000 "work/$SONG/vocals16k.wav"
"${PY[@]}" ctc_emissions.py vocals vocL vocR
"${PY[@]}" vocal_feats.py
"${PY[@]}" whisper_run.py turbo
"${PY[@]}" align.py --plots
"${PY[@]}" analyze.py
echo ANALYSIS_DONE
