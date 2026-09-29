#!/usr/bin/env bash
# Full lyric alignment + music analysis on a Linux/CUDA box (the GPU host), using the song scoring venv
# (~/dev/cuda-song/score: torch, torchaudio, transformers, demucs, librosa) plus a few extras.
#   analysis/run_linux.sh            # from the repo root; SONG is set in analysis/common.py
set -euo pipefail
cd "$(dirname "$0")"
PY=(uv run --project "$HOME/dev/cuda-song/score" --with numba --with matplotlib --with soxr --with scipy python)
SONG=$(python3 -c "import re;print(re.search(r'^SONG = \"(\w+)\"', open('common.py').read(), re.M)[1])")
AUDIO=$(python3 -c "import re;s=open('common.py').read();print(re.search(r'\"$SONG\": dict\(audio=\"([^\"]+)\"', s)[1])")
STEM=$(basename "$AUDIO" .wav)
mkdir -p stems/htdemucs_ft work/"$SONG"
if [ ! -f "stems/htdemucs_ft/$STEM/vocals.wav" ]; then
  TORCH_HOME="$PWD/.cache/torch" "${PY[@]}" -m demucs -n htdemucs_ft -d cuda -o stems "../$AUDIO"
fi
ffmpeg -loglevel error -y -i "stems/htdemucs_ft/$STEM/vocals.wav" -ac 1 -ar 16000 "work/$SONG/vocals16k.wav"
"${PY[@]}" ctc_emissions.py vocals vocL vocR
"${PY[@]}" vocal_feats.py
"${PY[@]}" whisper_run.py turbo
"${PY[@]}" align.py --plots
"${PY[@]}" analyze.py
echo ANALYSIS_DONE
