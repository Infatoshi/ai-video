#!/usr/bin/env bash
# Round 1 on the GPU host (under one GPU lease): generate both captions, then score (Demucs -> Whisper recall,
# Audiobox) and rhythm. Run from ~/dev/cuda-song.
set -euo pipefail
cd ~/dev/cuda-song
uv run --project ace python gen.py gen who-is-it/w1_capA.json
uv run --project ace python gen.py gen who-is-it/w1_capB.json
uv run --project score python score.py out/who-is-it-w1_capA out/who-is-it-w1_capB
uv run --project score python rhythm.py out/who-is-it-w1_capA
uv run --project score python rhythm.py out/who-is-it-w1_capB
echo CHAIN_DONE
