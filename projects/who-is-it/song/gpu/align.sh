#!/usr/bin/env bash
# Stage 3 on the GPU host (under the GPU lease): Demucs, CTC, Whisper, align, analyze for SONG in analysis/common.py.
set -euo pipefail
cd ~/dev/ai-video/projects/who-is-it/analysis
bash run_linux.sh
