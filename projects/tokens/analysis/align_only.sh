#!/usr/bin/env bash
# The last two alignment steps only (align.py + analyze.py), when the stems, CTC emissions, vocal features and a
# Whisper word list are already in work/<song>/ (tokens: the Whisper list is rhythm.py's GPU run on the take).
set -euo pipefail
cd "$(dirname "$0")"
PY=(uv run --offline --project "$HOME/dev/cuda-song/score" --with numba --with matplotlib --with soxr --with scipy python)
"${PY[@]}" align.py --plots
"${PY[@]}" analyze.py
echo ANALYSIS_DONE
