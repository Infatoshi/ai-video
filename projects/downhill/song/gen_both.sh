#!/bin/bash
# downhill: generate both caption runs in one GPU lease (run from ~/dev/cuda-song)
set -u
cd ~/dev/cuda-song
export PATH=$HOME/.local/bin:$PATH
uv run --project ace python downhill/gen_ts.py gen downhill/runs/capA.json > downhill/gen_capA.log 2>&1
uv run --project ace python downhill/gen_ts.py gen downhill/runs/capB.json > downhill/gen_capB.log 2>&1
echo GEN_DONE
