#!/bin/bash
# downhill v2 (an instrumental break after verse 3 and after verse 4): both captions in one GPU lease
set -u
cd ~/dev/cuda-song
export PATH=$HOME/.local/bin:$PATH
for r in capA2 capB2; do
  [ -f downhill/runs/$r.json ] && uv run --project ace python downhill/gen_ts.py gen downhill/runs/$r.json > downhill/gen_$r.log 2>&1
done
echo GEN_DONE
