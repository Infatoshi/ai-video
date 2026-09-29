#!/bin/bash
# tokens song stage on the GPU host, one GPU lease: generate both captions (8 seeds each), score (Demucs -> Whisper
# recall + Audiobox), rhythm-rank. Copy of this file lives in ~/dev/cuda-song/tokens/; logs next to it.
cd ~/dev/cuda-song
export PATH=$HOME/.local/bin:$PATH
L=tokens
uv run --project ace python gen.py gen tokens/m1_capA.json > $L/gen_m1A.log 2>&1
uv run --project ace python gen.py gen tokens/m1_capB.json > $L/gen_m1B.log 2>&1
uv run --project score python score.py out/tokens-m1_capA out/tokens-m1_capB > $L/score_m1.log 2>&1
uv run --project score python rhythm.py out/tokens-m1_capA > $L/rhythm_m1A.log 2>&1
uv run --project score python rhythm.py out/tokens-m1_capB > $L/rhythm_m1B.log 2>&1
echo CHAIN_DONE >> $L/score_m1.log
