#!/bin/bash
# Pull the run, the song chain's outputs and the alignment back from the GPU host (after train/after_train.sh), then
# export the run into the app. Run from the project root: bash train/fetch.sh
set -eu
source "$(git rev-parse --show-toplevel)/tools/hosts.sh"
R=$GPU:$REPODIR/projects/learning-to-write
S=$GPU:$SONGDIR
mkdir -p train/run song/takes audio lyrics data
scp -q $R/train/run/run.json $R/train/run/probe.json train/run/
scp -q $R/song/excerpts.json $R/song/lyrics.sing.txt song/
scp -q "$S/out/learning-to-write-capA/scores.json" song/takes/scores_capA.json
scp -q "$S/out/learning-to-write-capB/scores.json" song/takes/scores_capB.json
scp -q "$S/out/learning-to-write-capA/rhythm.json" song/takes/rhythm_capA.json || true
scp -q "$S/out/learning-to-write-capB/rhythm.json" song/takes/rhythm_capB.json || true
scp -q "$S/learning-to-write/pick.txt" "$S/learning-to-write/judge.log" song/takes/ || true
scp -q $R/song/takes/pick_raw.wav $R/song/takes/pick.words.json song/takes/ || true
scp -q $R/audio/write.wav audio/ && ffmpeg -hide_banner -loglevel error -y -i audio/write.wav -c:a libmp3lame -b:a 320k audio/write.mp3
scp -q $R/lyrics/write.src.js lyrics/ || true
rsync -a $R/data/ data/ || true
python3 train/export.py
