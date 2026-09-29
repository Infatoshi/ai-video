#!/bin/bash
# Runs at the end of the GPU training run, inside the same lease (train.py calls it when it exists):
# fill the sung lyrics from the run's samples, then generate, score and rhythm-check the song takes.
set -u
export PATH=$HOME/.local/bin:$PATH
P=$HOME/dev/ai-video/projects/learning-to-write
S=$HOME/dev/cuda-song
mkdir -p $S/learning-to-write
python3 $P/song/make_sing.py $P/train/run/run.json $P/song/lyrics.sing.tmpl.txt $S/learning-to-write/lyrics.sing.txt $P/song/excerpts.json > $P/train/make_sing.log 2>&1 || { echo MAKE_SING_FAILED; exit 1; }
echo SING_READY
cd $S
for v in A B; do
  uv run --project ace python gen.py gen learning-to-write/cap$v.json > learning-to-write/gen_cap$v.log 2>&1
  echo GEN_cap${v}_DONE $(ls out/learning-to-write-cap$v/*.wav 2>/dev/null | wc -l)
done
uv run --project score python score.py out/learning-to-write-capA out/learning-to-write-capB > learning-to-write/score.log 2>&1
echo SCORE_DONE
for v in A B; do uv run --project score python rhythm.py out/learning-to-write-cap$v > learning-to-write/rhythm_cap$v.log 2>&1; done
echo RHYTHM_DONE
# pick a take by rule (song/judge.py: word recall on the fixed lines + how closely the excerpts' letters were sung;
# >= 210 s, 76-84 BPM), master it (-14 LUFS / -1 dBTP, as tools/song/pick.sh), seed the lyric times and run the
# alignment (Demucs, CTC, Whisper: GPU) in this same lease. A different pick later re-runs these steps.
python3 $P/song/judge.py out/learning-to-write-capA out/learning-to-write-capB --excerpts $P/song/excerpts.json --pick learning-to-write/pick.txt > learning-to-write/judge.log 2>&1 || { echo JUDGE_FAILED; exit 1; }
read RUN SEED < learning-to-write/pick.txt
echo PICKED $RUN $SEED
mkdir -p $P/audio $P/lyrics $P/song/takes
cp out/$RUN/$SEED.wav $P/song/takes/pick_raw.wav
cp out/$RUN/$SEED.words.json $P/song/takes/pick.words.json
cp learning-to-write/lyrics.sing.txt $P/song/lyrics.sing.txt
M=$(ffmpeg -hide_banner -i $P/song/takes/pick_raw.wav -af loudnorm=I=-14:TP=-1:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
mi=$(echo "$M" | python3 -c "import json,sys;d=json.load(sys.stdin);print(f\"measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:offset={d['target_offset']}\")")
ffmpeg -hide_banner -loglevel error -y -i $P/song/takes/pick_raw.wav -af "loudnorm=I=-14:TP=-1:LRA=11:$mi:linear=true" -ar 44100 -c:a pcm_s24le $P/audio/write.wav
python3 $P/out/src_from_words.py $P/song/takes/pick.words.json $P/song/lyrics.sing.txt $P/lyrics/write.src.js "Watch It Learn to Write" > $P/song/takes/src.log 2>&1
echo MASTERED
bash $P/analysis/run_linux.sh > $P/analysis/run_linux.log 2>&1 && echo ALIGN_DONE || echo ALIGN_FAILED
