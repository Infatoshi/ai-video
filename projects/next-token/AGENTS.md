# next-token agent notes

The workflow and shared tools are in ../../AGENTS.md; this file is what is specific to this video.

"Next Token": a drum and bass music video about how an LLM writes, printed as a risograph. Forked from
`../cuda-roofline` (engine from mexicat/pdoom-video, MIT). Mac-native: renders use the local Chrome with
Metal (`--use-angle=metal`). `SPEC.md` is the brief, design system, rules and shot plan; `docs/ENGINE.md`
is the engine contract (its top section covers this project's changes).

## Commands (from `app/`)
- Deps: `bun install`. Don't keep a shared no-HMR server on :5173 while scenes are being added (its
  `import.meta.glob` list freezes at startup); render.ts starts a private server when none is reachable.
- Stills: `bun scripts/render.ts stills --t 12.5,13.0 --only tokens --out ../out/wip/tokens` (add
  `--portrait` for the 9:16 frame). Sheet: `... sheet --from 1 --to 9 --n 16 --cols 4 --only tokens --out ...`.
- Perf: `bun scripts/render.ts perf --from 3 --to 5 --only tokens` (target < 30 ms/frame at 1 sample).
- Per-beat sheets: `scripts/beatsheet.sh [--portrait]` -> `out/qa/beats*/`. Thumbnails: `bun scripts/render.ts stills --t 0.5,1.5,2.5 --only thumbA,thumbB,thumbC --out ../out/publish/thumbs`.
- Full render: `scripts/render-parallel.sh 4 12 --samples auto --max-samples 36 --shutter 0.2` ->
  `out/final/next_token.mp4`; vertical: `OUTNAME=next_token_vertical scripts/render-parallel.sh ... --portrait`.
- Activity gate: `uv run ../../tools/gate/gate.py out/final/next_token.mp4 [--vs other.mp4]`.

## Song (ACE-Step 1.5 on the GPU host)
- Lyrics: `song/lyrics.txt` (annotated) and `song/lyrics.sing.txt` (what the model sings; on the GPU host it is
  `~/dev/cuda-song/lyrics.nt.sing.txt`). Runs: `song/runs/n1_capA.json` (rapped verses), `n1_capB.json`
  (sung verses), 174 BPM, F minor, 170 s, seeds 801-808 / 901-908.
- the GPU host `~/dev/cuda-song` (copies of `../../tools/song/*.py`): `gen.py` (ace venv, `"offload": true` on the 24 GB 3090), `score.py` (Demucs ->
  Whisper large-v3 recall + Audiobox), `rhythm.py` (rhyme placement; tempo search within 5% of the take's
  BPM so 174 can't lock to half-time). GPU work goes through the lease:
  `~/bin/overnight-compute run --agent <unique-name> --resource gpu0 --ttl 90m --poll 15s -- env CUDA_VISIBLE_DEVICES=0 <cmd>`.
- Model run: `llm/dump.py` (copied to the GPU host as `~/dev/cuda-song/dump_llm.py`, score venv, HF cache
  `~/.cache/huggingface`) -> `data/llm.json`.
- Alignment: the GPU host `~/dev/ai-video/projects/next-token/analysis/run_linux.sh` (SONG = "token" in `common.py`; audio
  `audio/token.wav`, lyric seed times `lyrics/token.src.js` from `tools/song/pick.sh next-token <run> <seed>`); copy `data/` back.
  Sections: `SECTION_BARS_TOKEN` in `analyze.py` (names match the kit's `CUT_EVERY`).

## Rules
- Scene authors edit only `app/src/scenes/<name>.ts` and `<name>-*.ts`. Kit, tower, engine, HUD, story,
  words and timeline changes go through the lead.
- Every number on screen comes from `data/llm.json` or arithmetic on it (labelled), or a cited spec.
- Any render step over ~2 minutes: find out why before rerunning.

## Publish
- `publish.json` + `../../tools/publish/yt.py` (package / upload / update / status). Uploaded PRIVATE on Elliotcodes:
  next_token https://youtu.be/OK-Rf8VqfBQ, next_token_vertical https://youtu.be/MoB0stv1vKg (a Short).
- Thumbnails: scenes/thumb.ts (A strawberry "AI: 2" is on the video; B die 2|3 69.7%, C "HOW AI WRITES" for Test & Compare).
