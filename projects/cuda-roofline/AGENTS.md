# cuda-roofline agent notes

The workflow and shared tools are in ../../AGENTS.md (this project predates them: its song scripts now live in
../../tools/song, and the GPU host's copy of this folder is ~/dev/ai-video/projects/cuda-roofline).

Home: Mac-native - no The GPU host dependency; this folder is the source of truth. Renders use the
local Chrome with Metal (`--use-angle=metal`), which is why it runs on a Mac.

## Commands (from `app/`)
- Deps: `bun install`. Do not keep a shared no-HMR server on :5173 while scenes are being added: its
  `import.meta.glob` scene list is frozen at startup, so new scene files fail with "scene module not
  found". With :5173 down, render.ts starts a fresh private server per run (~1 s).
- Stills (then look at them): `bun scripts/render.ts stills --t 12.5,13.0 --only die --out ../out/wip/die`
- Sheet: `bun scripts/render.ts sheet --from 1 --to 9 --n 16 --cols 4 --only die --out ../out/wip/die/sheet.png`
- Motion check: `bun scripts/render.ts video --from 3 --to 6 --only die --out ../out/wip/die.mp4 --preset veryfast`
- Perf: `bun scripts/render.ts perf --from 3 --to 5 --only die` (target < 25 ms/frame at 1 sample).
- Typecheck one scene: `bunx tsc --noEmit -p tsconfig.json 2>&1 | grep scenes/die`
- Final: `bun scripts/render.ts video --samples auto --max-samples 108 --out ../out/cuda_pdoom.mp4`

## Song and analysis (v2)
- Song: `audio/roofline.wav` (master) and `.mp3` (what the app plays), take a4-401 from ACE-Step on
  the other GPU host (`~/dev/cuda-song`; tooling in `song/`: `gen.py`, `score.py`, `musical.py`, `runs/*.json`).
- `analysis/common.py` `SONG` picks the song ("roofline" writes `data/`).
  Order: Demucs htdemucs_ft stems into `analysis/stems/htdemucs_ft/roofline/` (ran on the other GPU host),
  `ctc_emissions.py vocals vocL vocR`, `vocal_feats.py`, `whisper_run.py turbo` (needs
  `work/roofline/vocals16k.wav`), `align.py --plots` (QA in `qa/roofline/`), `analyze.py`.

## v4 (beginner cut, current)
- Song: take b2-701 of `song/lyrics.sing.v4.txt`, generated and scored on **the GPU host** (the other GPU host was busy):
  `~/dev/cuda-song` there (ACE-Step XL-SFT + 4B LM with `"offload": true` for the 24 GB card),
  `score.py` (word recall + Audiobox), `rhythm.py` (rhyme/phrase placement on an onset-fitted grid).
- Alignment on <gpu_host>: `analysis/run_linux.sh` (Demucs, CTC, vocal feats, Whisper large-v3 via
  transformers, align, analyze) using the scoring venv; copy `data/` back. Pinned fixes for this take
  live in `align.py` (`ANCHORS`/`FIX` for lines 0, 10, 41) because the intro/break backing vocals fool it.
- the GPU host's GPU work goes through the lease tool: `~/bin/overnight-compute run --agent <unique-name>
  --resource gpu0 --ttl 30m --poll 15s -- env CUDA_VISIBLE_DEVICES=0 <cmd>` (agent names can't be reused).
  Copy between GPU hosts over the LAN, not a VPN relay (~150 ms).
- Scenes follow the SPEC.md "v4" rules; `scenes/_lock.ts` has the beat-lock helpers.

## Rules
- `docs/ENGINE.md` is the engine contract; `SPEC.md` is the treatment and plate table.
- Scene authors edit only `app/src/scenes/<name>.ts` and `<name>-*.ts`. Engine and timeline
  changes go through the lead.
- Chapter facts come from `~/dev/cuda/cuda-book/manuscript/CHnn_*.adoc` (read-only; never edit the book).
- The original video's scenes and treatment are in github.com/mexicat/pdoom-video (not copied here): technique reference only.
- Any render step over ~2 minutes: find out why (per-frame ms, sub-frame counts, serial
  bottleneck) and fix it before rerunning.
