# who-is-it agent notes

Read ../../AGENTS.md (the workflow) first, then SPEC.md. "Who's It?" is ML, slowly #4/5 (attention): one continuous
scene (`app/src/scenes/sentence.ts`) over the real attention of Llama 3.1 8B on two sentences.

## Data
- `llm/attn.py` (the GPU host, score venv, under the GPU lease; 21 s): every layer x head's attention for 5 sentence pairs
  -> out/llm/attn.npz + attn_meta.json. `uv run llm/flip.py out/llm --json data/attn.json` (Mac): the flip census and
  what the app reads. `FEATURED` in flip.py picks the pair/head the video shows (brief, L8 H29).

## Song and alignment
- Lyrics `song/lyrics.sing.txt`; runs `song/runs/w1_cap{A,B}.json` (the GPU host copies in ~/dev/cuda-song/who-is-it/,
  outputs out/who-is-it-w1_cap*/); chain `song/gpu/chain_w1.sh`. Pick: w1_capA/403 -> audio/whoisit.{wav,mp3}.
- Alignment ran on the Mac (tools are Mac-native; models cloned from ../cuda-roofline/analysis/.cache):
  `cd analysis && TORCH_HOME=$PWD/.cache/torch uv run python -m demucs -n htdemucs_ft -d mps -o stems ../audio/whoisit.wav`,
  ffmpeg vocals16k, then `uv run python ctc_emissions.py vocals vocL vocR; uv run python vocal_feats.py;
  uv run python whisper_run.py turbo; uv run python align.py --plots; uv run python analyze.py` (pins in align.py
  under `SONG == "whoisit"`, sections in analyze.py `SECTION_BARS_WHOISIT`). After any re-alignment re-check the
  busiest 30 s: it is exactly 120 words/min, the course ceiling.

## Render and gate (from app/)
- Stills/sheets: `bun scripts/render.ts stills --t 140 [--portrait] --out ../out/wip/x`. Don't render on :5173
  (another project may own it): pass `--url` to this project's own server (`PDOOM_NO_HMR=1 bunx vite --port 5947`).
- Finals: `../out/final/render_both.sh` (both aspects behind `tools/lock.sh render`, `--samples 1`: the 2D layer is
  drawn once per frame). Never edit app code while a render runs (later segments load the new code).
- Gate: `uv run ../../tools/gate/gate.py projects/who-is-it/out/final/who_is_it.mp4` (and `_vertical`) from the repo root.
- Thumbnails: `bun scripts/render.ts stills --only thumbA,thumbB,thumbC --t 0.5,1.5,2.5 --out ../out/publish/thumbs`.
- Hook Short: `python3 tools/publish/cut.py who-is-it who_is_it_hook`; metadata `python3 tools/publish/yt.py package who-is-it`.
