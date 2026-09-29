# tokens agent notes

Read ../../AGENTS.md (the workflow and "The course") first, then SPEC.md (brief, arc, data, shot plan) and DEVLOG.md.

"Tokens" (ML, slowly #2/5), song "Never the Letters": the model gets numbers, not letters. One scene for the whole
song (`app/src/scenes/world.ts` + `world-draw.ts`): one sheet, one camera, no cuts; `thumb.ts` for thumbnails.

## Data (data/tok.json, derived numbers only; the gated tokenizer stays in the GPU host's HF cache)
- `tok/llama_tok.py` (the GPU host, CPU) -> out/llama_tok.json; `tok/spell_test.py` (the GPU host's 3090, under the lease) ->
  out/spell_test.json; `tok/bpe.py` (BPE on song/lyrics.sing.txt); `python3 tok/build_data.py` assembles tok.json.
- `tok/chapters.py` -> data/timeline.json (YouTube chapters; the app timeline is one entry). `tok/pace.py`: a
  take's projected words/min from Whisper word times.

## Commands (from app/)
- render.ts never uses a shared vite server unless `--url` names it (every fork defaults to :5173); it starts a
  private one and checks the page says `project: 'tokens'`.
- Stills / sheets: `bun scripts/render.ts stills --t 30,150 [--portrait] --out ../out/wip/x`.
- Finals: `../../../tools/lock.sh render scripts/render-parallel.sh 4 12 --samples auto --max-samples 108 --shutter 0.2`
  (+ `OUTNAME=tokens_vertical ... --portrait`), ~6 min each. Never edit app/src while a render runs (each segment
  job loads the code when it starts).
- Gate: `uv run --offline ../../tools/gate/gate.py projects/tokens/out/final/tokens.mp4` (from the repo root).
- Alignment on the GPU host's CPU (no lease): `UV_OFFLINE=1 CUDA_VISIBLE_DEVICES= DEMUCS_DEV=cpu analysis/run_linux.sh`,
  or `analysis/align_only.sh` once stems, emissions and a Whisper word list exist.
