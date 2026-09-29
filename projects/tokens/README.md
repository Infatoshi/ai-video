# Tokens (ML, slowly #2 of 5)

Why a model can't count the r's in strawberry: it never sees letters. Words are cut into chunks, every chunk becomes a number, and " strawberry" is one number, 73700.

Not released yet.

## Real data

- `tok/llama_tok.py` (the Llama 3.1 tokenizer, CPU): real splits and ids.
- `tok/spell_test.py` (Llama 3.1 8B, GPU): the strawberry question 100 times as one token (2 in 67 answers) and with the letters spaced out (3 in 90).
- `tok/bpe.py`: byte-pair encoding trained on the song's own lyrics, every merge in order (140 merges; the first is e+r).
- `tok/build_data.py` -> `data/tok.json`.

Song: Motown girl group, call-and-response, 102 BPM (`audio/tokens.mp3`), generated with ACE-Step 1.5 (run configs in `song/runs/`), with every sung word timed in
`data/lyrics.json` and the beat grid and sections in `data/audio.json`.

`SPEC.md` has the brief, the arc and the shot plan; `DEVLOG.md` is how it was made, stage by stage; `AGENTS.md` has
the project's notes for coding agents; `docs/ENGINE.md` documents the engine and scene API.

## Preview

```sh
cd app
bun install
bunx vite
```

Open http://localhost:5173. `?t=60` starts at 60 s. Add `?aspect=portrait` for the 9:16 cut. Space plays and pauses, the arrow keys seek (shift for
5 s), `,` and `.` step a frame, `[` and `]` jump between scenes, `l` loops the current scene, `h` hides the controls.

## Render

```sh
cd app
bun scripts/render.ts video --samples auto --max-samples 108 --shutter 0.2 --out ../out/final/tokens.mp4
# or in parallel segments, then a lossless concat:
scripts/render-parallel.sh 4 12 --samples auto --max-samples 108 --shutter 0.2
OUTNAME=tokens_vertical scripts/render-parallel.sh 4 12 --samples auto --max-samples 108 --shutter 0.2 --portrait
```

1920x1080 (or 1080x1920) at 60 fps. Every frame is a function of song time, so the export matches the preview.
Other modes: `stills`, `sheet` (contact sheets; `--cuts` for every scene boundary) and `perf`.

## Layout

- `app/`: the renderer (TypeScript + three.js, bun + Vite). `src/engine/` is the engine, `src/scenes/` the scenes,
  `src/timeline.ts` the edit, `scripts/render.ts` the headless-Chrome exporter.
- `analysis/`: Demucs stems, CTC forced alignment cross-checked with Whisper, beat and section analysis ->
  `data/lyrics.json`, `data/audio.json` (uv; runs on the GPU host).
- `song/`: lyrics, the sung-spelling file the generator sings from, ACE-Step run configs.
- `publish.json`: YouTube titles, descriptions, chapters and thumbnails for `tools/publish/yt.py`.

See the [top-level README](../../README.md) for credits and licenses.
