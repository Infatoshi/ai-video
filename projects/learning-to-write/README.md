# Watch It Learn to Write (ML, slowly #5 of 5)

Training: a small GPT guesses the next letter, is told how wrong it was, adjusts a little, and repeats. The choir sings the model's real output at each checkpoint, gibberish first and Shakespeare-ish last, and the video shows where it starts memorizing.

Not released yet.

## Real data

- `train/train.py` (GPU host) trains a 10.7M-parameter character-level GPT on tiny Shakespeare (5,000 steps, 3.3 minutes on an RTX 3090) and samples one fixed prompt at checkpoints; `train/probe.py` and `train/export.py` -> `data/run.json`, `data/probe.json`, `data/excerpts.json`.

Song: gospel, 80 BPM, one voice growing to a choir (`audio/write.mp3`), generated with ACE-Step 1.5 (run configs in `song/runs/`), with every sung word timed in
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
bun scripts/render.ts video --samples auto --max-samples 108 --shutter 0.2 --out ../out/final/learning_to_write.mp4
# or in parallel segments, then a lossless concat:
scripts/render-parallel.sh 4 12 --samples auto --max-samples 108 --shutter 0.2
OUTNAME=learning_to_write_vertical scripts/render-parallel.sh 4 12 --samples auto --max-samples 108 --shutter 0.2 --portrait
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
