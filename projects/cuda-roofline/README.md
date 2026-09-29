# Chasing the Roofline

The chapters of *CUDA for Deep Learning*, in order, as a synth-pop music video. One orange thread (a single CUDA thread) multiplies into grids and warps, gets fed by memory, meets the roofline and ends up running on a hundred thousand GPUs.

Watch: https://youtu.be/_TJoJYLsEcQ

## Real data

- Numbers come from the book's chapters (`SPEC.md` has the plate-by-plate plan and sources).

Song: synth-pop, 133 BPM (`audio/roofline.mp3`), generated with ACE-Step 1.5 (run configs in `song/runs/`), with every sung word timed in
`data/lyrics.json` and the beat grid and sections in `data/audio.json`.

`SPEC.md` has the brief, the arc and the shot plan; `DEVLOG.md` is how it was made, stage by stage; `AGENTS.md` has
the project's notes for coding agents; `docs/ENGINE.md` documents the engine and scene API.

## Preview

```sh
cd app
bun install
bunx vite
```

Open http://localhost:5173. `?t=60` starts at 60 s. Space plays and pauses, the arrow keys seek (shift for
5 s), `,` and `.` step a frame, `[` and `]` jump between scenes, `l` loops the current scene, `h` hides the controls.

## Render

```sh
cd app
bun scripts/render.ts video --samples auto --max-samples 108 --shutter 0.2 --out ../out/final/roofline.mp4
# or in parallel segments, then a lossless concat:
scripts/render-parallel.sh 4 12 --samples auto --max-samples 108 --shutter 0.2
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
