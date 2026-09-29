# ai-video

Music videos that teach machine learning, made end to end by coding agents.

Each video is an original song about one idea, with real numbers on screen (a recorded model run, a real
tokenizer, a real training run) and code-rendered visuals. Coding agents (Claude Code, with Codex for the few
clicks no API offers) write the lyrics, generate and score the song, record the data, align every sung word,
build the scenes, render the video in 16:9 and native 9:16, and prepare the upload. A person approves the
brief, the song and anything posted.

## The videos

| Video | Teaches | Real data on screen | Song |
|---|---|---|---|
| [Chasing the Roofline](https://youtu.be/_TJoJYLsEcQ) | GPU programming, chapter by chapter of *CUDA for Deep Learning* | the book's numbers | synth-pop, 133 BPM |
| [Next Token](https://youtu.be/OK-Rf8VqfBQ) (Sep 29) | how an LLM writes, one token at a time | a recorded Llama 3.1 8B run: "There are 2 r's in the word 'strawberry'" (the 2 won a 69.7% roll) | drum and bass, 174 BPM |

**ML, slowly** is a five-episode course for beginners, after viewers said the first two were too fast: one
idea per video, about half the sung words per minute, one continuous scene instead of hundreds of cuts, and a
chorus that repeats the definition. Not released yet.

| # | Episode | Teaches | Real data on screen | Song |
|---|---|---|---|---|
| 1 | Downhill | how a model learns (gradient descent) | a 337-weight network learning two spirals, every step recorded | jazz waltz, 90 BPM |
| 2 | Tokens | why a model can't count the r's in strawberry | the Llama 3.1 tokenizer (" strawberry" is token 73700); Llama 3.1 8B asked 100 times | Motown call-and-response, 102 BPM |
| 3 | Meaning Is a Direction | embeddings | Llama 3.1 8B's embedding table: king - man + woman lands on King, KING, King, then queen | dream pop, 85 BPM |
| 4 | Who's It? | attention | all 1,024 attention heads of Llama 3.1 8B on one sentence and its twin | neo-soul duet, 75 BPM |
| 5 | Watch It Learn to Write | training | a 10.7M-parameter GPT trained from scratch on Shakespeare in 3.3 minutes; the choir sings its real output | gospel, 80 BPM |

## How a video is made

1. **Brief.** One idea, a story in real data, a genre, a look.
2. **Song.** Lyrics with every number real. [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) generates
   takes on a GPU box; `tools/song/` scores them (Demucs vocals, Whisper large-v3 word recall, Audiobox
   Aesthetics, whether the words sit on the beat), masters the pick to -14 LUFS and stages the runners-up for a
   listen.
3. **Real data.** Record the thing the video explains: model internals, a tokenizer, a training run.
4. **Align.** Demucs stems, CTC forced alignment cross-checked with Whisper, beat and section analysis: every
   sung word gets a time (`data/lyrics.json`, `data/audio.json`).
5. **Build.** Scenes in TypeScript + three.js, each a pure function of song time, cut on the beat grid.
6. **Gate and render.** Headless Chrome renders every frame deterministically with adaptive motion blur.
   `tools/gate/gate.py` measures each second (motion, detail, cuts; for the course, sung words per minute and
   shot length).
7. **Publish.** Thumbnails, titles, chapters and captions from `publish.json`; `tools/publish/yt.py` uploads and
   schedules, `cut.py` cuts a beat-aligned Short, `review.py` builds a local review page, `tt.py` sends to
   TikTok drafts.
8. **Measure.** Views and retention from the YouTube Analytics API.

[`AGENTS.md`](AGENTS.md) is the full workflow, written for the agents that run it. Each project's `DEVLOG.md` is
the record of how that video was actually made, stage by stage, with times and numbers.

## Layout

```
projects/<name>/     one video: app/ (engine + scenes), analysis/, song/, data/, audio/, lyrics/,
                     publish.json, SPEC.md (the brief and shot plan), DEVLOG.md, AGENTS.md
tools/song/          generate, score, check rhythm, master and pick song takes
tools/gate/          per-second activity and pace checks on a render
tools/publish/       YouTube and TikTok CLIs, Short cutter, review page, Studio prompts for Codex
tools/new_project.sh fork a new video from an existing one
tools/lock.sh        one full render at a time when several projects share a Mac
```

Each project keeps its own copy of the engine, so an old video still renders exactly as published after a
newer project changes the engine.

## Watch a project in your browser

Needs [bun](https://bun.sh), Google Chrome and ffmpeg.

```sh
cd projects/downhill/app
bun install
bunx vite                 # http://localhost:5173 (?t=60 starts at 60 s, ?aspect=portrait for 9:16)
bun scripts/render.ts video --samples auto --max-samples 108 --shutter 0.2 --out ../out/downhill.mp4
```

Space plays, the arrow keys seek, `[` and `]` jump between scenes, `h` hides the controls. The committed
`data/` and `audio/` files are all the renderer needs.

## Make a new one

The full pipeline wants a Mac (the renderer) and a Linux box with a 24 GB NVIDIA GPU for ACE-Step, Demucs,
Whisper and model runs, plus [uv](https://docs.astral.sh/uv/).

```sh
cp config.example.json config.json   # your GPU host's ssh name and paths
tools/new_project.sh my-video meaning
```

Then follow `AGENTS.md` stage by stage, or hand it to a coding agent.

## Credits

- Made with [Claude Code](https://claude.com/claude-code) (Claude Opus 5.5); Studio and TikTok clicks by Codex.
- The renderer began as a fork of [mexicat/pdoom-video](https://github.com/mexicat/pdoom-video) by Giacomo
  Magnanini (MIT).
- Songs: ACE-Step 1.5. Alignment and scoring: Demucs, Whisper large-v3, wav2vec2 CTC, Audiobox Aesthetics.
- **Built with Llama.** Next Token, Tokens, Meaning Is a Direction and Who's It? record Llama 3.1 8B Instruct
  (and Llama 3.2 1B as a speculative-decoding draft); data derived from them is under the Llama Community
  Licenses in [`LICENSES/`](LICENSES/).
- Fonts: Archivo, Cormorant Garamond and IBM Plex Mono (SIL Open Font License 1.1); EMS and Hershey
  single-stroke fonts (license notices inside each SVG).
- Training text for Watch It Learn to Write: Shakespeare (public domain), via tiny Shakespeare.

## License

The code is MIT ([`LICENSE`](LICENSE)). Fonts keep their own licenses, and Llama-derived data is under the
Llama 3.1 / 3.2 Community Licenses. The songs, lyrics and rendered videos are not covered by the MIT license.
