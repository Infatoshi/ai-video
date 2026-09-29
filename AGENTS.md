# ai-video agent notes

Code-rendered music videos that teach ML systems: an original song, real data on screen, one scene per idea,
cut to the beat, printed in both 16:9 and 9:16, published from the CLI. Each video is a workspace in
`projects/<name>/` (its own AGENTS.md, SPEC.md, DEVLOG.md, app, analysis config, song, data); shared tooling is in
`tools/`. Newest project = the template (`tools/new_project.sh <name>` forks it).

| project | song | status |
|---|---|---|
| `projects/cuda-roofline` | "Chasing the Roofline" (synth-pop), CUDA book chapters | v5 done; X 2026-09-27 12:29 am (378k impressions, reposted by Bryce Lelbach and Elon Musk); YouTube https://youtu.be/_TJoJYLsEcQ scheduled Mon Sep 28 8:00 am MT |
| `projects/next-token` | "Next Token" (174 BPM drum and bass), how an LLM writes, real Llama 3.1 8B run | v1 done; YouTube scheduled 8:00 am MT: https://youtu.be/OK-Rf8VqfBQ Tue Sep 29, hook Short https://youtu.be/JgXY8wtrp90 Wed Sep 30, Short https://youtu.be/MoB0stv1vKg Thu Oct 1 |
| `projects/downhill` | ML, slowly #1 "Downhill" (jazz waltz), how a model learns | v1 done 2026-09-28 (4:00, gate PASS); not uploaded |
| `projects/tokens` | ML, slowly #2 "Tokens" (Motown call-and-response), why it can't see the r's | v1 done 2026-09-28 (4:20, gate PASS); not uploaded; releases first |
| `projects/meaning` | ML, slowly #3 "Meaning Is a Direction" (dream pop), embeddings | v1 done 2026-09-28 (4:00, gate PASS); not uploaded |
| `projects/who-is-it` | ML, slowly #4 "Who's It?" (neo-soul duet), attention | v1 done 2026-09-28 (4:10, gate PASS); not uploaded |
| `projects/learning-to-write` | ML, slowly #5 "Watch It Learn to Write" (gospel), training a small GPT | v1 done 2026-09-28 (3:48, gate PASS); not uploaded |

## Machines
Host names and paths live in `config.json` (untracked; copy `config.example.json`). Scripts read it through
`tools/hosts.sh`; below, `<gpu_host>` means its `gpu_host`.
- **Mac** (this repo): the renderer (headless Chrome, Metal), ffmpeg, publishing. `cd projects/<p>/app && bun install`.
- **GPU host** (an RTX 3090 here, `ssh <gpu_host>`): song generation, scoring, alignment, model runs. On a shared box,
  run GPU work through a lease wrapper (here `overnight-compute run --agent <unique-name> --resource gpu0 --ttl 60m
  -- env CUDA_VISIBLE_DEVICES=0 <cmd>`); on your own box, run the command directly. Song runner: `gpu_song_dir` (an
  ACE-Step 1.5 checkout with `ace/` and `score/` uv venvs; the scripts are copies of `tools/song/*.py`, pushed with
  `scp`). Project mirrors for alignment: `gpu_repo_dir/projects/<p>/analysis`.

## The workflow (one project, stage by stage; each stage writes its outcome into the project's DEVLOG.md)
Priors that outlive one project live in `tools/priors.md` (read it at song-caption and scene-plan time).
0. **Brief.** Predict Elliot's prompt (topic arc, genre, voice, look, rules, deliverables); he approves; it goes
   verbatim into SPEC.md "The brief". Pick a topic with a story in real data (next-token: the model got the
   strawberry question wrong on a 69.7% roll).
1. **Song.** Lyrics topic-first, every number real, technical words' stresses on strong beats, a sung-spelling file
   (`song/lyrics.sing.txt`). Run configs in `song/runs/*.json` (two captions x 8 seeds). Captions name the harmony,
   default a repeating four-chord loop (I-V-vi-IV family) whose chorus order is the same every time; check candidates
   with musical.py's `4chords` column (tools/priors.md). On <gpu_host>:
   `gen.py gen runs/<run>.json` (ace venv), `score.py out/<run>` (Demucs -> Whisper large-v3 recall + Audiobox),
   `rhythm.py out/<run>` (rhymes on the grid; tempo within 5% of the requested BPM). Locally
   `uv run tools/song/structure.py <take.wav> <take.words.json> song/lyrics.sing.txt out.png` (drop, breakdown,
   ending). Pick by recall + rhythm + shape; stage the runners-up in `~/clips` for Elliot's ear.
   `tools/song/pick.sh <project> <run> <seed>` masters to -14 LUFS and seeds the lyric times.
2. **Real data.** Record the thing the video explains (next-token: `llm/dump.py` on the GPU host). Every on-screen number
   comes from it or is computed and labelled. Write the facts into SPEC.md.
3. **Align.** On <gpu_host>: `projects/<p>/analysis/run_linux.sh` under the lease (Demucs, CTC, Whisper, align, analyze);
   pin bad words with `FIX`/`ANCHORS` in align.py, sections with `SECTION_BARS_*` in analyze.py; copy `data/` back.
   Check against Whisper (median word offset ~0.06 s).
4. **Build.** Engine + kit first (overlay, shot cutter, materials, hero model), then a scene plan per lyric line in
   SPEC.md (written against tools/priors.md: the music-video grammar, which devices to use where), then scene authors
   in parallel (one brief each: rules, windows, lyric times, real data). Review every
   author's contact sheets; fix engine issues centrally.
5. **Gate and render.** `app/scripts/beatsheet.sh [--portrait]` (a frame per beat), `render.ts sheet --cuts`,
   `scripts/render-parallel.sh 4 12 --samples auto --max-samples 108 --shutter 0.2` (+ `OUTNAME=<p>_vertical ... --portrait`),
   `uv run tools/gate/gate.py projects/<p>/out/final/<p>.mp4` (activity per second; rework the worst seconds),
   share copies under X's 512 MB (`ffmpeg -crf 24/25`).
6. **Publish.** Thumbnails (3 variants, <= 3 big words, checked at 168x94; next-token: `render.ts stills --only thumbA,thumbB,thumbC`),
   `publish.json` (titles lead with the hook, chapters, tags, AI disclosure), then `tools/publish/yt.py package`,
   `upload` (PRIVATE), `update` (metadata edits, no re-upload), `playlist`. A 20-40 s hook Short: `tools/publish/cut.py`
   (beat-aligned segments of the vertical render, same bar position at every join). Series playlist: PLVnaIVGCV8DA
   "Music videos that teach ML" (private). X: `x-cli` only after Elliot approves
   the exact text. Elliot makes videos public in YouTube Studio.
   **Launch review:** `tools/publish/review.py <project>...` -> `out/review/launch.html` (thumbnails at feed and
   phone size, Test & Compare candidates, titles as they truncate, the two lines above the fold, links + live
   status, the decisions with copy buttons). A local HTML file only (it opens in the browser), never a Claude
   artifact. **Studio-only steps** go to Codex computer use:
   `codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C ~/dev/ai-video "$(cat tools/publish/studio_prompts/<step>.md)"`
   (`set_thumbnail.md`: swap a thumbnail when the API is locked out; `go_public.md`: visibility, playlist, Test &
   Compare, end screens, Shorts' related video, pin, only after Elliot's go on those exact videos).
   YouTube refuses Test & Compare, end screens and related-video links on private videos, and a staged Test &
   Compare set blocks API thumbnail changes (403 forbidden): stage tests only after going public.
   **Pinned comments:** YouTube refuses comments on private/scheduled videos (403). For comments Elliot approved
   word for word (`pinned_comment` per variant in publish.json): `tools/publish/schedule_comments.py <p>:<variant>...`
   loads one-shot launchd jobs 3 min after each release that post via the API and pin via Codex
   (`release_comment.sh`, logs in out/review/comment_<variant>.log); `--list` shows pending jobs.
7. **Measure.** `tools/publish/yt.py status|analytics`, `x-cli -j tweet get <id>` (public_metrics).

## The course: "ML, slowly" (working name; started 2026-09-28)
Beginners said the first two were too fast (replies on Next Token's X post). Measured: Next Token sings 169 words/min and cuts every 0.39 s (440 cuts in 170 s); Chasing the
Roofline 103 words/min, a cut every 0.87 s (218 in 190 s). The course slows down, one idea per video, the pieces
Next Token covered in 2.5 minutes; watched in order, Next Token (#6) is the recap and cuda-roofline is the hardware
epilogue. Elliot approved the five below on 2026-09-28 ("i trust your vision on these"). Playlist order below;
release order starts with tokens (it rides Next Token's strawberry hook).
1. `downhill`: learning = measure the error, find downhill, take a small step, repeat. A small network trained on
   two interleaved spirals (every step recorded) + a real 2D slice of its loss landscape (labelled a slice). One
   ball on one landscape while the network's boundary untangles beside it. Jazz waltz, ~90 BPM, 3/4 (6/8 or a
   swung 4/4 if the generator can't hold 3/4). The chorus is the algorithm.
2. `tokens`: the model never sees letters. The Llama 3.1 tokenizer's real split of the strawberry prompt, and a
   small BPE tokenizer trained on the song's own lyrics shown one merge at a time. One line of letters snapping
   into chunks. Motown girl group, call-and-response, ~100 BPM. Hook: the numbers the model gets instead of the word.
3. `meaning`: meaning is a direction. Llama 3.1 8B's embedding table (128,256 x 4,096), real nearest neighbours,
   the real king - man + woman result shown whatever it is. One continuous flight through the point cloud. Dream
   pop, ~85 BPM.
4. `who-is-it`: attention. Llama 3.1 8B on "The animal didn't cross the street because it was too tired" vs
   "...too wide": find a real head where "it" moves from animal to street; if none does it cleanly, show what the
   heads really do. One sentence with arcs sized by the real weights, morphing when the word changes. Neo-soul duet,
   ~75 BPM (one voice the word asking, the other the words answering).
5. `learning-to-write`: training. A small GPT trained from scratch on the GPU host on public-domain text, one fixed prompt
   sampled at checkpoints; the chorus lyrics are its real samples (gibberish first, words last) and the
   arrangement grows from one voice to a full choir as the loss falls. Gospel, ~80 BPM. Hook: "N minutes ago this
   model couldn't spell" (measured N). Hands off to Next Token.

**Pace rules** (the course gate enforces the measurable ones: `projects/<p>/gate.json` = {"profile": "course"}
switches `tools/gate/gate.py` from the calm-seconds floor to ceilings; exit 1 on a fail):
- 75-100 BPM, sung not rapped: <= 95 sung words/min over the song, <= 120 in the busiest 30 s.
- One continuous scene per section that transforms (morph, camera move, build-up) instead of cutting: <= 6 hard
  changes/min (cuts plus anything that pops in within a frame; fade or draw things in over >= 0.3 s), median shot
  >= 8 s. Something always moves gently; nothing frozen > 3 s (outside the last 5 s).
- The chorus is the episode's one definition: the same words every time over the same picture; each return adds
  one layer. Concrete example first, then the rule.
- One number on screen at a time, held until the lyric moves on. No concept-word slams. The sung lyric line stays
  (beginners read along). 4-8 instrumental bars after each new idea, the picture finishing it with no new text.
- 3.5-4.5 min. The last chorus reassembles every piece; the outro names the next episode.
- Shorts stay fast: the hook Short (cut.py) is the fast lane and is exempt from the course gate.
- Template for new course episodes: `tools/new_project.sh <name> meaning` (the forks before 2026-09-28 8 am
  divided the HUD by alpha twice, so every partial fade popped; meaning has the straight-alpha mix, render.ts on a
  private 7200-7899 port that refuses a page whose title isn't this episode's). A vertical over 3:00 is not a
  Short: no #Shorts in its title.

**Course look:** the Next Token riso engine as forked (paper, fluorescent pink, blue, ink), so colours mean the same
thing in every episode and in the recap: pink = the one thing being taught, while it is sung; blue = the data it
acts on; ink = structure, text, outlines. A small solid-ink marker top left: "ML, SLOWLY  N/5" (where Next Token's
model readout was); the rest of the HUD is per episode.

**Parallel agents** (one per episode, running at once):
- Each owns `projects/<slug>/` only. `tools/`, this file and other projects are read-only, except additive entries
  to `CANON` in `tools/song/score.py` (re-read right before editing, then scp the whole file to the GPU host). Anything
  else a tool needs: do it project-locally and note it in the DEVLOG for the lead.
- <gpu_host>: GPU only through the lease, agent name `course-<slug>`, one lease at a time, `--ttl` sized to the job and
  released after. At most 2 ssh connections per agent; long jobs in tmux (`course-<slug>`), polls >= 60 s apart.
  Song runner files in `<gpu_song_dir>/<slug>/` (lyrics_file "<slug>/lyrics.sing.txt"), run names
  "<slug>-<run>" (outputs `out/<slug>-<run>/`); never overwrite shared files there. Model runs and alignment in
  `<gpu_repo_dir>/projects/<slug>/` on the GPU host.
- Mac: full renders only as `tools/lock.sh render scripts/render-parallel.sh ...` (one at a time); sheets, stills
  and beat sheets at most 2 jobs at once.
- Git: `git add projects/<slug> && git commit -m "<slug>: ..." -- projects/<slug>` (retry after 10 s on
  index.lock). Never another path, never push.
- Nothing leaves the machine (no YouTube, TikTok or X uploads, comments or posts). Done = out/final 16:9 + 9:16 +
  share copies, the hook Short, 3 thumbnails, publish.json drafts (hook-first titles "... | ML, slowly #N",
  description, chapters, tags, AI disclosure, a lowercase funny pinned-comment draft), course gate PASS, DEVLOG.
  Songs: the agent picks (Elliot trusts the calls) and stages the two runners-up in `~/clips/<slug>/`.

## YouTube over the CLI (`tools/publish/yt.py`)
- Auth: youtubeuploader's token format in `~/.config/youtubeuploader/` (any Google OAuth desktop client with the
  YouTube Data API v3 and YouTube Analytics API enabled). `yt.py auth` asks for upload, comment and analytics scopes
  (one consent click).
- An unaudited Google project can only upload private. Scheduling works anyway: videos.update with
  status.privacyStatus private + status.publishAt (verified in Studio 2026-09-27: "Scheduled", no lock).
- Release time: 8:00 am MT on weekdays (picked from the channel's audience split between the US and India).
- Studio only (no API): making a video public (until audited), end screens, cards, Test & Compare thumbnails,
  a Short's related video, pinning a comment. These run through Codex computer use (prompts in
  `tools/publish/studio_prompts/`), never publishing without Elliot's go.
- TikTok via `tools/publish/tt.py` (auth / whoami / upload <file> / status): uploads land in the account's TikTok
  drafts inbox (scope video.upload), where the caption is edited and posted in the app. Needs a TikTok for Developers
  app (Desktop; Login Kit + Content Posting API; scopes user.info.basic + video.upload; redirect
  http://localhost:8765/callback; terms and privacy URLs are required to save it). Keys go in
  `~/.config/tiktok/client.json`, tokens in `token_<env>.json` (mode 600; the refresh token lasts 365 days). Files over
  64 MB go in >= 2 chunks. An unaudited app's direct posts are SELF_ONLY, so posting and scheduling go through
  TikTok's web uploader with Codex (`studio_prompts/tiktok_schedule.md`: caption, AI-generated label on, Schedule).
  TikTok release time: 6:00 pm MT.

## Rules
- Nothing goes out as Elliot (X post, public video, comment) without his go on that exact text; comments need
  `yt.py comment ... --yes`.
- Real numbers only; label computed ones; cite specs. No AI-hype lines.
- Times to Elliot in Mountain Time, 12-hour clock.
- Scene authors edit only their own scene files; kit/engine/overlay/timeline changes go through the lead.
- Any step over ~2 minutes: know why (per-frame ms, sub-frame counts, the serial bottleneck) before rerunning.
- Commit locally; never push without Elliot's go.
