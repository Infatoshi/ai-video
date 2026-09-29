# DEVLOG (times Mountain)

## 2026-09-28: tokens (ML, slowly #2)

- Forked from next-token (engine + kit).
- 2:17 am: read AGENTS.md "The course", SPEC brief, next-token SPEC/DEVLOG, ENGINE.md.
- 2:18 am, real data first (the lyrics are built on it). `tok/llama_tok.py` on the GPU host, CPU only, tokenizer from the
  HF cache (gated files stay there; only derived splits/ids leave, `out/llama_tok.json`):
  vocabulary 128,256 = 128,000 regular + 256 special. Question "How many r's are in strawberry?" = 8 tokens
  `How · many ·r 's ·are ·in ·strawberry ?` = 4438 1690 436 596 527 304 73700 30 (43 with the chat template).
  " strawberry" (leading space) = 1 token, 73700; "strawberry" (no space, as in the model's own answer after a
  quote) = str|aw|berry 496 675 15717; " Strawberry" = 89077 (one token, another number); " STRAWBERRY" = 4
  (STR|AW|B|ERRY); " strawberries" = 76203. " s t r a w b e r r y" = 10 tokens, " r" = 436 three times.
  Other splits: " Motown" Mot|own, " ChatGPT" Chat|G|PT, " Llama" L|lama but " llama" 1, " tokenization"
  token|ization, " HELLO" HEL|LO vs " hello" 1, " unhappiness" unh|appiness, numbers in 3-digit groups
  (" 1234567" = ' '|123|456|7). The tokenizer.json lists 280,147 merges: a tiktoken-conversion artifact (several
  merge paths per token), so the video never quotes a merge count for Llama.
- 2:18-2:20 am, `tok/spell_test.py` (lease course-tokens, 1 min on the 3090, bf16): the question asked 100 times
  each with the model's default sampling (T 0.6, top-p 0.9, seed 7), max 40 new tokens, first number tallied.
  Word as one token: greedy "There are 2 r's...", 67 of 100 said 2, 33 said 3 (Next Token's single run had
  69.7% / 30.3% at that step). Letters spaced out (every letter its own token): greedy 3, 90 of 100 said 3; the
  other 10 began counting out loud and hit the 40-token cap (8 parsed as "1" from list numbering, 2 no number;
  at least 2 of the 10 concluded 2). Hyphenated letters: noisy (57 said 3, many counted out loud), not used.
  So the video can say: one chunk -> it says two; letters spaced out -> it says three. It claims nothing more
  about why.
- 2:20 am, lyrics (`song/lyrics.txt` annotated, `song/lyrics.sing.txt` sung): "Never the Letters", verse 1 the
  strawberry example (8 chunks, 8 numbers), chorus = the one definition ("A token is a chunk of text / Every
  chunk gets a number / The model reads the numbers / Never the letters"), verse 2 = BPE on this song, verse 3
  = 73700 and the spaced-letters test, bridge = same letters different numbers, outro names #3. ~330 sung words
  (with the group's answers) for a 260 s take: ~80 words/min if the vocal spans ~245 s.
- `tok/bpe.py` (stdlib, ~60 lines): BPE trained on the sung lyrics (330 words, 119 unique, 1,333 letters,
  alphabet 24 letters + apostrophe): 140 merges until no pair repeats. Merge 1 is e+r (64 times), then t+h (52),
  th+e (35), b+er (27), ... "the" whole at merge 3, "number" 7, "chunk" 11, "letters" 22, "strawberry" 97.
  Verse 2 now sings the real first merge ("E and R, glue them together"; still e+r x64 after the edit).
- Additive CANON entries in tools/song/score.py (A-I, S-T-R A-W berry, "seven three seven oh oh"/73700), scp'd
  to the GPU host (md5 matched local before). Not committed (tools/ is the lead's path): lead, please commit.
- 2:23 am: song runs `song/runs/m1_capA.json` (seeds 201-208) and `m1_capB.json` (301-308): 100 BPM, Bb major,
  260 s, xl-sft + 4B LM, offload. `song/song_chain.sh` (gen A, gen B, score, rhythm) queued in tmux
  course-tokens under the lease (ttl 110m). The lease record from the spell test had to be cleared first
  (`overnight-compute clear --agent course-tokens`; "already done" otherwise). Queue at 2:24 am: downhill
  running (to 3:08 at most), learning-to-write queued, then tokens.
- 2:30-2:50 am, build started while the lease queue is ahead of us (downhill running, learning-to-write
  queued before tokens): `tok/build_data.py` -> `data/tok.json` (derived numbers only: splits, ids, tallies,
  merges, a 19 x 19 window of real vocabulary around " strawberry" with profanity blanked; the full answers stay
  in out/). Stated-total classifier: word 67 said 2 / 33 said 3; spaced 90 said 3 / 2 said 2 / 8 cut off
  counting. Provisional timing (`tok/provisional.py`, data/*.approx.json, gitignored) from the bar plan at
  100 BPM so the scene can be built before the take exists; the aligned take replaces it.
- Engine for the course: `data/tok.json` replaces the Llama run (llm.json/story.ts dropped); HUD rewritten
  (marker "ML, SLOWLY 2/5" top left, the sung line crossfading 0.35 s and holding 2.5 s, words easing to pink,
  a small source line bottom left set by the scene, no slams); app/analysis point at audio/tokens.* (main.ts,
  render.ts, render-parallel.sh OUTNAME=tokens, common.py SONG = "tokens", tempo window 92-108).
- One scene for the whole song (`scenes/world.ts` + `world-draw.ts`): one sheet, one camera, no cuts. Stations:
  Q the question as type sorts (typed in the intro, cut into Llama's 8 chunks on "cut", split-flap flips to the
  ids on "swapped", number slips into the model on "shown"), the chorus definition built on Q each time
  (brackets + "token = a chunk of text", slips + "every chunk gets a number", "reads the numbers", "never the
  letters" with the letters dimmed); B our BPE on "the model reads the numbers never the letters" (merge 1 on
  "glue", 2 on "Count", 3 on "glue", 4-29 through "Till the words...", the rest in the instrumental; our
  tokenizer's ids in chorus 2); V the real vocabulary in id order (358 per row) from a readable patch around
  " strawberry" out to all 128,256; S " strawberry" fuses into one chunk and flips to 73700 digit by digit on
  "seven three seven oh oh"; T the test (67 of 100 said 2 / 90 of 100 said 3, 10 x 10 tallies); D same
  letters, different numbers (73700 / 3492 364 496 675 15717 4527 / 89077); final chorus pulls back over all of
  it and every letter on the sheet flips to its number; outro: an arrow out of 73700, "next: meaning is a
  direction". Landscape puts S/T/D in a second column so the overview stays compact; portrait is one column.
- Pace check on a 46 s test render (34-80 s, provisional timing), gate.py's own measure: the first camera
  drift (slow push + pan, both capped) left 1,021 of 1,380 frames "still" (runs up to 7.7 s: a gate fail).
  Fix: the camera floats on a slow circle (36 px, one turn per 18 s, ~12 px/s) with a breathing zoom, always:
  84 still frames, none in a run over 0.5 s, 0 cuts detected.
- 3:00 am, review fixes: the vocabulary field moved clear of every station (it had been drawn under B); its
  readable text limited to a 19 x 19 window around " strawberry" (the full patch had " Pornhub" and "fuck"
  within reach of the zoom; profane tokens in the window are blanked, none were); a pink ring marks " strawberry"
  once zoomed out; the instrumental after chorus 2 now paces the line's merges (30 to the last one that changes
  it) over the first 62% and flips after; "never the letters" is a pink strike through the letters (dimmed
  letters printed as noisy dots); the 1-8 count under the ids became a brace (one number at a time); in the test
  the first result steps back when the second arrives. Portrait: test panel stacked (slips, then model + tally),
  outro arrow goes up from the tile. Perf: 17-23 ms/frame, adaptive sampling stops at 12 sub-frames.
- Thumbnails (`scenes/thumb.ts`, `out/publish/thumbs/thumb_{A,B,C}.jpg`, check sheet `thumbs_check.png` with the
  168x94 versions): A "SPACE IT OUT" (one chunk -> 2, 67 of 100; spaced letters -> 3, 90 of 100), B "FIND THE R"
  (one split-flap tile 73700 + the strawberry), C "IT READS" (the question's 8 ids). Next Token already uses
  "AI SEES 73700" and the sliced strawberry, so none repeat those. `thumb-berry.ts` copied from next-token.
- `publish.json` drafted (titles "... | ML, slowly #2", description with the real numbers and an AI disclosure,
  chapters anchored to sung lines via `tok/chapters.py` -> data/timeline.json since the edit is one scene, tags,
  lowercase pinned-comment drafts; the hook Short's cut list waits for the alignment). `tok/pace.py`: projected
  words/min per take from Whisper word times, for the pick.
- 3:41 am: lease acquired (who-is-it and downhill got it first: the queue is a race, not FIFO); song chain running (gen A, gen B, score, rhythm), ttl 110m.
- 3:45 am: gen runs at 199 s per batch of 2 (260 s takes). To skip a second lease race for the alignment, the
  analysis can run on the GPU host's CPU (Ryzen 9 9950X3D, 32 threads; CPU work needs no lease): whisper_run.py picks
  cuda or cpu (fp32 on cpu), run_linux.sh takes DEMUCS_DEV=cpu; the tokens Whisper prompt added; pron.py spells
  the letters sung by name (A-I = "ay eye", R's = "ars", S-T-R, A-W).
- 4:28 am: song chain done, lease released (gen 2 x 13.5 min, score + rhythm ~20 min). Scores (recall / wer /
  Audiobox PQ): capA 203 0.973/0.176/7.59, 201 0.951/0.061/8.00, 207 0.945/0.076/7.96, 205 0.942/0.076/8.11,
  202 0.556, 206 0.243, 208 0.191, 204 0.012 (4 of 8 capA takes collapsed: ad-libs, "SILENT REFLECTION");
  capB 301 0.976/0.030/7.79, 303 0.954/0.061/7.94, 307 0.945, 305 0.933, 308 0.836, 304 0.827, 302 0.280,
  306 0.246. Rhythm (tempo, rhyme on the 8th grid): 207 103.1 BPM 0.32, 205 100.6 0.39, 303 98.7 0.54,
  201 101.2 0.51, 305 101.2 0.41, 301 101.9 0.51.
- Pace is the binding constraint: Whisper's own word lists are useless for it (hallucinated repeats: 201 reads
  812 words), so `tok/pace.py` now maps Whisper's words onto the 325 lyric words (what data/lyrics.json will
  count) and interpolates misses. Projected words/min / busiest 30 s (gate ceilings 95 / 120): 207 89.3 / 114,
  205 87.3 / 118, 201 87.6 / 120, 303 88.0 / 120, 305 92.7 / 118, 203 95.4 / 130, 301 96.1 / 132, 307 96.4 / 140.
  The best-scoring takes (203, 301) sing the lyric too fast for the course gate.
- Structure plots (`song/takes/m*_*.png`, `tools/song/structure.py`, run with `uv run --offline` after a PyPI
  timeout): 207 is the only take with clear instrumental dips after chorus 1 (~70 s), chorus 2 (~132 s), before
  the bridge (~169 s) and before the outro (~215 s); its vocal enters at ~10.3 s after a band entry at 6 s.
- **Pick: m1_capA/207** (recall 0.945, wer 0.076, every line heard in the transcript, CE 7.50 / PQ 7.96, the
  most pace headroom, the breaks where the picture finishes each idea). Runners-up for Elliot's ear:
  m1_capA/205 (best Audiobox PQ 8.11, a 20 s intro, no breaks) and m1_capB/303 (tightest rhythm 0.54, recall
  0.954). All three at -14 LUFS in ~/clips/tokens/. `tools/song/pick.sh tokens tokens-m1_capA 207`: -14.0 LUFS,
  -0.9 dBTP -> audio/tokens.wav/.mp3; line 0's seed moved from Whisper's 0.0 to 10.0 and anchored (9.6-11.0 s).
- 4:44 am: alignment running on the GPU host's CPU in tmux course-tokens (no lease): Demucs htdemucs_ft on cpu, CTC,
  Whisper large-v3 (fp32), align, analyze. Model caches symlinked from next-token's analysis/.cache (read only).
- Render hazard (lead, 4:40 am): every fork's vite defaults to :5173, so render.ts could render another episode.
  render.ts now never uses a shared server unless --url names it; it starts a private one on a free random port,
  and the page reports `project: 'tokens'` (main.ts), which render.ts checks before rendering anything.
- 4:58 am: the first CPU alignment sat 12 min in `uv run` (uv's request for pypi.org/simple/python-dateutil times out on both machines while curl gets it in 0.2 s); rerun with UV_OFFLINE=1 (every package is cached). Demucs on CPU: ~3.7 s of audio per second per model.
- 5:15 am: CPU Whisper large-v3 (fp32) was still on its first pass after 17 min (~180 CPU-minutes), and
  align.py only reads a Whisper word list as a cross-check. Stopped it and handed align.py the word times the
  song stage already had on the GPU for this take (rhythm.py: Whisper large-v3-turbo on the Demucs vocal stem of
  207; `work/tokens/whisper_turbo_prompt.json` notes the source); `analysis/align_only.sh` runs align.py +
  analyze.py. Result: 102.07 BPM (period 0.58781 s, first beat 0.063 s), 443 beats, 111 bars; 43 lines, 325
  words; median |aligned - Whisper| 0.122 s over 282 matched words (turbo's word starts sit on the previous
  word's end; the QA plots, `out/qa/align/line_*.png`, show all three CTC models agreeing, e.g. line 15's
  echo "(together)" at 90.17 s, line 26's digits "seven three seven oh oh" at 143.33-145.87 s).
  Sections by bar in analyze.py (intro 0-4, verse1 4-20, chorus1 20-28, inst1 28-30, verse2 30-46, chorus2
  46-54, inst2 54-56, verse3 56-72, bridge 72-81, chorus3 81-92, outro 92-end).
- Course pace from data/lyrics.json (gate.py's definitions): 325 words over 218.2 s = 89.4 words/min (ceiling
  95), busiest 30 s 114 (ceiling 120, the window starting at 29.0 s). Vocals end at 228.6 s; the take plays an
  instrumental outro to 260 s, so the camera pulls back over the whole sheet (all numbers) and then settles on
  the question as the model got it (8 ids) before the fade.
- Chapters (`tok/chapters.py` -> data/timeline.json): 0:00 intro, 0:27 cut, 0:46 chorus, 1:11 bpe, 1:38 vocab,
  1:48 chorus2, 2:12 strawberry, 2:28 test, 2:50 bridge, 3:10 finale, 3:35 outro.
- Hook Short cut (publish.json tokens_hook): song 28.28-47.09 (8 bars: cut into chunks, swapped for numbers,
  eight numbers, into the model) + 141.14-157.60 (7 bars: "(one chunk) One number (seven three seven oh oh)",
  the test, 67 vs 90 of 100), both joins on downbeats, 35.3 s.
- 5:22 am: 16:9 final render started (tools/lock.sh render, 4 x 12 s segments, adaptive up to 108, shutter
  0.2): ~64 s per 12 s segment per job, every frame converges at 12 sub-frames.
- 5:28 am: first 16:9 final (6 min). Checked frames: it is this episode. Course gate FAIL: 17 "cuts" in
  three kinds, and 4 frozen runs. (1) 7-frame clusters at 143.0 and 165.0 s: the big " strawberry" tile's
  split-flap flip (0.55 s) in an otherwise sparse frame; now 1.2 s. (2) 101.5 / 105.6 s: the vocabulary switched
  from the prerendered dot field to live cells within one frame as the zoom crossed 10 px per cell; now they
  crossfade over 12-18 px, and the cell fill/text fade over 30-52 px (densities matched). (3) 95.8 s: a merge in
  the flurry against a near-still baseline. Frozen runs 161.1 (3.2 s), 220.2 (4.1), 224.4 (3.6), 228.0 (3.2):
  one tile on paper under a 12 px/s float reads as still (mean |diff| 0.4-0.5e-3 < 1e-3). The float is now
  60 px / 16 s (~24 px/s) with a 3.5% / 20 s breathing zoom. Test clips (samples 1): 138-170 s and 212-236 s no
  cuts, longest still run 1.6 s; 90-110 s no cuts. The 9:16 render that had started was stopped (it would have
  mixed old and new code across segments) and both aspects re-render from 5:35 am.
- 5:41 am, 16:9 final re-rendered (6 min): **course gate PASS**: 89 words/min, busiest 30 s 114, 0.0 hard
  changes/min (0 cuts detected), median shot 260 s (one continuous shot), nothing frozen; activity median 1.095,
  p10/median 0.59. Frames checked (30/105/160/225 s): this episode. Share copy crf 24: 344 MB.
- 5:48 am, first 9:16 final: gate PASS (2 detected changes, 0.5/min, median shot 84.5 s), share copy crf 25
  320 MB, and the hook Short cut from it (35.27 s, audio continuous across the join). Per-beat sheets
  (`out/qa/beats*/`, 10 sheets x 48 each aspect) then showed portrait-only clipping: the B line ("·the",
  "never" cut at the left edge with the camera float) and the bridge's second row (4527 cut at the right), and
  the HUD source line running off a 1080-wide frame. Portrait B/D tiles smaller (44 px), B framing wider, D
  centred, the source line wraps. Landscape output unchanged by these (portrait branches; wrapping never
  triggers at 1920). 9:16 re-render from 5:52 am; the hook Short and the vertical share copy are redone from it.
- 6:03 am, done. 9:16 re-render gate **PASS**: 89 words/min, busiest 30 s 114, 0.2 hard changes/min (1 cut
  detected), median shot 130 s, nothing frozen. Finals (sha256):
  out/final/tokens.mp4 1920x1080p60 260.0 s 640 MB 78c082de246ea7d34d97dbff425ee09fd28000005c8f28ee19ac618642bf4d22
  out/final/tokens_share.mp4 (crf 24) 344 MB 5a4d59a8fcd18adab9ec476deec50d1f233de7f90aeeed61b795ae0ede45e6c1
  out/final/tokens_vertical.mp4 1080x1920p60 640 MB 4c62ad245f41ade179376d487c8044ee2dab32cf7187fc251cb66e78254bd53b
  out/final/tokens_vertical_share.mp4 (crf 25) 309 MB 007222b998d49758c366a3c4c81d9ec9e43d4b65d7f4e2c26a3c27ddcb00447b
  out/final/tokens_hook.mp4 (tools/publish/cut.py) 1080x1920 35.27 s 39 MB 6fb6a8171bc8e7c4c39e5d7da6400c156d1bbe77603fe78cdb59ff1c15de3089
  Gate plots: out/final/*.gate.png. Thumbnails out/publish/thumbs/thumb_{A,B,C}.jpg (B "FIND THE R" set as the
  main one in publish.json). `tools/publish/yt.py package tokens` (local only) wrote meta.json + captions for
  all three variants (chapters 0:00 ... 3:36). Nothing uploaded, posted or pushed.
- For the lead: tools/song/score.py has the tokens CANON entries uncommitted (tools/ is yours); Elliot's ear
  on the pick (207) vs 205 / 303 in ~/clips/tokens/, and on the pinned-comment drafts in publish.json.
- 7:30 am, HUD fade fix (from the lead; the meaning agent found it): post.ts mixed the HUD as
  `mix(col, h.rgb / max(h.a, 1e-4), h.a * hud)`, but the HUD CanvasTexture is straight alpha, so every
  partial-alpha pixel was divided up to paper-white or full ink. Mid-fade still at 9.92 s (the first lyric line
  fading in), before: the paper strip is a hard white box while its text is half faded; after
  (`mix(col, h.rgb, h.a * hud)`): the strip and text fade together into the paper. render.ts private-server
  ports moved to 7200-7899 (clear of Postgres 5432 and Chrome's unsafe ports), 6 tries.
- 7:31-7:53 am, both finals re-rendered (same flags), share copies and the hook Short (same segments) rebuilt.
  Gate PASS: 16:9 89 words/min, busiest 30 s 114, 0.0 hard changes/min, median shot 260 s, nothing frozen;
  9:16 89 / 114, 0.2/min (1 cut), median shot 130 s, nothing frozen. Frames checked (10/60/150 s, 10/86/186 s,
  the Short): this episode. sha256:
  tokens.mp4 44a5329bc6fb4bb8362e93a7567c45378eb0ebbe26ab14a399d87be5da99dd40
  tokens_share.mp4 (344 MB) a308bd002531cd556aeac607e1ce1ef21e14233d98506201ff6dbe666fb204e4
  tokens_vertical.mp4 12d41f8e71f29ff0867fcbc8c0a6f2de48d1d6eb9920ec8ea626253e0bd21ce7
  tokens_vertical_share.mp4 (309 MB) 331bd59d99126a4c614236a7f8eb2e3546ab3124b32ccf1e20015fd3b2c1427d
  tokens_hook.mp4 (35.27 s) 0c8a81db414f72b751fb9d760451f1f2b7924222785a62653a2471ed9c88b476
