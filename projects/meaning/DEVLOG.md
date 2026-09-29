# DEVLOG (times Mountain)

## 2026-09-28: meaning

- Forked from next-token (engine + kit).
- 2:18 am, real data first (the lyrics depend on it). `llm/extract.py` on the GPU host, CPU only, no model load: the
  safetensors header of model-00001-of-00004 gives `model.embed_tokens.weight`'s byte range; its bf16 bits are
  copied as they are (128,256 x 4,096, sha256 e1598c85...) with every token's decoded string; 11 s. Copied to the
  Mac (out/data_src/, gitignored, 1 GB). `llm/embed.py` (numpy, 3 s) checks the sha256 and writes data/emb.json.
- What the table really says (cosine similarity over all 128,256 rows; tokens keep their leading space):
  - " rain" (id 11,422): nearest are its own spellings, then weather: ·Rain 0.520, Rain 0.404, ·rains 0.346,
    ·snow 0.277, ·raining 0.264, ·rainfall 0.250, rain 0.241, ·wind 0.209, ·sun 0.196, ·weather 0.194, ·storm 0.184.
  - " three": ·four 0.711, ·two 0.676, ·five 0.626 (the cleanest neighbourhood: other numbers first).
  - Random pairs of common lowercase words: mean cosine 0.038 (p99 0.109): the "unrelated" level.
  - **king - man + woman** (unit vectors, 3CosAdd, the three input tokens left out): 1 ·King 0.423, 2 ·KING 0.332,
    3 King 0.306, **4 ·queen 0.297**, 5 ·kings 0.285. Not queen: king's own spellings win, queen is fourth. (In
    king's plain neighbour list queen is 5th, behind ·King, ·KING, King, ·kings.) That is the story of verse 3.
  - The same arrow works cleanly elsewhere: boy - man + woman = ·girl #1 (0.461), father - man + woman = ·mother #1
    (0.443); 13 of 23 male/female pairs land on the partner at #1 (brother/sister, he/she, son/daughter,
    husband/wife, waiter/waitress ...); monk -> nun is rank 1,030.
  - 3D: PCA fitted to the 46 labelled tokens keeps 20.7% of their variance (8.7 + 6.5 + 5.5); a PCA of 12,801
    common words keeps 1.9% and scatters the neighbourhoods (9 of 36 nearest agree vs 21 of 36), so the flight uses
    the labelled-set PCA, labelled as such, with 8,000 common words projected the same way as the haze.
- Lyrics written from those results (song/lyrics.txt; sung spelling song/lyrics.sing.txt): verse 1 a word is a row
  of 4,096 numbers (its embedding), verse 2 close = pointing the same way (three -> four, two, five; rain -> its
  spellings, then snow, wind, storm), verse 3 the arrow man -> woman (boy -> girl, father -> mother, king ->
  King, KING, King, then queen fourth), a 2-line bridge ("the spelling pulls harder than the crown"), outro names
  #4. Chorus (the one definition, 3x): "Every word is a list of numbers / And every list is a place / Words that
  mean alike live close together / And meaning is a direction in the space". 309 words.
- CANON (tools/song/score.py, additive): "4,096" -> "four thousand ninety six", "chat bot" -> "chatbot". Pushed to
  the GPU host (md5 9611502...).
- 2:25 am: song runs queued on the GPU host (tmux course-meaning, lease course-meaning, ttl 90m): runs/a1.json (dream pop,
  female lead) and runs/b1.json (dreamy indie pop, male lead), 85 BPM, D major, 245 s, 8 seeds each
  (3301-3308, 3401-3408), then score.py and rhythm.py in the same lease. Queue: downhill running, learning-to-write
  and tokens ahead. Chain script: ~/dev/cuda-song/meaning/chain.sh, log chain.log.
- 2:30-3:00 am, build on provisional timing while the GPU queue runs (data/audio.json + data/lyrics.json are a
  PROVISIONAL 85 BPM grid from out/scratch/fake_align.py; the alignment replaces them). Engine changes (project-local):
  no Llm/Story load (engine.ts), hud.ts rewritten for the course (crop marks, "ML, SLOWLY 3/5" tag, source tag, the
  sung line fading between lines, no slams) plus `OVERLAYS` (scene type drawn with the HUD after the print: crisp,
  never halftoned), post.ts `crawl` (halftone screen offset), song paths -> audio/meaning.*, OUTNAME meaning.
  One timeline entry, scenes/flight.ts (+ flight-data.ts), and scenes/thumb.ts (3 thumbnails).
- Camera: a monotone cubic through lyric-keyed poses (Catmull-Rom overshot between far-apart islands and swung
  through empty space), plus a slow sway; a chase cam while the rain point flies from the table into the space.
- Pace check before any render (out/scratch/motion.sh: stills at 30 fps -> the gate's 320 px mean |diff|): the calm
  wide shots measured 0.0002-0.0003 per frame, under the gate's 0.001 "still" line (a 40 s test clip had 903 of 1,200
  still frames, runs up to 4.1 s, and two false cuts where a number faded in over a near-zero baseline). A 0.1 rad/s
  orbit only reached 0.0009. Fix: the halftone screen crawls (9, 4) px/s and the background wash is a bit heavier
  (0.22): every frame now moves 0.0019-0.0025 with no content change, which also lifts the cut detector's baseline.
- Thumbnails (out/wip/thumbs, checked at 168x94): A "QUEEN CAME 4TH" + the real top 4, B "RAIN = 4,096 NUMBERS" +
  the first 384 of its numbers, C "MEANING = DIRECTION" + the arrow carried to boy and father (real 3D squash).
- GPU queue: the lease is not first-come (who-is-it, queued after meaning, got it at 2:54 am).
- 3:10 am: to save a second trip through the GPU queue, the queued chain now ends with
  out/autopick_align.sh on the GPU host (copy in out/scratch/): provisional pick = best word recall among 200-280 s takes,
  mastered like tools/song/pick.sh (-14 LUFS / -1 dBTP), lyric times seeded (src_from_words.py with the display
  lyrics, so King/KING keep their case), then analysis/run_linux.sh. If the pick changes after the structure and
  rhythm check, the alignment re-runs for the new take. Display lyrics now match the sung words ("Four thousand
  ninety-six"). analyze.py: TEMPO_RANGE 78-92 for meaning, sections provisional ("song") until the first pass.
- Timeline: the flight is split into chapter entries (intro, table, definition, close, definition-2, direction,
  king, together, outro) that all run the same pure-function-of-time scene: seamless (no diff spike across a
  boundary, checked at 9.29 s and 56.47 s) and the app's timeline now carries the YouTube chapters.
- publish.json drafted (titles "Queen came 4th: ... | ML, slowly #3", description with the real numbers, chapters,
  tags, AI disclosure, lowercase pinned comments); the hook Short's cut waits for the aligned timing.
- 4:28 am: the course-meaning lease started (tokens held it 3:51-4:28, who-is-it 2:54-3:51); chain running in tmux course-meaning.
- 4:45 am, lead's render hazard (render.ts renders whatever serves localhost:5173): meaning's render.ts now defaults
  to http://localhost:6133 (off 5173 and outside the private-server range 5300-5800, so it normally starts its own
  private server) and refuses to render unless the page title is "Meaning Is a Direction"; vite.config port 6133
  strictPort. Every earlier still/sheet and the 60-100 s test clip were checked by eye: all this episode. No dev
  server of mine was ever left on 5173.
- Generation: 189 s per batch of 2 (LM 45 s, DiT 126 s); take 3301 is 245.0 s, ~86 BPM by librosa.
- 5:06-5:20 am, song run 1 results (score.py word recall / rhythm.py): a1 (female dream pop) 3305 0.981 (WER 0.026),
  3301 0.974, 3302/3303 0.964, 3307 0.948; 3306 0.259 and 3308 0.010 failed. b1 (male) 3407 0.977, 3403 0.968,
  3401/3402/3405/3406 0.951. All 245.0 s, tempo 85.0 (3302 80.8). Pace problem: from the Whisper word times matched
  to the lyrics, every take sings some stretch too densely for the course gate: busiest 30 s ~134 (a1/3307),
  136 (a1/3305), 144 (b1/3401), 158-186 (others) words/min vs the 120 ceiling (ACE-Step puts an 8-10 word line in
  1-1.5 bars at 85 BPM). The v1 lyrics (309 words) cannot pass with these takes.
- Lyrics v2 (song/lyrics.txt; v1 kept as lyrics_v1*.txt): the same story in shorter lines, 230 words. Chorus is now
  "Every word is a list of numbers / Every list is a place / Words that mean alike live close / Meaning is a
  direction". Verse tags ask for slow, sparse phrasing. Run a2 (runs/a2.json): the a1 caption plus "slow sparse vocal
  phrasing with long pauses between short lines", 240 s, seeds 3501-3508. Appended to the running chain
  (~/dev/cuda-song/meaning/regen.sh: gen, score, rhythm, then out/autopick_align2.sh, which backs up the a1
  alignment to data_a1/ and clears the stems so Demucs re-runs), so it runs in the same lease instead of another
  trip through the queue (~25 min of GPU).
- 5:42-5:56 am, run 2 (v2 lyrics, a2): recall 3505 0.996, 3507 0.996 (WER 0.004), 3504 0.991, 3502 0.978,
  3501 0.961, 3503 0.957, 3506 0.944, 3508 0.913; all 240.0 s. Pace from the matched Whisper words: busiest 30 s
  3505 82, 3507 90, 3506 98 (pass); 3501/3508 126, 3502 134, 3503 138, 3504 146 (fail). Structure
  (song/takes/a2_3507.png, a2_3505.png): 3505 has the best shape (a quiet bridge, then the last chorus lifts) but the
  generator ran out of time: the outro line is crammed into 237.6-240.0 s and the song stops mid-word. **Pick
  a2/3507**: recall 0.996, WER 0.004, tempo 85.0, the best couplet parity of the run (2.96), a soft intro (vocal at
  22.9 s), instrumental gaps after each chorus, the outro sung at 227.6-232.3 s then a fade to 236 s. Runners-up for
  Elliot's ear in ~/clips/meaning/: a2/3505 and a2/3506 (loudnorm -14 LUFS); the pick is meaning_a2-3507_PICK.wav.
  GPU lease 4:28-5:56 am (gen 2 x 16 min, score, rhythm, two alignments).
- Alignment (analysis/run_linux.sh on the GPU host, auto-run in the lease): 38 lines, 230 words; mastered -14.0 LUFS,
  -1.0 dBTP; 85.001 BPM, snare on beats 2 and 4. Against Whisper turbo the median |offset| is 0.14 s, but the big
  differences (1.7-3.9 s at six line starts) are Whisper being early: the Demucs vocal stem's energy rises exactly at
  the aligned times (22.9, 28.6, 34.2, 51.2, 76.9, 195.9 s), so no FIX pins were needed. Pace from data/lyrics.json:
  66 words/min, busiest 30 s 88 (ceilings 95 and 120). Sections by bar in analyze.py (intro 0-8, verse1 8-23,
  chorus1 23-31, inst1 31-34, verse2 34-47, chorus2 47-55, inst2 55-57, verse3 57-68, inst3 68-69, bridge 69-73,
  chorus3 73-80, outro 80-); instrumental gaps after the choruses are 11 s and 9.5 s (about 3-4 bars), after verse 3
  only 4.7 s: the generator's arrangement, noted as a deviation from the 4-8 bar rule.
- 5:59 am: camera keys closer than 0.8 s are merged (verse 2's last key landed exactly on chorus 2's first, 133.50 s,
  and the spline divided by zero: 130-144 s rendered empty paper). Sheets at 16:9 and 9:16 over the real timing
  checked (out/wip/sheet_real*.png). data/timeline.json from `render.ts timeline` (9 chapter entries); chapters
  preview: 0:00 Intro, 0:21, 1:05, 1:34, 2:13, 2:40, 2:56, 3:26, 3:47 (all >= 13 s).
- 6:01 am: final renders started under tools/lock.sh render (16:9 then 9:16; 4 jobs x 12 s, --samples auto
  --max-samples 108 --shutter 0.2). Log out/final_render.log. Hook Short cut chosen: 160.97-166.62 ("Draw an
  arrow from man to woman") + 175.09-192.03 (the famous one -> King, KING, King -> queen fourth), both on bar starts,
  22.6 s.
- 6:09 / 6:18 am, first finals (16:9 2.8 GB, 9:16 2.6 GB). Course gate on the 16:9: PASS (66 words/min, busiest
  30 s 88, 1.0 hard changes/min, median shot 19.5 s, nothing frozen). Frames pulled from the file: this episode.
  Its 4 detected "cuts" were content arriving: 106.4 s (the "four" thread + number), 125.9 s (rain's ranked list),
  145.4 s (the threads switching to pink in one frame on "close"), 227.6 s (the next-episode card). Fixes: thread and
  arrow colour changes fade over 0.3-0.4 s, the ranked lists fade over 1.0 s, numbers over 0.45 s, the card over
  1.8 s; the bridge now holds on king and queen (its last camera key had been merged into the last chorus's first,
  so the camera drifted past the haze core during "the spelling pulls harder than the crown"). Re-rendering both.
- 6:30 am, the real cause of those 4 "cuts": post.ts blended the HUD layer as if premultiplied
  (`mix(col, h.rgb / h.a, h.a)`), but Layer2D uploads straight alpha, so every partial-alpha fade printed at full
  strength from its first frame (a paper box became a white box in one frame; pink type popped in fully). Fixed to
  `mix(col, h.rgb, h.a)` (project-local; the forked engine has the same line, which is presumably why Next Token's HUD
  avoided partial alpha: worth a look by the lead for the other episodes). Stopped the second render, re-rendering both.
- 6:41 am: the third render lost segment 156-168 s: its private vite server drew port 5432, which Postgres holds
  (ERR_EMPTY_RESPONSE). render.ts now picks free ports in 6200-6999 and retries up to 6 times. Re-rendering both
  (Mac load average ~30-40 from the other episodes).
- 6:55 am, 16:9 final re-rendered with the blend fix: out/final/meaning.mp4 (1920x1080p60, 240.0 s, 2.8 GB). Course
  gate: **PASS**, 66 words/min, busiest 30 s 88, 0 hard changes (0.0/min), median shot 240 s (one continuous shot),
  nothing frozen (35 frames under the still line, no run over 3 s). Frames pulled from the file: this episode.
  Share copy (two-pass x264 15.5 Mb/s, -tune grain, AAC 256k): out/final/meaning_share.mp4, 471.7 MB, side by side
  with the final at 1:1 the halftone survives.
- The vertical's last segment (228-240 s) died on a Chrome "unsafe port" (6668 is in Chrome's IRC block list);
  render.ts's private ports moved to 7200-7899. That one segment is being re-rendered under the render lock (held by
  learning-to-write's finals at 7:15 am) and re-joined, instead of re-rendering the whole vertical.
- 7:22 am, vertical final: out/final/meaning_vertical.mp4 (1080x1920p60, 240.0 s, 14,400 frames, 2.6 GB; the seam at
  228 s has uniform timestamps). Course gate: **PASS**, 66 words/min, busiest 30 s 88, 0 hard changes, median shot
  240 s, nothing frozen. Share copy out/final/meaning_vertical_share.mp4, 472.9 MB (two-pass 15.5 Mb/s).
- Hook Short (tools/publish/cut.py, from the vertical): one clean segment 176.50-206.15 s (beat 3 of bar 62 to the
  downbeat of bar 73): "Now the famous one, start at king / You get King, KING, King again / And then queen, in fourth
  place / It isn't magic, it isn't perfect / The spelling pulls harder than the crown" -> out/final/meaning_hook.mp4,
  29.65 s, 1080x1920. (A first two-segment cut joined mid-line: "Start it at boy" cut off, then "...meaning".) The
  last 0.13 s of "crown" is past the cut.
- publish.json final drafts; `yt.py package meaning` (local only) wrote out/publish/<variant>/meta.json +
  captions.srt (+ thumbnail A for the 16:9). Nothing uploaded or posted.
- sha256 (first 16): meaning.mp4 fa70493d716bfdd0, meaning_vertical.mp4 f0cd951ab65ed4e0, meaning_share.mp4
  3302472403701159, meaning_vertical_share.mp4 57f06cb7a5326e4e, meaning_hook.mp4 6f60e9256083b49d.
- For the lead: tools/song/score.py has an uncommitted additive CANON entry from this episode ("4,096" ->
  "four thousand ninety six", "chat bot" -> "chatbot"; pushed to the GPU host), outside projects/meaning so not committed
  here. The HUD alpha-blend bug (post.ts) and render.ts's private-port choice (5432 Postgres, 6665-6669 Chrome
  unsafe) are fixed only in this project; the other forks have the same code.
- 7:45 am, lead's fix: the hook Short clipped the last 0.13 s of "crown". Re-cut 176.50-206.85 s (to the next beat,
  one segment, no joins) and faded the audio out over 206.66-206.83 s, because the next line ("Every") starts at
  ~206.80. Checked with Python on the Demucs vocal stem: "crown" decays from 0.091 to 0.007-0.010 RMS by
  206.30-206.40 s, well before the cut; the last 50 ms of the Short's mix is 0.005. out/final/meaning_hook.mp4 30.35 s,
  sha256 ef358cec68e34430...
