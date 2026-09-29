# DEVLOG (times Mountain)

## 2026-09-28: downhill

- Forked from next-token (engine + kit).

## 2026-09-28: stage 1-2 (song + real data), course agent

- 2:15 am: started from the approved brief (SPEC.md). Read AGENTS.md "The course", next-token SPEC/DEVLOG, ENGINE.md.
- Real-data probe (model/spiral.py, numpy float64, manual backprop, gradient-checked against finite differences):
  2-16-16-1 tanh network = **337 weights**, 200 spiral points (100 per class, 1.5 turns), mean cross-entropy loss,
  plain full-batch gradient descent. Learning-rate sweep (seed 1): 0.3 reaches 100% at step 3,121 (loss 0.696 ->
  0.0049 by step 5,000); 3.0 (ten times as big) never learns (loss never below 0.58, best accuracy 75.5%, 61% at step
  5,000); 5.0 explodes (loss 17). Every plain-GD run has a few loss bumps (edge-of-stability spikes); lr 0.3 seed 1
  has one small bump at steps 693-697 (0.55 -> 0.68), kept and shown as is.
- Landscape: the brief asked for two filter-normalized random directions (Li et al. 2018). Measured: that plane holds
  **1.8%** of the training path (|theta_t - theta*|^2), the path projects to a speck at the centre, and the slice's
  loss at the projected points (0.25) is nowhere near the real loss (0.70). Li et al. 2018 section 7 says exactly this
  (random directions miss optimisation paths) and plots trajectories on the **PCA directions of the path** instead:
  that plane holds 98.8% (PC1 94.9%, PC2 3.9%) and the slice's loss at the projected points tracks the real loss
  (0.62 vs 0.70 at step 0, 0.55 vs 0.55 at step 500). Decision: the landscape is the PCA slice, labelled as a 2D slice
  of a 337-dimensional space; the random-direction numbers stay in the DEVLOG.
- Too-big step, with a real run: lr 3.0 at its step 9 (loss 0.697): along the downhill direction the valley floor
  (0.622) is at step length ~1.05; a 0.3 step lands at 0.660, the 3.0 step lands at 0.818 (higher than it started).
- Lyrics (song/lyrics.txt, song/lyrics.sing.txt): topic first; weights, loss, gradient, learning rate each defined in
  plain words the first time they are sung; chorus = the algorithm, same words x3; numbers sung: 337 weights, ten
  times as big, 3,121 steps. 264 sung words (~80 words/min over a 4:00 take).
- Song runs (song/runs/capA.json female vocalist, capB.json male crooner; jazz waltz, 90 BPM, Bb major, 240 s, 8 seeds
  each). tools/song/gen.py hardcodes timesignature "4": project-local copy song/gen_ts.py passes the run's "timesig"
  ("3"); for the lead: upstream could read it from the run config.
- 2:23 am: generation started on the GPU host under the lease (tmux course-downhill, `downhill/gen_both.sh`).
- 2:49 am: 16 takes done (capA 1101-1108, capB 1201-1208; ~26 min of GPU). The the GPU host queue then held four other
  course agents, so scoring ran **on the Mac** (song/score_mac.py: Demucs htdemucs_ft on MPS, Whisper large-v3 via
  mlx-whisper, the same text normalisation as tools/song/score.py; tempo grid within 5% of 90 BPM; a 3-vs-4 meter
  measure; words/min and busiest 30 s). No Audiobox scores. For the lead: a project-local copy, not a tools/ change.
- capA: 1101 recall 0.981 (91.7 BPM, in 3), 1107 0.981 (90.0 BPM, strongly in 3), 1104 0.977 (90.0, in 3); 1102,
  1103, 1106 sit at 85.7 BPM (the window's edge) and read as not-3. **Pace problem**: the busiest 30 s is 118-146
  sung words/min in every take (1101 124, 1107 122; the course ceiling is 120), always verse 3 running straight into
  chorus 1 (61 words). The lyric broke the course's own rule there: no instrumental bars after verse 3's new idea
  (loss, the map). Same after verse 4. Fix in the song, not the gate: lyrics.sing.v2.txt adds an [Instrumental] after
  verse 3 and after verse 4 (words unchanged); runs capA2 (1301-1306) and capB2 (1401-1406), 256 s.
- 2:59 am: v2 queued on the GPU host (5th in the lease queue). Meanwhile the scene is built against approximate timings
  (data/*.approx.json), anchored to lyric lines by content so the real alignment drops in.
- Engine for the course (app/): one continuous scene `scenes/world.ts` for the whole song (timeline.ts has one
  entry); `engine/model.ts` loads data/model.json (the network, per-step weights, the slice); `story.ts` is the step
  clock anchored to sung words; the HUD (`engine/hud.ts`) is marks + "ML, SLOWLY 1/5" + the sung line, which slides
  in 0.45 s before it is sung and clears in instrumentals (no pops, no slams); `Lyrics.get('^...')` matches a line
  start ("Measure how wrong it is" is also inside verse 3). The panel is the real network evaluated per pixel in a
  shader from the recorded weights. Song paths point at audio/downhill.mp3; render-parallel's OUTNAME default is
  downhill; analysis/common.py SONG = "downhill"; analyze.py fits a 3-beat bar (bar phase from low-band onsets).
- Gate calibration on a 60 s test render: median frame difference 0.0002 in the 2D opening, frozen stretches up to
  3.9 s (the gate allows 3). Fix: every dot swells a little on each downbeat (the waltz's one, ~2 s apart) and the
  camera drifts faster over the landscape.
- Renders use the project's own Vite server (port 5931, `--url http://localhost:5931`): render.ts otherwise takes
  whatever answers on 5173, which could be another agent's project.
- 3:05 am, song pick: **downhill-capB/1201** (male crooner): recall 0.981, WER 0.019 (best of 16), tempo 90.0 BPM,
  the clearest 3/4 of all takes (meter measure +1.58), all 35 lines heard, sung numbers heard as "337" and "3,121
  steps". Busiest 30 s in lyric tokens 114 (under 120; 57 words from 64.6 s), 74 words/min over the song. Shape
  (song/takes structure plot): intro 0-16 s, 7-8 s piano breaks after verses 1 and 2 and chorus 1, a 36 s piano solo
  after the bridge (162-198 s), fade to 240 s. Runners-up staged in ~/clips/downhill/: 1205 (capB, recall 0.981,
  90.0 BPM, busiest 30 s 122) and 1107 (capA female, 0.981, 90.0 BPM, strongly in 3, busiest 122); the pick's master
  is there too. Mastered -14.0 LUFS / -1.0 dBTP -> audio/downhill.wav, .mp3.
  The v2 run (breaks after verses 3 and 4) stays queued: if a v2 take matches 1201 it replaces it (everything is
  anchored to lyric lines, so a re-align is all it needs).
- 3:09 am, alignment on the Mac (analysis/run_mac.sh: Demucs on MPS, MMS + LV60K CTC on MPS, mlx-whisper; align.py
  with ANCHORS for the first word; analyze.py with a 3-beat bar): **90.001 BPM**, first beat 0.664 s, bar phase from
  the bass (low-band onset sums per beat mod 3: 1.00 / 0.52 / 0.08), 120 bars of 2.0 s, sections by bar in
  SECTION_BARS_DOWNHILL. Aligned words vs Whisper: median offset 0.14 s; the worst (line-initial "Take", "And", ~1.1-1.4
  s) are Whisper being early: the vocal stem is silent until the aligned time (checked on the vocal envelope).
- 3:19 am: withdrew the queued v2 song run (still 4th behind three course agents after 20 min; ~1.5-3 h away). 1201
  already passes the gate's word pace (busiest 30 s 114). Known deviation, for the lead: 1201 has piano breaks after
  verses 1-2, chorus 1 and the bridge, but none after verse 3 (straight into chorus 1) or verse 4 (straight into
  chorus 2). v2's lyric (song/lyrics.sing.v2.txt, runs capA2/capB2) is ready if a re-sing is wanted.
- Draft render 1 (1 sample, all 240 s): gate 1.8 hard changes/min, median shot 8.4 s, words 74/min, busiest 30 s
  114, **FAIL frozen 3.6 s at 2.5 s** (the intro before the dots are in). Cuts found at 16.7 s and 96.7 s (the
  downbeat swell at its peak, 4-5e-3 per frame) and 113.5 s (verse 4's fog edge racing outward as it cleared). Fixes:
  dots all arrive by ~4.5 s; the swell eases in over 0.24 s and is 24% (1.7e-3 per frame, between the frozen floor
  1e-3 and the cut line); the fog fades as a whole (2.6 s) at a fixed radius; the first step counts roll over
  instead of popping.
- Small type (labels, captions, chart text) halftoned into broken dots: it now draws on a second canvas that the HUD
  prints after the riso pass (crisp solid ink; engine/hud.ts setOverlay). Big numbers stay in the print.
- publish.json drafted (titles, description, chapters from data/timeline.json via app/scripts/chapters.ts, tags,
  AI disclosure, pinned-comment drafts, the hook Short's cut).
- 3:35-3:55 am: draft renders (1 sample) and the course gate. 16:9: PASS (0 hard changes, nothing frozen). 9:16
  first FAILED (19 cuts, median shot 2.0 s): in the tall frame the downbeat swell of the knobs and dots hit 4e-3 per
  frame on every downbeat of verses 2-3, plus the first step ticks (the random network's line swings a long way on
  steps 1-3, which is real) and a lyric swap. Fixes, portrait only: swell 17% on the dots and none on the knobs,
  line swaps 0.7 s, number fades 0.6 s; both aspects: steps 1->2->3 tick over 1.0 s. Portrait draft: PASS (0 cuts,
  nothing frozen). Knobs fade and thin as they fly into the ball (they had piled into a black square).
- 3:57 am: final renders started (tools/lock.sh render, 4 x 12 s segments, adaptive motion blur up to 108 sub-frames,
  shutter 0.2), 16:9 then 9:16. ~10 ms per frame at 1 sample; the intro runs ~19 fps per job.
- Thumbnails (scenes/thumb.ts, the real scene at a chosen time, small type off, <= 3 words): A "HOW AI LEARNS" (the
  ball in the valley, chorus 3), B "3,121 STEPS" (the untangled spirals), C "STEP TOO BIG" (the jump over the valley
  floor); all three read at 168x94.
- 4:10 am: the first 16:9 final FAILED the gate by one stretch (frozen 4.0 s at 147.6 s: the bridge's cut chart over a
  90%-dimmed landscape, the downbeat swell at exactly 1.0e-3). Fixes: in the bridge the panel now shows the too-big
  run itself from its first line (steps 0 -> 9 while the jump is explained, step 9 being the one cut open, then all
  5,000 on "it bounces"), the landscape dims to 80% and sways slowly in 16:9, the side-panel swell is 32%; the
  downhill arrow fades with the bridge (its head had been left floating over the chart). Both aspects re-rendered.
- 4:18 am, finals (all 14,400 frames, 240.0 s, adaptive motion blur 12/36/108 sub-frames, shutter 0.2):
  out/final/downhill.mp4 1920x1080p60 409 MB sha256 a5216d781daf...
  out/final/downhill_vertical.mp4 1080x1920p60 377 MB sha256 effddd37abcc...
  share copies (x264 slow crf 23, both under X's 512 MB): downhill_share.mp4 164 MB, downhill_vertical_share.mp4 151 MB.
  **Course gate PASS on both**: 74 sung words/min, busiest 30 s 114, 0 hard changes/min, median shot 240 s (one
  continuous scene), nothing frozen (longest near-still stretch 2.27 s in the 16:9 intro, none over 2 s in the 9:16).
  Master audio in the mux peaks -0.9 dBFS (no clipping).
- Hook Short (tools/publish/cut.py, publish.json downhill_hook): chorus 1 (81.996-99.329: measure, find down, step,
  again), the bridge's first two lines (143.329-151.329: ten times as big, jumps the floor), the outro (215.328-
  224.661: 3,121 steps, every dot on its side); every join on beat 1 of a bar; 34.7 s, 42 MB; joins click-free
  (sample jumps at the joins 0.09 vs 0.29 typical).
- Thumbnails out/publish/thumbs/thumb_{A,B,C}.jpg (1280x720): A "HOW AI LEARNS", B "3,121 STEPS", C "STEP TOO BIG";
  legible at 168x94 (out/wip/thumbs_168.png). publish.json: A is the default, all three listed for Test & Compare.
- `yt.py package downhill` (local only): out/publish/<variant>/meta.json, captions.srt, thumbnail.jpg. Nothing
  uploaded or posted.

## 2026-09-28: HUD alpha fix (the lead's follow-up)

- The lead (found by the meaning agent): engine/post.ts mixed the HUD as `h.rgb / max(h.a, 1e-4)`, but the HUD is a
  straight-alpha CanvasTexture, so every partial-alpha pixel was divided again and blew out to white or full ink: the
  lyric crossfades showed a white box (visible in the hook Short at "And do it again, and again"), and the paper
  strip's antialiased edge had a white fringe even when settled. Now `mix(col, h.rgb, h.a * hud)`. Before/after crops
  at 95.35 s (mid line-swap) and 64.2 s (settled line): out/wip/fade_cmp_*.png; after the fix the strips blend into the
  paper, no box, no fringe.
- render.ts hardened the way meaning's is: default server http://localhost:5931 (this project's), else a private
  server on a random port in 7200-7899 with up to 6 retries, and the page title must be "Downhill" (index.html still
  said the p(doom) title) or it throws.
- Re-rendering both finals under tools/lock.sh render (queued behind the tokens agent's renders).
- 8:19 am: re-rendered both finals with the fix (same flags; queued ~35 min behind the tokens and learning-to-write
  renders). **Course gate PASS on both**: 16:9 74 words/min, busiest 30 s 114, 0 hard changes, nothing frozen; 9:16
  the same except 1 hard change (0.2/min, median shot 120 s) at 90.87 s: the lyric swap into "Take one small step
  that way", now a real 0.7 s fade, which the gate counts once. Share copies and the hook Short (same segments,
  34.7 s) rebuilt. Frame-checked: the finals and the hook are downhill (title card, landscape, bridge cut, outro).
  downhill 409 MB d6783be52731;downhill_vertical 377 MB f50993f0f9f6;downhill_share 164 MB e315be7af0b3;downhill_vertical_share 151 MB 69728942d93d;downhill_hook 42 MB c80e23c0749e;
