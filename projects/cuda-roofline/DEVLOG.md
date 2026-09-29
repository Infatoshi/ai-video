# DEVLOG

## 2026-09-26
- Forked mexicat/pdoom-video (MIT) engine; moved the original scenes to `ref/pdoom/` (removed before publishing). Reused the
  song's existing beat grid (132.007 BPM) and word-level lyric alignment in `data/`, so no audio
  analysis rerun was needed.
- Readout relabelled P(DOOM) → ACHIEVED / PEAK (`engine/hud.ts`), same step times.
- New 22-entry timeline, one plate per concept in book-chapter order (SPEC.md).
- Baseline timing on M4 Max: stills ~4 s per call incl. Chrome boot; a simple plate renders at
  9 ms/frame (1 sample).
- 3:38 pm MT: shared no-HMR vite on :5173 froze the scene glob before scene files existed ("scene
  module not found" for every new plate). Stopped it; render.ts now starts a private server per run.
- hook plate done by its author in ~7 min wall time; 7.5-7.9 ms/frame at 1 sample.
- post `flash` override is linear-additive: even 0.05 greys the whole frame (hostdev author).
- Plate authoring wall times (parallel, 14 authors from 3:28 pm): hook 7.1 min, flash 11.3, tensor 11.4,
  outro 11.5, mnist 11.8, hostdev 12.8, conv 13.6, optimize 15.0. All render at 7.5-11 ms/frame (1 sample).
- Transition QA sheet (conv→mnist, hook2→optimize, optimize→tensor, tensor→flash) renders in 2 s; the
  flatline at y=540 and hook2's orange bar hand off cleanly.
- 4:1x pm: session restart interrupted die, grid, pack+fuse authors; resumed from their transcripts.
- Final-render cost probe (adaptive samples, max 108, shutter 0.2, 1080p): 50-79 ms/frame average
  (hostdev 78.5, pipeline 55.3, hook4 52.7, optimize 49.9); sub-frames mostly 36-108. Whole song
  ≈ 9,400 frames × ~60 ms ≈ 9-10 min of GPU time, so the Mac is fine; no need for the other GPU host.
- Draft render (samples 1, veryfast, 4 pipelines) ran ~10 fps per pipeline while 11 authors were
  rendering concurrently: contention, not the engine.
- All 14 plates landed by 4:13 pm (die, grid, pack+fuse after the restart). Full-song QA: 1 fps contact
  sheets (out/wip/qa/full_*.png) and a --cuts sheet; every hand-off matches.
- render.ts video: added a throwaway warm-up still before streaming (a first still once came out solid
  red at 16.60). Verified a 20.0-20.5 segment's first frames are clean.
- Final render 4:14 pm: render-parallel.sh 4 20 --samples auto --max-samples 108 --shutter 0.2.
  ~5-6.6 fps per pipeline × 4 ≈ 24 fps total; GPU-bound on adaptive sub-frames (expected ~7 min).
- 4:16 pm final-render check: load avg 144 is x264 thread count (4 × 59 threads), CPU ~half used;
  4 pipelines × ~4 fps ≈ 16 fps total = one pipeline's 60 ms/frame, so the GPU (adaptive sub-frames)
  is saturated and more pipelines won't help. ETA ~10 min total (~4:25 pm). Capping --max-samples 36
  would save ~3 min at the cost of faint striations on whips; not worth a restart.
- v1 DONE 4:26 pm: out/final/cuda_pdoom.mp4, 1920×1080 60 fps, 9,399 frames, 156.65 s, AAC 320k,
  1.3 GB (CRF 16, grain). Render wall time 11 min 16 s (4:14:54 → 4:26:11), 2 batches of 4 segments at
  ~350 s each. Sub-frames: 12:2069 36:3481 108:3849. Seams at segment boundaries checked, continuous.
- Share copy: out/final/cuda_pdoom_share.mp4 (CRF 19, maxrate 25M, 448 MB) in 1 min 39 s.
- Timeline of the whole build: 3:21 pm first look at the reference → 3:31 engine forked + treatment →
  3:32-3:47 14 plates authored in parallel (3 finished after a 4:11 restart) → 4:13 QA → 4:26 v1.

## 2026-09-26 v2 song: "Chasing the Roofline" (music first)
- 4:36 pm MT: Elliot asked for a new song (CUDA lyrics in chapter order, voice synthesized by us),
  music first, then video. Source check: the p(doom) original was co-written by osmarks and Claude,
  generated on Udio (April 2024). Reference measured: 132 BPM, Eb major, -15.5 LUFS.
- Model: ACE-Step 1.5 XL-SFT (4B DiT, MIT) + 5Hz LM 4B planner, on the other GPU host (GPU 1) (~/dev/cuda-song).
  gen.py / score.py / runs/*.json live in song/ and are rsynced there.
- Setup was the slow part (~60 min): the other GPU host's 1 GbE link; the HF CLI's Xet path gave 4-8 MB/s vs 53 MB/s
  for plain HTTP, long-lived curl streams decayed to ~2 MB/s, aria2c per shard (x8) fixed it. uv pulls
  every PyPI wheel over one HTTP/2 connection (5-10 MB/s). The other GPU host's shared /data/hf is not writable:
  both scripts force HF_HOME/HF_HUB_CACHE to ~/dev/cuda-song/.hf.
- Speed once running: 8 takes of 170-188 s in ~5 min (LM codes 80-110 s + XL DiT 50 steps ~75 s per
  batch of 4). Scoring (htdemucs_ft vocals -> Whisper large-v3 sequential long-form -> word recall;
  Audiobox Aesthetics) ~6 min per 8.
- Metric note: WER blew up on p(doom) (1.27) from Whisper looping on "da da da"; word recall
  (hits / lyric words) is the ranking metric. p(doom) baseline: recall 0.83, WER 0.29,
  CE 7.46 CU 7.98 PC 6.91 PQ 8.29.
- 36 takes over 5 runs: t0 2B-turbo (best recall 0.84), a1 XL v1 lyrics (0.89), a3 XL v2 phonetic
  fixes (0.90: "Cuda, mem-copy", "M-NIST", "Q-BLAS", "Cut-lass", "All re-duce"), a4/a5 XL v3 with
  [Instrumental Break] + [Outro - instrumental] at 188 s (0.92 best). Remaining "misses" are mostly
  homophones (four/for, cut last).
- Pick: a4 seed 401 (recall 0.886, WER 0.146, CE 7.45 CU 7.93 PC 6.83 PQ 8.24): the only strong take
  with the full structure: 6 s intro, 2.5 s break after chorus 1, 18 s instrumental outro, clean stop
  at 185.5 s, hook contour consistent (0.9 st DTW). Mastered with a flat -1.2 dB to -14.0 LUFS,
  -2.1 dBFS peak: audio/roofline.wav (sha256 bed4255b...), audio/roofline.mp3, ~/clips/chasing_the_roofline.wav.
- 5:53 pm MT: song locked (77 min from the ask).

## 2026-09-26 v2 video on "Chasing the Roofline"
- 5:53 pm song locked; 6:08 pm alignment done (analysis/ made song-parametric: common.py SONG; stems
  from Demucs on the other GPU host; CTC fused6 + Whisper cross-check; no word under 0.6 conf; G/P/U, K/V, M/M/A,
  N/by/N, cut/lass carry syllable times). Grid 133.001 BPM, half-time groove, 14 sections.
- Timeline rewritten for the new lines; cluster now precedes pipeline (lyric and chapter order);
  beyond keeps 3 bars of the outro to resolve onto die's frame 0 (the loop point, DIE_FRAME0).
- ACHIEVED / PEAK now follows the book's CH6 GEMM ladder on the RTX 3090 (35.6 TFLOPS): 0 at start,
  0.01 / 0.26 / 0.70 / 0.78 on the four sung "roofline"s (naive 0.3, 1D 9.3, 2D 25.1, vectorized
  27.8 TFLOPS). v1's 0.15 / 0.42 / 0.81 / 0.99 contradicted the kernel labels (hook author caught it).
- 16 parallel plate authors, 6:10-6:36 pm (~10-25 min each). Lead fixes after review: mnist stamp
  ">100×" (book wording, not our 132×), pack "MAE ≤ s/2" (the 0.005 was CH9's INT8 example).
- Numbers spot-checked in the book this round: CH07 713 / 618 TFLOPS, 87% of cuBLAS; CH02 82 SMs on
  the 3090; CH04 async launches; CH12 "not automatically the nobler choice"; CH10 900/128 GB/s,
  8-12 vs ~476 GB/s busbw, 167,434 / 697,926 samples/s; CH08 N = 8192, 268 MB, 33.8 -> 2.18 ms.
- 6:39 pm final render started (render-parallel.sh 4 12, auto samples <= 108, shutter 0.2).
- 6:48 pm render done in 9 min 51 s (16 segments × 12 s, 4 parallel; faster than v1's 11 min for a
  longer song). out/final/cuda_roofline.mp4: 1920×1080 60 fps, 11,280 frames, 188.0 s, AAC 48 kHz,
  1.5 GB (sha256 e31eee72...). Seams at every 12 s checked frame-continuous. Share copy
  cuda_roofline_share.mp4 (CRF 19, maxrate 25M) 213 MB in 2 min 8 s (sha256 0c272d1d...).
- Total v2: 4:36 pm ask -> 5:53 pm song -> 6:54 pm video (2 h 18 min).

## 2026-09-26 v4 "beginner cut" (after Elliot's v2 review)
- 7:18 pm review: rhymes off the beat, GPU/cuBLAS sung badly, visuals not snappy/locked, kernel
  scenes too fast for beginners, curate instead of a line per chapter. Agreed plan: 9 beginner
  concepts, one idea per verse, repeated roofline chorus, music first.
- Lyrics v4 (`song/lyrics.v4.txt`): jargon only mid-line, no cuBLAS, AABB couplets of even length.
  16 takes (2 captions); all 8 caption-A takes >= 0.909 word recall (v1-v3 lyrics: many < 0.85).
- The other GPU host got busy (a watcher SIGKILLed the scorer): moved to the GPU host. Copy between the two hosts
  over the LAN (~110 MB/s), not the VPN relay (~150 ms). The GPU host's 3090 needs
  ACE-Step CPU offload with XL + 4B LM; scoring env pinned to torch 2.10 to install from uv cache.
- Rhythm scorer (`song/rhythm.py`): Whisper word times snapped to vocal onsets, tempo/phase fitted to
  mix onsets (beat-this frames are 20 ms-quantized, which drifted a median-based grid by ~3 s).
  Pick: b2-701 (133.0 BPM, line starts spread 0.59 beat, couplet rhymes within 1.1 beats, no
  stretched acronyms, recall 0.923, PQ 8.33). Two takes (604, 606) had drifted to ~100 BPM.
- Alignment on the GPU host (`analysis/run_linux.sh`); fixes for lines 0/10/41 (intro/break backing vocals).
  Every line is a 2-bar phrase starting ~1.2 beats before its bar; rhyme words land at bar +0.10.
- 13 scene authors (8:41-8:52 pm) under SPEC "v4" rules with `scenes/_lock.ts` (snap on the 8th grid,
  <= 90 ms, hold). Lead: title card in the intro, roof4 holds through the outro's first bar, engine
  resets canvas text/dash state per clear (a multi-scene lyric ghosting bug), removed v1/v2 scenes.
- 8:55-9:05 pm render (9 min 43 s): out/final/cuda_roofline.mp4, 1080p60, 11,400 frames, 190.0 s,
  1.0 GB (sha256 3bc7a999...); share copy 109 MB (1dbb7a57...). v2 kept as cuda_roofline_v2*.mp4.

## 2026-09-26 v5 "3D world" (after Elliot's v4 review)
- Ask: zoom in, more 3D, constant energy; a creative transition at every cut with big ones at the
  drops; model an H100 (or B200) die and a DGX node, take it apart, show bytes crossing flickering wires.
- Shared models (cf16a68): H100 SXM5 (GH100 die, 144 SM sites / 132 enabled, 6 HBM3 sites / 5 active,
  SM close-up) and DGX H100 (HGX baseboard, 8 SXM5, 4 NVSwitch, 144 NVLinks, teardown, traffic),
  checked against NVIDIA's DGX H100 user guide. Engine (e48a685): 9 transition kinds on overlapping
  entries, per-cut map in timeline.ts; `_cam.ts` beat-locked camera (drum-driven drive, kick pushes).
- 13 scene authors rewrote every scene in 3D on those models (8-10 ms/frame at 1 sample). Lead
  fixes: whip transition no longer lands shifted (0.3 screen width jump on the first frame after);
  transition centres aimed at each scene's focus (roof2 streak out of slow's MEMORY, roof3 shatter
  out of reuse's dot, memory dive out of the lit SM, bridge streak on tensor's dive target, outro iris
  out of roof4's orb, multi whip along quant's packets); tensor's rim light 2.2 -> 0.6 (top-of-frame
  glare). All 15 cuts checked with both neighbours loaded (out/wip/v5cuts/cuts1-3.png).
- Render probe: every frame hits the 108 sub-frame cap (the camera never stops), 4.2 fps per job.
- 10:23 pm final render started (render-parallel.sh 4 8, auto samples <= 108, shutter 0.2).
  v4 kept as out/final/cuda_roofline_v4*.mp4.
- 10:47 pm render done in 23 min 33 s (24 segments × 8 s, 4 parallel; most frames at the 108
  sub-frame cap, 1.5-3.5 fps per job). out/final/cuda_roofline.mp4: 1080p60, 11,400 frames, 190.0 s,
  AAC 48 kHz, 1.8 GB (sha256 25bfa13c...). Seams at every 8 s: frame diff at the join <= 1.35× its
  neighbours (no jumps). Share copy (CRF 19, maxrate 25M) 290 MB, 11,400 frames (sha256 ca5139f5...).

## 2026-09-27: YouTube
- Moved into ~/dev/ai-video. publish.json: "Why your GPU code is slow: CUDA and the roofline, as a music video",
  11 chapters (short scenes merged to clear YouTube's 10 s minimum), captions, thumbnails from real frames
  (A "YOUR GPU IS SLOW" over the 0.01 readout at 80 s; B "1% -> 78%" over the 0.78 vectorized-loads frame at 172 s;
  C the title card). Uploaded private https://youtu.be/_TJoJYLsEcQ with thumbnail A, AI disclosure on.
- Thumbnails v2: D = H100 with one SM lit, "YOU'RE USING 1%" (Elliot's pick, now on the video); E = all SMs lit
  "93x FASTER" (0.3 -> 27.8 TFLOPS, CH6) and B "1% -> 78%" stay for Test & Compare. scenes/thumb.ts.
