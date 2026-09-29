# DEVLOG (times Mountain)

## 2026-09-28: learning-to-write

- Forked from next-token (engine + kit).
- 2:16 am: course agent started (brief approved by Elliot, AGENTS.md "The course" #5). Song file renamed to
  audio/write.{wav,mp3} (main.ts, render.ts, render-parallel.sh + OUTNAME learning_to_write, analysis/common.py
  SONG = "write").
- 2:20 am: real data first (the chorus lyrics are the model's samples). `train/train.py`: a character-level GPT
  written from scratch after nanoGPT's shakespeare_char recipe (6 layers, 6 heads, 384 wide, 256-char context,
  dropout 0.2, AdamW 1e-3 cosine to 1e-4, batch 64, 5,000 steps, bf16), tiny Shakespeare (karpathy/char-rnn,
  1,115,394 chars, sha256 86c4e6aa...565ed), 90/10 split (the validation text is The Taming of the Shrew and The
  Tempest; ROMEO never appears in it). Records every step's loss and wall clock, train/val loss on 20 fixed
  batches, and 400 chars after the fixed prompt "ROMEO:\n" (temperature 1, sample seed 42 every time) at steps
  0 1 2 3 5 10 20 30 50 100 200 300 500 1000 1500 2000 3000 4000 5000, with checkpoints for `train/probe.py`
  (CPU: per-character surprise on the unseen line "Good morrow, neighbour Baptista."). CPU smoke test on the GPU host
  (3 steps): 10,745,088 parameters (10,646,784 without position embeddings, nanoGPT's 10.65M), 65-character
  vocabulary, step-0 loss 4.29 (ln 65 = 4.17). GPU run queued on the lease at 2:24 am behind course-downhill.
- 2:40 am: lease queue on gpu0: course-downhill running (their song takes), this project first in line, then
  tokens, meaning, who-is-it. To keep the GPU busy and skip a second wait, the queued training command hands its
  lease straight on to the song (`train/after_train.sh`, called by train.py after the full GPU run):
  `song/make_sing.py` fills `song/lyrics.sing.tmpl.txt` from the run (chorus excerpts = the start of what the model
  wrote after "ROMEO:" at steps 100 / 500 / 5000, whole words, 26-50 characters, sung as two lines; {N} = wall-clock
  minutes; {L0}/{L1} = validation loss at step 0 and at the end), then 16 takes (song/runs/capA.json female lead,
  capB.json male lead; gospel, 80 BPM, Ab major, 228 s, seeds 5101-5108 / 5201-5208), score.py and rhythm.py.
  Excerpt rule fixed before seeing the samples (the first words it wrote), so no cherry-picking.
- App: one continuous scene (`scenes/world.ts`): a world of printed sheets (the page with the model's writing, the
  weights, the book, the guess, the surprise, the loss curve, the chorus's loop ring) and a camera that moves on
  lyric lines; `clock.ts` maps song time to training step (finished model in the intro, rewind to step 0 on the
  hook's "ago", runs through each chorus's definition to its checkpoint); `engine/run.ts` loads data/run.json +
  probe.json; HUD = course marker, STEP n / 5,000, the sung line (crossfades, stays up); no slams. llm.ts,
  story.ts, words.ts removed. Dev layout checked with a 30-step CPU run and fake timing (out/dev, never shipped).
- 3:05 am: lease acquisition is a race, not FIFO (who-is-it took it at ~3:04 though queued last); still waiting.
  Chain extended (still one lease): `song/judge.py` ranks the 16 takes (word recall on the fixed lines, which
  Whisper can hear, plus how closely the excerpts' letters were sung: Whisper can't score gibberish as words;
  >= 210 s, 76-84 BPM) and writes the pick; then mastering (-14 LUFS / -1 dBTP, as tools/song/pick.sh),
  src_from_words, and `analysis/run_linux.sh` (Demucs, CTC, Whisper, align, analyze). The pick is checked by
  hand afterwards; a different pick re-runs mastering + alignment.
- First draft gate (fake timing, 0-75 s at 1 sample): 8 "cuts" = camera moves (an inOutCubic move from rest
  quadruples its frame difference within the gate's 1 s median window) and 4 frozen stretches. Fix: moves ease with
  a sine (speed grows linearly from rest) and the camera always circles 48 px on screen (~27 px/s): 0 cuts,
  one frozen 3.1 s over a near-blank page (radius raised after).
- 4:45 am: lead's render hazard (every fork's vite defaults to 5173): `app/scripts/render.ts` no longer renders
  whatever answers on 5173; without --url it always starts a private no-HMR vite on a random port 6100-6899 and
  checks the server is this app (src/scenes/world-panels.ts must come back as JavaScript: vite answers a missing
  file with index.html and 200). Finals get a frame check by eye before they count. Lease note: a done lease
  under the same agent name never re-acquires; reschedule/release it first.
- Lease race lost three times (who-is-it 3:04, tokens 3:41, meaning 4:28; queued since 2:23). Probe indexing
  fixed (chars[i] is the guess for character i of the line); verse 2's example character is chosen from the data
  (the letter inside a word whose right-answer chance grew most from step 0 to the end).
- 5:56 am: lease acquired (3.5 h in the queue). **The run** (train/run/run.json, RTX 3090, torch 2.10, bf16):
  10,745,088 weights, 5,000 steps in 200.07 s wall clock (3.33 min, of which 22.6 s evaluation and samples).
  Loss (20 fixed batches, no dropout): step 0 train 4.2879 / val 4.2821 (ln 65 = 4.17); val lowest **1.4652 at step
  2,000**; step 5,000 train **0.6147**, val **1.6956**. The real result differs from the brief's hope ("words by the
  last chorus", fine) in one way: past step 2,000 it overfits, the held-back loss rises from 1.47 to 1.70 while the
  practice loss keeps falling. The video says so (bridge lyric patched in place on the GPU host before the generator read
  it, ~6:00 am: "On the plays it practised, the loss fell to point six, / On a play it never read, it fell to one
  point five, then rose, / Past two thousand steps it learned the lines by heart"; template + make_sing.py updated
  to match, output identical). Takes checked: 5101.json carries the patched lyrics.
  Samples after "ROMEO:" (the same seed): step 0 `dF3u :';RnXbzDP'CnT---`; step 100 `I\nO: asafnte tist te othuared
  tharnd he, preckn,`; step 500 `It tile first; 'tis good a moderal.`; step 1000 `It in the temper, thou dost, but
  and leave:`; step 5000 `I cannot not sup. Tell him, do you know,\nYou shall be amended for lawful knees`.
  Sung excerpts (rule above): step 100 "I O: asafnte tist te / othuared tharnd he, preckn,", step 500 "It tile first;
  'tis / good a moderal.", step 5,000 "I cannot not sup. Tell / him, do you know, You shall". Hook: "Three minutes
  ago" (3.33 min rounded; the screen says 3.3).
- Probe (CPU, train/run/probe.json): on the unseen line "Good morrow, neighbour Baptista." the average loss per
  character goes 4.330 (step 0) -> 3.009 (100) -> 1.811 (500) -> 1.042 (1000) -> 1.024 (1500) -> 1.292 (5000): the
  overfitting shows there too. Verse 2's example is the "r" of "neighbour": 1.75% at step 0, 16.2% at step 100
  (its top guess then was a space, 30.5%; loss 1.82), 83.4% at 1000, 100.0% at 5000. One step moves 99.99999% of
  the weights; at step 1 (warmup lr 1e-5) the median move is 0.0000099 on a median weight of 0.0074.
- 6:16 am: mistake: train.py ran after_train.sh as a child process, so the finished training process kept ~2 GB of
  the 3090 while ACE-Step ran; the 3rd and 4th batch of each caption ran out of memory. 8 takes instead of 16
  (5101-5104 female lead, 5201-5204 male lead). train.py now exec()s the chain (same pid keeps the lease, the CUDA
  context goes). The chain continues (score, rhythm, pick, alignment) on the 8; the missing 8 seeds get a
  supplementary run only if the 8 don't give a good take.
- 6:25 am: **song pick: capA/5103** (female lead), chosen by `song/judge.py`'s rule and checked: word recall 0.916
  overall (score.py; its misses are mostly the gibberish), 0.936 on the fixed lines, excerpt letter similarity
  0.62 / 0.69 / 0.97 (step 100 / 500 / 5000: the gibberish is sung close to its letters; the words line almost
  exactly), 80.0 BPM, rhyme_grid 0.46 (best of 8), Audiobox CE 7.22 PQ 7.46, 228.0 s. The build is real: loudness
  of the last third minus the first third +8.6 dB (per 20 s: -37 dB at the start rising to -19 dB at the last
  chorus, back to -31 dB in the one-voice outro). Structure (out/takes/5103_structure.png): verse 1 from ~11 s,
  a ~15 s instrumental after it, chorus 1 ~46-62 s, ~12 s instrumental, verse 2 ~72-95 s, ~12 s instrumental,
  chorus 2 ~107-128 s, bridge ~132-160 s, ~16 s instrumental, chorus 3 ~176-196 s, outro to ~220 s.
  Runners-up staged in ~/clips/learning-to-write/: capB/5203 (male lead; recall 0.912, fixed lines 0.946,
  excerpts 0.43 / 0.77 / 0.89, the biggest build +10.2 dB, 78.5 BPM, rhyme_grid 0.21) and capA/5101 (female;
  0.891, excerpts 0.43 / 0.73 / 0.95, 78.7 BPM, no build: -0.2 dB). Gibberish lines scored by letter similarity,
  not word recall (Whisper can't recall words that aren't words).
- 6:28-6:35 am: alignment. The chain's run_linux.sh stopped at whisper_run.py (no prompt entry for "write"); Demucs
  and CTC were done. Whisper re-run in a 20-min lease (rescheduled the done agent first), align + analyze on CPU
  outside it. Pins: ANCHORS line 0 at 9.6-11.0 s (the voice enters ~10.3 s; Whisper put "Three" at 0.0), FIX
  chorus 2's "Guess" 106.7 s (the aligner had pulled it back to 95.5; Whisper 106.66). The prompted Whisper run
  hallucinated its own prompt over the instrumental, so align.py's cross-check uses the plain run for this song.
  Check: aligned word starts sit a median 0.033 s from the nearest vocal onset (Demucs stem); Whisper's plain
  word times lead them by 0.30 s median (0.063 s from their nearest onsets but systematically early on this
  slow, held singing), so the aligned times stand. 80.016 BPM, first downbeat 0.45 s, 304 beats; sections by bar
  in analyze.py (intro, verse1 9.4, inst1 33.4, chorus1 45.4, verse2 69.4, inst2 96.4, chorus2 105.4, bridge
  129.4, inst3 162.4, chorus3 171.4, outro 195.4-228.0). Mastered -13.9 LUFS / -1.0 dBTP.
- 6:46 am: first full draft (1 sample, render lock): out/final/draft1.mp4, frame-checked as this episode. Course
  gate PASS: 68 sung words/min (100 in the busiest 30 s), 0 hard changes/min, one continuous shot (median 228 s),
  nothing frozen; activity p10/median 0.55. Build fixes from the per-beat sheets (out/qa/beats*): the step-0 page
  types from "so this is what it wrote"; long camera moves fly (pull back mid-move so both ends are in view)
  instead of whipping; ring labels step back in the page close-ups; the excerpt framing keeps "STEP N, VERBATIM"
  in frame; the outro ends on a slow pull-back over the whole world with the NEXT TOKEN card. Thumbnails
  (scenes/thumb.ts -> out/publish/thumbs/thumb_{A,B,C}.jpg): "3 MINUTES" (step 0 vs step 5,000), "GOSPEL
  GIBBERISH" (the step-100 excerpt), "82 MILLION GUESSES" (the two loss curves), all legible at 168x94.
- 6:48 am: portrait close-ups of the excerpts framed tighter (the page text fills the width), portrait weights
  moved up out of those frames. Short check (out/wip/short_check*.png).
- 7:08-7:21 am: finals under the render lock (4 x 12 s segments, adaptive motion blur up to 108 sub-frames,
  shutter 0.2; ~4 min per aspect once the lock was free), frame-checked by eye as this episode:
  out/final/learning_to_write.mp4 1920x1080p60, 228.0 s, 849,172,476 bytes, sha256 8a4d7da5...c59ae
  out/final/learning_to_write_vertical.mp4 1080x1920p60, 228.0 s, 970,945,395 bytes, sha256 381fc3dc...6b0
  Course gate PASS on both: 68 sung words/min, 100 in the busiest 30 s, 0 hard changes/min, median shot 228 s
  (one continuous shot), nothing frozen; activity p10/median 0.55 (16:9) and 0.61 (9:16). Plots:
  out/final/*.gate.png.
  X share copies: learning_to_write_share.mp4 (crf 21, 327 MB, sha256 fc3ec72b...ce53),
  learning_to_write_vertical_share.mp4 (crf 22, 339 MB, sha256 53fae33f...e7f4).
- 7:23 am: hook Short (tools/publish/cut.py, from the vertical): out/final/learning_to_write_hook.mp4, 34.5 s,
  segments 6.45-15.45 (the finished model writing, "Three minutes ago" and the rewind), 57.44-69.44 (the step-100
  gibberish sung verbatim), 183.41-196.91 (the step-5,000 words), every join on a downbeat; audio joins checked
  (no step above the take's own 99th percentile). publish.json drafts (hook-first titles "... | ML, slowly #5",
  description with chapters, tags, AI disclosure, lowercase pinned comments) packaged locally with
  `yt.py package` (out/publish/<variant>/meta.json, captions.srt, thumbnail.jpg). Nothing uploaded or posted.
- 7:30 am (lead's fix): the HUD mix in post.ts divided the straight-alpha HUD texture by alpha again, so every
  partial-alpha pixel blew out: mid-fade the sung line's paper strip printed as a pure white box and the ink went
  grey (out/wip/hudfade_before vs hudfade_after at 10.0 s and 16.12 s). Now `mix(col, h.rgb, h.a * hud)`: the strip
  fades in over the paper, the ink fades. render.ts: private server on a random port 7200-7899, up to 6 tries, and
  a page-title check ("Watch It Learn to Write", index.html's title was still "I'm Upping My P(doom)").
  Re-rendering both finals under the lock (queued behind tokens at 7:32 am).
- 7:53-8:06 am: both finals re-rendered with the fix (same flags), frame-checked as this episode (mid-fade frames at
  10.0 s and 16.12 s: no white strip). Course gate PASS on both: 68 words/min, 100 in the busiest 30 s, 0 hard
  changes/min, median shot 228 s, nothing frozen; p10/median 0.55 (16:9) and 0.61 (9:16).
  learning_to_write.mp4 sha256 6b3a45aa...e003; learning_to_write_vertical.mp4 770c000d...b7; share copies rebuilt
  (327 MB b6b8c34e...f98a, 340 MB e6693ac2...a947); hook Short re-cut on the same segments (34.5 s, 9efb73cf...05cd).
  publish.json keeps the lead's vertical-title fix (4458dbf); packaged again locally.
