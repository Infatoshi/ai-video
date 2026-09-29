# DEVLOG (times Mountain)

## 2026-09-28: who-is-it

- Forked from next-token (engine + kit).
- 2:18 am: real data first (it decides what the song can say). `llm/attn.py` on the GPU host (score venv, Llama 3.1 8B
  Instruct from the HF cache, bf16, `.cuda()`, eager attention; 21 s under the lease `course-who-is-it`): five
  sentence pairs, each sentence run once as plain text after `<|begin_of_text|>`, full attention of all 32 layers x 32
  heads saved (out/llm/attn.npz, attn_meta.json), plus two behaviour probes per sentence. `llm/flip.py` (Mac) ranks
  heads by flip margin = min(dA, -dB), d = w(ref0) - w(ref1); clean = margin >= 0.05 each way.
- The brief's pair, at "it": **no head can flip**. "it" comes before "tired"/"wide" and each word only looks back, so
  the 1,024 attention rows at "it" are identical in both sentences (max |difference| 0.0). The flip happens one word
  later, at "tired"/"wide": 118 heads lean the right way in both sentences, 4 clearly (>= 5 points each way: L0 H11,
  L3 H31, L5 H17, L8 H29), 23 lean the wrong way, 883 don't switch. Best: **L8 H29** (0-indexed): "tired" -> animal
  11.1%, street 4.0% ("it" 19.5%, start token 22.5%); "wide" -> street 68.0%, animal 0.3%. The same head at "it"
  (both sentences): animal 8.3%, street 7.1%, start 60.5%. Mean weight on the start token over all heads ~75%.
- The model as a whole gets it right: "... too tired. The one that was too tired was the" -> " animal" 33.4% vs
  " street" 0.4%; "... too wide. The one that was too wide was the" -> " street" 43.1% vs " animal" 6.6%. Chat
  ("what does 'it' refer to? one word"): "Animal." (Animal 85.0% + animal 14.8%) / "Street." (Street 49.6% +
  street 23.4%, Animal 20.7%).
- Other pairs tried (all recorded in data/attn.json): trophy/suitcase big/small (at "it" identical again; at the
  adjective 3 clean, L13 H11 best, L8 H29 2nd; the chat answer is "Trophy." for both, the probe gets both right);
  "before" (the deciding word moved before "it": "Too tired, it" / "Too wide, it"): 1 clean flip at "it" itself, L8 H29
  again (14.1% / 4.0% -> 3.9% / 13.5%); "plural" (animal/streets vs animals/street): 16 clean at "it", best L8 H31
  28.3% / 0.7% -> 1.4% / 55.1%; "pronoun" (they vs it): 13 clean, L16 H22 46.7% / 4.3% -> 8.7% / 32.0%.
- Decision: the video keeps the brief's sentence and shows the real result: "it" can't tell yet (the same weights both
  times), "tired"/"wide" looks back and does the switching (L8 H29), few heads switch clearly, and the model's answer
  is right anyway. This is the beginner rule's own consequence (each word looks back only), so it teaches, not hedges.
- 2:23 am: lyrics (song/lyrics.sing.txt, 260 sung words: target <= 95 wpm over the sung span, verses <= 45 words so no
  30 s window passes 120). Run configs song/runs/w1_capA.json (neo-soul, male tenor asks / female alto answers,
  Rhodes, brushed drums) and w1_capB.json (swung lo-fi neo-soul R&B, both on the chorus), 75 BPM, Eb major, 250 s,
  seeds 401-408 / 501-508. Queued on the GPU host as one chain (gen A, gen B, score, rhythm) under the lease, tmux
  `course-who-is-it`, log ~/dev/cuda-song/who-is-it/chain_w1.log; behind course-downhill and course-learning-to-write.
- 2:26 am: lease gotcha (for the lead): `overnight-compute run` with an agent name whose ephemeral lease already
  finished loops forever on "course-who-is-it is already done" (every poll) instead of queueing. Fix: `overnight-compute
  clear --agent course-who-is-it` before each new `run`. Requeued 5th (behind downhill, learning-to-write, tokens,
  meaning), so the song lands late; building the app on approximate 75 BPM timings meanwhile.
- 2:30-2:52 am, build (on approximate 75 BPM timings from out/scratch/approx.py -> data/*.approx.json, replaced
  after alignment). Engine: `engine/attn.ts` (data/attn.json) replaces llm.ts/story.ts/words.ts; the HUD is now the
  course overlay only (print marks, "ML, SLOWLY  4/5", the sung line sliding in over 0.35 s and staying up until the
  next, a crisp source caption passed from the scene as `post.caption`); no slams. Song paths point at
  audio/whoisit.* (main.ts, render.ts, render-parallel.sh OUTNAME who_is_it, analysis/common.py SONG "whoisit",
  analyze.py tempo window 70-80 BPM).
- One scene for the whole song (`scenes/sentence.ts`), timeline entries only name the chapters (they touch, no
  transitions). The real Llama tokens as a sentence (column in 9:16), arcs as tapered ribbons sized by the real
  weights of L8 H29 with paper dots flowing from each word into the asker, the tired/wide roll, a reading cursor
  with the unread words ghosted, QUERY/KEY tags, the weights stacked into one bar "= 1", verse 3's pull-back from
  one head to its row to all 1,024 (each cell that head's real arcs, morphing), the probe line, the next-episode
  title. One number at a time (`Sentence.calls()`); the arc a number belongs to is lit, the others dim.
- Gate check on a 12 s test clip (the course gate's own measure): the first camera drift (~3 px/s) left 570 of 600
  frames "still" (d < 1e-3). A two-period float (~20 px/s) and a faster background wash: 167 of 360 still, longest
  run 1.2 s (limit 3 s). The title and outro text float with it.
- Per-frame cost 14 ms at 1 sample (render.ts perf); the Canvas layer draws once per output frame (perFrame), so
  sub-frames are identical and the adaptive sampler stops early. Clip renders were ~3-4 fps with other agents'
  renders on the Mac (load ~30).
- Thumbnails (scenes/thumb.ts, extends the scene): A "WHO'S IT?" over the real arcs from "wide", B "68%" (the street
  arc alone), C "4 OF 1,024" (the grid, 4 clean switches in pink). All read at 168x94 (out/publish/thumbs/small.png).
- 2:49 am: the song chain got the lease (not FIFO: whoever polls first); ~3.5 min per batch of 2 takes.
- 2:55 am: publish.json drafted (title "Who's "it"? How AI attention really works | ML, slowly #4", description
  opening on the two sentences and listing the real results, chapters per timeline entry, tags, AI disclosure
  (containsSyntheticMedia + the made-with line), lowercase pinned comments). Hook Short cut added after alignment.
  Fades that change a large area (the unread-word mask, the arcs retracting before a chorus) lengthened to 0.8-0.9 s
  so the gate can't read them as cuts; arc widths no longer jump while a chorus retracts them.
- 3:41 am: chain done (16 takes, 250 s each, ~192 s per batch of 2 in offload mode; lease released). Recall
  (Whisper large-v3 on the Demucs vocal): capA 403 0.982, 401 0.971 (best Audiobox PQ 8.20, but one female voice
  throughout), 405 0.964; capB 503 0.982, 505 0.982, 507 0.978, 501 0.975; 404/506/408 failed (0.43-0.71). rhythm.py
  used the takes' own bpm (75, window 71-79): all lock at 75 (402 at 73). Rhyme-grid numbers are low (0.2-0.42) for
  every take: at 75 BPM the neo-soul phrasing lands behind the beat, so they didn't decide anything.
- Who sings what (out/scratch/shape.py: median F0 of each line on the vocal stem, male < 210 Hz): **403 is the real
  duet**: verse 1 alternates female / male line by line (308 / 155 / 391 / 156 / 369 Hz), verse 2 and the bridge are
  mostly the male voice, the choruses female. 507 also trades voices; 503, 501, 401 are one female voice with a few
  male lines. Aesthetics 403: CE 7.61 CU 7.74 PC 6.38 PQ 7.87 (503 is equal, 507 lower: PQ 7.31).
- **Pick: w1_capA/403** (duet per the brief, recall 0.982). Runners-up for Elliot's ear in ~/clips/who-is-it/: capB 503
  (same recall, one voice, the most pace headroom) and capB 507 (the other duet). tools/song/pick.sh mastered it to
  -14.0 LUFS / -1.0 dBTP (audio/whoisit.wav, .mp3; ~/clips/who-is-it/who_is_it_capA_403_PICK.wav). Whisper put
  "The" at 0.0 over the intro: line 0 seeded at 14.0 s instead, ANCHORS (0,0,0) 13.6-14.9 s.
- The take sings the outro three times (204, 224, 237 s): the two repeats were added to lyrics/whoisit.src.js as
  sung, so the read-along line and the captions show them.
- 3:46 am, alignment on the Mac instead of the GPU host (the GPU queue was 3 agents deep, ~2 h; the analysis tools are
  Mac-native: Demucs on MPS 54 s, CTC emissions 30 s, vocal features 49 s, mlx-whisper turbo 24 s, align 45 s,
  analyze 10 s; models cloned (APFS clone, no copy) from ../cuda-roofline/analysis/.cache). The queued the GPU host job
  was withdrawn. FIX (15, 6) end 89.9 s: "start" is held to 89.9, refinement had run it to the next chorus.
  Aligned vs Whisper (prompted turbo): median |dt| 0.11 s over 267 of 297 words (the outliers are the outro
  repeats Whisper matched to the other pass). Tempo 75.003 BPM, first downbeat 0.323 s, 3.200 s per bar.
- Shape: intro 4 bars; verse 1 14.6-34.4 s; **5 bars instrumental**; chorus 1 51.5; verse 2 71.7-89.9; **4 bars**;
  chorus 2 102.7; bridge 122.2-142.1; **6 bars**; verse 3 161.3-179.4; chorus 3 185.9; outro 204.7-248; fade to 250.
  So each new idea (verse 1, verse 2, bridge) is followed by 4-6 bars without words; verse 3 runs into the last
  chorus after 2 bars.
- Pace from data/lyrics.json (the gate's own formula): 297 sung words over 233.6 s = 76 words/min; busiest 30 s
  (166.1-196.1 s: verse 3 into the last chorus) = **120**, exactly the ceiling (the gate fails only above 120).
  No margin: any re-alignment must be re-checked.
- 3:55 am: per-beat sheets (out/qa/beats/, out/qa/beats_portrait/, 313 beats each): every beat has the sentence or
  the grid on screen, no empty frames; the only paper-only moments are the lyric strip fading between lines. Outro
  reworked for its real length (the take sings it three times, 204-248 s): "NEXT · 5/5 Watch It Learn to Write" on
  the first "Next time", then it shrinks out of the way and the sentence comes back, flipping tired/wide, to the fade.
  Final chorus adds a 32 x 32 inset of every head ("ONE HEAD OF MANY", the 4 clean switches pink, L8 H29 boxed).
- Hook Short planned (publish.json who_is_it_hook): 13.92-25.92 (verse 1 to "...too wide") + 121.92-142.72 (the
  bridge: left to right, the same weights at "it", "tired" looks back, "wide" finds the street 68.0%); both joins on
  beat 1 of a bar, 32.8 s.
- Renders queued behind the Mac render lock (downhill held it) on this project's own no-HMR server (:5947, so a
  stray server on :5173 from another project can't be rendered by mistake): out/final/render_both.sh, 4 jobs x 12 s
  segments, `--samples 1` (the scene's 2D layer is drawn once per output frame, so motion-blur sub-frames would be
  identical; only the background wash moves between them).
- 4:03 am, first 16:9 final (on the old code): course gate FAIL, frozen 3.8 s at 14.5 s (the first sung words alone
  on paper) and 4.9 s at 210 s (the outro title holding), 2 cuts at 102.47/102.93 (the weight bar vanished for the
  0.4 s the chorus-2 arcs were fully retracted); 76 words/min, busiest 30 s 120, 0.5 hard changes/min. The gate
  counts a frame as still under a mean change of 1e-3 at 320 px, and ~60 of the 250 seconds averaged under it.
  Fixes: the camera float is ~30 px/s (two periods per axis), the wash drifts faster (amt 0.2); the bar no longer
  depends on the arcs being drawn; choruses 2 and 3 redraw at the real weights (no jump to uniform); the asker's
  pink fades with its arcs; the flow dots fade in; the title glides into a small header during verse 1 instead of
  fading in place; the outro title settles after 3.2 s. Clip checks at 12-21, 99-106, 207-216 s: longest still
  run 0.7 s, no cuts. The vertical render (old code) was stopped and both are re-queued.
- 4:26 am, 16:9 final (4 jobs x 12 s, ~24 fps per job, 2.5 min once it had the lock): course gate **PASS**: 76
  words/min, busiest 30 s 120, 0.0 hard changes/min, median shot 250 s (one continuous shot), nothing frozen;
  activity p10/median 0.53, 0 cuts. 4:32 am, 9:16 final: **PASS** with the same pace numbers, p10/median 0.63.
- Share copies (crf 24 / 25, -tune grain): 452 MB and 356 MB, both under X's 512 MB.
- 4:33 am, hook Short (tools/publish/cut.py, from the vertical): 13.92-25.92 + 121.92-142.72 (snapped 13.923-25.922,
  121.918-142.717), 32.8 s, 50 MB; both joins on beat 1; audio level continuous across the 12 ms crossfade at 12.0 s.
- `yt.py package who-is-it` (local only): meta.json, captions.srt and the thumbnail per variant in out/publish/;
  chapters 0:00 Intro, 0:14 Who's "it"?, 0:51 Attention, 1:12 Query and keys, 1:42 Attention again, 2:02 left to
  right, 2:41 All 1,024 heads, 3:05 Attention all together, 3:24 Next. Nothing uploaded or posted.
- Finals (sha256 prefix): out/final/who_is_it.mp4 852,947,091 B 86f38de59f20fcd7; who_is_it_share.mp4 452,050,725 B
  575c6060bb2040e3; who_is_it_vertical.mp4 746,356,042 B 71644d394e3fdfcb; who_is_it_vertical_share.mp4 356,443,244 B
  dae4127faa9d6952; who_is_it_hook.mp4 50,320,396 B ad37e0cab98afbff. Thumbnails out/publish/thumbs/thumb_{A,B,C}.jpg.
- For the lead: (1) the overnight-compute "already done" loop (2:26 am entry); (2) a render.ts without `--url` uses
  any server on :5173, whichever project it belongs to; (3) the analysis pipeline runs on the Mac in ~4 min when the
  GPU queue is long.

## 2026-09-28 (morning): HUD fade fix (from the lead)
- post.ts mixed the HUD as `mix(col, h.rgb / max(h.a, 1e-4), h.a * hud)`, but the HUD layer is straight alpha
  (CanvasTexture, premultiplyAlpha off): dividing again blew partial-alpha pixels out, so every HUD fade popped.
  Now `mix(col, h.rgb, h.a * hud)`. Mid-fade stills (out/wip/fade_lyric.png, t = 18.35 s, the lyric crossfade):
  before, both lines' paper strips printed as bright white boxes with a hard ink rule; after, the strips blend into
  the paper at half strength and the two lines cross-fade. The caption fade (51.4 s) changes only slightly (its
  ink is dark either way).
- render.ts hardened like meaning's: a server is reused only when passed with --url; otherwise a private no-HMR vite
  on a random port 7200-7899 (clear of 5173, 5300-5800, Postgres 5432 and Chrome's unsafe ports), up to 6 tries; and
  it throws unless the page title is "Who's It? (ML, slowly #4)".
- 8:23-8:29 am: both finals re-rendered under tools/lock.sh render (same flags: 4 x 12 s, --url :5947, --samples 1;
  the lock went to four other renders first, ~50 min wait). Course gate PASS on both: 76 words/min, busiest 30 s
  120, 0.0 hard changes/min, median shot 250 s, nothing frozen, 0 cuts (p10/median 0.53 and 0.63). Share copies
  and the hook Short rebuilt (same segments, 32.8 s). Frame check: 16:9 at 18.35 s shows the fixed lyric crossfade
  over "Who's It?"; 9:16 at 140 s the 68.0% street arc; the Short at 20 s the bridge. New files (size, sha256 prefix):
  who_is_it.mp4 852834293 f937c9378791d1e0
  who_is_it_share.mp4 451906321 37793ccf6ab4701c
  who_is_it_vertical.mp4 746283180 f6a2cf7c704accc5
  who_is_it_vertical_share.mp4 356483588 29351e2c99abb4e3
  who_is_it_hook.mp4 50304661 6d978a2961a0be5c
