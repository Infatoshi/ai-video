# DEVLOG (times Mountain)

## 2026-09-27: Next Token, v1

- 3:49 am: brief approved ("yup banger do it"): new genre, voice, topic and palette (SPEC.md "The brief").
  Forked `~/dev/cuda-video` to `~/dev/token-video` (git clone, origin removed); CUDA scenes, lyric
  versions and run configs dropped.
- Lyrics written topic-first (tokens -> vectors -> attention -> feed-forward -> logits/softmax/temperature
  -> append + KV cache -> memory-bound decode -> batching, speculative decoding -> end token). Every number
  in them is Llama 3.1 8B's config (32 layers, 32 heads, 4,096 wide, 14,336 feed-forward, 128,256 vocab,
  16 GB bf16). Sung-spelling file writes "K-V cache".
- 3:53 am: 16 takes on the GPU host's 3090 (the other GPU host's GPUs stay off overnight): capA = UK MC rapping verses, capB =
  sung melodic verses; both 174 BPM, F minor, 170 s. ~2.3 min per batch of 2 (offload mode). Done 4:13 am.
- Real data instead of invented numbers: Llama 3.1 8B Instruct + Llama 3.2 1B (draft) downloaded to the GPU host
  (official repos; the HF token has access). `llm/dump.py` records the chat tokens, the sampled answer with
  each step's top 10 / nucleus / uniform draw, logit lens, all attention rows, feed-forward activations at
  layers 0/15/31, embedding PCA of a word list, a speculative round at every answer position, and measured
  tok/s at batch 1 and 100.
- Engine: risograph post (three-ink separation by optical density, area-coverage halftone per ink at
  15/75/45 degrees, per-shot misregistration and paper seed, ink mottle, paper fibre; no bloom); portrait
  mode (`?aspect=portrait`, W/H 1080x1920); `f.ft` frame time so in-scene cuts never blend two shots under
  motion blur; global overlay (print marks, model readout from the story clock, the sung line on a paper
  strip, concept-word slams from `words.ts`); `story.ts` maps answer tokens to lyric words.
- Kit (`scenes/_kit.ts`): beat-locked shot cutter with section rates, `inkMat` (coverage follows a
  camera-fixed light; instanced variant with a per-instance hit channel), constant-width outlines,
  `Wash`, token rain, specks, token tiles. `_tower.ts`: the model as a building. Look test stills:
  `out/wip/look/`.
- rhythm.py tempo search now stays within 5% of the requested BPM (174 locked to 87 before); analysis
  tempo range per song (165-183 for this one).
- 4:13-4:36 am: scoring (Demucs -> Whisper large-v3 recall, Audiobox) and rhythm. Best: capA (rapped) 801 / 803 /
  805 / 807 recall 0.97-0.98, capB (sung) 901 / 907 / 905 recall 0.97-0.98 with slightly higher aesthetics.
  Picked **n1_capA/803**: rapped verses per the brief, recall 0.979, rhymes on the 8th grid 0.73, couplet parity
  2.9, and the clearest drum-and-bass shape (quiet 11 s intro, breakdown 97-105 s, a +20 dB drop at 104.8 s,
  quiet final chorus, loud outro to a hard stop at ~165.5 s). 801 and 907 staged in ~/clips for Elliot's ear.
  Mastered to -14.0 LUFS / -1.0 dBTP (two-pass loudnorm) -> audio/token.wav, .mp3, ~/clips/next_token_take803_rap_PICK.wav.
- Alignment on the GPU host (run_linux.sh, SONG = "token"): 173.995 BPM; the refine step walked "Type" back onto the
  intro (0.07 s) and ran "done" to the stop: pinned with FIX (10.6 s; end 143.4 s). Aligned words vs Whisper:
  median 0.06 s. Sections by bar in analyze.py (intro 0-8, verse1 8-24, chorus1 24-32, verse2 32-48, chorus2
  48-56, verse3 56-72, breakdown 72-76, drop 76-84, verse4 84-92, chorus3 92-100, outro 100-).
- The model run (4:39 am, after two fixes: no `accelerate` in the venv -> `.cuda()`; embeddings detached):
  **"There are 2 r's in the word 'strawberry'."** 15 tokens. Step 3's nucleus: "2" 69.7% / "3" 30.3%, u = 0.657
  -> "2". " strawberry" is one token in the question; the answer spells str | aw | berry. 42.8 tok/s at batch 1,
  1,805.5 at batch 100 (3090, HF eager, bf16). The 1B draft's 4 guesses were accepted at every position; the
  round shown is at position 10 (str aw berry '.), chosen by rule in dump.py. Story anchors: chorus 2's "pick"
  is step 3.
- 4:58 am: seven scene authors launched (tokens; vectors + ffn; finale + outro; attention; logits + append;
  memory + roofline; spec). intro and hook written by the lead as references (8 ms/frame at 1 sample).
- 5:00-6:25 am: author reports in; lead fixes from them: render.ts sheet cells follow the aspect (and
  the `PORTRAIT` flag is passed into page.evaluate), video mux points at audio/token.mp3, `shot()` counts
  cuts from the entry's cut (`ctx.cut`, the start without the transition overlap) and carries each cut's
  own section rate so a scene that spans a section change doesn't jump; hook2 / finale transitions mostly
  after their cuts (pre 0.25 / 0.2) so ffn's "power" and spec's "go" payoffs read; verse 1's strawberry slam
  is "1 TOKEN" (it is one token in the question); tokenRain 17 px / 600.
- Honesty fixes from review: a decode step reads 15.01 GB (16.06 GB of weights minus the 1.05 GB
  embedding table, of which one row is looked up), so per-token counters, 16.0 ms at 936 GB/s and 642 GB/s
  achieved (computed) replaced 16.06 GB / 687 GB/s in memory, roofline and spec; the batch-100 wall shows the
  greedy picks (the speed run was greedy) and stops at step 9 instead of typing the rest of the answer.
- `render.ts plates` renders clean frames (no HUD, no print: engine.plainOut) of every entry, 3 each, 16:9
  and 9:16 (public/plates*, plates.json); the outro band reprints them one per beat.
- Per-beat sheets (out/qa/beats*, 11 sheets x 48 frames each aspect): no empty frames; fixes from them:
  intro's top-down shot (read as a grey square) became an orbit with the tower's real dimensions; the
  chorus rider sits on the roof and turns '?' -> the picked token after "pick", facing the lens; the ffn
  city's base ink is heavier; roof and lobby lighter (solid black printed as heavy dot fields).
- 6:31 am: final render, 4 parallel x 12 s segments, adaptive motion blur up to 108 sub-frames, shutter 0.2
  (~3 fps per job: most frames reach the cap because something always moves).
- 6:41 am first full render (10 min); gate: median 1.04, p10/median 0.67, weak seconds 3 (intro low angle),
  47 (attention skyline), 66 (hook2's streak transition blurred a beat to paper), plus 165-169 (the intended
  blank paper after the stop). Fixes: intro drive base 1.1; attention's skyline shot dollies from the query
  back to the 67.1% strawberry bar; hook2 transition -> glitch 0.35 s. Re-render 6:45-7:05 am (both aspects).
- Gate after fixes (16:9): median 1.049, p10/median 0.69 (0.72 over the music), 1 s under the 0.55 floor
  during the music (s 3, 0.576 vs 0.577), 440 cuts. Same measure on the CUDA v5 share copy: p10/median 0.48,
  26 s under the floor, 218 cuts in 190 s. Weakest seconds left: 3 (intro), 129 (finale's floor-guess shot
  before its labels arrive), 66 (hook2's first shot, the roof close-up), 61 (ffn's floor close-up).
- The outro's full poster panned past the safe clip (it filled 96% of the width and the pan starts at 90%):
  fit at 84%, no pan. Re-rendered 156-170 s in both aspects and re-joined (pts steps uniform at the seam).
- Final (7:37 am), all 10,200 frames, 170.0 s, AAC:
  out/final/next_token.mp4 1920x1080p60 1.45 GB sha256 100ce84d1f76d21d407eadf1e92c79dc00d406f148cecc811b324427fa4cb0b9
  out/final/next_token_share.mp4 (crf 24) 498 MB sha256 fff4f0663463e138cedadc68b14ed257890ebfd7c3a5756321e401cc495a6072
  out/final/next_token_vertical.mp4 1080x1920p60 1.59 GB sha256 baf1e10a3be62662e5e44e6d83e13bff32c0567a145c55b96f856359e26be3d3
  out/final/next_token_vertical_share.mp4 (crf 25) 485 MB sha256 da67c3e9a975924706d161b00d6c959130f7530e3dbf796d7dda94ea41417f14
  Halftone barely compresses: crf 20 share copies were 710 / 773 MB; crf 24/25 keeps the dots and fits X's 512 MB.
  Gate plot: out/final/next_token.gate.png. Per-beat sheets: out/qa/beats/, out/qa/beats_portrait/.
- 2:10-2:25 pm: publish step. youtubeuploader 1.25.5 (checksum-verified release) with gog's OAuth client;
  Elliot consented and enabled YouTube Data API v3 on the OAuth client's Google project; token = channel Elliotcodes.
  publish/package.py (metadata, chapters from data/timeline.json, SRT from lyrics.json, poster thumbnail) and
  publish/youtube.sh (private upload). First try failed only on the disabled API (SERVICE_DISABLED).
- 2:29 pm: uploaded PRIVATE to Elliotcodes: https://youtu.be/OK-Rf8VqfBQ (16:9, category Education, custom
  thumbnail, English captions serving) and https://youtu.be/MoB0stv1vKg (9:16, a Short). youtubeuploader drops
  containsSyntheticMedia; set it with videos.update (both True now) and youtube.sh does it after every upload.
- 2:40 pm: engagement pass. Channel data (44 public videos, 138k views): the winners are concrete payoffs
  ("Building a ChatGPT Voice Assistant on Raspberry Pi", 23k) with a face + one object; the earlier "Claude
  Opus 4.6 Music -- AI-Generated Visualizer Demo" got 573. So the YouTube title leads with the hook, not the
  tool: "Why AI says strawberry has 2 r's (how LLMs work, as a music video)"; description opens on the
  question and the 2, hashtags #LLM #AI #MachineLearning, strawberry/how-LLMs-work tags. Thumbnails
  (scenes/thumb.ts): A strawberry + "AI: 2" (set on the video), B die 2|3 + 69.7%, C "HOW AI WRITES" + tower.
  Applied to both private videos with publish/yt_update.py.
- 2026-09-27 afternoon: moved into ~/dev/ai-video (monorepo; generic scripts in ../../tools). yt.py auth added
  comment + analytics scopes (Elliot's click; Analytics API enabled). Hook Short with tools/publish/cut.py: song
  16.21-21.73 (strawberry is one token) + 65.87-76.91 (chorus 2, the 2 vs 3 roll) + 162.42-166.21 (the poster to
  the hard stop), every join on beat 4 or a downbeat, 20.35 s, uploaded private https://youtu.be/JgXY8wtrp90
  ("The AI rolled a 2. The answer was 3. #Shorts"). Playlist PLVnaIVGCV8DA (private) holds both 16:9 videos.
- Thumbnails v2 (Elliot: "AI: 2" and the die aren't catchy): E = strawberry with a price tag "73700" (its token
  id in the question), "AI SEES" is on the video; D = sliced strawberry str|aw|berry and C = tower stay for Test &
  Compare. The API refused the swap (403: Codex had staged a Test & Compare set while private); Codex deleted the
  staged test in Studio and set E.
- Release (Elliot: "make your best judgment and you can upload it and go"): 8:00 am MT, one per day. Channel audience
  (365 d): mostly the US and India, mostly desktop, weekdays even; 8 am MT = 10 am ET = 7:30 pm IST. Mon Sep 28 CUDA video,
  Tue Sep 29 Next Token, Wed Sep 30 hook Short, Thu Oct 1 vertical Short. Set via the API (publishAt); Codex confirmed
  all four "Scheduled" in Studio, no unverified-project lock. Playlist PLVnaIVGCV8DA public. Test & Compare refused
  while not public ("Your video is ineligible because: Your video is not public"); staged tests deleted.
  After Wednesday: studio_prompts/go_public.md (Test & Compare, end screens, Shorts' related video).
- Elliot posted Next Token on X himself: https://x.com/elliotarledge/status/2104294957095997882 (Sep 27 1:39 pm MT;
  16,517 impressions, 231 likes, 119 bookmarks by 11:55 pm). The CUDA post was at 465,951 impressions, 2,121 likes,
  908 bookmarks by then. Pinned comments drafted per video in publish.json (lowercase, Elliot's ask).
- 2026-09-28 12:05 am: Elliot's go on the four pinned comments. Comments can't be posted before release (403
  "insufficient permissions" on the scheduled CUDA video), so one-shot launchd jobs post and pin each at 8:03 am MT
  Mon-Thu (tools/publish/schedule_comments.py; logs out/review/comment_<variant>.log).
- 2026-09-28 early am, TikTok: Elliot signed in to TikTok for Developers. Codex created app ai-video
  (production Draft) and sandbox ai-video-sandbox; both refuse to save
  without Terms/Privacy URLs and a Web/Desktop URL. Pages drafted for elliotarledge.com (Next.js app router,
  app/privacy and app/terms), not pushed.
- 12:40 am: elliotarledge.com/privacy and /terms pushed (Elliot's go; commit 730b455, build checked, live in ~30 s).
  Sandbox saved (description capped at 120 chars, no special characters). "Add account" for the target user signs
  TikTok out and needs Elliot's login. tools/publish/tt.py written (desktop PKCE with hex SHA-256, 64 MB chunks,
  inbox upload + status).
- 8:51 am: tt.py auth (sandbox, user.info.basic + video.upload), whoami = Elliot's account. Hook Short uploaded to TikTok
  drafts: 103,120,302 bytes in 2 chunks (206, 201), SEND_TO_USER_INBOX. First try failed "The chunk size is invalid":
  a single 103 MB chunk declared as 64 MB; files over 64 MB now split evenly into >= 2 chunks.
- 1:25 am: both Shorts in TikTok drafts (vertical share copy 484,572,979 bytes in 8 chunks; chunk count now rounds up),
  then scheduled on tiktok.com by Codex (Elliot's go on captions and times), AI-generated label on: hook
  7690483676160724244 Wed Sep 30 6:00 pm MT, vertical 7690484310033337620 Thu Oct 1 6:00 pm MT. TikTok Studio shows
  both with future dates, 0 views; public oEmbed returns 400 for both (not live yet).
