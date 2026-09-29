# tokens: ML, slowly #2/5

## The brief
Elliot, 2026-09-28, on the five-episode plan (after beginners said the first two were too fast): "i trust your
vision on these. fire off a subagent for each". The approved episode, as proposed:

the model never sees letters. The Llama 3.1 tokenizer's real split of the strawberry prompt, and a small BPE tokenizer trained on the song's own lyrics shown one merge at a time. One line of letters snapping into chunks. Motown girl group, call-and-response, ~100 BPM. Hook: the numbers the model gets instead of the word.

Series rules: ../../AGENTS.md "The course" (pace rules, course look, parallel-agent rules). Template and reference
for every stage: ../next-token (its SPEC.md and DEVLOG.md). Real numbers only: name the source of every number on
screen.

## Topic arc (one idea: the model gets a list of numbers, not letters)
1. The example (verse 1): "How many r's are in strawberry?" Llama 3.1 8B answered 2 (Next Token's run). Before
   it reads, the question is cut into 8 chunks and each chunk is swapped for a number: that list is all it gets.
2. The definition (chorus, same words every time): "A token is a chunk of text / Every chunk gets a number / The
   model reads the numbers / Never the letters". Each return adds a layer (1: the question; 2: + our own
   tokenizer's numbers on a line of this song; 3: + every piece on the sheet turning into numbers).
3. Where chunks come from (verse 2): byte-pair encoding, run for real on this song's lyrics, one merge at a time
   (e+r first, 64 times), until the words sung most are whole; Llama's own chunks were made the same way: 128,256.
4. The payoff (verse 3): " strawberry" is one chunk, one number, 73700; no r in that. Asked 100 times: as one
   chunk it said 2 (67 of 100); with the letters spaced out (every letter its own token) it said 3 (90 of 100).
   Claimed: what the model was given, and what it answered. Not claimed: any mechanism beyond that.
5. Same letters, different numbers (bridge): no space in front -> str|aw|berry (how it wrote the word in its own
   answer), capital S -> 89077.
6. Outro: what does a number mean? Next: #3 "Meaning Is a Direction".

## Real data (data/tok.json; sources in DEVLOG)
- Llama 3.1 tokenizer (meta-llama/Llama-3.1-8B-Instruct, HF cache on the GPU host, CPU): vocabulary 128,256 = 128,000
  regular + 256 special. Question = 8 tokens `How ·many ·r 's ·are ·in ·strawberry ?` = 4438 1690 436 596 527 304
  73700 30 (43 with the chat template). " strawberry" 73700 (1 token); "strawberry" str|aw|berry 496 675 15717;
  " Strawberry" 89077; " STRAWBERRY" STR|AW|B|ERRY; " s t r a w b e r r y" 10 tokens (274 259 436 264 289 293
  384 436 436 379). Others: " Motown" Mot|own, " ChatGPT" Chat|G|PT, " Llama" L|lama, " tokenization" token|ization.
- Spelled-letters test (Llama 3.1 8B Instruct, RTX 3090, bf16, T 0.6, top-p 0.9, seed 7, 100 answers each, the
  total each answer states): word as one token: 67 said 2, 33 said 3 (greedy: 2). Letters spaced out: 90 said
  3, 2 said 2, 8 counted out loud past the 40-token cap (greedy: 3).
- Our BPE (tok/bpe.py) on the sung lyrics: 330 words, 119 unique, 1,333 letters, 25 symbols; 140 merges until no
  pair repeats; vocabulary 165. Merge 1 e+r x64, 2 t+h x52, 3 th+e x35; "the" whole at 3, "number" 7, "letters"
  22, "strawberry" 97.

## Song
"Never the Letters": Motown girl group, call and response (the group answers every line), 100 BPM, Bb major, 260 s
takes. Lyrics `song/lyrics.txt` (annotated) and `song/lyrics.sing.txt` (sung). Runs `song/runs/m1_capA.json`,
`m1_capB.json` (seeds 201-208, 301-308). **Pick: m1_capA/207** (recall 0.945, 102.07 BPM, vocal 10.35-228.6 s,
instrumental outro to 260 s; 325 sung words, 89.4 words/min, busiest 30 s 114). Runners-up in ~/clips/tokens/:
m1_capA/205, m1_capB/303. Master: audio/tokens.wav (-14.0 LUFS, -0.9 dBTP).

## Look
The Next Token riso engine as forked (paper, fluorescent pink, riso blue, ink; halftone print, misregistration).
Pink = the thing being taught while it is sung (the pair being glued, the r's, the chunk just formed); blue = the
numbers (ids, tallies, the vocabulary); ink = letters, outlines, labels. HUD: "ML, SLOWLY 2/5" top left, the sung
line top centre (crossfades, holds), a source line bottom left. No slams.

## Shot plan (one scene, one sheet, one camera: app/src/scenes/world.ts)
The whole song is one continuous scene. Every idea lives at a station on one sheet of paper and stays there; the
camera flies between stations (far moves zoom out on the way) and floats slowly on a circle between moves, so
nothing freezes and nothing cuts. Labels draw in over 0.5 s and stay up.

| section | station | what moves |
|---|---|---|
| intro | Q | "Tokens" title; the question typed letter by letter as type sorts; the model box; its answer (2) |
| verse 1 | Q | r's counted 1 2 3 (pink); "it never got the letters" (crossed arrow); on "cut" the letters fuse into Llama's 8 chunks; on "swapped" each chunk flips (split-flap) to its id; "8 chunks, 8 numbers"; on "shown" the numbers slide into the model as slips |
| chorus (x3) | Q | the definition built in order: brackets + "token = a chunk of text", slips + "every chunk gets a number", slips into the model + "reads the numbers", letters dim + "never the letters" |
| instrumental 1 | Q | the numbers inside the model light one by one |
| verse 2 | B, V | a line of this song ("the model reads the numbers never the letters") breaks into letters; the most frequent pair (e r, 64) glows; merges 1, 2, 3 on "glue", "Count", "glue"; 4-29 through "Till the words..."; the camera flies to V: the real vocabulary in id order around " strawberry" (id 73700), then out to all 128,256 |
| chorus 2 | Q + B | the definition again, plus our tokenizer's ids on the song line |
| instrumental 2 | B | merges run until every word in the line is whole, then they flip to our ids |
| verse 3 | S, T | " strawberry": "a space" + fuse into one chunk; flips to 73700 digit by digit on "seven three seven oh oh"; ghost letters, "no r in 73700"; the test: 73700 -> model -> 67 of 100 said 2; spaced letters -> model -> 90 of 100 said 3; back to the chunk with its r's in pink; flips to the number again |
| bridge | D | " strawberry" 73700 / its own answer "·word ·' str aw berry '." (str, aw, berry light up as sung) / " Strawberry" 89077; "same letters, different numbers" |
| chorus 3 | Q -> all | the definition on Q, then the camera pulls back over the whole sheet and every letter on it flips to its number |
| outro | S | "what does a number mean?"; an arrow out of 73700: "next: meaning is a direction" (ML, slowly #3); fade to paper |

Layout: landscape puts S/T/D in a second column so the final overview stays compact; portrait is one column.

## Gates
- Course gate (`gate.json` = {"profile": "course"}): <= 95 sung words/min, <= 120 in the busiest 30 s, <= 6 hard
  changes/min, median shot >= 8 s, nothing frozen > 3 s outside the last 5 s.
- Per-beat contact sheets, both aspects.

>>> NEXT
Done 2026-09-28 (see DEVLOG for hashes): out/final/tokens.mp4 (16:9) + tokens_vertical.mp4 (9:16), share copies,
tokens_hook.mp4 (35.3 s Short), thumbnails A/B/C, publish.json drafts; course gate PASS on both finals. Nothing
uploaded or posted. Open: Elliot's ear on the song pick and the pinned-comment drafts.
