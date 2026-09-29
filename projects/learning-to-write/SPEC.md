# learning-to-write: ML, slowly #5/5 "Watch It Learn to Write"

## The brief
Elliot, 2026-09-28, on the five-episode plan (after beginners said the first two were too fast): "i trust your
vision on these. fire off a subagent for each". The approved episode, as proposed:

training. A small GPT trained from scratch on the GPU host on public-domain text, one fixed prompt sampled at checkpoints; the chorus lyrics are its real samples (gibberish first, words last) and the arrangement grows from one voice to a full choir as the loss falls. Gospel, ~80 BPM. Hook: "N minutes ago this model couldn't spell" (measured N). Hands off to Next Token.

Series rules: ../../AGENTS.md "The course" (pace rules, course look, parallel-agent rules). Template and reference
for every stage: ../next-token (its SPEC.md and DEVLOG.md). Real numbers only: name the source of every number on
screen.

## Topic arc (one idea: how a model learns to write)
Concrete first, then the rule, then the repetition, then the payoff.
1. Hook (intro + verse 1 line 1): the finished model writes after "ROMEO:"; on "N minutes ago" the page rewinds
   to step 0 and the text erases itself: it couldn't spell.
2. A model is numbers that guess what comes next; ten million of them, the weights, all random. What it wrote at
   step 0 (gibberish, typed out in the instrumental).
3. Chorus = the one definition (the same words every time): guess the next letter, see how wrong it was, nudge
   every weight a little, and do it again. Then the model's own writing from that chorus's checkpoint, sung
   verbatim and labelled "step N, verbatim" (step 100: gibberish; step 500: half words; step 5,000: words).
4. Verse 2, the rule: the book (tiny Shakespeare), one guess out of 65 characters, the right one, how surprised
   it was = the loss.
5. Bridge, the repetition: 16,384 guesses then one nudge = a step; 5,000 steps = 81,920,000 guesses. The real
   result (not the one the brief hoped for): the practice loss fell to 0.61, the loss on a play it never read fell to
   1.47 at step 2,000 and then rose to 1.70: past step 2,000 it was learning the lines by heart. The video sings it
   and shades it on the curve.
6. Outro: 3.3 minutes on one graphics card; the big ones learn the same way; next: Next Token.

## The run (data/run.json, data/probe.json, data/excerpts.json)
Recorded 2026-09-28 5:56 am MT. Headline: 10,745,088 weights, 5,000 steps, 200.07 s wall clock (3.33 min);
loss 4.28 at step 0 (both texts); practice text 0.61 at the end; held-back text lowest 1.47 at step 2,000, 1.70
at the end (overfitting after step 2,000). Samples after "ROMEO:": step 100 "I\nO: asafnte tist te othuared tharnd
he, preckn,", step 500 "It tile first; 'tis good a moderal.", step 5,000 "I cannot not sup. Tell him, do you
know,\nYou shall be amended for lawful knees". Verse 2's example: the "r" of "neighbour" in the unseen line, 1.75%
at step 0, 16.2% at step 100 (loss 1.82), 100.0% at step 5,000.
`train/train.py` on the GPU host's RTX 3090:
a character-level GPT written from scratch after nanoGPT's shakespeare_char recipe: 6 layers, 6 heads, 384
wide, 256-character context, dropout 0.2, AdamW 1e-3 (warmup 100, cosine to 1e-4), batch 64 x 256 = 16,384
characters (guesses) per step, 5,000 steps, bf16 autocast; 10,745,088 weights; 65-character vocabulary.
Data: tiny Shakespeare (karpathy/char-rnn, 1,115,394 characters, sha256 86c4e6aa...565ed), first 90% to train
on (1,003,854), last 10% held back (111,540: The Taming of the Shrew and The Tempest; ROMEO never appears in it).
Samples: 400 characters after "ROMEO:\n" at steps 0 1 2 3 5 10 20 30 50 100 200 300 500 1000 1500 2000 3000
4000 5000, temperature 1, the same sample seed (42) every time, so only the weights change between samples.
`train/probe.py` (CPU, checkpoints): per-character loss and the full 65-way guess on the unseen line
"GREMIO: Good morrow, neighbour Baptista.", a 65 x 48 corner of the character embedding table, one weight ("e",
number 1) at every checkpoint, and how far one step moves the weights.

## Song
Gospel, 80 BPM, Ab major, 228 s; one voice and piano growing to a full choir (section tags carry the build:
one voice -> organ -> backing singers answer -> drums, choir grows -> full gospel choir, handclaps). Lyrics:
`song/lyrics.txt` (annotated), `song/lyrics.sing.tmpl.txt` (what is sung, before the run's numbers and excerpts
are filled in by `song/make_sing.py`), the filled `song/lyrics.sing.txt`.

## Look
The course look (AGENTS.md): the Next Token riso engine, paper + pink + blue + ink; pink = what is being taught
while it is sung (the sung excerpt, the loss, the right answer, the ring's active label), blue = the data (the
prompt, Shakespeare, the weights' positive values, every step's batch loss), ink = structure and type. "ML,
SLOWLY 5/5" marker top left; STEP n / 5,000 top right; the sung line top centre (stays up, crossfades). One
print registration for the whole video (reg 0): no reprint pops.

## Shot plan (scenes/world.ts: one scene, no cuts; the camera moves on lyric lines)
World (16:9): the page in the middle (ROMEO:, what the model wrote next, STEP n), the loop ring around it
(the chorus's picture), the weights to the left, the book / the guess / the surprise to the right, the loss curve
beyond. 9:16: the same pieces stacked, the ring's labels on its diagonals.

| section | lyric | camera and what moves |
|---|---|---|
| intro | (instrumental) | title written in; down to the page: the finished model (step 5,000) types after ROMEO: |
| verse 1 | N minutes ago ... couldn't spell | the page; on "ago" the clock rewinds 5,000 -> 0 and the text erases itself |
| | A model is numbers ... / Ten million numbers ... | pan to the weights (65 x 48 corner, real values), 10,745,088 written in on "million" |
| | All of them random ... | back to the page; the step-0 sample types itself out through the instrumental |
| chorus (x3) | Guess ... / See ... / Nudge ... / And do it again | the page inside the ring; each label written in when first sung, pink while sung; the clock runs to the chorus's checkpoint (the text morphs cell by cell, the bead spins) |
| | (the excerpt) | push in to the excerpt: pink as sung, "STEP N, VERBATIM" |
| verse 2 | Its only book is Shakespeare | the book: 1,115,394 characters, the 90/10 split, real opening lines |
| | It guesses the next character ... | the guess: the unseen line typed to the blank, 65 bars |
| | Then it sees the real one | the right character revealed (pink), its chance written over its bar |
| | That surprise we call the loss | the surprise: the loss bar under every letter; the average written in |
| instrumental | | guess + surprise together while the clock runs: the bars shrink |
| bridge | Sixteen thousand guesses ... that's a step | the weights shimmer as the clock runs; one weight circled, its value changing |
| | Five thousand steps ... | fly to the loss curve: the practice loss (blue, evaluated every 250 steps) drawn to step 5,000 as the clock runs |
| | On the plays it practised ... point six | 0.61 written in at the end of the blue line |
| | On a play it never read ... then rose | the held-back loss drawn in pink; its lowest point circled: 1.47 at step 2,000 |
| | Past two thousand steps it learned the lines by heart | everything past step 2,000 shaded, "LEARNING THE LINES BY HEART"; 1.70 at the end |
| instrumental | | pull back to the whole world |
| chorus 3 | | the page, the ring, the weights, the guess in one frame, then in to the final excerpt |
| outro | Three minutes on one graphics card / The big ones ... / Next time, Next Token | the page; wide; down to the next-episode card (NEXT TOKEN); a last slow pull-back over the whole world; fade to paper |

>>> NEXT
Done 2026-09-28 7:23 am: out/final/learning_to_write.mp4 + _vertical.mp4 (course gate PASS on both), share copies,
the hook Short (learning_to_write_hook.mp4), thumbnails out/publish/thumbs/thumb_{A,B,C}.jpg, publish.json drafts.
Song capA/5103; runners-up 5203 and 5101 in ~/clips/learning-to-write/. Nothing uploaded or posted.
Open: 8 of 16 takes (the other 8 seeds ran out of memory, DEVLOG 6:16 am); Whisper word times on this slow take
lead the vocal onsets by ~0.3 s, the aligned times were kept; the wide shots print small text as texture.
