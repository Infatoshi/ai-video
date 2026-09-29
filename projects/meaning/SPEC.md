# meaning: ML, slowly #3/5

## The brief
Elliot, 2026-09-28, on the five-episode plan (after beginners said the first two were too fast): "i trust your
vision on these. fire off a subagent for each". The approved episode, as proposed:

meaning is a direction. Llama 3.1 8B's embedding table (128,256 x 4,096), real nearest neighbours, the real king - man + woman result shown whatever it is. One continuous flight through the point cloud. Dream pop, ~85 BPM.

Series rules: ../../AGENTS.md "The course" (pace rules, course look, parallel-agent rules). Template and reference
for every stage: ../next-token (its SPEC.md and DEVLOG.md). Real numbers only: name the source of every number on
screen.

## The data (data/emb.json, from llm/extract.py + llm/embed.py)
Source: `model.embed_tokens.weight` of meta-llama/Llama-3.1-8B-Instruct (snapshot 0e9e39f2..., shard
model-00001-of-00004), its bf16 bytes copied as they are (sha256 e1598c85...). 128,256 rows x 4,096 numbers.
Similarity is cosine over all 4,096 numbers and all 128,256 rows; tokens keep their leading space (shown as `·`).

- " rain" is row 11,422; its first number is -0.0016 (range -0.037 to 0.036).
- Nearest to ·rain: ·Rain 0.520, Rain 0.404, ·rains 0.346, ·snow 0.277, ·raining 0.264, ·rainfall 0.250, rain 0.241,
  ·wind 0.209, ·sun 0.196, ·weather 0.194, ·storm 0.184. Its own spellings first, then weather.
- Nearest to ·three: ·four 0.711, ·two 0.676, ·five 0.626.
- Two random common words: mean cosine 0.038 (20,000 pairs of 12,699 lowercase whole-word tokens).
- king - man + woman (unit vectors, the three inputs left out): ·King 0.423, ·KING 0.332, King 0.306, **·queen 0.297
  (4th)**, ·kings 0.285. boy - man + woman: ·girl #1 (0.461). father - man + woman: ·mother #1 (0.443). The same
  arrow lands on the partner at #1 for 13 of 23 male/female pairs.
- The 3D view: PCA fitted to the 46 labelled tokens, 3 axes keep 20.7% of their spread (8.7 / 6.5 / 5.5%); 8,000
  common words squashed the same way form the haze in the middle. Landing points in 3D are the real sums, squashed
  (PCA is linear, so the arrow drawn in 3D is exactly the squashed woman - man).

## Topic arc
1. A word is a list of numbers (verse 1): before a chatbot reads a word it looks it up in a table; ·rain's row is
   4,096 numbers; that list is its embedding.
2. Close means alike (verse 2): close = the lists point the same way (cosine: 1 same, 0 unrelated, random pairs
   0.038); three's nearest are four, two, five; rain's nearest are its own spellings, then snow, wind, storm.
3. A direction can mean something (verse 3): the man -> woman arrow carried to boy lands on girl, to father on
   mother; carried to king it lands on King, KING, King, and queen is fourth. The bridge says so plainly: "the
   spelling pulls harder than the crown".
Chorus (3x, the one definition): "Every word is a list of numbers / Every list is a place / Words that mean alike
live close / Meaning is a direction". Outro names #4 (Who's It?, attention).

## Song
Dream pop, 85 BPM, D major, ~240 s. Lyrics song/lyrics.txt, sung spelling song/lyrics.sing.txt (v2, 230 words).
Run 1 (v1 lyrics, 309 words): runs/a1.json (female lead) and b1.json (male lead), 8 seeds each; recall up to 0.981
(a1/3305) but every take too dense for the pace ceiling (a1/3305 aligned: 89 words/min overall, busiest 30 s 138 vs
120). Run 2: runs/a2.json (v2 lyrics, female lead, "slow sparse vocal phrasing"), seeds 3501-3508.
Pick: **a2/3507** (recall 0.996, WER 0.004, 85.0 BPM, 240 s; vocal 22.9-232.3 s, fade to 236 s). Aligned pace
66 words/min, busiest 30 s 88. Runners-up in ~/clips/meaning/: a2/3505 (best shape, but the outro is cut off at
240 s) and a2/3506.

## Look
The course riso look (paper, pink, blue, ink; post.ts halftone). Pink = the word or relation being taught while it
is sung (the rain row, the current neighbour thread, the carried arrow); blue = the data (rows, the points, the
haze); ink = type, axes, outlines, settled threads and arrows. All type (labels, numbers, panels, the sung line)
prints with the HUD: crisp solid ink, never halftoned. HUD: crop marks, "ML, SLOWLY 3/5" tag top left, the source
tag bottom left, the sung line top centre (fades between lines, sung words go from a tint to solid ink).

## Shot plan: one continuous flight (scenes/flight.ts; no cuts)
The camera is a monotone cubic through keyed poses (flight-data.ts `Spline`: no overshoot, no stops) plus a slow
sway, keyed to lyric words (times from data/lyrics.json).

| section (aligned time) | camera | on screen |
|---|---|---|
| intro 0-22.6 s | the whole space from far, drifting in | title MEANING IS A DIRECTION |
| V1 "Before a chatbot ... in a table ... every word it knows" | the table wall arrives (48 real rows around ·rain, their tokens down the left) | THE EMBEDDING TABLE; 128,256 rows |
| V1 "Find rain ... Four thousand ninety-six ... embedding" | along ·rain's row (pink bars = its 4,096 real numbers) | 11,422 (its id), 4,096, EMBEDDING, -0.0016 |
| V1 "It doesn't know ... Only where it is" | the wall fades, the row folds into one point, the camera chases it into the space | |
| chorus x3 | the same move: from rain out to the whole space, turning | rain's first 96 numbers on line 1; 20.7% on "place"; labels draw in on "close"; each return keeps what the verses added and turns it pink on its line (threads on "close", arrows on "direction"); the last chorus leans in on the arrows |
| V2 "Close means ... One means the same, zero ..." | the weather side; inset dial: cosine 1 same way, 0 at right angles | 0.038 random pairs |
| V2 "Next to three ... Nobody put them there" | the numbers: threads three->four, two, five | 0.711, 0.676, 0.626 one at a time |
| V2 "Next to rain ... Then words that mean the same" | out through the middle to rain: threads to ·Rain, ·rains, ·snow, ·wind, ·storm; ranked list | 0.520 ... 0.184 |
| V3 "Draw an arrow ... A direction carries meaning" | the people, from the side: man->woman arrow, carried to boy (lands by girl), father (by mother) | 0.461, 0.443 |
| V3 "Now the famous one ... in fourth place" | the arrow travels to king; lands among King, KING, King; queen fourth | 0.423, 0.332, 0.306, 0.297; top-5 panel |
| bridge | holds on king's spellings and queen, then flies to rain | 0.297 held |
| outro | out past everything | NEXT 4/5 WHO'S IT? |

>>> NEXT
Done 2026-09-28 7:30 am: out/final/meaning.mp4 + meaning_vertical.mp4 (course gate PASS on both: 66 words/min,
busiest 30 s 88, 0 hard changes, one continuous shot), share copies under 512 MB, hook Short meaning_hook.mp4
(29.65 s), thumbnails out/publish/thumbs/thumb_A/B/C.jpg, publish.json drafts. Nothing uploaded or posted; Elliot
picks the thumbnail and approves any text before it goes out. Weak points: the instrumental after verse 3 is only
1.7 bars (the generator's arrangement); the 3D squash keeps 20.7% of the labelled words' spread, so the king landing
point looks far from King in 3D even though King is its true nearest (the dashed threads carry the real ranking);
wide shots are sparse (lots of paper around the cloud).
