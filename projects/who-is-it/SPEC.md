# who-is-it: ML, slowly #4/5

## The brief
Elliot, 2026-09-28, on the five-episode plan (after beginners said the first two were too fast): "i trust your
vision on these. fire off a subagent for each". The approved episode, as proposed:

attention. Llama 3.1 8B on "The animal didn't cross the street because it was too tired" vs "...too wide": find a real head where "it" moves from animal to street; if none does it cleanly, show what the heads really do. One sentence with arcs sized by the real weights, morphing when the word changes. Neo-soul duet, ~75 BPM (one voice the word asking, the other the words answering).

Series rules: ../../AGENTS.md "The course" (pace rules, course look, parallel-agent rules). Template and reference
for every stage: ../next-token (its SPEC.md and DEVLOG.md). Real numbers only: name the source of every number on
screen.

## The real data (data/attn.json)

Llama 3.1 8B Instruct (bf16, eager attention, RTX 3090), each sentence run once as plain text after the
`<|begin_of_text|>` token it always sees first (`llm/attn.py` -> out/llm/attn.npz; `llm/flip.py` -> data/attn.json).
Tokens: `start · The · animal · didn · 't · cross · the · street · because · it · was · too · tired|wide · .`
(14; "didn't" is two tokens). Heads are 0-indexed (layer 0-31, head 0-31).

- **At "it" nothing can flip.** Each word only looks back, and "it" comes before "tired"/"wide", so all 1,024 heads
  give "it" exactly the same weights in both sentences (max difference 0.0). The featured head at "it": start 60.5%,
  animal 8.3%, street 7.1% (both times).
- **The switch happens at the changed word.** At "tired"/"wide": 118 of 1,024 heads lean the right way in both
  sentences (more weight on animal after "tired", on street after "wide"), 4 clearly (>= 5 points each way: L0 H11,
  L3 H31, L5 H17, L8 H29), 23 lean the wrong way, 883 don't switch; 524 of all 1,024 heads put under 2% on both words in both sentences.
- **The featured head, layer 8 head 29** (best flip margin): "tired" -> start 22.5%, it 19.5%, animal 11.1%, street
  4.0%; "wide" -> street 68.0%, start 18.6%, animal 0.3%. The same head is the top switcher in the "before" pair
  and second in trophy/suitcase.
- **The model gets it right anyway** (behaviour, not one head): "... The one that was too tired was the" -> animal
  33.4% (street 0.4%); "... too wide ... was the" -> street 43.1% (animal 6.6%). Asked in chat: "Animal." / "Street.".
- Most weight rests on the start token: ~75% on average over all heads at these positions.
- All pairs tried are in data/attn.json `pairs` and DEVLOG (trophy/suitcase, "before", plural, pronoun).

What the video says, in order: you know who "it" is; the model reads left to right, so at "it" it can't know
yet (same weights both times); the next word looks back and does the switching in some heads (L8 H29: 11.1% ->
animal, 68.0% -> street); most heads don't switch, one head is one small part; the model's answer is right.

## Topic arc
1. Concrete first (verse 1): the sentence, who's "it"? the animal. Change one word: the street. A program reads one
   word at a time: how does it know?
2. The rule (chorus = the one definition, same words each time): every word looks back at the words before, asks
   each how much it matters, gives each a weight, the weights add up to 1. That's attention.
3. The names (verse 2): the asking word is the query, the answering words are the keys; the closer the match, the
   bigger the weight; each weight is a line; spare weight rests on the start.
4. The twist (bridge): it reads left to right, so at "it" the deciding word hasn't come: the same weights both
   times. "tired" arrives and looks back; "wide" looks back and finds the street (68.0%).
5. Honest scale (verse 3): each way of looking is a head; 32 heads x 32 layers; most don't switch; one head is one
   small part; ask the whole model and it says the animal.
6. Final chorus reassembles it all on "tired"/"wide"; outro names #5.

## Song
"Who's It?", neo-soul duet, 75 BPM, Eb major, ~250 s requested: male tenor = the word asking, female alto = the
words answering (song/lyrics.sing.txt, 260 sung words). Runs song/runs/w1_capA.json (Rhodes, brushed drums, call
and response) and w1_capB.json (swung lo-fi neo-soul R&B), seeds 401-408 / 501-508. **Pick: w1_capA/403** (recall
0.982; the one take that is a real duet: verse 1 trades female/male line by line, verse 2 and the bridge are the male
voice, the choruses female), mastered to -14 LUFS (audio/whoisit.wav). Runners-up in ~/clips/who-is-it/ (capB 503,
capB 507). 75.003 BPM, 3.200 s bars, 250 s: intro 4 bars; verse 1 14.6-34.4 s; 5 bars instrumental; chorus 1 51.5;
verse 2 71.7-89.9; 4 bars; chorus 2 102.7; bridge 122.2-142.1; 6 bars; verse 3 161.3-179.4; chorus 3 185.9; outro
204.7-248 (sung three times), fade. Pace: 76 sung words/min, busiest 30 s = 120 (the ceiling).

## Look
The course look (riso: paper, pink, blue, ink). Pink = attention (the asking word, its arcs, the weight bar);
blue = the words it acts on (KEY tags, the heads that lean wrong); ink = the sentence, numbers, structure. The
marker "ML, SLOWLY  4/5" top left, the sung line top centre (slides in, stays up), the source of the data bottom
left (crisp, via the overlay's caption). One continuous picture (scenes/sentence.ts), no cuts: the camera moves
(verse 1 close on the words, choruses on the whole arc fan, the bridge leans in on "it .. tired", verse 3 pulls back
from one head to its row of 32 to all 1,024 and flies back in), words are drawn in over >= 0.35 s, arcs have paper
dots flowing from each word into the asker, and the camera always drifts a little. 9:16 is a native re-layout: the
sentence becomes a column, arcs bulge to the left.

## Shot plan (one scene, windows = timeline.ts entries, used for chapters)
| entry | lines | the picture |
|---|---|---|
| intro | instrumental | title "Who's It?" drawn in, slow push |
| who-is-it | verse 1 | the words appear as sung; "it" turns pink; a dashed "YOU" arc to animal; "tired" rolls into "wide", the arc swings to street; a pink cursor reads token by token, unread words ghosted. Instrumental: the word flips tired/wide and your arc follows |
| attention | chorus 1 | START token drawn in; arcs from "it" to every earlier word, all alike, then grown to the real weights; they stack into one bar "= 1" |
| query-and-keys | verse 2 | QUERY tag on "it", KEY tags on every earlier word; arcs regrow ("8.3%" animal); "60.5%" on the start arc |
| attention-2 | chorus 2 | the same picture, tags up |
| left-to-right | bridge | the cursor reads to "it", later words ghosted; animal and street arcs lit, near equal; the ghost word flips tired/wide and the arcs don't move ("all 1,024 heads: the same weights"); the cursor reaches "tired", the arcs slide over to it ("11.1%" animal); "tired" rolls into "wide" and the street arc swells ("68.0%"). Instrumental: tired/wide back and forth |
| all-the-heads | verse 3 | a card "ONE HEAD: LAYER 8, HEAD 29"; pull back to its row of 32, then all 32 x 32, every cell its head's real arcs morphing with the word; "883 of 1,024 don't switch" (they dim); "4 switch clearly" (pink frames) + legend; fly back in; "The one that was too tired was the animal" ("33.4%") |
| attention-3 | chorus 3 | everything at once on "tired": tags, arcs, bar; on "That's attention" it rolls to "wide" ("68.0%") |
| outro | outro | back to "tired", arcs fade, "NEXT · 5/5 Watch It Learn to Write", fade to paper |

One number on screen at a time (Sentence.calls()), each held until its line moves on.

>>> NEXT
v1 done 2026-09-28 4:35 am: out/final/who_is_it.mp4 (1920x1080p60, 250 s) and who_is_it_vertical.mp4 (1080x1920),
share copies (crf 24/25), the hook Short who_is_it_hook.mp4 (32.8 s), thumbnails out/publish/thumbs/thumb_{A,B,C}.jpg,
publish.json drafts (packaged locally by yt.py package). Course gate PASS on both finals: 76 words/min, busiest 30 s
120 (the ceiling), 0 hard changes/min, one 250 s shot, nothing frozen. Nothing uploaded or posted. Open: the
busiest 30 s sits exactly at the ceiling; verse 3 runs into the last chorus after 2 bars (the other new ideas get
4-6); the grid's 1,024 cells read as texture more than as individual heads at 16:9.
