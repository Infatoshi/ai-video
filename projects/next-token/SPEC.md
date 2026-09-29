# Next Token: the music video

A code-rendered drum and bass music video about how an LLM writes, one token at a time, for people
who use ChatGPT and have never looked inside. Every number on screen comes from one real model,
Llama 3.1 8B Instruct, recorded answering one question on the GPU host's 3090 (`llm/dump.py` ->
`data/llm.json`; the scenes read it through `ctx.llm` and `ctx.story`). Engine forked from
`../cuda-roofline` (itself from github.com/mexicat/pdoom-video, MIT); `docs/ENGINE.md` is binding.

## The brief (Elliot, 2026-09-27)

- Topic, in order: text becomes tokens (the "strawberry" R-count joke); tokens become vectors;
  attention; the MLP where facts are stored, stacked N layers; logits, softmax, temperature (the model
  rolls weighted dice); append and repeat, the KV cache; why it's slow (decode is memory-bound, a
  roofline cameo), batching, speculative decoding; outro: the answer assembles itself token by token.
- Song: drum and bass, 174 BPM, male UK MC, rapped verses and a big sung hook, "just one more token".
- Look: risograph print. Off-white paper, fluorescent pink, riso blue, black used sparingly. Real 3D
  printed as halftone, slightly misregistered, paper grain, flat shading, no bloom. Hero object: the
  model as a building (a tower of layer slabs, the residual stream a light shaft up the middle, tokens
  as printed tiles, the KV cache a shelf that keeps growing). One recurring character: the token about
  to be generated.
- Edit: fast (a new shot at least every bar, every beat on the drop, hard cuts on the grid); dense
  (at least 3 moving layers plus a HUD in every frame, never an empty frame); readable (the word being
  sung is the biggest thing on screen); no camera shake or zoom pulse; low variance (shared kit first,
  a bar-by-bar shot list, an activity gate, a per-beat contact sheet, rework the worst first).
- Deliverables: `out/final/next_token.mp4` (1080p60) and a native 9:16 re-render (not a crop), the
  song wav in `~/clips`, the activity plot, the per-beat contact sheet, and the weakest seconds left.
  Nothing pushed or posted.

## The model run (data/llm.json)

The question is "How many r's are in strawberry?" in the Llama 3.1 chat template. The answer is the
model's own, sampled with its default settings (temperature 0.6, top-p 0.9, fixed seed); every step
keeps the top 10, the nucleus it sampled from, the uniform draw and the pick, the logit lens up the
32 layers, every head's attention row, and feed-forward activations at layers 0, 15 and 31. Plus the
input embeddings of a word list (PCA to 3D), a speculative round (Llama 3.2 1B drafting 4 tokens) at
every answer position, and measured decode speed at batch 1 and 100. Whatever it answers is what the
video shows. Config numbers (all real): 32 layers, 32 query heads, 8 KV heads, hidden 4,096,
feed-forward 14,336, vocabulary 128,256, 8.03 B parameters, 16.06 GB in bf16, 128 KiB of KV cache per
token.

**What it did** (the story of the video): the chat-templated prompt is 43 tokens (Llama's template adds a
system header with "Cutting Knowledge Date: December 2023"; the question itself is 8 tokens: `How · many ·r
's ·are ·in ·strawberry ?`). **" strawberry" is ONE token in the question** (id in llm.d.prompt): the
model never sees its letters. The answer is 15 tokens: `There ·are · 2 ·r 's ·in ·the ·word ·' str aw berry
'. <END>`: "There are 2 r's in the word 'strawberry'." Wrong, and honestly so. Step 3 is the crux: the
nucleus was **"2" 69.7% vs "3" 30.3%**, the uniform draw was u = 0.657, and it landed on "2" (the right
answer had a 30% chance). In its own answer it spells the word as three tokens, str | aw | berry. Most other
steps are a sure thing (one token left after top-p); steps 4 (" r" 78% / " R" 12% / " '" 10%), 7 and 9 have
real choices. Speed on the 3090: 42.8 tok/s at batch 1, 1,805.5 tok/s at batch 100 (42x). Speculative
decoding: the 1B draft's 4 greedy guesses were all accepted by the 8B at every position; the round shown is
at answer position 10 (`str aw berry '.`, all 4 kept, the 8B's bonus 5th pick is `<END>`).

`ctx.story` (app/src/story.ts) is the clock over that run: which answer token is in flight at song
time t, how far up the tower it has climbed, when each is emitted (anchored to lyric words: chorus 1's
"pick" = "There", chorus 2's "pick" = the "2", "Roll the dice" = " r", the breakdown = " in" " the", the
drop = " word" " '", verse 4's "go" = the speculative round's 4 tokens, "says it's done" = `<END>`), and the mode (prefill / decode /
batch in the drop / spec in verse 4). The HUD and every scene read it, so the whole video agrees on
where the model is.

## Design system

- **Inks** (`palette.ts`, GLSL `C_PAPER C_PINK C_BLUE C_INK`, Canvas `rgba('pink')`): paper, pink,
  blue, black. Pink = the hero (the token in flight, what is being sung, a hit). Blue = structure (the
  tower, grids, the model). Black = type, outlines, numbers. Two inks overprint (pink over blue prints
  purple): draw both, with `multiply` in Canvas2D. Any other colour prints as muddy black.
- **Print** (`post.ts`): every frame is separated into the three inks and halftoned (8 px cells, pink
  15°, blue 75°, black 45°), each plate out of register by ~1.6 px, on paper with fibre and ink
  mottle. Return `reg: shot.seed` from `render()` so every shot is a new print (registration and paper
  change on the cut). Tints become dot patterns, so **small type must be solid ink** (alpha 1).
  `solid: 1` skips the halftone for a clean moment; `flood` floods pink; `flash` whites out to paper.
- **3D** (`scenes/_kit.ts`): `inkMat` (coverage follows a camera-fixed key light: a light tint on the
  lit side, solid ink in shadow, optional second ink overprinting the shadow and rim, `uniforms.hot`
  floods it with the second ink) and `outline()` (constant-width black edges). Every 3D object gets
  both. Backgrounds are paper or a `Wash` (two soft ink blobs that print as halftone gradients),
  never black.
- **Type**: Archivo (grotesk, widths 62-125, weights 300-900) for display, IBM Plex Mono for data.
  Slams are Archivo 125/900. Tokens always print with `showTok()` (a leading space shows as `·`).
- **Shared models**: `scenes/_tower.ts` (the tower: lobby = embedding, 32 layer floors each with a
  slab, a ring of 32 head pillars and a feed-forward block, a KV shelf per floor, the pink residual
  shaft, `ride()` a tile up it, `light(climb)`, `setHeads(l, weights)`, `setKV(n)`), `tokenTile()`
  (a slab printed with a real token and optionally its id).

## Rules (every scene, checked at review)

1. **Overlay is global** (engine HUD, `hud.ts`): crop and registration marks, the model tag, POS / KV
   / tok/s, the floor ladder, the next-token odds, the answer strip, the sung line (top centre, on a
   paper strip), and the **slam**: the concept word of the line (`src/words.ts`) printed huge the moment
   it is sung. A scene that stages the word itself returns `kinetic: 0` for that moment; otherwise
   leave it on. Scenes never draw their own lyric subtitles.
2. **Cut on the grid**: use `shot(ctx, f)`. It cuts every bar in verses, every 2 beats in choruses
   and verse 4, every beat on the drop and the outro band (`CUT_EVERY`; the final chorus is the quiet
   one in this take, so it cuts every 2 beats and gets its density from the replay). A shot is a different
   framing or a different idea, not the same framing nudged. Pick hard switches with `f.ft`, animate
   with `f.t`.
3. **Three layers minimum** in every shot: a background field (Wash, token rain, grid, tower
   silhouette), the hero, and a foreground field (specks, flying tiles, a second object crossing).
   Plus the HUD. No empty paper.
4. **Fast**: things arrive in under 100 ms (`snapIn`, `whip`), land on a beat and hold. No eases
   longer than half a bar. Speed ramps allowed (slow into a hit, snap out).
5. **Readable**: the sung concept is the brightest, biggest thing. Repeat an idea from 3-4 angles
   across its line rather than showing it slowly once.
6. **No camera shake, no zoom pulse.** Camera moves are continuous (`drive`) or whip on a beat.
7. **Real numbers only**: from `ctx.llm` (the run and config) or arithmetic on them. Label a
   computed number as computed. Never random digits.
8. **Both aspects**: lay out from `W`/`H` (`portrait()` in the kit); check stills at 16:9 and with
   `--portrait`. The vertical cut is re-framed per shot, never cropped.
9. Performance: < 30 ms/frame at 1 sample (`render.ts perf`).

## Scenes and shot plan

Windows come from `app/src/timeline.ts` (anchored to lyric lines, snapped to beats). Each scene
cuts on the grid; the ideas below are the shots to rotate through, in order.

| id | section / lyric | shots |
|---|---|---|
| `intro` | intro (instrumental) | NEXT TOKEN title slammed in riso type; the tower flashing floor by floor; the empty chat box with a blinking cursor; token rain; each bar a new angle on the tower; the model tag and "8,030,261,248 PARAMETERS" |
| `tokens` | V1 1-4: Type a question ... watch how it goes | the question typed into the chat box (the real prompt); ENTER; the sentence chopped into tiles with their real ids; "strawberry" split into its real pieces; the model's view: ids only, no letters; R's crossed out / "?" |
| `vectors` | V1 5-8: Every token is a vector ... climb them all | a tile unrolls into a bar code of its real embedding values (first 64 of 4,096); the 3D embedding cloud (real PCA: king/queen, cat/kitten, Paris/France side by side); the sentence as a stack of vectors; crane up the tower, 32 floors counted |
| `hook` ×3 | chorus: Just one more token ... all night | the in-flight token at the roof; the real nucleus as a stacked odds bar / wheel; the dice roll lands on `u`; PICK: the tile drops; FEED IT BACK: it falls down the shaft to the lobby; AGAIN: the loop, each chorus one step further |
| `attention` | V2 1-4: Every token looks back ... own way | the context as a row of tiles, arcs from the new token weighted by real attention; the weights as bars; the values blending into the token; 32 HEADS as a grid of 32 real attention rows (different heads look at different words) |
| `ffn` | V2 5-8: Then the feed-forward fires ... its power | 14,336 neurons as a field of bars (real activations, layers 0 / 15 / 31); the few that fire; one floor of the tower in close-up (heads ring + FF block); stack ×32, the tower rising floor by floor |
| `logits` | V3 1-4: At the top of the tower ... it's a bet | the scoreboard: 128,256 scores as a long ribbon; softmax squashes them into odds; the temperature knob: low (one winner), 0.6 (the real one), high (flat) from the real top-10; SAFE vs BET |
| `append` | V3 5-8: Roll the dice ... past comes cheap | roll, APPEND the tile to the end of the sentence; the whole thing goes back in; KV cache: the shelf of K/V cards per floor growing; only the new token climbs, the past is cards (cheap) |
| `memory` | breakdown: Every single token ... math just waits | 16 GB of weights streaming from memory as sheets of paper for every token; the math unit idle (a clock, a WAIT stamp); bytes counter; slower cuts but full frames |
| `roofline` | drop (instrumental) | the roofline chart printed: decode at batch 1 sits at ~1 FLOP/byte on the memory slope; the drop: batch slider 1 → 100, the dot climbs, 100 answer streams; measured tok/s at batch 1 vs 100; tokens fly (the story emits many here) |
| `spec` | V4: Stuck on the slope ... in one go | roofline again; one read, a hundred users; the small tower (Llama 3.2 1B, 16 floors) guesses 4 tiles; the big tower checks all 4 in one pass; real accept / reject marks |
| `finale` | final chorus | a replay of every roll the model made, one per cut, needles landing on the real u; step 3 ("2" 69.7% vs "3" 30.3%) gets the big beat; the `<END>` die spins and does not land |
| `outro` | outro: Till it picks the one that says it's done | `<END>` lands on "done"; then the band: the whole video reprinted at speed from clean plates (`render.ts plates`), poster beats between, ending on the poster (question, the wrong answer, "The dice said 2 (69.7%). The answer was 3 (30.3%)."); blank paper from the hard stop |

## Gates

- `bun scripts/gate.ts`: per-second visual activity (frame difference, detail, cuts) of a render,
  plotted under the song's energy; flags seconds under the floor; v-to-v comparison.
- Per-beat contact sheet (one frame per beat), not per scene.
- Rework the bottom sections first; repeat until the spread is tight.

>>> NEXT
v1 done 2026-09-27 7:37 am: out/final/next_token.mp4 (1080p60, sha256 100ce84d...), next_token_vertical.mp4
(1080x1920, sha256 baf1e10a...), share copies under 512 MB, song in ~/clips (take 803; 801 and 907 beside it).
Gate p10/median 0.69 (CUDA v5: 0.48), 440 cuts. Open weak points: the intro's second shot and the finale's
floor-guess shot are the quietest seconds; the HUD slam position is hashed, so it sometimes covers a scene's
chart (finale roll 8, attention's grid); logits' top-down coil reads as rings more than 128k slots; the
batch-100 wall is 2D-heavy. Nothing pushed or posted.
