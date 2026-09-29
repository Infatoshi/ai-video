# downhill: ML, slowly #1/5

## The brief
Elliot, 2026-09-28, on the five-episode plan (after beginners said the first two were too fast): "i trust your
vision on these. fire off a subagent for each". The approved episode, as proposed:

learning = measure the error, find downhill, take a small step, repeat. A small network trained on two interleaved spirals (every step recorded) + a real 2D slice of its loss landscape (labelled a slice). One ball on one landscape while the network's boundary untangles beside it. Jazz waltz, ~90 BPM, 3/4 (6/8 or a swung 4/4 if the generator can't hold 3/4). The chorus is the algorithm.

Series rules: ../../AGENTS.md "The course" (pace rules, course look, parallel-agent rules). Template and reference
for every stage: ../next-token (its SPEC.md and DEVLOG.md). Real numbers only: name the source of every number on
screen.

## Topic arc (one idea: how a model learns)
1. The task (verse 1): two interleaved spirals of dots, blue and black; draw a line that keeps them apart. The
   machine's line at the start is wrong (105 of 200 dots on the wrong side).
2. Weights (verse 2): the machine is a network: 337 knobs, each a weight. Turn one knob and the line bends (the
   real network, one weight swept). They start random.
3. Loss (verse 3): measure how wrong, dot by dot (each dot's real cross-entropy as a pink bar), add it up: one number,
   the loss (0.696 at the start). Every setting of the knobs is a place on a map; the loss is the height.
4. The chorus, the one definition, the same words x3: measure how wrong it is / find which way is down / take one
   small step that way / and do it again, and again.
5. Gradient and learning rate (verse 4): it can't see the valley, only the slope under its feet; which way is down =
   the gradient; how big a step = the learning rate (0.3).
6. Too big a step (bridge), a real run: learning rate 3.0 (ten times as big). Cut open along its own downhill
   direction at its step 9: the valley floor (0.622) sits at step length ~1.05; a 0.3 step lands at 0.660, the 3.0
   step jumps the floor and lands at 0.818, higher than where it stood (0.697). Over 5,000 steps it bounces and never
   gets down (loss never below 0.58; 61% right at step 5,000).
7. Last chorus: every piece at once. Outro: 3,121 steps and every dot is on its own side; next time (2/5): why it
   never sees a letter (tokens).

## The real data (model/train.py -> data/model.json; model/spiral.py)
- Data: 200 points, two spirals (100 per class, 1.5 turns, noise 0.04, seed 7), in [-1.25, 1.25]^2.
- Network: 2-16-16-1, tanh, Glorot-uniform init (seed 1), **337 weights**. Mean binary cross-entropy. Plain
  full-batch gradient descent (every step measures every dot), **learning rate 0.3**, steps 0..5,000, all recorded
  (loss, accuracy, weights). Gradient checked against finite differences.
- The run: loss 0.696 at step 0 (47.5% right, 105 wrong); one bump at steps 693-697 (0.55 -> 0.68, plain GD's edge
  of stability, shown as is); 1 wrong at step 3,000; **every dot right first at step 3,121** (loss 0.049); loss
  0.0049 at step 5,000.
- The too-big run: same start, **learning rate 3.0**: loss never below 0.582 (step 1,973), best accuracy 75.5%
  (step 1,415), 61% at step 5,000.
- The landscape: Li et al. 2018. Their filter-normalized random directions hold 1.6% of the training path (measured;
  the path would be a speck and the slice's loss at it 0.22 vs the real 0.70), so the video uses their section-7
  method for trajectories: the top two PCA directions of theta_t - theta_5000 (95.8% + 3.1% = 99.0% of the path),
  loss on a 161 x 99 grid around the final weights. On-screen label: "A 2D SLICE OF A 337-DIMENSIONAL LANDSCAPE (THE
  TWO DIRECTIONS THE WEIGHTS MOVED MOST)". The ball sits on the slice at the path's projection; the number shown is the
  real loss (the slice's value at the ball differs by at most 0.12).
- Height = 2.1 x 1.2 tanh(loss / 1.2) (monotone; compresses the far walls above ~1). Contours at loss 0.1, 0.2, ...
  and 0.05, 0.025, ... toward the floor.

## Song
"Downhill", jazz waltz, 3/4, requested 90 BPM, Bb major, 240 s; ACE-Step 1.5 XL SFT + 5 Hz LM 4B; capA soft female
jazz vocalist, capB male crooner (song/runs/). Lyrics: song/lyrics.txt (annotated), song/lyrics.sing.txt (sung).
**Pick: downhill-capB/1201** (male crooner): Whisper recall 0.981, WER 0.019, 90.001 BPM (analysis), clearly in 3
(bar phase from the bass), 74 sung words/min, busiest 30 s 114 (course ceilings 95 / 120). Shape: intro 0-16 s, verse 1
16-32, break, verse 2 40-57, break, verse 3 64-81, chorus 1 82-99, break, verse 4 104-121, chorus 2 122-139, bridge
144-162, piano solo 162-198, chorus 3 198-215, outro 216-228, fade to 240 s. Runners-up in ~/clips/downhill/ (1205
capB, 1107 capA). Master: audio/downhill.wav (-14 LUFS, -1 dBTP). Scored on the Mac (song/score_mac.py), aligned on
the Mac (analysis/run_mac.sh); see DEVLOG for why.

## Look
The course look (AGENTS.md): the Next Token riso print (paper, pink, blue, ink; halftone, misregistration fixed for
the whole song: one print, no reseeding). Pink = what is being taught while it is sung (the wish line, wrong-dot rings,
error bars, the loss number, the ball, the arrow, the trail, the too-big jump). Blue = the data (blue dots, the blue
side of the network's line). Ink = structure (the landscape block, contours, the network's line, knobs, type).
HUD: crop/registration marks, "ML, SLOWLY 1/5" top left, the sung line top centre (slides in 0.45 s before it is
sung, stays until the next, clears in instrumentals). No slams. One featured number at a time (top of the panel).

## Shot plan (one scene, app/src/scenes/world.ts; phases in app/src/story.ts, anchored to sung words)
| phase | lyric | picture |
|---|---|---|
| intro | instrumental | the 200 dots arrive along their arms, centre panel |
| task | verse 1 | "Draw me a line": the final network's line drawn in pink round the spiral (the line it will learn); "can't do it": the step-0 network's line and region tint, wrong dots ringed pink; number 105 |
| knobs | verse 2 | panel slides right; 337 dials draw in (each at its real weight); number 337; "Turn any knob": one W2 weight swept +-2.6, the line bends live; "random": rings back |
| loss | verse 3 | per-dot error bars rise along the arms, then sum: 0.696; "a place on a map": the dials fly into one point, the contour map draws outward from it (top-down); "how high": the map rises into the block, camera tilts to 3/4, HEIGHT = LOSS |
| chorus x3 | the loop | same picture each time: loss number and wrong-dot rings (measure), the pink downhill arrow (find), the ball steps (step), the clock runs (again). Chorus 1 = steps 0-3 then ~12; chorus 2 = 600-1,500 and adds the trail; chorus 3 = 2,000-3,000 and adds the loss curve |
| slope | verse 4 | camera in close on the ball, fog beyond a small radius (it can't see the valley), a ring under it (the slope under its feet), GRADIENT on the arrow, "ONE STEP = 0.3 x THE SLOPE" (a tick at 30% of the arrow); number 0.3 |
| big | bridge | landscape dims; the cut chart: the real valley profile, the 0.3 hop (0.660) and the 3.0 jump over the floor to 0.818; then both runs' loss over 5,000 steps (0.3 falls, 3.0 bounces); panel shows the too-big run's line; number 61% |
| outro | 3,121 steps | number 3,121 on "steps", then 200 / 200; the panel comes forward; NEXT: ML, SLOWLY 2/5 TOKENS; fade to paper |

>>> NEXT
v1 done 2026-09-28 4:18 am MT: out/final/downhill.mp4 (1080p60, sha256 a5216d78...), downhill_vertical.mp4 (1080x1920,
sha256 effddd37...), share copies 164 / 151 MB, hook Short out/final/downhill_hook.mp4 (34.7 s), thumbnails
out/publish/thumbs/thumb_{A,B,C}.jpg, publish.json drafts (packaged in out/publish/). Course gate PASS on both finals
(74 words/min, busiest 30 s 114, 0 hard changes, nothing frozen). Nothing uploaded or posted. Open points: the song has
no instrumental bars after verses 3 and 4 (v2 lyric with those breaks is ready: song/lyrics.sing.v2.txt); the
landscape is the PCA slice (Li et al. section 7), not the random-direction slice the brief named (measured 1.6% of
the path on the random plane); scoring and alignment ran on the Mac, not the GPU host.
