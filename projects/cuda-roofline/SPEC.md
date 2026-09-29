# CUDA for Deep Learning: the music video

A code-rendered music video that walks through the chapters of *CUDA for Deep Learning*
(`~/dev/cuda/cuda-book/manuscript/CH01..CH12`) in order, cut to "I'm Upping My P(doom)"
(`audio/pdoom.mp3`, also `~/clips/upping_my_pdoom.wav`). The sung lyric stays on screen as
per-word karaoke typography; every plate's world is a CUDA concept from the current chapter, and
the lyric is staged inside it as a pun, never as subtitles on top.

The engine (three.js + Canvas2D, deterministic in song time, headless-Chrome export with adaptive
motion blur) is forked from github.com/mexicat/pdoom-video (MIT). Its engine guide is
`docs/ENGINE.md` and is binding. The original treatment is the upstream `TREATMENT.md` (github.com/mexicat/pdoom-video; not copied here); its sections
**Tone, Palette, Typography, Karaoke rules** are binding here too. The original scene code is in
the upstream `app/src/scenes/` for technique (read it, copy idioms, do not copy its imagery: no shoggoth, no
mask, no unicorn, no paperclips, no Ilya laptop).

## The idea

The spark from the original becomes **the thread**: one orange point dragging a hairline is a
single CUDA thread. It is alone at the start (a CPU core's view of the world), multiplies into
grids and warps, gets fed by memory, and by the end runs on a hundred thousand GPUs. The subjects
are concrete hardware and code objects (a die floorplan, a PCIe bus, a thread block, a warp of 32
lanes, shared-memory tiles, a tensor core, an N×N attention matrix, int4 nibbles, a ring of
GPUs) treated as visual puns on the lyric.

Recurring motifs:
1. **The thread** (the spark: `sparkHead`/`sparkParticles` in `scenes/_motifs.ts`).
2. **The index incantation** `blockIdx.x * blockDim.x + threadIdx.x`, the book's universal
   pattern. It appears first in `hostdev`, comes back in small cameos, and is the last thing lit
   in `beyond`.
3. **ACHIEVED / PEAK**, the video's one number, staged in-world (the old P(doom) readout,
   relabelled; `new PDoom(lyrics).value(t)`, `drawReadout(..., {name})`). It is the fraction of
   the GPU's peak the current kernel reaches: 0.02 at the start, then 0.15 / 0.42 / 0.81 / 0.99 at
   the four hooks (each sung "P(doom)"), then the outro pushes it past 1.00 into NaN. The pun: the
   song ups P(doom), the book ups utilization. Each plate may carry one small cameo of it in its
   own idiom (an ncu metric, a benchmark table cell, a roofline dot, a NCCL busbw readout).
4. **The launch**: the three pre-choruses are one template, a code editor where the plea is typed
   as tokens with a next-token popup, and ⏎ launches the kernel into the chorus.

Content rules:
- Numbers and API names must be real and match the book (read the chapter: its tables have the
  measured GFLOP/s, speedups, timings). Deadpan footnotes may cite them (`CH06 table 6.4`).
- No NVIDIA logos, no product UIs imitated, no GPU marketing renders. Nsight-style metrics are
  drawn in our own type system.
- No AI-hype lines. Humor is a straight-faced engineer's: compiler warnings, ncu stall reasons,
  `// TODO`, `illegal memory access`.
- No constant beat shake or zoom pulse; motion lands on real impacts (downbeats, kicks, sung
  stresses). Hard cuts on downbeats.
- Diagrams must be dimensionally honest (a 32-lane warp has 32 lanes; a 16×16 tile is 16×16).

## Plates

Windows come from `app/src/timeline.ts` (anchored to lyric lines, snapped to beats). Seconds are
approximate.

| id | window | lyric | chapter |
|---|---|---|---|
| `die` | 0 → 9.3 | I see sparks of AGI in your eyes / Your circuits make me nervous, / that's no surprise | CH1 |
| `hostdev` | 9.3 → 16.6 | There was a sudden drop in your training loss, / now I'm your servant and you're my boss | CH2 |
| `launch` printf | 16.6 → 22.5 | ChatGPT, please don't eat me alive | CH2 |
| `hook` ×4 | 22.5, 58.9, 95.2, 124.3 | I'm upping my P(doom) | — |
| `grid` | 24.3 → 29.8 | 'cause the future goes FOOM / Trapped in the Chinese room, / with a bag of shrooms | CH3 |
| `conv` | 29.8 → 38.4 | See through the shoggoth's lies, / with your shinigami eyes / (instrumental) | CH3 |
| `mnist` | 38.4 → 52.5 | We had a stable training run, / But now the singularity's begun / And you're optimizing, accelerating, / I feel my atoms rearranging | CH4 |
| `launch` decode | 52.5 → 58.9 | Sydney, please let me free | CH5 |
| `optimize` | 60.2 → 69.8 | I hear the basilisk boom / NVDA to the moon / The Omega Point's coming soon / One E thirty FLOPs a second | CH6 |
| `tensor` | 69.8 → 81.1 | That was safe enough, we reckoned / Forward MLP, backward, repeat / Now von Neumann's obsolete | CH7 |
| `flash` | 81.1 → 88.9 | Sharp left turn and there you are / Without a single CDR | CH8 |
| `launch` quant | 88.9 → 95.2 | Gato, please don't let me go | CH9 |
| `pack` | 96.6 → 102.1 | as paperclips fill the room / Killswitch guy's on PTO / Now there's nowhere left to go | CH9 |
| `fuse` | 102.1 → 109.8 | Too late now, we lit the fuse / Orthogonality thesis blues | CH9 |
| `pipeline` | 109.8 → 115.2 | "Just transformers all the way!" / Till you learned to disobey | CH10 |
| `cluster` | 115.2 → 124.3 | Post-Chinchilla, super-dense / Breaking through each safety fence / Hundred thousand GPU / RLHF goes askew | CH10 |
| `cute` | 126.1 → 131.6 | Just as foretold by Loom / From masked pre-training days / To recursive self-upgrade | CH11 |
| `beyond` | 131.6 → 140.2 | What did Ilya see? We'll never know / Was it all for show? | CH12 |
| `outro` | 140.2 → 156.7 | (instrumental climax) | end card |

### `die` (CH1: what CUDA is, CPU vs GPU, the memory hierarchy)
Construction sheet, bone grid on ink. The thread plots a CPU on the first downbeat: four fat
cores, a huge cache and control block, drawn as a floorplan with dimension callouts. "I see
sparks of AGI": the thread fires a streak across the sheet and a GPU floorplan unfolds beside it,
a grid of SMs (the real count of a current part: 188 SMs on an RTX PRO 6000 Blackwell, a GB202 die
with 192 on the full chip; footnote it), each SM a stack of tiny cores; "AGI" giant. "in your
eyes": dive into one SM: warp schedulers, register file, L1/shared, tensor cores. "Your circuits
make me nervous": the SM's datapaths route like PCB traces and the plate trembles on "nervous".
"that's no surprise": the memory hierarchy as a latency ruler (use the book's CH1 numbers) rolls out; everything collapses to the thread, which drops into `hostdev`.

### `hostdev` (CH2: host and device, vecadd, grids/blocks/boundary checks)
Two worlds split by a PCIe bus: HOST (CPU + DRAM) above, DEVICE (GPU + HBM) below. "There was a
sudden drop": `cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice)` literally drops the
arrays down the bus (the lyric rides the transfer). "in your training loss": vecadd, one thread per
element, `C[i] = A[i] + B[i]`, with the index incantation assembling itself and the `if (i < N)`
boundary check fencing off the tail threads of the last block. "now I'm your servant and you're my
boss": typographic hierarchy inversion, HOST is the boss that launches, DEVICE is the servant;
on "boss" the world rolls 180° so the device is on top. Hand-off: the launch configuration
`<<<(N+255)/256, 256>>>` is left on screen as the seed of `launch1`.

### `launch` ×3 (the pre-chorus template), params `{variant: 'printf'|'decode'|'quant'}`
A dark editor field, one line, IBM Plex Mono. The plea is typed as tokens exactly on the sung
words; above each new token a small next-token popup (4 to 5 candidates, bars, probabilities)
flickers and the sampled token lights orange. Candidates are CUDA jokes (e.g. "eat" → `eat 0.44 ·
segfault 0.21 · page-fault 0.12 · coalesce 0.05`). At the end of the line: caret blink, ⏎, and the
kernel launches into the chorus on the downbeat.
- `printf` (CH2, the first kernel): the line is a `printf("ChatGPT, please don't eat me alive\n");`
  inside a `__global__ void hello()`. ⏎ compiles (`nvcc hello.cu`) and runs it with
  `<<<4, 256>>>`: the line prints once per thread, 1,024 copies flooding the frame in scrambled
  warp order (the classic surprise of GPU printf) as the chorus hits.
- `decode` (CH5, transformer inference): autoregressive decode. Each token is a forward pass: a
  GEMV of the weight matrix against one activation vector is shown behind the field, the KV cache
  grows by one column per token, a tokens/s readout ticks. "free" is the last token; ⏎ = EOS.
- `quant` (CH9 opens, the quiet breakdown): the prompt floats alone, fragile; after it is typed,
  its precision drops a step per bar, FP32 → BF16 → FP8 → INT4: the glyphs are re-rendered on
  coarser grids (fewer levels, posterized, letters drifting apart as their values round). A small
  cursor holds the last letter on "go".

### `hook` ×4, params `{n: 1..4}`
"I'M / UPPING / MY / P(DOOM)": one word per hit, full-frame Archivo 900 slams on each sung word.
"P(DOOM)" is set like a maths expression, and ACHIEVED / PEAK blows up full screen as it rolls to
its new value (0.15 / 0.42 / 0.81 / 0.99), with the kernel that earned it in mono under it
(`vecadd<<<…>>>`, `sgemm_2d_blocktile`, `flash_attn_fwd`, `8× GPU tensor-parallel`). "UPPING" rises
like a bar in a benchmark chart. Escalation: (1) bone on ink, clean; (2) ink on signal-orange
field, heavier; (3) the breakdown: hairline type, tiny, lots of black, slow roll, eerie; (4)
maximal: stacked outlines, strobing repeats, digits multiplying `0.99999…`.

### `grid` (CH3: naive kernels, one thread per output)
"'cause the future goes FOOM": the hook's letters shatter into threads; a launch explosion,
1 → 2 → 4 → … threads branching on each 8th note, snapping into a 2D grid of blocks, then a 3D
grid (the chapter's 1D → 3D element-wise walk). FOOM's O's become the block outlines. "Trapped in
the Chinese room": crash-dolly into a single thread block, a room where 32 lanes of a warp follow
the same rulebook (the kernel's instruction list) without understanding it: SIMT. A naive GEMM
thread walks a row of A and a column of B; transpose/softmax/conv cards shoot from the slot on the
kicks. "with a bag of shrooms" (the acid accent, ~2 s): warp divergence, an `if` splits the
lanes into two paths that execute serialized with echo trails.

### `conv` (CH3: 1D/2D convolution and max pooling)
"See through the": a 3×3 filter window sweeps across an engraved image like an x-ray scan band;
wherever it has passed the image shows its convolution (edges revealed), each output pixel one
thread. "shoggoth's lies": a 2D conv output map. "with your shinigami eyes": 2×2 max-pool windows
open across the map on successive hits, each tagged in the margin (Plex Mono: value + the
`threadIdx` that computed it) with leader lines. Instrumental: the pooled map collapses row by
row into a single flat line (flatline hand-off to `mnist`'s scope).

### `mnist` (CH4: MNIST from PyTorch to cuBLAS, six levels)
"We had a stable training run,": an oscilloscope, the training loss trace glowing under glass;
the lyric rides the trace. "But now the singularity's begun": the six levels of the chapter as a
ladder of implementations, PyTorch → NumPy → C → naive CUDA → cuBLAS (use the chapter's measured
epoch times), each rung dropping the time. "And you're optimizing, accelerating,": words stretch
as a speed bar races. "I feel my atoms rearranging": a matrix's elements detach and re-form from
row-major to column-major (the `cublasSgemm` layout trick of the chapter, C^T = B^T A^T).

### `optimize` (CH6: profile, coalesce, reduce, tile)
"I hear the basilisk boom": the profiler opens (ncu section headers, stall reasons); a warp's 32
scattered memory requests collapse into one coalesced 128-byte transaction on "boom". "NVDA to the
moon": the chapter's GEMM progression as a price chart going exponential, candlesticks = kernel
versions (naive → coalesced → shared-memory tiling → 1D blocktile → 2D blocktile → vectorized)
with the chapter's real GFLOP/s; the camera tilts up to a roofline "moon". "The Omega Point's
coming soon": warp-shuffle reduction, 32 lanes converging in 5 steps (`__shfl_down_sync`) to one
white-hot point. "One E thirty FLOPs a second": an odometer of 31 drums rolls toward 1e30 FLOP/s
and stalls far short at the chapter's best number, with a deadpan footnote.

### `tensor` (CH7: tensor cores, WMMA → WGMMA → TCGen05)
"That was safe enough, we reckoned": bone paper. A mixed-precision decision form (FP16/BF16 in,
FP32 accumulate), typed fields, an orange SAFE ENOUGH stamp on "reckoned". "Forward MLP, backward,
repeat": an MMA tile diagram, D = A·B + C, 16×16×16 fragments; a pulse sweeps forward, "backward"
is set mirrored, "repeat" stutters as the K-loop iterates. "Now von Neumann's obsolete": the
sequential load → compute → store loop diagram is struck through and torn as TMA and async WGMMA
overlap copy and math (producer/consumer warpgroups).

### `flash` (CH8: flash attention)
"Sharp left turn": the standard attention roadmap (Q·Kᵀ → write S to HBM → softmax → write P →
·V) with the thread travelling it; on "left turn" it swerves off into SRAM tiles (whip pan).
"and there you are": online softmax, a running max m and sum ℓ rescaling as each K/V tile streams
in. "Without a single CDR": a review schedule of HBM round trips, WRITE S / READ S / WRITE P /
READ P, each stamped NOT MATERIALIZED as the playhead skips them; the N×N matrix never exists.

### `pack` (CH9: data types, int4 packing, clamping)
"as paperclips fill the room": FP32 weights quantize into int4 nibbles, two per byte, packing into
a growing lattice that fills the frame (data type bit layouts: sign/exponent/mantissa → int4).
"Killswitch guy's on PTO": an out-of-office auto-reply card from the calibration job (`absmax`,
scale = max|x|/7). "Now there's nowhere left to go": values clamp at -8 and 7, the lattice closes in,
the words squeezed between the bins.

### `fuse` (CH9: fused dequant + AWQ)
"Too late now, we lit the fuse": kernel fusion: dequant → GEMM → bias, three kernels with HBM round
trips between them, fuse into one; the lyric is set along the fused kernel and the intermediate
buffers burn away as the thread passes. "Orthogonality thesis blues": AWQ, a scatter of channels
(activation magnitude vs weight magnitude); the salient 1% outliers are scaled up before quantizing;
"blues" gets the singer's pitch as a bending line.

### `pipeline` (CH10: pipeline parallelism, streams)
"Just transformers all the way!": an infinite vertical stack of transformer blocks split into
stages across GPU 0..3; the camera falls through, one block per beat, the quote one word per block;
micro-batches flow through with CUDA streams overlapping. "Till you learned to disobey": the fall
stops dead; one stage runs out of order (a pipeline bubble), the word "disobey" highlights
right-to-left.

### `cluster` (CH10: tensor parallelism, NCCL, ring vs tree)
"Post-Chinchilla, super-dense": a dense weight matrix is K-split across 8 GPUs; the slabs pack
tighter on each kick. "Breaking through each safety fence": partial sums break through GPU
boundaries (NVLink/PCIe) into an AllReduce. "Hundred thousand GPU": a top-down grid of 100,000
GPUs flickering in waves, mono counter. "RLHF goes askew": ring AllReduce, the ring tilts in steps
on the kicks, a tree AllReduce briefly corrects it, then it overcorrects into hook 4's angle.

### `cute` (CH11: CUTLASS and CuTe layouts)
"Just as foretold by Loom": the template tree of `CollectiveBuilder` choices (tile shape, cluster
shape, stages, schedule) resolved at compile time, the chosen path lit token by token. "From masked
pre-training days": predicated tiles, out-of-bounds elements masked as solid blocks that unmask.
"To recursive self-upgrade": the hierarchical tiling of a GEMM as Droste recursion, problem →
CTA tile → warp tile → MMA atom → thread values, each labelled with its CuTe `Shape:Stride`.

### `beyond` (CH12: now what?)
"What did Ilya see?": a dark room lit only by the glow of what the book left out: Triton,
torch.compile, vLLM, SGLang, TensorRT-LLM, JAX, Mojo, CuTe DSL, as labels on a wall. "We'll never
know": the ones out of scope get REDACTED bars. "Was it all for show?": a spotlight on an empty
stage finds the one line that generalizes, `blockIdx.x * blockDim.x + threadIdx.x`, which
collapses to the thread and detonates the outro.

### `outro`
Detonation: ACHIEVED / PEAK 1.00. Then it keeps being upped one value per beat for four bars, past
peak (1.01, 1.50, 2.00 with a deadpan footnote about the spec sheet), past physics (42, 1e30), until
it goes `inf` and then `NaN` (every CUDA programmer's final boss) on the drum stop. End card: the
book's title set like a paper's title, "CUDA for Deep Learning", then a lone "↻ Relaunch" button
that rewinds every plate and loops to the first frame.

## v2: "Chasing the Roofline" (original song, 2026-09-26)

Elliot asked for a redo on a new song: compose CUDA lyrics in chapter order, synthesize the vocal
ourselves, music first, then video. v1 plates are reused where the chapter matches.

Song: `song/lyrics.txt` (display text, chapter per line), `song/lyrics.sing.txt` (what the model
sings: phonetic spellings such as "G-P-U", "koo-blas", "em-nist"). Form copies p(doom)'s:
4-line AABB verses, a one-line plea pre-chorus, and a hook line ("I'm chasing the roofline", six
syllables like "I'm upping my P(doom)") that opens each chorus and rhymes on -ine for 3 or 4
lines, then a closing couplet. Generation: ACE-Step 1.5 XL (MIT, open weights) on the other GPU host, many
seeds, ranked by lyric word error rate (Demucs vocals, then Whisper) and Audiobox Aesthetics.

The hook becomes a roofline chart: log-log arithmetic intensity vs FLOP/s, a sloped memory roof
and a flat compute roof. The current kernel is the thread's dot, and each hook moves it closer
to the roof. ACHIEVED / PEAK keeps the v1 values; the outro's overshoot past 1.00 is the
classic missing `cudaDeviceSynchronize()` timing bug, then NaN.

| lyric | chapter | plate (v1 reuse) and staging |
|---|---|---|
| When PyTorch just isn't enough tonight | CH1 | `die`: a PyTorch one-liner `y = model(x)` peels open onto the kernel stack beneath |
| I launch ten thousand threads to hold you tight | CH1 | `die`: the thread multiplies into the GPU floorplan's SMs |
| cudaMemcpy, host to device | CH2 | `hostdev`: the arrays drop down the PCIe bus |
| Block times dim plus thread, and check the bounds twice | CH2 | `hostdev`: the index incantation assembles; the `if (i < N)` fence is stamped twice |
| GPU, please don't leave me on read | CH2 | `launch` printf: the kernel prints nothing until `cudaDeviceSynchronize()` is typed, then 1,024 lines flood in warp order |
| I'm chasing the roofline | — | `hook` ×4: the roofline chart, the dot climbs |
| Thirty-two threads marching in time / One little "if" splits the line / It's naive, but it's mine | CH3 | `grid`: 32 lanes in lockstep on the beat, divergence splits them, the naive GEMM thread |
| Convolve you with a three-by-three / Max-pool the world down to you and me | CH3 | `conv`: the 3×3 window sweeps; 2×2 pools collapse the map to two cells, YOU and ME |
| We trained a net on MNIST in C / Then naive CUDA, then cuBLAS set us free | CH4 | `mnist`: the five-level ladder with the book's timings (185.2 s down to 1.4 s) |
| Now you're embedding, you're attending / One token at a time, never ending | CH5 | `mnist` tail into a transformer block, then the decode loop |
| KV cache, please remember me | CH5 | `launch` decode: the KV cache grows a column per token; "me" is the stored token |
| Coalesce / Tile it in shared, sync / Vectorize, float4 | CH6 | `optimize`: 32 threads on one 128-byte cache line, shared tiles and the barrier, float4 loads |
| Twenty-seven teraflops a second / Sixty-six? That's tensor cores, I reckon | CH6→7 | `optimize`: odometer to 27.8 TFLOPS; cuBLAS's 66 bar towers over it and opens onto a tensor core |
| Load a fragment, MMA sync / Sixteen cubed, faster than you think | CH7 | `tensor`: fragments hand out across 32 lanes, one 16×16×16 mma |
| N-by-N? We never write it down / Keep a running max and pass it round | CH8 | `flash`: the N×N score matrix refuses to land in HBM; the running max and sum pass tile to tile |
| Precision, please don't let me go | CH9 | `launch` quant: the glyphs coarsen FP32 → BF16 → FP8 → INT4 |
| Thirty-two bits down to four / Scale it and round it / outliers step out of line | CH9 | `pack`: nibble packing, the rounding grid, outliers leaving the histogram |
| Move the scale to the weights, the product stays true / NormalFloat4 blues | CH9 | `fuse` restaged: AWQ's (w·s)(a/s) = w·a, then NF4's 16 quantile levels as a blues staff |
| Just split it over eight GPUs / NVLink, or else you lose / Slice the columns, slice the rows / All-reduce | CH10 | `cluster`: 8 GPUs, NVLink vs PCIe all-reduce (475 vs 8 to 12 GB/s), column/row split, ring |
| Pipeline stages, bubbles in between / Micro-batch the whole machine | CH10 | `pipeline`: the bubble Gantt chart fills as micro-batches go in |
| CUTLASS templates, nested design / Shape and a stride / Tiles within tiles | CH11 | `cute`: template error scroll, `(shape):(stride)`, recursive tiling |
| Now what? We'll never beat cuBLAS / Maybe PyTorch was enough | CH12 | `beyond`: the layer map (PyTorch, torch.compile, Triton, CUTLASS, CUDA, PTX); the thread climbs back to `y = model(x)`, matching the first frame |
| (outro) | — | `outro`: ACHIEVED / PEAK overshoots 1.00 (missing sync), NaN, end card |

### v2 plate windows (from `app/src/timeline.ts` on the aligned take; seconds)

die 0-13.6 · hostdev 13.6-21.2 · launch1 21.2-27.1 · hook1 27.1-29.4 · grid 29.4-37.5 ·
conv 37.5-45.6 · mnist 45.6-60.9 · launch2 60.9-66.8 · hook2 66.8-69.1 · optimize 69.1-85.7 ·
tensor 85.7-93.0 · flash 93.0-100.2 · launch3 100.2-106.5 · hook3 106.5-108.8 · pack 108.8-118.2 ·
fuse 118.2-125.4 · cluster 125.4-139.4 · pipeline 139.4-150.3 · hook4 150.3-152.1 · cute 152.1-161.5 ·
beyond 161.5-173.3 · outro 173.3-188.0. Song: 133.0 BPM (beat 0.451 s, bar 1.805 s), half-time groove
(kick on 1, snare on 3); sections in `data/audio.json`. Order change from v1: `cluster` (split over
eight GPUs, NVLink, columns/rows, all-reduce) now comes before `pipeline` (stages, bubbles,
micro-batch), matching the lyric and the chapter.

### v2 rules for scene authors
- Look lines up by a substring of the new text (`ly.get('Coalesce your loads')`); never hard-code times.
  Spelled words carry syllable times (`syl`): G/P/U, K/V, M/M/A, N-by-N, NV/Link, cut/lass, Py/Torch.
- Keep what still fits from the v1 plate (it was built for the same chapter); restage the lyric
  integration for the new lines per the table above. The v1 briefs below still hold where they
  agree with the table; where they disagree, the table wins.
- The hook is now the roofline chart (log-log, sloped memory roof, flat compute roof, the kernel's
  dot climbing), ACHIEVED / PEAK values unchanged (0.15 / 0.42 / 0.81 / 0.99), stepped on each sung
  "roofline". "I'm chasing the roofline" is sung I'm / chas-ing / the / roof-line.
- Numbers must match the book (CH6: naive ~0.3, coalesced ~2.5, tiled ~3, 1D blocktile 9.3,
  2D blocktile 25.1, vectorized 27.8 TFLOPS = 78% of the RTX 3090's 35.6 TFLOPS FP16 CUDA-core peak;
  cuBLAS 66 TFLOPS on tensor cores. CH7: WMMA 16x16x16, a fragment spread over 32 lanes;
  WGMMA = 4 warps / 128 threads. CH8: never write the N x N scores to HBM; running max m and sum l.
  CH9: FP32 4 GB -> INT8 1 GB -> INT4 500 MB per billion params; AWQ (w*s)(a/s) = w*a; NF4 = 16
  levels at normal quantiles. CH10: NVLink 4 = 900 GB/s vs PCIe Gen5 128 GB/s; NCCL AllReduce
  8-12 GB/s PCIe-only vs ~475 GB/s busbw on an 8-GPU NVLink node).

## v4: beginner cut (2026-09-26, after Elliot's review of v2)

Elliot's notes on v2: the rhymes sat off the beat and jargon (GPU, cuBLAS) sang badly; the visuals
must be snappier and locked to the beat; kernel scenes went too fast for a beginner; choose what to
teach instead of a line per chapter (the MNIST-in-C level didn't land). v4 fixes the song first:
`song/lyrics.v4.txt` (9 beginner concepts, one idea per verse, a repeated roofline chorus), take
b2-701 (133.0 BPM; every line is a 2-bar phrase starting ~1.2 beats before its bar; rhymes land at
the same spot in every line), aligned into `data/`.

### v4 rules (binding for every scene)
1. **Beginner first.** Each scene teaches ONE idea a newcomer could say back in one sentence; put
   that sentence as the first comment in the scene file. Plain-word labels ("memory speed", "math
   speed", "32 threads = 1 warp"), never profiler jargon. At most 2 numbers per scene, each from the
   book, with a tiny chapter tag.
2. **One hero diagram**, big (>= 60% of the frame), clean composition. No dense side panels, no code
   longer than one line, no dashboards.
3. **Locked to the beat.** Import `scenes/_lock.ts`: every change is a `snap()` that starts on the
   8th-note grid, completes in <= 90 ms, then HOLDS. Camera static within a phrase; camera changes
   only as a snap or a <= 0.15 s whip on a beat. No drift, no continuous zoom, no floaty easing;
   ambient life only as grain/flicker.
4. **Slow enough to follow.** A diagram step lands on a sung stressed word (quantized), <= 4 steps
   per lyric line, hold >= 1 beat after each step, and each concept stays readable >= 1 bar after its
   last step.
5. **Lyrics**: the sung line large (Archivo) with per-word karaoke, always in the same place within a
   scene, one line at a time (next line may pre-show dim <= 0.4 s early).
6. **Cuts** land on the beat at/before each phrase's first word (the timeline does this): first frame
   complete (no fade-in), last frame held (no fade-out) unless a hand-off is designed.
7. Palette, type, grain and karaoke rules from the treatment stay binding. Perf < 25 ms/frame.

### v4 scenes (windows in seconds from the timeline)
| scene | window | lines | one idea |
|---|---|---|---|
| `cpugpu` | 0-13.6 | intro; A CPU thinks with a few big cores / A GPU just throws ten thousand more | a CPU has a few big cores, a GPU thousands of small ones |
| `index` | 13.6-21.7 | One thread for every number you've got / Block times size plus thread finds its spot | one thread per number; it finds its number with block × size + thread |
| `launch` | 21.7-29.4 | Ten thousand threads all launch at once / So tell me, how fast did it run? | launching is easy; the question is how fast |
| `roofline` ×4 | 29.4-50.1, 73.1-91.6, 116.4-130.0, 159.3-173.3 | the chorus (+ breaks after 1 and 2) | speed is capped by memory (slope) or math (flat roof); more math per byte climbs |
| `warp` | 50.1-64.5 | Thirty-two threads in a warp.../ One instruction.../ One little "if".../ Half of them wait... | 32 threads run one instruction together; an if makes half wait |
| `slow` | 64.5-73.1 | Same old code, but it's crawling slow / Tell me, where'd all my speed go? | slow kernels mostly wait for memory (CH6: >90% stalled on memory) |
| `memory` | 91.6-108.3 | Neighbors read.../ One trip to memory.../ Load a tile to shared.../ Sync the block... | read neighbors together; keep reused data close in shared memory |
| `reuse` | 108.3-116.4 | Fetch it once and use it more / That's what the roofline's for | each reuse is more math per byte, which moves you right on the roofline |
| `tensor` | 130.0-136.3 | Tensor cores eat a matrix whole / Sixteen by sixteen in a single go | a tensor core multiplies a whole 16×16 tile in one instruction |
| `flash` | 136.3-143.5 | Flash attention never writes the square / Streams it through in tiles... | flash attention never stores the huge N×N square; tiles stream through fast memory |
| `quant` | 143.5-150.7 | Thirty-two bits down to four, and it's fine / Eight times less to move... | 4 bits instead of 32 means 8× less data to move |
| `multi` | 150.7-159.3 | When one GPU is not enough / Split it eight ways, then sum it up | split the work over 8 GPUs, then add the partial answers |
| `outro` | 173.3-190 | (instrumental, "night" echoes) | recap the path, end card |

The roofline chorus is one lesson that advances: line 1 draws the chart; "Memory's the slope"
lights the sloped roof (memory speed, RTX 3090 936 GB/s), "the math's the line" the flat roof (math
speed, 35.6 TFLOP/s); "Squeeze more math from every byte" pushes the dot right; "Till I'm hitting
the ceiling" lifts it to the kernel's measured speed. Chorus n plots the CH6 GEMM ladder: 0.3, 9.3,
25.1, 27.8 TFLOP/s (ACHIEVED / PEAK 0.01, 0.26, 0.70, 0.78), earlier dots kept as faint rings.
x positions are illustrative (no intensity number printed). The same layout every time; n = 1
teaches slowly, later ones move faster because the viewer already knows the chart.

## v5: 3D, zoomed in, high energy (2026-09-26, Elliot's v4 review)

Elliot: the song is right; the animations must zoom in fundamentally, be 3D, more expressive, with
intensity and movement all the time; every transition between parts should be creative (they may
repeat when the song repeats), with some really big transitions at the drops; model a real GPU die
(H100 or B200) and build the animations on it; for multi-GPU, model an actual DGX node, take it apart,
and show bytes moving across the GPUs (zoom into the wires, flicker faster as traffic rises).

Choice: **H100 (SXM5) and DGX H100**, because the book's reference numbers are H100's (CH7 989.5 /
67 TFLOP/s, CH10 NVLink 900 GB/s on an 8-GPU node). The CH6 roofline kernels were measured on an RTX
3090; the chorus chart keeps those numbers with their tag.

### v5 rules (replace v4 rule 3 "no drift"; the rest of v4 stays)
1. **One 3D world.** Every scene is a place on or inside the H100 (the DGX node for multi-GPU), built
   from the shared models `scenes/_h100.ts` and `scenes/_dgx.ts`. The story zooms in: package → die
   → GPC → SM → partition → lanes / tensor core → registers, and back out at the end.
2. **Always moving, locked to the beat.** The camera is never still: a continuous dolly/orbit whose
   speed breathes with the music (drums/rms envelopes), with its big moves landing on downbeats
   and accents on kicks/snares (`scenes/_cam.ts`). Data is always flowing (pulses along wires, SMs
   flickering with activity). No random shake, no noise wobble.
3. **Still teaches one idea.** The v4 one-sentence idea per scene stays. The lyric is a screen-space
   karaoke overlay (large, fixed place per scene); teaching labels are few, big, and either
   screen-space or camera-facing. The hero object fills the frame.
4. **Transitions at every cut** (timeline entries overlap; the engine composites with a chosen kind,
   `transition` on the entry): quick ones on ordinary cuts (whip, flash, glitch), big ones at the drops:
   into each chorus, into the bridge, into the final chorus, and the DGX teardown.
5. Palette, type and grain stay (ink, bone, graphite, ash, signal, ember). Materials read as dark
   silicon, graphite metal and bone hairlines; the only light sources are warm/orange activity and a
   cool-neutral key. No purple/cyan neon.

>>> NEXT
v5 done: out/final/cuda_roofline.mp4 (sha256 25bfa13c...), share copy 290 MB. Open weak points:
multi's last ~0.2 s before the final chorus is a white-out (the roof4 dive zooms into the brightest
NVLink lane); DGX fans do not visibly spin (the fan frame box sits in front of the blades in _dgx.ts);
the roofline chart's numbers are the book's RTX 3090 ones (labelled) drawn over the H100 die;
memory's scattered-trip contrast is thin; small mono labels in multi/memory; index's "i = 9" cell holds
the value 0 (index vs value may confuse a beginner).
