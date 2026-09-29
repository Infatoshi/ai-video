// The edit: which scene plays when. Boundaries are anchored to lyric lines and snapped
// to the beat grid, so they follow the aligned data (data/lyrics.json, data/audio.json).
import type { TimelineEntry } from './engine/engine';
import type { TransitionSpec } from './engine/transitions';
import type { SceneClass } from './engine/scene';
import type { Lyrics } from './engine/lyrics';
import type { AudioData } from './engine/audio';

// Scene modules are discovered lazily so a missing/broken scene never breaks the build.
const modules = import.meta.glob<{ default: SceneClass }>('./scenes/*.ts');
const scene = (name: string) => () => {
  const m = modules[`./scenes/${name}.ts`];
  return m ? m() : Promise.reject(new Error(`scene module not found: scenes/${name}.ts`));
};

export function makeTimeline(ly: Lyrics, au: AudioData): TimelineEntry[] {
  // YouTube thumbnails (scenes/thumb.ts): only with --only thumbD,thumbE, never in the edit
  if (typeof location !== 'undefined' && new URLSearchParams(location.search).get('only')?.startsWith('thumb'))
    return (['D', 'E'] as const).map((v, i) => ({ id: `thumb${v}`, load: scene('thumb'), start: i, end: i + 1, params: { v } }));
  /** Cut on the last beat at/before the first word of the matching line (never after the word). */
  const cut = (q: string, nth = 0, tol = 0.02) => {
    const s = ly.get(q, nth).words[0]!.start;
    return au.timeOfBeat(Math.floor(au.beatAt(s + tol)));
  };
  /** The n-th downbeat after t. */
  const nextDownbeat = (t: number, n = 0) => {
    const k = au.downbeats.findIndex((d) => d >= t - 0.01);
    return au.downbeats[Math.min(au.downbeats.length - 1, (k < 0 ? au.downbeats.length - 1 : k) + n)]!;
  };
  /** Nearest downbeat to the end of a line. */
  const after = (q: string, nth = 0) => {
    const e = ly.get(q, nth).end;
    return au.downbeats.reduce((b, d) => (Math.abs(d - e) < Math.abs(b - e) ? d : b), au.downbeats[0] ?? e);
  };

  const b = {
    index: cut('One thread for every'),
    launch: cut('Ten thousand threads'),
    roof1: cut("I'm chasing", 0),
    warp: cut('Thirty-two threads in a warp'),
    slow: cut('Same old code'),
    roof2: cut("I'm chasing", 1),
    memory: cut('Neighbors read'),
    reuse: cut('Fetch it once'),
    roof3: cut("I'm chasing", 2),
    tensor: cut('Tensor cores'),
    flash: cut('Flash attention'),
    quant: cut('Thirty-two bits'),
    multi: cut('When one GPU'),
    roof4: cut("I'm chasing", 3),
    // roof4 holds its final chart through the outro's first bar; the outro opens on its recap
    outro: nextDownbeat((au.sections.find((x) => x.name === 'outro')?.start ?? after('ceiling tonight', 3)) + 0.1),
    end: au.duration,
  };

  const E = (id: string, file: string, start: number, end: number, extra: Partial<TimelineEntry> = {}): TimelineEntry =>
    ({ id, load: scene(file), start, end, ...extra });

  // v5: every cut gets a transition; the drops (into each chorus, the bridge, the final chorus) get the
  // big travelling ones. A scene that designs its own entrance sets `handlesTransition` and ignores these.
  const TR: Record<string, TransitionSpec> = {
    index: { kind: 'whip', dur: 0.3 },
    launch: { kind: 'flash', dur: 0.3 },
    roof1: { kind: 'dive', dur: 1.2 },          // drop: into chorus 1
    warp: { kind: 'shatter', dur: 0.8 },
    slow: { kind: 'glitch', dur: 0.35 },
    roof2: { kind: 'streak', dur: 1.2, center: [0.5, 0.2] },  // drop: into chorus 2, bursting out of slow's MEMORY
    memory: { kind: 'dive', dur: 0.8, center: [0.4, 0.45] },  // opens out of memory's lit SM
    reuse: { kind: 'pixel', dur: 0.45 },
    roof3: { kind: 'shatter', dur: 1.0, center: [0.52, 0.53] },  // drop: into chorus 3, breaking out of reuse's dot
    tensor: { kind: 'streak', dur: 1.3, center: [0.5, 0.6] },  // drop: the bridge, on tensor's dive target
    flash: { kind: 'glitch', dur: 0.35 },
    quant: { kind: 'pixel', dur: 0.45 },
    multi: { kind: 'whip', dur: 0.4, dir: [-0.6, 0.8] },  // content slides down-right, with quant's packets
    roof4: { kind: 'dive', dur: 1.5 },          // the biggest drop: the final chorus
    outro: { kind: 'iris', dur: 0.9, center: [0.54, 0.8] },  // opens out of roof4's orb
  };
  /** Overlap each entry with the previous one around their cut, as its transition asks. */
  const withTransitions = (list: TimelineEntry[]) => {
    for (let i = 1; i < list.length; i++) {
      const e = list[i]!, prev = list[i - 1]!, spec = TR[e.id];
      if (!spec || prev.end !== e.start) continue;
      const d = spec.dur ?? 0.5, pre = spec.pre ?? 0.5, cut = e.start;
      e.start = cut - d * pre;
      prev.end = cut + d * (1 - pre);
      e.transition = spec;
    }
    return list;
  };

  // One idea per scene, in the book's chapter order (SPEC.md "v4" and "v5").
  return withTransitions([
    E('cpugpu', 'cpugpu', 0, b.index),                                                   // CH1 few big cores vs thousands
    E('index', 'index', b.index, b.launch),                                              // CH2 one thread per number, the index
    E('launch', 'launch', b.launch, b.roof1),                                            // CH2 launch 10k threads; how fast?
    E('roof1', 'roofline', b.roof1, b.warp, { params: { n: 1 } }),                       // roofline lesson 1 (+ break)
    E('warp', 'warp', b.warp, b.slow),                                                   // CH3 warps in lockstep, divergence
    E('slow', 'slow', b.slow, b.roof2),                                                  // waiting on memory
    E('roof2', 'roofline', b.roof2, b.memory, { params: { n: 2 } }),                     // lesson 2 (+ break)
    E('memory', 'memory', b.memory, b.reuse),                                            // CH6 coalescing, shared tiles
    E('reuse', 'reuse', b.reuse, b.roof3),                                               // CH6 reuse = more math per byte
    E('roof3', 'roofline', b.roof3, b.tensor, { params: { n: 3 } }),                     // lesson 3
    E('tensor', 'tensor', b.tensor, b.flash),                                            // CH7 tensor cores
    E('flash', 'flash', b.flash, b.quant),                                               // CH8 flash attention
    E('quant', 'quant', b.quant, b.multi),                                               // CH9 fewer bits
    E('multi', 'multi', b.multi, b.roof4),                                               // CH10 split across GPUs
    E('roof4', 'roofline', b.roof4, b.outro, { params: { n: 4 } }),                      // lesson 4
    E('outro', 'outro', b.outro, b.end),
  ]);
}
