// The edit: which scene plays when. Boundaries are anchored to lyric lines and snapped to the beat
// grid, so they follow the aligned data (data/lyrics.json, data/audio.json). SPEC.md has the shot plan.
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
  // YouTube thumbnails (scenes/thumb.ts): only with --only thumbA,...,thumbE, never in the edit
  if (typeof location !== 'undefined' && new URLSearchParams(location.search).get('only')?.startsWith('thumb'))
    return (['A', 'B', 'C', 'D', 'E', 'I'] as const).map((v, i) => ({ id: `thumb${v}`, load: scene('thumb'), start: i, end: i + 1, params: { v } }));
  /** Cut on the last beat at/before the first word of the matching line (never after the word). */
  const cut = (q: string, nth = 0, tol = 0.02) => {
    const s = ly.get(q, nth).words[0]!.start;
    return au.timeOfBeat(Math.floor(au.beatAt(s + tol)));
  };
  const sec = (name: string) => au.sections.find((x) => x.name === name);
  const snapDown = (t: number) => au.downbeats.reduce((b, d) => (Math.abs(d - t) < Math.abs(b - t) ? d : b), au.downbeats[0] ?? t);

  const b = {
    tokens: cut('Type a question'),
    vectors: cut('Every token is a vector'),
    hook1: cut('Just one more token', 0),
    attention: cut('Every token looks back'),
    ffn: cut('Then the feed-forward'),
    hook2: cut('Just one more token', 1),
    logits: cut('At the top of the tower'),
    append: cut('Roll the dice'),
    memory: cut('Every single token, it reads'),
    roofline: snapDown(sec('drop')?.start ?? ly.get('Sixteen gigs').end + 1),
    spec: cut('Stuck on the slope'),
    finale: cut('Just one more token', 2),
    outro: snapDown(sec('outro')?.start ?? ly.get('Till it picks').start - 3),
    end: au.duration,
  };

  const E = (id: string, file: string, start: number, end: number, extra: Partial<TimelineEntry> = {}): TimelineEntry =>
    ({ id, load: scene(file), start, end, ...extra });

  // every cut gets a transition; the drops (into each hook, the drop itself, the finale) get the big ones
  const TR: Record<string, TransitionSpec> = {
    tokens: { kind: 'flash', dur: 0.3 },
    vectors: { kind: 'whip', dur: 0.3 },
    hook1: { kind: 'dive', dur: 0.9 },
    attention: { kind: 'shatter', dur: 0.7 },
    ffn: { kind: 'glitch', dur: 0.3 },
    hook2: { kind: 'glitch', dur: 0.35 },   // short, and keeps detail (the streak blurred a whole beat to paper)
    logits: { kind: 'pixel', dur: 0.4 },
    append: { kind: 'whip', dur: 0.3, dir: [-1, 0] },
    memory: { kind: 'iris', dur: 0.8 },
    roofline: { kind: 'dive', dur: 1.2 },
    spec: { kind: 'shatter', dur: 0.7 },
    finale: { kind: 'streak', dur: 1.0, pre: 0.2 },   // spec's four tiles leave the roof on "go" just before the cut
    outro: { kind: 'iris', dur: 0.9 },
  };
  const withTransitions = (list: TimelineEntry[]) => {
    for (let i = 1; i < list.length; i++) {
      const e = list[i]!, prev = list[i - 1]!, spec = TR[e.id];
      if (!spec || prev.end !== e.start) continue;
      const d = spec.dur ?? 0.5, pre = spec.pre ?? 0.5, c = e.start;
      e.start = c - d * pre;
      prev.end = c + d * (1 - pre);
      e.transition = spec;
    }
    return list;
  };

  return withTransitions([
    E('intro', 'intro', 0, b.tokens),
    E('tokens', 'tokens', b.tokens, b.vectors),
    E('vectors', 'vectors', b.vectors, b.hook1),
    E('hook1', 'hook', b.hook1, b.attention, { params: { n: 0 } }),
    E('attention', 'attention', b.attention, b.ffn),
    E('ffn', 'ffn', b.ffn, b.hook2),
    E('hook2', 'hook', b.hook2, b.logits, { params: { n: 1 } }),
    E('logits', 'logits', b.logits, b.append),
    E('append', 'append', b.append, b.memory),
    E('memory', 'memory', b.memory, b.roofline),
    E('roofline', 'roofline', b.roofline, b.spec),
    E('spec', 'spec', b.spec, b.finale),
    E('finale', 'finale', b.finale, b.outro),
    E('outro', 'outro', b.outro, b.end),
  ]);
}
