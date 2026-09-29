// The edit: one continuous flight (scenes/flight.ts, the course's pace rules: no cuts). The flight is split into
// chapter entries that all run the same pure-function-of-time scene, so the switch between them is invisible and
// the timeline carries the chapters (render.ts timeline -> data/timeline.json -> publish chapters).
import type { TimelineEntry } from './engine/engine';
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
  // YouTube thumbnails (scenes/thumb.ts): only with --only thumbA,..., never in the edit
  if (typeof location !== 'undefined' && new URLSearchParams(location.search).get('only')?.startsWith('thumb'))
    return (['A', 'B', 'C'] as const).map((v, i) => ({ id: `thumb${v}`, load: scene('thumb'), start: i, end: i + 1, params: { v } }));
  /** The last beat at/before the first word of the matching line. */
  const at = (q: string, nth = 0) => au.timeOfBeat(Math.floor(au.beatAt(ly.get(q, nth).words[0]!.start + 0.02)));
  const marks: [string, number][] = [
    ['intro', 0],
    ['table', at('Before a chatbot') - 2],
    ['definition', at('Every word is a list of numbers', 0)],
    ['close', at('Close means they point') - 2],
    ['definition-2', at('Every word is a list of numbers', 1)],
    ['direction', at('Draw an arrow from man') - 2],
    ['king', at('Now the famous one')],
    ['together', at('Every word is a list of numbers', 2)],
    ['outro', at('Next time')],
  ];
  return marks.map(([id, s], i) => ({ id, load: scene('flight'), start: Math.max(0, s), end: marks[i + 1]?.[1] ?? au.duration }));
}
