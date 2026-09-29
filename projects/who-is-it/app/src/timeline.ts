// The edit: one continuous picture (scenes/sentence.ts renders the whole song from song time), split into
// entries only so chapters and per-section checks have names. Entries touch with no transition: the
// picture is the same object on both sides of every boundary. Boundaries are anchored to lyric lines and
// snapped to the beat grid, so they follow the aligned data (data/lyrics.json, data/audio.json).
import type { TimelineEntry } from './engine/engine';
import type { SceneClass } from './engine/scene';
import type { Lyrics } from './engine/lyrics';
import type { AudioData } from './engine/audio';

const modules = import.meta.glob<{ default: SceneClass }>('./scenes/*.ts');
const scene = (name: string) => () => {
  const m = modules[`./scenes/${name}.ts`];
  return m ? m() : Promise.reject(new Error(`scene module not found: scenes/${name}.ts`));
};

export function makeTimeline(ly: Lyrics, au: AudioData): TimelineEntry[] {
  // YouTube thumbnails (scenes/thumb.ts): only with --only thumbA,..., never in the edit
  if (typeof location !== 'undefined' && new URLSearchParams(location.search).get('only')?.startsWith('thumb'))
    return (['A', 'B', 'C'] as const).map((v, i) => ({ id: `thumb${v}`, load: scene('thumb'), start: i, end: i + 1, params: { v } }));
  /** The beat at/before the first word of the matching line (never after the word). */
  const cut = (q: string, nth = 0, tol = 0.02) => {
    const s = ly.get(q, nth).words[0]!.start;
    return au.timeOfBeat(Math.floor(au.beatAt(s + tol)));
  };
  const b: [string, number][] = [
    ['intro', 0],
    ['who-is-it', cut('The animal didn')],
    ['attention', cut('Every word looks back', 0)],
    ['query-and-keys', cut("The word that's asking")],
    ['attention-2', cut('Every word looks back', 1)],
    ['left-to-right', cut('But it reads from left')],
    ['all-the-heads', cut('Each way of looking')],
    ['attention-3', cut('Every word looks back', 2)],
    ['outro', cut("Who's it? Just look back")],
  ];
  return b.map(([id, start], i) => ({ id, load: scene('sentence'), start, end: b[i + 1]?.[1] ?? au.duration }));
}
