// The edit: one continuous scene for the whole song (the course's pace rules: no cuts, the camera moves
// instead). scenes/world.ts places every camera move and label on the lyric lines. SPEC.md has the shot plan.
import type { TimelineEntry } from './engine/engine';
import type { SceneClass } from './engine/scene';
import type { Lyrics } from './engine/lyrics';
import type { AudioData } from './engine/audio';

const modules = import.meta.glob<{ default: SceneClass }>('./scenes/*.ts');
const scene = (name: string) => () => {
  const m = modules[`./scenes/${name}.ts`];
  return m ? m() : Promise.reject(new Error(`scene module not found: scenes/${name}.ts`));
};

export function makeTimeline(_ly: Lyrics, au: AudioData): TimelineEntry[] {
  // YouTube thumbnails (scenes/thumb.ts): only with --only thumbA,..., never in the edit
  if (typeof location !== 'undefined' && new URLSearchParams(location.search).get('only')?.startsWith('thumb'))
    return (['A', 'B', 'C'] as const).map((v, i) => ({ id: `thumb${v}`, load: scene('thumb'), start: i, end: i + 1, params: { v } }));
  return [{ id: 'world', load: scene('world'), start: 0, end: au.duration + 0.01 }];
}
