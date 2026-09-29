// data/timeline.json for yt.py's chapters: the lesson's parts (story.ts phases, anchored to the sung lines), since
// the edit itself is one continuous scene.   bun scripts/chapters.ts
import { Lyrics } from '../src/engine/lyrics';
const ly = new Lyrics(await Bun.file('../data/lyrics.json').json());
const au = await Bun.file('../data/audio.json').json();
const at = (q: string, n = 0) => ly.get(q, n).start;
const rows: [string, number][] = [
  ['intro', 0], ['task', at('Two spirals')], ['weights', at('Inside the machine')], ['loss', at('So measure how wrong')],
  ['loop', at('^Measure how wrong it is', 0)], ['gradient', at("It can't see the valley")], ['loop2', at('^Measure how wrong it is', 1)],
  ['toobig', at('Now make every step')], ['loop3', at('^Measure how wrong it is', 2)], ['outro', at('Three thousand one hundred')],
];
const tl = rows.map(([id, s], i) => ({ id, start: Math.max(0, s - 0.5), end: rows[i + 1]?.[1] ?? au.duration }));
await Bun.write('../data/timeline.json', JSON.stringify(tl, null, 1));
console.log(tl.map((e) => `${e.id} ${e.start.toFixed(1)}`).join('  '));
