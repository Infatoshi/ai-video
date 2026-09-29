// The story clock: which training step the network is at, at song time t, and which part of the lesson is on.
// Steps are anchored to sung words (data/lyrics.json), so the picture and the lyric agree: chorus 1's "step" is
// the first real step, the outro's "steps" lands on step 3,121 (the first step with every dot right).
import type { AudioData } from './engine/audio';
import type { Lyrics, Line } from './engine/lyrics';
import type { Model } from './engine/model';
import { clamp, ease } from './engine/util';

export type Phase = 'intro' | 'task' | 'knobs' | 'loss' | 'chorus' | 'slope' | 'big' | 'outro';

export class Story {
  /** [song time, step] keys of the run's clock. */
  keys: [number, number][] = [];
  /** [song time, step] keys of the too-big run (the bridge). */
  bigKeys: [number, number][] = [];
  /** When the bridge's picture (the too-big run) hands back to the landscape. */
  brx = 0;
  /** Phase windows in song time. */
  phases: { t0: number; t1: number; phase: Phase; n?: number }[] = [];
  /** Chorus windows (first word of "Measure" .. last word of "again"), 0..2. */
  choruses: { t0: number; t1: number; lines: Line[] }[] = [];

  constructor(public model: Model, public ly: Lyrics, public au: AudioData) {
    const line = (q: string, nth = 0) => { try { return ly.get(q, nth); } catch { return null; } };
    const word = (q: string, w: string, nth = 0, which: 'start' | 'end' = 'start') => {
      const l = line(q, nth);
      const x = l?.words.find((y) => y.w.toLowerCase().replace(/[^a-z']/g, '').startsWith(w));
      return x ? x[which] : undefined;
    };
    const dur = au.duration;
    const v1 = line('Two spirals')?.start ?? 16;
    const v2 = line('Inside the machine')?.start ?? v1 + 30;
    const v3 = line('So measure how wrong')?.start ?? v2 + 30;
    const ch = [0, 1, 2].map((n) => {
      const a = line('^Measure how wrong it is', n), d = line('And do it again', n);
      const lines = [a, line('Find which way is down', n), line('Take one small step', n), d].filter((x): x is Line => !!x);
      return a && d ? { t0: a.start, t1: d.end, lines } : null;
    });
    const v4 = line("It can't see the valley")?.start ?? (ch[0]?.t1 ?? v3 + 40) + 8;
    const br = line('Now make every step')?.start ?? (ch[1]?.t1 ?? v4 + 40) + 2;
    const brEnd = line('It bounces and bounces')?.end ?? br + 20;
    const bounce = line('It bounces and bounces')?.start ?? brEnd - 4;
    const out = line('Three thousand one hundred')?.start ?? (ch[2]?.t1 ?? brEnd + 30) + 4;
    // the bridge's picture: until the last chorus, or 18 s after the bridge's last line if the break is long
    const c2s = ch[2]?.t0 ?? brEnd + 12;
    const brx = (this.brx = Math.min(c2s - 3, brEnd + 18));
    this.choruses = ch.filter((x): x is NonNullable<typeof x> => !!x);
    const c0 = ch[0] ?? { t0: v3 + 24, t1: v3 + 40, lines: [] }, c1 = ch[1] ?? { t0: v4 + 24, t1: v4 + 40, lines: [] }, c2 = ch[2] ?? { t0: brEnd + 10, t1: brEnd + 26, lines: [] };

    // the run's clock: frozen at step 0 until chorus 1's "step"; one step per "again"; then it rolls
    const step1 = word('Take one small step', 'step', 0, 'end') ?? c0.t0 + 8;
    const again1 = word('And do it again', 'again', 0) ?? c0.t1 - 2;
    const again2 = line('And do it again', 0)?.words.filter((w) => w.w.toLowerCase().startsWith('again'))[1]?.start ?? again1 + 1.5;
    const steps = word('Three thousand one hundred', 'steps') ?? out + 3;
    const k: [number, number][] = [
      [0, 0], [step1 - 0.9, 0], [step1, 1], [again1, 1], [again1 + 1.0, 2], [again2, 2], [again2 + 1.0, 3],
      [c0.t1 + 1.0, 12],
      [v4, 300],                        // the instrumental after chorus 1 rolls the ball across the plateau
      [c1.t0, 600],                     // verse 4 (the slope, the step size) creeps on
      [c1.t1, 1500], [br, 2000],        // chorus 2 runs it
      [brx + 1.5, 2000],                // the bridge holds the run (the too-big run plays)
      [c2.t0, brx + 8 < c2.t0 ? 2400 : 2000], // a long break after it: the ball rolls on under the solo
      [c2.t1, 3000], [steps, this.model.f.first_all_right], [dur, this.model.f.first_all_right],
    ];
    this.keys = k.sort((a, b) => a[0] - b[0]);
    for (let i = 1; i < this.keys.length; i++) this.keys[i]![1] = Math.max(this.keys[i]![1], this.keys[i - 1]![1]);
    // the too-big run: its first steps while the bridge explains the jump (step 9 is the one cut open), then all 5,000
    this.bigKeys = [[br + 0.5, 0], [bounce, 9], [Math.max(bounce + 6, brx - 3), 5000]];

    const P = (t0: number, t1: number, phase: Phase, n?: number) => this.phases.push({ t0, t1, phase, n });
    P(0, v1, 'intro'); P(v1, v2, 'task'); P(v2, v3, 'knobs'); P(v3, c0.t0, 'loss');
    P(c0.t0, v4, 'chorus', 0); P(v4, c1.t0, 'slope'); P(c1.t0, br, 'chorus', 1);
    P(br, brx, 'big'); P(brx, c2.t0, 'slope'); P(c2.t0, out, 'chorus', 2); P(out, dur + 1, 'outro');
  }

  /** Training step at song time t (fractional: the picture moves between steps). */
  step(t: number): number {
    const k = this.keys;
    if (t <= k[0]![0]) return k[0]![1];
    for (let i = 0; i < k.length - 1; i++) {
      const [ta, sa] = k[i]!, [tb, sb] = k[i + 1]!;
      if (t < tb) {
        const u = clamp((t - ta) / Math.max(1e-3, tb - ta));
        // big ranges move in log space (slow at first, then rolling), single steps ease
        if (sb - sa > 20) return Math.expm1(Math.log1p(sa) + (Math.log1p(sb) - Math.log1p(sa)) * u);
        return sa + (sb - sa) * ease.inOutCubic(u);
      }
    }
    return k[k.length - 1]![1];
  }
  bigStep(t: number): number {
    const k = this.bigKeys;
    if (t <= k[0]![0]) return 0;
    if (t >= k[2]![0]) return k[2]![1];
    if (t < k[1]![0]) return (k[1]![1] * (t - k[0]![0])) / Math.max(1e-3, k[1]![0] - k[0]![0]);
    const u = clamp((t - k[1]![0]) / (k[2]![0] - k[1]![0]));
    return Math.expm1(Math.log1p(k[1]![1]) + (Math.log1p(k[2]![1]) - Math.log1p(k[1]![1])) * u);
  }
  phaseAt(t: number) {
    return this.phases.find((p) => t >= p.t0 && t < p.t1) ?? this.phases[this.phases.length - 1]!;
  }
}
