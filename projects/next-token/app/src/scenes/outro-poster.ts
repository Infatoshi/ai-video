// The outro's riso poster: the whole exchange as printed type, laid out from W/H (landscape or portrait).
// Every string comes from the recorded run (data/llm.json): the question, the answer exactly as it came
// out, the footnote from step 3's nucleus (and the letter count of the word it was asked about), and the
// colophon from the run's settings. `rects` are the named regions the edit zooms into.
import { F, font, glyphX, measure } from '../engine/type';
import type { Llm } from '../engine/llm';
import { portrait } from './_kit';
import { PINK, BLUE, INK, pct } from './finale-kit';

export type Focus = 'full' | 'title' | 'q' | 'ans1' | 'ans2' | 'two' | 'straw' | 'foot' | 'footTwo' | 'footThree' | 'three' | 'colo' | 'digits' | 'bigTwo' | 'bigThree';
export interface Rect { x: number; y: number; w: number; h: number }
export interface Inks { title: string; titleUnder: string; body: string; q: string }

interface Line { text: string; x: number; y: number; family: string; size: number }

export class Poster {
  question: string;
  answer: string;
  lines: { title: Line[]; q: Line[]; ans: Line[]; foot: Line[]; colo: Line[] };
  digits: { two: string; three: string; x2: number; y2: number; x3: number; y3: number; size: number };
  rects = {} as Record<Focus, Rect>;
  /** the footnote's pieces: [text, colour key] */
  footText: string;
  twoTok: string;
  threeTok: string;

  constructor(llm: Llm) {
    const P = portrait();
    this.question = llm.d.question;
    this.answer = llm.d.answer.replace(/<\|eot_id\|>$/, '');
    const st = llm.steps[3]!;
    const picked = st.nucleus[st.pick]!, other = st.nucleus.find((_, i) => i !== st.pick)!;
    this.twoTok = picked[0];
    this.threeTok = other[0];
    // the right answer: the r's in the word it was asked about (it matches the other face of step 3's die)
    const word = llm.d.question_tokens.find((t) => /straw/.test(t))!.trim();
    const count = [...word].filter((ch) => ch === 'r').length;
    if (String(count) !== this.threeTok) console.warn(`outro: r count ${count} is not step 3's other token ${this.threeTok}`);
    this.footText = `The dice said ${this.twoTok} (${pct(picked[1])}). The answer was ${count} (${pct(other[1])}).`;
    const model = llm.d.model.split('/').pop()!.replace(/-/g, ' ');
    const gpu = llm.d.gpu.replace(/^NVIDIA (GeForce )?/, '');
    const s = llm.d.sampling;
    const colo = [model, `temperature ${s.temperature}`, `top-p ${s.top_p}`, `seed ${s.seed}`, gpu, `${llm.steps.length} tokens`];

    const titleF = F.archivo(125, 900), mono = F.mono(700), footF = F.archivo(100, 700);
    const L = (text: string, x: number, y: number, family: string, size: number): Line => ({ text, x, y, family, size });
    // wrap words to a width
    const wrap = (text: string, family: string, size: number, maxW: number) => {
      const out: string[] = [];
      let cur = '';
      for (const w of text.split(' ')) {
        const next = cur ? `${cur} ${w}` : w;
        if (measure(next, family, size) > maxW && cur) { out.push(cur); cur = w; } else cur = next;
      }
      if (cur) out.push(cur);
      return out;
    };
    if (!P) {
      const x = 140;
      const ansSize = 88;
      const ans = wrap(this.answer, mono, ansSize, 1200);
      const foot = [this.footText];
      this.lines = {
        title: [L('NEXT TOKEN', x, 238, titleF, 150)],
        q: [L(this.question, x + 70, 362, mono, 40)],
        ans: ans.map((t, i) => L(t, x, 478 + i * 104, mono, ansSize)),
        foot: foot.map((t, i) => L(t, x, 478 + ans.length * 104 + 18 + i * 52, footF, 40)),
        colo: [L(colo.join(' · '), x, 478 + ans.length * 104 + 18 + foot.length * 52 + 48, mono, 24)],
      };
      this.digits = { two: this.twoTok, three: String(count), x2: 1555, y2: 530, x3: 1680, y3: 585, size: 460 };
    } else {
      const x = 130;
      const ansSize = 84;
      const ans = wrap(this.answer, mono, ansSize, 830);
      const q = wrap(this.question, mono, 36, 800);
      const foot = [this.footText.slice(0, this.footText.indexOf('. ') + 1), this.footText.slice(this.footText.indexOf('. ') + 2)];
      const qy = 610, ay = qy + q.length * 48 + 104, fy = ay + ans.length * 98 + 20;
      this.lines = {
        title: [L('NEXT', x, 330, titleF, 170), L('TOKEN', x, 490, titleF, 170)],
        q: q.map((t, i) => L(t, x + 60, qy + i * 48, mono, 36)),
        ans: ans.map((t, i) => L(t, x, ay + i * 98, mono, ansSize)),
        foot: foot.map((t, i) => L(t, x, fy + i * 50, footF, 38)),
        colo: [L(colo.slice(0, 1).join(' · '), x, 1480, mono, 24), L(colo.slice(1, 4).join(' · '), x, 1516, mono, 24), L(colo.slice(4).join(' · '), x, 1552, mono, 24)],
      };
      this.digits = { two: this.twoTok, three: String(count), x2: 760, y2: 1240, x3: 890, y3: 1300, size: 330 };
    }
    this.measureRects();
  }

  private bbox(ls: Line[]): Rect {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const l of ls) {
      const w = measure(l.text, l.family, l.size);
      x0 = Math.min(x0, l.x); x1 = Math.max(x1, l.x + w);
      y0 = Math.min(y0, l.y - l.size * 0.5); y1 = Math.max(y1, l.y + l.size * 0.5);
    }
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }
  /** Rect of text[i..j) in a line. */
  private sub(l: Line, i: number, j: number): Rect {
    const a = glyphX(l.text, i, l.family, l.size), b = glyphX(l.text, j, l.family, l.size);
    return { x: l.x + a, y: l.y - l.size * 0.42, w: b - a, h: l.size * 0.84 };
  }
  private find(ls: Line[], s: string, nth = 0): Rect {
    let k = 0;
    for (const l of ls) {
      let i = l.text.indexOf(s);
      while (i >= 0) {
        if (k++ === nth) return this.sub(l, i, i + s.length);
        i = l.text.indexOf(s, i + 1);
      }
    }
    return this.bbox(ls);
  }

  private measureRects() {
    const L = this.lines, d = this.digits;
    const all = [...L.title, ...L.q, ...L.ans, ...L.foot, ...L.colo];
    const r = this.rects;
    r.full = this.bbox(all);
    const dg: Rect = { x: d.x2 - d.size * 0.36, y: d.y2 - d.size * 0.42, w: d.x3 - d.x2 + d.size * 0.72, h: d.y3 - d.y2 + d.size * 0.84 };
    r.full = union(r.full, dg);
    r.title = this.bbox(L.title);
    r.q = this.bbox(L.q);
    r.ans1 = this.bbox(L.ans.slice(0, 1));
    r.ans2 = this.bbox(L.ans.slice(1));
    r.two = this.find(L.ans, this.twoTok);
    r.straw = this.find(L.ans, 'strawberry');
    r.foot = this.bbox(L.foot);
    const f2 = `${this.twoTok} (`, f3 = `${this.digits.three} (`;
    const i2 = this.footText.indexOf(f2), i3 = this.footText.indexOf(f3);
    const j2 = this.footText.indexOf(')', i2) + 1, j3 = this.footText.indexOf(')', i3) + 1;
    r.footTwo = this.findRange(L.foot, i2, j2);
    r.footThree = this.findRange(L.foot, i3, j3);
    r.three = this.findRange(L.foot, i3, i3 + 1);
    r.colo = this.bbox(L.colo);
    r.digits = dg;
    r.bigTwo = { x: d.x2 - d.size * 0.34, y: d.y2 - d.size * 0.38, w: d.size * 0.68, h: d.size * 0.76 };
    r.bigThree = { x: d.x3 - d.size * 0.34, y: d.y3 - d.size * 0.38, w: d.size * 0.68, h: d.size * 0.76 };
  }
  /** Rect of footText[i..j) across the (possibly split) footnote lines. */
  private findRange(ls: Line[], i: number, j: number): Rect {
    let off = 0;
    for (const l of ls) {
      const k = this.footText.indexOf(l.text, off);
      if (i >= k && j <= k + l.text.length) return this.sub(l, i - k, j - k);
      off = k + l.text.length;
    }
    return this.bbox(ls);
  }

  /** Draw the whole poster in poster space (the caller sets the view transform). */
  draw(c: CanvasRenderingContext2D, inks: Inks, misreg: [number, number]) {
    const L = this.lines, d = this.digits;
    c.save();
    c.textBaseline = 'middle';
    c.textAlign = 'left';
    // the big digits: the dice said 2, the answer was 3 (overprinted)
    c.font = font(F.archivo(125, 900), d.size);
    c.textAlign = 'center';
    c.fillStyle = PINK;
    c.fillText(d.two, d.x2, d.y2);
    c.globalCompositeOperation = 'multiply';
    c.fillStyle = BLUE;
    c.fillText(d.three, d.x3, d.y3);
    c.globalCompositeOperation = 'source-over';
    c.textAlign = 'left';
    // title: two inks, the underprint out of register by this print's offset
    for (const l of L.title) {
      c.font = font(l.family, l.size);
      c.fillStyle = inks.titleUnder;
      c.fillText(l.text, l.x + misreg[0] * l.size * 0.03, l.y + misreg[1] * l.size * 0.03);
      c.globalCompositeOperation = 'multiply';
      c.fillStyle = inks.title;
      c.fillText(l.text, l.x, l.y);
      c.globalCompositeOperation = 'source-over';
    }
    // the question, marked Q; the answer, marked A, its digit in pink
    for (const [i, l] of L.q.entries()) {
      c.font = font(l.family, l.size);
      if (i === 0) { c.fillStyle = PINK; c.fillText('Q', l.x - l.size * 1.5, l.y); }
      c.fillStyle = inks.q;
      c.fillText(l.text, l.x, l.y);
    }
    for (const l of L.ans) this.pieces(c, l, [[this.twoTok, PINK]], inks.body);
    // footnote: the pick in pink, the answer in blue
    const f2 = `${this.twoTok} (`, f3 = `${d.three} (`;
    for (const l of L.foot) {
      const marks: [string, string][] = [];
      const a = l.text.indexOf(f2), b = l.text.indexOf(f3);
      if (a >= 0) marks.push([l.text.slice(a, l.text.indexOf(')', a) + 1), PINK]);
      if (b >= 0) marks.push([l.text.slice(b, l.text.indexOf(')', b) + 1), BLUE]);
      this.pieces(c, l, marks, inks.body);
    }
    // rule and colophon
    const top = L.colo[0]!;
    c.fillStyle = INK;
    c.fillRect(top.x, top.y - top.size * 1.3, this.rects.colo.w, 3);
    for (const l of L.colo) { c.font = font(l.family, l.size); c.fillStyle = INK; c.fillText(l.text, l.x, l.y); }
    c.restore();
  }

  /** A line with some substrings in other inks (kerned as one run). */
  private pieces(c: CanvasRenderingContext2D, l: Line, marks: [string, string][], base: string) {
    c.font = font(l.family, l.size);
    const spans: { i: number; j: number; col: string }[] = [];
    for (const [s, col] of marks) {
      const i = l.text.indexOf(s);
      if (i >= 0) spans.push({ i, j: i + s.length, col });
    }
    spans.sort((a, b) => a.i - b.i);
    let at = 0;
    const draw = (i: number, j: number, col: string) => {
      if (j <= i) return;
      c.fillStyle = col;
      c.fillText(l.text.slice(i, j), l.x + glyphX(l.text, i, l.family, l.size), l.y);
    };
    for (const s of spans) { draw(at, s.i, base); draw(s.i, s.j, s.col); at = s.j; }
    draw(at, l.text.length, base);
  }
}

const union = (a: Rect, b: Rect): Rect => {
  const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  return { x, y, w: Math.max(a.x + a.w, b.x + b.w) - x, h: Math.max(a.y + a.h, b.y + b.h) - y };
};

