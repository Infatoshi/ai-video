// hook (the chorus, three entries via params.n): "Just one more token, / One more roll of the dice, /
// Guess it, pick it, feed it back in, / And do it again all night". Each chorus is one real sampling
// step: k = the answer token the story emits on this chorus's "pick". Its nucleus (temperature 0.6,
// top-p 0.9) is the die and the odds bar; its uniform draw u is where the needle lands. Cuts every
// 2 beats; the four lines each rotate through their own shots.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease } from '../engine/util';
import { showTok } from '../engine/llm';
import { Wash, tokenRain, specks, makeCam, drive, shot, snapIn, portrait, perFrame, tokenFaceTexture, inkMat, outline, tokenTile } from './_kit';
import { Tower, riderTile } from './_tower';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');

interface Cand { t: string; p: number; inDraw: boolean }

export default class Hook extends Scene {
  scene = new THREE.Scene();
  cam = makeCam(36);
  wash = new Wash();
  back = new Layer2D();
  mid = new Layer2D();
  front = new Layer2D();
  tower = new Tower();
  die!: THREE.Mesh;
  dieQ = new THREE.Quaternion();
  /** The rider: a '?' until the pick, then the picked token. */
  riderQ!: THREE.Mesh;
  riderT!: THREE.Mesh;
  k = 0;
  cands: Cand[] = [];
  /** Word times: line starts and the guess / pick / feed / again words. */
  tm = { l1: 0, l2: 0, l3: 0, l4: 0, guess: 0, pick: 0, feed: 0, again: 0 };

  override init() {
    const { lyrics: ly, llm, story } = this.ctx;
    const n = this.ctx.params.n ?? 0;
    const word = (line: string, w: string, nth: number) => {
      const l = ly.get(line, nth);
      return (l.words.find((x) => x.w.toLowerCase().replace(/[^a-z]/g, '').startsWith(w)) ?? l.words[0]!).start;
    };
    this.tm = {
      l1: ly.get('Just one more token', n).start, l2: ly.get('One more roll of the dice', n).start,
      l3: ly.get('Guess it, pick it', n).start, l4: ly.get('do it again all night', n).start,
      guess: word('Guess it, pick it', 'guess', n), pick: word('Guess it, pick it', 'pick', n),
      feed: word('Guess it, pick it', 'feed', n), again: word('do it again all night', 'again', n),
    };
    // the step emitted on this chorus's "pick"
    let best = 0;
    story.emits.forEach((t, i) => { if (Math.abs(t - this.tm.pick) < Math.abs(story.emits[best]! - this.tm.pick)) best = i; });
    this.k = best;
    const st = llm.steps[this.k]!;
    const inDraw = new Set(st.nucleus.map(([t]) => t));
    // the die's six faces: the nucleus it sampled from, then the next candidates at T=1 (outside top-p)
    this.cands = st.nucleus.slice(0, 6).map(([t, p]) => ({ t, p, inDraw: true }));
    for (const [t, p] of st.top10_t1) if (this.cands.length < 6 && !inDraw.has(t)) this.cands.push({ t, p, inDraw: false });
    const faces = this.cands.map((c, i) => new THREE.MeshBasicMaterial({
      map: tokenFaceTexture(c.t, i === st.pick ? 'pink' : c.inDraw ? 'paper' : 'blue', i === st.pick ? 'paper' : c.inDraw ? 'ink' : 'paper', { w: 256, h: 256 }),
    }));
    // BoxGeometry face order +x -x +y -y +z -z
    this.die = new THREE.Mesh(new THREE.BoxGeometry(2.4, 2.4, 2.4), faces);
    outline(this.die, 3);
    this.scene.add(this.die);
    // the orientation that shows face `pick` to the camera (+z)
    const toFront: Record<number, THREE.Euler> = {
      0: new THREE.Euler(0, -Math.PI / 2, 0), 1: new THREE.Euler(0, Math.PI / 2, 0), 2: new THREE.Euler(Math.PI / 2, 0, 0),
      3: new THREE.Euler(-Math.PI / 2, 0, 0), 4: new THREE.Euler(0, 0, 0), 5: new THREE.Euler(0, Math.PI, 0),
    };
    this.dieQ.setFromEuler(toFront[Math.min(st.pick, 5)] ?? toFront[4]!);
    this.tower.outlineAll();
    this.riderQ = riderTile('?');
    this.riderT = riderTile(st.tok.t);
    this.tower.ride(this.riderQ, 1);
    this.scene.add(this.tower.group);
    void inkMat; void tokenTile;
  }

  /** The odds: a stacked bar of the nucleus (what was sampled from), the needle at u. */
  private oddsBar(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, reveal: number, needle: number) {
    const st = this.ctx.llm.steps[this.k]!;
    let acc = 0;
    st.nucleus.forEach(([t, p], i) => {
      const x0 = x + acc * w, pw = p * w * reveal;
      c.fillStyle = i === st.pick && needle >= 1 ? PINK : i % 2 ? BLUE : INK;
      c.fillRect(x0, y, Math.max(2, pw - 3), h);
      if (pw > 90) {
        c.fillStyle = PAPER;
        c.font = font(F.mono(700), Math.min(34, h * 0.4));
        c.textBaseline = 'middle';
        c.fillText(showTok(t).slice(0, 12), x0 + 14, y + h * 0.36);
        c.font = font(F.mono(500), Math.min(24, h * 0.26));
        c.fillText(`${(p * 100).toFixed(1)}%`, x0 + 14, y + h * 0.72);
      }
      acc += p;
    });
    // the 0..1 ruler and the needle at the real uniform draw
    c.fillStyle = INK;
    c.fillRect(x, y + h + 18, w, 3);
    for (let i = 0; i <= 10; i++) c.fillRect(x + (w * i) / 10, y + h + 10, 3, i % 5 ? 10 : 18);
    c.font = font(F.mono(600), 20);
    c.textBaseline = 'top';
    c.fillText('0', x - 6, y + h + 32);
    c.fillText('1', x + w - 6, y + h + 32);
    if (needle > 0) {
      const u = st.u;
      const nx = x + w * u * ease.outExpo(clamp(needle));
      c.fillStyle = PINK;
      c.fillRect(nx - 3, y - 40, 6, h + 60);
      c.beginPath(); c.moveTo(nx - 16, y - 56); c.lineTo(nx + 16, y - 56); c.lineTo(nx, y - 34); c.fill();
      c.fillStyle = INK;
      c.font = font(F.mono(700), 26);
      c.textBaseline = 'bottom';
      c.fillText(`u = ${u.toFixed(3)}`, nx + 12, y - 44);
      // the edge of the picked segment: how close the roll was
      if (st.nucleus.length > 1 && needle >= 1) {
        const edge = st.nucleus.slice(0, st.pick + 1).reduce((a, [, p]) => a + p, 0);
        const ex = x + w * edge;
        c.fillStyle = INK;
        c.fillRect(ex - 2, y - 14, 4, h + 28);
        c.font = font(F.mono(700), 22);
        c.textBaseline = 'top';
        c.fillText(edge.toFixed(3), ex + 8, y + h + 32);
      }
    }
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, au = this.ctx.audio, llm = this.ctx.llm, story = this.ctx.story;
    const sh = shot(this.ctx, f);
    const P = portrait();
    const d = drive(au, f.t, 0.6, 1.8);
    const tm = this.tm;
    const st = llm.steps[this.k]!;
    const line = f.ft < tm.l2 ? 1 : f.ft < tm.l3 ? 2 : f.ft < tm.l4 ? 3 : 4;
    const alt = sh.i % 3;
    const picked = f.ft >= tm.pick;

    // background: wash + rain of the candidates' ids and the context
    this.wash.render(r, out, { seed: sh.seed, drift: d, amt: line === 3 ? 0.75 : 0.5, c1: line % 2 ? 'pink' : 'blue', c2: line % 2 ? 'blue' : 'pink' });
    this.ctx.comp.draw(r, perFrame(this.back, f, () => tokenRain(this.back.ctx, llm.d.prompt, drive(au, f.ft, 0.6, 1.8) * 0.5, { seed: sh.seed, cols: P ? 8 : 13 })), out);

    // 3D: the tower (lines 1 and 4) or the die (lines 2 and 3)
    const showTower = line === 1 || (line === 4 && alt !== 1) || (line === 3 && f.ft >= tm.feed);
    this.tower.group.visible = showTower;
    this.die.visible = !showTower;
    const Ht = this.tower.height;
    const dist = P ? 1.5 : 1;
    if (showTower) {
      let climb = 1; // on the roof: the output head, about to be scored
      if (line === 3) climb = clamp(1 - (f.t - tm.feed) / Math.max(0.3, tm.l4 - tm.feed)); // falls back down the shaft
      if (line === 4) climb = (d * 0.4) % 1; // again, and again
      this.tower.ride(picked ? this.riderT : this.riderQ, climb);
      this.tower.light(climb);
      const a = d * 0.1 + sh.i * 1.7;
      const y = this.tower.floorY(climb * 33);
      const framings = [
        [Math.cos(a) * 11 * dist, y + 2.5, Math.sin(a) * 11 * dist, y],
        [Math.cos(a) * 5 * dist, y + 1.6, Math.sin(a) * 5 * dist, y + 0.3],
        [Math.cos(a) * 30 * dist, Ht * 0.6, Math.sin(a) * 30 * dist, Ht * 0.5],
      ][alt]!;
      this.cam.position.set(framings[0]!, framings[1]!, framings[2]!);
      this.cam.lookAt(0, framings[3]!, 0);
      // the rider shows its face to the lens
      const rd = picked ? this.riderT : this.riderQ;
      rd.rotation.y = Math.atan2(framings[0]!, framings[2]!);
    } else {
      // the die tumbles with the drums and settles on the pick
      const tumble = new THREE.Quaternion().setFromEuler(new THREE.Euler(d * 1.9, d * 1.3 + sh.i, d * 0.7));
      const settle = line === 3 ? ease.outBack(clamp((f.t - tm.pick) / 0.16)) : 0;
      this.die.quaternion.copy(tumble).slerp(this.dieQ, clamp(settle));
      const s = (line === 3 ? 0.85 : 1) * (line === 3 && picked ? 1 + 0.2 * snapIn(f.t, tm.pick) : 1);
      this.die.scale.setScalar(s);
      if (line === 3) this.die.position.set(P ? 0 : 3.4, P ? 1.2 : 0.5, 0);
      else this.die.position.set(P ? 0 : [-3.2, 0, 3.2][alt]!, P ? [1.6, 0.8, 2.2][alt]! : 0.4, 0);
      this.cam.position.set(0, 0.6, (P ? 13 : 9.5) - (alt === 1 && line !== 3 ? 2.5 : 0));
      this.cam.lookAt(0, 0.2, 0);
    }
    this.cam.up.set(0, 1, 0);
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.scene, this.cam);

    // mid layer: the odds, the context strip, the counters (crisp, once per frame)
    this.ctx.comp.draw(r, perFrame(this.mid, f, () => {
      const c = this.mid.ctx;
      const ft = f.ft;
      const bw = P ? W * 0.84 : W * (line === 2 ? 0.5 : 0.52), bh = P ? 110 : 120;
      const bx = P ? (W - bw) / 2 : line === 2 && alt !== 1 ? (alt === 0 ? W * 0.44 : W * 0.06) : line === 3 ? W * 0.06 : (W - bw) / 2;
      const by = P ? H * 0.64 : line === 3 ? H * 0.6 : H * 0.62;
      if (line === 2 || line === 3) {
        const reveal = line === 2 ? ease.outExpo(clamp((ft - tm.l2) / 0.3)) : 1;
        const needle = line === 3 ? clamp((ft - tm.pick + 0.12) / 0.12) : 0;
        this.oddsBar(c, bx, by, bw, bh, reveal, needle);
        c.fillStyle = INK;
        c.font = font(F.mono(700), 22);
        c.textBaseline = 'bottom';
        const nuc = st.nucleus_size;
        c.fillText(nuc === 1 ? `TOP-P 0.9 · ONE TOKEN LEFT IN THE DRAW` : `TOP-P 0.9 · ${nuc} TOKENS IN THE DRAW · T 0.6`, bx, by - 70);
      }
      if (line === 3 && ft >= tm.guess && ft < tm.pick) {
        // GUESS: the top candidates at T=1 as bars
        const top = st.top10_t1.slice(0, 5);
        top.forEach(([t, p], i) => {
          const yy = (P ? H * 0.42 : H * 0.24) + i * 58;
          const x0 = P ? W * 0.08 : W * 0.08;
          const k = snapIn(ft, tm.guess + i * 0.05);
          c.fillStyle = i === 0 ? PINK : BLUE;
          c.fillRect(x0 + 230, yy - 20, (P ? 500 : 560) * p * k, 36);
          c.fillStyle = INK;
          c.font = font(F.mono(700), 30);
          c.textBaseline = 'middle';
          c.fillText(showTok(t).slice(0, 12), x0, yy);
          c.fillText(`${(p * 100).toFixed(1)}%`, x0 + 240 + (P ? 500 : 560) * p * k, yy);
        });
      }
      if (line === 1 || line === 4) {
        // the context so far with an empty slot at the end (line 1), or the step counter (line 4)
        const toks = [...llm.d.question_tokens, ...story.textAt(ft)];
        const size = P ? 30 : 36;
        c.font = font(F.mono(700), size);
        let x = P ? 60 : 140, y = P ? H * 0.2 : H * 0.8;
        for (const t of toks) {
          const lab = showTok(t), w = c.measureText(lab).width + 22;
          if (x + w > W - 60) { x = P ? 60 : 140; y += size * 1.9; }
          c.fillStyle = INK; c.fillRect(x, y - size, w, size * 1.5);
          c.fillStyle = PAPER; c.textBaseline = 'alphabetic'; c.fillText(lab, x + 11, y + size * 0.2);
          x += w + 8;
        }
        if (line === 1) {
          // the empty slot, blinking on the 8ths
          if (Math.floor(au.beatAt(ft) * 2) % 2 === 0) { c.strokeStyle = PINK; c.lineWidth = 6; c.setLineDash([12, 8]); c.strokeRect(x, y - size, 120, size * 1.5); c.setLineDash([]); }
        } else {
          const n = story.at(ft).emitted;
          c.fillStyle = PINK;
          c.font = font(F.archivo(125, 900), P ? 150 : 190);
          c.textBaseline = 'middle';
          c.textAlign = 'center';
          c.fillText(`${n} / ${llm.steps.length}`, W / 2, P ? H * 0.5 : H * 0.42);
          c.textAlign = 'left';
          c.fillStyle = INK;
          c.font = font(F.mono(700), 26);
          c.fillText('TOKENS WRITTEN', W / 2 - 110, (P ? H * 0.5 : H * 0.42) + (P ? 110 : 130));
        }
      }
      if (line === 3 && picked) {
        // PICK: the chosen token, huge, then FEED IT BACK
        const k = snapIn(ft, tm.pick);
        const lab = showTok(st.tok.t);
        c.save();
        c.translate(P ? W / 2 : W * 0.31, P ? H * 0.45 : H * 0.33);
        c.scale(1.3 - 0.3 * k, 1.3 - 0.3 * k);
        c.font = font(F.mono(700), P ? 150 : 200);
        c.textAlign = 'center';
        c.textBaseline = 'middle';
        const w = c.measureText(lab).width + 80;
        c.fillStyle = PINK;
        c.fillRect(-w / 2, -120, w, 240);
        c.fillStyle = PAPER;
        c.fillText(lab, 0, 8);
        c.restore();
      }
    }), out);

    // front: specks, faster on the last line
    const c = this.front.ctx;
    this.front.clear();
    specks(c, d * (line === 4 ? 1.8 : 1), { seed: sh.seed, n: line === 4 ? 110 : 70 });
    this.ctx.comp.draw(r, this.front.upload(), out);

    return { reg: sh.seed, kinetic: line === 3 && picked && f.ft < tm.feed ? 0 : 1 };
  }
}
