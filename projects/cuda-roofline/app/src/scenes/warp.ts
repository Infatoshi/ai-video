// A warp is 32 threads running the same instruction together; an `if` makes half of them wait.
//
// warp (v5, CH3), 50.10-64.54 s, inside one SM of the shared H100 (_h100.ts smDetail). The 32 FP32
// lanes of partition 0 are the warp: exactly 32, one per thread.
//  L1 "Thirty-two threads in a warp, in line,": the camera dives from above the SM into partition 0;
//     the lanes light 16 on "Thir-ty" and 16 on "two"; "1 warp = 32 threads" boxes them on "warp";
//     on "line" the 2 × 16 lanes rise off the silicon and snap into one straight line of 32 pipes.
//  L2 "One instruction, marching in time,": the partition's warp scheduler and dispatch light on
//     "instruction"; on the three beats from "marching" one instruction token drops down all 32 pipes
//     at once and every lane fires together (load a, load b, c = a + b). The camera dollies the row.
//  L3 "One little “if” and the warp splits in two,": an acid `if` gate drops across the pipes, the
//     row splits 16 | 16 on "splits", tagged IF / ELSE on "two". The camera cranes up over the split.
//  L4 "Half of them wait while the others go through": the IF half runs while the ELSE half goes dark
//     and hatched ("waiting"), then they swap on "others", and on the beat after "through" the halves
//     close up and all 32 fire together again (store c). The camera pulls back out of the SM.
// Always moving: the orbit breathes with the drums (_cam.ts drive), faint operand tokens keep
// streaming down the active pipes, the die flickers with the music; accents snap on the grid (_lock).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H, makeRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font, layout, type TextLayout } from '../engine/type';
import { Lyrics, type Line } from '../engine/lyrics';
import { clamp, ease, lerp } from '../engine/util';
import { H100, SMDetail, glowMat } from './_h100';
import { snap, hit, q } from './_lock';
import { drive, kickPush } from './_cam';

const N = 32, HALF = 16;           // threads in a warp (CH3); the if splits it 16 | 16
const SM = 58;                     // a mid-die SM site
const PITCH = 0.034;               // mm between pipes once in line
const LIFT = 0.15;                 // mm the line floats above the partition
const PIPE = { w: 0.022, h: 0.03, len: 0.34 };
const GAP = 0.16;                  // mm between the halves once split
const LYR = { x: 120, y: 986, size: 72 };

type RowKind = 'all' | 'if' | 'else';
interface Row { kind: RowKind; label: string; at: number }
interface Key { t: number; dist: number; el: number; az: number; tx: number; tz: number }

const lay = (() => {
  const cache = new Map<string, TextLayout>();
  return (text: string, fam: string, size: number) => {
    const k = `${text}|${fam}|${size}`;
    let l = cache.get(k);
    if (!l) { l = layout(text, fam, size); cache.set(k, l); }
    return l;
  };
})();

function hatchTexture(): THREE.CanvasTexture {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 128;
  const c = cv.getContext('2d')!;
  c.clearRect(0, 0, 128, 128);
  c.strokeStyle = 'rgba(238,233,223,0.9)';
  c.lineWidth = 7;
  for (let i = -128; i < 256; i += 32) { c.beginPath(); c.moveTo(i, 0); c.lineTo(i + 128, 128); c.stroke(); }
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

export default class Warp extends Scene {
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(34, W / H, 0.01, 500);
  gpu!: H100;
  det!: SMDetail;
  ms = makeRT(W, H, { samples: 4 });
  ui = new Layer2D();
  famL = F.archivo(100, 800);
  famM = F.mono(500);

  // the warp rig (SM-local, parented to the SM close-up)
  rig = new THREE.Group();
  pipes!: THREE.InstancedMesh;
  pipeGlow!: THREE.InstancedMesh;
  tokens!: THREE.InstancedMesh;
  gate!: THREE.Mesh;
  gateEdge!: THREE.LineSegments;
  hatch: THREE.Mesh[] = [];
  box!: THREE.LineSegments;
  tmp = new THREE.Object3D();
  col = new THREE.Color();

  home: THREE.Vector3[] = [];      // each thread's lane on the silicon (SM-local)
  cx = 0; cz = 0;                  // partition 0 lane-region centre

  lines: Line[] = [];
  rows: Row[] = [];
  keys: Key[] = [];
  tIn1 = 0; tIn2 = 0; tNums = 0; tBox = 0; tLine = 0; tPC = 0;
  tIf = 0; tSplit = 0; tTags = 0; tHalf = 0; tWait = 0; tSwap = 0; tJoin = 0;
  /** acid owns the "if" and the split for about 2 s, then the running half goes back to signal */
  tAcid = 0;

  override async init() {
    const ly = this.ctx.lyrics, au = this.ctx.audio;
    this.gpu = new H100({ renderer: this.ctx.renderer });
    // lights: H100.rig()'s key, a weaker ember rim (the shared rig's rim hazes this close-up), dim fill
    const key = new THREE.DirectionalLight(new THREE.Color().setRGB(0.92, 0.95, 1.0), 1.35);
    key.position.set(-60, 120, 80);
    const rim = new THREE.DirectionalLight(new THREE.Color().setRGB(LIN.ember[0], LIN.ember[1], LIN.ember[2]), 0.3);
    rim.position.set(90, 40, -140);
    const fill = new THREE.HemisphereLight(new THREE.Color().setRGB(0.1, 0.1, 0.11), new THREE.Color().setRGB(0.02, 0.018, 0.016), 0.3);
    this.scene.add(this.gpu.group, key, rim, fill);
    this.scene.environment = this.gpu.env;
    // the camera grazes the die here, so the env reflection would haze the frame at full strength
    this.scene.environmentIntensity = 0.35;
    this.scene.background = new THREE.Color().setRGB(LIN.ink[0], LIN.ink[1], LIN.ink[2], THREE.LinearSRGBColorSpace);
    this.det = this.gpu.smDetail(SM);
    this.det.group.add(this.rig);

    // homes: the 32 FP32 lanes of partition 0 (2 rows of 16)
    for (let i = 0; i < N; i++) this.home.push(this.det.lanePos(0, 'fp32', i));
    const xs = this.home.map((v) => v.x), zs = this.home.map((v) => v.z);
    this.cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    this.cz = (Math.min(...zs) + Math.max(...zs)) / 2;

    const body = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(LIN.graphite[0] * 0.35, LIN.graphite[1] * 0.35, LIN.graphite[2] * 0.35), metalness: 0.5, roughness: 0.4, envMapIntensity: 0.4 });
    const geo = new THREE.BoxGeometry(1, 1, 1);
    this.pipes = new THREE.InstancedMesh(geo, body, N);
    this.pipeGlow = new THREE.InstancedMesh(geo, glowMat(), N);
    this.tokens = new THREE.InstancedMesh(geo, glowMat(), N * 8);
    for (let i = 0; i < N; i++) this.pipeGlow.setColorAt(i, this.col.setRGB(0, 0, 0));
    for (let i = 0; i < N * 8; i++) this.tokens.setColorAt(i, this.col.setRGB(0, 0, 0));
    this.pipes.frustumCulled = this.pipeGlow.frustumCulled = this.tokens.frustumCulled = false;
    this.rig.add(this.pipes, this.pipeGlow, this.tokens);

    // the if gate: a thin acid slab across the north end of the pipes
    this.gate = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), glowMat());
    this.gateEdge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: new THREE.Color().setRGB(LIN.acid[0], LIN.acid[1], LIN.acid[2]), transparent: true, depthWrite: false }));
    this.rig.add(this.gate, this.gateEdge);

    // hatched "waiting" covers, one per half
    const htex = hatchTexture();
    for (let s = 0; s < 2; s++) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ map: htex, transparent: true, opacity: 0, depthWrite: false }));
      this.hatch.push(m);
      this.rig.add(m);
    }
    // the "1 warp" wire box
    this.box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: new THREE.Color().setRGB(LIN.bone[0], LIN.bone[1], LIN.bone[2]), transparent: true, depthWrite: false }));
    this.rig.add(this.box);

    // ---- times
    const l1 = ly.get('Thirty-two threads in a warp'), l2 = ly.get('One instruction'),
      l3 = ly.get('warp splits in two'), l4 = ly.get('Half of them wait');
    this.lines = [l1, l2, l3, l4];
    const w = (l: Line, s: string) => l.words.find((x) => x.w.toLowerCase().replace(/[^a-z-]/g, '').startsWith(s))!;
    const tw = w(l1, 'thirty');
    this.tIn1 = tw.start;
    this.tIn2 = tw.syl?.[1]?.[0] ?? tw.start + 0.3;
    this.tNums = w(l1, 'threads').start;
    this.tBox = w(l1, 'warp').start;
    this.tLine = l1.words[l1.words.length - 1]!.start;
    this.tPC = w(l2, 'instruction').start;
    const b0 = Math.round(au.beatAt(w(l2, 'marching').start));
    const beat = (k: number) => au.timeOfBeat(b0 + k);
    this.tIf = w(l3, 'if').start;
    this.tAcid = au.timeOfBeat(Math.round(au.beatAt(this.tIf + 2.0)));
    this.tSplit = w(l3, 'splits').start;
    this.tTags = l3.words[l3.words.length - 1]!.start;
    this.tHalf = w(l4, 'half').start;
    this.tWait = w(l4, 'wait').start;
    this.tSwap = w(l4, 'others').start;
    const through = l4.words[l4.words.length - 1]!;
    this.tJoin = au.timeOfBeat(Math.ceil(au.beatAt(through.start) + 0.25));
    this.rows = [
      { kind: 'all', label: 'load a', at: beat(0) },
      { kind: 'all', label: 'load b', at: beat(1) },
      { kind: 'all', label: 'c = a + b', at: beat(2) },
      { kind: 'if', label: 'if (thread < 16): c = c * 2', at: this.tHalf },
      { kind: 'else', label: 'else: c = 0', at: this.tSwap },
      { kind: 'all', label: 'store c', at: this.tJoin },
    ];

    // ---- camera keys (SM-local target; spherical offset), big moves land on downbeats
    const D = au.downbeats.filter((d) => d > this.ctx.start - 0.5 && d < this.ctx.end + 2);
    const at = (x: number) => D.reduce((b, d) => (Math.abs(d - x) < Math.abs(b - x) ? d : b), D[0]!);
    const xl = this.cx - (N / 2) * PITCH, xr = this.cx + (N / 2) * PITCH;
    this.keys = [
      { t: this.ctx.start, dist: 3.4, el: 1.12, az: 0.35, tx: this.cx, tz: this.cz },
      { t: at(52.36), dist: 1.05, el: 0.9, az: 0.1, tx: this.cx, tz: this.cz },
      { t: at(54.16), dist: 0.95, el: 0.42, az: -0.55, tx: this.cx, tz: this.cz },
      { t: at(55.97), dist: 0.62, el: 0.3, az: -1.05, tx: xl + 0.1, tz: this.cz },
      { t: at(57.77), dist: 0.62, el: 0.34, az: -0.8, tx: xr - 0.1, tz: this.cz },
      { t: at(59.58), dist: 1.55, el: 1.12, az: -0.2, tx: this.cx, tz: this.cz },
      { t: at(63.18), dist: 1.2, el: 0.72, az: 0.55, tx: this.cx, tz: this.cz },
      { t: this.ctx.end + 0.2, dist: 3.8, el: 1.05, az: 0.8, tx: this.cx, tz: this.cz - 0.2 },
    ];
  }

  /** Interpolated camera parameters (eased between keys; keys land on downbeats). */
  camAt(t: number): Key {
    const k = this.keys;
    if (t <= k[0]!.t) return k[0]!;
    for (let i = 0; i < k.length - 1; i++) {
      const a = k[i]!, b = k[i + 1]!;
      if (t < b.t) {
        const u = ease.inOutCubic(clamp((t - a.t) / (b.t - a.t)));
        return { t, dist: lerp(a.dist, b.dist, u), el: lerp(a.el, b.el, u), az: lerp(a.az, b.az, u), tx: lerp(a.tx, b.tx, u), tz: lerp(a.tz, b.tz, u) };
      }
    }
    return k[k.length - 1]!;
  }

  /** 0..1 how far the lanes have risen into the line (a one-beat move landing on "line"). */
  lineK(t: number) {
    const au = this.ctx.audio;
    const t1 = q(au, this.tLine), t0 = t1 - 60 / au.bpm * 0.75;
    return ease.inOutCubic(clamp((t - t0) / (t1 - t0)));
  }
  splitK(t: number) {
    const au = this.ctx.audio;
    return snap(au, t, this.tSplit, 0.12) * (1 - snap(au, t, this.tJoin, 0.12));
  }
  /** SM-local centre of pipe i at t (home lane → line → split). */
  pipePos(i: number, t: number): THREE.Vector3 {
    const m = this.lineK(t);
    const side = i < HALF ? -1 : 1;
    const line = new THREE.Vector3(this.cx + (i - (N - 1) / 2) * PITCH + side * (GAP / 2) * this.splitK(t), this.home[i]!.y + LIFT + PIPE.h / 2, this.cz);
    return this.home[i]!.clone().lerp(line, m);
  }

  /** Which pipes are running at t: all, the IF half, or the ELSE half. */
  runs(i: number, t: number): number {
    if (t < this.tHalf || t >= this.tJoin - 0.05) return 1;
    const inIf = i < HALF;
    return t < this.tSwap ? (inIf ? 1 : 0) : (inIf ? 0 : 1);
  }

  render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp, audio: au } = this.ctx;
    const t = f.t;
    const g = this.gpu, d = this.det;
    g.update(t, f.a);
    // the rest of the die stays alive but dim, so the warp is the hero
    g.setSMActivity((i) => (i === SM ? 0.02 : 0.05 + 0.12 * f.a.drums * ((i * 7) % 5 === 0 ? 1 : 0.4)));
    g.smDetail(SM);
    const m = this.lineK(t);

    // ---- the silicon lanes of partition 0 light as the warp assembles (then fade as they lift away)
    const lit = (i: number) => (i < HALF ? snap(au, t, this.tIn1) : snap(au, t, this.tIn2));
    d.setLanes((p, kind, idx) => {
      if (p !== 0 || kind !== 'fp32') return 0.05 + 0.1 * f.a.drums;
      return lit(idx) * (1 - m) * (0.8 + 0.6 * hit(au, t, idx < HALF ? this.tIn1 : this.tIn2, 0.1));
    });
    const pc = snap(au, t, this.tPC);
    d.setUnits((u, p) => (p === 0 && (u === 'sched' || u === 'dispatch') ? pc * (0.15 + 0.35 * this.rowHit(t)) : u === 'regfile' && p === 0 ? 0.08 + 0.15 * f.a.drums : 0));
    d.setTensor(() => 0.04 * f.a.drums);

    // ---- pipes
    const acidOn = snap(au, t, this.tIf) * (1 - snap(au, t, this.tAcid));
    for (let i = 0; i < N; i++) {
      const p = this.pipePos(i, t);
      const len = lerp(0.03, PIPE.len, m), wdt = lerp(0.028, PIPE.w, m), hgt = lerp(0.05, PIPE.h, m);
      this.tmp.position.copy(p);
      this.tmp.scale.set(wdt, hgt, len);
      this.tmp.updateMatrix();
      this.pipes.setMatrixAt(i, this.tmp.matrix);
      this.tmp.scale.set(wdt * 1.08, hgt * 1.08, len * 1.01);
      this.tmp.updateMatrix();
      this.pipeGlow.setMatrixAt(i, this.tmp.matrix);
      const run = this.runs(i, t);
      const fire = this.pipeHit(i, t);
      const base = lit(i) * (m > 0 ? 0.08 + 0.1 * f.a.drums : 0);
      const a = run * (base + 0.65 * fire);
      const c = acidOn > 0.5 && i < HALF ? LIN.acid : LIN.signal;
      this.pipeGlow.setColorAt(i, this.col.setRGB(c[0] * a, c[1] * a, c[2] * a));
    }
    this.pipes.instanceMatrix.needsUpdate = true;
    this.pipeGlow.instanceMatrix.needsUpdate = true;
    this.pipeGlow.instanceColor!.needsUpdate = true;
    this.pipes.visible = this.pipeGlow.visible = snap(au, t, this.tIn1) > 0;

    // ---- tokens: one instruction per row drops down every running pipe and lands on its beat; in
    // between, faint operand tokens keep streaming down the running pipes
    let n = 0;
    const travel = 60 / au.bpm * 0.5;
    const put = (i: number, u: number, gain: number, len = 0.03) => {
      if (n >= N * 8) return;
      const p = this.pipePos(i, t);
      this.tmp.position.set(p.x, p.y + PIPE.h * 0.62, p.z - PIPE.len / 2 + u * PIPE.len);
      this.tmp.scale.set(PIPE.w * 0.9, PIPE.h * 0.35, len);
      this.tmp.updateMatrix();
      this.tokens.setMatrixAt(n, this.tmp.matrix);
      const c = acidOn > 0.5 && i < HALF ? LIN.acid : LIN.ember;
      this.tokens.setColorAt(n, this.col.setRGB(c[0] * gain, c[1] * gain, c[2] * gain));
      n++;
    };
    if (m > 0.98) {
      for (const r of this.rows) {
        const th = q(au, r.at);
        const u = (t - (th - travel)) / travel;
        if (u < 0 || u > 1) continue;
        for (let i = 0; i < N; i++) if (this.rowRuns(r.kind, i)) put(i, ease.inQuad(u), 1.6, 0.05);
      }
      // the stream: 3 faint tokens per running pipe, a pure function of t
      const rate = 1.6 + 1.2 * f.a.drums;
      for (let i = 0; i < N; i++) {
        if (!this.runs(i, t)) continue;
        for (let s = 0; s < 3; s++) {
          const u = ((drive(au, t, rate, 1.5) * 0.35 + s / 3 + (i % 4) * 0.07) % 1);
          put(i, u, 0.35 + 0.35 * f.a.drums, 0.018);
        }
      }
    }
    this.tokens.count = n;
    this.tokens.instanceMatrix.needsUpdate = true;
    if (this.tokens.instanceColor) this.tokens.instanceColor.needsUpdate = true;

    // ---- the if gate (drops in on "if", lifts at the join)
    const gIn = snap(au, t, this.tIf, 0.12), gOut = snap(au, t, this.tJoin, 0.12);
    const gy = lerp(0.35, 0, gIn) + 0.35 * gOut;
    const gA = gIn * (1 - gOut);
    const gw = N * PITCH + GAP * this.splitK(t) + 0.06;
    const gz = this.cz - PIPE.len / 2 - 0.03;
    this.gate.position.set(this.cx, this.home[0]!.y + LIFT + PIPE.h / 2 + gy, gz);
    this.gate.scale.set(gw, 0.07, 0.006);
    this.gateEdge.position.copy(this.gate.position);
    this.gateEdge.scale.copy(this.gate.scale);
    const ga = gA * (0.35 + 0.8 * hit(au, t, this.tIf, 0.12));
    // acid while it owns the split, then a dim bone bar until the join
    const gc = (j: number) => lerp(LIN.bone[j]! * 0.35, LIN.acid[j]!, acidOn) * ga;
    (this.gate.material as THREE.MeshBasicMaterial).color.setRGB(gc(0), gc(1), gc(2));
    (this.gateEdge.material as THREE.LineBasicMaterial).opacity = gA;
    this.gate.visible = this.gateEdge.visible = gA > 0.01;

    // ---- hatched covers on the waiting half
    for (let s = 0; s < 2; s++) {
      const h = this.hatch[s]!;
      const waiting = t >= q(au, this.tHalf) && t < q(au, this.tJoin) && ((s === 1 && t < q(au, this.tSwap)) || (s === 0 && t >= q(au, this.tSwap)));
      const k = waiting ? 1 : 0;
      const c0 = this.pipePos(s === 0 ? 0 : HALF, t), c1 = this.pipePos(s === 0 ? HALF - 1 : N - 1, t);
      h.position.set((c0.x + c1.x) / 2, c0.y + 0.004, c0.z);
      h.scale.set(Math.abs(c1.x - c0.x) + PIPE.w * 1.6, PIPE.h * 1.25, PIPE.len * 1.04);
      (h.material as THREE.MeshBasicMaterial).opacity = 0.55 * k;
      (h.material as THREE.MeshBasicMaterial).map!.repeat.set(4, 1);
      h.visible = k > 0;
    }

    // ---- the warp box (on "warp"), wraps the line and opens with the split
    const bA = snap(au, t, this.tBox);
    const b0 = this.pipePos(0, t), b1 = this.pipePos(N - 1, t);
    this.box.position.set((b0.x + b1.x) / 2, (b0.y + b1.y) / 2, (b0.z + b1.z) / 2);
    this.box.scale.set(Math.abs(b1.x - b0.x) + 0.07, lerp(0.09, PIPE.h + 0.05, m), lerp(0.2, PIPE.len + 0.06, m));
    (this.box.material as THREE.LineBasicMaterial).opacity = 0.75 * bA * (1 - 0.6 * this.splitK(t));
    this.box.visible = bA > 0.01;

    // ---- camera: keyed moves that land on downbeats, an orbit that breathes with the drums, and a
    // small push on kicks
    const k = this.camAt(t);
    const az = k.az + 0.045 * drive(au, t, 0.6, 1.4, this.ctx.start);
    const dist = k.dist * (1 - 0.035 * kickPush(au, t));
    const tgtL = new THREE.Vector3(k.tx, this.home[0]!.y + LIFT * m * 0.8, k.tz);
    const tgt = d.group.localToWorld(tgtL.clone());
    const cam = this.cam;
    cam.position.set(tgt.x + Math.sin(az) * Math.cos(k.el) * dist, tgt.y + Math.sin(k.el) * dist, tgt.z + Math.cos(az) * Math.cos(k.el) * dist);
    cam.up.set(0, 1, 0);
    cam.lookAt(tgt);
    H100.fitClip(cam, dist);

    renderer.setRenderTarget(this.ms);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    renderer.render(this.scene, cam);
    comp.draw(renderer, this.ms.texture, out);

    // ---- screen-space: labels, then the karaoke
    const c = this.ui.ctx;
    this.ui.clear();
    this.drawLabels(c, t);
    this.drawLyric(c, t);
    comp.draw(renderer, this.ui.upload(), out);
    return { bloom: 0.42, bloomThreshold: 0.92 };
  }

  rowRuns(kind: RowKind, i: number) { return kind === 'all' || (kind === 'if' ? i < HALF : i >= HALF); }

  /** Fire envelope of pipe i: rows landing on their beats. */
  pipeHit(i: number, t: number) {
    const au = this.ctx.audio;
    let v = 0;
    for (const r of this.rows) if (this.rowRuns(r.kind, i)) v = Math.max(v, hit(au, t, r.at, 0.13));
    return v;
  }
  rowHit(t: number) {
    const au = this.ctx.audio;
    let v = 0;
    for (const r of this.rows) v = Math.max(v, hit(au, t, r.at, 0.13));
    return v;
  }

  /** Project an SM-local point to screen px (null behind the camera). */
  screen(pL: THREE.Vector3): [number, number] | null {
    const v = this.det.group.localToWorld(pL.clone()).project(this.cam);
    if (v.z > 1) return null;
    return [(v.x * 0.5 + 0.5) * W, (-v.y * 0.5 + 0.5) * H];
  }

  drawLabels(c: CanvasRenderingContext2D, t: number) {
    const au = this.ctx.audio;
    const label = (pL: THREE.Vector3, text: string, a: number, opts: { size?: number; col?: string; dx?: number; dy?: number; align?: CanvasTextAlign } = {}) => {
      if (a <= 0.01) return;
      const s = this.screen(pL);
      if (!s) return;
      const [x, y] = s;
      if (x < 80 || x > W - 80 || y < 70 || y > H - 170) return;
      c.save();
      c.globalAlpha = a;
      c.font = font(this.famM, opts.size ?? 26);
      c.textAlign = opts.align ?? 'left';
      c.textBaseline = 'alphabetic';
      c.fillStyle = 'rgba(10,10,11,0.72)';
      const tw = c.measureText(text).width;
      const bx = (opts.align === 'center' ? x - tw / 2 : x) + (opts.dx ?? 0) - 8;
      c.fillRect(bx, y + (opts.dy ?? 0) - (opts.size ?? 26) - 4, tw + 16, (opts.size ?? 26) + 14);
      c.fillStyle = opts.col ?? rgba('bone', 1);
      c.fillText(text, x + (opts.dx ?? 0), y + (opts.dy ?? 0));
      c.restore();
    };
    const m = this.lineK(t);
    const mid = this.pipePos(HALF, t), p0 = this.pipePos(0, t), pN = this.pipePos(N - 1, t);
    const above = (p: THREE.Vector3, dy = 0.07) => p.clone().add(new THREE.Vector3(0, dy, 0));
    // L1: the count, then the warp label
    const cA = snap(au, t, this.tNums) * (1 - snap(au, t, this.tBox));
    label(above(mid, 0.09), '32 threads', cA, { size: 40, align: 'center' });
    const wA = snap(au, t, this.tBox) * (1 - snap(au, t, this.tIf));
    // sits above the row label once the rows start
    label(above(mid, lerp(0.09, 0.2, m)), '1 warp = 32 threads', wA, { size: 38, align: 'center' });
    // L2: the instruction on each row, above the line
    for (let r = 0; r < this.rows.length; r++) {
      const row = this.rows[r]!, next = this.rows[r + 1];
      const a = snap(au, t, row.at) * (next ? 1 - snap(au, t, next.at) : 1) * (1 - snap(au, t, this.ctx.end - 0.4));
      if (a <= 0) continue;
      const at = row.kind === 'if' ? this.pipePos(HALF / 2, t) : row.kind === 'else' ? this.pipePos(HALF + HALF / 2, t) : mid;
      const col = row.kind === 'if' && t < this.tAcid ? `rgba(216,255,60,1)` : rgba('signal', 1);
      label(above(at, 0.1), row.label, a, { size: 40, col, align: 'center' });
    }
    // lane ends
    const eA = snap(au, t, this.tNums) * m * (1 - snap(au, t, this.tIf));
    label(above(p0, 0.035), '0', eA, { size: 22, align: 'center' });
    label(above(pN, 0.035), '31', eA, { size: 22, align: 'center' });
    // L3: the gate and the halves
    const gA = snap(au, t, this.tIf) * (1 - snap(au, t, this.tHalf));
    label(new THREE.Vector3(this.cx, this.home[0]!.y + LIFT + 0.12, this.cz - PIPE.len / 2), 'if (thread < 16)', gA, { size: 40, col: t < this.tAcid ? 'rgba(216,255,60,1)' : rgba('bone', 1), align: 'center' });
    const tagA = snap(au, t, this.tTags) * (1 - snap(au, t, this.tJoin));
    label(above(this.pipePos(HALF / 2, t), -0.02).add(new THREE.Vector3(0, 0, PIPE.len / 2 + 0.05)), 'IF', tagA, { size: 40, align: 'center' });
    label(above(this.pipePos(HALF + HALF / 2, t), -0.02).add(new THREE.Vector3(0, 0, PIPE.len / 2 + 0.05)), 'ELSE', tagA, { size: 40, align: 'center' });
    // L4: waiting…
    const wt = snap(au, t, this.tWait) * (1 - snap(au, t, this.tJoin));
    if (wt > 0) {
      const waitingSide = t < q(au, this.tSwap) ? HALF + HALF / 2 : HALF / 2;
      label(above(this.pipePos(waitingSide, t), 0.05).add(new THREE.Vector3(0, 0, 0.05)), 'waiting…', wt, { size: 36, col: rgba('ash', 1), align: 'center' });
    }
    const jA = snap(au, t, this.tJoin) * (1 - snap(au, t, this.ctx.end - 0.4));
    label(above(mid, 0.2), 'back in step', jA, { size: 40, align: 'center' });
    // chapter tag, small, top left (clear of the karaoke band)
    c.save();
    c.globalAlpha = 1 - snap(au, t, this.tIf);
    c.font = font(F.mono(400), 18);
    c.fillStyle = rgba('ash', 0.9);
    c.textAlign = 'left';
    c.fillText('CH3 · a warp is 32 threads · H100 SM, partition 0, 32 FP32 lanes', 96, 60);
    c.restore();
  }

  drawLyric(c: CanvasRenderingContext2D, t: number) {
    let line = this.lines[0]!;
    for (const l of this.lines) if (l.start <= t + 0.35) line = l;
    const fam = this.famL;
    const size = LYR.size;
    const text = line.words.map((w) => w.w).join(' ');
    const L = lay(text, fam, size);
    c.save();
    // a soft dark band under the line so it reads over the 3D
    const grad = c.createLinearGradient(0, LYR.y - size * 1.6, 0, H);
    grad.addColorStop(0, 'rgba(10,10,11,0)');
    grad.addColorStop(0.45, 'rgba(10,10,11,0.62)');
    grad.addColorStop(1, 'rgba(10,10,11,0.8)');
    c.fillStyle = grad;
    c.fillRect(0, LYR.y - size * 1.6, W, H - (LYR.y - size * 1.6));
    c.font = font(fam, size);
    c.textBaseline = 'alphabetic';
    c.textAlign = 'left';
    c.fillStyle = rgba('bone', 0.28);
    c.fillText(text, LYR.x, LYR.y);
    let idx = 0;
    for (const w of line.words) {
      const p = Lyrics.wordProgress(w, t);
      const gx = LYR.x + (L.glyphs[idx]?.x ?? 0);
      const wx = lay(w.w, fam, size).width;
      if (p > 0) {
        c.save();
        c.beginPath();
        c.rect(gx - 4, LYR.y - size * 1.05, wx * p + 4, size * 1.4);
        c.clip();
        c.fillStyle = p >= 1 ? rgba('bone', 1) : rgba('signal', 1);
        c.fillText(text, LYR.x, LYR.y);
        c.restore();
      }
      idx += w.w.length + 1;
    }
    c.restore();
  }
}
