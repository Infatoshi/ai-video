// flight: the whole episode is one continuous camera flight through Llama 3.1 8B Instruct's input embedding
// table (data/emb.json, llm/embed.py). No cuts: the table wall and the rain row (verse 1), the rain row
// collapsing into a point that flies into the 3D squash of the space (chorus), neighbourhoods (verse 2),
// the man -> woman arrow carried to boy, father and king (verse 3), everything together (last chorus).
// Colours: pink = the thing being taught while it is sung; blue = the data (points, rows); ink = structure, type.
// Every number is from emb.json; one number on screen at a time, held until the lyric moves on.
import * as THREE from 'three';
import { Scene, type Frame } from '../engine/scene';
import { W, H, SCALE } from '../engine/gl';
import { OVERLAYS } from '../engine/hud';
import { rgba, LIN } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp } from '../engine/util';
import { Wash, inkMat, outline, portrait } from './_kit';
import { loadEmb, toWorld, tableValues, fmtCos, tokLabel, Spline, R, type Emb } from './flight-data';

const PINK = rgba('pink'), BLUE = rgba('blue'), INK = rgba('ink'), PAPER = rgba('paper');
const V3 = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const ANCHORS = new Set([' rain', ' three', ' man', ' woman', ' king', ' queen', ' red', ' ocean', ' cat', ' coffee', ' snow', ' boy', ' girl', ' King']);
const sm = (x: number) => { const u = clamp(x); return 0.5 - 0.5 * Math.cos(Math.PI * u); };

// the table wall (verse 1): the 48 real rows around " rain", 4,096 values across
const WALL_Z = 30, WALL_W = 40, ROW_H = 0.42;

interface NumCue { t0: number; t1: number; text: string; cap: string; at: () => THREE.Vector3 | null }
interface Thread { a: string; b: string; t: number; cur1: number; cos: number }
interface Arrow { tail: (t: number) => THREE.Vector3; vec: THREE.Vector3; t0: number; grow: number; cur0: number; cur1: number; label?: string }

export default class Flight extends Scene {
  emb!: Emb;
  s3 = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(36, W / H, 0.05, 400);
  camF = new THREE.PerspectiveCamera(36, W / H, 0.05, 400); // the overlay's camera, at the frame's own time
  wash = new Wash();
  path!: Spline;
  pos = new Map<string, THREE.Vector3>(); // featured token -> world position (rain's is animated in verse 1)
  sph = new Map<string, { mesh: THREE.Mesh; mat: THREE.ShaderMaterial }>();
  hazeMat!: THREE.ShaderMaterial;
  wallMat!: THREE.ShaderMaterial;
  barMat!: THREE.ShaderMaterial;
  T: Record<string, number> = {};
  labelAt = new Map<string, number>(); // featured token -> time its label draws in
  pinks: { tok: string; t0: number; t1: number }[] = [];
  threads: Thread[] = [];
  arrows: Arrow[] = [];
  nums: NumCue[] = [];
  land: Record<string, THREE.Vector3> = {};
  wallCenter = V3(0, 0, WALL_Z);

  override async init() {
    const e = (this.emb = await loadEmb());
    for (const f of e.space.featured) this.pos.set(f.t, toWorld(f.p));
    for (const [k, p] of Object.entries(e.space.land)) this.land[k] = toWorld(p);
    this.times();
    this.build();
    this.plan();
    // labels, numbers and arrows print with the HUD: crisp solid ink, never halftoned
    OVERLAYS.length = 0;
    OVERLAYS.push((c, t) => this.overlay(c, t));
  }

  // ------------------------------------------------------------------ lyric anchors
  private times() {
    const ly = this.ctx.lyrics, au = this.ctx.audio;
    const L = (q: string, n = 0) => ly.get(q, n);
    const w = (q: string, word: string, n = 0) => {
      const l = L(q, n);
      const x = l.words.find((y) => y.w.toLowerCase().replace(/[^a-z0-9]/g, '').startsWith(word));
      if (!x) throw new Error(`word ${word} not in "${l.text}"`);
      return x.start;
    };
    const T = this.T;
    T.end = au.duration;
    T.v1 = L('Before a chatbot').start; T.v1a = L('It looks it up in a table').start; T.v1b = L('One row for every word').start;
    T.v1c = L('Find rain').start; T.v1d = L('Four thousand').start; T.v1e = L('That list is the word').start;
    T.v1g = L("It doesn't know what rain").start; T.v1f = (T.v1e + T.v1g) / 2;
    T.v1h = L('Only where it is').start; T.v1hEnd = L('Only where it is').end;
    for (const n of [0, 1, 2]) {
      T[`c${n}`] = L('Every word is a list of numbers', n).start;
      T[`c${n}b`] = L('Every list is a place', n).start;
      T[`c${n}c`] = L('Words that mean alike', n).start;
      T[`c${n}d`] = L('Meaning is a direction', n).start;
      T[`c${n}End`] = L('Meaning is a direction', n).end;
    }
    T.v2 = L('Close means they point').start; T.v2b = L('One means the same').start; T.v2zero = w('One means the same', 'zero'); T.v2same = w('One means the same', 'same');
    T.v2c = L('Next to three').start;
    T.v2four = w('Next to three', 'four'); T.v2two = w('Next to three', 'two'); T.v2five = w('Next to three', 'five');
    T.v2d = L('Nobody put them there').start; T.v2e = L('Next to rain is Rain').start;
    T.v2f = L('Then rains, then snow').start; T.v2rains = w('Then rains', 'rains'); T.v2snow = w('Then rains', 'snow');
    T.v2wind = w('Then rains', 'wind'); T.v2storm = w('Then rains', 'storm');
    T.v2g = L('Its own spelling comes first').start; T.v2h = L('Then words that mean the same').start;
    T.v2hEnd = L('Then words that mean the same').end;
    // "Next to rain is Rain": the second "rain" is ·Rain
    const lr = L('Next to rain is Rain');
    const rains = lr.words.filter((y) => y.w.toLowerCase().startsWith('rain'));
    T.v2Rain = (rains[1] ?? rains[0]!).start;
    T.v3 = L('Draw an arrow from man').start; T.v3b = L('Start it at boy').start; T.v3c = w('Start it at boy', 'you');
    T.v3girl = w('Start it at boy', 'girl'); T.v3d = L('Start at father').start; T.v3mother = w('Start at father', 'mother');
    T.v3e = L('A direction carries').start; T.v3f = L('Now the famous one').start; T.v3king = w('Now the famous one', 'king');
    T.v3g = L('You get King').start;
    const kg = L('You get King').words.filter((y) => y.w.toLowerCase().startsWith('king'));
    T.v3k1 = kg[0]!.start; T.v3k2 = (kg[1] ?? kg[0]!).start; T.v3k3 = (kg[2] ?? kg[kg.length - 1]!).start;
    T.v3h = L('And then queen').start; T.v3queen = w('And then queen', 'queen'); T.v3hEnd = L('And then queen').end;
    T.br = L("It isn't magic").start; T.br2 = L('The spelling pulls').start; T.brEnd = L('The spelling pulls').end;
    T.out = L('Next time').start; T.outEnd = L('Next time').end;
  }

  // ------------------------------------------------------------------ geometry
  private build() {
    const e = this.emb;
    // the haze: 8,000 common words, same squash
    const hp = e.space.haze.p, n = e.space.haze.n;
    const g = new THREE.BufferGeometry();
    const arr = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) arr[i] = hp[i]! * R;
    g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    this.hazeMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { pxs: { value: 1 }, size: { value: 0.05 }, col: { value: new THREE.Vector3(...LIN.blue) }, vis: { value: 1 } },
      vertexShader: /* glsl */ `
        uniform float pxs, size; varying float vA;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          float px = size * pxs / max(0.05, -mv.z);
          gl_PointSize = clamp(px, 1.6 * ${SCALE.toFixed(1)}, 14.0 * ${SCALE.toFixed(1)});
          vA = clamp(px / (2.4 * ${SCALE.toFixed(1)}), 0.45, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 col; uniform float vis; varying float vA;
        void main() {
          vec2 q = gl_PointCoord * 2.0 - 1.0; float r = dot(q, q);
          if (r > 1.0) discard;
          gl_FragColor = vec4(col, smoothstep(1.0, 0.55, r) * vA * vis);
        }`,
    });
    const haze = new THREE.Points(g, this.hazeMat);
    haze.frustumCulled = false;
    this.s3.add(haze);

    // featured words: inked spheres (blue; pink while taught)
    const geo = new THREE.SphereGeometry(0.065, 20, 14);
    for (const f of e.space.featured) {
      const mat = inkMat({ ink: 'blue', lit: 1, shade: 1 });
      const m = new THREE.Mesh(geo, mat);
      outline(m, 1.6);
      m.position.copy(this.pos.get(f.t)!);
      this.s3.add(m);
      this.sph.set(f.t, { mesh: m, mat });
    }

    // the table wall: 48 real rows x 4,096 values; near, each row is a bar chart of its numbers (solid ink, so
    // the print keeps it); far, the bars average into a tint band (mipmapped |value|)
    const tv = tableValues(e.table), rows = e.table.rows;
    const pxA = new Uint8Array(4096 * rows * 4), pxS = new Uint8Array(4096 * rows * 4);
    for (let r = 0; r < rows; r++) for (let c = 0; c < 4096; c++) {
      const v = tv[r * 4096 + c]!, k = ((rows - 1 - r) * 4096 + c) * 4; // texture row 0 = bottom
      const g = Math.sign(v) * Math.pow(Math.abs(v), 0.6);
      pxA[k] = Math.round(Math.abs(g) * 255); pxA[k + 3] = 255;
      pxS[k] = Math.round((g * 0.5 + 0.5) * 255); pxS[k + 3] = 255;
    }
    const texA = new THREE.DataTexture(pxA, 4096, rows, THREE.RGBAFormat, THREE.UnsignedByteType);
    texA.generateMipmaps = true; texA.minFilter = THREE.LinearMipmapLinearFilter; texA.magFilter = THREE.LinearFilter;
    texA.anisotropy = 16; texA.needsUpdate = true;
    const texS = new THREE.DataTexture(pxS, 4096, rows, THREE.RGBAFormat, THREE.UnsignedByteType);
    texS.minFilter = THREE.NearestFilter; texS.magFilter = THREE.NearestFilter; texS.needsUpdate = true;
    const rainRow = e.rows[' rain']!.id - e.table.first_id;
    this.wallMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      uniforms: {
        texA: { value: texA }, texS: { value: texS }, rows: { value: rows }, rainRow: { value: rainRow }, reveal: { value: 0 }, vis: { value: 0 },
        hot: { value: 0 }, hideRain: { value: 0 }, paper: { value: new THREE.Vector3(...LIN.paper) }, blue: { value: new THREE.Vector3(...LIN.blue) },
        pink: { value: new THREE.Vector3(...LIN.pink) }, ink: { value: new THREE.Vector3(...LIN.ink) },
      },
      vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D texA, texS; uniform float rows, rainRow, reveal, vis, hot, hideRain; uniform vec3 paper, blue, pink, ink;
        varying vec2 vUv;
        void main() {
          float rf = (1.0 - vUv.y) * rows;           // row from the top
          float row = floor(rf), fy = fract(rf);
          float tv = (rows - 1.0 - row + 0.5) / rows;
          float col = vUv.x * 4096.0, fp = fwidth(col);
          float y = (0.5 - fy) * 2.0;                 // +1 top of the row, -1 bottom
          float yw = fwidth(y);
          // near: bars (78% of a column wide) from the row's midline to the value
          float v = texture(texS, vec2((floor(col) + 0.5) / 4096.0, tv)).r * 2.0 - 1.0;
          float fc = fract(col);
          float inX = smoothstep(0.0, fp * 0.7, fc) * (1.0 - smoothstep(0.78 - fp * 0.7, 0.78, fc));
          float h = v * 0.86;
          float inY = h >= 0.0 ? smoothstep(-yw, 0.0, y) * (1.0 - smoothstep(h, h + yw, y)) : smoothstep(h - yw, h, y) * (1.0 - smoothstep(0.0, yw, y));
          float near = inX * inY;
          // far: the average ink of the columns under this pixel
          float far = textureGrad(texA, vec2(vUv.x, tv), vec2(dFdx(vUv.x), 0.0), vec2(dFdy(vUv.x), 0.0)).r * 1.1
                      * smoothstep(0.0, 0.25, 1.0 - abs(y));
          float a = mix(near, far, smoothstep(0.5, 1.4, fp));
          float shown = clamp((reveal * (rows + 6.0) - row) / 6.0, 0.0, 1.0);
          float isRain = 1.0 - step(0.5, abs(row - rainRow));
          a *= (1.0 - isRain * hideRain);
          // the row's rule: a thin ink line along its bottom edge
          float rule = 1.0 - smoothstep(0.0, fwidth(fy) * 1.2, fy);
          vec3 c = mix(blue, pink, isRain * hot);
          vec3 colr = paper * mix(vec3(1.0), c / paper, clamp(a, 0.0, 1.0));
          colr = mix(colr, ink, rule * 0.5);
          gl_FragColor = vec4(colr, vis * shown);
        }`,
    });
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(WALL_W, rows * ROW_H), this.wallMat);
    // rain's row centred on y = 0
    wall.position.set(0, (rows / 2 - rainRow - 0.5) * ROW_H, WALL_Z);
    wall.renderOrder = 10; // drawn after the haze, so the sheet hides the space behind it
    this.s3.add(wall);

    // rain's 4,096 numbers as bars standing on its row (flat, printed on the wall), collapsing to a point
    const vals = e.rows[' rain']!.values;
    const vmax = Math.max(...vals.map(Math.abs));
    const bg = new THREE.InstancedBufferGeometry();
    bg.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-0.5, 0, 0, 0.5, 0, 0, 0.5, 1, 0, -0.5, 1, 0]), 3));
    bg.setIndex([0, 1, 2, 0, 2, 3]);
    const ax = new Float32Array(4096), av = new Float32Array(4096);
    for (let i = 0; i < 4096; i++) { ax[i] = (i + 0.5) / 4096; av[i] = vals[i]! / vmax; }
    bg.setAttribute('aX', new THREE.InstancedBufferAttribute(ax, 1));
    bg.setAttribute('aV', new THREE.InstancedBufferAttribute(av, 1));
    bg.instanceCount = 4096;
    this.barMat = new THREE.ShaderMaterial({
      side: THREE.DoubleSide,
      uniforms: { grow: { value: 0 }, collapse: { value: 0 }, hs: { value: 0.95 }, col: { value: new THREE.Vector3(...LIN.pink) } },
      vertexShader: /* glsl */ `
        attribute float aX, aV; uniform float grow, collapse, hs;
        void main() {
          float x = mix(-${(WALL_W / 2).toFixed(1)} + aX * ${WALL_W.toFixed(1)}, 0.0, collapse);
          float bw = ${(WALL_W / 4096 * 0.72).toFixed(5)} * (1.0 - collapse);
          float h = aV * hs * grow * (1.0 - collapse);
          vec3 p = vec3(x + position.x * bw, position.y * h, ${(WALL_Z + 0.03).toFixed(2)});
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }`,
      fragmentShader: /* glsl */ `uniform vec3 col; void main() { gl_FragColor = vec4(col, 1.0); }`,
    });
    const bars = new THREE.Mesh(bg, this.barMat);
    bars.frustumCulled = false;
    this.s3.add(bars);
  }

  // ------------------------------------------------------------------ the plan: camera keys and cues
  private island(toks: string[]) {
    const c = V3(0, 0, 0);
    for (const t of toks) c.add(this.pos.get(t)!);
    return c.multiplyScalar(1 / toks.length);
  }

  private plan() {
    const T = this.T, P = this.pos, land = this.land;
    const rainP = P.get(' rain')!.clone();
    const Crain = this.island([' rain', ' Rain', 'Rain', ' rains', ' raining', ' rainfall', 'rain']);
    const Cweather = this.island([' rain', ' snow', ' wind', ' storm', ' Rain', ' rains']);
    const Cnum = this.island([' three', ' four', ' two', ' five']);
    const Cpeople = this.island([' man', ' woman', ' boy', ' girl', ' father', ' mother']);
    const Cking = this.island([' king', ' King', ' KING', 'King']).lerp(land.king!, 0.3);
    const queen = P.get(' queen')!;
    const mid = V3(3.2, -2.6, 0); // the middle of everything labelled
    const wide = (a: number, d = 27) => mid.clone().add(V3(Math.sin(a) * d, 0.12 * d, Math.cos(a) * d));
    const k: [number, THREE.Vector3, THREE.Vector3, number][] = [];
    const key = (t: number, pos: THREE.Vector3, tgt: THREE.Vector3, fov = 36) => k.push([t, pos, tgt, fov]);
    const off = (c: THREE.Vector3, o: THREE.Vector3) => c.clone().add(o);

    // intro: the whole space from far away, drifting in
    key(0, off(mid, V3(14, 5, 104)), mid);
    key(T.v1! - 3, V3(5, 2, 82), V3(1, -1, 5));
    // verse 1: the table, then along rain's row, then the row folds into a point
    key(T.v1b!, V3(0, 0.6, WALL_Z + 44), V3(0, 0, WALL_Z));
    key(T.v1c!, V3(-18.8, 2.0, WALL_Z + 6.2), V3(-15.6, 0, WALL_Z));
    key(T.v1e!, V3(-6.8, 2.2, WALL_Z + 6.4), V3(-3.6, 0, WALL_Z));
    key(T.v1f!, V3(5.8, 2.4, WALL_Z + 6.8), V3(9, 0, WALL_Z));
    key(T.v1g!, V3(0, 2.5, WALL_Z + 30), V3(0, 0, WALL_Z));
    key(T.v1h!, V3(0, 1.5, WALL_Z + 17), V3(0, 0, WALL_Z));
    // the chorus picture (the same move every time): from rain, out to the whole space, turning slowly
    const chorus = (n: number) => {
      key(T[`c${n}`]!, off(rainP, V3(2.2, 2.4, 8.5)), rainP.clone().lerp(mid, 0.12));
      key(T[`c${n}c`]!, wide(-0.24, 27), mid);
      if (n < 2) key(T[`c${n}End`]! + 1.5, wide(0.14, 28), mid);
      else {
        // the last chorus leans in on "direction": the arrows, man -> woman and its copies, up close
        const ar = Cpeople.clone().lerp(Cking, 0.4);
        key(T.c2d! + 1.8, off(ar, V3(10, 2.5, 12)), ar);
      }
    };
    chorus(0);
    key(T.v2! - 4, wide(0.3, 30), mid);
    // verse 2: close = pointing the same way; three's neighbours; rain's neighbours
    key(T.v2!, off(Cweather, V3(3.5, 2, 11)), Cweather.clone().add(V3(-1, 0, 0)));
    key(T.v2c! - 3.2, off(Cweather, V3(3, 2.4, 10)), Cweather.clone().add(V3(-1.2, 0, 0)));
    key(T.v2c!, off(Cnum, V3(-1, 2, 7.5)), Cnum);
    key(T.v2d!, off(Cnum, V3(-3, 1.5, 7)), Cnum);
    key(T.v2d! + 1.6, off(Cnum, V3(-3.5, 1.4, 6.8)), Cnum);
    key((T.v2d! + T.v2e!) / 2 + 1.0, off(mid, V3(-2, 3, 20)), V3(2, 1.5, -1));
    key(T.v2e!, off(Crain, V3(2.5, 1.5, 6.5)), Crain);
    key(T.v2f!, off(Cweather, V3(3.5, 2, 10)), Cweather);
    key(T.v2g!, off(Cweather, V3(4.5, 2.2, 10.5)), Cweather.clone().add(V3(2.8, 0, 0)));
    key(T.v2hEnd!, off(Cweather, V3(5.5, 2.5, 11)), Cweather.clone().add(V3(2.8, 0, 0)));
    chorus(1);
    key(T.v3! - 4, wide(0.5, 28), mid);
    // verse 3: the arrow, seen from the side (it points mostly along the 3rd direction)
    key(T.v3!, off(Cpeople, V3(5.8, 1.2, 1.2)), Cpeople);
    key(T.v3c!, off(Cpeople, V3(5.4, 1.8, 2)), Cpeople);
    key(T.v3e!, off(Cpeople, V3(6.2, 1.2, 0.2)), Cpeople);
    const way = Cpeople.clone().lerp(Cking, 0.5);
    key(T.v3f! - 1.5, off(Cpeople, V3(6.4, 1.3, 0.6)), Cpeople);
    key(T.v3f! + 2.6, off(way, V3(27, 3, 8)), way);
    key(T.v3g!, off(Cking, V3(6.5, 1.2, 1.5)), Cking);
    const kq = Cking.clone().lerp(queen, 0.4);
    key(T.v3h!, off(kq, V3(10.5, 1.6, 2.5)), kq);
    key(T.br!, off(Cking.clone().lerp(queen, 0.45), V3(12.5, 2.2, 4)), Cking.clone().lerp(queen, 0.45));
    key(Math.min(T.br2! + 2, T.c2! - 3.2), off(Cking.clone().lerp(queen, 0.5), V3(15, 3, 7)), Cking.clone().lerp(queen, 0.5));
    chorus(2);
    // outro: out past everything
    key(T.out!, wide(0.4, 48), mid);
    key(T.end!, wide(0.6, 90), mid);
    k.sort((a, b) => a[0] - b[0]);
    // keys closer than 0.8 s (a section's last key can land on the next one's first) would make the spline divide by
    // ~0: the earlier one gives way
    for (let i = k.length - 1; i > 0; i--) if (k[i]![0] - k[i - 1]![0] < 0.8) k.splice(i - 1, 1);
    this.path = new Spline(k.map((x) => x[0]), k.map(([, p, t, f]) => [p.x, p.y, p.z, t.x, t.y, t.z, f]));

    // labels: rain first (verse 1), everything at "Words that mean alike live close together"
    this.labelAt.set(' rain', T.v1c! + 0.2);
    const others = this.emb.space.featured.map((f) => f.t).filter((t) => t !== ' rain');
    others.forEach((t, i) => this.labelAt.set(t, T.c0c! + (i / others.length) * 2.4));

    // pink: the word being taught, while it is sung
    const pk = (tok: string, t0: number, t1: number) => this.pinks.push({ tok, t0, t1 });
    pk(' rain', T.v1c!, T.c0c!);
    pk(' three', T.v2c!, T.v2d!);
    pk(' four', T.v2four!, T.v2two!); pk(' two', T.v2two!, T.v2five!); pk(' five', T.v2five!, T.v2d!);
    pk(' rain', T.v2e!, T.v2hEnd!);
    pk(' Rain', T.v2Rain!, T.v2f!);
    pk(' rains', T.v2rains!, T.v2snow!); pk(' snow', T.v2snow!, T.v2wind!); pk(' wind', T.v2wind!, T.v2storm!); pk(' storm', T.v2storm!, T.v2g!);
    pk(' man', T.v3!, T.v3b!); pk(' woman', T.v3!, T.v3b!);
    pk(' boy', T.v3b!, T.v3c!); pk(' girl', T.v3girl!, T.v3d!);
    pk(' father', T.v3d!, T.v3mother!); pk(' mother', T.v3mother!, T.v3e!);
    pk(' king', T.v3king!, T.v3g!);
    pk(' King', T.v3k1!, T.v3k2!); pk(' KING', T.v3k2!, T.v3k3!); pk('King', T.v3k3!, T.v3h!);
    pk(' queen', T.v3queen!, T.brEnd!);

    // neighbour threads (verse 2): real cosines over the whole table
    const nb = (s: string, tok: string) => this.emb.neighbours[s]!.find((x) => x.t === tok)!.cos!;
    const th = (a: string, b: string, t: number, cur1: number) => this.threads.push({ a, b, t, cur1, cos: nb(a, b) });
    th(' three', ' four', T.v2four!, T.v2two!); th(' three', ' two', T.v2two!, T.v2five!); th(' three', ' five', T.v2five!, T.v2d!);
    th(' rain', ' Rain', T.v2Rain!, T.v2f!); th(' rain', ' rains', T.v2rains!, T.v2snow!); th(' rain', ' snow', T.v2snow!, T.v2wind!);
    th(' rain', ' wind', T.v2wind!, T.v2storm!); th(' rain', ' storm', T.v2storm!, T.v2g!);

    // the arrow (verse 3): woman - man, carried to boy, father, king (3D landing = the real sum, squashed)
    const vec = P.get(' woman')!.clone().sub(P.get(' man')!);
    const man = P.get(' man')!, boy = P.get(' boy')!, father = P.get(' father')!, king = P.get(' king')!;
    const slide = (a: THREE.Vector3, b: THREE.Vector3, t0: number, t1: number) => (t: number) => a.clone().lerp(b, sm((t - t0) / (t1 - t0)));
    this.arrows.push({ tail: () => man, vec, t0: T.v3! + 0.3, grow: 1.6, cur0: T.v3!, cur1: T.v3b! });
    this.arrows.push({ tail: slide(man, boy, T.v3b! + 0.2, T.v3c!), vec, t0: T.v3b!, grow: 0.01, cur0: T.v3b!, cur1: T.v3d! });
    this.arrows.push({ tail: slide(boy, father, T.v3d! + 0.1, T.v3d! + 1.4), vec, t0: T.v3d!, grow: 0.01, cur0: T.v3d!, cur1: T.v3e! });
    this.arrows.push({ tail: slide(father, king, T.v3f! + 0.2, T.v3g! - 0.2), vec, t0: T.v3f!, grow: 0.01, cur0: T.v3f!, cur1: T.v3hEnd! });

    // one number at a time
    const at = (tok: string) => () => this.pos.get(tok)!;
    const n = (t0: number, t1: number, text: string, cap: string, a: () => THREE.Vector3 | null) => this.nums.push({ t0, t1, text, cap, at: a });
    const e = this.emb;
    n(T.v1b!, T.v1c!, e.vocab.toLocaleString('en-US'), 'rows: one for every token (a word or a piece of one)', () => null);
    n(T.v1c! + 0.3, T.v1d!, e.rows[' rain']!.id.toLocaleString('en-US'), '·rain’s row (its token id)', () => V3(-WALL_W / 2 + 0.3, 0.5, WALL_Z));
    n(T.v1d!, T.v1e!, e.dim.toLocaleString('en-US'), 'numbers in every row', () => null);
    n(T.v1f!, T.v1g!, e.rows[' rain']!.values[0]!.toFixed(4).replace('-', '−'), '·rain’s first number', () => null);
    n(T.c0b!, T.c0c!, `${(e.space.var_kept_total * 100).toFixed(1)}%`, 'of the labelled words’ spread survives the squash to 3D (PCA)', () => null);
    n(T.v2zero! + 1.2, T.v2c!, fmtCos(e.baseline.mean_cos), 'average for two random words', () => null);
    for (const t of this.threads) n(t.t, t.cur1, fmtCos(t.cos), `${tokLabel(t.a)} and ${tokLabel(t.b)}`, at(t.b));
    const an = e.analogies;
    n(T.v3girl!, T.v3d!, fmtCos(an.boy!.top5[0]!.cos!), `#1 nearest: ${tokLabel(an.boy!.top5[0]!.t)}`, () => this.land.boy!);
    n(T.v3mother!, T.v3e!, fmtCos(an.father!.top5[0]!.cos!), `#1 nearest: ${tokLabel(an.father!.top5[0]!.t)}`, () => this.land.father!);
    const kt = an.king!.top5;
    n(T.v3k1!, T.v3k2!, fmtCos(kt[0]!.cos!), `#1 ${tokLabel(kt[0]!.t)}`, at(kt[0]!.t));
    n(T.v3k2!, T.v3k3!, fmtCos(kt[1]!.cos!), `#2 ${tokLabel(kt[1]!.t)}`, at(kt[1]!.t));
    n(T.v3k3!, T.v3queen!, fmtCos(kt[2]!.cos!), `#3 ${tokLabel(kt[2]!.t)}`, at(kt[2]!.t));
    n(T.v3queen!, T.brEnd!, fmtCos(kt[3]!.cos!), `#4 ${tokLabel(kt[3]!.t)}`, at(kt[3]!.t));
  }

  // ------------------------------------------------------------------ per frame
  private setCam(cam: THREE.PerspectiveCamera, t: number) {
    const v = this.path.at(t), T = this.T;
    const tgt = V3(v[3]!, v[4]!, v[5]!);
    const pos = V3(v[0]!, v[1]!, v[2]!);
    // follow the rain point on its way from the wall to its place (blend in, then back to the path)
    const fw = sm((t - (T.v1h! - 1.0)) / 1.5) * (1 - sm((t - (T.c0b! + 0.5)) / (T.c0c! - T.c0b! - 0.5)));
    if (fw > 0) {
      const rp = this.rainPos(t);
      const chase = rp.clone().add(V3(1.2, 1.6, 7.5));
      tgt.lerp(rp, fw);
      pos.lerp(chase, fw);
    }
    // a gentle drift so the picture never freezes: a slow sway around the target
    const off = pos.clone().sub(tgt);
    const a = 0.035 * Math.sin(t * 0.21) + 0.02 * Math.sin(t * 0.13 + 1.3);
    off.applyAxisAngle(V3(0, 1, 0), a);
    off.y += 0.25 * Math.sin(t * 0.17 + 0.4) * off.length() * 0.05;
    cam.position.copy(tgt).add(off);
    cam.up.set(0, 1, 0);
    cam.lookAt(tgt);
    const fov = v[6]!;
    cam.fov = portrait() ? (2 * Math.atan(Math.tan((fov * Math.PI) / 360) * 1.85) * 180) / Math.PI : fov;
    cam.aspect = W / H;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
  }

  /** Where the rain point is: on the wall until verse 1 ends, then flying to its place. */
  private rainPos(t: number) {
    const T = this.T, home = this.pos.get(' rain')!;
    const k = sm((t - T.v1h!) / (T.c0b! - T.v1h!));
    return this.wallCenter.clone().lerp(home, k);
  }

  private pinkOf(tok: string, t: number) {
    let k = 0;
    for (const p of this.pinks) if (p.tok === tok) k = Math.max(k, clamp((t - p.t0) / 0.3) * (1 - clamp((t - p.t1) / 0.4)));
    return k;
  }

  render(f: Frame, out: THREE.WebGLRenderTarget) {
    const r = this.ctx.renderer, t = f.t, T = this.T;
    this.wash.render(r, out, { seed: 11, drift: t * 0.08, amt: 0.22, c1: 'blue', c2: 'blue' });

    // verse 1 state
    const wallIn = clamp((t - (T.v1a! - 0.6)) / 3.0);
    const wallOut = clamp((t - T.v1g!) / 2.2);
    this.wallMat.uniforms.reveal!.value = wallIn;
    this.wallMat.uniforms.vis!.value = 1 - wallOut;
    this.wallMat.uniforms.hot!.value = clamp((t - T.v1c!) / 0.5);
    this.wallMat.uniforms.hideRain!.value = clamp((t - T.v1c! - 0.4) / 0.6);
    this.barMat.uniforms.grow!.value = sm((t - T.v1c! - 0.2) / 1.2);
    this.barMat.uniforms.collapse!.value = ease.inOutCubic(clamp((t - T.v1g! - 0.3) / (T.v1h! - T.v1g! + 0.5)));

    // featured spheres: rain animated, pink while taught
    const pinkV = new THREE.Vector3(...LIN.pink).max(V3(0.03, 0.03, 0.03)), blueV = new THREE.Vector3(...LIN.blue).max(V3(0.03, 0.03, 0.03));
    for (const [tok, s] of this.sph) {
      const k = this.pinkOf(tok, t);
      (s.mat.uniforms.inkA!.value as THREE.Vector3).copy(blueV).lerp(pinkV, k);
      s.mesh.scale.setScalar(1 + 0.6 * k);
      if (tok === ' rain') {
        s.mesh.position.copy(this.rainPos(t));
        const born = clamp((t - T.v1g! - 0.3) / (T.v1h! - T.v1g! + 0.5));
        s.mesh.visible = t > T.v1g!;
        s.mesh.scale.setScalar((1 + 0.35 * k) * (t < T.c0! ? lerp(0.3, 3.4, ease.outCubic(born)) - (t > T.v1h! ? 1.8 * sm((t - T.v1h!) / (T.c0b! - T.v1h!)) : 0) : 1 + 0.6 * k));
      }
    }

    this.setCam(this.cam, t);
    for (const [, sp] of this.sph) {
      const d = this.cam.position.distanceTo(sp.mesh.position);
      sp.mesh.scale.multiplyScalar(Math.max(1, (d * 0.003) / 0.065));
    }
    this.hazeMat.uniforms.pxs!.value = (H * SCALE) / (2 * Math.tan((this.cam.fov * Math.PI) / 360));
    r.setRenderTarget(out);
    r.clearDepth();
    r.render(this.s3, this.cam);

    const fadeEnd = clamp((t - (T.end! - 3.5)) / 3);
    return { reg: 7, kinetic: 0, fade: fadeEnd, crawl: [f.ft * 9, f.ft * 4] as [number, number] };
  }

  // ------------------------------------------------------------------ overlay (Canvas2D)
  private proj(p: THREE.Vector3): [number, number, boolean] {
    const v = p.clone().project(this.camF);
    return [(v.x * 0.5 + 0.5) * W, (-v.y * 0.5 + 0.5) * H, v.z < 1 && v.z > -1];
  }

  private overlay(c: CanvasRenderingContext2D, t: number) {
    const T = this.T, P = portrait();
    this.setCam(this.camF, t);
    const rainHere = this.rainPos(t);
    const posOf = (tok: string) => (tok === ' rain' ? rainHere : this.pos.get(tok)!);

    // intro title
    const title = clamp((t - 0.8) / 1.2) * (1 - clamp((t - (T.v1! - 2.5)) / 1.5));
    if (title > 0) this.titleCard(c, title);

    // verse 1: the table's words (row labels at the wall's left edge) and the rain row's name
    this.wallLabels(c, t);

    // threads (verse 2), drawn from a toward b
    for (const th of this.threads) {
      const g = clamp((t - th.t) / 0.6);
      if (g <= 0) continue;
      const [ax, ay, aok] = this.proj(posOf(th.a)), [bx, by, bok] = this.proj(posOf(th.b));
      if (!aok || !bok) continue;
      const pk = Math.max(1 - clamp((t - th.cur1 - 0.3) / 0.4), this.lineK(t, 'c1c'), this.lineK(t, 'c2c'));
      const x1 = lerp(ax, bx, ease.outCubic(g)), y1 = lerp(ay, by, ease.outCubic(g));
      c.save();
      c.strokeStyle = INK; c.lineWidth = 1.8; c.globalAlpha = 0.75 * (1 - pk);
      c.beginPath(); c.moveTo(ax, ay); c.lineTo(x1, y1); c.stroke();
      c.strokeStyle = PINK; c.lineWidth = 3.5; c.globalAlpha = pk;
      c.beginPath(); c.moveTo(ax, ay); c.lineTo(x1, y1); c.stroke();
      c.restore();
    }

    // arrows (verse 3)
    for (const a of this.arrows) {
      const g = clamp((t - a.t0) / a.grow);
      if (g <= 0) continue;
      const tail = a.tail(t), head = tail.clone().add(a.vec.clone().multiplyScalar(ease.outCubic(g)));
      const [x0, y0, ok0] = this.proj(tail), [x1, y1, ok1] = this.proj(head);
      if (!ok0 || !ok1) continue;
      const pk = Math.max(clamp((t - a.cur0) / 0.3) * (1 - clamp((t - a.cur1) / 0.4)), this.lineK(t, 'c2d'));
      if (pk < 0.999) { c.save(); c.globalAlpha = 1 - pk; this.arrow2D(c, x0, y0, x1, y1, INK, 3); c.restore(); }
      if (pk > 0.001) { c.save(); c.globalAlpha = pk; this.arrow2D(c, x0, y0, x1, y1, PINK, 5); c.restore(); }
    }
    // landing marks (the real sum, squashed)
    for (const [k, tt] of [['boy', T.v3girl!], ['father', T.v3mother!], ['king', T.v3g!]] as const) {
      const a = clamp((t - tt + 0.3) / 0.4);
      if (a <= 0) continue;
      const [x, y, ok] = this.proj(this.land[k]!);
      if (!ok) continue;
      c.save(); c.globalAlpha = a; c.strokeStyle = INK; c.lineWidth = 2;
      c.beginPath(); c.arc(x, y, 11, 0, Math.PI * 2); c.stroke();
      c.beginPath(); c.moveTo(x - 17, y); c.lineTo(x + 17, y); c.moveTo(x, y - 17); c.lineTo(x, y + 17); c.stroke();
      c.restore();
    }

    // from each landing point, threads to the real nearest rows (in all 4,096 numbers), as they are sung
    const lt = (k: string, tok: string, t0: number, t1: number) => {
      const g = clamp((t - t0) / 0.5);
      if (g <= 0) return;
      const [ax, ay, aok] = this.proj(this.land[k]!), [bx, by, bok] = this.proj(posOf(tok));
      if (!aok || !bok) return;
      const cur = t < t1;
      c.save(); c.setLineDash([7, 6]); c.strokeStyle = cur ? PINK : INK; c.lineWidth = cur ? 3 : 1.6; c.globalAlpha = cur ? 1 : 0.7;
      c.beginPath(); c.moveTo(ax, ay); c.lineTo(lerp(ax, bx, ease.outCubic(g)), lerp(ay, by, ease.outCubic(g))); c.stroke(); c.restore();
    };
    lt('boy', ' girl', T.v3girl!, T.v3d!);
    lt('father', ' mother', T.v3mother!, T.v3e!);
    const kt = this.emb.analogies.king!.top5;
    lt('king', kt[0]!.t, T.v3k1!, T.v3k2!); lt('king', kt[1]!.t, T.v3k2!, T.v3k3!); lt('king', kt[2]!.t, T.v3k3!, T.v3queen!);
    lt('king', kt[3]!.t, T.v3queen!, T.brEnd!);

    // "Every word is a list of numbers": rain's first 96 numbers beside its point, on each chorus's first line
    for (const n of [0, 1, 2]) {
      const k = clamp((t - T[`c${n}`]! + 0.2) / 0.5) * (1 - clamp((t - T[`c${n}b`]! - 0.6) / 0.5));
      if (k > 0) this.miniRow(c, posOf(' rain'), k);
    }

    // featured labels
    this.labels(c, t, posOf);

    // the axes of the squash (from the first chorus's "place")
    this.axes(c, t);

    // verse 2 inset: what "close" means (cosine), then rain's ranked neighbours
    this.cosInset(c, t);
    this.rankPanel(c, t, 'rain', T.v2g! - 0.2, T.c1c!, ' rain', this.emb.neighbours[' rain']!.slice(0, 11).map((x) => x.t), 'nearest to ·rain, of every row in the table');
    this.rankPanel(c, t, 'king', T.v3g! - 0.1, T.c2c!, ' king', this.emb.analogies.king!.top5.map((x) => x.t), 'king − man + woman: nearest rows (king, man, woman left out)');

    // the one number
    this.number(c, t);

    // fine print: what the 3D view is
    const fp = clamp((t - T.c0b!) / 0.8) * (1 - clamp((t - T.out!) / 1));
    if (fp > 0) {
      c.save(); c.globalAlpha = fp; c.fillStyle = INK; c.font = font(F.mono(500), P ? 15 : 13); c.textAlign = 'right';
      c.letterSpacing = '1px';
      const s = '3D VIEW: THE NUMBERS SQUASHED TO THREE DIRECTIONS (PCA OF THE LABELLED WORDS)';
      if (P) { c.fillText('3D VIEW: THE NUMBERS SQUASHED TO THREE', W - 36, H - 110); c.fillText('DIRECTIONS (PCA OF THE LABELLED WORDS)', W - 36, H - 90); }
      else c.fillText(s, W - 36, H - 52);
      c.restore();
    }

    // outro: the next episode
    const nx = clamp((t - T.out! + 0.4) / 1.8);
    if (nx > 0) this.nextCard(c, nx);
  }

  /** 0..1 while chorus line `key` is sung, fading in and out over 0.4 s (no one-frame colour switches). */
  private lineK(t: number, key: string) {
    const T = this.T, n = key.slice(0, 2), part = key.slice(2);
    const order = ['', 'b', 'c', 'd'];
    const t0 = T[`${n}${part}`]!, i = order.indexOf(part);
    const t1 = i < 3 ? T[`${n}${order[i + 1]}`]! : T[`${n}End`]!;
    return clamp((t - t0 + 0.2) / 0.4) * (1 - clamp((t - t1 + 0.2) / 0.4));
  }

  /** A strip of rain's first 96 real numbers as pink bars, hung under its point. */
  private miniRow(c: CanvasRenderingContext2D, p: THREE.Vector3, k: number) {
    const [x, y, ok] = this.proj(p);
    if (!ok) return;
    const vals = this.emb.rows[' rain']!.values, n = 96, bw = 3, h = 34;
    const vmax = Math.max(...vals.slice(0, n).map(Math.abs));
    const x0 = clamp(x - (n * bw) / 2, 30, W - n * bw - 30), y0 = clamp(y + 58, 60, H - 90);
    c.save();
    c.globalAlpha = k * 0.9; c.fillStyle = PAPER; c.fillRect(x0 - 10, y0 - h - 8, n * bw + 20, 2 * h + 40);
    c.globalAlpha = k;
    c.fillStyle = PINK;
    for (let i = 0; i < n; i++) {
      const v = (vals[i]! / vmax) * h * Math.min(1, k * 1.5 - i / n * 0.5);
      c.fillRect(x0 + i * bw, v > 0 ? y0 - v : y0, bw - 1, Math.abs(v));
    }
    c.fillStyle = INK; c.fillRect(x0, y0 - 0.75, n * bw, 1.5);
    c.font = font(F.archivo(100, 600), 18);
    c.fillText('·rain: the first 96 of its numbers', x0, y0 + h + 22);
    c.restore();
  }

  private titleCard(c: CanvasRenderingContext2D, k: number) {
    const P = portrait();
    c.save();
    c.globalAlpha = k;
    c.textAlign = 'center'; c.textBaseline = 'middle';
    let size = P ? 104 : 128;
    c.font = font(F.archivo(112.5, 900), size);
    const tw = c.measureText(P ? 'A DIRECTION' : 'MEANING IS A DIRECTION').width;
    if (tw > W * 0.86) { size *= (W * 0.86) / tw; c.font = font(F.archivo(112.5, 900), size); }
    c.fillStyle = INK;
    if (P) { c.fillText('MEANING IS', W / 2, H * 0.4); c.fillText('A DIRECTION', W / 2, H * 0.4 + size * 1.02); }
    else c.fillText('MEANING IS A DIRECTION', W / 2, H * 0.47);
    c.font = font(F.mono(600), P ? 26 : 24); c.letterSpacing = '3px';
    c.fillStyle = BLUE;
    c.fillText('HOW A MODEL STORES WHAT WORDS MEAN', W / 2, P ? H * 0.4 + size * 2.0 : H * 0.47 + size * 0.72);
    c.restore();
  }

  private nextCard(c: CanvasRenderingContext2D, k: number) {
    const P = portrait();
    c.save();
    c.globalAlpha = ease.outCubic(k);
    c.textAlign = 'center'; c.textBaseline = 'middle';
    const y = P ? H * 0.46 : H * 0.5;
    c.fillStyle = PAPER; c.globalAlpha = ease.outCubic(k) * 0.9;
    c.fillRect(W / 2 - (P ? 480 : 620), y - (P ? 170 : 160), P ? 960 : 1240, P ? 320 : 300);
    c.globalAlpha = ease.outCubic(k);
    c.font = font(F.mono(700), P ? 28 : 26); c.letterSpacing = '3px'; c.fillStyle = BLUE;
    c.fillText('NEXT · ML, SLOWLY 4/5', W / 2, y - (P ? 110 : 96));
    c.font = font(F.archivo(112.5, 900), P ? 110 : 120); c.letterSpacing = '0px'; c.fillStyle = INK;
    c.fillText('WHO’S IT?', W / 2, y);
    c.font = font(F.archivo(100, 500), P ? 34 : 34); c.fillStyle = INK;
    c.fillText('how the words look at each other (attention)', W / 2, y + (P ? 100 : 92));
    c.restore();
  }

  private arrow2D(c: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, col: string, w: number) {
    const a = Math.atan2(y1 - y0, x1 - x0), L = Math.hypot(x1 - x0, y1 - y0), hl = Math.min(22, L * 0.45);
    c.save();
    c.strokeStyle = col; c.fillStyle = col; c.lineWidth = w; c.lineCap = 'round';
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1 - Math.cos(a) * hl * 0.8, y1 - Math.sin(a) * hl * 0.8); c.stroke();
    c.beginPath(); c.moveTo(x1, y1);
    c.lineTo(x1 - Math.cos(a - 0.42) * hl, y1 - Math.sin(a - 0.42) * hl);
    c.lineTo(x1 - Math.cos(a + 0.42) * hl, y1 - Math.sin(a + 0.42) * hl);
    c.closePath(); c.fill();
    c.restore();
  }

  private wallLabels(c: CanvasRenderingContext2D, t: number) {
    const T = this.T, e = this.emb, vis = clamp((t - (T.v1a! - 0.6)) / 3.0) * (1 - clamp((t - T.v1g!) / 1.6));
    if (vis <= 0) return;
    const rows = e.table.rows, rainRow = e.rows[' rain']!.id - e.table.first_id;
    c.save();
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (let r = 0; r < rows; r++) {
      const y = (rainRow - r) * ROW_H;
      const [x0, y0, ok] = this.proj(V3(-WALL_W / 2 - 0.25, y, WALL_Z));
      const [, y1] = this.proj(V3(-WALL_W / 2 - 0.25, y + ROW_H, WALL_Z));
      if (!ok) continue;
      const hpx = Math.abs(y1 - y0);
      const isRain = r === rainRow;
      const size = Math.min(isRain ? 34 : 20, hpx * 0.72);
      if (size < 7 || x0 < -200) continue;
      const shown = clamp((vis * (rows + 6) - r) / 6);
      c.globalAlpha = shown * (isRain ? 1 : 0.8);
      c.font = font(F.mono(isRain ? 700 : 500), size);
      c.fillStyle = isRain && t > T.v1c! ? PINK : INK;
      c.fillText(tokLabel(e.table.t[r]!).replace(/\n/g, '↵'), x0, y0);
    }
    c.restore();
    // the wall's name, while it arrives
    const nm = clamp((t - T.v1a!) / 0.8) * (1 - clamp((t - T.v1c!) / 0.8));
    if (nm > 0) {
      c.save(); c.globalAlpha = nm; c.fillStyle = INK; c.textAlign = 'center';
      c.font = font(F.archivo(100, 800), portrait() ? 34 : 34);
      c.fillText('THE EMBEDDING TABLE', W / 2, H * (portrait() ? 0.17 : 0.2));
      c.font = font(F.archivo(100, 500), 26);
      c.fillText('one row of numbers for every token it knows', W / 2, H * (portrait() ? 0.17 : 0.2) + 38);
      c.restore();
    }
    // "embedding" defined on its line, under the row
    const em = clamp((t - T.v1e!) / 0.5) * (1 - clamp((t - T.v1g!) / 0.8));
    if (em > 0) {
      c.save(); c.globalAlpha = em; c.textAlign = 'center';
      const P = portrait();
      c.font = font(F.archivo(112.5, 900), P ? 64 : 72); c.fillStyle = PINK;
      c.fillText('EMBEDDING', W / 2, H * (P ? 0.3 : 0.29));
      c.font = font(F.archivo(100, 500), P ? 30 : 30); c.fillStyle = INK;
      c.fillText('a word’s list of numbers', W / 2, H * (P ? 0.3 : 0.29) + (P ? 52 : 56));
      c.restore();
    }
  }

  private labels(c: CanvasRenderingContext2D, t: number, posOf: (tok: string) => THREE.Vector3) {
    const P = portrait();
    type Lb = { tok: string; x: number; y: number; w: number; size: number; k: number; pink: number; draw: number };
    const list: Lb[] = [];
    for (const [tok, t0] of this.labelAt) {
      const draw = clamp((t - t0) / 0.5);
      if (draw <= 0) continue;
      const [x, y, ok] = this.proj(posOf(tok));
      if (!ok || x < -100 || x > W + 100 || y < -60 || y > H + 60) continue;
      const pink = this.pinkOf(tok, t);
      const d = this.camF.position.distanceTo(posOf(tok));
      const raw = (P ? 300 : 260) / d;
      const size = clamp(raw, P ? 15 : 14, P ? 30 : 28) * (1 + 0.25 * pink);
      // far away only the anchor words keep their labels (the rest fade with distance, back when near)
      const near = ANCHORS.has(tok) || pink > 0.05 ? 1 : clamp((raw - 10) / 4);
      if (near <= 0.01) continue;
      c.font = font(F.archivo(100, 700), size);
      list.push({ tok, x, y, w: c.measureText(tokLabel(tok)).width, size, k: near, pink, draw });
    }
    // simple vertical de-overlap: later labels step down past earlier ones
    list.sort((a, b) => a.y - b.y);
    const placed: { x0: number; x1: number; y0: number; y1: number }[] = [];
    c.save();
    c.textBaseline = 'middle';
    for (const l of list) {
      let lx = l.x + 14, ly = l.y - 12;
      for (let it = 0; it < 12; it++) {
        const hit = placed.find((p) => lx < p.x1 && lx + l.w > p.x0 && ly - l.size * 0.55 < p.y1 && ly + l.size * 0.55 > p.y0);
        if (!hit) break;
        ly = hit.y1 + l.size * 0.6;
      }
      placed.push({ x0: lx - 2, x1: lx + l.w + 2, y0: ly - l.size * 0.55, y1: ly + l.size * 0.55 });
      // leader line when pushed away
      if (Math.abs(ly - (l.y - 12)) > 4) {
        c.globalAlpha = l.draw * l.k * 0.6; c.strokeStyle = INK; c.lineWidth = 1;
        c.beginPath(); c.moveTo(l.x + 5, l.y); c.lineTo(lx - 2, ly); c.stroke();
      }
      c.globalAlpha = l.k;
      c.font = font(F.archivo(100, 700), l.size);
      // draw in left to right
      c.save();
      c.beginPath(); c.rect(lx - 4, ly - l.size, (l.w + 8) * ease.outCubic(l.draw), l.size * 2); c.clip();
      if (l.pink > 0.01) { c.fillStyle = PAPER; c.globalAlpha = 0.85 * l.pink; c.fillRect(lx - 4, ly - l.size * 0.62, l.w + 8, l.size * 1.24); c.globalAlpha = l.k; }
      c.fillStyle = l.pink > 0.5 ? PINK : INK;
      c.fillText(tokLabel(l.tok), lx, ly);
      c.restore();
    }
    c.restore();
    // the space-dot legend, once, with the first label
    const T = this.T, lg = clamp((t - T.v1c! - 0.6) / 0.5) * (1 - clamp((t - T.v1e!) / 0.6));
    if (lg > 0) {
      c.save(); c.globalAlpha = lg; c.fillStyle = INK; c.font = font(F.mono(500), P ? 22 : 20);
      c.fillText('· = the space before the word (part of the token)', P ? 60 : 80, H * (P ? 0.78 : 0.84));
      c.restore();
    }
  }

  private axes(c: CanvasRenderingContext2D, t: number) {
    const T = this.T, k = clamp((t - T.c0b!) / 1.2) * (1 - clamp((t - T.v2!) / 1.2)) + clamp((t - T.c1b!) / 1.2) * (1 - clamp((t - T.v3!) / 1.2)) + clamp((t - T.c2b!) / 1.2) * (1 - clamp((t - T.out!) / 1.2));
    if (k <= 0) return;
    const o = V3(0, 0, 0), L = R * 1.15;
    c.save();
    c.globalAlpha = clamp(k) * 0.8;
    c.strokeStyle = INK; c.lineWidth = 1.4; c.setLineDash([6, 6]);
    const names = ['1st', '2nd', '3rd'];
    [V3(1, 0, 0), V3(0, 1, 0), V3(0, 0, 1)].forEach((d, i) => {
      const [x0, y0, a] = this.proj(o.clone().addScaledVector(d, -L)), [x1, y1, b] = this.proj(o.clone().addScaledVector(d, L));
      if (!a || !b) return;
      const g = ease.outCubic(clamp(k * 1.2 - i * 0.1));
      c.beginPath(); c.moveTo(lerp((x0 + x1) / 2, x0, g), lerp((y0 + y1) / 2, y0, g)); c.lineTo(lerp((x0 + x1) / 2, x1, g), lerp((y0 + y1) / 2, y1, g)); c.stroke();
      c.setLineDash([]);
      c.fillStyle = INK; c.font = font(F.mono(600), 15);
      c.fillText(`${names[i]} direction`, x1 + 6, y1 - 6);
      c.setLineDash([6, 6]);
    });
    c.restore();
  }

  private cosInset(c: CanvasRenderingContext2D, t: number) {
    const T = this.T, k = clamp((t - T.v2! + 0.2) / 0.6) * (1 - clamp((t - T.v2c!) / 0.6));
    if (k <= 0) return;
    const P = portrait();
    const bw = P ? 860 : 640, bh = P ? 420 : 380;
    const x0 = P ? (W - bw) / 2 : W * 0.07, y0 = P ? H * 0.645 : H * 0.36;
    const ox = x0 + 70, oy = y0 + bh - 50, Rr = P ? 270 : 225;
    // the pink list swings: a middling angle, then "the same" (0 deg), then "unrelated" (90 deg), then it settles
    // at the real angle of two random words, arccos(0.038)
    const deg = Math.PI / 180, rnd = Math.acos(this.emb.baseline.mean_cos);
    const keysA: [number, number][] = [[T.v2!, 50 * deg], [T.v2b! - 0.8, 35 * deg], [T.v2same!, 2 * deg], [T.v2zero! - 0.3, 2 * deg], [T.v2zero! + 0.4, 90 * deg], [T.v2zero! + 1.6, rnd]];
    let ang = keysA[0]![1];
    for (let i = 0; i < keysA.length - 1; i++) {
      const [ta, a] = keysA[i]!, [tb, b] = keysA[i + 1]!;
      if (t >= ta) ang = lerp(a, b, sm((t - ta) / Math.max(0.2, tb - ta)));
    }
    c.save();
    c.globalAlpha = k * 0.92; c.fillStyle = PAPER; c.fillRect(x0, y0, bw, bh);
    c.globalAlpha = k;
    c.strokeStyle = INK; c.lineWidth = 2; c.strokeRect(x0, y0, bw, bh);
    c.fillStyle = INK; c.font = font(F.archivo(100, 800), P ? 30 : 27);
    c.fillText('close = two lists point the same way', x0 + 26, y0 + 46);
    c.font = font(F.archivo(100, 500), P ? 24 : 21);
    c.fillText('(the cosine of the angle between them)', x0 + 26, y0 + 78);
    this.arrow2D(c, ox, oy, ox + Rr, oy, BLUE, 5);
    this.arrow2D(c, ox, oy, ox + Math.cos(ang) * Rr, oy - Math.sin(ang) * Rr, PINK, 5);
    c.strokeStyle = INK; c.lineWidth = 1.5;
    c.beginPath(); c.arc(ox, oy, 52, -ang, 0); c.stroke();
    c.font = font(F.mono(600), P ? 24 : 21); c.fillStyle = INK;
    c.globalAlpha = k * clamp((t - T.v2same! + 0.2) / 0.4);
    c.fillText('1 = same way', ox + Rr + 16, oy + 7);
    c.globalAlpha = k * clamp((t - T.v2zero! + 0.1) / 0.4);
    c.fillText('0 = at right angles', ox + 14, oy - Rr + 8);
    c.restore();
  }

  private rankPanel(c: CanvasRenderingContext2D, t: number, _id: string, t0: number, t1: number, _q: string, toks: string[], head: string) {
    const k = clamp((t - t0) / 1.0) * (1 - clamp((t - t1) / 1.0));
    if (k <= 0) return;
    const P = portrait(), T = this.T;
    const x = P ? W * 0.1 : W * 0.7, y0 = P ? H * 0.715 : H * 0.28, rh = P ? 36 : 38;
    c.save();
    c.globalAlpha = k * 0.9; c.fillStyle = PAPER;
    c.fillRect(x - 24, y0 - 64, (P ? W * 0.8 : W * 0.27), rh * toks.length + 90);
    c.globalAlpha = k;
    c.fillStyle = INK; c.font = font(F.archivo(100, 700), P ? 22 : 19);
    c.fillText(head, x, y0 - 30);
    toks.forEach((tok, i) => {
      const show = clamp((t - t0 - i * 0.12) / 0.3);
      if (show <= 0) return;
      const y = y0 + i * rh;
      // rain: its spellings are pink on "its own spelling", the weather words on "the words that mean the same"
      const spell = /rain/i.test(tok);
      let pink = 0;
      if (toks.length > 6) pink = spell ? clamp((t - T.v2g!) / 0.3) * (1 - clamp((t - T.v2h!) / 0.3)) : clamp((t - T.v2h!) / 0.3) * (1 - clamp((t - T.v2hEnd! - 1) / 0.3));
      else pink = this.pinkOf(tok, t);
      c.globalAlpha = k * show;
      c.font = font(F.mono(600), P ? 24 : 22); c.fillStyle = INK;
      c.fillText(String(i + 1).padStart(2, ' '), x, y);
      c.font = font(F.archivo(100, pink > 0.5 ? 800 : 600), P ? 30 : 28); c.fillStyle = pink > 0.5 ? PINK : INK;
      c.fillText(tokLabel(tok), x + (P ? 56 : 52), y);
    });
    c.restore();
  }

  private number(c: CanvasRenderingContext2D, t: number) {
    const P = portrait();
    for (const n of this.nums) {
      const k = clamp((t - n.t0) / 0.45) * (1 - clamp((t - n.t1 + 0.05) / 0.35));
      if (k <= 0) continue;
      let x = P ? W / 2 : W * 0.5, y = P ? H * 0.57 : H * 0.8, align: CanvasTextAlign = 'center';
      const p = n.at();
      if (p) {
        const [px, py, ok] = this.proj(p);
        if (ok) { x = clamp(px + 60, 80, W - (P ? 330 : 420)); y = clamp(py + 120, H * 0.24, P ? H * 0.64 : H * 0.84); align = 'left'; }
      }
      c.save();
      c.globalAlpha = k;
      c.textAlign = align; c.textBaseline = 'alphabetic';
      const size = P ? 76 : 84;
      c.font = font(F.mono(700), size);
      const wNum = c.measureText(n.text).width;
      c.font = font(F.archivo(100, 600), P ? 26 : 26);
      const wCap = c.measureText(n.cap).width;
      const bw = Math.max(wNum, wCap) + 36, bx = align === 'center' ? x - bw / 2 : x - 18;
      c.fillStyle = PAPER; c.globalAlpha = k * 0.88; c.fillRect(bx, y - size * 0.9, bw, size * 0.9 + 52);
      c.globalAlpha = k;
      c.font = font(F.mono(700), size); c.fillStyle = INK;
      c.fillText(n.text, x, y);
      c.font = font(F.archivo(100, 600), P ? 26 : 26); c.fillStyle = INK;
      c.fillText(n.cap, x, y + 38);
      c.restore();
    }
  }
}
