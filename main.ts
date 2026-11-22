// Sokoban solver for make.fun's generated levels.
//
// The generator's levels are a one-square-wide hallway with one to four rooms off it.
// A box pushed out of its room can never come back, so a level's push optimum is the sum
// of its rooms' optima, and each room can be solved on its own: the player walks the
// hallway to a room's door, solves that room, walks out, and goes on to the next room.
//
// Finding the rooms: take the floor graph and remove its articulation squares (the
// hallway and the door passages are all articulation squares). What is left falls into
// blocks; a block holding boxes is a room's core. Growing the core over the neighbouring
// non-hallway squares absorbs its door passage and any goal niches. The hallway is the
// player's own row, which no room growth crosses.
//
// Solving a room: A* over pushes. A state is the boxes' squares and the player's square;
// a step pushes one box one square, when the player can walk to the square behind it.
// Squares a box can never leave again without reaching a goal (dead squares, found by
// pushing backwards from the goals) prune the search, and the estimate is a minimum
// matching of boxes to goals over plain grid distances, which never overstates the
// pushes still needed. The search is over pushes alone, so its first solution is
// push-optimal for the room as modelled here.
//
// Everything is deterministic: fixed tie-breaking, budgets counted in expansions,
// never the clock, so two runs of the program write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Set<number>;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#' ? 1 : 0;
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet: new Set(goals), boxes, player };
}

function neighbours(lv: Level, i: number): number[] {
  const x = i % lv.w;
  const y = (i / lv.w) | 0;
  const out: number[] = [];
  for (const d of DIRS) {
    const nx = x + d.dx;
    const ny = y + d.dy;
    if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
    const j = ny * lv.w + nx;
    if (!lv.wall[j]) out.push(j);
  }
  return out;
}

// Articulation squares of the floor graph (Tarjan), then the blocks that remain when
// they are removed.
function blocksAndArticulations(lv: Level): { blocks: number[][]; art: Set<number> } {
  const floors: number[] = [];
  for (let i = 0; i < lv.w * lv.h; i++) if (!lv.wall[i]) floors.push(i);
  const adj = new Map<number, number[]>();
  for (const f of floors) adj.set(f, neighbours(lv, f));
  const disc = new Map<number, number>();
  const low = new Map<number, number>();
  const art = new Set<number>();
  let time = 0;
  const dfs = (u: number, parent: number) => {
    disc.set(u, time);
    low.set(u, time);
    time += 1;
    let children = 0;
    for (const v of adj.get(u)!) {
      if (v === parent) continue;
      if (!disc.has(v)) {
        children += 1;
        dfs(v, u);
        low.set(u, Math.min(low.get(u)!, low.get(v)!));
        if (parent !== -1 && low.get(v)! >= disc.get(u)!) art.add(u);
      } else {
        low.set(u, Math.min(low.get(u)!, disc.get(v)!));
      }
    }
    if (parent === -1 && children > 1) art.add(u);
  };
  dfs(lv.player, -1);
  const seen = new Set<number>();
  const blocks: number[][] = [];
  for (const f of floors) {
    if (art.has(f) || seen.has(f)) continue;
    const block: number[] = [];
    const stack = [f];
    seen.add(f);
    while (stack.length) {
      const u = stack.pop()!;
      block.push(u);
      for (const v of adj.get(u)!) {
        if (!art.has(v) && !seen.has(v)) {
          seen.add(v);
          stack.push(v);
        }
      }
    }
    blocks.push(block);
  }
  return { blocks, art };
}

interface Room {
  area: Set<number>; // squares the room's boxes and player may use (not the doorstep)
  doorstep: number; // the hallway square the player stands on to enter
  doorsteps: number[]; // every hallway square next to the room (usually just the one)
  boxes: number[];
  goals: number[];
}

function findRooms(lv: Level): { rooms: Room[]; spine: Set<number> } {
  const { blocks, art } = blocksAndArticulations(lv);
  const playerRow = (lv.player / lv.w) | 0;
  const spine = new Set<number>();
  for (let x = 0; x < lv.w; x++) {
    const i = playerRow * lv.w + x;
    if (!lv.wall[i]) spine.add(i);
  }
  const boxSet = new Set(lv.boxes);
  const cores = blocks.filter((b) => b.some((c) => boxSet.has(c)));
  const claimed = new Set<number>();
  const rooms: Room[] = [];
  for (const core of cores) {
    const area = new Set(core);
    // Grow over neighbouring squares that are not hallway and not another room's:
    // the door passage and goal niches join the room.
    let grew = true;
    while (grew) {
      grew = false;
      for (const c of [...area]) {
        for (const n of neighbours(lv, c)) {
          if (area.has(n) || spine.has(n) || claimed.has(n)) continue;
          if (cores.some((o) => o !== core && o.includes(n))) continue;
          area.add(n);
          grew = true;
        }
      }
    }
    for (const c of area) claimed.add(c);
    // The doorsteps: the hallway squares next to the room. A room is built with one
    // door, but every candidate is tried when solving and the best is kept.
    const doorsteps: number[] = [];
    for (const c of area) {
      for (const n of neighbours(lv, c)) {
        if (spine.has(n) && !doorsteps.includes(n)) doorsteps.push(n);
      }
    }
    doorsteps.sort((a, b) => a - b);
    rooms.push({
      area,
      doorstep: doorsteps[0] ?? -1,
      doorsteps,
      boxes: lv.boxes.filter((b) => area.has(b)),
      goals: lv.goals.filter((g) => area.has(g)),
    });
  }
  rooms.sort((a, b) => a.doorstep - b.doorstep);
  return { rooms, spine };
}

// Squares a box may sit on and still reach some goal, found by pulling boxes backwards
// from the goals. The box itself stays inside `boxCells`; the player who pulled it may
// stand on any `standCells` square (the doorstep included: the first push of a doorway
// box is made from the hallway). A box on any other square can never finish.
function liveSquares(lv: Level, boxCells: Set<number>, standCells: Set<number>, goals: number[]): Set<number> {
  const live = new Set<number>(goals);
  const queue = [...goals];
  while (queue.length) {
    const b = queue.pop()!;
    const bx = b % lv.w;
    const by = (b / lv.w) | 0;
    for (const d of DIRS) {
      const from = (by - d.dy) * lv.w + (bx - d.dx); // the box came from here
      const stand = (by - 2 * d.dy) * lv.w + (bx - 2 * d.dx); // the player stood here
      if (!boxCells.has(from) || live.has(from)) continue;
      if (!standCells.has(stand)) continue;
      live.add(from);
      queue.push(from);
    }
  }
  return live;
}

// Plain grid distances from every cell of `cells` to the nearest goal, per goal, for the
// matching estimate.
function goalDistances(lv: Level, cells: Set<number>, goal: number): Map<number, number> {
  const dist = new Map<number, number>([[goal, 0]]);
  const queue = [goal];
  while (queue.length) {
    const u = queue.shift()!;
    for (const v of neighbours(lv, u)) {
      if (!cells.has(v) || dist.has(v)) continue;
      dist.set(v, dist.get(u)! + 1);
      queue.push(v);
    }
  }
  return dist;
}

// The player's walk to a target square, avoiding boxes: the move string, or null.
function walkPath(
  lv: Level,
  passable: (i: number) => boolean,
  boxes: Set<number>,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const prev = new Map<number, { at: number; ch: string }>([[from, { at: -1, ch: '' }]]);
  const queue = [from];
  while (queue.length) {
    const u = queue.shift()!;
    const ux = u % lv.w;
    const uy = (u / lv.w) | 0;
    for (const d of DIRS) {
      const v = (uy + d.dy) * lv.w + (ux + d.dx);
      if (v < 0 || v >= lv.w * lv.h || lv.wall[v] || boxes.has(v) || prev.has(v) || !passable(v)) continue;
      prev.set(v, { at: u, ch: d.ch });
      if (v === to) {
        let s = '';
        let cur = v;
        while (prev.get(cur)!.at !== -1) {
          s = prev.get(cur)!.ch + s;
          cur = prev.get(cur)!.at;
        }
        return s;
      }
      queue.push(v);
    }
  }
  return null;
}

// Which squares the player can reach, and one walk string to each.
function reachable(
  lv: Level,
  passable: (i: number) => boolean,
  boxes: Set<number>,
  from: number,
): Map<number, string> {
  const paths = new Map<number, string>([[from, '']]);
  const queue = [from];
  while (queue.length) {
    const u = queue.shift()!;
    const ux = u % lv.w;
    const uy = (u / lv.w) | 0;
    for (const d of DIRS) {
      const v = (uy + d.dy) * lv.w + (ux + d.dx);
      if (v < 0 || v >= lv.w * lv.h || lv.wall[v] || boxes.has(v) || paths.has(v) || !passable(v)) continue;
      paths.set(v, paths.get(u)! + d.ch);
      queue.push(v);
    }
  }
  return paths;
}

interface HeapItem {
  f: number;
  g: number;
  boxes: number[];
  player: number;
  moves: string;
}

class Heap {
  items: HeapItem[] = [];
  push(it: HeapItem) {
    const a = this.items;
    a.push(it);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p].f <= a[i].f) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop(): HeapItem | undefined {
    const a = this.items;
    const top = a[0];
    const last = a.pop()!;
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && a[l].f < a[m].f) m = l;
        if (r < a.length && a[r].f < a[m].f) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
  get size() {
    return this.items.length;
  }
}

// Solves one room push-optimally from the given doorstep. Returns the move string (the
// player starts on the doorstep) or null when this model of the room has no solution.
function solveRoom(lv: Level, room: Room, doorstep: number, budget: number): { moves: string; endPlayer: number; pushes: number } | null {
  const playable = new Set(room.area);
  playable.add(doorstep);
  const passable = (i: number) => playable.has(i);
  const live = liveSquares(lv, room.area, playable, room.goals);
  for (const b of room.boxes) if (!live.has(b)) return null;
  const dists = room.goals.map((g) => goalDistances(lv, room.area, g));
  const estimate = (boxes: number[]): number => {
    // Minimum matching of boxes to goals over grid distances (boxes <= 4: permutations).
    const n = boxes.length;
    const cost: number[][] = boxes.map((b) => dists.map((d) => d.get(b) ?? 1e9));
    let best = Infinity;
    const perm = [...Array(n).keys()];
    const go = (k: number) => {
      if (k === n) {
        let s = 0;
        for (let i = 0; i < n; i++) s += cost[i][perm[i]];
        if (s < best) best = s;
        return;
      }
      for (let i = k; i < n; i++) {
        [perm[k], perm[i]] = [perm[i], perm[k]];
        go(k + 1);
        [perm[k], perm[i]] = [perm[i], perm[k]];
      }
    };
    go(0);
    return best;
  };
  const start = [...room.boxes].sort((a, b) => a - b);
  const keyOf = (p: number, bs: number[]) => String.fromCharCode(p, ...bs);
  const open = new Heap();
  open.push({ f: estimate(start), g: 0, boxes: start, player: doorstep, moves: '' });
  const seen = new Map<string, number>([[keyOf(doorstep, start), 0]]);
  let expansions = 0;
  while (open.size) {
    const cur = open.pop()!;
    if (cur.boxes.every((b) => lv.goalSet.has(b))) return { moves: cur.moves, endPlayer: cur.player, pushes: cur.g };
    if (++expansions > budget) return null;
    const boxSet = new Set(cur.boxes);
    const reach = reachable(lv, passable, boxSet, cur.player);
    for (let bi = 0; bi < cur.boxes.length; bi++) {
      const b = cur.boxes[bi];
      const bx = b % lv.w;
      const by = (b / lv.w) | 0;
      for (const d of DIRS) {
        const stand = (by - d.dy) * lv.w + (bx - d.dx);
        const target = (by + d.dy) * lv.w + (bx + d.dx);
        if (!room.area.has(target) || boxSet.has(target) || lv.wall[target]) continue;
        if (!live.has(target)) continue;
        const walk = reach.get(stand);
        if (walk === undefined) continue;
        const next = [...cur.boxes];
        next[bi] = target;
        next.sort((a, b2) => a - b2);
        const g = cur.g + 1;
        const key = keyOf(b, next);
        if (seen.has(key) && seen.get(key)! <= g) continue;
        seen.set(key, g);
        open.push({ f: g + estimate(next), g, boxes: next, player: b, moves: cur.moves + walk + d.ch });
      }
    }
  }
  return null;
}

function solveLevel(text: string): string {
  const lv = parseLevel(text);
  if (lv.boxes.every((b) => lv.goalSet.has(b))) return '';
  const { rooms } = findRooms(lv);
  const allBoxes = new Set(lv.boxes);
  let player = lv.player;
  let out = '';
  const solvedAreas: Set<number>[] = [];
  for (const room of rooms) {
    if (room.boxes.every((b) => lv.goalSet.has(b))) continue;
    // Try each doorstep of the room and keep the solution with the fewest pushes.
    // Walking never moves a box (walkPath steps around them), so the whole floor is
    // fair to cross, the hallway above all.
    let best: { walk: string; moves: string; endPlayer: number; pushes: number; doorstep: number } | null = null;
    for (const ds of room.doorsteps) {
      const walk = walkPath(lv, (i) => !lv.wall[i], allBoxes, player, ds);
      if (walk === null) continue;
      const solved = solveRoom(lv, room, ds, 3_000_000);
      if (!solved) continue;
      if (!best || solved.pushes < best.pushes) best = { walk, moves: solved.moves, endPlayer: solved.endPlayer, pushes: solved.pushes, doorstep: ds };
    }
    if (!best) return out; // the room would not solve; keep what is solved
    out += best.walk + best.moves;
    room.doorstep = best.doorstep;
    player = best.endPlayer;
    // The room's boxes are on its goals now; update the world view and walk back out.
    for (const b of room.boxes) allBoxes.delete(b);
    for (const g of room.goals) allBoxes.add(g);
    solvedAreas.push(room.area);
    const back = walkPath(lv, (i) => !lv.wall[i], allBoxes, player, room.doorstep);
    if (back === null) return out;
    out += back;
    player = room.doorstep;
  }
  return out;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
