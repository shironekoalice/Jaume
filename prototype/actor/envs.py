"""Two tiny decision environments for Jaume-Actor prototyping.

Both are procedural, deterministic under a seed, and have **exact ground truth**,
so a score is never ambiguous.

1. `UIScreen` — a fake browser page. An instruction names an element by
   color + shape + position band. The agent must pick the right element among
   N distractors that share some attributes. Ground truth = candidate index.
   This is the "browser control" side: element grounding + discrete choice.

2. `Maze` — an 8x8 grid world rendered as an image. The agent sees the frame
   and must emit one of {up,down,left,right} each tick until it reaches the
   goal. Ground truth at every state = the first move of a BFS shortest path.
   This is the "game" side: closed-loop control from pixels at a fixed tick.

No downloads, no fonts, pure Pillow.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- UI screens

COLORS = {
    "red": (214, 69, 65),
    "green": (60, 160, 90),
    "blue": (60, 110, 200),
    "gray": (140, 140, 145),
}
SHAPES = ("circle", "square", "triangle")
BANDS = ("top", "middle", "bottom")  # vertical thirds
COLS = ("left", "right")

IMG = 128
CROP = 32


@dataclass
class Candidate:
    crop: np.ndarray  # (3, CROP, CROP) float32 in [0,1]
    box: np.ndarray  # (4,) x0,y0,x1,y1 normalised
    color: str
    shape: str
    cx: float
    cy: float


@dataclass
class UITask:
    screen: np.ndarray  # (3, IMG, IMG) float32 in [0,1]
    candidates: list[Candidate]
    instruction: list[int]  # word ids
    answer: int


def _draw_shape(d: ImageDraw.ImageDraw, shape: str, box, color):
    x0, y0, x1, y1 = box
    if shape == "circle":
        d.ellipse(box, fill=color, outline=(20, 20, 20))
    elif shape == "square":
        d.rectangle(box, fill=color, outline=(20, 20, 20))
    else:
        d.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=color, outline=(20, 20, 20))


def make_ui_task(rng: random.Random, n_cand: int = 6, with_screen: bool = True) -> UITask:
    """Render a screen, draw n_cand elements, and ask for one of them."""
    im = Image.new("RGB", (IMG, IMG), (248, 248, 246))
    d = ImageDraw.Draw(im)
    # page furniture: header bar + a couple of fake text rows
    d.rectangle([0, 0, IMG, 14], fill=(228, 230, 236))
    for i in range(3):
        y = 22 + i * 12
        d.rectangle([8, y, 8 + rng.randint(40, 110), y + 4], fill=(205, 208, 214))

    # choose cells on a 3x2 lattice so elements never overlap
    cells = [(c, r) for r in range(3) for c in range(2)]
    rng.shuffle(cells)
    cells = cells[:n_cand]

    cands: list[Candidate] = []
    used = set()
    for (c, r) in cells:
        # avoid duplicate (color, shape, band, col) combos -> instruction stays unique
        for _ in range(20):
            color = rng.choice(list(COLORS))
            shape = rng.choice(SHAPES)
            if (color, shape) not in used and (color, r, c) not in used:
                break
        used.add((color, shape))
        used.add((color, r, c))
        cw, ch = 22, 22
        x0 = 10 + c * 58 + rng.randint(0, 6)
        y0 = 20 + r * 36 + rng.randint(0, 4)
        box = [x0, y0, x0 + cw, y0 + ch]
        _draw_shape(d, shape, box, COLORS[color])
        crop = np.asarray(im.crop(box).resize((CROP, CROP)), dtype=np.float32) / 255.0
        cands.append(
            Candidate(
                crop=crop.transpose(2, 0, 1),
                box=np.array([box[0] / IMG, box[1] / IMG, box[2] / IMG, box[3] / IMG], dtype=np.float32),
                color=color,
                shape=shape,
                cx=(box[0] + box[2]) / 2 / IMG,
                cy=(box[1] + box[3]) / 2 / IMG,
            )
        )

    ans = rng.randrange(len(cands))
    tgt = cands[ans]
    # instruction = color + shape + band (+ column), enough to be unique
    band = BANDS[min(2, int(tgt.cy * 3))]
    col = COLS[0] if tgt.cx < 0.5 else COLS[1]
    words = ["click", tgt.color, tgt.shape, "at", band, col]
    instruction = [word_id(w) for w in words]

    screen = np.asarray(im, dtype=np.float32) / 255.0
    if not with_screen:
        screen = np.zeros_like(screen)
    return UITask(screen=screen.transpose(2, 0, 1), candidates=cands, instruction=instruction, answer=ans)


VOCAB = ["click", "red", "green", "blue", "gray", "circle", "square", "triangle",
         "at", "top", "middle", "bottom", "left", "right"]
_VID = {w: i + 1 for i, w in enumerate(VOCAB)}
N_WORDS = len(VOCAB) + 1


def word_id(w: str) -> int:
    return _VID.get(w, 0)


def heuristic_ui(task: UITask) -> int:
    """Hand-written baseline: match color first, then shape. No position use.

    This is the 'greedy is already strong' control from the planning report —
    a learned scorer has to beat this, not random.
    """
    words = [VOCAB[i - 1] if 0 < i <= len(VOCAB) else "" for i in task.instruction]
    color = next((w for w in words if w in COLORS), None)
    shape = next((w for w in words if w in SHAPES), None)
    best, best_key = 0, -1.0
    for i, c in enumerate(task.candidates):
        key = 2.0 * (c.color == color) + 1.0 * (c.shape == shape)
        if key > best_key:
            best, best_key = i, key
    return best


# -------------------------------------------------------------------- maze

MAZE = 8


@dataclass
class MazeState:
    agent: tuple[int, int]
    goal: tuple[int, int]
    walls: np.ndarray  # (MAZE, MAZE) bool
    frame: np.ndarray  # (3, IMG, IMG)


def make_maze(rng: random.Random, p_wall: float = 0.22, randomize: bool = True) -> MazeState:
    """Build a solvable maze.

    `randomize=True` places the agent and the goal at random free cells.
    This matters: with fixed start/goal the position labels are constant, so
    a localisation probe can reach 1.000 by learning a constant and every
    downstream measurement is meaningless. Randomised endpoints also make
    the closed-loop task non-trivial (the goal is not always bottom-right).
    """
    while True:
        walls = np.zeros((MAZE, MAZE), dtype=bool)
        for r in range(MAZE):
            for c in range(MAZE):
                if 0 < r < MAZE - 1 and 0 < c < MAZE - 1 and rng.random() < p_wall:
                    walls[r, c] = True
        if randomize:
            free = [(r, c) for r in range(MAZE) for c in range(MAZE) if not walls[r, c]]
            start, goal = rng.sample(free, 2)
        else:
            start, goal = (1, 1), (MAZE - 2, MAZE - 2)
        walls[start] = False
        walls[goal] = False
        if bfs_dist(walls, start, goal) is not None:
            return MazeState(agent=start, goal=goal, walls=walls, frame=render_maze(start, goal, walls))


STEPS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up


def bfs_dist(walls, start, goal, size: int = None):
    """Shortest-path distance on the grid. `size` defaults to the maze edge
    length inferred from the walls array, so it also works for the 12x12
    generalisation domain."""
    size = size or walls.shape[0]
    if start == goal:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), d = q.popleft()
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if walls[nr, nc] or (nr, nc) in seen:
                continue
            if (nr, nc) == goal:
                return d + 1
            seen.add((nr, nc))
            q.append(((nr, nc), d + 1))
    return None


def best_action(walls, start, goal, rng=None) -> int:
    """First move of a BFS shortest path.

    With `rng` given, ties among equally-short moves are broken randomly.
    This matters: with a fixed action order the oracle's action distribution
    collapses onto right/down (97% of frames), the label carries almost no
    information about which of the *equally optimal* moves to take, and the
    policy never learns to disambiguate symmetric cases.
    """
    cands = []
    for a, (dr, dc) in enumerate(STEPS):
        nr, nc = start[0] + dr, start[1] + dc
        if not (0 <= nr < MAZE and 0 <= nc < MAZE) or walls[nr, nc]:
            continue
        d = bfs_dist(walls, (nr, nc), goal)
        if d is not None:
            cands.append((d, a))
    if not cands:
        return 0
    best_d = min(c for c, _ in cands)
    tied = [a for c, a in cands if c == best_d]
    if rng is None or len(tied) == 1:
        return tied[0]
    return tied[rng.randrange(len(tied))]


def render_maze(start, goal, walls, size: int = None, palette: str = None) -> np.ndarray:
    """Render the maze.

    The agent and the goal MUST be visually distinct in shape, not only in
    colour. Probe result (probe_localise.py): with a blue circle for the
    agent and a green *square* for the goal, a linear probe on the spatial
    trunk localised the agent at 0.902 but the goal at only 0.098 — the
    network latched onto "the round blue thing" and treated the goal as
    noise. Making the goal a distinctly different shape fixes the
    ambiguity at the source.

    `size` renders a different grid resolution into the same 128x128 canvas
    (used for the generalisation shift D3), and `palette="shift"` recolours
    everything (used for the colour-shift control D5).
    """
    size = size or MAZE
    cell = IMG // size
    pal = {"agent": (45, 85, 215), "goal": (40, 175, 85), "wall": (70, 74, 84)}
    if palette == "shift":
        pal = {"agent": (215, 85, 45), "goal": (175, 140, 40), "wall": (60, 84, 74)}
    im = Image.new("RGB", (IMG, IMG), (250, 250, 252))
    d = ImageDraw.Draw(im)
    for r in range(size):
        for c in range(size):
            box = [c * cell, r * cell, c * cell + cell - 1, r * cell + cell - 1]
            if walls[r, c]:
                d.rectangle(box, fill=pal["wall"])
            else:
                d.rectangle(box, outline=(226, 228, 234))
    # goal: a DIAMOND (distinct silhouette from the agent's circle)
    gr, gc = goal
    m = max(2, cell // 4)
    cx, cy = gc * cell + cell // 2, gr * cell + cell // 2
    d.polygon([(cx, cy - m), (cx + m, cy), (cx, cy + m), (cx - m, cy)], fill=pal["goal"])
    # agent: a CIRCLE
    ar, ac = start
    m2 = max(2, cell // 3)
    d.ellipse([ac * cell + m2, ar * cell + m2, ac * cell + cell - m2, ar * cell + cell - m2], fill=pal["agent"])
    return (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)


def step_maze(state: MazeState, action: int) -> tuple[MazeState, bool]:
    dr, dc = STEPS[action]
    nr, nc = state.agent[0] + dr, state.agent[1] + dc
    if not (0 <= nr < MAZE and 0 <= nc < MAZE) or state.walls[nr, nc]:
        nr, nc = state.agent  # bump: stay (counts as a tick, not a failure)
    new = MazeState(agent=(nr, nc), goal=state.goal, walls=state.walls,
                    frame=render_maze((nr, nc), state.goal, state.walls))
    return new, (nr, nc) == state.goal
