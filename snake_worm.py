"""
Snake, steered by the OpenWorm C. elegans connectome.

Run:        python3 snake_worm.py
Benchmark:  python3 snake_worm.py --headless 20

The game logic and the brain <-> game interface live here; the sci-fi HUD
lives in worm_hud.py.
"""
import os

# numpy's BLAS would otherwise spin up one thread per core for tiny arrays
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import math
import random

import numpy as np

from worm_brain import WormBrain

GRID_W, GRID_H = 24, 24
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left (clockwise)

ODOR_LENGTH = 7.0      # cells; food odour falls off as exp(-d / ODOR_LENGTH)
TOXIN_LENGTH = 2.2     # cells; toxin is sensed within a few cells
FOOD_MARGIN = 2        # random food spawns at least this many cells from a wall
TOXIN_GAIN = 3.0       # toxin concentration -> nociceptor drive
TOXIN_LIFETIME = 60    # game steps a dropped toxin stays on the board


class SnakeGame:
    def __init__(self, w=GRID_W, h=GRID_H, rng=None, lethal=True):
        self.w, self.h = w, h
        self.rng = rng or random.Random()
        # lethal=False: a blocked move just bumps the nose and the worm stays put
        self.lethal = lethal
        self.reset()

    def reset(self):
        cx, cy = self.w // 2, self.h // 2
        self.dir = 1
        self.body = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.alive = True
        self.death_cause = None
        self.bumped = None  # what the nose hit on the last step (non-lethal mode)
        self.bumps = 0
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.toxins = []  # [(x, y), expiry_step]
        self.place_food()

    def place_food(self, cell=None):
        if cell is None:
            # keep random food off the walls so its odour field is not cut in half
            m = FOOD_MARGIN
            free = [(x, y) for x in range(m, self.w - m) for y in range(m, self.h - m)
                    if (x, y) not in self.body and not self.is_toxic((x, y))]
            if not free:
                free = [(x, y) for x in range(self.w) for y in range(self.h)
                        if (x, y) not in self.body and not self.is_toxic((x, y))]
            cell = self.rng.choice(free)
        self.food = cell

    def add_toxin(self, cell):
        if cell not in self.body and cell != self.food:
            self.toxins = [t for t in self.toxins if t[0] != cell]
            self.toxins.append([cell, self.steps + TOXIN_LIFETIME])

    def is_toxic(self, cell):
        return any(t[0] == cell for t in self.toxins)

    def in_bounds(self, cell):
        return 0 <= cell[0] < self.w and 0 <= cell[1] < self.h

    def blocked(self, cell):
        return (not self.in_bounds(cell) or cell in self.body[:-1]
                or self.is_toxic(cell))

    def cell_in(self, d, n=1):
        hx, hy = self.body[0]
        dx, dy = DIRS[d % 4]
        return (hx + dx * n, hy + dy * n)

    def odor_at(self, cell):
        """Food odour concentration 0..1 at a cell."""
        d = math.dist(cell, self.food)
        return math.exp(-d / ODOR_LENGTH)

    def toxin_at(self, cell):
        """Toxin concentration at a cell (sum over all toxins)."""
        return sum(math.exp(-math.dist(cell, t[0]) / TOXIN_LENGTH) for t in self.toxins)

    def reverse(self):
        """Crawl backwards: the tail becomes the head."""
        self.body.reverse()
        (hx, hy), (nx, ny) = self.body[0], self.body[1]
        self.dir = DIRS.index((hx - nx, hy - ny))

    def step(self, turn):
        """turn: -1 left, 0 straight, +1 right (relative to heading)."""
        self.toxins = [t for t in self.toxins if t[1] > self.steps]
        self.dir = (self.dir + turn) % 4
        new = self.cell_in(self.dir)
        self.steps += 1
        self.steps_since_food += 1
        self.bumped = None
        if self.blocked(new):
            cause = ("toxin" if self.is_toxic(new) else
                     "wall" if not self.in_bounds(new) else "own body")
            if self.lethal:
                self.alive = False
                self.death_cause = cause
            else:
                self.bumped = cause
                self.bumps += 1
            return
        self.body.insert(0, new)
        if new == self.food:
            self.score += 1
            self.steps_since_food = 0
            if len(self.body) >= self.w * self.h:
                self.alive = False
                return
            self.place_food()
        else:
            self.body.pop()


class WormController:
    """Turns game state into sensory input and motor output into actions.

    Sensing:
      * obstacle ahead/left/right -> nose-touch neurons (FLP, ASH, OLQ, IL1)
      * food odour rising / falling -> ASEL, AWA / ASER, AWC; the signal is
        stronger where the odour is stronger (near the food)
      * toxin nearby               -> nociceptors ASH, ADL, ASK
      * user pokes the body        -> ALM/AVM (front) or PLM/PVM (back)
    Acting:
      * head muscle activity above `min_drive` -> bend, i.e. turn; left vs right
        head muscles decide the direction
      * reverse command (AVA, AVD, AVE) dominating -> crawl backwards
      * forward command (AVB, PVC) dominating      -> sprint forwards
    """

    def __init__(self, brain, ticks_per_step=6, noise=3.0, min_drive=50.0,
                 ahead_weight=0.6, turn_away=True, seed=None):
        self.brain = brain
        self.ticks = ticks_per_step
        self.noise = noise
        self.min_drive = min_drive
        self.ahead_weight = ahead_weight
        # The worm lies on its side and bends dorso-ventrally, so mapping its
        # left/right onto the game's left/right is a convention. With this
        # setting the nose-touch reflex (IL1 -> same-side head muscles) steers
        # the snake away from the obstacle.
        self.turn_away = turn_away
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.brain.reset()
        self.prev_odor = None
        self.pokes = []           # [sensor kind, strength, steps left]
        self.reverse_cooldown = 0
        self.sprint = 0
        self.spike_frames = []    # fired vectors of the last step, for the HUD
        self.last = {}
        self.events = []          # (kind, text) produced during the last step

    def poke(self, kind, strength=1.3, steps=3):
        self.pokes.append([kind, strength, steps])

    def sense(self, game):
        d = game.dir
        # The nose feels 1-2 cells ahead (closer = stronger) and both flanks.
        # "Ahead" presses both sides of the nose equally, so it is kept weaker
        # than a flank contact - in a corner the blocked flank still dominates.
        ahead = self.ahead_weight * (1.0 if game.blocked(game.cell_in(d, 1)) else (
            0.4 if game.blocked(game.cell_in(d, 2)) else 0.0))
        left = 1.0 if game.blocked(game.cell_in(d - 1)) else 0.0
        right = 1.0 if game.blocked(game.cell_in(d + 1)) else 0.0

        head = game.body[0]
        odor = game.odor_at(head)
        change = 0.0 if self.prev_odor is None else odor - self.prev_odor
        self.prev_odor = odor
        # ASE responds to relative change; a stronger odour gives a stronger response
        gain = 0.6 + 0.9 * odor
        food_up = gain if change > 1e-9 else 0.0
        food_down = gain if change < -1e-9 else 0.0

        # toxin: intensity from concentration, side from its bearing
        tox = min(2.0, TOXIN_GAIN * game.toxin_at(head))
        tox_l = tox_r = 0.0
        if tox > 0.05:
            hx, hy = head
            fx, fy = DIRS[d]
            side = 0.0
            for (cx, cy), _ in game.toxins:
                vx, vy = cx - hx, cy - hy
                n = math.hypot(vx, vy) or 1.0
                side += (fx * vy - fy * vx) / n  # >0: toxin is to the right
            side = max(-1.0, min(1.0, side))
            tox_l = tox * (1.0 - 0.3 * side)
            tox_r = tox * (1.0 + 0.3 * side)

        return dict(touch=(ahead + left, ahead + right), odor=odor,
                    food=(food_up, food_down), toxin=(tox_l, tox_r))

    def decide(self, game):
        b = self.brain
        s = self.sense(game)
        touch_l, touch_r = s["touch"]
        food_up, food_down = s["food"]
        tox_l, tox_r = s["toxin"]
        pokes = [p for p in self.pokes if p[2] > 0]

        L = R = F = B = 0.0
        self.spike_frames = []
        for t in range(self.ticks):
            if t % 2 == 0:
                if touch_l or touch_r:
                    b.stimulate("touch", touch_l, touch_r)
                if food_up:
                    b.stimulate("food_up", food_up, food_up)
                if food_down:
                    b.stimulate("food_down", food_down, food_down)
            if tox_l or tox_r:
                # nociceptors keep firing as long as the toxin is there
                b.stimulate("noxious", tox_l, tox_r)
            for kind, strength, _ in pokes:
                b.stimulate(kind, strength, strength)
            b.add_noise(self.noise, self.rng)
            out = b.tick()
            L += out["head_L"]; R += out["head_R"]
            F += out["forward"]; B += out["reverse"]
            self.spike_frames.append(b.fired.copy())
        for p in pokes:
            p[2] -= 1
        self.pokes = [p for p in self.pokes if p[2] > 0]

        self.events = []
        total = L + R
        asym = (L - R) / total if total > 1e-6 else 0.0
        turn = 0
        if total > self.min_drive:
            contracted = -1 if asym > 0 else 1  # -1: left side contracted more
            turn = -contracted if self.turn_away else contracted

        reverse = False
        self.reverse_cooldown = max(0, self.reverse_cooldown - 1)
        if B >= 6 and B > 2 * F + 2 and self.reverse_cooldown == 0:
            reverse = True
            turn = 0
            self.reverse_cooldown = 8
            self.events.append(("reverse", f"REVERSE CMD  AVA/AVD/AVE x{B:.0f} > AVB/PVC x{F:.0f}"))
        elif F >= 4 and F > B:
            if self.sprint == 0:
                self.events.append(("sprint", f"FORWARD ESCAPE  AVB/PVC x{F:.0f}"))
            self.sprint = 5
        else:
            self.sprint = max(0, self.sprint - 1)

        self.last = dict(L=L, R=R, F=F, B=B, asym=asym, turn=turn,
                         reverse=reverse, sprint=self.sprint > 0,
                         touch=s["touch"], food=s["food"], odor=s["odor"],
                         toxin=s["toxin"])
        return self.last

    def act(self, game):
        """Run one brain step and apply it. Returns the number of moves made."""
        out = self.decide(game)
        if out["reverse"]:
            game.reverse()
            self.prev_odor = None
        game.step(out["turn"])
        moves = 1
        if out["sprint"] and game.alive:
            # the forward escape run: an extra move straight ahead
            game.step(0)
            moves += 1
        return moves


def play(ctrl, game, max_steps=3000, starve=400):
    ctrl.reset()
    while game.alive and game.steps < max_steps and game.steps_since_food < starve:
        ctrl.act(game)
    return game.score, game.steps


class RandomController:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, game):
        game.step(self.rng.choice([-1, 0, 0, 0, 1]))


def run_headless(games, seed=0, **params):
    rng = random.Random(seed)
    seeds = [rng.random() for _ in range(games)]
    results = {}
    for name, ctrl in [("connectome", WormController(WormBrain(), seed=seed, **params)),
                       ("random", RandomController(seed))]:
        runs = [play(ctrl, SnakeGame(rng=random.Random(s))) for s in seeds]
        scores = [r[0] for r in runs]
        steps = [r[1] for r in runs]
        results[name] = np.mean(scores)
        print(f"[{name:10s}] {games} games | mean score {np.mean(scores):.2f} "
              f"(max {max(scores)}) | mean steps {np.mean(steps):.0f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=int, metavar="N", help="play N games without a window and print stats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=3.0, help="game steps per second")
    args = ap.parse_args()
    if args.headless:
        run_headless(args.headless, args.seed)
    else:
        from worm_hud import run
        run(args.speed)
