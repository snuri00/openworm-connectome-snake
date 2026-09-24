"""
Agar-plate world: the connectome-driven soft-bodied worm (worm_animal) crawls
freely on a round plate with a food spot, toxin droplets and touch stimuli.

It exposes the same interface the console uses for the grid game (it plays
both the "game" and the "controller" role), so every console panel works
unchanged. One console step = one decision window of TICKS brain ticks.
"""
import math
import random
from collections import deque

import numpy as np

from worm_animal import DT, WormAnimal
from worm_brain import WormBrain

PLATE_RADIUS = 4.0       # mm
ODOR_LENGTH = 1.0        # mm, food odour C = exp(-d / ODOR_LENGTH)
TOXIN_LENGTH = 0.35      # mm
TOXIN_GAIN = 3.0
TOXIN_LIFETIME = 40.0    # s
TOXIN_CORE = 0.1         # mm, touching this is lethal in lethal mode
EAT_RADIUS = 0.15        # mm
TOUCH_RANGE = 0.12       # mm, the nose feels the plate rim this close
WARMUP_TICKS = 500       # 5 s of crawling before a new worm is placed
TICKS = 6                # brain ticks per decision window (= 60 ms)
STEPS_PER_SECOND = 1.0 / (TICKS * DT)

PIROUETTE_RATE = 0.03    # reversal probability per window per AIB spike
CHEMO_ADAPT_TAU = 1.5    # s; chemosensory responses fade under a sustained change
ODOR_DEADZONE = 0.0      # relative change below this gives no chemosensory drive
ODOR_MAX = 1.5           # cap on chemosensory drive
ODOR_SENSITIVITY = 20.0  # relative odour change -> chemosensory drive (Weber-like)
WEATHERVANE_GAIN = 1.0   # steering per relative lateral odour gradient [1/mm]
WEATHERVANE_MAX = 0.35
WEATHERVANE_WINDOWS = 33 # ~2 s of head sweeps
ODOR_WINDOWS = 8         # chemosensors compare the last ~0.5 s with the 0.5 s before
TURN_THRESHOLD = 100.0   # smoothed |L-R| head muscle drive that starts a turn
TURN_STEER = 0.45        # head bias during a turn (see worm_animal.pacemaker)
OMEGA_STEER = -0.5       # ventral bias of the omega turn after a reversal
OMEGA_PROB = 0.5         # otherwise the reversal ends with a shallow turn
SHALLOW_STEER = 0.3


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


class PlateWorld:
    ticks = TICKS

    def __init__(self, brain=None, seed=None, lethal=False):
        self.brain = brain or WormBrain()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.lethal = lethal
        self.aib = [self.brain.idx["AIBL"], self.brain.idx["AIBR"]]
        self.reset()

    # ------------------------------------------------------------ setup ---
    def reset(self):
        self.brain.reset()
        self.animal = WormAnimal(brain=self.brain, seed=self.rng.randrange(1 << 30))
        # warm-up: let the rhythm and the dorsal/ventral muscle balance settle
        # off-screen, otherwise a fresh worm curls into a C for its first seconds
        for _ in range(WARMUP_TICKS):
            self.animal.step(DT)
        body = self.animal.body
        body.center = np.array([0.0, 0.0])
        body.angle = self.rng.uniform(-math.pi, math.pi)
        self.alive = True
        self.death_cause = None
        self.bumped = None
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.t = 0.0
        self.dir = 0
        self.toxins = []          # [[(x, y), expiry_time]]
        self.pokes = []
        self.trail = []
        self.spike_frames = []
        self.events = []
        self.last = {}
        self.odor_hist = deque(maxlen=2 * ODOR_WINDOWS)
        self.sweep_hist = deque(maxlen=WEATHERVANE_WINDOWS)   # (lateral head offset, odour)
        self.weathervane = 0.0
        self.chemo_adapt = np.zeros(2)          # adaptation of the [rising, falling] channels
        # behavioural state timers (seconds)
        self.reverse_left = 0.0
        self.omega_left = 0.0
        self.omega_steer = OMEGA_STEER
        self.turn_left = 0.0
        self.turn_sign = 0
        self.sprint_left = 0.0
        self.reverse_cooldown = 0.0
        self.diff_avg = 0.0
        self.place_food()

    def place_food(self, pos=None):
        if pos is None:
            r = (PLATE_RADIUS - 1.8) * math.sqrt(self.rng.random())
            a = self.rng.uniform(-math.pi, math.pi)
            pos = (r * math.cos(a), r * math.sin(a))
            if math.dist(pos, self.head) < 1.5:
                return self.place_food()
        self.food = (float(pos[0]), float(pos[1]))

    def add_toxin(self, pos):
        if math.hypot(*pos) < PLATE_RADIUS:
            self.toxins.append([(float(pos[0]), float(pos[1])), self.t + TOXIN_LIFETIME])

    def poke(self, kind, strength=1.3, steps=3):
        self.pokes.append([kind, strength, steps])

    # ------------------------------------------------------------ queries --
    @property
    def body(self):
        return [tuple(p) for p in self.animal.body.nodes()]

    @property
    def head(self):
        return tuple(self.animal.body.nodes()[0])

    def heading_vec(self):
        p = self.animal.body.nodes()
        d = p[0] - p[2]
        return d / (np.linalg.norm(d) + 1e-12)

    def odor_at(self, pos):
        return math.exp(-math.dist(pos, self.food) / ODOR_LENGTH)

    def toxin_at(self, pos):
        return sum(math.exp(-math.dist(pos, t[0]) / TOXIN_LENGTH) for t in self.toxins)

    def body_region(self, pos, radius=0.12):
        """Which touch receptors a poke at pos (mm) would hit, or None."""
        nodes = self.animal.body.nodes()
        d = np.linalg.norm(nodes - np.asarray(pos), axis=1)
        i = int(np.argmin(d))
        if d[i] > radius:
            return None
        if i <= 1:
            return "touch"
        return "body_anterior" if i / (len(nodes) - 1) <= 0.5 else "body_posterior"

    # ------------------------------------------------------------ sensing --
    def sense(self):
        head = np.array(self.head)
        h = self.heading_vec()
        # plate rim: nose touch, side from where the rim is relative to the heading
        r = np.linalg.norm(head)
        touch_l = touch_r = 0.0
        self.bumped = None
        if r > PLATE_RADIUS - TOUCH_RANGE:
            n = head / r
            near = min(1.0, (r - (PLATE_RADIUS - TOUCH_RANGE)) / TOUCH_RANGE + 0.3)
            ahead = max(0.0, float(np.dot(h, n)))
            side = cross(h, n)                      # >0: rim on the left
            touch_l = near * (0.6 * ahead + max(0.0, side))
            touch_r = near * (0.6 * ahead + max(0.0, -side))
            self.bumped = "plate rim"

        odor = self.odor_at(head)
        # ASE/AWC integrate concentration over time, so the side-to-side head
        # sweep does not read as "odour falling" on every stroke
        self.odor_hist.append(odor)
        rel = 0.0
        if len(self.odor_hist) == self.odor_hist.maxlen:
            h = list(self.odor_hist)
            before, now = np.mean(h[:ODOR_WINDOWS]), np.mean(h[ODOR_WINDOWS:])
            # Weber-like: the response scales with the relative change
            rel = (now - before) / (before + 1e-3)
        # chemosensory neurons respond to clear changes only; without this dead
        # zone they are driven all the time and the neck muscles never relax
        food_up = min(ODOR_MAX, ODOR_SENSITIVITY * (rel - ODOR_DEADZONE)) if rel > ODOR_DEADZONE else 0.0
        food_down = min(ODOR_MAX, ODOR_SENSITIVITY * (-rel - ODOR_DEADZONE)) if rel < -ODOR_DEADZONE else 0.0
        if CHEMO_ADAPT_TAU:
            # sensory adaptation (e.g. cGMP signalling in AWC/ASE): a sustained
            # change gives a strong transient that fades within ~tau seconds
            raw = np.array([food_up, food_down])
            food_up, food_down = np.clip(raw - self.chemo_adapt, 0, None)
            self.chemo_adapt += (raw - self.chemo_adapt) * min(1.0, TICKS * DT / CHEMO_ADAPT_TAU)
            food_up, food_down = float(food_up), float(food_down)

        # weathervaning: while the head sweeps side to side, compare the odour
        # on either side of the body axis and lean towards the richer side
        nodes = self.animal.body.nodes()
        axis = nodes[4] - nodes[10]
        axis /= np.linalg.norm(axis) + 1e-12
        lateral = cross(axis, nodes[0] - nodes[4])       # >0: head is to the left
        self.sweep_hist.append((lateral, odor))
        self.weathervane = 0.0
        if len(self.sweep_hist) == self.sweep_hist.maxlen:
            lat, c = np.array(self.sweep_hist).T
            var = lat.var()
            if var > 1e-6:
                slope = ((lat - lat.mean()) * (c - c.mean())).mean() / var
                grad = slope / (c.mean() + 1e-3)             # relative gradient, left positive
                # positive steer turns the worm right (clockwise), so leaning left is negative
                self.weathervane = float(np.clip(-WEATHERVANE_GAIN * grad,
                                                 -WEATHERVANE_MAX, WEATHERVANE_MAX))

        tox = min(2.0, TOXIN_GAIN * self.toxin_at(head))
        tox_l = tox_r = 0.0
        if tox > 0.05:
            side = 0.0
            for (tx, ty), _ in self.toxins:
                v = np.array([tx, ty]) - head
                side += cross(h, v) / (np.linalg.norm(v) + 1e-9)
            side = max(-1.0, min(1.0, side))           # >0: toxin on the left
            tox_l = tox * (1.0 + 0.3 * side)
            tox_r = tox * (1.0 - 0.3 * side)
        return dict(touch=(touch_l, touch_r), odor=odor, food=(food_up, food_down),
                    toxin=(tox_l, tox_r))

    # ------------------------------------------------------------ step -----
    def decide(self, _game=None):
        """One decision window: sense, run TICKS brain+body steps, act."""
        b = self.brain
        a = self.animal
        s = self.sense()
        touch_l, touch_r = s["touch"]
        food_up, food_down = s["food"]
        tox_l, tox_r = s["toxin"]
        pokes = [p for p in self.pokes if p[2] > 0]

        L = R = F = B = 0.0
        aib = 0
        self.spike_frames = []
        for t in range(TICKS):
            if t % 2 == 0:
                if touch_l or touch_r:
                    b.stimulate("touch", touch_l, touch_r)
                if food_up:
                    b.stimulate("food_up", food_up, food_up)
                if food_down:
                    b.stimulate("food_down", food_down, food_down)
            if tox_l or tox_r:
                b.stimulate("noxious", tox_l, tox_r)
            for kind, strength, _ in pokes:
                b.stimulate(kind, strength, strength)
            out = a.step(DT)
            self.keep_on_plate()
            L += out["head_L"]; R += out["head_R"]
            F += out["forward"]; B += out["reverse"]
            self.spike_frames.append(b.fired.copy())
            aib += int(b.fired[self.aib].sum())
        for p in pokes:
            p[2] -= 1
        self.pokes = [p for p in self.pokes if p[2] > 0]
        window = TICKS * DT
        self.t += window

        # ---- behaviour from the command interneurons and head muscles
        self.events = []
        reverse = sprint_start = False
        self.reverse_cooldown = max(0.0, self.reverse_cooldown - window)
        if B >= 6 and B > 2 * F + 2 and self.reverse_left <= 0 and self.reverse_cooldown <= 0:
            reverse = True
            self.reverse_left = 1.5
            self.reverse_cooldown = 3.0
            self.events.append(("reverse", f"REVERSE CMD  AVA/AVD/AVE x{B:.0f} > AVB/PVC x{F:.0f}"))
        elif (aib and self.reverse_left <= 0 and self.reverse_cooldown <= 0
              and self.omega_left <= 0 and self.np_rng.random() < PIROUETTE_RATE * aib):
            # klinokinesis: falling odour excites AWC/ASER -> AIB, and AIB
            # activity raises the chance of a reversal ("pirouette")
            reverse = True
            self.reverse_left = 1.5
            self.reverse_cooldown = 3.0
            self.events.append(("reverse", f"PIROUETTE  AIB x{aib} (odour falling)"))
        elif F >= 4 and F > B and self.reverse_left <= 0:
            if self.sprint_left <= 0:
                sprint_start = True
                self.events.append(("sprint", f"FORWARD ESCAPE  AVB/PVC x{F:.0f}"))
            self.sprint_left = 1.5

        turn = 0
        diff = L - R
        # the rhythm itself causes brief left/right blips; a reflex is sustained
        self.diff_avg = 0.5 * self.diff_avg + 0.5 * diff
        if abs(self.diff_avg) > TURN_THRESHOLD and self.reverse_left <= 0 and self.omega_left <= 0:
            # the nose-touch reflex contracts the same-side head muscles; the
            # worm bends away from that side (same convention as the grid game)
            turn = 1 if self.diff_avg > 0 else -1
            self.turn_sign = turn
            self.turn_left = 0.6

        # ---- apply to the body
        if self.reverse_left > 0:
            a.direction = -1
            a.steer = 0.0
            self.reverse_left -= window
            if self.reverse_left <= 0:
                # a reversal ends in a deep ventral omega turn or a shallow turn
                if self.np_rng.random() < OMEGA_PROB:
                    self.omega_steer, self.omega_left = OMEGA_STEER, 1.2
                else:
                    self.omega_steer = SHALLOW_STEER * (1 if self.np_rng.random() < 0.5 else -1)
                    self.omega_left = 0.8
        else:
            a.direction = 1
            if self.omega_left > 0:
                a.steer = self.omega_steer
                self.omega_left -= window
            elif self.turn_left > 0:
                a.steer = TURN_STEER * self.turn_sign
                self.turn_left -= window
            else:
                a.steer = self.weathervane
        a.cpg_freq = 0.8 if self.sprint_left > 0 else 0.5
        self.sprint_left = max(0.0, self.sprint_left - window)

        # ---- world events
        head = self.head
        self.steps += 1
        self.steps_since_food += 1
        self.trail.append(head)
        if len(self.trail) > 400:
            self.trail.pop(0)
        if math.dist(head, self.food) < EAT_RADIUS:
            self.score += 1
            self.steps_since_food = 0
            self.place_food()
        self.toxins = [tx for tx in self.toxins if tx[1] > self.t]
        if self.lethal and any(math.dist(head, tx[0]) < TOXIN_CORE for tx in self.toxins):
            self.alive = False
            self.death_cause = "toxin"

        self.last = dict(L=L, R=R, F=F, B=B,
                         asym=diff / (L + R) if L + R > 1e-6 else 0.0,
                         turn=turn, reverse=reverse or self.reverse_left > 0,
                         sprint=self.sprint_left > 0, touch=s["touch"], food=s["food"],
                         odor=s["odor"], toxin=s["toxin"], state=self.state(),
                         weathervane=self.weathervane)
        return self.last

    def state(self):
        if self.reverse_left > 0:
            return "reversal"
        if self.omega_left > 0:
            return "omega turn"
        if self.turn_left > 0:
            return "turning"
        if self.sprint_left > 0:
            return "forward run"
        return "forward"

    def act(self, _game=None):
        self.decide()
        return 1

    def step(self, _turn=0):
        """Manual mode is not meaningful on the plate; just advance."""
        self.decide()

    def keep_on_plate(self):
        body = self.animal.body
        nodes = body.nodes()
        r = np.linalg.norm(nodes, axis=1)
        over = r.max() - PLATE_RADIUS
        if over > 0:
            i = int(np.argmax(r))
            body.center = body.center - nodes[i] / r[i] * over


def run_trials(seconds=120, seed=0, sense_food=True):
    """Headless chemotaxis test: food eaten in `seconds` of simulated time."""
    world = PlateWorld(seed=seed)
    if not sense_food:
        world.odor_at = lambda pos: 0.0
    for _ in range(int(seconds * STEPS_PER_SECOND)):
        world.decide()
    return world.score, world
