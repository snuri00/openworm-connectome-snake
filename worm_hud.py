"""
Experiment console for the connectome-driven snake, styled after lab
acquisition software: calibrated plots, a spike raster, a stimulus-response
table and CSV export.

Interaction:
  LMB on the worm   gentle touch (nose / anterior body / posterior body)
  LMB on the floor  place a toxin droplet (sensed by the nociceptors)
  RMB               move the food source
  Step-rate slider and buttons below the arena, or the keys:
  SPACE pause   M manual (arrow keys)   + / - rate   K lethal collisions
  R new specimen   E export CSV   ESC quit
"""
import os

# numpy's BLAS would otherwise spin up one thread per core for tiny arrays
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import math
import time
from collections import deque
from datetime import datetime

import numpy as np
import pygame

from neuron_atlas import load_atlas
from snake_worm import (GRID_H, GRID_W, ODOR_LENGTH, TOXIN_LENGTH,
                        SnakeGame, WormController)
from worm_brain import WormBrain

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ layout --
CANVAS = (1600, 960)
CELL = 27
ARENA_PANEL = pygame.Rect(20, 66, 720, 712)
ARENA = pygame.Rect(56, 104, GRID_W * CELL, GRID_H * CELL)         # 648 x 648
WALL = 10                                                          # hatched wall band
CONTROLS = pygame.Rect(20, 788, 720, 52)
READOUT = pygame.Rect(20, 850, 720, 90)
MAP = pygame.Rect(760, 92, 820, 250)
RASTER = pygame.Rect(760, 362, 500, 250)
CIRC = pygame.Rect(1276, 362, 304, 250)
TRACES = pygame.Rect(760, 632, 500, 308)
TRIALS = pygame.Rect(1276, 632, 304, 308)

# ------------------------------------------------------------------ colours --
BG = (17, 19, 23)
PANEL = (24, 27, 32)
BORDER = (56, 61, 70)
GRID = (36, 40, 47)
TEXT = (222, 226, 232)
MUTED = (135, 142, 153)
FAINT = (85, 91, 101)
ACCENT = (86, 156, 240)
ODOUR = (64, 186, 160)
TOXIN = (228, 84, 72)
FORWARD = (112, 196, 96)
REVERSE = (196, 112, 222)
TOUCH = (232, 178, 70)
SENSORY = (96, 170, 232)
INTER = (228, 182, 86)
MOTOR = (214, 110, 160)
TYPE_COLORS = {"sensory": SENSORY, "interneuron": INTER, "motor": MOTOR}

STIMULI = {
    "touch": ("nose touch", "FLP OLQ IL1 ASH"),
    "body_anterior": ("anterior body touch", "ALM AVM"),
    "body_posterior": ("posterior body touch", "PLM PVM"),
    "toxin": ("toxin droplet", "ASH ADL ASK"),
}

KEY_NEURONS = {"ASEL": "ASE", "AWCL": "AWC", "ASHL": "ASH", "ALML": "ALM", "AVM": "AVM",
               "PLML": "PLM", "AIYL": "AIY", "AIBL": "AIB", "AVAL": "AVA", "AVDL": "AVD",
               "AVBL": "AVB", "PVCL": "PVC", "SMDDL": "SMD", "VA06": "VA", "VB06": "VB"}

RASTER_TICKS = 240        # brain ticks shown in the raster (= 40 game steps)
TRACE_SECONDS = 20.0
TRIAL_WINDOW = 36         # ticks after a stimulus in which a response is counted
RATE_MIN, RATE_MAX = 0.5, 20.0   # game steps per second
FPS = 30                         # screen refresh; data changes at most 20 Hz


def font(size, mono=False, bold=False):
    base = "/usr/share/fonts/truetype/dejavu/"
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    if os.path.exists(base + name):
        return pygame.font.Font(base + name, size)
    return pygame.font.SysFont("monospace" if mono else "sans", size, bold=bold)


def mix(c1, c2, k):
    return tuple(int(a + (b - a) * k) for a, b in zip(c1, c2))


def colormap(v):
    """Perceptually ordered dark -> teal -> pale yellow map for concentrations."""
    stops = [(0.0, (22, 26, 34)), (0.35, (24, 70, 88)), (0.7, (44, 150, 138)), (1.0, (214, 226, 150))]
    v = max(0.0, min(1.0, v))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            return mix(ca, cb, (v - a) / (b - a))
    return stops[-1][1]


class Console:
    def __init__(self, speed):
        self.speed = speed
        self.brain = WormBrain()
        self.ctrl = WormController(self.brain)
        self.game = SnakeGame()
        self.canvas = pygame.Surface(CANVAS)
        self.f_xs = font(11)
        self.f_s = font(12)
        self.f_m = font(14)
        self.f_title = font(17, bold=True)
        self.f_mono = font(12, mono=True)
        self.f_mono_m = font(15, mono=True)
        self.f_mono_l = font(20, mono=True, bold=True)
        self._text_cache = {}

        self.exp_id = datetime.now().strftime("EXP-%Y%m%d-%H%M%S")
        self.specimen = 1
        self.paused = False
        self.manual = False
        self.manual_turn = 0
        self.best = 0
        self.sim_time = 0.0
        self.step_acc = 0.0
        self.tick = 0
        self.hover = None
        self.banner = ("", -10.0)          # (text, sim time) shown over the arena
        self.dragging = False
        self.prev_body = list(self.game.body)
        self.step_progress = 1.0
        self.game.lethal = False
        self._build_controls()
        self.markers = []                 # stimulus markers on the arena: [cell, t, color]

        self.recording = []               # one row per game step
        self.trials = []                  # stimulus-response trials
        self.pending = []                 # trials still inside their response window
        self.history = deque()            # (t, odour, toxin, fwd, rev)
        self.log = deque(maxlen=6)

        self._build_atlas()
        self.raster = np.zeros((RASTER_TICKS, len(self.raster_order)), dtype=bool)
        self.raster_stim = deque(maxlen=12)   # (tick, color) stimulus onsets
        self.circuit_level = np.zeros(len(self.brain.circuits))
        self.activity = np.zeros(len(self.brain.names))
        self._odour_food = None
        self._grid = pygame.Surface(ARENA.size, pygame.SRCALPHA)
        for k in range(GRID_W + 1):
            pygame.draw.line(self._grid, (255, 255, 255, 14), (k * CELL, 0), (k * CELL, ARENA.h))
            pygame.draw.line(self._grid, (255, 255, 255, 14), (0, k * CELL), (ARENA.w, k * CELL))
        self.note("session started")

    # ------------------------------------------------------------ setup ---
    def _build_atlas(self):
        atlas = load_atlas()
        b = self.brain
        inner = MAP.inflate(-80, -86).move(10, 6)
        self.map_inner = inner
        # anterior-posterior axis in micrometres; head region is expanded
        self.ap_breaks = (-320.0, -180.0, 430.0)
        self.ap_split = 0.45
        self.map_pts = []
        kinds = {}
        for i, name in enumerate(b.names):
            if not b.is_neuron[i] or name not in atlas:
                continue
            (x, y, z), kind = atlas[name]
            main = kind.split(";")[0].strip()
            kinds[i] = main
            px = inner.x + self.ap_to_u(y) * inner.w
            py = inner.centery + (-z * 0.85 + x * 0.45) / 95 * inner.h * 0.5
            color = TYPE_COLORS.get(main, MUTED)
            self.map_pts.append((i, (int(px), int(py)), color, mix((45, 50, 58), color, 0.35)))

        # raster rows: sensory, inter, motor; each sorted head -> tail
        rank = {"sensory": 0, "interneuron": 1, "motor": 2}
        order = sorted(kinds, key=lambda i: (rank.get(kinds[i], 3), atlas[b.names[i]][0][1]))
        self.raster_order = np.array(order)
        self.raster_bands = []
        for kind in ("sensory", "interneuron", "motor"):
            rows = [r for r, i in enumerate(order) if kinds[i] == kind]
            if rows:
                self.raster_bands.append((kind, rows[0], rows[-1] + 1))
        self.raster_colors = np.array([TYPE_COLORS.get(kinds[i], MUTED) for i in order], dtype=np.uint8)

    def _build_controls(self):
        self.slider = pygame.Rect(CONTROLS.x + 104, CONTROLS.y + 32, 140, 6)
        specs = [
            ("pause", lambda: "Resume" if self.paused else "Pause", self.toggle_pause),
            ("lethal", lambda: "Lethal: " + ("on" if self.game.lethal else "off"),
             self.toggle_lethal),
            ("mode", lambda: "Manual" if self.manual else "Autonomous", self.toggle_manual),
            ("reset", lambda: "New specimen", lambda: self.new_specimen("reset by operator")),
            ("export", lambda: "Export CSV", self.export),
        ]
        widths = [62, 84, 96, 104, 88]
        x = CONTROLS.right - 10 - sum(widths) - 6 * (len(widths) - 1)
        self.buttons = []
        for (key, label, action), w in zip(specs, widths):
            self.buttons.append((pygame.Rect(x, CONTROLS.y + 5, w, 42), label, action))
            x += w + 6

    def rate_to_u(self, rate):
        return math.log(rate / RATE_MIN) / math.log(RATE_MAX / RATE_MIN)

    def set_rate_from_x(self, x):
        u = max(0.0, min(1.0, (x - self.slider.x) / self.slider.w))
        rate = RATE_MIN * (RATE_MAX / RATE_MIN) ** u
        self.speed = round(rate * 4) / 4 if rate < 4 else round(rate * 2) / 2

    def toggle_pause(self):
        self.paused = not self.paused
        self.note("acquisition paused" if self.paused else "acquisition resumed")

    def toggle_manual(self):
        self.manual = not self.manual
        self.note("condition: manual steering" if self.manual else "condition: autonomous")

    def toggle_lethal(self):
        self.game.lethal = not self.game.lethal
        self.note("collisions are now " + ("lethal" if self.game.lethal else "non-lethal (nose contact only)"))

    def ap_to_u(self, y):
        a, m, b = self.ap_breaks
        if y < m:
            return (y - a) / (m - a) * self.ap_split
        return self.ap_split + (y - m) / (b - m) * (1 - self.ap_split)

    # ------------------------------------------------------------ helpers --
    def note(self, text):
        self.log.append((self.sim_time, text))

    def text(self, txt, pos, color=TEXT, f=None, anchor="topleft"):
        f = f or self.f_s
        key = (txt, color, id(f))
        img = self._text_cache.get(key)
        if img is None:
            if len(self._text_cache) > 4000:
                self._text_cache.clear()
            img = self._text_cache[key] = f.render(txt, True, color)
        r = img.get_rect(**{anchor: pos})
        self.canvas.blit(img, r)
        return r

    def panel(self, rect, title, subtitle=""):
        pygame.draw.rect(self.canvas, PANEL, rect)
        pygame.draw.rect(self.canvas, BORDER, rect, 1)
        pygame.draw.line(self.canvas, BORDER, (rect.x, rect.y + 24), (rect.right - 1, rect.y + 24))
        r = self.text(title.upper(), (rect.x + 10, rect.y + 5), TEXT, self.f_s)
        if subtitle:
            self.text(subtitle, (r.right + 10, rect.y + 6), MUTED, self.f_xs)

    def cell_at(self, pos):
        if not ARENA.collidepoint(pos):
            return None
        return (int(pos[0] - ARENA.x) // CELL, int(pos[1] - ARENA.y) // CELL)

    def cell_center(self, cell):
        return (ARENA.x + cell[0] * CELL + CELL // 2, ARENA.y + cell[1] * CELL + CELL // 2)

    def body_region(self, cell):
        body = self.game.body
        if cell not in body:
            return None
        i = body.index(cell)
        if i == 0:
            return "touch"
        return "body_anterior" if i / max(1, len(body) - 1) <= 0.5 else "body_posterior"

    # ------------------------------------------------------------ trials ---
    def start_trial(self, kind, cell):
        label, cells = STIMULI[kind]
        trial = {"n": len(self.trials) + 1, "t": self.sim_time, "tick": self.tick,
                 "stimulus": label, "cells": cells, "cell": cell,
                 "first_command": None, "command": "-", "behaviour": "none", "latency": None}
        self.trials.append(trial)
        self.pending.append(trial)
        color = TOXIN if kind == "toxin" else TOUCH
        self.raster_stim.append((self.tick, color))
        self.markers.append([cell, self.sim_time, color])

    def update_trials(self, frames, start_tick):
        b = self.brain
        last = self.ctrl.last
        for trial in list(self.pending):
            for k, fired in enumerate(frames):
                t = start_tick + k
                if t < trial["tick"] or trial["first_command"] is not None:
                    continue
                rev = fired[b.reverse_cmd].sum()
                fwd = fired[b.forward_cmd].sum()
                if rev or fwd:
                    trial["first_command"] = t - trial["tick"]
                    trial["command"] = "AVA/AVD/AVE" if rev >= fwd else "AVB/PVC"
            if trial["behaviour"] == "none":
                if last["reverse"]:
                    trial["behaviour"] = "reversal"
                    trial["latency"] = self.tick - trial["tick"]
                elif last["sprint"]:
                    trial["behaviour"] = "forward run"
                    trial["latency"] = self.tick - trial["tick"]
            if trial["behaviour"] != "none" or self.tick - trial["tick"] > TRIAL_WINDOW:
                self.pending.remove(trial)

    # ------------------------------------------------------------ input ---
    def on_mouse_down(self, pos, button):
        if button == 1:
            if self.slider.inflate(16, 24).collidepoint(pos):
                self.dragging = True
                self.set_rate_from_x(pos[0])
                return
            for rect, _, action in self.buttons:
                if rect.collidepoint(pos):
                    action()
                    return
        self.on_click(pos, button)

    def on_mouse_move(self, pos):
        self.hover = pos
        if self.dragging:
            self.set_rate_from_x(pos[0])

    def on_mouse_up(self):
        self.dragging = False

    def on_click(self, pos, button):
        cell = self.cell_at(pos)
        if cell is None:
            return
        g = self.game
        if button == 3:
            if cell not in g.body and not g.is_toxic(cell):
                g.place_food(cell)
                self.note(f"food source moved to {cell}")
            return
        region = self.body_region(cell)
        if region:
            self.ctrl.poke(region)
            self.start_trial(region, cell)
            self.note(f"stimulus: {STIMULI[region][0]} at {cell}")
        elif cell != g.food:
            g.add_toxin(cell)
            self.start_trial("toxin", cell)
            self.note(f"stimulus: toxin droplet at {cell}")

    def handle_key(self, key):
        if key == pygame.K_SPACE:
            self.toggle_pause()
        elif key == pygame.K_m:
            self.toggle_manual()
        elif key == pygame.K_k:
            self.toggle_lethal()
        elif key == pygame.K_r:
            self.new_specimen("reset by operator")
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(RATE_MAX, round(self.speed * 1.25, 2))
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(RATE_MIN, round(self.speed / 1.25, 2))
        elif key == pygame.K_e:
            self.export()
        elif self.manual and key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT):
            want = {pygame.K_UP: 0, pygame.K_RIGHT: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3}[key]
            self.manual_turn = {0: 0, 1: 1, 3: -1}.get((want - self.game.dir) % 4, 0)

    def new_specimen(self, reason):
        self.best = max(self.best, self.game.score)
        self.note(f"specimen #{self.specimen} ended: {reason} (score {self.game.score})")
        self.banner = (f"specimen #{self.specimen} ended: {reason}  →  specimen #{self.specimen + 1}",
                       self.sim_time)
        self.specimen += 1
        self.game.reset()
        self.ctrl.reset()
        self.pending.clear()
        self.prev_body = list(self.game.body)

    def export(self):
        out = os.path.join(HERE, "recordings")
        os.makedirs(out, exist_ok=True)
        ts_path = os.path.join(out, f"{self.exp_id}_timeseries.csv")
        tr_path = os.path.join(out, f"{self.exp_id}_trials.csv")
        if self.recording:
            with open(ts_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.recording[0]))
                w.writeheader()
                w.writerows(self.recording)
        with open(tr_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "t_s", "stimulus", "sensory_neurons", "cell",
                        "first_command_latency_ticks", "first_command", "behaviour",
                        "behaviour_latency_ticks"])
            for t in self.trials:
                w.writerow([t["n"], f"{t['t']:.2f}", t["stimulus"], t["cells"], t["cell"],
                            t["first_command"], t["command"], t["behaviour"], t["latency"]])
        self.note(f"exported -> recordings/{self.exp_id}_*.csv")
        self.status_msg = (f"saved recordings/{self.exp_id}_*.csv", time.time())

    # ------------------------------------------------------------ update --
    def step(self):
        g, c = self.game, self.ctrl
        if not g.alive:
            self.new_specimen(f"collision with {g.death_cause}")
        score = g.score
        start_tick = self.tick
        self.prev_body = list(g.body)
        if self.manual:
            c.decide(g)
            g.step(self.manual_turn)
            self.manual_turn = 0
        else:
            c.act(g)
        frames = c.spike_frames
        self.tick += len(frames)
        last = c.last
        if last.get("reverse") and not self.manual:
            self.prev_body.reverse()
        if g.bumped:
            self.note(f"nose contact with {g.bumped} (non-lethal)")

        # raster ring buffer
        block = np.array([f[self.raster_order] for f in frames])
        self.raster = np.vstack([self.raster[len(block):], block])
        fired = np.array(frames).mean(axis=0)
        for k, (_, mask, _) in enumerate(self.brain.circuits):
            self.circuit_level[k] = 0.5 * self.circuit_level[k] + 0.5 * fired[mask].mean()
        self.activity = np.maximum(self.activity * 0.5, np.array(frames).max(axis=0))

        self.update_trials(frames, start_tick)
        if g.score > score:
            self.note(f"food acquired (score {g.score})")
        tox = max(last["toxin"])
        self.history.append((self.sim_time, last["odor"], tox, last["F"], last["B"]))
        while self.history and self.history[0][0] < self.sim_time - TRACE_SECONDS:
            self.history.popleft()
        self.recording.append({
            "t_s": round(self.sim_time, 3), "specimen": self.specimen, "step": g.steps,
            "head_x": g.body[0][0], "head_y": g.body[0][1], "score": g.score,
            "odour": round(last["odor"], 4), "bumped": g.bumped or "", "toxin_L": round(last["toxin"][0], 3),
            "toxin_R": round(last["toxin"][1], 3), "touch_L": last["touch"][0],
            "touch_R": last["touch"][1], "head_muscle_L": round(last["L"], 1),
            "head_muscle_R": round(last["R"], 1), "forward_cmd": last["F"],
            "reverse_cmd": last["B"], "turn": last["turn"], "reverse": int(last["reverse"]),
            "sprint": int(last["sprint"]), "alive": int(g.alive),
        })

    def update(self, dt):
        if self.paused:
            return
        self.sim_time += dt
        self.step_acc += dt
        interval = 1.0 / self.speed
        while self.step_acc >= interval:
            self.step_acc -= interval
            self.step()
        self.step_progress = min(1.0, self.step_acc / interval)
        self.markers = [m for m in self.markers if self.sim_time - m[1] < 2.0]

    # ------------------------------------------------------------ drawing --
    def draw_header(self):
        s = self.canvas
        pygame.draw.rect(s, PANEL, (0, 0, CANVAS[0], 64))
        pygame.draw.line(s, BORDER, (0, 64), (CANVAS[0], 64))
        self.text("OpenWorm Connectome Behaviour Rig", (20, 10), TEXT, self.f_title)
        self.text("C. elegans hermaphrodite connectome (Cook et al. 2019)  ·  "
                  f"{self.brain.n_neurons} neurons  ·  {self.brain.n_synapses} synapses  ·  "
                  "integrate-and-fire, 6 ticks / step", (20, 36), MUTED, self.f_s)
        fields = [
            ("EXPERIMENT", self.exp_id),
            ("SPECIMEN", f"#{self.specimen:03d}"),
            ("CONDITION", "manual" if self.manual else "autonomous"),
            ("STEP RATE", f"{self.speed:g} Hz"),
            ("ELAPSED", f"{self.sim_time:8.1f} s"),
            ("SCORE / BEST", f"{self.game.score} / {max(self.best, self.game.score)}"),
        ]
        x = CANVAS[0] - 20
        for label, value in reversed(fields):
            w = max(self.f_mono_m.size(value)[0], self.f_xs.size(label)[0]) + 22
            x -= w
            pygame.draw.line(s, BORDER, (x, 10), (x, 54))
            self.text(label, (x + 11, 12), MUTED, self.f_xs)
            self.text(value, (x + 11, 30), TEXT, self.f_mono_m)
        rec = "PAUSED" if self.paused else "ACQUIRING"
        col = TOUCH if self.paused else (TOXIN if int(time.time() * 2) % 2 else (120, 50, 45))
        pygame.draw.circle(s, col, (x - 96, 36), 5)
        self.text(rec, (x - 86, 29), TEXT, self.f_s)

    def draw_arena(self):
        s = self.canvas
        g = self.game
        self.panel(ARENA_PANEL, "Arena",
                   f"24 × 24 grid · food odour C = exp(-d / {ODOR_LENGTH:.0f} cells)")
        # odour field, evaluated per pixel so the gradient is continuous
        if self._odour_food != g.food:
            self._odour_food = g.food
            px = (np.arange(ARENA.w) + 0.5) / CELL - 0.5
            py = (np.arange(ARENA.h) + 0.5) / CELL - 0.5
            d = np.hypot(px[:, None] - g.food[0], py[None, :] - g.food[1])
            lut = np.array([colormap(k / 255) for k in range(256)], dtype=np.uint8)
            img = lut[(np.exp(-d / ODOR_LENGTH) * 255).astype(np.uint8)]
            self._odour = pygame.surfarray.make_surface(img)
            self._odour.blit(self._grid, (0, 0))
        outer = ARENA.inflate(2 * WALL, 2 * WALL)
        s.set_clip(outer)
        for k in range(-outer.h, outer.w, 7):
            pygame.draw.line(s, (72, 77, 86), (outer.x + k, outer.bottom), (outer.x + k + outer.h, outer.y))
        s.set_clip(None)
        s.blit(self._odour, ARENA.topleft)
        clip = s.get_clip()
        s.set_clip(ARENA)

        # food source and iso-concentration contours
        fc = self.cell_center(g.food)
        for level in (0.8, 0.6, 0.4, 0.2):
            r = -math.log(level) * ODOR_LENGTH * CELL
            pygame.draw.circle(s, (150, 175, 170), fc, int(r), 1)
            self.text(f"{level:.1f}", (fc[0] + int(r * 0.71) + 2, fc[1] - int(r * 0.71) - 12), (170, 190, 185), self.f_xs)
        pygame.draw.circle(s, (240, 244, 210), fc, 8)
        pygame.draw.circle(s, BG, fc, 8, 2)

        # toxins: sensing radius and remaining lifetime
        for cell, expiry in g.toxins:
            c = self.cell_center(cell)
            pygame.draw.circle(s, TOXIN, c, int(TOXIN_LENGTH * CELL * 1.5), 1)
            pygame.draw.circle(s, mix(TOXIN, PANEL, 0.6), c, int(TOXIN_LENGTH * CELL * 0.75), 1)
            r = pygame.Rect(0, 0, CELL - 6, CELL - 6)
            r.center = c
            pygame.draw.rect(s, TOXIN, r)
            pygame.draw.line(s, BG, r.topleft, r.bottomright, 2)
            pygame.draw.line(s, BG, r.topright, r.bottomleft, 2)
            left = (expiry - g.steps) / 60
            pygame.draw.arc(s, TEXT, r.inflate(10, 10), math.pi / 2, math.pi / 2 + left * 2 * math.pi, 2)

        self.draw_worm()

        # stimulus markers (crosshair, fades over 2 s)
        for cell, t0, color in self.markers:
            a = 1 - (self.sim_time - t0) / 2.0
            c = self.cell_center(cell)
            col = mix(PANEL, color, a)
            r = int(12 + (1 - a) * 18)
            pygame.draw.circle(s, col, c, r, 1)
            pygame.draw.line(s, col, (c[0] - r - 6, c[1]), (c[0] - r + 4, c[1]))
            pygame.draw.line(s, col, (c[0] + r - 4, c[1]), (c[0] + r + 6, c[1]))
            pygame.draw.line(s, col, (c[0], c[1] - r - 6), (c[0], c[1] - r + 4))
            pygame.draw.line(s, col, (c[0], c[1] + r - 4), (c[0], c[1] + r + 6))

        # hover probe
        if self.hover:
            cell = self.cell_at(self.hover)
            if cell:
                r = pygame.Rect(ARENA.x + cell[0] * CELL, ARENA.y + cell[1] * CELL, CELL, CELL)
                pygame.draw.rect(s, TEXT, r, 1)
                region = self.body_region(cell)
                lines = [f"cell {cell}   odour C = {g.odor_at(cell):.3f}"]
                if g.toxins:
                    lines.append(f"toxin C = {g.toxin_at(cell):.3f}")
                if region:
                    lines.append(f"LMB: {STIMULI[region][0]} → {STIMULI[region][1]}")
                elif cell != g.food:
                    lines.append("LMB: toxin droplet → ASH ADL ASK")
                lines.append("RMB: move food source here")
                w = max(self.f_s.size(t)[0] for t in lines) + 14
                box = pygame.Rect(r.right + 8, r.y, w, 8 + 16 * len(lines))
                if box.right > ARENA.right:
                    box.right = r.left - 8
                if box.bottom > ARENA.bottom:
                    box.bottom = ARENA.bottom - 2
                pygame.draw.rect(s, PANEL, box)
                pygame.draw.rect(s, BORDER, box, 1)
                for i, t in enumerate(lines):
                    self.text(t, (box.x + 7, box.y + 4 + i * 16), TEXT if i == 0 else MUTED, self.f_s)
        text, t0 = self.banner
        if self.sim_time - t0 < 3.0:
            img = self.f_m.render(text, True, TEXT)
            box = img.get_rect(midtop=(ARENA.centerx, ARENA.y + 12)).inflate(24, 14)
            pygame.draw.rect(s, PANEL, box)
            pygame.draw.rect(s, TOUCH, box, 1)
            s.blit(img, img.get_rect(center=box.center))
        s.set_clip(clip)

        # the wall around the plate (hatched) and cell coordinates: everything
        # inside the wall is the whole 24 x 24 world
        outer = ARENA.inflate(2 * WALL, 2 * WALL)
        pygame.draw.rect(s, (150, 156, 166), outer, 1)
        pygame.draw.rect(s, (170, 176, 186), ARENA.inflate(2, 2), 1)
        for k in range(0, GRID_W, 4):
            x = ARENA.x + k * CELL + CELL // 2
            y = ARENA.y + k * CELL + CELL // 2
            self.text(str(k), (x, outer.bottom + 2), FAINT, self.f_xs, "midtop")
            self.text(str(k), (outer.x - 4, y), FAINT, self.f_xs, "midright")
        self.text("WALL", (outer.right, outer.bottom + 2), MUTED, self.f_xs, "topright")

        # colour bar
        cb = pygame.Rect(ARENA_PANEL.right - 170, ARENA_PANEL.y + 9, 110, 8)
        for k in range(cb.w):
            pygame.draw.line(s, colormap(k / cb.w), (cb.x + k, cb.y), (cb.x + k, cb.bottom))
        self.text("0", (cb.x - 4, cb.y - 3), MUTED, self.f_xs, "topright")
        self.text("1  C", (cb.right + 4, cb.y - 3), MUTED, self.f_xs)

    def worm_points(self):
        """Segment centres interpolated between the previous and current step."""
        g = self.game
        p = self.step_progress
        p = p * p * (3 - 2 * p)  # ease in-out
        prev = self.prev_body
        pts = []
        for k, cell in enumerate(g.body):
            a = self.cell_center(prev[min(k, len(prev) - 1)])
            b = self.cell_center(cell)
            if math.dist(a, b) > CELL * 2.5:  # reset or teleport: no interpolation
                a = b
            pts.append((a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p))
        return pts

    def draw_worm(self):
        s = self.canvas
        pts = self.worm_points()
        n = len(pts)

        def radius(k):
            return 10.5 - 4.5 * k / max(1, n - 1)

        # dark outline, then a continuous tapered body
        for layer, grow, color_of in ((0, 2.0, lambda k: (10, 12, 15)),
                                      (1, 0.0, lambda k: mix((236, 238, 240), (128, 134, 144), k / max(1, n - 1)))):
            for k in range(n - 1, -1, -1):
                r = radius(k) + grow
                col = color_of(k)
                if k + 1 < n:
                    a, b = pts[k], pts[k + 1]
                    steps = 4
                    for j in range(1, steps):
                        t = j / steps
                        pygame.draw.circle(s, col, (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t),
                                           r - (radius(k) - radius(k + 1)) * t)
                pygame.draw.circle(s, col, pts[k], r)
        # nose and heading
        hx, hy = pts[0]
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.game.dir]
        pygame.draw.line(s, ACCENT, (hx + dx * 11, hy + dy * 11), (hx + dx * 22, hy + dy * 22), 2)
        pygame.draw.circle(s, ACCENT, (hx, hy), 4)
        if n > 2:
            tail = pts[-1]
            self.text("head", (hx + 12, hy - 22), MUTED, self.f_xs)
            self.text("tail", (tail[0] + 10, tail[1] - 20), FAINT, self.f_xs)

    def draw_controls(self):
        s = self.canvas
        pygame.draw.rect(s, PANEL, CONTROLS)
        pygame.draw.rect(s, BORDER, CONTROLS, 1)
        x = CONTROLS.x + 12
        self.text("STEP RATE", (x, CONTROLS.y + 5), MUTED, self.f_xs)
        self.text(f"{self.speed:g} Hz", (x, CONTROLS.y + 18), TEXT, self.f_mono_m)
        self.text(f"{1000 / self.speed:.0f} ms / step", (x, CONTROLS.y + 37), FAINT, self.f_xs)
        sl = self.slider
        pygame.draw.rect(s, GRID, sl, border_radius=3)
        u = self.rate_to_u(self.speed)
        pygame.draw.rect(s, ACCENT, (sl.x, sl.y, int(sl.w * u), sl.h), border_radius=3)
        for rate in (0.5, 1, 2, 5, 10, 20):
            tx = sl.x + sl.w * self.rate_to_u(rate)
            pygame.draw.line(s, FAINT, (tx, sl.y - 5), (tx, sl.y - 2))
            self.text(f"{rate:g}", (tx, sl.y - 18), FAINT, self.f_xs, "midtop")
        knob = (int(sl.x + sl.w * u), sl.centery)
        pygame.draw.circle(s, TEXT if self.dragging else ACCENT, knob, 8)
        pygame.draw.circle(s, BG, knob, 8, 2)
        for rect, label, _ in self.buttons:
            hot = self.hover and rect.collidepoint(self.hover)
            pygame.draw.rect(s, (38, 43, 50) if hot else (31, 35, 41), rect, border_radius=4)
            pygame.draw.rect(s, ACCENT if hot else BORDER, rect, 1, border_radius=4)
            txt = label()
            f = self.f_xs if self.f_s.size(txt)[0] > rect.w - 10 else self.f_s
            self.text(txt, rect.center, TEXT, f, "center")

    def bar(self, x, y, w, label, value, vmax, color):
        self.text(label, (x, y), MUTED, self.f_xs)
        txt = f"{value:.0f}" if vmax >= 10 else f"{value:.2f}"
        self.text(txt, (x + w, y), TEXT, self.f_mono, "topright")
        r = pygame.Rect(x, y + 16, w, 6)
        pygame.draw.rect(self.canvas, GRID, r)
        pygame.draw.rect(self.canvas, color, (x, y + 16, int(w * min(1.0, value / vmax)), 6))

    def draw_readout(self):
        last = self.ctrl.last
        self.panel(READOUT, "Sensory input / motor output", "current step")
        if not last:
            return
        x0, y0, w = READOUT.x + 12, READOUT.y + 30, 154
        up, down = last["food"]
        cols = [
            [("odour C at head", last["odor"], 1.0, ODOUR),
             ("ASEL/AWA  C rising", up, 1.5, ODOUR)],
            [("ASER/AWC  C falling", down, 1.5, ODOUR),
             ("ASH/ADL  left", last["toxin"][0], 2.6, TOXIN)],
            [("ASH/ADL  right", last["toxin"][1], 2.6, TOXIN),
             ("nose touch", max(last["touch"]), 1.6, TOUCH)],
            [("fwd cmd AVB/PVC", last["F"], 12, FORWARD),
             ("rev cmd AVA/AVD/AVE", last["B"], 12, REVERSE)],
        ]
        for ci, col in enumerate(cols):
            for ri, (label, v, vmax, c) in enumerate(col):
                self.bar(x0 + ci * (w + 22), y0 + ri * 30, w, label, v, vmax, c)

    def draw_map(self):
        s = self.canvas
        self.panel(MAP, "Anatomical soma map", "neuron positions from c302 NeuroML · lateral view · "
                   "highlighted = fired in the last step")
        inner = self.map_inner
        # schematic outline
        top, bot = [], []
        for k in range(81):
            u = k / 80
            width = (0.3 + 0.7 * math.sin(math.pi * min(1.0, u * 1.5 + 0.1)) ** 0.5) * (1 - 0.75 * u ** 3)
            x = inner.x - 10 + u * (inner.w + 20)
            top.append((x, inner.centery - width * inner.h * 0.62))
            bot.append((x, inner.centery + width * inner.h * 0.62))
        pygame.draw.polygon(s, (30, 34, 40), top + bot[::-1])
        pygame.draw.polygon(s, BORDER, top + bot[::-1], 1)
        # axis
        ay = MAP.bottom - 26
        pygame.draw.line(s, FAINT, (inner.x, ay), (inner.right, ay))
        for um in (-300, -250, -200, 0, 200, 400):
            x = inner.x + self.ap_to_u(um) * inner.w
            pygame.draw.line(s, FAINT, (x, ay), (x, ay + 4))
            self.text(f"{um}", (x, ay + 5), MUTED, self.f_xs, "midtop")
        self.text("[µm]", (inner.x - 22, ay + 5), MUTED, self.f_xs, "topright")
        self.text("anterior → posterior  (head region expanded)",
                  (inner.right, MAP.y + 30), MUTED, self.f_xs, "topright")
        xs = inner.x + self.ap_to_u(-180) * inner.w
        pygame.draw.line(s, FAINT, (xs, inner.y), (xs, ay), 1)
        self.text("head ganglia / nerve ring", (inner.x, MAP.y + 30), MUTED, self.f_xs)

        act = self.activity
        for i, p, color, dim in self.map_pts:
            if act[i] > 0.5:
                pygame.draw.circle(s, color, p, 4)
                pygame.draw.circle(s, TEXT, p, 5, 1)
            else:
                pygame.draw.circle(s, dim, p, 2)
        # key neuron labels on two rows above/below the body, with leader lines
        labelled = []
        for name, label in KEY_NEURONS.items():
            i = self.brain.idx.get(name)
            pt = next((m[1] for m in self.map_pts if m[0] == i), None)
            if pt is not None:
                labelled.append((pt[0], pt, label, act[i] > 0.5))
        labelled.sort()
        rows = [MAP.y + 46, ay - 16]
        next_free = [inner.x - 30, inner.x - 30]
        for k, (_, pt, label, on) in enumerate(labelled):
            row = k % 2
            w = self.f_xs.size(label)[0]
            lx = max(pt[0] - w // 2, next_free[row])
            next_free[row] = lx + w + 8
            ly = rows[row]
            anchor_y = ly + (12 if row == 0 else 0)
            pygame.draw.line(s, TEXT if on else FAINT, pt, (lx + w // 2, anchor_y))
            self.text(label, (lx, ly), TEXT if on else MUTED, self.f_xs)
        lx = MAP.x + 250
        for label, col in [("sensory", SENSORY), ("interneuron", INTER), ("motor", MOTOR)]:
            pygame.draw.circle(s, col, (lx + 4, MAP.y + 37), 4)
            lx += self.text(label, (lx + 12, MAP.y + 30), MUTED, self.f_xs).w + 24

    def draw_raster(self):
        s = self.canvas
        self.panel(RASTER, "Spike raster", f"{len(self.raster_order)} neurons × last {RASTER_TICKS} ticks")
        plot = pygame.Rect(RASTER.x + 78, RASTER.y + 32, RASTER.w - 90, RASTER.h - 58)
        pygame.draw.rect(s, (19, 21, 26), plot)
        img = np.zeros((RASTER_TICKS, len(self.raster_order), 3), dtype=np.uint8)
        img[:] = (19, 21, 26)
        on = self.raster
        img[on] = np.broadcast_to(self.raster_colors, img.shape)[on]
        surf = pygame.surfarray.make_surface(img)
        s.blit(pygame.transform.scale(surf, plot.size), plot.topleft)
        rows = len(self.raster_order)
        for kind, a, b in self.raster_bands:
            y0 = plot.y + a / rows * plot.h
            y1 = plot.y + b / rows * plot.h
            pygame.draw.line(s, BORDER, (plot.x, y0), (plot.right, y0))
            self.text(kind, (plot.x - 6, (y0 + y1) / 2), MUTED, self.f_xs, "midright")
        # stimulus onsets
        for t, color in self.raster_stim:
            age = self.tick - t
            if 0 <= age < RASTER_TICKS:
                x = plot.right - age / RASTER_TICKS * plot.w
                pygame.draw.line(s, color, (x, plot.y), (x, plot.bottom), 1)
                pygame.draw.polygon(s, color, [(x - 4, plot.y - 6), (x + 4, plot.y - 6), (x, plot.y)])
        pygame.draw.rect(s, BORDER, plot, 1)
        for k in range(0, RASTER_TICKS + 1, 60):
            x = plot.right - k / RASTER_TICKS * plot.w
            pygame.draw.line(s, FAINT, (x, plot.bottom), (x, plot.bottom + 4))
            self.text(f"-{k}" if k else "0", (x, plot.bottom + 5), MUTED, self.f_xs, "midtop")
        self.text("[ticks]", (plot.x - 18, plot.bottom + 5), MUTED, self.f_xs, "topright")

    def draw_circuits(self):
        s = self.canvas
        self.panel(CIRC, "Circuit activity", "% of neurons firing")
        y = CIRC.y + 34
        for k, (label, mask, role) in enumerate(self.brain.circuits):
            v = float(self.circuit_level[k]) * 100
            self.text(label.title(), (CIRC.x + 10, y), TEXT if v > 2 else MUTED, self.f_xs)
            bx, bw = CIRC.x + 118, 130
            pygame.draw.rect(s, GRID, (bx, y + 3, bw, 9))
            color = {"REVERSE CMD": REVERSE, "FORWARD CMD": FORWARD, "NOCICEPTION": TOXIN,
                     "CHEMOSENSORY": ODOUR, "NOSE TOUCH": TOUCH, "BODY TOUCH": TOUCH}.get(label, ACCENT)
            pygame.draw.rect(s, color, (bx, y + 3, int(bw * min(1.0, v / 40)), 9))
            self.text(f"{v:4.1f}", (CIRC.right - 10, y), TEXT, self.f_mono, "topright")
            y += 23
        self.text("bar full scale = 40 %", (CIRC.x + 10, CIRC.bottom - 18), FAINT, self.f_xs)

    def draw_traces(self):
        s = self.canvas
        self.panel(TRACES, "Time series", f"last {TRACE_SECONDS:.0f} s")
        lanes = [("odour C", ODOUR, 1, 1.0, "1.0"), ("toxin (nociceptor drive)", TOXIN, 2, 2.0, "2.0"),
                 ("forward cmd [spikes/step]", FORWARD, 3, 12, "12"),
                 ("reverse cmd [spikes/step]", REVERSE, 4, 12, "12")]
        plot = pygame.Rect(TRACES.x + 44, TRACES.y + 32, TRACES.w - 56, TRACES.h - 60)
        lane_h = plot.h / len(lanes)
        now = self.sim_time
        hist = list(self.history)
        for n, (label, color, idx, vmax, top_label) in enumerate(lanes):
            top = plot.y + n * lane_h
            base = top + lane_h - 6
            pygame.draw.rect(s, (19, 21, 26), (plot.x, top, plot.w, lane_h - 6))
            for sec in range(0, int(TRACE_SECONDS) + 1, 5):
                x = plot.right - sec / TRACE_SECONDS * plot.w
                pygame.draw.line(s, GRID, (x, top), (x, base))
            pygame.draw.line(s, FAINT, (plot.x, base), (plot.right, base))
            self.text(top_label, (plot.x - 4, top), MUTED, self.f_xs, "topright")
            self.text("0", (plot.x - 4, base - 12), MUTED, self.f_xs, "topright")
            self.text(label, (plot.x + 6, top + 2), color, self.f_xs)
            pts = [(plot.right - (now - t) / TRACE_SECONDS * plot.w,
                    base - min(1.0, h[idx - 1] / vmax) * (lane_h - 22)) for t, *h in hist]
            if len(pts) > 1:
                pygame.draw.lines(s, color, False, pts, 2)
        for sec in range(0, int(TRACE_SECONDS) + 1, 5):
            x = plot.right - sec / TRACE_SECONDS * plot.w
            self.text(f"-{sec}" if sec else "0", (x, plot.bottom + 2), MUTED, self.f_xs, "midtop")
        self.text("t [s]", (plot.x - 16, plot.bottom + 2), MUTED, self.f_xs, "topright")
        # stimulus onsets on the time axis
        for tr in self.trials[-20:]:
            age = now - tr["t"]
            if age < TRACE_SECONDS:
                x = plot.right - age / TRACE_SECONDS * plot.w
                col = TOXIN if tr["stimulus"].startswith("toxin") else TOUCH
                pygame.draw.line(s, col, (x, plot.y), (x, plot.bottom), 1)

    def draw_trials(self):
        s = self.canvas
        self.panel(TRIALS, "Stimulus–response trials", f"n = {len(self.trials)}")
        x = TRIALS.x + 10
        y = TRIALS.y + 32
        heads = [("#", 0), ("stimulus", 24), ("1st cmd", 150), ("resp.", 214), ("lat.", 268)]
        for h, dx in heads:
            self.text(h, (x + dx, y), MUTED, self.f_xs)
        pygame.draw.line(s, BORDER, (x, y + 16), (TRIALS.right - 10, y + 16))
        y += 22
        for t in self.trials[-8:][::-1]:
            pending = t in self.pending
            col = MUTED if pending else TEXT
            beh_col = {"reversal": REVERSE, "forward run": FORWARD}.get(t["behaviour"], MUTED)
            self.text(str(t["n"]), (x, y), col, self.f_mono)
            self.text(t["stimulus"].replace(" touch", ""), (x + 24, y), col, self.f_xs)
            self.text(t["cells"], (x + 24, y + 13), FAINT, self.f_xs)
            cmd = "…" if pending and t["first_command"] is None else (
                "-" if t["first_command"] is None else f"{t['command'][:3]} {t['first_command']}t")
            self.text(cmd, (x + 150, y), col, self.f_xs)
            beh = "…" if pending and t["behaviour"] == "none" else t["behaviour"].replace("forward run", "fwd run")
            self.text(beh, (x + 214, y), beh_col, self.f_xs)
            self.text("-" if t["latency"] is None else f"{t['latency']}t", (x + 268, y), col, self.f_xs)
            y += 30
        # summary statistics per stimulus
        pygame.draw.line(s, BORDER, (x, TRIALS.bottom - 64), (TRIALS.right - 10, TRIALS.bottom - 64))
        yy = TRIALS.bottom - 58
        for key in ("body_anterior", "body_posterior", "touch", "toxin"):
            label = STIMULI[key][0]
            done = [t for t in self.trials if t["stimulus"] == label and t not in self.pending]
            if not done:
                continue
            rev = sum(t["behaviour"] == "reversal" for t in done)
            fwd = sum(t["behaviour"] == "forward run" for t in done)
            self.text(f"{label[:14]:<14} n={len(done):<3} rev {100 * rev // len(done):3d}%  fwd {100 * fwd // len(done):3d}%",
                      (x, yy), MUTED, self.f_mono)
            yy += 14

    def draw_footer(self):
        s = self.canvas
        y = CANVAS[1] - 14
        if self.log:
            t, msg = self.log[-1]
            self.text(f"[{t:7.1f} s] {msg}", (20, y), MUTED, self.f_xs, "midleft")
        self.text("LMB worm: touch  ·  LMB floor: toxin  ·  RMB: move food  ·  SPACE pause  ·  "
                  "M manual  ·  +/− rate  ·  K lethal  ·  R new specimen  ·  E export CSV  ·  ESC quit",
                  (CANVAS[0] - 20, y), FAINT, self.f_xs, "midright")

    def draw(self):
        self.canvas.fill(BG)
        self.draw_header()
        self.draw_arena()
        self.draw_controls()
        self.draw_readout()
        self.draw_map()
        self.draw_raster()
        self.draw_circuits()
        self.draw_traces()
        self.draw_trials()
        self.draw_footer()


def run(speed=3.0, max_frames=None, screenshot=None, script=None):
    # SCALED lets SDL resize the fixed-size canvas with the window, keeping the
    # aspect ratio (letterboxing) and mapping mouse positions back to canvas pixels
    os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "best")
    pygame.init()
    window = pygame.display.set_mode(CANVAS, pygame.SCALED | pygame.RESIZABLE)
    pygame.display.set_caption("OpenWorm Connectome Behaviour Rig")
    con = Console(speed)
    clock = pygame.time.Clock()
    frames = 0
    while True:
        dt = min(0.1, clock.tick(FPS) / 1000)
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return con
            if e.type == pygame.KEYDOWN:
                con.handle_key(e.key)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
                con.on_mouse_down(e.pos, e.button)
            elif e.type == pygame.MOUSEBUTTONUP:
                con.on_mouse_up()
            elif e.type == pygame.MOUSEMOTION:
                con.on_mouse_move(e.pos)
        if script:
            script(con, frames)
        con.update(dt)
        con.draw()
        window.blit(con.canvas, (0, 0))
        pygame.display.flip()
        frames += 1
        if max_frames and frames >= max_frames:
            if screenshot:
                pygame.image.save(con.canvas, screenshot)
            pygame.quit()
            return con


if __name__ == "__main__":
    run()
