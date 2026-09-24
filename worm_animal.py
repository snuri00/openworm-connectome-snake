"""
Closed loop between the connectome (worm_brain) and the soft body (worm_body).

  body-wall muscles  -> preferred curvature of each of the 24 body segments
  body curvature     -> proprioceptive input to motor neurons:
      * B-type motor neurons (VB, DB) feel the bending of the body just
        anterior to them and push their own segment the same way, so a bend
        travels from head to tail (Wen et al. 2012)
      * head motor neurons (SMD, RMD) feel the head bend and pull it back the
        other way (negative feedback), which makes the head sweep from side to side
"""
import math
import re

import numpy as np

from worm_body import N_SEGMENTS, WormBody
from worm_brain import WormBrain

DT = 0.01  # s per brain tick / physics step


class WormAnimal:
    def __init__(self, brain=None, body=None, seed=None,
                 muscle_tau=0.1, muscle_gain=0.1, max_kappa=8.0,
                 b_gain=2.5, b_lag=2, head_gain=0.0, head_rows=4,
                 forward_drive=0.0, noise=3.0, cpg_freq=0.5, cpg_gain=1.2,
                 balance_tau=4.0, gain_cap=1.5, soft_saturation=True):
        self.brain = brain or WormBrain()
        self.body = body or WormBody()
        self.rng = np.random.default_rng(seed)
        self.muscle_tau = muscle_tau
        self.muscle_gain = muscle_gain
        self.max_kappa = max_kappa
        self.b_gain = b_gain
        self.b_lag = b_lag
        self.head_gain = head_gain
        self.head_rows = head_rows
        self.forward_drive = forward_drive
        self.noise = noise
        self.gain_cap = gain_cap              # max boost for weakly innervated rows
        self.soft_saturation = soft_saturation
        self.cpg_freq = cpg_freq          # Hz, head sweep rhythm
        self.cpg_gain = cpg_gain          # 0 disables the pacemaker
        self.steer = 0.0                  # -1..1, biases the head dorsal/ventral
        self.direction = 1                # +1 forward (B-type), -1 reverse (A-type)
        self.cpg_phase = 0.0
        self.t = 0.0
        self._wire()
        self.muscle = np.zeros((2, N_SEGMENTS))   # [dorsal, ventral] activation per row
        # slow dorsal/ventral balance per body row (homeostasis): there are 11 VB
        # but only 7 DB neurons, which would otherwise keep the body curled ventrally
        self.balance_tau = balance_tau
        self.balance_from_row = 6                 # the head stays free for steering
        self.drive_mean = np.full((2, N_SEGMENTS), 0.3)

    def _wire(self):
        b = self.brain
        n = N_SEGMENTS
        # muscle cell -> (side, row); side 0 = dorsal, 1 = ventral
        m_idx, m_slot = [], []
        for i, name in enumerate(b.names):
            m = re.fullmatch(r"([dv])BWM[LR](\d+)", name)
            if m:
                m_idx.append(i)
                m_slot.append((0 if m.group(1) == "d" else 1) * n + int(m.group(2)) - 1)
        self.m_idx = np.array(m_idx)
        self.m_slot = np.array(m_slot)

        # each motor neuron's segment = weighted mean of the muscle rows it innervates
        row_sum = {}
        for s, d, w, kind in b.edges:
            m = re.fullmatch(r"[dv]BWM[LR](\d+)", b.names[d])
            if m and b.is_neuron[s] and kind == "chemical":
                acc = row_sum.setdefault(s, [0.0, 0.0])
                acc[0] += w * (int(m.group(1)) - 1)
                acc[1] += w
        self.neuron_row = {s: acc[0] / acc[1] for s, acc in row_sum.items()}

        # Calibration: some rows/sides receive far more motor synapses than
        # others (the tail 3-4x less than mid-body, ventral more than dorsal).
        # Scale each row/side so equal firing gives equal force.
        exc = np.clip(b.W, 0, None)[b.is_neuron].sum(axis=0)
        in_d, in_v = self.muscle_rows(exc)
        ref = np.median(np.concatenate([in_d, in_v]))
        self.side_gain = np.stack([np.clip(ref / np.maximum(in_d, 1), 0.25, self.gain_cap),
                                   np.clip(ref / np.maximum(in_v, 1), 0.25, self.gain_cap)])

        def cells(prefixes):
            return [i for i, nm in enumerate(b.names)
                    if b.is_neuron[i] and nm.startswith(prefixes) and i in self.neuron_row]

        self.db = cells(("DB",))
        self.vb = cells(("VB",))
        self.da = cells(("DA",))
        self.va = cells(("VA",))
        # tail pacemaker targets for backward crawling: the most posterior A cells
        self.tail_dorsal = sorted(self.da, key=lambda i: self.neuron_row[i])[-2:]
        self.tail_ventral = sorted(self.va, key=lambda i: self.neuron_row[i])[-2:]

        # proprioceptive wiring as arrays: which cell, which row it feels, sign
        def sensing(dorsal, ventral, lag):
            cells = dorsal + ventral
            rows = [max(0, min(n - 1, int(round(self.neuron_row[i])) + lag)) for i in cells]
            signs = [1.0] * len(dorsal) + [-1.0] * len(ventral)
            return np.array(cells), np.array(rows), np.array(signs)

        self.sense_fwd = sensing(self.db, self.vb, -self.b_lag)   # B-type feel anterior
        self.sense_rev = sensing(self.da, self.va, self.b_lag)    # A-type feel posterior

        self.head_dorsal = [b.idx[c] for c in ("SMDDL", "SMDDR", "RMDDL", "RMDDR")]
        self.head_ventral = [b.idx[c] for c in ("SMDVL", "SMDVR", "RMDVL", "RMDVR")]
        self.avb = [b.idx["AVBL"], b.idx["AVBR"]]
        # the pacemaker drives RMD only: SMD also innervates rows 1-16 and would
        # bend the whole front half in synchrony
        self.pace_dorsal = [b.idx[c] for c in ("RMDDL", "RMDDR")]
        self.pace_ventral = [b.idx[c] for c in ("RMDVL", "RMDVR")]

    def muscle_rows(self, values):
        """Sum a per-cell array over body-wall muscles -> (dorsal[24], ventral[24])."""
        out = np.bincount(self.m_slot, weights=values[self.m_idx], minlength=2 * N_SEGMENTS)
        return out.reshape(2, N_SEGMENTS)

    # ------------------------------------------------------------ loop -----
    def proprioception(self):
        b = self.brain
        k = self.body.curvature_at_rows()          # >0 = dorsal bend
        amp = b.threshold
        # forward: B-type cells feel the body anterior to them;
        # reverse: A-type cells feel the body posterior to them
        cells, rows, signs = self.sense_fwd if self.direction > 0 else self.sense_rev
        drive = np.clip(signs * k[rows] / self.max_kappa, 0, None)
        b.v[cells] += amp * self.b_gain * drive
        head = k[:self.head_rows].mean() / self.max_kappa
        # head bent dorsally -> ventral head motor neurons pull it back, and vice versa
        if head > 0:
            b.v[self.head_ventral] += amp * self.head_gain * head
        else:
            b.v[self.head_dorsal] += amp * self.head_gain * -head

    def pacemaker(self, dt):
        """Head rhythm: an oscillating current injected into the head motor
        neurons (not into muscles), standing in for their intrinsic
        oscillatory properties that an integrate-and-fire cell lacks."""
        if not self.cpg_gain:
            return
        b = self.brain
        self.cpg_phase += 2 * math.pi * self.cpg_freq * dt
        wave = math.sin(self.cpg_phase)
        amp = b.threshold * self.cpg_gain
        if self.direction > 0:
            wave += 0.8 * self.steer
            dorsal, ventral = self.pace_dorsal, self.pace_ventral
        else:
            dorsal, ventral = self.tail_dorsal, self.tail_ventral
        if wave > 0:
            b.v[dorsal] += amp * wave
        else:
            b.v[ventral] += amp * -wave

    def step(self, dt=DT):
        b = self.brain
        self.pacemaker(dt)
        self.proprioception()
        if self.forward_drive:
            b.v[self.avb] += b.threshold * self.forward_drive
        b.add_noise(self.noise, self.rng)
        out = b.tick()
        m = b.muscle_out
        drive = self.muscle_rows(m) * self.side_gain * self.muscle_gain
        if self.balance_tau:
            self.drive_mean += (drive - self.drive_mean) * (dt / self.balance_tau)
            mean = self.drive_mean.mean(axis=0, keepdims=True)
            scale = np.clip(mean / np.maximum(self.drive_mean, 1e-3), 1 / self.gain_cap, self.gain_cap)
            scale[:, :self.balance_from_row] = 1.0
            drive = drive * scale
        self.muscle += (np.clip(drive, 0, 3) - self.muscle) * (dt / self.muscle_tau)
        target = (self.muscle[0] - self.muscle[1]) * self.max_kappa
        if self.soft_saturation:
            # muscles saturate gradually instead of pinning the bend at its maximum
            target = self.max_kappa * np.tanh(target / self.max_kappa)
        else:
            target = np.clip(target, -self.max_kappa, self.max_kappa)
        # joint j sits between rows j and j+1
        self.body.step(0.5 * (target[:-1] + target[1:]), dt)
        self.t += dt
        return out


def measure(animal, seconds=20.0):
    """Speed, head-sweep frequency and head->tail wave propagation."""
    steps = int(seconds / DT)
    start = animal.body.center.copy()
    kym = np.zeros((steps, N_SEGMENTS))
    forward = 0.0
    for k in range(steps):
        p_before = animal.body.head.copy()
        animal.step()
        d = animal.body.head - p_before
        h = animal.body.heading
        forward += d[0] * math.cos(h) + d[1] * math.sin(h)
        kym[k] = animal.body.curvature_at_rows()
    head = kym[:, 1]
    crossings = np.sum(np.diff(np.sign(head - head.mean())) != 0)
    freq = crossings / 2 / seconds
    # lag (in steps) that best aligns row 4 with row 16: >0 = wave travels head->tail
    a, c = kym[:, 4] - kym[:, 4].mean(), kym[:, 16] - kym[:, 16].mean()
    lags = range(-150, 151)
    corr = [np.dot(a[max(0, -L):len(a) - max(0, L)], c[max(0, L):len(c) - max(0, -L)]) for L in lags]
    best = list(lags)[int(np.argmax(corr))]
    amp = np.abs(kym).mean()
    return dict(speed_um_s=1000 * forward / seconds,
                net_mm=float(np.linalg.norm(animal.body.center - start)),
                head_freq_hz=freq, wave_lag_s=best * DT, mean_abs_kappa=amp), kym
