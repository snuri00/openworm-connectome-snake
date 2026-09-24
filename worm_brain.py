"""
A simple integrate-and-fire model that runs the C. elegans connectome
(OpenWorm / c302, Cook et al. 2019 hermaphrodite edge list).

Every cell keeps an accumulator. When a neuron fires, it adds its synaptic
weight to each postsynaptic target. A neuron whose accumulator crosses the
threshold fires on the next tick and is reset to zero. Chemical synapses from
GABAergic neurons are treated as inhibitory (negative weight). Muscles never
fire; their accumulated input is read out and cleared every tick
(the approach popularised by Busbice's GoPiGo connectome robot).
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "c302", "c302", "data", "herm_full_edgelist.csv")

# GABAergic (inhibitory) neurons, McIntire et al. 1993
GABAERGIC = {
    "RMED", "RMEV", "RMEL", "RMER", "AVL", "DVB", "RIS",
    *[f"DD{i:02d}" for i in range(1, 7)],
    *[f"VD{i:02d}" for i in range(1, 14)],
}

# Gentle-touch receptor neurons. They talk to the command interneurons mostly
# through gap junctions (ALM/AVM <-> AVD, PLM <-> PVC), so those junctions keep
# a much stronger weight than the rest of the network.
TOUCH_RECEPTORS = {"ALML", "ALMR", "AVM", "PLML", "PLMR", "PVM"}

# Chalfie et al. 1985: anterior touch cells inhibit the forward command
# interneurons through their chemical synapses (glutamate-gated Cl- channels).
INHIBITORY_CHEMICAL = {
    (src, dst) for src in ("ALML", "ALMR", "AVM")
    for dst in ("PVCL", "PVCR", "AVBL", "AVBR")
}

# Sensory neurons driven by the game, as (left side, right side)
SENSORS = {
    # food concentration rising -> ASEL (ON cell), AWA (attractive odour)
    "food_up": (["ASEL", "AWAL"], ["AWAR"]),
    # food concentration falling -> ASER (OFF cell), AWC (triggers pirouettes)
    "food_down": (["AWCL"], ["ASER", "AWCR"]),
    # nose touch / obstacle (wall or own body) -> avoidance
    "touch": (["FLPL", "ASHL", "OLQDL", "OLQVL", "IL1L", "IL1DL", "IL1VL"],
              ["FLPR", "ASHR", "OLQDR", "OLQVR", "IL1R", "IL1DR", "IL1VR"]),
    # gentle touch on the front half of the body
    "body_anterior": (["ALML", "AVM"], ["ALMR"]),
    # gentle touch on the back half of the body
    "body_posterior": (["PLML", "PVM"], ["PLMR"]),
    # noxious chemicals / toxins -> polymodal nociceptors
    "noxious": (["ASHL", "ADLL", "ASKL"], ["ASHR", "ADLR", "ASKR"]),
}

# Functional circuits shown in the HUD: (label, neuron name prefixes, role)
CIRCUITS = [
    ("CHEMOSENSORY", ("ASE", "AWA", "AWC", "AWB", "ADF", "ASI", "ASG", "ASJ"),
     "smell & taste of food"),
    ("NOCICEPTION", ("ASH", "ADL", "ASK", "PVD", "PHA", "PHB"),
     "pain / toxin detection"),
    ("NOSE TOUCH", ("FLP", "OLQ", "IL1", "CEP"), "mechanosensory nose"),
    ("BODY TOUCH", ("ALM", "AVM", "PLM", "PVM"), "gentle body touch"),
    ("NAVIGATION", ("AIY", "AIZ", "AIB", "AIA", "RIA", "RIB", "RIM"),
     "first-layer interneurons"),
    ("REVERSE CMD", ("AVA", "AVD", "AVE"), "backward command"),
    ("FORWARD CMD", ("AVB", "PVC"), "forward command"),
    ("HEAD STEER", ("SMD", "RMD", "RME", "SMB"), "head bending motor"),
    ("BODY MOTOR", ("VA", "DA", "VB", "DB", "AS", "VD", "DD"),
     "ventral cord motor neurons"),
]

HEAD_MUSCLE_ROWS = 6  # body wall muscles 1..6 steer the head


class WormBrain:
    def __init__(self, csv_path=DEFAULT_CSV, threshold=60.0, decay=0.8,
                 gap_scale=0.1, touch_gap_scale=4.0, adapt_gain=10.0,
                 adapt_decay=0.9):
        self.threshold = threshold
        self.decay = decay
        # spike-frequency adaptation: each spike raises that neuron's threshold
        # for a while, which stops the network locking into runaway firing
        self.adapt_gain = adapt_gain
        self.adapt_decay = adapt_decay
        rows = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            for src, dst, w, kind in reader:
                rows.append((src.strip(), dst.strip(), float(w), kind.strip()))

        names = sorted({r[0] for r in rows} | {r[1] for r in rows})
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        n = len(names)

        W = np.zeros((n, n), dtype=np.float32)
        for src, dst, w, kind in rows:
            if kind == "chemical":
                if src in GABAERGIC or (src, dst) in INHIBITORY_CHEMICAL:
                    w = -w
            elif src in TOUCH_RECEPTORS or dst in TOUCH_RECEPTORS:
                w *= touch_gap_scale
            else:
                # Gap junctions share current rather than drive spikes; at full
                # weight the network falls into runaway (epileptic) firing.
                w *= gap_scale
            W[self.idx[src], self.idx[dst]] += w
        self.W = W
        self.n_synapses = len(rows)
        self.edges = [(self.idx[s], self.idx[d], w, k) for s, d, w, k in rows]

        # Neurons are upper-case names; muscles/organs (dBWML1, pm3d, ...) are not.
        self.is_muscle = np.array(["BWM" in n for n in names])
        self.is_neuron = np.array([n[0].isupper() for n in names])
        self.muscle_L = np.array(["BWML" in n for n in names])
        self.muscle_R = np.array(["BWMR" in n for n in names])
        head = np.array(["BWM" in n and int(n[5:]) <= HEAD_MUSCLE_ROWS for n in names])
        self.head_L = self.muscle_L & head
        self.head_R = self.muscle_R & head

        def group(prefixes):
            return np.array([self.is_neuron[i] and n.startswith(prefixes)
                             for i, n in enumerate(names)])

        self.forward_cmd = group(("AVB", "PVC"))
        self.reverse_cmd = group(("AVA", "AVD", "AVE"))
        self.circuits = [(label, group(prefixes), role)
                         for label, prefixes, role in CIRCUITS]

        self.sensor_idx = {
            k: ([self.idx[s] for s in L], [self.idx[s] for s in R])
            for k, (L, R) in SENSORS.items()
        }
        self.reset()

    @property
    def n_neurons(self):
        return int(self.is_neuron.sum())

    def reset(self):
        n = len(self.names)
        self.v = np.zeros(n, dtype=np.float32)
        self.fired = np.zeros(n, dtype=bool)
        self.adapt = np.zeros(n, dtype=np.float32)
        self.activity = np.zeros(n, dtype=np.float32)  # smoothed, for display
        self.muscle_out = np.zeros(n, dtype=np.float32)

    def stimulate(self, kind, left, right):
        """Excite a sensory group with strength 0..~1.5 on each side."""
        L, R = self.sensor_idx[kind]
        amp = self.threshold * 1.2
        self.v[L] += amp * left
        self.v[R] += amp * right

    def add_noise(self, sigma, rng):
        """Spontaneous background activity - a nervous system is never silent."""
        self.v[self.is_neuron] += rng.normal(0, sigma, self.n_neurons)

    def tick(self):
        """Advance one step. Returns a dict with the motor readout."""
        self.v *= self.decay
        if self.fired.any():
            # summing only the rows of neurons that fired is much cheaper than a
            # dense matrix product (typically <10% of cells fire per tick)
            self.v += self.W[self.fired].sum(axis=0)
        self.fired = (self.v > self.threshold + self.adapt) & ~self.is_muscle
        self.v[self.fired] = 0.0
        self.adapt = self.adapt * self.adapt_decay + self.adapt_gain * self.fired

        muscle = np.clip(self.v, 0, None) * self.is_muscle
        self.muscle_out = muscle
        self.v[self.is_muscle] = 0.0
        self.activity = np.maximum(self.activity * 0.85, self.fired.astype(np.float32))
        return {
            "head_L": float(muscle[self.head_L].sum()),
            "head_R": float(muscle[self.head_R].sum()),
            "forward": float(self.fired[self.forward_cmd].sum()),
            "reverse": float(self.fired[self.reverse_cmd].sum()),
        }
