# OpenWorm Connectome Snake

A snake game steered by the real nervous system of *C. elegans*, plus a lab-style
console for poking the worm and watching which neurons respond.

The worm's brain is the full hermaphrodite connectome from
[OpenWorm / c302](https://github.com/openworm/c302) (Cook et al. 2019):
300 neurons, 148 muscles and other end organs, and 7,379 chemical and electrical synapses.
No steering rules are hard-coded. The game feeds sensory neurons, the network
runs, and the head muscles and command interneurons decide what the snake does.

![Experiment console](docs/screenshot.png)

## Quick start

```bash
git clone --recursive https://github.com/snuri00/openworm-connectome-snake.git
cd openworm-connectome-snake
pip install -r requirements.txt
python3 snake_worm.py
```

If you cloned without `--recursive`, fetch the connectome data with
`git submodule update --init`.

Benchmark the connectome against a random-walking snake, without a window:

```bash
python3 snake_worm.py --headless 50
```

## Interacting with the worm

| Action | What happens in the nervous system |
|---|---|
| Left-click the **head** | nose touch: FLP, OLQ, IL1, ASH |
| Left-click the **front half** of the body | gentle anterior touch: ALM, AVM |
| Left-click the **back half** of the body | gentle posterior touch: PLM, PVM |
| Left-click the **floor** | toxin droplet, sensed by the nociceptors ASH, ADL, ASK |
| Right-click | move the food source |

The panel under the arena has a step-rate slider (0.5–20 Hz) and buttons for
pause, lethal/non-lethal collisions, manual steering, a new specimen and CSV
export. Keyboard shortcuts: `SPACE` pause, `M` manual (arrow keys), `+`/`-`
rate, `K` lethal collisions, `R` new specimen, `E` export CSV, `ESC` quit.

### Console panels

- **Arena**: the food odour field `C = exp(-d / 7 cells)` with iso-concentration
  contours, toxins, and the worm.
- **Anatomical soma map**: every neuron at its real position, taken from the
  c302 NeuroML model. Neurons that fired in the last step are highlighted.
- **Spike raster**: all 300 neurons over the last 240 ticks, grouped into
  sensory, inter- and motor neurons. Stimulus onsets are marked.
- **Circuit activity**: percentage of neurons firing in 9 functional circuits
  (chemosensory, nociception, nose/body touch, navigation, forward/reverse
  command, head steering, body motor).
- **Time series**: odour, toxin drive, and forward and reverse command activity.
- **Stimulus–response trials**: for every poke or toxin, the latency of the
  first command-neuron spike, the resulting behaviour and its latency.
- **Export (`E`)**: writes `recordings/<experiment>_timeseries.csv` and
  `recordings/<experiment>_trials.csv`.

## How it works

### Brain model (`worm_brain.py`)

An integrate-and-fire network in the style of Busbice's GoPiGo connectome robot:

- Every cell accumulates input. When a neuron crosses its threshold it fires,
  adds its synaptic weights to its targets and resets.
- Chemical synapses from GABAergic neurons (DD, VD, RME, RIS, AVL, DVB) are
  inhibitory.
- **Spike-frequency adaptation**: each spike raises that neuron's threshold for
  a while. Without it the network locks into runaway, seizure-like firing.
- Gap junctions are scaled down, because at full strength they also cause
  runaway firing. The exception is the touch receptor neurons, which reach the
  command interneurons mainly through gap junctions (ALM/AVM ↔ AVD, PLM ↔ PVC).
- Following Chalfie et al. (1985), the chemical synapses from ALM/AVM onto
  the forward command neurons (AVB, PVC) are inhibitory. Without that sign,
  the anatomy alone makes an anterior touch drive the worm forwards.
- Muscles never fire. Their input is read out every tick.

### Game ↔ brain interface (`snake_worm.py`)

| Game event | Sensory neurons |
|---|---|
| obstacle ahead / left / right | nose touch FLP, ASH, OLQ, IL1 (per side) |
| odour rising | ASEL, AWA |
| odour falling | ASER, AWC |
| toxin nearby | ASH, ADL, ASK (strength from concentration, side from bearing) |
| click on the body | ALM/AVM or PLM/PVM |

On each game step the brain runs for 6 ticks. The motor readout works like this:

- **Turn**: head muscles 1–6 active above a threshold. The side with the
  stronger contraction sets the direction.
- **Reverse**: the reverse command neurons (AVA/AVD/AVE) clearly dominate the
  forward ones. The snake crawls backwards, so its tail becomes its head.
- **Forward run**: the forward command neurons (AVB/PVC) dominate. The snake
  makes an extra move.

Food finding is not a left-versus-right comparison. It emerges as
**klinotaxis / pirouettes**, as in real worms. When the odour is falling, activity
builds over a few steps along the path AWC → AIB → head motor neurons (SMD/RMD),
and the worm turns. When the odour is rising, the network stays quiet and the worm
keeps going.

## Measured behaviour

Measured with `--headless` and scripted trials:

| Test | Result |
|---|---|
| Mean score, connectome vs random | 2.2 vs 0.1 |
| Food sensing ablated | score drops from 2.4 to 0.5 |
| Anterior body touch | 97 % reversal (AVA/AVD fire 2–3 ticks after the stimulus) |
| Nose touch | 80 % reversal |
| Posterior body touch | 70 % forward run |
| Toxin 5 cells ahead | reaches an adjacent cell in 4 % of trials (26 % with nociceptors ablated) |

## Limitations

- The worm is not a skilled snake player. As its body grows it tends to trap itself.
- The threshold, adaptation, gap-junction scaling and stimulus gains were tuned by
  hand so that the network neither saturates nor stays silent. The wiring
  itself is used unchanged, apart from the inhibitory signs described above.
- The worm crawls on its side and bends dorsoventrally. Mapping its left/right
  onto the game's left/right is a convention.
- Posterior touch is noisy. The strong anatomical PVC → AVA synapse sometimes
  also triggers a reversal.

## Files

| File | Purpose |
|---|---|
| `worm_brain.py` | connectome loader and integrate-and-fire network |
| `snake_worm.py` | game, sensory/motor interface, headless benchmark |
| `worm_hud.py` | experiment console (pygame) |
| `neuron_atlas.py` | soma positions and cell types from the c302 NeuroML file |
| `c302/` | OpenWorm c302 (git submodule, MIT licence): connectome and NeuroML data |

## Credits

- Connectome and NeuroML data: [OpenWorm c302](https://github.com/openworm/c302);
  Cook et al., *Nature* 571, 63–71 (2019).
- Touch circuit: Chalfie et al., *J. Neurosci.* 5, 956–964 (1985).
- GABAergic neurons: McIntire et al., *Nature* 364, 337–341 (1993).
- Integrate-and-fire approach: Timothy Busbice's connectome robot.
