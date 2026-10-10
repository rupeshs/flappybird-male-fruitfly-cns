# flybrainmap

Play Flappy Bird with a real fruit-fly connectome.

`brain.py` runs the wiring of an adult male *Drosophila* central nervous system (MaleCNS v1.0,
166,700 neurons, 25.6 million connections) as a spiking network. The game state is turned into a
"the ground is rushing up" signal. That signal is injected into the fly's own looming-detector
neurons, and when the fly's **giant-fiber escape neuron (DNp01)** fires, the bird flaps.

Nothing in the network is trained. Only the wiring the fly was born with decides whether a flap happens.

---

## Background: what is MaleCNS?

A **connectome** is a detailed map of the neural connections within a brain or nervous system. It
shows which neurons are connected to which other neurons, and through which synapses. You can think
of it as the wiring diagram of the brain.

MaleCNS is the reconstructed adult male fruit fly central nervous system (brain plus ventral nerve
cord) and the wiring between its neurons:

```
MaleCNS
  = neurons
  + synapses
  + connectivity map
  + anatomical information
```

It was produced by FlyEM (HHMI Janelia), the University of Cambridge, the MRC Laboratory of
Molecular Biology and Google Research (Berg et al., 2026, *Cell*), and is released under CC BY 4.0.

---

## How the pieces fit together

```
 Flappy Bird env                brain.py                        flybrain package
 (flappy_bird_gymnasium)                                        (MaleCNS as a spiking net)
 ─────────────────────   ─────────────────────────────   ─────────────────────────────────
  obs (11 numbers)  ───►  get_game_state()
                          bird height, velocity, gap
                                   │
                                   ▼
                          loom_strength()  0.0 … 1.0
                          "how close is the ground"
                                   │
                                   ▼  inject voltage
                          brain.step(inject=[(LC4+LPLC2, loom)])  ───►  LIF update of all
                          brain.step()                                   166,700 neurons
                                   │                                     along the connectome
                                   ▼  fired neuron ids
                          count spikes of DNp01
                                   │
   action 0 / 1     ◄───  spikes > 0  →  flap (1)
```

Every game frame the loop in `fly_action()` does this:

1. Read the game observation and work out where the bird is relative to the next gap.
2. Convert that to a single **loom amplitude** between 0 and 1.
3. Add that much voltage to every LC4 and LPLC2 neuron (both eyes) for one brain step.
4. Run a second brain step so the signal can propagate to the descending neurons.
5. If DNp01 spiked, flap. Otherwise do nothing.

---

## How the fly CNS (MaleCNS) is used

The wiring comes from the [`flybrain`](https://github.com/alextitonis/fly.ai) package, which
prebuilds MaleCNS into two files:

| File | Contents |
|------|----------|
| `weights.npz` | The signed, normalised connection matrix (25,582,938 connections). Weight is the synapse count. The sign comes from the predicted neurotransmitter: GABA, glutamate and histamine are inhibitory, the rest excitatory. Each neuron's input is normalised. |
| `brain.npz` | Per-neuron metadata: cell type, side (L/R), superclass, position, and eye layout. |

They are downloaded automatically on the first `FlyBrain()` call (~260 MB, checked by sha256) into
`~/fly-data`, or into `$FLY_DATA` if you set it.

### The neuron model

`FlyBrain` is a leaky integrate-and-fire (LIF) network over the whole connectome. One step is 20 ms
of fly time:

```
v ← exp(-dt/τ)·v  +  gain · W @ spikes  +  tonic  +  noise  +  injected input
v ≥ 1  →  the neuron spikes, then v resets to 0
```

with `τ = 100 ms`, `gain = 3.0`, `tonic = 0.14`, and sparse random noise. These constants are
hand-calibrated, not measured. They are chosen so that descending neurons stay almost silent at
rest (~1 Hz) while real sensory pathways still get through. The network is noisy, so a neuron can
occasionally fire on its own.

### The pathway `brain.py` uses

```
 LC4, LPLC2          ──────►         DNp01
 (visual projection                  (descending neuron, "giant fiber":
  neurons that detect                 the fly's take-off / escape command)
  looming objects)
```

`make_brain()` picks the neurons by cell type:

```python
brain      = FlyBrain(device="auto")               # GPU if available, else CPU
loom_cells = brain.cells(["LC4", "LPLC2"])          # input: looming detectors
dnp01      = brain.cells(["DNp01"])                 # output: giant fiber
```

- **Input, LC4 and LPLC2.** In the real fly these cells respond to something large approaching, such
  as a predator or the ground. Here the "approaching thing" is the ground or the bottom pipe.
- **Output, DNp01.** Its spikes mean take-off, which the game maps to a flap.
- `LC10a -> DNp01` was tested and never made DNp01 fire, so it is not used.
- The real eye is not simulated. The loom signal is injected as voltage directly into LC4/LPLC2.


## Tuning constants

| Constant | Default | Effect |
|----------|---------|--------|
| `FLAP_MARGIN` | 4 px | How far above the gap bottom the bird flaps. A flap lifts the bird ~45 px and the gap is 100 px, so this is tight. |
| `LOOM_AT_LINE` | 0.28 | Loom amplitude at the flap line. Must sit just above DNp01's firing threshold. |
| `LOOM_PX_SCALE` | 40 | Pixels per 1.0 of loom amplitude. Smaller means a sharper response. |
| `BRAIN_STEPS_PER_FRAME` | 2 | Brain steps per game frame (step 1 receives the pulse, step 2 reads the answer). |
| `SWITCH_FRAMES` | 1 | Lookahead for deciding a pipe has been passed. |
| `RISING_LOCKOUT` | -1 | Ignore flap commands while rising faster than this. |

---

## Live brain view in the browser

```bat
python brain.py --web --headless
```

This opens a page at <http://localhost:8765/> that shows the fly's nervous system firing while it
plays. It keeps playing new episodes until you press Ctrl+C. `--headless` skips the pygame window,
since the page has its own game view.

What is on the page:

| Area | Shows |
|------|-------|
| 3D CNS map | 140,638 neurons at their reconstructed positions (brain on the left, nerve cord on the right), coloured by region. Each neuron that fires flashes and fades. The loom-input cells (LC4/LPLC2) and the two DNp01 neurons are labelled, and DNp01 sends out a ring when it spikes. Drag to rotate, scroll to zoom, double-click to reset. |
| Sensorimotor loop | The game, plus the chain loom input → connectome → DNp01 → wing motor, lit as each stage is active. The wing motor shows `FLAP`, or `HELD` when DNp01 fired but the rising lockout ignored it. |
| Loom → DNp01 | The last 10 s of loom level, the 0.28 line where DNp01 starts to fire, and every DNp01 spike. |
| Firing by region | The share of each region's neurons that fired in the last frame. |
| Network activity | The share of all 166,700 neurons firing per frame, over the last 20 s. |

How it works: `brain.py` calls `Monitor.publish()` in `webview.py` once per game frame. That
starts a small HTTP server on a background thread (standard library only) and streams each frame to
the page over Server-Sent Events. A frame is a bitmap of which drawable neurons fired (~24 KB), the
per-region counts, and the game and motor state. The neuron layout (`positions.bin`, 1.6 MB) is
downloaded once when the page loads. The page itself is `web/index.html`, a single file that
draws the neurons with WebGL2.

It needs a browser with WebGL2 (any current Chrome, Edge, Firefox or Safari). The 26,062 neurons
with no reconstructed position (mostly sensory neurons) are left out of the map, but still count in
the firing statistics. Options: `--port N` (default 8765, the next free port is used if it is
taken) and `--no-open` (do not open the browser automatically).

---

## Running it

```bat
env\Scripts\activate
pip install -r requirements.txt

python brain.py                     :: watch the game window
python brain.py --headless          :: no rendering
python brain.py --verbose           :: print bird position, loom, and DNp01 spikes every frame
python brain.py --web --headless    :: live brain view in the browser (see above)
```

The first run downloads the brain files (~260 MB). The first steps on CPU are slow while numba
compiles, and later steps take about 12–15 ms on 24 threads (about 1.4 ms on an RTX 4060 with
`flybrain[gpu]`).

To smoke-test the brain alone (no game), run `python testflyb.py`. It stimulates left-eye
LC4/LPLC2 and prints which cells fired.

---

## Credits

- Connectome: MaleCNS v1.0, FlyEM (HHMI Janelia) et al. Berg, S. et al. (2026), *Sexual dimorphism in
  the complete connectome of the Drosophila male central nervous system*, *Cell*. CC BY 4.0.
- Simulator: [`flybrain`](https://github.com/alextitonis/fly.ai) by alextitonis (MIT), whose neuron
  model follows [Fly64](https://github.com/ornata/fly) by Jessica Paquette.
- Game: [`flappy-bird-gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium).
