# Drive shaping workflow

Drive shaping builds the time-dependent drive Hamiltonian — amplitude and detuning waveforms, and optionally a detuning map — applied to a register during quantum solving. Each drive shaping algorithm is callable directly, or picked and configured through `Solver`/`SolverConfig`.

## Goal of drive shaping

The [embedding](embedding.md) step encodes the QUBO's off-diagonal coefficients $Q_{ij}$ into the physical interaction between atoms. The diagonal coefficients $Q_{ii}$ are encoded separately, as local detunings $\delta_i$ applied to each atom: $Q_{ii} \longleftrightarrow -2\delta_i$ (see [The quantum pipeline](intro.md)). Drive shaping is the step that builds this detuning schedule, together with the amplitude (Rabi frequency) schedule that drives the system, so that the ground state of the resulting Hamiltonian encodes a good solution of the QUBO.

A drive shaping algorithm returns a [`qoolqit.Drive`][] — an amplitude waveform, a detuning waveform, and optionally a per-atom weighted detuning (a Detuning Map Modulator, or DMM) for site-dependent control — together with a `Solution`. Algorithms that only build the drive analytically from the problem's structure return an empty `Solution`, since no sampling has happened yet; algorithms that also run a quantum simulation internally return a populated one.

`qubosolver` currently ships three drive shaping algorithms — proportional-diagonal, local-energy-scale, and Bayesian search, described below — plus the option to write your own.

## Proportional-diagonal

The proportional-diagonal shaper builds a fixed drive directly from the diagonal of the QUBO matrix, without any numerical optimization: the final local detuning $\delta_i(T)$ is set proportional to $-Q_{ii}$, and the peak Rabi frequency $\Omega_{\max}$ is set proportional to the peak detuning via a `kappa` coefficient.

- API reference: [`qubosolver.drive_shaping.proportional_diagonal`](../../api/drive_shaping.md)
- Tutorial: [Proportional-diagonal notebook](../../tutorials/04-drive-shaping/01-proportional-diagonal.ipynb)
- Under the hood: [Proportional-diagonal](../../under_the_hood/driveshaping/proportional_diagonal.md)

### Code example
```python exec="on" source="tabbed-left" session="drive_shaping" result="text"
from qubosolver import Instance, matrix, embedding, drive_shaping
import qoolqit

# Private utility to set seed.
from qubosolver.utils._random import manual_seed
manual_seed(958)

instance = Instance(matrix.tensor([
    [-1, 1, 2, 1],
    [ 1,-3, 3, 0],
    [ 2, 3,-1, 5],
    [ 1, 0, 5,-2],
    ]))
register = embedding.blade.embed(instance)
device = qoolqit.AnalogDeviceWithDMM()

drive = drive_shaping.proportional_diagonal.build_drive(
    instance, register, device=device, dmm=True, kappa=0.25)
print(drive)
```
Plot the resulting drive:

```python exec="on" source="tabbed-right" session="drive_shaping"
from pathlib import Path
from matplotlib import pyplot as plt

output_dir = Path.cwd()
output_dir = Path("{{ config.site_dir.split('"')[0] }}/user_guide/quantum/drive_shaping")  # markdown-exec: hide
output_dir.mkdir(parents=True, exist_ok=True)

drive.draw()
fig = plt.gcf()
fig.savefig(output_dir / "quantum_drive_shaping_proportional_diagonal.svg")
plt.close(fig)

print('<figure><figcaption>Drive</figcaption><img src="quantum_drive_shaping_proportional_diagonal.svg" alt="Proportional-diagonal drive"></figure>')  # markdown-exec: hide
```

## Local-energy-scale

The local-energy-scale shaper also builds a fixed drive analytically, but derives the peak Rabi frequency from the average local physical energy scale $E_i = |\delta_i(T)| + \sum_{j \neq i} |V_{ij}|$ — combining the target detuning with the interaction strengths induced by the embedded register — rather than from the detuning alone.

- API reference: [`qubosolver.drive_shaping.local_energy_scale`](../../api/drive_shaping.md)
- Tutorial: [Local-energy-scale notebook](../../tutorials/04-drive-shaping/02-local-energy-scale.ipynb)
- Under the hood: [Local-energy-scale](../../under_the_hood/driveshaping/local_energy_scale.md)

### Code example
```python exec="on" source="tabbed-left" session="drive_shaping_local_energy_scale" result="text"
from qubosolver import Instance, matrix, embedding, drive_shaping
import qoolqit

# Private utility to set seed.
from qubosolver.utils._random import manual_seed
manual_seed(958)

instance = Instance(matrix.tensor([
    [-1, 1, 2, 1],
    [ 1,-3, 3, 0],
    [ 2, 3,-1, 5],
    [ 1, 0, 5,-2],
    ]))
register = embedding.blade.embed(instance)
device = qoolqit.AnalogDeviceWithDMM()

drive = drive_shaping.local_energy_scale.build_drive(
    instance, register, device=device, dmm=True, kappa=0.25)
print(drive)
```
Plot the resulting drive:

```python exec="on" source="tabbed-right" session="drive_shaping_local_energy_scale"
from pathlib import Path
from matplotlib import pyplot as plt

output_dir = Path.cwd()
output_dir = Path("{{ config.site_dir.split('"')[0] }}/user_guide/quantum/drive_shaping")  # markdown-exec: hide
output_dir.mkdir(parents=True, exist_ok=True)

drive.draw()
fig = plt.gcf()
fig.savefig(output_dir / "quantum_drive_shaping_local_energy_scale.svg")
plt.close(fig)

print('<figure><figcaption>Drive</figcaption><img src="quantum_drive_shaping_local_energy_scale.svg" alt="Local-energy-scale drive"></figure>')  # markdown-exec: hide
```

## Bayesian search

Unlike the two heuristics above, Bayesian search is not really a drive-shaping algorithm — it is a hybrid quantum-classical solver in its own right: it runs a Bayesian optimization loop ([`skopt.gp_minimize`](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html)) over six waveform knots (three for the amplitude, three for the detuning), running a quantum simulation at each evaluation and minimizing a configurable objective of the resulting `Solution`. Because that loop already produces both a tuned drive and a sampled solution, it can also be reused as a drive shaper for another solving step — which is why it is included here.

This dual role means it is more expensive than the other two heuristics, but able to adapt the drive to the problem and the backend rather than following a fixed rule. Because it simulates the problem while shaping the drive, it needs a `backend` in addition to the `device`, and it returns a populated `Solution` alongside the drive — reused directly if you're also using it as your solving step.

- API reference: [`qubosolver.solving.drive_bayesian_search`](../../api/quantum_solving.md)
- Tutorial: [Bayesian search notebook](../../tutorials/04-drive-shaping/03-bayesian-search.ipynb)
- Under the hood: [Bayesian search](../../under_the_hood/driveshaping/bayesian_search.md)

The example below uses a small `n_evaluations` for a quick, self-contained doc run; see [`solving.drive_bayesian_search.Config`](../../api/quantum_solving.md) for the full set of parameters, including the initial waveform knots, the random seed, and the objective function.

### Code example
```python exec="on" source="tabbed-left" session="drive_shaping_bayesian_search" result="text"
from qubosolver import Instance, matrix, embedding, solving, analysis, LocalEmulator
import qoolqit

# Private utility to set seed.
from qubosolver.utils._random import manual_seed
manual_seed(958)

instance = Instance(matrix.tensor([
    [-1, 1, 2, 1],
    [ 1,-3, 3, 0],
    [ 2, 3,-1, 5],
    [ 1, 0, 5,-2],
    ]))
register = embedding.blade.embed(instance)
device = qoolqit.AnalogDeviceWithDMM()
backend = LocalEmulator()

config = solving.drive_bayesian_search.Config(n_evaluations=11)

solution, drive = solving.drive_bayesian_search.solve(
    instance, register, backend=backend, device=device, dmm=True, config=config)
print(analysis.to_dataframe([solution]))
```
Plot the resulting drive:

```python exec="on" source="tabbed-right" session="drive_shaping_bayesian_search"
from pathlib import Path
from matplotlib import pyplot as plt

output_dir = Path.cwd()
output_dir = Path("{{ config.site_dir.split('"')[0] }}/user_guide/quantum/drive_shaping")  # markdown-exec: hide
output_dir.mkdir(parents=True, exist_ok=True)

drive.draw()
fig = plt.gcf()
fig.savefig(output_dir / "quantum_drive_shaping_bayesian_search.svg")
plt.close(fig)

print('<figure><figcaption>Drive</figcaption><img src="quantum_drive_shaping_bayesian_search.svg" alt="Bayesian-search drive"></figure>')  # markdown-exec: hide
```

## Custom drive

You can skip the built-in algorithms altogether and build a [`qoolqit.Drive`][] directly from waveforms, as in the example below, or reuse a drive computed elsewhere:

```python exec="on" source="tabbed-right" session="drive_shaping"
import qoolqit
from pathlib import Path
from matplotlib import pyplot as plt

drive = qoolqit.Drive(
    amplitude=qoolqit.InterpolatedWaveform(500, [1e-9, 4.0, 4.0, 1e-9]),
    detuning=qoolqit.InterpolatedWaveform(500, [-6.0, -6.0, 2.0, 2.0]),
)

output_dir = Path.cwd()
output_dir = Path("{{ config.site_dir.split('"')[0] }}/user_guide/quantum/drive_shaping")  # markdown-exec: hide
output_dir.mkdir(parents=True, exist_ok=True)

drive.draw()
fig = plt.gcf()
fig.savefig(output_dir / "quantum_drive_shaping_custom_drive.svg")
plt.close(fig)

print('<figure><figcaption>Drive</figcaption><img src="quantum_drive_shaping_custom_drive.svg" alt="Custom drive"></figure>')  # markdown-exec: hide
```

## The `Solver` shortcut

Rather than calling `drive_shaping.proportional_diagonal.build_drive`, `drive_shaping.local_energy_scale.build_drive`, or `solving.drive_bayesian_search.solve` directly, you can select and configure the drive shaping algorithm through `DriveShapingConfig`, nested in `QuantumSolvingConfig` and `SolverConfig`; `Solver` then runs it as part of the full quantum pipeline. Leaving `DriveShapingConfig` unset falls back to the proportional-diagonal shaper with DMM enabled — see [`SolverConfig`][qubosolver.SolverConfig] for the full set of defaults.

`DriveShapingConfig` only exposes the most commonly tuned parameters of each algorithm (e.g. `proportional_diagonal_kappa`, `local_energy_scale_kappa`, `bayesian_search_n_calls`). Finer-grained parameters — such as Bayesian search's `objective_fn` or its optimization callback — are not settable this way; call `solving.drive_bayesian_search.solve` directly with a full `solving.drive_bayesian_search.Config` if you need those.

```python exec="on" source="tabbed-left" session="drive_shaping" result="text"
from qubosolver import SolverConfig, QuantumSolvingConfig, DriveShapingConfig
from dataclasses import asdict
import pprint

drive_shaping_config = DriveShapingConfig(
    algorithm = "local_energy_scale",
    # algorithm = "proportional_diagonal",
    # algorithm = "bayesian_search",
    dmm = True,
    local_energy_scale_kappa = 0.3,
)
quantum_config = QuantumSolvingConfig(
    drive_shaping = drive_shaping_config,
)
solver_config = SolverConfig(
    solving = quantum_config,
)
print(pprint.pformat(asdict(solver_config.solving.drive_shaping)))
```
