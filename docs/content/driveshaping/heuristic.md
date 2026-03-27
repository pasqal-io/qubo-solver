## Heuristic Drive Shaper

`HeuristicDriveShaper` builds a fixed, closed-form quantum drive directly from
the structure of the QUBO diagonal — no optimization loop required.
It is faster than the [`OptimizedDriveShaper`](./optimized.md) and more
principled for problems with heterogeneous diagonal terms.

It outputs the generated drive and an empty solution object (bitstrings and
costs are populated only after the quantum simulation is executed by the solver).

### Key Idea

The diagonal entries $Q_{ii}$ encode the local bias of each qubit.
The shaper scales them with a factor $\alpha$ to produce per-site target
detunings:

$d_i = -\alpha \, Q_{ii}$

These are then realized by splitting the total detuning into:

- A **global detuning** $\delta_g(T) = d_{\max}$ — the top of the range, applied
  uniformly to all qubits.
- A **local DMM detuning** $\delta_{\text{dmm}}(T) = -(d_{\max} - d_{\min}) \leq 0$
  — a site-dependent negative shift applied via the Detuning Map Modulator (DMM).

Per-site weights $w_i \in [0, 1]$ are chosen so that the combined detuning
at the final time recovers $d_i$:

$\delta_i(T) = \delta_g(T) + \delta_{\text{dmm}}(T) \cdot w_i = d_i$

When no DMM is available (or `dmm=False`), only the global detuning is used.

### Schedule Shape

The drive follows a simple 4-point, 3-phase schedule:

**Amplitude** — a flat-top plateau:

$\Omega = [0,\ \Omega_{\max},\ \Omega_{\max},\ 0]$

**Global detuning** — stays negative during the drive-on phase, then sweeps to the
final encoded value and holds:

$\delta_g = [\delta_0,\ \delta_0,\ \delta_g(T),\ \delta_g(T)]$

where $\delta_0 = -|\delta_{\max}|$ (the most negative hardware-allowed value).

**DMM detuning map** — ramps from zero to the negative local shift $\delta_{\text{dmm}}(T)$
weighted by $w_i$, if DMM is active.

### Configuration Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `drive_shaping_method` | `DriveType \| str` | — | Must be set to `DriveType.HEURISTIC` or `"heuristic"` |
| `dmm` | `bool` | `True` | Whether to use the Detuning Map Modulator for site-dependent detuning |
| `heuristic_alpha_safety` | `float` | `0.8` | Safety fraction applied to $\alpha_{\max}$ to keep the encoding within hardware bounds |
| `heuristic_kappa` | `float` | `0.25` | Fraction of the detuning energy scale used to set $\Omega_{\max}$ |

`heuristic_alpha_safety` and `heuristic_kappa` are passed via `DriveShapingConfig`
and read by the shaper at generation time.

### How $\alpha$ is Chosen

$\alpha_{\max}$ is computed so that the final encoding fits within both the global
detuning bounds and the DMM magnitude bounds of the selected device:

$\alpha_{\max} = \min\!\left(\frac{\delta_{\text{dmm,max}}}{Q_{\max} - Q_{\min}},\ \frac{\delta_{g,\max}}{-Q_{\min}}\right)$

The effective $\alpha$ is then:

$\alpha = \texttt{heuristic\_alpha\_safety} \times \alpha_{\max}$

### How $\Omega_{\max}$ is Chosen

The Rabi amplitude plateau is set as a fraction of the detuning energy scale:

$\Omega_{\max} = \texttt{heuristic\_kappa} \times \max_i |d_i|$

This value is clamped to the hardware `max_amp` and `min_avg_amp` channel
constraints before being used in the waveform.

### Example

```python exec="on" source="material-block" html="1"
import torch

from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, DriveShapingConfig
from qubosolver.solver import QuboSolver
from qubosolver.qubo_types import DriveType

Q = torch.tensor([[-1.0, 0.5, 0.2], [0.5, -2.0, 0.3], [0.2, 0.3, -3.0]])

instance = QUBOInstance(Q)

config = SolverConfig(
    use_quantum=True,
    drive_shaping=DriveShapingConfig(
        drive_shaping_method=DriveType.HEURISTIC,
        dmm=True,
        heuristic_alpha_safety=0.8,
        heuristic_kappa=0.25,
    ),
)

solver = QuboSolver(instance, config)
solution = solver.solve()
print(solution)
```

This returns a `QUBOSolution` containing the solution bitstrings, counts,
probabilities, and costs produced by the quantum simulation.

---
