## Optimized Pulse Shaper

`OptimizedPulseShaper` uses bayesian optimization to find pulse parameters (amplitude and detuning) in order to solve a QUBO problem using quantum simulation.

It outputs both the optimized pulse and a solution object containing bitstrings, counts, probabilities, and associated costs.

### Features:
- Computes normalized weights from the QUBO diagonal to support later application of the Detuning Map Modulator (DMM).
- Uses Bayesian optimization to tune six parameters: three for the Rabi amplitude ($\Omega$), and three for the global detuning ($\delta$).
- Executes quantum simulations at each iteration to evaluate candidate pulse parameters and their performance on the QUBO.
- Returns the final optimized pulse and best QUBO solution, with full metadata (counts, probabilities, and costs).

### Initialization Parameters:

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `instance`   | `QUBOInstance` | Qubo instance. |
| `config` | `SolverConfig` | Configuration for solving. |


### Pulse Parameterization
The optimized pulse is built from an `InterpolatedWaveform` with:

Amplitude:
$\Omega = [0, \Omega_1, \Omega_2, \Omega_3, 0]$

Detuning:
$\delta = [\delta_1, \delta_2, \delta_3]$

These waveforms:

Always start and end in zero amplitude;
Use 3 intermediate amplitude values ($\Omega_1$ to $\Omega_3$) and 3 detuning values ($\delta_1$ to $\delta_3$), which are the parameters optimized.

The pulse starts with an `InterpolatedWaveform` with the points:

- $\Omega = [0, 5, 10, 5, 0]$
- $\delta = [-10, 0, 10]$

### Methods Overview
- `generate(self, target: Register, instance: QUBOInstance) -> tuple[Pulse, QUBOSolution]`:
Runs the Bayesian optimization loop and returns the optimized pulse and corresponding solution. Handles fallback cases if simulation fails.

- `build_pulse(self, params: list) -> Pulse`:
Creates a Pulse from a 6-element parameter list: the first 3 for amplitude, the last 3 for detuning.

- `_compute_norm_weights(self, QUBO: torch.Tensor) -> list[float]`:
Normalizes the QUBO diagonal weights (used in DMM shaping).

- `run_simulation(...) -> tuple[...]`:
Runs a simulation of the current pulse on a quantum backend and returns bitstring results, probabilities, and QUBO costs.

- `compute_qubo_cost(self, bitstring: str, QUBO: torch.Tensor) -> float`:
Computes the QUBO cost of a specific bitstring.


### Output Structure
After the final round of optimization, the following attributes are populated:

- `pulse`: Final Pulse object with optimized waveform parameters.
- `best_cost`: Minimum cost found during optimization.
- `best_bitstring`: Corresponding bitstring with the lowest cost.
- `bitstrings, counts, probabilities, costs`: Full result distributions as PyTorch tensors.

### Example

```python exec="on" source="material-block" html="1"
import torch

from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, PulseShapingConfig
from qoolqit._solvers.types import BackendType, DeviceType
from qubosolver.solver import QuboSolver
from qubosolver.qubo_types import PulseType


Q = torch.tensor([[-63.9423,   0.0000], [0.0000, -44.1916]])

instance = QUBOInstance(Q)

default_config = SolverConfig(
    use_quantum = True, pulse_shaping=PulseShapingConfig(pulse_shaping_method=PulseType.OPTIMIZED), n_calls = 25
)
solver = QuboSolver(instance, default_config)

solution = solver.solve()
print(solution)

```
This will return a `QUBOSolution` instance, which comprehends the solution bitstrings, the counts of each bitstring, their probabilities and costs.

---
