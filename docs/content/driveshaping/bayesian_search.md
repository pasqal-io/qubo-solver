## Bayesian Search Drive Shaper

`BayesianSearchDriveShaper` uses bayesian optimization to find drive parameters (amplitude and detuning) in order to solve a QUBO problem using quantum simulation.

It outputs both the optimized drive and a solution object containing bitstrings, counts, probabilities, and associated costs.

### Features:
- Computes normalized weights from the QUBO diagonal to support later application of the Detuning Map Modulator (DMM).
- Uses Bayesian optimization to tune six parameters: three for the Rabi amplitude ($\Omega$), and three for the global detuning ($\delta$).
- Executes quantum simulations at each iteration to evaluate candidate drive parameters and their performance on the QUBO.
- Returns the final optimized drive and best QUBO solution, with full metadata (counts, probabilities, and costs).

### Initialization Parameters:

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `instance`   | `Instance` | Qubo instance. |
| `config` | `SolverConfig` | Configuration for solving. |


### Drive Parameterization
The optimized drive is built from an `InterpolatedWaveform` with:

Amplitude:
$\Omega = [0, \Omega_1, \Omega_2, \Omega_3, 0]$

Detuning:
$\delta = [\delta_1, \delta_2, \delta_3]$

These waveforms:

- Always start and end in zero amplitude;
- Use 3 intermediate amplitude values ($\Omega_1$ to $\Omega_3$) and 3 detuning values ($\delta_1$ to $\delta_3$), which are the parameters that are optimized on, as ratios of the device maximal amplitude and detuning.

The drive starts with an `InterpolatedWaveform` with the points:

- $\Omega = [0, 0.5, 0.9, 0.5, 0] \times \Omega_{\mathrm{hw,max}}$
- $\delta = [-0.8, 0.0, 0.8] \times \delta_{\mathrm{hw,max}}$

### Methods Overview
- `generate(self, register: Register, instance: Instance) -> tuple[Drive, Solution]`:
Runs the Bayesian optimization loop and returns the optimized drive and corresponding solution. Handles fallback cases if simulation fails.

- `build_drive(self, params: list) -> Drive`:
Creates a Drive from a 6-element parameter list: the first 3 for amplitude, the last 3 for detuning.

- `_compute_norm_weights(self, QUBO: torch.Tensor) -> list[float]`:
Normalizes the QUBO diagonal weights (used in DMM shaping).

- `run_simulation(...) -> tuple[...]`:
Runs a simulation of the current drive on a quantum backend and returns bitstring results, probabilities, and QUBO costs.

- `compute_qubo_cost(self, bitstring: str, QUBO: torch.Tensor) -> float`:
Computes the QUBO cost of a specific bitstring.


### Output Structure
After the final round of optimization, the following attributes are populated:

- `drive`: Final Drive object with optimized waveform parameters.
- `best_cost`: Minimum cost found during optimization.
- `best_bitstring`: Corresponding bitstring with the lowest cost.
- `bitstrings, counts, probabilities, costs`: Full result distributions as PyTorch tensors.

### Example

```python exec="on" source="material-block" html="1"
import torch

from qubosolver import Instance, SolverConfig, DriveShapingConfig, Solver, drive_shaping


Q = torch.tensor([[-1.0, 0.5, 0.2], [0.5, -2.0, 0.3], [0.2, 0.3, -3.0]])

instance = Instance(Q)

default_config = SolverConfig(
    use_quantum = True, drive_shaping=DriveShapingConfig(drive_shaping_method=drive_shaping.Algorithm.BAYESIAN_SEARCH, bayesian_search_n_calls = 25),
)
solver = Solver(instance, default_config)

solution = solver.solve()
print(solution)

```
This will return a `Solution` instance, which comprehends the solution bitstrings, the counts of each bitstring, their probabilities and costs.

---
