# `postprocess` Method Documentation

## Overview

The `postprocess` method of the `Fixtures` class refines raw QUBO solutions by applying a local bit-flip search to each candidate bitstring. It evaluates each modified solution against the original QUBO instance, aiming to lower the objective cost.

## Method Signature

```python
def postprocess(self, solution: QUBOSolution) -> QUBOSolution
```

### Parameters

- `solution` (`QUBOSolution`): The QUBO solver's output, containing:
  - `bitstrings`: Tensor of shape `(num_samples, num_variables)` representing candidate solutions.
  - `costs`: Tensor of shape `(num_samples,)` with corresponding objective values.
  - Optional `counts` and `probabilities` attributes.

### Returns

- `QUBOSolution`: A new solution object where each bitstring has been locally optimized and costs recomputed. It includes:
  - Updated `bitstrings` (dtype `float32`) of shape `(num_samples, num_variables)`.
  - Updated `costs` (dtype `float32`) of shape `(num_samples,)`.
  - `solution_status`: A description indicating postprocessing and listing the improved costs.
  - Preserved `counts` and `probabilities` if originally present.

## Behavior

1. **No-op for Empty Solutions**
   If `solution.bitstrings` is empty, the method returns the input `solution` unchanged.

2. **Objective Wrapper**
   Defines an inner function:
   ```python
   def qubo_objective(s_arr: np.ndarray) -> float:
       return self.instance.evaluate_solution(s_arr.tolist())
   ```
   to compute the cost of any candidate bitstring via the original QUBO instance.

3. **Local Bit-Flip Search**
   For each bitstring in `solution.bitstrings`:
   - Convert the tensor row to a NumPy integer array.
   - Call `bit_flip_local_search(qubo_objective, s_orig)` to greedily flip bits and reduce cost.
   - Collect the improved bitstring and its new cost.

4. **Assemble Tensors**
   Stack all improved bitstrings and costs into new PyTorch tensors (`dtype=torch.float32`).

5. **Update and Return**
   - Replace `solution.bitstrings` and `solution.costs` with the improved tensors.
   - Set `solution.solution_status` to:
     ```text
     postprocessed (improved costs: [c1, c2, …])
     ```
   - Return the modified `QUBOSolution`.

## Example Usage with `Fixtures`

```python exec="on" source="material-block" html="1"
import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, ClassicalConfig
from qubosolver.pipeline import Fixtures
from qubosolver.solver import QuboSolver

# Create a random 4-variable QUBO instance
graphics = torch.randn(4, 4)
graphics = (graphics + graphics.T) / 2  # Ensure symmetry
n = graphics.size(0)
off_diag_mask = ~torch.eye(n, dtype=torch.bool)
graphics[off_diag_mask] = graphics[off_diag_mask].abs() # Ensure Abs off-diagonal
qubo = QUBOInstance(coefficients=graphics)

# Configure solver to enable postprocessing
cplex = ClassicalConfig(classical_solver_type="cplex", cplex_log_path="solver.log", cplex_maxtime=300.0,)
config = SolverConfig(
    use_quantum=False,
    classical=cplex,
    do_postprocessing=True
)

# Solve with classical solver
classical_solver = QuboSolver(qubo, config)
raw_solution = classical_solver.solve()

# Apply postprocessing
fixture = Fixtures(qubo, config)
final_solution = fixture.postprocess(raw_solution)

print("Optimized bitstrings:", final_solution.bitstrings)
print("Optimized costs:", final_solution.costs)
print("Status:", final_solution.solution_status)
```

---

---

## Example Usage without Direct Fixture Manipulation

Instead of manually instantiating `Fixtures`, you can enable postprocessing directly through your solver configuration:

```python exec="on" source="material-block" html="1"
import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, ClassicalConfig
from qubosolver.solver import QuboSolver
import emu_mps
from qoolqit._solvers.types import BackendType


# Assume `first_qubo_coefficients` is your 2×2 QUBO matrix (e.g., identity):
first_qubo_coefficients = torch.eye(2)
instance = QUBOInstance(coefficients=first_qubo_coefficients)

# Configure solver with postprocessing enabled
cplex = ClassicalConfig(classical_solver_type="cplex", cplex_log_path="solver.log", cplex_maxtime=300.0,)
config = SolverConfig(
    classical=cplex,
    do_postprocessing=True,                  # Enable postprocessing
    use_quantum=False
)

# Instantiate and run the classical solver
classical_solver = QuboSolver(instance, config)
solution = classical_solver.solve()

print("Final bitstrings:", solution.bitstrings)
print("Final costs:", solution.costs)
print("Status:", solution.solution_status)
```

---
