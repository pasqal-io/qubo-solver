# `CplexSolver` Class Documentation

## Overview

The `CplexSolver` class implements a classical QUBO solver using IBM ILOG CPLEX. It extends the `BaseClassicalSolver` abstract base class and translates a QUBO (Quadratic Unconstrained Binary Optimization) instance into a CPLEX quadratic optimization model.

## Initialization

```python
class CplexSolver(BaseClassicalSolver):
    def __init__(
        self,
        instance: QUBOInstance,
        config: Optional[Dict[str, Any]] = None
    )
```

- **Parameters**:
  - `instance` (`QUBOInstance`): QUBO problem containing a square coefficient matrix (`torch.Tensor`).
  - `config` (`Optional[Dict[str, Any]]`): Dictionary supporting:
    - `cplex_maxtime` (`float`, default `600.0`): Maximum solve time in seconds.
    - `cplex_log_path` (`str`, default `"solver.log"`): Path for CPLEX log output.

## Configuration Options

| Key              | Type    | Default      | Description                                   |
|------------------|---------|--------------|-----------------------------------------------|
| `cplex_maxtime`  | `float` | `600.0`      | Time limit for the CPLEX solver, in seconds.  |
| `cplex_log_path` | `str`   | `"solver.log"` | Log file path for CPLEX output streams.        |

Configure the solver by passing a dict at instantiation:

```python
config = {
    "cplex_maxtime": 300.0,
    "cplex_log_path": "cplex_run.log",
}
solver = CplexSolver(qubo_instance, config)
```

## `solve()` Method

```python
def solve(self) -> QUBOSolution
```

Solves the QUBO problem via CPLEX and returns a `QUBOSolution` containing bitstrings and costs.

### Behavior

1. **Validate Input**
   Raises `ValueError` if `instance.coefficients` is `None`.

2. **Handle Empty Problem**
   If the coefficient matrix has size zero, returns an empty `QUBOSolution`.

3. **Convert to Sparse Format**
   Calls `qubo_instance_to_sparsepairs(instance)` to obtain `List[cplex.SparsePair]` for quadratic terms.

4. **Build CPLEX Model**
   - Instantiate `cplex.Cplex()`.
   - Redirect log, error, warning, and result streams to the file at `cplex_log_path`.
   - Set `timelimit` parameter to `cplex_maxtime`.
   - Specify minimization objective.
   - Add `N` binary variables (`types="B" * N`).
   - Assign quadratic objective via `objective.set_quadratic(sparsepairs)`.

5. **Solve**
   Invoke `problem.solve()`.

6. **Extract Results**
   - Retrieve variable values (`problem.solution.get_values()`) and objective cost (`get_objective_value()`).
   - Close the log file.

7. **Format Output**
   - Build a `torch.Tensor` for bitstrings of shape `(1, N)`, dtype `float32`.
   - Build a `torch.Tensor` for cost of shape `(1,)`, dtype `float32`.
   - Return `QUBOSolution(bitstrings, costs)`.

### Exceptions

- `ValueError`: Thrown if the QUBO instance has no coefficients.

## Example Usage

```python exec="on" source="material-block" html="1"
import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, ClassicalConfig
from qubosolver.solver import QuboSolver

# Define a simple 2×2 QUBO matrix (identity)
matrix = torch.eye(2)
instance = QUBOInstance(coefficients=matrix)

# Prepare solver configuration
cplex = ClassicalConfig(
    classical_solver_type="cplex",
    cplex_maxtime=120.0,
    cplex_log_path="cplex_run.log",
)
config = SolverConfig(
    classical=cplex,
    use_quantum=False
)

# Directly obtain solution via dispatcher
classical_solver = QuboSolver(instance, config)
solution = classical_solver.solve()

print("Bitstrings:", solution.bitstrings)
print("Costs:", solution.costs)
```

---
