## `TabuSearchSolver`

Classical solver using Tabu Search. Designed to integrate with the solver factory.

### Signature

```python
class TabuSearchSolver(BaseClassicalSolver):
    def solve(self) -> QUBOSolution
```

### Description

This solver applies a Tabu Search metaheuristic to escape local minima and explore the solution space. It is suitable for solving QUBO instances classically without relying on quantum hardware. The implementation is based on `TabuSearchSolver` from the Ocean SDK, and returns solutions compatible with the `QUBOSolution` interface used across the `qubo-solver` package.

## Fields

| Field                   | Type                    | Description                                                                                |
| ----------------------- | ----------------------- | ------------------------------------------------------------------------------------------ |
| `use_quantum`           | `bool`                  | Have to be `False` to use a classical solver.                                              |
| `classical_solver_type` | `str`                   | Set to `"tabu_search"` to use Tabu Search as the solving method.                           |
| `tabu_x0`               | `torch.Tensor` | `None` | The initial binary solution tensor.                                                        |
| `tabu_tenure`           | `int`                   | Number of iterations a move (bit flip) remains tabu.                                       |
| `tabu_max_no_improve`   | `int`                   | Maximum number of consecutive iterations without improvement before termination.           |
| `tabu_time_limit`       | `float`                 | Maximum execution time for Tabu Search, in seconds. If infinite, no time limit is applied. |

### Usage

```python exec="on" source="material-block" html="1"
from qubosolver import QUBOInstance
from qubosolver.solver import QuboSolver
from qubosolver.config import SolverConfig, ClassicalConfig

qubo = QUBOInstance(coefficients=[[-2.0, 1.0], [1.0, -2.0]])
config = SolverConfig(use_quantum = False, classical=ClassicalConfig(classical_solver_type="tabu_search", tabu_time_limit=300.0))

solver = QuboSolver(qubo, config)

solution = solver.solve()
print(solution)
```

### Notes

Recommended for classical heuristics when reproducibility and control over local search dynamics are desired.
