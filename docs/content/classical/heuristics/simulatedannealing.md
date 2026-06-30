## `SimulatedAnnealingSolver`

Simple classical solver class using Simulated Annealing. Designed to integrate with the solver factory.

### Signature

```python
class SimulatedAnnealingSolver(BaseClassicalSolver):
    def solve(self) -> QUBOSolution
```

### Description

This solver uses a Simulated Annealing to probabilistically explore the solution space. It is suitable for approximating solutions on medium-sized QUBO instances. Computation is entirely classical and based on the `SimulatedAnnealingSolver`. The output is fully compatible with the `QUBOSolution` structure used in the `qubo-solver` package.

## Fields

| Field                   | Type              | Description                                                                                        |
| ----------------------- | ----------------- | -------------------------------------------------------------------------------------------------- |
| `use_quantum`           | `bool`            | Have to be `False` to use a classical solver.                                                      |
| `classical_solver_type` | `str`             | Set to `"simulated_annealing"` to use Simulated Annealing as the solving method.                   |
| `max_iter`              | `int`             | Maximum number of iterations to perform for simulated annealing or tabu search.                    |
| `sa_initial_temp`       | `float`           | Starting temperature (controls exploration).                                                       |
| `sa_final_temp`         | `float`           | Minimum temperature threshold for stopping.                                                        |
| `sa_alpha`              | `float`           | Cooling rate - should be slightly below 1 (e.g., 0.95–0.99).                                       |
| `sa_time_limit`         | `float`           | Maximum execution time for simulated annealing, in seconds. If infinite, no time limit is applied. |

### Usage

```python exec="on" source="material-block" html="1"
from qubosolver import QUBOInstance, QuboSolver, SolverConfig, ClassicalConfig, matrix

qubo = QUBOInstance(matrix=matrix.tensor([[-2.0, 1.0], [1.0, -2.0]]))
config = SolverConfig(
    use_quantum=False,
    classical=ClassicalConfig(
        classical_solver_type="simulated_annealing",
        max_iter=1000,
        sa_time_limit=300.0,
    ),
)

solver = QuboSolver(qubo, config)

solution = solver.solve()
print(solution)
```

### Notes

Recommended for local, classical solving when exact optimization is not required.
