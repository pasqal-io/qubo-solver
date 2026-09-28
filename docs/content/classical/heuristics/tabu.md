## `DwaveTabuSolver`

Classical solver using Tabu Search. Designed to integrate with the solver factory.

### Signature
```python
class DwaveTabuSolver(BaseClassicalSolver):
    def solve(self) -> QUBOSolution
```

### Description
This solver applies a Tabu Search metaheuristic to escape local minima and explore the solution space. It is suitable for solving QUBO instances classically without relying on quantum hardware. The implementation is based on `TabuSampler` from the Ocean SDK, and returns solutions compatible with the `QUBOSolution` interface used across the `qubo-solver` package.

## Fields

| Field                  | Type    | Description |
|------------------------|---------|-------------|
| `use_quantum`           | `bool`  | Have to be `False` to uses a classical solver. |
| `classical_solver_type` | `str`   | Set to `"dwave_tabu"` to use Tabu Search as the solving method. |


### Usage
```python exec="on" source="material-block" html="1"
from qubosolver import QUBOInstance
from qubosolver.solver import QuboSolver
from qubosolver.config import SolverConfig, ClassicalConfig

qubo = QUBOInstance(coefficients=[[-2.0, 1.0], [1.0, -2.0]])
config = SolverConfig(use_quantum = False, classical=ClassicalConfig(classical_solver_type="dwave_tabu"))

solver = QuboSolver(qubo, config)

solution = solver.solve()
print(solution)
```

### Notes
Recommended for classical heuristics when reproducibility and control over local search dynamics are desired.

### References

- D-Wave Systems Inc., *Ocean SDK — TabuSampler*
  [doc](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/index.html#tabu)
