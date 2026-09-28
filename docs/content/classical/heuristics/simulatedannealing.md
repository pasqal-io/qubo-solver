## `DwaveSASolver`

Simple classical solver class using Simulated Annealing. Designed to integrate with the solver factory.

### Signature
```python
class DwaveSASolver(BaseClassicalSolver):
    def solve(self) -> QUBOSolution
```

### Description
This solver uses a Simulated Annealing backend to probabilistically explore the solution space. It is suitable for approximating solutions on medium-sized QUBO instances. Computation is entirely classical and based on the `SimulatedAnnealingSampler` from DWave's Ocean SDK [add the ref below ?]. The output is fully compatible with the `QUBOSolution` structure used in the `qubo-solver` package.

## Fields

| Field                  | Type    | Description |
|------------------------|---------|-------------|
| `use_quantum`           | `bool`  | Have to be `False` to uses a classical solver. |
| `classical_solver_type` | `str`   | Set to `"dwave_sa"` to use Simulated Annealing as the solving method. |


### Usage
```python exec="on" source="material-block" html="1"
from qubosolver import QUBOInstance
from qubosolver.solver import QuboSolver
from qubosolver.config import SolverConfig, ClassicalConfig

qubo = QUBOInstance(coefficients=[[-2.0, 1.0], [1.0, -2.0]])
config = SolverConfig(use_quantum = False, classical=ClassicalConfig(classical_solver_type="dwave_sa"))

solver = QuboSolver(qubo, config)

solution = solver.solve()
print(solution)
```

### Notes
Recommended for local, classical solving when exact optimization is not required.

### References

- D-Wave Systems Inc., *Ocean SDK — SimulatedAnnealingSampler*
  [doc](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/)
