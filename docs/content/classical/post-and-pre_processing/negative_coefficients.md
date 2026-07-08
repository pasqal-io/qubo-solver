# Negative coefficient preprocessing

This page explains how to handle negative off-diagonal QUBO coefficients with:

- bit-flip preprocessing;
- explicit zeroing of remaining negative coefficients.

## Configuration

Negative coefficient preprocessing is controlled from `SolverConfig`.

| Field | Type | Description |
|---|---|---|
| `bitflip_preprocessing` | `BitFlipPreprocessingConfig` | Enables GLPK-based bit-flip preprocessing. |
| `negative_handling` | `"error"` \| `"zeroing"` | Defines what to do if negative off-diagonal coefficients remain after preprocessing. |

The default behavior is:

```python
negative_handling="error"
```

This means that the quantum solver raises an error if negative off-diagonal coefficients remain after preprocessing.

## Bit-flip preprocessing

Bit-flip preprocessing searches for a flip vector.

For each variable:

- `flip_vector[i] = 0`: the variable is kept unchanged;
- `flip_vector[i] = 1`: the variable is complemented.

The transformed QUBO is equivalent to the original problem up to this change of variables.

Internally, the current implementation uses GLPK to select a flip vector that minimizes the remaining negative off-diagonal weight.

```python exec="on" source="material-block"
import torch

from qubosolver import QUBOInstance
from qubosolver.config import BitFlipPreprocessingConfig, SolverConfig
from qubosolver.pipeline.bitflip_preprocessing import has_negative_offdiagonal
from qubosolver.solver import QuboSolverClassical

Q = torch.tensor(
    [
        [1.0, -3.0, -2.0, 1.0],
        [-3.0, 6.0, 3.0, -1.0],
        [-2.0, 3.0, 2.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
    ]
)

config = SolverConfig(
    use_quantum=False,
    do_preprocessing=True,
    activate_trivial_solutions=False,
    bitflip_preprocessing=BitFlipPreprocessingConfig(
        enabled=True,
        time_limit_s=5.0,
    ),
)

solver = QuboSolverClassical(QUBOInstance(Q), config)
solver.preprocess()

print("Bitflip applied:", solver.fixtures.bitflip_applied)
print("Flip vector:", solver.fixtures.bitflip_vector)
print("Bitflip status:", solver.fixtures.bitflip_status)
print("Bitflip metrics:", solver.fixtures.bitflip_metrics)

print("Has negative off-diagonal coefficients after preprocessing:")
print(has_negative_offdiagonal(solver.instance.coefficients))
```

## Classical check

For classical solvers, preprocessing is optional. It can be used to check that the transformed problem gives the same cost after postprocessing.

```python exec="on" source="material-block"
from qubosolver import ClassicalSolverType
from qubosolver.config import ClassicalConfig

classical_config = ClassicalConfig(
    classical_solver_type=ClassicalSolverType.CPLEX,
    max_bitstrings=1,
)

config_without_preprocessing = SolverConfig(
    use_quantum=False,
    do_preprocessing=False,
    activate_trivial_solutions=False,
    classical=classical_config,
)

solver_without_preprocessing = QuboSolverClassical(
    QUBOInstance(Q),
    config_without_preprocessing,
)

solution_without_preprocessing = solver_without_preprocessing.solve()

config_with_bitflip = SolverConfig(
    use_quantum=False,
    do_preprocessing=True,
    activate_trivial_solutions=False,
    classical=classical_config,
    bitflip_preprocessing=BitFlipPreprocessingConfig(
        enabled=True,
        time_limit_s=5.0,
    ),
)

solver_with_bitflip = QuboSolverClassical(
    QUBOInstance(Q),
    config_with_bitflip,
)

solution_with_bitflip = solver_with_bitflip.solve()

print("Best cost without preprocessing:", float(solution_without_preprocessing.costs.min()))
print("Best cost with bit-flip preprocessing:", float(solution_with_bitflip.costs.min()))
```

## Remaining negative coefficients

Bit-flip preprocessing reduces the negative off-diagonal coefficients, but it may not remove all of them.

If `negative_handling="error"`, remaining negative coefficients are kept. The quantum solver will raise an error before embedding.

```python exec="on" source="material-block"
Q_hard = torch.tensor(
    [
        [0.0, -2.0, 1.0, 1.0],
        [-2.0, 0.0, -2.0, 1.0],
        [1.0, -2.0, 0.0, -2.0],
        [1.0, 1.0, -2.0, 0.0],
    ]
)

config_error = SolverConfig(
    use_quantum=False,
    do_preprocessing=True,
    activate_trivial_solutions=False,
    negative_handling="error",
    bitflip_preprocessing=BitFlipPreprocessingConfig(
        enabled=True,
        time_limit_s=5.0,
    ),
)

solver_error = QuboSolverClassical(QUBOInstance(Q_hard), config_error)
solver_error.preprocess()

print("Bitflip applied:", solver_error.fixtures.bitflip_applied)
print("Zeroing applied:", solver_error.fixtures.zeroing_applied)
print("Bitflip metrics:", solver_error.fixtures.bitflip_metrics)

print("Has negative off-diagonal coefficients after preprocessing:")
print(has_negative_offdiagonal(solver_error.instance.coefficients))
```

## Zeroing

Remaining negative coefficients can be explicitly set to zero with:

```python
negative_handling="zeroing"
```

This makes the QUBO compatible with the quantum solver, but it changes the QUBO objective.

```python exec="on" source="material-block"
config_zeroing = SolverConfig(
    use_quantum=False,
    do_preprocessing=True,
    activate_trivial_solutions=False,
    negative_handling="zeroing",
    bitflip_preprocessing=BitFlipPreprocessingConfig(
        enabled=True,
        time_limit_s=5.0,
    ),
)

solver_zeroing = QuboSolverClassical(QUBOInstance(Q_hard), config_zeroing)
solver_zeroing.preprocess()

print("Bitflip applied:", solver_zeroing.fixtures.bitflip_applied)
print("Zeroing applied:", solver_zeroing.fixtures.zeroing_applied)

print("Has negative off-diagonal coefficients after preprocessing:")
print(has_negative_offdiagonal(solver_zeroing.instance.coefficients))
```

## Quantum solving

When using a quantum solver, the final QUBO is checked after preprocessing.

If negative off-diagonal coefficients remain, the solver raises an error before embedding.

```python exec="on" source="material-block"
from qoolqit import AnalogDeviceWithDMM
from pulser_simulation import QutipBackendV2

from qubosolver.config import LocalEmulator
from qubosolver.solver import QuboSolverQuantum

config_quantum_zeroing = SolverConfig(
    use_quantum=True,
    do_preprocessing=True,
    activate_trivial_solutions=False,
    negative_handling="zeroing",
    backend=LocalEmulator(
        backend_type=QutipBackendV2,
        num_shots=100,
    ),
    device=AnalogDeviceWithDMM(),
    bitflip_preprocessing=BitFlipPreprocessingConfig(
        enabled=True,
        time_limit_s=300.0,
    ),
)

quantum_solver = QuboSolverQuantum(QUBOInstance(Q_hard), config_quantum_zeroing)
quantum_solver.preprocess()

print("Has negative off-diagonal coefficients before quantum solve:")
print(has_negative_offdiagonal(quantum_solver.instance.coefficients))
```

The QUBO can then be solved with the selected backend:

```python exec="on" source="material-block"
quantum_solver = QuboSolverQuantum(QUBOInstance(Q_hard), config_quantum_zeroing)
quantum_solution = quantum_solver.solve()

print("Quantum solution bitstrings:")
print(quantum_solution.bitstrings)

print("Quantum solution costs:")
print(quantum_solution.costs)
```

## Notes

Bit-flip preprocessing is the preferred approach because it preserves the QUBO objective up to a variable transformation.

Zeroing should be used only when this approximation is acceptable.

The current bit-flip implementation uses GLPK to compute the flip vector exactly. This is useful for validation and small QUBOs, but it is not expected to scale to large industrial instances.

Future versions should replace this exact GLPK step with an efficient classical heuristic for selecting good flip vectors at larger scale.

Future work should also add a dedicated method to represent negative interactions directly during the quantum resolution.