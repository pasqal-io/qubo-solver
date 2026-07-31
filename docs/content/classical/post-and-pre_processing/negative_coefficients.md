# Negative coefficient preprocessing

This page explains how to handle negative off-diagonal QUBO coefficients with:

- bit-flip preprocessing;
- explicit zeroing of remaining negative coefficients.

## Configuration

Negative coefficient preprocessing is controlled from `SolverConfig.do_preprocessing`.

When `do_preprocessing=True`, the solver chains variable-fixing and bit-flip preprocessing
before solving. If negative off-diagonal coefficients remain afterwards, the quantum solver
automatically zeros them out (logging a message) before embedding, since the backend cannot
encode negative interactions. There is no configuration flag to opt out of this fallback or to
make it raise instead: the `Solver` API always runs the full chain the same way.

!!! note
    The `Solver` API does not let you run bit-flip preprocessing and zeroing independently of
    each other, or independently of variable-fixing. If you need that flexibility — for example,
    to inspect the QUBO after bit-flip preprocessing but before deciding whether to zero it — use
    the functional API directly, as shown below: `transforms.negative_bitflip.apply` and
    `transforms.zeroing.apply` can each be called on their own.

## Bit-flip preprocessing

Bit-flip preprocessing searches for a flip vector.

For each variable:

- `flip_vector[i] = 0`: the variable is kept unchanged;
- `flip_vector[i] = 1`: the variable is complemented.

The transformed QUBO is equivalent to the original problem up to this change of variables.

Internally, the current implementation uses GLPK to select a flip vector that minimizes the remaining negative off-diagonal weight.

The transform lives in `qubosolver.transforms.negative_bitflip`. Calling
`negative_bitflip.apply` solves the ILP, applies the flips, and returns an `Instance`
that records the flip vector, status, and metrics.

```python exec="on" source="tabbed-left" session="negative" result="text"
import json
from qubosolver import Instance, transforms, matrix, Analyzer

Q = Instance(matrix.tensor(
    [
        [1.0, -3.0, -2.0, 1.0],
        [-3.0, 6.0, 3.0, -1.0],
        [-2.0, 3.0, 2.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
    ]
))

reduced = transforms.negative_bitflip.apply(Q, time_limit_s=5.0)

print(f"Flip vector: {reduced.flips}")
print(f"Bitflip status: {reduced.status}")
print(f"Bitflip metrics: {json.dumps(reduced.metrics, indent=4)}")
```

## Classical check

For classical solvers, preprocessing is optional. It can be used to check that the transformed problem gives the same cost after postprocessing.

End-to-end solving goes through the public `Solver` dispatcher, which picks the
classical or quantum solver from the `SolverConfig`.

```python exec="on" source="tabbed-left" session="negative" result="text"
from qubosolver import Solver, SolverConfig, solvers

config_without_preprocessing = SolverConfig(
    use_quantum=False,
    do_preprocessing=False,
    activate_trivial_solutions=False,
)

solution_without_preprocessing = Solver(
    Q,
    config_without_preprocessing,
).solve()

config_with_bitflip = SolverConfig(
    use_quantum=False,
    do_preprocessing=True,
    activate_trivial_solutions=False,
)

solution_with_bitflip = Solver(
    Q,
    config_with_bitflip,
).solve()

print(f"Best cost without preprocessing: {solution_without_preprocessing[0].cost}")
print(f"Best cost with bit-flip preprocessing: {solution_with_bitflip[0].cost}")
```

## Remaining negative coefficients

Bit-flip preprocessing reduces the negative off-diagonal coefficients, but it may not remove all of them.

```python exec="on" source="tabbed-left" session="negative" result="text"
Q_hard = Instance(matrix.tensor(
    [
        [0.0, -2.0, 1.0, 1.0],
        [-2.0, 0.0, -2.0, 1.0],
        [1.0, -2.0, 0.0, -2.0],
        [1.0, 1.0, -2.0, 0.0],
    ]
))

reduced_hard = transforms.negative_bitflip.apply(Q_hard, time_limit_s=5.0)

print(f"Bitflip status: {reduced_hard.status}")
print(f"Bitflip metrics: {json.dumps(reduced_hard.metrics, indent=4)}")
```

## Zeroing

If negative off-diagonal coefficients remain after bit-flip preprocessing, they can be
explicitly set to zero with `zeroing.apply`, applied on top of the bit-flip result.

This makes the QUBO compatible with the quantum solver, but it changes the QUBO objective. It
should therefore be used only when this approximation is acceptable.

`zeroing.apply` returns a new `Instance`; it does not modify its argument in place.

```python exec="on" source="tabbed-left" session="negative" result="text"
zeroed_hard = transforms.zeroing.apply(reduced_hard)

print(f"Zeroed edges:\n{zeroed_hard.zeroed_edges}")
print(f"Negative matrix:\n{zeroed_hard.negative_matrix}")
```

## Quantum solving

When using the `Solver` API with `use_quantum=True`, the final QUBO is checked after
preprocessing. If negative off-diagonal coefficients remain, the quantum solver zeros them out
automatically (logging a message) so the QUBO can be embedded and solved with the selected
backend — there is no separate flag to request this.

```python exec="on" source="tabbed-left" session="negative" result="text"
from qoolqit import AnalogDeviceWithDMM
from pulser_simulation import QutipBackendV2

from qubosolver import LocalEmulator

config_quantum = SolverConfig(
    use_quantum=True,
    do_preprocessing=True,
    activate_trivial_solutions=False,
)

quantum_solution = Solver(Q_hard, config_quantum).solve()

print(f"Quantum solution:\n{Analyzer(quantum_solution).df}")
```

To control bit-flip preprocessing and zeroing independently — for example to zero a QUBO without
running bit-flip preprocessing first, or to inspect intermediate results — call
`transforms.negative_bitflip.apply` and `transforms.zeroing.apply` directly, as in the sections
above, and pass the resulting `Instance` to `embedding`/`drive_shaping`/`solvers` yourself instead
of going through `Solver`.

## Notes

Bit-flip preprocessing is the preferred approach because it preserves the QUBO objective up to a variable transformation.

Zeroing should be used only when this approximation is acceptable.

The current bit-flip implementation uses GLPK to compute the flip vector exactly. This is useful for validation and small QUBOs, but it is not expected to scale to large industrial instances.

Future versions should replace this exact GLPK step with an efficient classical heuristic for selecting good flip vectors at larger scale.

Future work should also add a dedicated method to represent negative interactions directly during the quantum resolution.
