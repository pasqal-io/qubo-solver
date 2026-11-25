# Transforms

Transforms reduce or change a QUBO `Instance` before solving — fixing variables, or eliminating
negative off-diagonal coefficients — and record enough history to map a `Solution` of the
transformed problem back to the original one. Each is a plain function you call directly.

- API reference: [`qubosolver.transforms`](../../api/transforms.md)

## Apply and lift

Every transform exposes the same pair of functions:

- `apply` takes an `Instance` and returns a *wrapper* `Instance` — a
  subclass that keeps a reference to the parent instance and records what was changed (fixed
  variables, flipped bits, or zeroed edges).
- `lift` takes a `Solution` obtained by solving the wrapper instance, together with that wrapper
  instance, and reconstructs a `Solution` for the original problem: bitstrings are mapped back to
  their original length and ordering, and costs are recomputed against the original QUBO matrix.

### Code example
```python exec="on" source="tabbed-left" session="transforms" result="text"
import numpy as np

from qubosolver import Instance, transforms, solving, bitstrings

instance = Instance(np.asarray([[-2.0, 1.0], [1.0, -2.0]], dtype=np.float64))

reduced_instance = transforms.variable_fixing.apply_recursively(instance)
reduced_solution = solving.brute_force.solve(reduced_instance)
solution = transforms.variable_fixing.lift(reduced_solution, reduced_instance)

print(f"Fixed variables: {reduced_instance.fixed_variables}")
print(f"Full-size bitstrings: {bitstrings.to_strings(solution.bitstrings)}")
print(f"Costs: {solution.costs}")
```

Transforms can be chained: apply one to the output of another, solve the final instance, then
lift back through each transform in reverse order.

## Reduction with variable fixing

Variable fixing deterministically fixes variables to 0 or 1 when the structure of the QUBO matrix
alone already settles their optimal value, without introducing any approximation or randomness —
every fixation is guaranteed to be optimal with respect to the original QUBO. It is particularly
effective for sparse or structured matrices, where many variables can often be fixed early.

The transform lives in `qubosolver.transforms.variable_fixing` and applies **fixation rules**
— functions that inspect the QUBO matrix and return the variables they can fix. `apply` runs each
rule once, in order, reducing the matrix after each rule fixes variables; `apply_recursively`
repeats that until a full pass fixes no additional variables.

The default (and currently only) rule, `hansen_fixing`, is based on the diagonal and off-diagonal
entries of the QUBO matrix. For each variable $i$, it computes a lower bound

$$
c_i + 2 \sum_{j \neq i} \min(0, Q_{ij})
$$

and an upper bound

$$
c_i + 2 \sum_{j \neq i} \max(0, Q_{ij})
$$

on that variable's contribution to the objective; the variable is fixed to $0$ when the lower
bound is already non-negative (it cannot improve the objective by being $1$) and to $1$ when the
upper bound is already non-positive (it can only improve it by being $1$). Several such rules can
be applied in sequence until convergence, reducing the QUBO instance before it is passed to the
solver.

- API reference: [`qubosolver.transforms.variable_fixing`](../../api/transforms.md)

### Code example
```python exec="on" source="tabbed-left" session="varfix" result="text"
import numpy as np

from qubosolver import Instance, transforms, solving, bitstrings

Q = Instance(np.asarray([[-2.0, 1.0], [1.0, -2.0]], dtype=np.float64))

reduced = transforms.variable_fixing.apply_recursively(Q)
reduced_solution = solving.brute_force.solve(reduced)
solution = transforms.variable_fixing.lift(reduced_solution, reduced)

print(f"Fixed variables: {reduced.fixed_variables}")
print(f"Reduced size: {reduced.size}")
print(f"Full-size bitstrings: {bitstrings.to_strings(solution.bitstrings)}")
```

If no variables were fixed, `lift` returns a deep copy of the reduced solution unchanged.

### References

- Hansen, P. (1979). *Method of non-linear 0-1 programming*. Annals of Discrete Mathematics, 5:53–70.

## Negative coefficients bitflip

Quantum (Rydberg) solvers cannot embed attractive (negative off-diagonal) interactions, so a QUBO
must have non-negative off-diagonal coefficients to be embeddable — see
[Embedding](../quantum/embedding.md). Bit-flip preprocessing searches for a change of variables
that removes as much negative off-diagonal weight as possible while preserving the QUBO objective
exactly.

For each variable, a flip vector entry of `0` keeps it unchanged and `1` complements it
(`x_i -> 1 - x_i`); the transformed QUBO is equivalent to the original problem up to this change of
variables. Choosing the flip vector that minimizes the remaining negative off-diagonal weight is
itself a QUBO. `apply` solves it with one of two methods:

- `"local_search"` (default): a local search from random starts plus the no-flip start. It is fast,
  but not guaranteed to be optimal.
- `"tabu_search"`: the same local search, refined with a tabu search.

The heuristics are reproducible when a seeded `rng` is passed to `apply`.

The transform lives in `qubosolver.transforms.negative_bitflip`. `apply` finds the flips, applies
them, and returns an `Instance` that records the flip vector and metrics.

- API reference: [`qubosolver.transforms.negative_bitflip`](../../api/transforms.md)

### Code example
```python exec="on" source="tabbed-left" session="bitflip" result="text"
import numpy as np

from qubosolver import Instance, transforms, numpy_rng

Q_hard = Instance(
    np.asarray(
        [
            [0.0, -2.0, 1.0, 1.0],
            [-2.0, 0.0, -2.0, 1.0],
            [1.0, -2.0, 0.0, -2.0],
            [1.0, 1.0, -2.0, 0.0],
        ],
        dtype=np.float64,
    )
)

reduced_hard = transforms.negative_bitflip.apply(
    Q_hard, time_limit_s=60.0, rng=numpy_rng(223130196)
)

print(f"Flip vector: {reduced_hard.flips}")
print(f"Flipped matrix:\n{reduced_hard.matrix}")
```

Bit-flip preprocessing reduces the negative off-diagonal coefficients, but — as in the example
above — it may not remove all of them.

## Negative coefficients zeroing

If negative off-diagonal coefficients remain after bit-flip preprocessing, `zeroing.apply` sets
each of them to zero, applied on top of the bit-flip result. This makes the QUBO embeddable, but
it changes the QUBO objective, unlike bit-flip preprocessing — so it should be used only when this
approximation is acceptable, as a last resort.

The transform lives in `qubosolver.transforms.zeroing`. `apply` returns a new `Instance`; it does
not modify its argument in place.

- API reference: [`qubosolver.transforms.zeroing`](../../api/transforms.md)

### Code example
```python exec="on" source="tabbed-left" session="zeroing" result="text"
import numpy as np

from qubosolver import Instance, transforms, numpy_rng

Q_hard = Instance(
    np.asarray(
        [
            [0.0, -2.0, 1.0, 1.0],
            [-2.0, 0.0, -2.0, 1.0],
            [1.0, -2.0, 0.0, -2.0],
            [1.0, 1.0, -2.0, 0.0],
        ],
        dtype=np.float64,
    )
)

reduced_hard = transforms.negative_bitflip.apply(
    Q_hard, time_limit_s=60.0, rng=numpy_rng(223130196)
)
zeroed_hard = transforms.zeroing.apply(reduced_hard)

print(f"Zeroed edges:\n{zeroed_hard.zeroed_edges}")
print(f"Negative matrix:\n{zeroed_hard.negative_matrix}")
```

Zeroing is never applied implicitly. The embedders reject instances with negative off-diagonal
coefficients, so before a quantum solve, call `transforms.negative_bitflip.apply` and, if negative
coefficients remain, `transforms.zeroing.apply`, then lift the solution back through both in
reverse order. Classical solvers have no such restriction.
