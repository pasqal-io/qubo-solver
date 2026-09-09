# Transforms

Transforms reduce or change a QUBO `Instance` before solving — fixing variables, or eliminating
negative off-diagonal coefficients — and record enough history to map a `Solution` of the
transformed problem back to the original one. Each is callable directly, or chained automatically
by `Solver`/`SolverConfig` via `preprocessing=True`.

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
from qubosolver import Instance, transforms, solving, matrix, bitstrings

instance = Instance(matrix.tensor([[-2.0, 1.0], [1.0, -2.0]]))

reduced_instance = transforms.variable_fixing.apply_recursively(instance)
reduced_solution = solving.brute_force.solve(reduced_instance)
solution = transforms.variable_fixing.lift(reduced_solution, reduced_instance)

print(f"Fixed indices: {reduced_instance.fixed_indices}")
print(f"Full-size bitstrings: {bitstrings.to_strings(solution.bitstrings)}")
print(f"Costs: {solution.costs}")
```

Transforms can be chained: apply one to the output of another, solve the final instance, then
lift back through each transform in reverse order. `Solver` does this for you (see
[The `Solver` shortcut](#the-solver-shortcut)).

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
from qubosolver import Instance, transforms, solving, matrix, bitstrings

Q = Instance(matrix.tensor([[-2.0, 1.0], [1.0, -2.0]]))

reduced = transforms.variable_fixing.apply_recursively(Q)
reduced_solution = solving.brute_force.solve(reduced)
solution = transforms.variable_fixing.lift(reduced_solution, reduced)

print(f"Fixed indices: {reduced.fixed_indices}")
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
variables. The current implementation uses GLPK to solve an integer linear program that selects
the flip vector minimizing the remaining negative off-diagonal weight — exact, which is useful for
validation and small QUBOs, but not expected to scale to large industrial instances.

The transform lives in `qubosolver.transforms.negative_bitflip`. `apply` solves the ILP, applies
the flips, and returns an `Instance` that records the flip vector, status, and metrics.

- API reference: [`qubosolver.transforms.negative_bitflip`](../../api/transforms.md)

### Code example
```python exec="on" source="tabbed-left" session="bitflip" result="text"
import json
from qubosolver import Instance, transforms, matrix

Q_hard = Instance(matrix.tensor(
    [
        [0.0, -2.0, 1.0, 1.0],
        [-2.0, 0.0, -2.0, 1.0],
        [1.0, -2.0, 0.0, -2.0],
        [1.0, 1.0, -2.0, 0.0],
    ]
))

reduced_hard = transforms.negative_bitflip.apply(Q_hard, time_limit_s=60.0)

print(f"Flip vector: {reduced_hard.flips}")
print(f"Bitflip status: {reduced_hard.status}")
print(f"Bitflip metrics: {json.dumps(reduced_hard.metrics, indent=4)}")
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
from qubosolver import Instance, transforms, matrix

Q_hard = Instance(matrix.tensor(
    [
        [0.0, -2.0, 1.0, 1.0],
        [-2.0, 0.0, -2.0, 1.0],
        [1.0, -2.0, 0.0, -2.0],
        [1.0, 1.0, -2.0, 0.0],
    ]
))

reduced_hard = transforms.negative_bitflip.apply(Q_hard, time_limit_s=60.0)
zeroed_hard = transforms.zeroing.apply(reduced_hard)

print(f"Zeroed edges:\n{zeroed_hard.zeroed_edges}")
print(f"Negative matrix:\n{zeroed_hard.negative_matrix}")
```

Zeroing is only ever applied automatically by the `Solver`/`SolverConfig` shortcut below, and only
when solving with a `QuantumSolvingConfig` — regardless of whether `preprocessing` is enabled.
Before embedding, the final QUBO is always checked, and if negative off-diagonal coefficients
remain, the quantum solver zeros them out automatically (logging a message) so it can be embedded
— there is no separate flag to request this, and no flag to opt out of it or make it raise
instead. Classical solvers have no such restriction and never trigger this fallback.

## The `Solver` shortcut

Going through the `Solver` dispatcher with a `QuantumSolvingConfig` chains the transforms above
for you:

1. if `SolverConfig.preprocessing=True`, variable-fixing reduction
   (`transforms.variable_fixing.apply_recursively`) followed by negative-coefficients bit-flip
   preprocessing (`transforms.negative_bitflip.apply`);
2. regardless of `preprocessing`, if negative off-diagonal coefficients remain, automatic zeroing
   (`transforms.zeroing.apply`) — classical solvers skip this step entirely.

After solving, `Solver` lifts the solution back through each transform in reverse order, so the
returned `Solution` always refers to the bitstrings and costs of the *original* instance you
passed in.

### Code example
```python exec="on" source="tabbed-left" session="shortcut" result="text"
from qubosolver import Instance, Solver, SolverConfig, QuantumSolvingConfig, matrix

Q_hard = Instance(matrix.tensor(
    [
        [0.0, -2.0, 1.0, 1.0],
        [-2.0, 0.0, -2.0, 1.0],
        [1.0, -2.0, 0.0, -2.0],
        [1.0, 1.0, -2.0, 0.0],
    ]
))

config = SolverConfig(
    solving=QuantumSolvingConfig(),
    preprocessing=True,
    activate_trivial_solutions=False,
)

# Bit-flip preprocessing cannot remove every negative off-diagonal coefficient here;
# the quantum solver zeros out the remainder automatically before embedding (see the
# logged message).
solution = Solver(Q_hard, config).solve()

print(f"Solution bitstrings shape: {solution.bitstrings.shape}")
print(f"Original instance size: {Q_hard.size}")
```

!!! note
    The `Solver` API does not let you run variable-fixing, bit-flip preprocessing, and zeroing
    independently of each other. If you need that flexibility — for example, to inspect the QUBO
    after bit-flip preprocessing but before deciding whether to zero it — use the functional API
    directly, as shown in the sections above: `transforms.variable_fixing.apply`,
    `transforms.negative_bitflip.apply`, and `transforms.zeroing.apply` can each be called on
    their own, and their matching `lift` functions used to map solutions back afterwards.
