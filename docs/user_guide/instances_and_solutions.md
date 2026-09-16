# Instances and Solutions

Two classes sit at the center of `qubosolver`'s API: [`Instance`][qubosolver.types.instance.Instance], the problem to solve, and [`Solution`][qubosolver.types.solution.Solution], the result a solver returns. Every solver in the package (`qubosolver.solving`) takes an `Instance` and produces a `Solution`.

## Instance

`Instance` represents a single QUBO problem. It wraps the coefficient matrix $Q$ and exposes helpers to evaluate candidate solutions and inspect the problem.

### Features

- Store the QUBO coefficient matrix (`matrix`) and its size (`size`, also available as `len(instance)`).
- Evaluate a candidate bitstring's cost via `instance.cost(bitstring)`.
- Serialize to and from disk.

### Code example

```python exec="on" source="tabbed-left" session="instance" result="text"
from qubosolver import Instance, matrix, bitstring

instance = Instance(matrix.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]]))

solution = bitstring.from_string("101")
cost = instance.cost(solution)
print(f"Solution Cost: {cost}")
```

### Save / Load

```python exec="on" source="tabbed-left" session="instance" result="text"
import tempfile
from pathlib import Path

file = Path(tempfile.mkdtemp()) / "qubo_instance.bin"

with file.open("wb") as f:
    instance.save(f)

with file.open("rb") as f:
    loaded_instance = Instance.load(f)

print(loaded_instance.matrix)
```

### Transforms

Some preprocessing steps in `qubosolver.transforms` (variable fixing, zeroing, negative bitflip) reduce a QUBO problem before solving it, and record what they did as `Instance` subclasses. Wrapping an `Instance` this way keeps the applied transform attached to the problem, so it can later be used to lift a solution of the reduced problem back to a solution of the original one:

```python exec="on" source="tabbed-left" result="text"
from qubosolver import Instance, matrix, solving, transforms, analysis

instance = Instance(matrix.tensor([
    [10.0, 1.0, 1.0],
    [ 1.0, -3.0, 2.0],
    [ 1.0, 2.0, -1.0],
]))

reduced_instance = transforms.variable_fixing.apply_recursively(instance)
reduced_solution = solving.brute_force.solve(reduced_instance)
solution = transforms.variable_fixing.lift(reduced_solution, reduced_instance)

print(f"Reduced size: {reduced_instance.size}, original size: {instance.size}")
print(f"Best bitstring: {solution[0].string}, cost: {solution[0].cost}")
print(analysis.to_dataframe([solution]))
```

`Instance.load` dispatches automatically to whichever subclass wrote the file, so a plain `Instance.load(f)` correctly restores a transformed instance produced by any of these transforms.

## Solution

`Solution` represents a collection of candidate bitstrings for a QUBO problem, together with their costs, sample counts, and probabilities. Solvers return one `Solution`, sorted by ascending cost, so the best candidate is always first.

### Features

- Store candidate `bitstrings`, their `costs`, `counts`, and `probabilities`.
- Iterable: iterating (or indexing) a `Solution` yields [`Candidate`][qubosolver.types.solution.Candidate] objects, one per candidate.
- Serialize to and from disk.

### Code example

```python exec="on" source="tabbed-left" session="solution" result="text"
from qubosolver import Instance, Solution, matrix, solving

instance = Instance(matrix.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]]))

solution = solving.brute_force.solve(instance)

# Best candidate first
best = solution[0]
print(f"Best bitstring: {best.string}, cost: {best.cost}")

# Iterate over every candidate
for candidate in solution:
    print(candidate.string, candidate.cost, candidate.probability)
```

### Save / Load

```python exec="on" source="tabbed-left" session="solution" result="text"
import tempfile
from pathlib import Path
from qubosolver import analysis

file = Path(tempfile.mkdtemp()) / "qubo_solution.bin"

with file.open("wb") as f:
    solution.save(f)

with file.open("rb") as f:
    loaded_solution = Solution.load(f)

print(analysis.to_dataframe([loaded_solution]))
```
