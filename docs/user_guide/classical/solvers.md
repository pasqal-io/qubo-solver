# Classical Solvers

Classical solvers tackle a QUBO `Instance` entirely on CPU, without going through the quantum pipeline. Each one is a standalone function under `solving`, callable directly with the instance and its own set of keyword arguments — no `Solver`/`SolverConfig` wiring required. They differ in the trade-off between solution quality, runtime, and whether they need a starting point.

## Without an initial solution

These solvers start from scratch:

- `solving.cplex.solve` — exact MIP solve via IBM CPLEX.
- `solving.random_sampling.solve` — uniform random sampling baseline.

## With an initial solution

These solvers take a starting point. Because they accept a starting point, they can also be used to **refine a previous solution** — for example, post-processing a quantum solver's output:

- `solving.tabu_search.solve` — neighborhood search with a tabu memory. See example below.
- `solving.simulated_annealing.solve` — stochastic temperature-cooling search.
- `solving.iterative_bitflip_local_search.solve` — greedy local search that iteratively flips bits until no single flip improves the solution.

### Example: Tabu Search

```python exec="on" source="tabbed-left" session="tabu" result="text"
from qubosolver import Instance, solving, matrix, bitstrings, torch_rng, analysis

instance = Instance(matrix.tensor([
    [-2.0, 1.0, 0.0, 1.5, 0.0],
    [ 1.0,-1.5, 1.0, 0.0, 0.5],
    [ 0.0, 1.0,-2.0, 1.0, 1.0],
    [ 1.5, 0.0, 1.0,-1.0, 0.5],
    [ 0.0, 0.5, 1.0, 0.5,-1.5],
]))

start = bitstrings.rand(5, instance.size, rng=torch_rng(15))
solution = solving.tabu_search.solve(instance, start, time_limit=10.0)

print("Tabu Search solution:")
print(analysis.to_dataframe([solution]))
```

### Example: refining a quantum solution

Run the quantum pipeline (see [quantum solving](../quantum/intro.md)), then try local bitflips for refinement:

```python exec="on" source="tabbed-left" session="refine" result="text"
from qubosolver import (
    Instance, Solution, LocalEmulator,
    embedding, drive_shaping, solving,
    matrix, analysis,
)
import qoolqit

instance = Instance(matrix.tensor([
    [-2.0, 1.0, 0.0, 1.5, 0.0],
    [ 1.0,-1.5, 1.0, 0.0, 0.5],
    [ 0.0, 1.0,-2.0, 1.0, 1.0],
    [ 1.5, 0.0, 1.0,-1.0, 0.5],
    [ 0.0, 0.5, 1.0, 0.5,-1.5],
]))
device = qoolqit.AnalogDeviceWithDMM()
backend = LocalEmulator()

register = embedding.blade.embed(instance)
drive = drive_shaping.proportional_diagonal.build_drive(instance, register, device=device, dmm=True)
program = solving.analog_quantum_sampling.compile(register, drive, device)
job = backend.run(program)
quantum_solution = Solution.from_results(job.results(), instance)

print("Quantum solution:")
print(analysis.to_dataframe([quantum_solution]))

# Refine the quantum solution classically.
refined_solution = solving.iterative_bitflip_local_search.solve(instance, quantum_solution)

print("Refined solution:")
print(analysis.to_dataframe([refined_solution]))
```

For full parameter details, see the [classical solvers API reference](../../api/classical_solving.md).

## The `Solver` shortcut

For the common case, [`SolverConfig`][qubosolver.SolverConfig] and [`Solver`][qubosolver.Solver] wrap solver selection through [`ClassicalSolvingConfig`][qubosolver.ClassicalSolvingConfig], and (optionally) the initial-solution sampling into a single call:

```python exec="on" source="tabbed-left" session="shortcut" result="text"
from qubosolver import (
    Instance, Solver, SolverConfig, ClassicalSolvingConfig,
    matrix, analysis,
)
from dataclasses import asdict
import pprint

instance = Instance(matrix.tensor([
    [-2.0, 1.0, 0.0, 1.5, 0.0],
    [ 1.0,-1.5, 1.0, 0.0, 0.5],
    [ 0.0, 1.0,-2.0, 1.0, 1.0],
    [ 1.5, 0.0, 1.0,-1.0, 0.5],
    [ 0.0, 0.5, 1.0, 0.5,-1.5],
]))

classical_config = ClassicalSolvingConfig(
    algorithm="tabu_search",
    # algorithm="simulated_annealing",
    # algorithm="cplex",
    )
solver_config = SolverConfig(solving=classical_config)
solver = Solver(instance, solver_config)
solution = solver.solve()

print(pprint.pformat(asdict(solver_config.solving)))
print()
print(analysis.to_dataframe([solution]))
```
