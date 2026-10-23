# Embedding workflow

Embedding maps a QUBO instance's variables onto atom positions on a quantum device, so that the physical interaction between atoms approximates the QUBO's coefficients. Each embedding algorithm is callable directly, or picked and configured through `Solver`/`SolverConfig`.

## Goal of embedding

Solving a QUBO on a neutral-atom quantum device requires translating the problem's coefficients into a physical layout of atoms. Each QUBO variable $i$ is represented by an atom, placed at some position on the device. The interaction strength between two atoms is a function of the physical distance between them (typically decaying with distance, e.g. as $1/r^6$ for Rydberg interactions), whereas the QUBO matrix specifies the desired coupling strength $Q_{ij}$ between each pair of variables $i$ and $j$.

The goal of embedding is therefore to find atom positions such that the interaction strengths induced by their pairwise distances reproduce, as closely as possible, the couplings encoded in the QUBO matrix — subject to the constraints of the target device (register size, minimal distance between atoms, layout geometry, etc.). A good embedding is what allows the quantum device to encode the optimization problem faithfully, so that its ground state (or low-energy states) correspond to good solutions of the original QUBO.

Because the induced interaction $J_{ij} \propto 1/r_{ij}^6$ is always positive, an embedding can only reproduce non-negative off-diagonal couplings. QUBO matrices with negative off-diagonal coefficients cannot be embedded directly; see [Negative coefficient preprocessing](../classical/transforms.md) for how such coefficients are eliminated or zeroed out before embedding.

Since embedding is an approximation, it is worth having a way to judge how good it is. The relative norm error between the QUBO matrix and the induced interaction matrix (shown in the examples below) is the most direct such proxy, and cheap to compute. But it is only a proxy: what actually matters is whether the embedding leads the quantum device to good solutions of the original QUBO, and a lower matrix error does not always mean a better solution — e.g. a small number of badly reproduced couplings on variables that matter more to the optimum can hurt more than a uniformly larger error spread evenly across all pairs.

See the [embedding overview notebook](../../tutorials/03-embedding/00-embedding.ipynb) for a hands-on introduction to this step.

`qubosolver` currently ships two embedding algorithms — BLaDE and greedy layout, described below — plus the option to write your own embedding function or supply a precomputed register.

!!! warning
    Each algorithm exposes two entry points: `embed(instance, config=...)` and `embed_for_device(instance, device)`. `embed` takes a fully-resolved `config` as-is and does not look at any device, so use it to tune parameters freely — but tread carefully with device-dependent ones: a value unsuited to your target device can produce a register that device cannot realize. `embed_for_device` derives those device-dependent fields automatically and is the safer default whenever you have a target device. See the [API reference](../../api/embedding.md) for the exact fields each algorithm derives.

## BLaDE

BLaDE places atoms by progressively projecting the problem from a high-dimensional layout down to the device's 2D plane, refining positions layer by layer so that pairwise distances keep approximating the QUBO couplings at each step.

- API reference: [`qubosolver.embedding.blade`](../../api/embedding.md)
- Tutorial: [BLaDE notebook](../../tutorials/03-embedding/01-blade.ipynb)
- Under the hood: [BLaDE](../../under_the_hood/embedding/blade.md)

The example below runs BLaDE sized for a target device via `embed_for_device` — the sequence of dimension layers, number of steps per round, and starting positions can all be tuned by combining `embedding.blade.Config(device=...)` with `embed` directly. See [Qoolqit's documentation](https://docs.pasqal.com/qoolqit/reference/embedding/#qoolqit.embedding.BladeConfig) for the available parameters.

### Code example
```python exec="on" source="tabbed-left" session="embedding" result="text"
from qubosolver import Instance, matrix, embedding
import torch
import qoolqit

# Private utility to set seed.
from qubosolver.utils._random import manual_seed

manual_seed(958)

instance = Instance(
    matrix.tensor(
        [
            [0, 1, 2, 1],
            [1, 0, 3, 0],
            [2, 3, 0, 5],
            [1, 0, 5, 0],
        ]
    )
)
register = embedding.blade.embed_for_device(instance, qoolqit.AnalogDevice())
interaction_matrix = matrix.as_tensor(register.interaction_matrix())

torch.set_printoptions(precision=2)
print(f"QUBO matrix:\n{instance.matrix}\n")
print(f"Interaction matrix:\n{interaction_matrix}\n")

relative_error = 100 * (instance.matrix - interaction_matrix).norm() / instance.matrix.norm()
print(f"Relative error = {relative_error:.2f} %")
```
Plot the resulting register:

```python exec="on" source="tabbed-right" session="embedding"
from pathlib import Path
from matplotlib import pyplot as plt

output_dir = Path.cwd()
output_dir = Path("{{ config.site_dir.split('"')[0] }}/user_guide/quantum/embedding")  # markdown-exec: hide
output_dir.mkdir(parents=True, exist_ok=True)

register.draw()
fig = plt.gcf()
fig.savefig(output_dir / "quantum_embedding_blade_register.svg")
plt.close(fig)

print('<figure><figcaption>Register</figcaption><img src="quantum_embedding_blade_register.svg" alt="Register of atoms embedding the QUBO instance"></figure>')  # markdown-exec: hide
```

Note that by default, before running BLaDE, the positions are initialized using MDS (Multi-Dimensional Scaling). You can disable that by instantiating a config with
`embedding.blade.Config(initialize_with_mds=False)`.

## Greedy embedder

The greedy embedder picks atom positions from a fixed lattice of candidate trap sites (triangular or square), assigning variables one at a time to the site that best matches their required couplings to already-placed variables.

- API reference: [`qubosolver.embedding.greedy_layout`](../../api/embedding.md)
- Tutorial: [Greedy layout notebook](../../tutorials/03-embedding/02-greedy-layout.ipynb)
- Under the hood: [Greedy Layout](../../under_the_hood/embedding/greedy_layout.md)

### Code example
```python exec="on" source="tabbed-left" session="embedding" result="text"
from qubosolver import Instance, matrix, embedding
import torch
import qoolqit

# Private utility to set seed.
from qubosolver.utils._random import manual_seed

manual_seed(851)

instance = Instance(
    matrix.tensor(
        [
            [0, 1, 2, 1],
            [1, 0, 3, 0],
            [2, 3, 0, 5],
            [1, 0, 5, 0],
        ]
    )
)

register = embedding.greedy_layout.embed_for_device(
    instance,
    qoolqit.AnalogDevice(),
)
interaction_matrix = matrix.as_tensor(register.interaction_matrix())

torch.set_printoptions(precision=2)
print(f"QUBO matrix:\n{instance.matrix}\n")
print(f"Interaction matrix:\n{interaction_matrix}\n")

relative_error = 100 * (instance.matrix - interaction_matrix).norm() / instance.matrix.norm()
print(f"Relative error = {relative_error:.2f} %")
```
Plot the resulting register:

```python exec="on" source="tabbed-right" session="embedding"
from pathlib import Path
from matplotlib import pyplot as plt

output_dir = Path.cwd()
output_dir = Path("{{ config.site_dir.split('"')[0] }}/user_guide/quantum/embedding")  # markdown-exec: hide
output_dir.mkdir(parents=True, exist_ok=True)

register.draw()
fig = plt.gcf()
fig.savefig(output_dir / "quantum_embedding_greedy_layout_register.svg")
plt.close(fig)

print('<figure><figcaption>Register</figcaption><img src="quantum_embedding_greedy_layout_register.svg" alt="Register of atoms embedding the QUBO instance"></figure>')  # markdown-exec: hide
```

## Partial embedding and decomposition

QUBO instances with more variables than a device can hold, or whose couplings cannot all be reproduced at once, can be solved by **decomposition**: embed part of the instance, solve the sub-instance on the embedded variables, fix them, and repeat on what is left.

The geometric embedder, `embedding.geometric`, is the partial embedder for this: rather than placing every variable, it grows a cluster of variables, from a random starting one, as long as their couplings can be reproduced within the device constraints. It returns the register together with the placed variables, where qubit $i$ embeds `variables[i]`. At least one variable is always placed.

Fixing a solved cluster is a job for [variable fixing](../classical/transforms.md#reduction-with-variable-fixing): `variable_fixing.from_candidate` turns a candidate of the sub-instance into a fixation rule on the instance. Fixing a variable to 1 folds its couplings into the diagonal of the remaining variables, so each round works on a reduced instance conditioned on all previous rounds, and `variable_fixing.lift` reconstructs the full solution at the end.

- API reference: [`qubosolver.embedding.geometric`](../../api/embedding.md), [`qubosolver.transforms.variable_fixing`](../../api/transforms.md)

!!! warning
    The geometric embedder requires negative diagonal coefficients, so start with `variable_fixing.apply_recursively`, which fixes the other variables to 0. For the same reason, fix in **two separate calls** at each round: the candidate rule alone, then `apply_recursively`, since fixing variables to 1 can turn diagonal coefficients non-negative. The candidate rule's indices refer to the instance it was built for: it raises `ValueError` if another rule fixed variables before it.

### Quantum subproblems

Each subproblem runs on the register found by the embedder, with a drive shaped for the exact sub-instance. The remaining variables are solved classically.

```python exec="on" source="tabbed-left" session="decomposition" result="text"
import qoolqit
import torch
from qubosolver import Instance, LocalEmulator, Solution, drive_shaping, embedding, matrix, solving
from qubosolver.transforms import variable_fixing

# Private utility to set seed.
from qubosolver.utils._random import manual_seed

manual_seed(2)

# A random QUBO: negative diagonal, sparse non-negative off-diagonal coefficients
n = 12
upper = (20 * torch.rand(n, n) * (torch.rand(n, n) < 0.5)).triu(1)
instance = Instance(matrix.as_tensor(upper + upper.T - torch.diag(100 * torch.rand(n))))

device = qoolqit.AnalogDeviceWithDMM()
backend = LocalEmulator(num_shots=100)
stop, min_atoms = 4, 1

reduced = variable_fixing.apply_recursively(instance)  # the embedder needs diag < 0
while reduced.size > stop:
    # register qubit i <-> variables[i]; always at least one atom
    register, variables = embedding.geometric.embed_for_device(reduced, device)
    if len(variables) <= min_atoms:  # not worth a QPU job
        break
    sub = reduced.subinstance(variables)  # exact (conditioned) sub-QUBO
    drive = drive_shaping.proportional_diagonal.build_drive(sub, register, device=device)
    job = backend.run(solving.analog_quantum_sampling.compile(register, drive, device))
    sub_solution = Solution.from_results(job.results(), sub)
    rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)  # best candidate
    variable_fixing.apply(reduced, (rule,), inplace=True)  # candidate rule alone
    variable_fixing.apply_recursively(reduced, inplace=True)  # Hansen to a fixed point
solution = variable_fixing.lift(solving.tabu_search.solve(reduced), reduced)

print(f"Fixed variables, in fixing order: {reduced.fixed_history}")
print(f"Solution: {solution[0].string}, cost {solution[0].cost:.2f}")
```

`reduced.fixed_history` lists the variables fixed by each rule application, in order and in the indices of the original instance: the initial Hansen presolve, then, for each subproblem, the fixed candidate followed by the Hansen passes it triggered. `reduced.fixed_variables` merges them.

The loop leaves every decision to you: rejecting a subproblem is simply not fixing it, stopping early is leaving the loop, and the remainder is `reduced`, which can be solved by any means. To see which QUBO the register and drive actually implement, including what drive shaping did to the detunings, use `extract_qubo(register, drive)`.

### Classical subproblems

The same loop works with classical sub-solvers. The geometric embedder still selects device-sized clusters of variables; its register is simply unused:

```python exec="on" source="tabbed-left" session="decomposition" result="text"
manual_seed(10)

# A larger random QUBO
n = 50
upper = (20 * torch.rand(n, n) * (torch.rand(n, n) < 0.3)).triu(1)
instance = Instance(matrix.as_tensor(upper + upper.T - torch.diag(100 * torch.rand(n))))

stop, min_atoms = 15, 3

reduced = variable_fixing.apply_recursively(instance)
while reduced.size > stop:
    _, variables = embedding.geometric.embed_for_device(reduced, device)
    if len(variables) <= min_atoms:
        break
    sub = reduced.subinstance(variables)
    sub_solution = solving.tabu_search.solve(sub)
    rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
    variable_fixing.apply(reduced, (rule,), inplace=True)
    variable_fixing.apply_recursively(reduced, inplace=True)
solution = variable_fixing.lift(solving.tabu_search.solve(reduced), reduced)

print(f"{len(reduced.fixed_variables)} variables fixed, {reduced.size} solved at the end")
print(f"Cost: {solution[0].cost:.2f}")
```

### Register-derived subproblems

The sub-instance `reduced.subinstance(variables)` is the exact sub-QUBO. The QUBO the register implements instead has the interactions $1/r^6$ between the placed atoms off the diagonal; `extract_qubo(register, exact)` builds it, taking the diagonal from the exact sub-instance. Solving it classically shows what the register alone can achieve:

```python exec="on" source="tabbed-left" session="decomposition" result="text"
from qubosolver import extract_qubo

manual_seed(10)

reduced = variable_fixing.apply_recursively(instance)
while reduced.size > stop:
    register, variables = embedding.geometric.embed_for_device(reduced, device)
    if len(variables) <= min_atoms:
        break
    exact = reduced.subinstance(variables)
    sub = extract_qubo(register, exact)  # off-diagonal from the register, diagonal from `exact`
    sub_solution = solving.tabu_search.solve(sub)
    rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
    variable_fixing.apply(reduced, (rule,), inplace=True)
    variable_fixing.apply_recursively(reduced, inplace=True)
solution = variable_fixing.lift(solving.tabu_search.solve(reduced), reduced)

print(f"Cost: {solution[0].cost:.2f}")
```

Candidates are ranked here by their cost on the register-derived QUBO. To rank them on the exact sub-QUBO instead, pick `min(sub_solution, key=lambda c: exact.cost(c.bitstring))`.

## Custom register

You can skip embedding algorithms altogether and build a [`qoolqit.Register`][] directly from coordinates, as in the example below, or reuse a register computed elsewhere or encode problem-specific knowledge about where atoms should sit:

```python exec="on" source="tabbed-right" session="embedding"
import qoolqit
from pathlib import Path
from matplotlib import pyplot as plt

coords = [[i, 0] for i in range(10)]
register = qoolqit.Register.from_coordinates(coords)

output_dir = Path.cwd()
output_dir = Path("{{ config.site_dir.split('"')[0] }}/user_guide/quantum/embedding")  # markdown-exec: hide
output_dir.mkdir(parents=True, exist_ok=True)

register.draw()
fig = plt.gcf()
fig.savefig(output_dir / "quantum_embedding_custom_register.svg")
plt.close(fig)

print('<figure><figcaption>Register</figcaption><img src="quantum_embedding_custom_register.svg" alt="Register of atoms embedding the QUBO instance"></figure>')  # markdown-exec: hide
```

## The `Solver` shortcut

Rather than calling `embedding.blade.embed` or `embedding.greedy_layout.embed` directly, you can select and configure the embedding algorithm through `EmbeddingConfig`, nested in `QuantumSolvingConfig` and `SolverConfig`; `Solver` then runs it as part of the full quantum pipeline. Leaving `EmbeddingConfig` unset falls back to the BLaDE embedder — see [`SolverConfig`][qubosolver.SolverConfig] for the full set of defaults.

`EmbeddingConfig` only exposes the most commonly tuned parameters of each algorithm (e.g. `blade_steps_per_round`, `greedy_layout_lattice`). Finer-grained parameters — such as BLaDE's `pca` flag or its `compute_*` schedule functions — are not settable this way; call `embedding.blade.embed`/`embedding.greedy_layout.embed` directly with a full `blade.Config`/`greedy_layout.Config` if you need those.

```python exec="on" source="tabbed-left" session="embedding" result="text"
from qubosolver import SolverConfig, QuantumSolvingConfig, EmbeddingConfig
from dataclasses import asdict
import pprint

embedding_config = EmbeddingConfig(
    algorithm="blade",
    # algorithm = "greedy_layout",
    # greedy_layout_lattice = "triangular",
    greedy_layout_lattice="square",
)
quantum_config = QuantumSolvingConfig(
    embedding=embedding_config,
)
solver_config = SolverConfig(
    solving=quantum_config,
)
print(pprint.pformat(asdict(solver_config.solving.embedding)))
```
