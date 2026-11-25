# Embedding workflow

Embedding maps a QUBO instance's variables onto atom positions on a quantum device, so that the physical interaction between atoms approximates the QUBO's coefficients. Each embedding algorithm is a function you call directly.

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
from qubosolver import Instance, embedding
import numpy as np
import qoolqit

# Private utility to set seed.
from qubosolver.utils._random import manual_seed

manual_seed(958)

instance = Instance(
    np.asarray(
        [
            [0, 1, 2, 1],
            [1, 0, 3, 0],
            [2, 3, 0, 5],
            [1, 0, 5, 0],
        ],
        dtype=np.float64,
    )
)
register = embedding.blade.embed_for_device(instance, qoolqit.AnalogDevice())
interaction_matrix = np.asarray(register.interaction_matrix(), dtype=np.float64)

np.set_printoptions(precision=2)
print(f"QUBO matrix:\n{instance.matrix}\n")
print(f"Interaction matrix:\n{interaction_matrix}\n")

relative_error = (
    100 * np.linalg.norm(instance.matrix - interaction_matrix) / np.linalg.norm(instance.matrix)
)
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
from qubosolver import Instance, embedding
import numpy as np
import qoolqit

# Private utility to set seed.
from qubosolver.utils._random import manual_seed

manual_seed(851)

instance = Instance(
    np.asarray(
        [
            [0, 1, 2, 1],
            [1, 0, 3, 0],
            [2, 3, 0, 5],
            [1, 0, 5, 0],
        ],
        dtype=np.float64,
    )
)

register = embedding.greedy_layout.embed_for_device(
    instance,
    qoolqit.AnalogDevice(),
)
interaction_matrix = np.asarray(register.interaction_matrix(), dtype=np.float64)

np.set_printoptions(precision=2)
print(f"QUBO matrix:\n{instance.matrix}\n")
print(f"Interaction matrix:\n{interaction_matrix}\n")

relative_error = (
    100 * np.linalg.norm(instance.matrix - interaction_matrix) / np.linalg.norm(instance.matrix)
)
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

The partial greedy embedder, `embedding.partial_greedy`, is the partial embedder for this: rather than placing every variable, it grows a cluster of variables, from a random starting one, as long as their couplings can be reproduced within the device constraints. It returns the register together with the placed variables, where qubit $i$ embeds `variables[i]`. At least one variable is always placed.

Fixing a solved cluster is a job for [variable fixing](../classical/transforms.md#reduction-with-variable-fixing): `variable_fixing.apply_candidate` fixes the placed variables to the bits of a candidate of the sub-instance. Fixing a variable to 1 folds its couplings into the diagonal of the remaining variables, so each round works on a reduced instance conditioned on all previous rounds, and `variable_fixing.lift` reconstructs the full solution at the end.

- API reference: [`qubosolver.embedding.partial_greedy`](../../api/embedding.md), [`qubosolver.transforms.variable_fixing`](../../api/transforms.md)

!!! warning
    The partial greedy embedder requires non-positive diagonal coefficients, so start with `variable_fixing.apply_recursively`, which fixes variables with a positive diagonal coefficient to 0. At each round, call `apply_recursively` again after `apply_candidate`, since fixing variables to 1 can turn diagonal coefficients positive.

### Quantum subproblems

Each subproblem runs on the register found by the embedder, with a drive shaped for the exact sub-instance. The remaining variables are solved classically.

```python exec="on" source="tabbed-left" session="decomposition" result="text"
import qoolqit
import numpy as np
from qubosolver import (
    Instance,
    LocalEmulator,
    Solution,
    analysis,
    drive_shaping,
    embedding,
    solving,
)
from qubosolver.transforms import variable_fixing

# Private utility to set seed.
from qubosolver.utils._random import manual_seed

manual_seed(10)

# A random QUBO: negative diagonal, sparse non-negative off-diagonal coefficients
n = 50
upper = np.triu(20 * np.random.rand(n, n) * (np.random.rand(n, n) < 0.5), k=1)
Q = upper + upper.T - np.diag(100 * np.random.rand(n))
instance = Instance(Q)

device = qoolqit.AnalogDeviceWithDMM()
backend = LocalEmulator(num_shots=1000)

# Pre-process to fix variables with positive linear weight
reduced_instance = variable_fixing.apply_recursively(instance)
reduced_solution = solving.trivial_solution_search.solve(reduced_instance)

while not reduced_solution:
    # register qubit i <-> variables[i]; always at least one atom
    register, variables = embedding.partial_greedy.embed_for_device(reduced_instance, device)
    # Exact (conditioned) sub-QUBO
    sub_instance = reduced_instance.subinstance(variables)

    # Solve sub-qubo with the quantum pipeline
    drive = drive_shaping.local_energy_scale.build_drive(sub_instance, register, device=device)
    program = solving.analog_quantum_sampling.compile(register, drive, device)
    job = backend.run(program)
    sub_solution = Solution.from_results(job.results(), sub_instance)

    # Fix variables from the best candidate
    variable_fixing.apply_candidate(reduced_instance, sub_solution[0], variables, inplace=True)
    # Fix variables that have been balanced to a positive linear weight
    variable_fixing.apply_recursively(reduced_instance, inplace=True)

    reduced_solution = solving.trivial_solution_search.solve(reduced_instance)

solution = variable_fixing.lift(reduced_solution, reduced_instance)

print(f"Fixed variables, in fixing order:")
for f in reduced_instance.fixed_history:
    print(f"\t{f}")
print(f"Solution:\n{analysis.to_dataframe([solution])}")
```

`reduced_instance.fixed_history` lists the variables fixed by each rule application, in order and in the indices of the original instance: the initial Hansen presolve, then, for each subproblem, the fixed candidate followed by the Hansen passes it triggered. `reduced_instance.fixed_variables` merges them.

The loop leaves every decision to you: rejecting a subproblem is simply not fixing it, stopping early is leaving the loop, and the remainder is `reduced_instance`, which can be solved by any means. To see which QUBO the register and drive actually implement, including what drive shaping did to the detunings, use [`extract_qubo(register, drive)`][qubosolver.extract_qubo].

### Classical subproblems

The same loop works with classical sub-solvers. The partial greedy embedder still selects device-sized clusters of variables; its register is simply unused:

```python exec="on" source="tabbed-left" session="decomposition" result="text"
manual_seed(10)

# A sparser random QUBO
n = 50
upper = np.triu(20 * np.random.rand(n, n) * (np.random.rand(n, n) < 0.3), k=1)
Q = upper + upper.T - np.diag(100 * np.random.rand(n))
instance = Instance(Q)

# Pre-process to fix variables with positive linear weight
reduced_instance = variable_fixing.apply_recursively(instance)
reduced_solution = solving.trivial_solution_search.solve(reduced_instance)

while not reduced_solution:
    # The register is unused: only the placed variables matter
    _, variables = embedding.partial_greedy.embed_for_device(reduced_instance, device)
    # Exact (conditioned) sub-QUBO
    sub_instance = reduced_instance.subinstance(variables)

    # Solve sub-qubo with a classical solver
    sub_solution = solving.tabu_search.solve(sub_instance)

    # Fix variables from the best candidate
    variable_fixing.apply_candidate(reduced_instance, sub_solution[0], variables, inplace=True)
    # Fix variables that have been balanced to a positive linear weight
    variable_fixing.apply_recursively(reduced_instance, inplace=True)

    reduced_solution = solving.trivial_solution_search.solve(reduced_instance)

solution = variable_fixing.lift(reduced_solution, reduced_instance)

print(f"Fixed variables, in fixing order:")
for f in reduced_instance.fixed_history:
    print(f"\t{f}")
print(f"Solution:\n{analysis.to_dataframe([solution])}")
```


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
