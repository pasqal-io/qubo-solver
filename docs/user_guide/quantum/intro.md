# The quantum pipeline

A QUBO problem on $N$ variables consists in a symmetric matrix $Q$ of size $N\times N$.

Solving a QUBO problem means to find the bitstring $z=(z_1,...,z_N)\in \{0, 1\}^N$ that minimizes the quantity

$$
f(z) = z^TQz= \sum_i Q_{ii} z_i + \sum_{i \neq j} Q_{ij} z_i z_j, \quad z_i \in \{0,1\}.
$$

## Problem formulation in Rydberg Hamiltonian

To use a Rydberg Analog model, we need to map the QUBO problem onto the Rydberg Hamiltonian.

This is achieved by identifying the binary variables with atomic occupations
$$
z_i = n_i
$$

where $ n_i \in \{0,1\} $ denotes whether atom $i$ is in the Rydberg state.

The effective Hamiltonian in the classical (diagonal $\tilde{\Omega}=0$) limit of the driven Rydberg system can be written as

$$
H = - \sum_i \tilde{\delta}_i n_i + \sum_{i<j} \tilde{J}_{ij} n_i n_j,
$$

where $ \tilde{\delta} $ is the local detuning and $ J_{ij}=1/r_{ij}^6 $ is the interaction energy between atoms $i$ and $j$. By direct comparison, we obtain the mapping:

$$
Q_{ii} \;\longleftrightarrow\; - 2\tilde{\delta}_i \tag{1}
$$

$$
Q_{ij} \;\longleftrightarrow\; \tilde{J}_{ij} \tag{2}
$$

Mapping $(1)$ is the final detuning part of the [drive shaping](drive_shaping.md). Mapping $(2)$ is called the [embedding](embedding.md).

In this way, tuning the interaction strengths between atoms (e.g., via their spatial separation) we can match the QUBO coefficients, while adjusting $Q_{ii}$ maps to change the local detunings. Under this correspondence, the ground state of the Rydberg Hamiltonian minimizes $H$ and therefore encodes the optimal solution of the original QUBO problem.

!!! note "Factor 2"
    The interaction between two atoms is physically counted only once, but our symmetric matrix representation of $Q$ counts each off-diagonal pair twice ($Q_{ij}$ and $Q_{ji}$), which is what introduces the factor $2$ on the diagonal. Equivalently, the Hamiltonian corresponds to a triangular matrix representation.

See [QoolQit](https://docs.pasqal.com/qoolqit/get_started/qoolqit_model/)'s documentation for more details on the Rydberg Hamiltonian.

Both [`drive`][qoolqit.Drive] and [`register`][qoolqit.Register] are then compiled into a [`QuantumProgram`][qoolqit.QuantumProgram], to run on a quantum backend (emulator or QPU).

Qubo Solver can solve a QUBO instance using Pasqal's Rydberg analog devices, either on local/remote emulators or on a real QPU. Solving with a quantum approach is a pipeline of three steps, each exposed as a standalone function you can call directly:

1. **[Embedding](embedding.md)** — map the QUBO instance's variables to atoms on a device, producing a `Register`.
2. **[Drive shaping](drive_shaping.md)** — build the time-dependent drive Hamiltonian applied to the register.
3. **Compiling and running** — compile the register and drive into a program, and run it on a [backend](backend.md): a local emulator, a remote emulator, or a real QPU.

## Example

```python exec="on" session="negative" source="tabbed-left" result="text"
from qubosolver import (
    Instance, Solution, LocalEmulator,
    embedding, drive_shaping, solving,
    matrix, analysis,
)
import qoolqit

# Private utility to set seed.
from qubosolver.utils._random import manual_seed
manual_seed(147)

instance = Instance(matrix.tensor([
    [-2.0, 1.0, 1.0],
    [ 1.0,-4.0, 1.0],
    [ 1.0, 1.0,-1.0],
]))

device = qoolqit.AnalogDeviceWithDMM()
backend = LocalEmulator()

# 1. Embedding: map the instance onto a register of atoms.
register = embedding.blade.embed(instance)

# 2. Drive shaping: build the drive Hamiltonian for that register.
drive = drive_shaping.proportional_diagonal.build_drive(
    instance, register, device=device, dmm=True)

# 3. Compile and run on the chosen backend.
program = solving.analog_quantum_sampling.compile(register, drive, device)
job = backend.run(program)

# 4. Turn the raw results into a Solution and inspect it.
solution = Solution.from_results(job.results(), instance)
print(analysis.to_dataframe([solution]))
```

You can plot the register and the drive.

```python exec="on" session="negative" source="tabbed-right" tabs="Source|Plots"
from pathlib import Path
from matplotlib import pyplot as plt

output_dir = Path.cwd()
output_dir = Path("{{ config.site_dir.split('"')[0] }}/user_guide/quantum/intro")  # markdown-exec: hide
output_dir.mkdir(parents=True, exist_ok=True)

register.draw()
fig = plt.gcf()
fig.savefig(output_dir / "quantum_intro_register.svg")
plt.close(fig)

drive.draw()
fig = plt.gcf()
fig.savefig(output_dir / "quantum_intro_drive.svg")
plt.close(fig)

print('<figure><figcaption>Register of atoms embedding the QUBO instance</figcaption><img src="quantum_intro_register.svg" alt="Register of atoms embedding the QUBO instance"></figure>')  # markdown-exec: hide
print('<figure><figcaption>Drive Hamiltonian applied to the register</figcaption><img src="quantum_intro_drive.svg" alt="Drive Hamiltonian applied to the register"></figure>')  # markdown-exec: hide
```

Each call returns a plain object you can inspect or pass to a different step: swap `embedding.blade.embed` for [`embedding.greedy.embed`](embedding.md), try another [drive shaping method](drive_shaping.md), or run on a different [backend](backend.md) — the rest of the pipeline is unaffected.

To run remotely (on a remote emulator or a real QPU), pass a `RemoteEmulator` or `QPU` backend instead of `LocalEmulator`. Remote runs are asynchronous: `backend.run(program)` returns as soon as the job is queued, so you can save its identifiers and retrieve the results later. See the [qubosolver-in-full tutorial](../../tutorials/02-qubosolver-in-full.ipynb) for the full remote and save/retrieve example, as well as the equivalent classical functional call.

## The `Solver` shortcut

For the common case, [`SolverConfig`][qubosolver.SolverConfig] and [`Solver`][qubosolver.Solver] wrap the four steps above into a single call, using sensible defaults for anything you don't specify:

```python exec="on" source="tabbed-left" session="negative" result="text"
from qubosolver import (
    Instance, Solver, SolverConfig,
    matrix, analysis,
)
# Private utility to set seed.
from qubosolver.utils._random import manual_seed
manual_seed(147)

instance = Instance(matrix.tensor([
    [-2.0, 1.0, 1.0],
    [ 1.0,-4.0, 1.0],
    [ 1.0, 1.0,-1.0],
]))

config = SolverConfig()
solver = Solver(instance, config)
solution = solver.solve()

print(analysis.to_dataframe([solution]))
```

## Where to go next

- Learn how variables are mapped onto atoms in [Embedding](embedding.md).
- Learn how the drive Hamiltonian is built in [Drive shaping](drive_shaping.md).
- Choose between local emulators, remote emulators and a QPU in [Backends](backend.md).
