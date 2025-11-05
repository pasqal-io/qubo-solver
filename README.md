# Qubo Solver

Solving combinatorial optimization (CO) problems using quantum computing is one of those promising applications for the near term. The Quadratic Unconstrained Binary Optimization (QUBO) (also known as unconstrained binary quadratic programming) model enables to formulate many CO problems that can be tackled using quantum hardware. QUBO offers a wide range of applications from finance and economics to machine learning.
The Qubo Solver is a Python library designed for solving Quadratic Unconstracined Binary Optimization (QUBO) problems on a neutral atom quantum processor.

The core of the library is focused on the development of several algorithms for solving QUBOs: classical (tabu-search, simulated annealing, ...), quantum (Variational Quantum Algorithms, Quantum Adiabatic Algorithm, ...) or hybrid quantum-classical.

Users setting their first steps into quantum computing will learn how to implement the core algorithm in a few simple steps and run it using the Pasqal Neutral Atom QPU. More experienced users will find this library to provide the right environment to explore new ideas - both in terms of methodologies and data domain - while always interacting with a simple and intuitive QPU interface.

!!! warning "Usage restrictions"
    At the moment, only QUBO matrices in symmetric form with non-negative off diagonal terms are supported.
    We plan to handle negative off diagonal terms in a future release.

## Development tools

## Installation

### Install as a dependency

Using `hatch`, `uv` or any pyproject-compatible Python manager

Edit file `pyproject.toml` to add the line

```
  "qubo-solver"
```

### Using `pip` or `pipx`

To install the `pipy` package using `pip` or `pipx`

1. Create a `venv` if that's not done yet

```sh
python -m venv .venv
```

2. Enter the venv

```sh
source .venv/bin/activate
```
or on Windows:
```sh
.venv\Scripts\activate
```

3. Install the package

```sh
pip install qubo-solver
```
or
```sh
pipx install qubo-solver
```

Alternatively, you can also:

* install with `pip` in development mode by simply running `pip install -e .`. Notice that in this way
  you will install all the dependencies, including extras.
* install it with `conda` by simply using `pip` inside the Conda environment.

### Windows Note

This package require features available on Unix systems. Under Windows, these features can be installed as
part of the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/).

### Cplex Installation

The `cplex` package is only available under some combinations of platforms and versions of Python. We
recommend using python `3.11` or `3.12`, which we have tested to work with cplex.

If you wish to use the licensed version of cplex, you will need to set the environment
variable `ILOG_LICENSE_FILE` to the location of the license file -- for more details, see the documentation
of cplex.


## QuickStart

### With a quantum solver

```python
import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.solver import QuboSolver


# define QUBO
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
instance = QUBOInstance(coefficients=Q)

# Create a SolverConfig object to use a quantum backend
config = SolverConfig(use_quantum=True)

# Instantiate the quantum solver.
solver = QuboSolver(instance, config)

# Solve the QUBO problem.
solution = solver.solve()
print(solution)

# Returns the following
# QUBOSolution(bitstrings=tensor([[0, 0]]), costs=tensor([0.]), counts=None, probabilities=None, solution_status=<SolutionStatusType.TRIVIALZERO: 'trivial-zero'>)
```

The solver returns a `QUBOSolution` instance containing candidates or bitstrings solutions found by the solver,
with their respective QUBO costs. If sampling was performed, we would also obtain respective counts (frequencies a solution has been sampled), and the respective probabilities (counts divided by the number of samples). Finally, the `solution_status` determines if preprocessing (technique to reduce the instance to another smaller instance) or postprocessing were applied (modification of the solution after solving), or if the solution found is trivial (obtaining the solution from the QUBO instance is straighforward as the case above where we have only positive coefficients, hence all variables must be set to 0).

### With a classical solver

```python
import torch
from qubosolver import QUBOInstance
from qubosolver.config import ClassicalConfig, SolverConfig
from qubosolver.solver import QuboSolverClassical, QuboSolverQuantum

# define QUBO
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
instance = QUBOInstance(coefficients=Q)

# Create a SolverConfig object with classical solver options.
classical_config = ClassicalConfig(
    classical_solver_type="cplex",
    cplex_maxtime=10.0,
    cplex_log_path="test_solver.log",
)
config = SolverConfig(use_quantum=False, classical=classical_config)

# Instantiate the classical solver via the pipeline's classical solver dispatcher.
classical_solver = QuboSolver(instance, config)

# Solve the QUBO problem.
solution = classical_solver.solve()
print(solution)
```


## Documentation

- [Documentation](https://pasqal-io.github.io/qubo-solver/latest/)
- [Notebooks Tutorials](https://pasqal-io.github.io/qubo-solver/latest/tutorial/01-dataset-generation-and-loading/).
- [Full API documentation](https://pasqal-io.github.io/qubo-solver/latest/api/qubo_instance/).


## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [Github repository](https://github.com/pasqal-io/qubo-solver) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)
