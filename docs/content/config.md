# SolverConfig – Solver Configuration Reference

The `SolverConfig` class defines how a QUBO problem should be solved — specifying whether to use a quantum or classical approach, which backend to run on, and additional execution parameters.

This configuration is passed into any solver (e.g., `QuboSolver`) and guides its behavior.
Note that `SolverConfig` uses three other configuration objects: `EmbeddingConfig`, `ClassicalConfig` and `DriveShapingConfig`.
Besides `ClassicalConfig`, the other configurations represents different parts of the solver when using a quantum approach:

---

## Fields for SolverConfig

::: qubosolver.config.SolverConfig

### Embedding configuration

When solving with a quantum approach, we need to define an embedding method, that is how we define the geometry (register) of atoms based on the QUBO instance and compatibility with a device.
The embedding configuration part (the `embedding` field of `SolverConfig`) is divided into several attributes that concerns the `embedding_method` chosen (`BLaDE` or `Greedy`, for which a prefix enables defning to which method they belong):

::: qubosolver.config.EmbeddingConfig


### Drive Shaping configuration

Quantum devices can be programmed by specifying a Drive. A program in the Rydberg analog model is defined as a time-dependent drive Hamiltonian that is imposed on the qubits.
The drive shaping configuration part (the `drive_shaping` field of `SolverConfig`) is set via the `DriveShapingConfig` class, and defines how the drive parameters are constructed (heuristically from the QUBO diagonal, via bayesian optimization, ...).
Note, for parameters concerning exclusively the heuristic drive shaping method, a `heuristic_` prefix is present.
Similarly, for parameters concerning exclusively the optimized drive shaping method (bayesian optimization), an `optimized_` prefix is present.

::: qubosolver.config.DriveShapingConfig

### Classical solver configuration

For the classical solver, its configuration can be set via the `ClassicalConfig` class:

::: qubosolver.config.ClassicalConfig

Note, for parameters concerning exclusively simulated annealing, an `sa_` prefix is present.
Similarly for tabu search, the prefix is `tabu_`.


### Pre-Post processing parameters

We can also apply preprocessing of the QUBO instance (to reduce it to another smaller instance) or postprocessing the solution after solving.

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `do_postprocessing`    | `bool` | Whether we apply post-processing (`True`) or not (`False`). |
| `do_preprocessing`    | `bool` | Whether we apply pre-processing (`True`) or not (`False`). |

---

## Example
The `SolverConfig` is designed in such way that all parameters have a default value which fulfilled the minimum required configuration to execute the necessary steps to solve a QUBO.

All the parameters are optional which allows for running `SolverConfig` without specifying any parameter:
```python exec="on" source="material-block"
from qubosolver.config import SolverConfig
from qubosolver.qubo_types import EmbedderType

config = SolverConfig()
print(config.specs())
```
Although the default configuration is straightforward, all parameters can be modified by the user to better suit the specific QUBO instance. Below is an example of a configuration that uses a different embedder with customized parameters on a specific device:
```python exec="on" source="material-block"
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, EmbeddingConfig

coefficients = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
instance = QUBOInstance(coefficients=coefficients)

embedding_config = EmbeddingConfig(embedding_method="greedy", greedy_traps=-1)

config = SolverConfig(
    config_name="my_config",
    use_quantum=True,
    embedding = embedding_config,
)
```

Equivalently, one can instantiate a `SolverConfig` simply using the keyword arguments of the other configs via the `SolverConfig.from_kwargs` method:

```python exec="on" source="material-block"
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig

coefficients = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
instance = QUBOInstance(coefficients=coefficients)

config = SolverConfig.from_kwargs(
    config_name="my_config",
    use_quantum=True,
    embedding_method="greedy",
    greedy_traps=-1
)
```
