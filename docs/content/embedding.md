# Embedding workflow

When using quantum approaches, we have to define a map from the instance variables to atoms on a quantum device. We call this step an embedding. Here is a breakdown of the current workflow implementation for the embedding of a QUBO matrix, using several embedding methods and parameters.

## Using default configuration

The `SolverConfig()` without argument has a default behavior (e.g. Solver Config page) that allows for embedding using the minimum configuration according to the device.

### Code example
```python exec="on" source="material-block" html="1" session="embedding"
import torch

from qubosolver.config import SolverConfig, EmbeddingConfig
from qubosolver.solver import QUBOInstance, QuboSolver

# define qubo matrix
coefficients = torch.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

# Instantiate a QUBOInstance with coefficients
instance = QUBOInstance(coefficients)

# define the solver with default configuration
default_config = SolverConfig()
solver = QuboSolver(instance, default_config)
register = solver.embedding()

# draw the register
# register.draw()
```

## BLaDE config
The following configuration uses the BLaDE method with specific dimension layers and a number of steps per round (i.e. the number of iterations per layer). Starting positions are implicitely defined but it can be set here, as long as it matches the first dimension layer.
TODO: refer to qoolqit documentation

```python exec="on" source="material-block" html="1" session="embedding"
embedconfig = EmbeddingConfig(embedding_method="blade", blade_dimensions=[5, 4, 3, 2], blade_steps_per_round=300)
blade_config = SolverConfig(
    use_quantum=True,
    embedding=embedconfig,
)

solver = QuboSolver(instance, blade_config)
register = solver.embedding()

# register.draw()
```

## Greedy embedder config
The following uses the greedy embedding method on a triangular lattice layout with 200 traps (`greedy_traps`). It also allows working on a square lattice layout, as well as increasing the number of traps according to the device specifications.

```python exec="on" source="material-block" html="1" session="embedding"
from qoolqit import AnalogDevice

embedconfig = EmbeddingConfig(embedding_method="greedy", greedy_traps=200, greedy_layout="triangular")
greedy_config = SolverConfig(
    use_quantum=True,
    embedding=embedconfig,
    device = AnalogDevice(),
)

solver = QuboSolver(instance, greedy_config)
register = solver.embedding()

# register.draw()
```

## Custom embedder config

If one desires to develop their own embedding method, a subclass of `qubosolver.pipeline.embedder.BaseEmbedder` should be implemented with a mandatory `embed` method.

The `embed` method `def embed(self) -> qoolqit.Register` specifies how the problem is mapped into a register of qubits when running using a quantum device. Let us show a simple example where each variable $i$ is mapped into a qubit lying on a horizontal line (with coordinates $[i, 0]$, $i$ normalized by the device specs).

```python exec="on" source="material-block" html="1" session="embedding"
import typing
from qoolqit import Register
from qubosolver.pipeline.embedder import BaseEmbedder
from qubosolver.config import (
    EmbeddingConfig,
    SolverConfig,
)

class FixedEmbedder(BaseEmbedder):

    @typing.no_type_check
    def embed(self) -> Register:
        # The register is scaled so that the minimal distance between atoms is 1 (plus a margin). This is the recommended minimal distance.
        scale = 1.001
        qubits = {f"q{i}": (i * scale ,0) for i in range(self.instance.coefficients.shape[0])}
        register = Register(qubits)
        return register


config = SolverConfig(
    use_quantum=True,
    embedding=EmbeddingConfig(embedding_method=FixedEmbedder),
)

solver = QuboSolver(instance, config)
register = solver.embedding()

# register.draw()
```
