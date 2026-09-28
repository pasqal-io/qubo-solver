# Embedding workflow

Here is a breakdown of the current workflow implementation for the embedding of a QUBO matrix, using several embedding methods and parameters.

## Using default configuration

The `SolverConfig()` without argument has a default behavior (e.g. Solver Config page) that allows for embedding using the minimum configuration according to the device.

### Code example
```python exec="on" source="material-block" html="1" session="embedding"
import torch

from qubosolver.config import SolverConfig, EmbeddingConfig, BackendConfig
from qubosolver.solver import QUBOInstance, QuboSolver

# define qubo matrix
coefficients = torch.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

# Instantiate a QUBOInstance with coefficients
instance = QUBOInstance(coefficients)

# define the solver with default configuration
default_config = SolverConfig()
solver = QuboSolver(instance, default_config)
geometry = solver.embedding()

# draw the register
# geometry.register.draw()
```

## Greedy embedder config
The following uses the greedy embedding method on a triangular lattice layout with a number of traps equals to the size of the QUBO. It also allows working on a square lattice layout, as well as increasing the number of traps according to the device specifications.

```python exec="on" source="material-block" html="1" session="embedding"
from qoolqit._solvers.types import DeviceType

embedconfig = EmbeddingConfig(embedding_method="greedy", traps=instance.size, layout_greedy_embedder="triangular",)
backend = BackendConfig(device=DeviceType.ANALOG_DEVICE)
greedy_config = SolverConfig(
    use_quantum=True,
    embedding=embedconfig,
    backend_config = backend,
)

solver = QuboSolver(instance, greedy_config)
geometry = solver.embedding()

# geometry.register.draw()
```

## Custom embedder config
If one desires to develop his own embedding method, a subclass of `qubosolver.pipeline.embedder.BaseEmbedder` should be implemented with a mandatory `embed` method.

The `embed` method `def embed(self) -> qubosolver.pipeline.targets.Register` specify how the problem is mapped into a register of qubits when running using a quantum device. Let us show a simple example where each variable $i$ is mapped into a qubit lying on a horizontal line (with coordinates $[i, 0]$).

```python exec="on" source="material-block" html="1" session="embedding"
import typing
from qubosolver.pipeline.embedder import BaseEmbedder
from qubosolver.pipeline.targets import Register as TargetRegister
from qubosolver.config import (
    EmbeddingConfig,
    SolverConfig,
)
from pulser.register import Register as PulserRegister

class FixedEmbedder(BaseEmbedder):

    @typing.no_type_check
    def embed(self) -> TargetRegister:
        qubits = {f"q{i}": (i,0) for i in range(self.instance.coefficients.shape[0])}
        register = PulserRegister(qubits)
        return TargetRegister(self.config.backend_config.device, register)


config = SolverConfig(
    use_quantum=True,
    embedding=EmbeddingConfig(embedding_method=FixedEmbedder),
)
```
