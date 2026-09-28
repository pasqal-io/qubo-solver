# Analyzing qubo solutions

## QUBOAnalyzer example

To analyze the solutions from one or many QUBO solvers, we can instantiate a `QUBOAnalyzer` with solutions and labels as follows:

```python exec="on" source="material-block" html="1" session="analyzer"
import torch
from qubosolver.data import QUBOSolution
from qubosolver.qubo_analyzer import QUBOAnalyzer

num_bitstrings=100
bit_length=3

costs = torch.randint(1, 20, (2**bit_length,), dtype=torch.float)

bitstrings = torch.randint(0, 2, (num_bitstrings, bit_length))
bitstrings,counts=bitstrings.unique(dim=0,return_counts=True)
solution1 = QUBOSolution(bitstrings, costs, counts)

bitstrings = torch.randint(0, 2, (num_bitstrings, bit_length))
bitstrings,counts=bitstrings.unique(dim=0,return_counts=True)
solution2 = QUBOSolution(bitstrings, costs, counts)

# Create the analyzer with our two solutions
analyzer = QUBOAnalyzer([solution1, solution2], labels=["sol1", "sol2"])
df = analyzer.df
```

This will generate a pandas dataframe internally (accessible via the `df` attribute) for several `QUBOAnalyzer` methods for plotting or comparing solutions,
described below. More examples are demonstrated in the [`QUBOAnalyzer` tutorial](/tutorial/08-qubo_analyzer/).

## QUBOAnalyzer API description

::: qubosolver.qubo_analyzer
