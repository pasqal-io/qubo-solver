## Install from PyPI

Qubo Solver can be installed from [PyPI](https://pypi.org/project/qubo-solver/) with your favorite pyproject-compatible Python manager.
Using `pip`, for example:

```sh
pip install qubo-solver
```

!!! tip "Don't forget to create a virtual environment first!"

##  Your first ⚛️**Quantum**⚛️ solve

!!! info "Runs on an emulator, no QPU required"
    By default, the solver runs locally using Pasqal's [emulators](https://docs.pasqal.com/qpu-emulators/emulators/), so you can try it out without access to real quantum hardware.

=== "Code"

    ```python
    from qubosolver import Instance, Solver, matrix, analysis

    # Define an Instance from a symmetric matrix
    Q = matrix.tensor([
        [-0.2, 0.0, 1.0],
        [ 0.0, 0.0, 1.5],
        [ 1.0, 1.5, 0.0],
    ])
    instance = Instance(Q)

    # Instantiate the quantum solver (default)
    solver = Solver(instance)
    solution = solver.solve()

    print(analysis.to_dataframe([solution]))
    ```

=== "Result"

    ```
      labels bitstrings  costs  counts  probs
    0      0        110   -0.2      43  0.043
    1      0        100   -0.2     180  0.180
    2      0        010    0.0     325  0.325
    3      0        001    0.0     320  0.320
    4      0        000    0.0     132  0.132
    ```

The solver returns a [`Solution`][qubosolver.Solution] instance holding the candidate bitstrings and their QUBO costs. As the quantum solver samples the solution space, it also reports each bitstring's count (how many times it was measured) and probability.
