::: qubosolver.solvers
    options:
      filters:
        - "!^Solver$"
        - "!^QuboSolver$"
        - "!^[a-z0-9_]+$"
        - "^quantum$"
        - "^hybrid$"
        - "^classical$"
        - "!^_[^_]"

::: qubosolver.solvers.quantum
    options:
      show_submodules: true

::: qubosolver.solvers.hybrid
    options:
      show_submodules: true

::: qubosolver.solvers.classical
    options:
      show_submodules: true
