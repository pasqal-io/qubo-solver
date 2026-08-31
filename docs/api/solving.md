::: qubosolver.solving
    options:
      filters:
        - "!^Solver$"
        - "!^QuboSolver$"
        - "!^[a-z0-9_]+$"
        - "^quantum$"
        - "^hybrid$"
        - "^classical$"
        - "!^_[^_]"

::: qubosolver.solving.quantum
    options:
      show_submodules: true
      filters:
        - "!^config$"
        - "!^_[^_]"

::: qubosolver.solving.hybrid
    options:
      show_submodules: true

::: qubosolver.solving.classical
    options:
      show_submodules: true
