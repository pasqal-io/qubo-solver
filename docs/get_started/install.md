## Install from PyPI

Qubo Solver can be installed from [PyPI](https://pypi.org/project/qubo-solver/) with your favorite pyproject-compatible Python manager.
Using `pip`, for example:

```sh
pip install qubo-solver
```

!!! tip "Don't forget to create a virtual environment first!"

## Add Qubo Solver as a dependency
For usage within a project with a corresponding `pyproject.toml` file, you can add
`qubo-solver` to the list of dependencies as follows:

```toml
[project]
dependencies = [
  "qubo-solver"
]
```

## Install from source
If you are developing code for Qubo Solver, you can install it directly from source:

1) Clone the [Qubo Solver GitHub repository](https://github.com/pasqal-io/qubo-solver)

  ```sh
  git clone https://github.com/pasqal-io/qubo-solver.git
  ```

2) From your `qubo-solver` folder, create a virtual environment and install the project in editable mode.
  Using `venv` and `pip`, for example:

  ```sh
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .[dev]
  ```
