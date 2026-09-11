<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/extras/assets/logo/qubosolver_logo_white.svg" width="60%">
    <source media="(prefers-color-scheme: light)" srcset="./docs/extras/assets/logo/qubosolver_logo_darkgreen.svg" width="60%">
    <img alt="Qubo Solver logo" src="./docs/extras/assets/logo/qubosolver_logo_darkgreen.svg" width="60%">
  </picture>
</p>

<p align="center">
  <strong>Qubo Solver</strong> is a Python library for solving Quadratic Unconstrained Binary Optimization (QUBO) problems on Pasqal's neutral-atom quantum processors.
</p>

> 🔥 Can't wait to try it? Jump straight to the [30-Second Quickstart](https://pasqal-io.github.io/qubo-solver/latest/get_started/quickstart/).
>
> 🆘 Need help? Check out the [Help](https://pasqal-io.github.io/qubo-solver/latest/help/faq/) section for FAQ, troubleshooting, and contact info.

**For more detailed information, [check out the documentation](https://pasqal-io.github.io/qubo-solver/latest/)**.

## Install from PyPI
Qubo Solver can be installed from PyPI with your favorite pyproject-compatible Python manager.
On `pip`, for example:

```sh
pip install qubo-solver
```

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
If you wish to install directly from the source, for example, if you are developing code for Qubo Solver, you can:

1) Clone the [Qubo Solver GitHub repository](https://github.com/pasqal-io/qubo-solver)

  ```sh
  git clone https://github.com/pasqal-io/qubo-solver.git
  ```

2) Setup an environment for developing. From your `qubo-solver` folder, again install with your favorite environment/package managers.
  On `venv`/`pip`, for example:

  ```sh
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .[dev]
  ```


# Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [GitHub Repository](https://github.com/pasqal-io/qubo-solver/) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)
