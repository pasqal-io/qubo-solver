We're grateful for your interest in contributing to Qubo Solver! Please follow our guidelines to ensure a smooth contribution process.

## Reporting an Issue or Proposing a Feature

Your course of action will depend on your objective, but generally, you should start by creating an issue. If you've discovered a bug or have a feature you'd like to see added, feel free to create an issue on [the issue tracker](https://github.com/pasqal-io/qubo-solver/issues). Here are some steps to take:

1. Quickly search the existing issues using relevant keywords to ensure your issue hasn't been addressed already.
2. If your issue is not listed, create a new one. Try to be as detailed and clear as possible in your description.

- If you're merely suggesting an improvement or reporting a bug, that's already excellent! We thank you for it. Your issue will be listed and, hopefully, addressed at some point.
- However, if you're willing to be the one solving the issue, that would be even better! In such instances, you would proceed by preparing a [Pull Request](#submitting-a-pull-request).

## Submitting a Pull Request

We're excited that you're eager to contribute to Qubo Solver! We use the standard GitHub fork workflow: fork the repository, make your changes on a branch in your fork, and once you are satisfied with your feature and all the tests pass, open a [Pull Request](https://github.com/pasqal-io/qubo-solver/pulls) from your fork's branch to `main`.

Here's the process for making a contribution:

1. [Fork the repository](https://github.com/pasqal-io/qubo-solver/fork) to your own GitHub account.

2. Clone your fork locally:

  ```shell
  git clone https://github.com/<your username>/qubo-solver.git
  cd qubo-solver
  git remote add upstream https://github.com/pasqal-io/qubo-solver.git
  ```

3. Create a new branch for your change, using a `topic/<short-description>` naming convention, where `<short-description>` is a few hyphenated words describing the change (e.g. `topic/solver-timeout`):

  ```shell
  git checkout -b topic/<short-description>
  ```

4. Push your branch to your fork:

  ```shell
  git push --set-upstream origin topic/<short-description>
  ```

5. Once your feature is ready and all the tests pass, open a Pull Request from your fork's branch against the `main` branch of `pasqal-io/qubo-solver`.

If your fork falls behind, sync it by fetching and merging (or rebasing on) `upstream/main`:

```shell
git fetch upstream
git merge upstream/main
```

## Setting up your development environment

1) Clone your fork of the [Qubo Solver GitHub repository](https://github.com/pasqal-io/qubo-solver)

  ```sh
  git clone https://github.com/<your username>/qubo-solver.git
  ```

2) Setup an environment for developing. From your `qubo-solver` folder run:

  ```sh
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .[dev]
  ```

### Useful things for your workflow: Linting and Testing

Use `pre-commit` hooks to make sure that the code is properly linted before pushing a new commit. Make sure that the unit tests and type checks are passing since the merge request will not be accepted if the automatic CI/CD pipeline do not pass.

```shell
pip install pre-commit
pre-commit install
pre-commit run --all-files
pytest
```

## A Word on Patents

Quantum computing is a heavily patented field. If your contribution introduces a new quantum algorithm, rather than an improvement to the existing library, please check beforehand that it is not covered by an existing patent.

## Contributing Terms

Finally, before contributing, please take a moment to review the following:

- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Contributor Agreement](./CONTRIBUTOR_AGREEMENT.md)
- [License](https://github.com/pasqal-io/qubo-solver/blob/main/LICENSE.md)

By contributing, you agree to follow and accept these documents.
