# FAQ

## How do I report a bug?

??? quote "Open an issue on the [GitHub issue tracker](https://github.com/pasqal-io/qubo-solver/issues)"
    To help us investigate quickly, please include:

    - if possible, a **minimal reproducible example** — the smallest snippet that triggers the issue,
    - the full **traceback**, if an exception was raised,
    - the full [`DEBUG`-level log output](troubleshooting.md#enabling-logging).

## I have a Maximum Independent Set (MIS) formulation, not a QUBO. Can I still use the Pasqal stack to solve it?

??? quote "Yes, first convert your MIS instance into a QUBO formulation"
    The QUBO Solver expects a QUBO instance, so you'll need to map your MIS problem to one before solving it. The [MIS-to-QUBO tutorial](../tutorials/08-mis-to-qubo.ipynb) walks through this conversion with a worked example.

    If you have any questions, please [reach out to us](contact.md).
