# The classical pipeline

A QUBO problem on $N$ variables consists in a symmetric matrix $Q$ of size $N\times N$.

Solving a QUBO problem means to find the bitstring $z=(z_1,...,z_N)\in \{0, 1\}^N$ that minimizes the quantity

$$
f(z) = z^TQz= \sum_i Q_{ii} z_i + \sum_{i \neq j} Q_{ij} z_i z_j, \quad z_i \in \{0,1\}.
$$

Qubo Solver's classical machinery runs entirely on CPU, without going through the quantum pipeline, and serves three purposes:

1. **[Solvers](solvers.md)** as a baseline — exact or heuristic algorithms to solve a QUBO instance from scratch, to validate or compare against the quantum pipeline.
2. **[Solvers](solvers.md)** as a refinement step — some solvers take an initial solution and improve it, which is useful to post-process the output of a quantum solver.
3. **[Transforms](transforms.md)** to reduce or adapt an instance before solving — fixing variables to shrink the problem, or removing negative off-diagonal coefficients so it becomes embeddable on a quantum device.

## Baseline: solving from scratch

Solvers that don't need a starting point are the simplest way to get a reference solution: an
exact solve, or a fast heuristic to compare quality and runtime against the quantum pipeline. See
[Solvers](solvers.md) for the full list.

## Refinement: improving an existing solution

Solvers that take an initial solution can post-process the result of any other solver, including
a quantum one — for example, running local bitflips on a quantum sampler's output to squeeze out
a better bitstring. See [Solvers](solvers.md) for the refinement algorithms and an example.

## Adapting the instance: transforms

Transforms change the `Instance` itself before it reaches a solver, and record enough history to
map a `Solution` of the transformed problem back to the original one. Some transforms are exact —
variable fixing only fixes a variable when it is guaranteed to be optimal, so the transformed
problem is equivalent to the original — while others are approximations: zeroing negative
off-diagonal coefficients (needed to make an instance embeddable on a quantum device) changes the
objective, and should be used with that trade-off in mind. See [Transforms](transforms.md) for
details on each.

## Where to go next

- Choose a baseline or refinement algorithm in [Solvers](solvers.md).
- Reduce or adapt an instance before solving in [Transforms](transforms.md).
