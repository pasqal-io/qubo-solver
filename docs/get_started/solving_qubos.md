# Solving QUBOs

QUBO is a general-purpose formulation: many combinatorial optimization problems — portfolio selection, graph partitioning, scheduling, feature selection — can be cast as a QUBO by choosing an appropriate matrix $Q$, spanning applications from finance and economics to machine learning[^1][^2]. This generality comes at a cost. QUBO is NP-hard: the number of candidate bitstrings grows as $2^n$, and no known algorithm finds the exact optimum for every instance in polynomial time. Beyond a modest number of variables, exhaustive search is out of reach.

In practice, this means solving a QUBO is a search for good bitstrings within a time budget, not a guarantee of the true optimum. Different solving methods trade off solution quality, runtime, and the size of instance they can handle.

## Classical, quantum, and hybrid approaches

**Classical solvers** — heuristics such as tabu search or simulated annealing — explore the space of bitstrings directly on a CPU. They are the natural default: no special hardware is required, and they scale to fairly large instances, though solution quality can degrade as the instance grows or as the coefficients become denser.

**Quantum solvers** take a different route. Rather than exploring bitstrings one by one, they encode the QUBO instance into a physical system — a register of neutral atoms — and let its quantum dynamics sample low-cost bitstrings directly. For instances that map well onto the hardware, this can find good solutions where classical heuristics struggle, though it depends on device size and connectivity constraints.

**Hybrid quantum-classical solvers** combine both: a classical optimization loop steers the parameters of a quantum computation, using the QPU to sample candidate solutions while a classical routine tunes how the search is conducted. This can improve solution quality beyond what either approach achieves alone, at the cost of additional runtime.

There is no universally best choice — which approach to use depends on the instance size, its structure, and the resources available.

## Solving on a neutral-atom processor

Pasqal's quantum processors operate in the Rydberg analog model: a register of neutral atoms is driven by laser pulses, and the resulting quantum evolution can be steered so that the states measured at the end favor low-cost bitstrings for the QUBO instance encoded into the register. Running on real hardware requires embedding the instance onto a physical layout of atoms and shaping a drive compatible with the target device — details covered in the [User guide](../user_guide/intro.md) section — but Qubo Solver applies sensible defaults for both, so a quantum solve can be run out of the box, on an emulator or on Pasqal's QPU, without configuring either step by hand.

## Where to go next

The [tutorials](../tutorials/01-qubosolver-basics.ipynb) walk through solving QUBO instances with both the object-oriented and functional APIs, covering classical, quantum, and hybrid solving in practice.

[^1]: [Glover et al., A Tutorial on Formulating and Using QUBO Models (2018)](https://arxiv.org/abs/1811.11538)
[^2]: [Glover et al., Quantum bridge analytics I: a tutorial on formulating and using QUBO models (2022)](https://link.springer.com/article/10.1007/s10479-022-04634-2)
