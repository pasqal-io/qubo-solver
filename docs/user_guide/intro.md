# Solving a QUBO instance

Solving a QUBO instance means feeding an [`Instance`](instances_and_solutions.md) to a solver. `qubosolver` gives you two ways to do that:

- Call the [classical](classical/intro.md) or [quantum](quantum/intro.md) pipeline functions directly, composing [transforms](classical/transforms.md) and solvers as needed.
- Or use [`Solver`][qubosolver.Solver] with a [`SolverConfig`][qubosolver.SolverConfig], which wraps a common pipeline (transform → solve → lift) behind a single call, and lets you switch between a classical and a quantum approach by configuration alone.

## Composition of solvers and transforms

A solve is a pipeline: an optional chain of transforms reduces or adapts the instance, a solver produces a solution for the transformed instance, and each transform is "lifted" in reverse to map that solution back to the original instance.

<figure markdown>
<img src="../../extras/assets/pipeline_composition.svg" alt="Composition of solvers and transforms" style="max-width: 700px; width: 100%;">
<figcaption>An optional chain of transforms reduces the instance, a solver produces a solution, then each transform is lifted in reverse to map that solution back to the original instance.</figcaption>
</figure>

[`Solver`][qubosolver.Solver]/[`SolverConfig`][qubosolver.SolverConfig] builds this pipeline for you: `preprocessing=True` chains the configured transforms before solving, `postprocessing=True` chains refinement solvers after, and the solver itself is either the [classical](classical/intro.md) or [quantum](quantum/intro.md) pipeline depending on `solving`. See [Transforms](classical/transforms.md) and [Solvers](classical/solvers.md) for the building blocks.

## The quantum pipeline

Solving with a quantum approach turns the instance into a program that runs on a neutral-atom device — an [embedding](quantum/embedding.md) step maps variables onto atoms, [drive shaping](quantum/drive_shaping.md) builds the laser pulse, and the result is compiled and run on a [backend](quantum/backend.md) (a local emulator, a remote emulator, or a real QPU).

<figure markdown>
<img src="../../extras/assets/quantum_pipeline_vertical.svg" alt="The quantum pipeline" style="max-width: 400px; width: 100%;">
<figcaption>Embedding maps the instance onto a register of atoms; drive shaping builds the drive from that register; both feed into a compiled <code>QuantumProgram</code>, which a backend (a QPU or an emulator) runs. <code>run</code> returns a <code>Job</code> as soon as it's queued — the <code>Solution</code> is fetched later through that job, via an async fetch, once it's ready.</figcaption>
</figure>

See [The quantum pipeline](quantum/intro.md) for the full walkthrough, including a runnable example and how to plot the register and drive.

## Where to go next

- Understand the core [`Instance`](instances_and_solutions.md) and [`Solution`](instances_and_solutions.md) types.
- Learn about the building blocks of the [classical pipeline](classical/intro.md): solvers and transforms.
- Learn about the steps of the [quantum pipeline](quantum/intro.md): embedding, drive shaping, compiling, and backends.
- Configure either approach through [`qubosolver.SolverConfig`][].
