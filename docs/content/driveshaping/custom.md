## Custom Drive Shaper config
If one desires to develop his own drive shaping method, a subclass of `qubosolver.pipeline.drive.BaseDriveShaper` should be implemented with a mandatory `generate` method.

The `generate` method syntax is `generate(register: Register, instance: Instance) -> tuple[Drive, Solution]`
 with arguments:

- a `Register` instance specifying the qubits we work with.
- a `Instance` specifying the qubo problem we target.

It returns:

- an instance of `qoolqit.Drive`
- a `Solution` specyfing the solution that may be used by a solver.

For concrete examples, we have the [`ProportionalDiagonalDriveShaper`](./proportional_diagonal.md) and the [`BayesianSearchDriveShaper`](./bayesian_search.md) and their current implementations lie in `qubosolver.pipeline.drive.py`.

Let us show an example of a custom simple hard-coded drive shaper.

```python exec="on" source="material-block" html="1"
# TODO: revamp with functional
# from qoolqit import Drive, Constant, Ramp, Register
# from qubosolver.pipeline.drive import BaseDriveShaper
# from qubosolver.data import Solution
# from qubosolver.config import (
#     DriveShapingConfig,
#     SolverConfig,
# )
#
# class SimpleShaper(BaseDriveShaper):
#     def generate(
#         self,
#         register: Register,
#     ) -> tuple[Drive, Solution]:
#
#         # Defining the drive parameters
#         omega = 2.0
#         delta_i = -4.0 * omega
#         delta_f = -delta_i
#         T = 200.0
#
#         # Defining the drive
#         wf_amp = Constant(T, omega)
#         wf_det = Ramp(T, delta_i, delta_f)
#         drive = Drive(amplitude=wf_amp, detuning=wf_det)
#
#         return drive, Solution()
#
# config = SolverConfig(
#     use_quantum=True,
#     drive_shaping=DriveShapingConfig(drive_shaping_method=SimpleShaper),
# )
```
