from __future__ import annotations

from dwave.samplers import SimulatedAnnealingSampler

from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.classical_solver.classical_solver_conversion_tools import run_sampler


def quboSAdwave(qubo: QUBOInstance) -> QUBOSolution:
    """
    Solves a QUBO optimization problem using D-Wave's Simulated Annealing sampler.

    Parameters:
    qubo (QUBOInstance): A QUBO problem instance encoded in the custom QUBOInstance format.

    Returns:
    dict: The result of the QUBO optimization, as returned by the run_sampler function.
    """
    # Initialize the Simulated Annealing sampler from D-Wave
    sampler = SimulatedAnnealingSampler()

    # Execute the QUBO instance using the sampler and return the results

    solution: QUBOSolution = run_sampler(sampler, qubo)

    return solution
