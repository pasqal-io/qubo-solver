from __future__ import annotations

import warnings
from typing import List

import cplex
import dimod
import torch

from qubosolver import QUBOInstance, QUBOSolution


def qubo_instance_to_dimod_bqm(
    qubo_instance: QUBOInstance,
) -> dimod.BinaryQuadraticModel:
    """
    Converts a QUBO instance into a dimod BinaryQuadraticModel.

    Args:
        qubo_instance (QUBOInstance): An instance containing the QUBO coefficient matrix
            (as a torch.Tensor).

    Returns:
        dimod.BinaryQuadraticModel: The resulting Binary Quadratic Model ready for use.
    """
    if qubo_instance.coefficients is None:
        raise ValueError("The QUBO instance does not have coefficients.")

    # Convert the coefficient tensor to a NumPy array.
    Q = qubo_instance.coefficients.cpu().numpy()

    if qubo_instance.size is None:
        raise ValueError("QUBO instance size is None.")

    size: int = qubo_instance.size

    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    offset: float = 0.0  # Adjust this offset if needed

    # Populate the linear and quadratic terms.
    # We assume that the matrix Q is symmetric.
    for i in range(size):
        linear[i] = Q[i, i]
        for j in range(i + 1, size):
            if Q[i, j] != 0:
                quadratic[(i, j)] = Q[i, j]

    # Create a BinaryQuadraticModel using binary variables (0,1).
    return dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)


def run_sampler(sampler: dimod.Sampler, qubo: QUBOInstance) -> QUBOSolution:
    """
    Runs a given sampler on a QUBO instance and returns the sampled solutions
    along with their costs.

    Parameters:
        sampler (dimod.Sampler): The sampler used to solve the QUBO problem.
        qubo (QUBOInstance): A QUBO problem instance encoded in the custom QUBOInstance format.

    Returns:
        QUBOSolution: An object containing the sampled solutions (binary vectors)
            and their corresponding costs.
    """
    # Convert the QUBO instance to a Binary Quadratic Model (BQM) compatible with dimod.
    bqm: dimod.BinaryQuadraticModel = qubo_instance_to_dimod_bqm(qubo)

    # Use the provided sampler to find solutions to the BQM.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampleset: dimod.SampleSet = sampler.sample(bqm)

    # Convert the sampled solutions into a PyTorch tensor of binary values (0/1).
    bitstrings = torch.tensor((sampleset.record.sample > 0).astype(int), dtype=torch.float32)

    # Compute the cost of each solution using the QUBO's evaluation function.
    costs = torch.tensor(
        [qubo.evaluate_solution(sample.tolist()) for sample in bitstrings],
        dtype=torch.float32,
    )

    # Return the solutions and their costs wrapped in a QUBOSolution object.
    return QUBOSolution(bitstrings=bitstrings, costs=costs)


def qubo_instance_to_sparsepairs(
    instance: QUBOInstance, tol: float = 1e-8
) -> List[cplex.SparsePair]:
    if instance.coefficients is None:
        raise ValueError("The QUBO instance does not have coefficients.")

    matrix = instance.coefficients.cpu().numpy()
    size = matrix.shape[0]
    sparsepairs: List[cplex.SparsePair] = []

    for i in range(size):
        indices: List[int] = []
        values: List[float] = []
        for j in range(size):
            coeff = matrix[i, j] * 2
            if abs(coeff) > tol:
                indices.append(j)
                values.append(float(coeff))  # <<< conversion ici
        sparsepairs.append(cplex.SparsePair(ind=indices, val=values))

    return sparsepairs
