from __future__ import annotations
from typing import List

import cplex

from qubosolver import QUBOInstance


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
