from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from qubosolver.data import QUBODataset, QUBOSolution

# Modules to be automatically added to the qubosolver.utils namespace
__all__ = [
    "save_qubo_dataset",
    "load_qubo_dataset",
]


def save_qubo_dataset(dataset: QUBODataset, filepath: str | Path) -> None:
    """
    Saves a QUBODataset to a file.

    Args:
        dataset (QUBODataset):
            The QUBODataset object to save.
        filepath (str | Path):
            Path to the file where the QUBODataset will be saved.

    Notes:
        The saved data includes:
            - Coefficients (size x size x num_instances tensor)
            - Solutions (optional, includes bitstrings, counts, probabilities, and costs)
    """
    data: dict[str, Any] = {"coefficients": dataset.coefficients, "solutions": None}
    if dataset.solutions is not None:
        data["solutions"] = [
            {
                "bitstrings": solution.bitstrings,
                "counts": solution.counts,
                "probabilities": solution.probabilities,
                "costs": solution.costs,
            }
            for solution in dataset.solutions
        ]
    torch.save(data, filepath)


def load_qubo_dataset(filepath: str | Path) -> QUBODataset:
    """
    Loads a QUBODataset from a file.
    Notes:
        The file should contain data saved in the format used by `save_qubo_dataset`.

    Args:
        filepath (str | Path):
            Path to the file from which the QUBODataset will be loaded.

    Returns:
        QUBODataset:
            The loaded QUBODataset object.


    """
    data = torch.load(filepath)
    solutions = None
    if data["solutions"] is not None:
        solutions = [
            QUBOSolution(
                bitstrings=solution["bitstrings"],
                counts=solution["counts"],
                probabilities=solution["probabilities"],
                costs=solution["costs"],
            )
            for solution in data["solutions"]
        ]
    return QUBODataset(coefficients=data["coefficients"], solutions=solutions)
