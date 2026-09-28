from __future__ import annotations

import torch

from qubosolver.data import QUBODataset, QUBOSolution
from qubosolver.qubo_instance import QUBOInstance

# Modules to be automatically added to the qubosolver.utils namespace
__all__ = [
    "load_qubo_instance",
    "save_qubo_instance",
    "save_qubo_dataset",
    "load_qubo_dataset",
]


def save_qubo_instance(instance: QUBOInstance, filepath: str) -> None:
    """
    Saves a QUBOInstance to a file.

    Args:
        instance (QUBOInstance):
            The QUBOInstance object to save.
        filepath (str):
            Path to the file where the QUBOInstance will be saved.

    Notes:
        The saved data includes:
            - Coefficients (N x N matrix)
            - Device (e.g., 'cpu' or 'cuda')
            - Data type (e.g., torch.float32)
            - Solution (optional)
    """
    data = {
        "coefficients": instance.coefficients,  # N x N
        "device": instance.device,
        "dtype": instance.dtype,
        "solution": instance.solution,
    }
    torch.save(data, filepath)


def load_qubo_instance(filepath: str) -> QUBOInstance:
    """
    Loads a QUBOInstance from a file.

    Args:
        filepath (str):
            Path to the file from which the QUBOInstance will be loaded.

    Returns:
        QUBOInstance:
            The loaded QUBOInstance object.

    Notes:
        The file should contain data saved in the format used by `save_qubo_instance`.
    """
    data = torch.load(filepath, weights_only=False)
    instance = QUBOInstance(
        coefficients=data["coefficients"],
        device=data["device"],
        dtype=data["dtype"],
    )
    instance.solution = data["solution"]
    return instance


def save_qubo_dataset(dataset: QUBODataset, filepath: str) -> None:
    """
    Saves a QUBODataset to a file.

    Args:
        dataset (QUBODataset):
            The QUBODataset object to save.
        filepath (str):
            Path to the file where the QUBODataset will be saved.

    Notes:
        The saved data includes:
            - Coefficients (size x size x num_instances tensor)
            - Solutions (optional, includes bitstrings, counts, probabilities, and costs)
    """
    data = {"coefficients": dataset.coefficients, "solutions": None}
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


def load_qubo_dataset(filepath: str) -> QUBODataset:
    """
    Loads a QUBODataset from a file.
    Notes:
        The file should contain data saved in the format used by `save_qubo_dataset`.

    Args:
        filepath (str):
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
