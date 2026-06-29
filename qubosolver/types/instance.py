from __future__ import annotations

import torch
import io

from ._checks import debug_runtime_typecheck
from . import matrix
from .linalg import Matrix, Bitstring
from .enums import DensityType
from qubosolver._io import utils as io_utils
from qubosolver._utils import costs


@debug_runtime_typecheck
class QUBOInstance:
    """
    Represents a single instance of a Quadratic Unconstrained Binary Optimization (QUBO) problem.

    Attributes:
        coefficients (Matrix):
            Tensor of shape ``(size, size)``, representing the QUBO coefficients.
        device (torch.device):
            Device where tensors are allocated (e.g., ``cpu`` or ``cuda``).
        dtype (torch.dtype):
            Data type of the tensors (e.g., ``torch.float32``).
        solution (QUBOSolution):
            Solution to the QUBO problem. Defaults to an empty :class:`QUBOSolution`.
        density (float):
            Fraction of non-zero entries in the coefficient matrix.
        density_type (DensityType):
            Classification of the density (SPARSE, MEDIUM, or HIGH).
    """

    def __init__(
        self,
        matrix: Matrix = matrix.zeros(0),
    ):
        """
        Initializes a QUBOInstance.

        Args:
            coefficients (Matrix):
                Square coefficient matrix of the QUBO problem.
                Defaults to an empty ``(0, 0)`` matrix.
        """
        self._matrix: Matrix = matrix

    @property
    def size(self) -> int:
        """
        Get the size of the QUBO matrix (number of variables).

        Returns:
            int:
                Size of the QUBO matrix.
        """
        return self.matrix.shape[0]

    @property
    def matrix(self) -> torch.Tensor:
        """
        Getter for the QUBO coefficient matrix.

        Returns:
            torch.Tensor:
                Tensor of shape (size, size) representing the QUBO coefficients.
        """
        assert self._matrix.ndim == 2 and self._matrix.shape[0] == self._matrix.shape[1]  # nosec B101
        return self._matrix

    @property
    def _max_off_diag(self) -> float:
        mask = ~torch.eye(self.size, dtype=torch.bool, device=self.matrix.device)
        return self.matrix[mask].max().item()

    @property
    def _normalized_matrix(self) -> torch.Tensor:
        """Returns the coefficient matrix normalized by the maximum off-diagonal value."""
        return self.matrix / self._max_off_diag

    def evaluate_solution(self, solution: Bitstring) -> float:
        """
        Evaluates a solution for the QUBO problem as ``solution^T Q solution``.

        Args:
            solution (Bitstring):
                Binary solution tensor of shape ``(size,)``.

        Returns:
            float:
                The cost of the given solution.
        """
        cost = costs.quadratic_cost(solution, self.matrix)
        assert type(cost) is float  # nosec B101
        return cost

    def __repr__(self) -> str:
        """
        Returns a string representation of the QUBOInstance.

        Returns:
            str: A dictionary-like string summarizing the instance.
        """
        density = _calculate_density(self.matrix)
        return repr(f"QUBOInstance of size = {self.size}," f"density = {round(density, 2)},")

    @staticmethod
    def save(file_like: io_utils.FileLike[bytes], instance: QUBOInstance) -> None:
        """
        Saves a QUBOInstance to a file-like object.

        Args:
            file_like (io_utils.FileLike[bytes]):
                File-like object opened in binary write mode where the instance will be saved.
            instance (QUBOInstance):
                The QUBOInstance object to be saved.

        Returns:
            None
        """
        with io_utils.open(file_like, "wb") as f:
            buffer = io.BytesIO()
            torch.save(instance.matrix, buffer)
            io_utils.save_sized_buffer(f, buffer.getbuffer())

    @staticmethod
    def load(file_like: io_utils.FileLike[bytes]) -> QUBOInstance:
        """
        Loads a QUBOInstance from a file-like object.

        Args:
            file_like (io_utils.FileLike[bytes]):
                File-like object opened in binary read mode containing the saved QUBOInstance data.

        Returns:
            QUBOInstance:
                A new QUBOInstance object reconstructed from the saved data.
        """
        with io_utils.open(file_like, "rb") as f:
            # torch.load might consume too much of the src buffer.
            #  Use a dedicated limited buffer
            buffer = io.BytesIO(io_utils.load_sized_buffer(f))
            Q = torch.load(buffer, weights_only=True)

        return QUBOInstance(Q)


# Density thresholds
_SPARSE_THRESHOLD: tuple[float, float] = (0.0, 0.3)
_MEDIUM_THRESHOLD: tuple[float, float] = (0.3, 0.7)
_HIGH_THRESHOLD: tuple[float, float] = (0.7, 1.0)


def _classify_density(density: float) -> DensityType:
    """
    Classifies the density of a QUBO problem based on predefined thresholds.

    Args:
        density (float):
            The density value to classify. Should be in the range [0.0, 1.0].

    Returns:
        DensityType:
            The classification of the density (SPARSE, MEDIUM, or HIGH).
    """
    if _SPARSE_THRESHOLD[0] <= density < _SPARSE_THRESHOLD[1]:
        return DensityType.SPARSE
    elif _MEDIUM_THRESHOLD[0] <= density < _MEDIUM_THRESHOLD[1]:
        return DensityType.MEDIUM
    elif _HIGH_THRESHOLD[0] <= density <= _HIGH_THRESHOLD[1]:
        return DensityType.HIGH
    else:
        raise ValueError(f"Density {density} is outside the defined thresholds.")


def _calculate_density(m: Matrix) -> float:
    """
    Calculates the density of a QUBO coefficient matrix.

    Density is defined as the fraction of non-zero elements in the matrix.

    Args:
        m (Matrix):
            The QUBO coefficient matrix.

    Returns:
        float:
            The density value, ranging from 0.0 (completely sparse) to 1.0 (completely dense).
            Returns 0.0 for empty matrices.
    """
    if m.numel() == 0:
        return 0.0
    return torch.count_nonzero(m).item() / m.numel()
