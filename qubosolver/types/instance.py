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
class Instance:
    """A single QUBO (Quadratic Unconstrained Binary Optimization) problem instance.

    Wraps a symmetric square coefficient matrix ``Q`` and exposes helpers for
    evaluation, serialisation, and introspection.  The objective to minimise is:

        cost(x) = x^T Q x,   x ∈ {0, 1}^n

    Attributes:
        matrix (Matrix):
            Read-only property returning the ``(size, size)`` coefficient tensor.
            Asserts squareness on every access (see [`matrix`][]).

    Args:
        matrix (Matrix):
            Square coefficient matrix ``Q`` of shape ``(n, n)``.
            Defaults to an empty ``(0, 0)`` zero matrix, which represents
            a trivial problem with no variables.
    """

    def __init__(
        self,
        matrix: Matrix = matrix.zeros(0),
    ):
        self._matrix: Matrix = matrix

    @property
    def size(self) -> int:
        """Number of binary variables in the QUBO problem (side length of ``Q``)."""
        return self.matrix.shape[0]

    @property
    def matrix(self) -> torch.Tensor:
        """The ``(size, size)`` QUBO coefficient matrix.

        Returns:
            Coefficient matrix of shape ``(size, size)``.

        Raises:
            AssertionError: If the internal tensor is not 2-D or not square.
        """
        assert (
            self._matrix.ndim == 2 and self._matrix.shape[0] == self._matrix.shape[1]
        )  # nosec B101
        return self._matrix

    @property
    def _max_off_diag(self) -> float:
        """Maximum absolute value among all off-diagonal entries of [`matrix`][].

        Used internally to normalise the coefficient matrix before embedding or
        solving.  Off-diagonal entries are identified via a boolean mask that
        excludes the main diagonal.
        """
        mask = ~torch.eye(self.size, dtype=torch.bool, device=self.matrix.device)
        return self.matrix[mask].max().item()

    @property
    def _normalized_matrix(self) -> torch.Tensor:
        """Coefficient matrix scaled so that the largest off-diagonal entry equals 1.

        Divides every element of [`matrix`][] by [`_max_off_diag`][].
        Used to bring coefficients into a hardware-friendly range before
        embedding.
        """
        return self.matrix / self._max_off_diag

    def evaluate_solution(self, solution: Bitstring) -> float:
        """Compute the QUBO objective ``x^T Q x`` for a candidate solution *x*.

        Args:
            solution (Bitstring):
                Binary vector ``x`` of shape ``(size,)`` with values in ``{0, 1}``.

        Returns:
            float: Scalar cost value.  Lower is better for minimisation problems.

        Note:
            The result type is asserted to be a plain ``float`` (not a tensor
            or numpy scalar) before returning.
        """
        cost = costs.quadratic_cost(solution, self.matrix)
        assert type(cost) is float  # nosec B101
        return cost

    def __repr__(self) -> str:
        """Human-readable summary of the instance.

        Returns:
            str: A quoted string of the form ``"Instance of size = <n>,density = <d>,"`` where *d* is rounded to two decimal places.

        Note:
            The outer `repr` call intentionally wraps the f-string in
            quotes, so the return value includes surrounding single quotes.
        """
        density = _calculate_density(self.matrix)
        return repr(f"Instance of size = {self.size}," f"density = {round(density, 2)},")

    @staticmethod
    def save(file_like: io_utils.FileLike[bytes], instance: Instance) -> None:
        """Serialise *instance* to *file_like* using `torch.save`.

        The coefficient matrix is written into an internal
        `io.BytesIO` buffer and then flushed to *file_like* using
        `io_utils.save_sized_buffer`, which prefixes the payload with its
        byte length.  This framing allows multiple objects to be stored
        contiguously in the same stream.

        Args:
            file_like (io_utils.FileLike[bytes]):
                Destination — a file path (`str` or `os.PathLike`),
                or a binary-writable `typing.IO` stream.
            instance (Instance):
                The instance to serialise.  Only [`matrix`][] is persisted;
                any derived state is recomputed on load.
        """
        with io_utils.open(file_like, "wb") as f:
            buffer = io.BytesIO()
            torch.save(instance.matrix, buffer)
            io_utils.save_sized_buffer(f, buffer.getbuffer())

    @staticmethod
    def load(file_like: io_utils.FileLike[bytes]) -> Instance:
        """Deserialise a `Instance` previously saved with `save`.

        Reads a length-prefixed byte block from *file_like* into a dedicated
        `io.BytesIO` buffer before calling `torch.load`.  The
        isolated buffer prevents `torch.load` from over-consuming the source
        stream when multiple objects are packed together.

        Args:
            file_like (io_utils.FileLike[bytes]):
                Source — a file path (`str` or `os.PathLike`),
                or a binary-readable `typing.IO` stream.  Must contain
                data written by `save`.

        Returns:
            A new instance whose `matrix` is the deserialised coefficient tensor.

        Note:
            `torch.load` is called with `weights_only=True` to prevent
            arbitrary code execution from untrusted checkpoint files.
        """
        with io_utils.open(file_like, "rb") as f:
            # torch.load might consume too much of the src buffer.
            #  Use a dedicated limited buffer
            buffer = io.BytesIO(io_utils.load_sized_buffer(f))
            Q = torch.load(buffer, weights_only=True)

        return Instance(Q)


# Density classification thresholds — half-open intervals [lo, hi).
# The HIGH bracket is closed on the right to include a fully dense matrix (1.0).
_SPARSE_THRESHOLD: tuple[float, float] = (0.0, 0.3)  # [0.0, 0.3)
_MEDIUM_THRESHOLD: tuple[float, float] = (0.3, 0.7)  # [0.3, 0.7)
_HIGH_THRESHOLD: tuple[float, float] = (0.7, 1.0)  # [0.7, 1.0]


def _classify_density(density: float) -> DensityType:
    """Map a density value to a :class:`~.DensityType` category.

    The boundaries follow half-open intervals so that every value in
    ``[0.0, 1.0]`` maps to exactly one category:

    +-----------+------------------+
    | Category  | Range            |
    +===========+==================+
    | SPARSE    | ``[0.0, 0.3)``   |
    +-----------+------------------+
    | MEDIUM    | ``[0.3, 0.7)``   |
    +-----------+------------------+
    | HIGH      | ``[0.7, 1.0]``   |
    +-----------+------------------+

    Args:
        density (float): Non-zero ratio in ``[0.0, 1.0]`` as returned by
            `_calculate_density`.

    Returns:
        DensityType: Corresponding :class:`~.DensityType` member.

    Raises:
        ValueError: If *density* falls outside ``[0.0, 1.0]`` (e.g. negative
            values or values greater than 1).
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
    """Compute the fraction of non-zero entries in a coefficient matrix.

    Args:
        m (Matrix): QUBO coefficient matrix of any shape.

    Returns:
        float: Value in ``[0.0, 1.0]`` where ``0.0`` means all entries are zero
        and ``1.0`` means no entry is zero.  Returns ``0.0`` for empty matrices
        (``m.numel() == 0``) to avoid division by zero.
    """
    if m.numel() == 0:
        return 0.0
    return torch.count_nonzero(m).item() / m.numel()
