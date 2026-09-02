"""QUBO problem instances.

An [`Instance`][qubosolver.types.instance.Instance] wraps a symmetric square coefficient matrix
``Q`` and defines the QUBO objective to minimize:

$$\\text{cost}(x) = x^T Q x, \\quad x \\in \\{0, 1\\}^n$$
"""

from __future__ import annotations

import torch
import io
from typing import TYPE_CHECKING, TypeVar

from ._checks import debug_runtime_typecheck
from . import matrix
from .linalg import Matrix, Bitstring
from ._enums import _DensityType
from qubosolver._io import utils as io_utils

if TYPE_CHECKING:
    from qubosolver.transforms import negative_bitflip, variable_fixing, zeroing

_InstanceT = TypeVar("_InstanceT", bound="Instance")


@debug_runtime_typecheck
class Instance:
    """A single QUBO problem instance.

    Wraps a symmetric square matrix $Q$ and exposes helpers for
    evaluation, serialization, and introspection.  The objective to minimize is:

    $$\\text{cost}(x) = x^T Q x, \\quad x \\in \\{0, 1\\}^n$$

    Args:
        matrix: Symmetric matrix $Q$ of shape ``(n, n)``.
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
        """Number of binary variables in the QUBO problem."""
        return self.matrix.shape[0]

    def __len__(self) -> int:
        """Number of binary variables in the QUBO problem (same as [`size`][])."""
        return self.size

    @property
    def matrix(self) -> Matrix:
        """The QUBO symmetric matrix.

        Returns:
             QUBO symmetric matrix of shape ``(size, size)``.

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

        Raises:
            RuntimeError: If [`size`][] is less than 2, since there are then no
                off-diagonal entries and the maximum is undefined.
        """
        mask = ~torch.eye(self.size, dtype=torch.bool, device=self.matrix.device)
        off_diag = self.matrix[mask]
        if off_diag.numel() == 0:
            raise RuntimeError(
                "_max_off_diag is undefined for an instance with no off-diagonal entries "
                f"(size={self.size})."
            )
        return off_diag.max().item()

    def cost(self, solution: Bitstring) -> float:
        """Compute the QUBO objective $x^T Q x$ for a candidate solution $x$.

        Args:
            solution: Binary vector $x$ of shape ``(size,)``.

        Returns:
            Scalar cost value.
        """
        # Import here to avoid circular imports
        from qubosolver.utils import _costs

        cost = _costs.quadratic_cost(solution, self.matrix)
        assert type(cost) is float  # nosec B101
        return cost

    @classmethod
    def _tag(cls) -> str:
        """Type tag written first by [`save`][] and checked first by [`load`][].

        Derived from the class's fully-qualified name, so subclasses never
        need to hand-write a duplicate string. Lets a stream be
        self-describing: [`load`][] fails fast with a clear error when handed
        a stream produced by a different `Instance` subclass, instead of
        misinterpreting its bytes as its own state.
        """
        return f"{cls.__module__}.{cls.__qualname__}"

    _registry: dict[str, type[Instance]] = {}
    """Tag -> `Instance` subclass, populated by [`__init_subclass__`][] for [`load`][] to dispatch on."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register `cls` under its [`_tag`][] so [`load`][] can dispatch to it.

        Automatic, so a new `Instance` subclass never needs to be added to a
        separate registry by hand.
        """
        super().__init_subclass__(**kwargs)  # type: ignore[arg-type]
        Instance._registry[cls._tag()] = cls

    def _write_body(self, f: io_utils.FileLike[bytes]) -> None:
        """Write this instance's state to `f`, without any type tag.

        The base implementation writes only [`matrix`][]. Subclasses that
        carry extra state override this to write that state too (typically
        after calling `super()._write_body(f)`).
        """
        # The coefficient matrix is written into an internal `io.BytesIO`
        # buffer and then flushed to *f* using `io_utils.save_sized_buffer`,
        # which prefixes the payload with its byte length. This framing
        # allows multiple objects to be stored contiguously in the same
        # stream.
        buffer = io.BytesIO()
        torch.save(self.matrix, buffer)
        io_utils.save_sized_buffer(f, buffer.getbuffer())  # type: ignore[arg-type]

    @classmethod
    def _read_body(cls, f: io_utils.FileLike[bytes]) -> Instance:
        """Read an instance of `cls` back from `f`, without any type tag.

        The base implementation reads only [`matrix`][]. Subclasses that
        override [`_write_body`][] to write extra state override this to
        read it back in the same order.

        Note:
            [`torch.load`][] is called with `weights_only=True` to prevent
            arbitrary code execution from untrusted checkpoint files.
        """
        # Reads a length-prefixed byte block from *f* into a dedicated
        # `io.BytesIO` buffer before calling `torch.load`. The isolated buffer
        # prevents `torch.load` from over-consuming the source stream when
        # multiple objects are packed together.
        buffer = io.BytesIO(io_utils.load_sized_buffer(f))  # type: ignore[arg-type]
        Q = torch.load(buffer, weights_only=True)
        return Instance(Q)

    def save(self, file_like: io_utils.FileLike[bytes]) -> None:
        """Serialize this instance to ``file_like``, tagged with its type.

        Args:
            file_like: Destination — a file path ([`str`][] or [`os.PathLike`][]),
                or a binary-writable [`typing.IO`][] stream.

        Example:
            ```python
            from pathlib import Path

            with Path("instance.bin").open("wb") as f:
                instance.save(f)
            ```
        """
        with io_utils.open(file_like, "wb") as f:
            io_utils.save_header(f)
            io_utils.save_string(f, self._tag())
            self._write_body(f)

    @classmethod
    def load(cls, file_like: io_utils.FileLike[bytes]) -> Instance:
        """Deserialize an [`Instance`][] previously saved with [`save`][].

        Called on the base class [`qubosolver.Instance`][], it loads any instance with automatic dispatch.

        Called on a subclass (e.g. [`variable_fixing.Instance.load(f)`][qubosolver.transforms.variable_fixing.Instance.load]),
        it additionally requires the loaded `Instance` to be an instance of that subclass (raising [`TypeError`][]
        otherwise).

        Args:
            file_like: Source file path or readable binary file object,
                as produced by [`save`][].

        Returns:
            A new instance of whichever concrete type wrote the tag.

        Raises:
            ValueError: If the stream is not a qubosolver file, or if its type
                tag is missing or unrecognized.
            TypeError: If the loaded instance's type is not `cls` or a subclass thereof.

        Example:
            ```python
            from pathlib import Path
            from qubosolver import Instance
            from qubosolver.transforms import variable_fixing

            file = Path("instance.bin")
            instance = variable_fixing.Instance(Instance())

            with file.open("wb") as f:
                instance.save(f)

            # Three ways to load it back, from least to most strict:
            with file.open("rb") as f:
                loaded = Instance.load(f)                  # accepts any Instance subtype
                loaded = Instance.load(f).variable_fixing  # loads, then narrows (fails after loading)
                loaded = variable_fixing.Instance.load(f)  # narrows first (fails before loading)
            ```
        """
        with io_utils.open(file_like, "rb") as f:
            io_utils.load_header(f)
            tag = io_utils.load_string(f)
            target_cls = Instance._registry.get(tag)
            if target_cls is None:
                raise ValueError(f"Cannot load Instance: unrecognized type tag {tag!r}.")
            instance = target_cls._read_body(f)
        if not isinstance(instance, cls):
            raise TypeError(
                f"Cannot load {cls.__module__}.{cls.__qualname__}: "
                f"stream contains a {type(instance).__module__}.{type(instance).__qualname__}."
            )
        return instance

    def _narrow(self, cls: type[_InstanceT]) -> _InstanceT:
        """Narrow ``self`` to a transform-specific `Instance` subclass.

        Shared implementation for convenience properties (`variable_fixing`,
        `zeroing`, `negative_bitflip`) that let call sites avoid
        ``assert isinstance(instance, <transform>.Instance)`` boilerplate before
        using a method specific to that subclass. These properties exist purely
        to satisfy static type checkers (mypy) and enable IDE code completion —
        the runtime `isinstance` check and the `TypeError` here just mirror the
        guarantee that the `assert` would otherwise provide.

        Raises:
            TypeError: If ``self`` is not an instance of *cls*.
        """
        if not isinstance(self, cls):
            raise TypeError(
                f"Expected a {cls.__module__}.{cls.__qualname__}, got {type(self).__name__}."
            )
        return self

    @property
    def variable_fixing(self) -> variable_fixing.Instance:
        """View of this instance as a variable-fixing [`Instance`][qubosolver.transforms.variable_fixing.Instance].

        Convenience property to avoid the boilerplate of
        ``assert isinstance(instance, variable_fixing.Instance)`` before calling
        a method specific to that subclass. It exists purely to satisfy static
        type checkers (mypy) and enable IDE code completion — the runtime
        [`isinstance`][] check and the [`TypeError`][] below just mirror the guarantee
        that the [`assert`](https://docs.python.org/3/reference/simple_stmts.html#index-18) would otherwise provide.

        Returns:
            This instance, narrowed to the variable-fixing subclass.

        Raises:
            TypeError: If this instance is not a
                [`variable_fixing.Instance`][qubosolver.transforms.variable_fixing.Instance].
        """
        from qubosolver.transforms import variable_fixing

        return self._narrow(variable_fixing.Instance)

    @property
    def zeroing(self) -> zeroing.Instance:
        """View of this instance as a zeroing [`Instance`][qubosolver.transforms.zeroing.Instance].

        Convenience property to avoid the boilerplate of
        ``assert isinstance(instance, zeroing.Instance)`` before calling
        a method specific to that subclass. It exists purely to satisfy static
        type checkers (mypy) and enable IDE code completion — the runtime
        [`isinstance`][] check and the [`TypeError`][] below just mirror the guarantee
        that the [`assert`](https://docs.python.org/3/reference/simple_stmts.html#index-18) would otherwise provide.

        Returns:
            This instance, narrowed to the zeroing subclass.

        Raises:
            TypeError: If this instance is not a
                [`zeroing.Instance`][qubosolver.transforms.zeroing.Instance].
        """
        from qubosolver.transforms import zeroing

        return self._narrow(zeroing.Instance)

    @property
    def negative_bitflip(self) -> negative_bitflip.Instance:
        """View of this instance as a negative-bitflip [`Instance`][qubosolver.transforms.negative_bitflip.Instance].

        Convenience property to avoid the boilerplate of
        ``assert isinstance(instance, negative_bitflip.Instance)`` before calling
        a method specific to that subclass. It exists purely to satisfy static
        type checkers (mypy) and enable IDE code completion — the runtime
        [`isinstance`][] check and the [`TypeError`][] below just mirror the guarantee
        that the [`assert`](https://docs.python.org/3/reference/simple_stmts.html#index-18) would otherwise provide.

        Returns:
            This instance, narrowed to the negative-bitflip subclass.

        Raises:
            TypeError: If this instance is not a
                [`negative_bitflip.Instance`][qubosolver.transforms.negative_bitflip.Instance].
        """
        from qubosolver.transforms import negative_bitflip

        return self._narrow(negative_bitflip.Instance)


# __init_subclass__ registers every subclass, but never fires for Instance
# itself, so it must be registered here explicitly.
Instance._registry[Instance._tag()] = Instance


# Density classification thresholds — half-open intervals [lo, hi).
# The HIGH bracket is closed on the right to include a fully dense matrix (1.0).
_SPARSE_THRESHOLD: tuple[float, float] = (0.0, 0.3)  # [0.0, 0.3)
_MEDIUM_THRESHOLD: tuple[float, float] = (0.3, 0.7)  # [0.3, 0.7)
_HIGH_THRESHOLD: tuple[float, float] = (0.7, 1.0)  # [0.7, 1.0]


def _classify_density(density: float) -> _DensityType:
    """Map a density value to a :class:`~._DensityType` category.

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
        density: Non-zero ratio in ``[0.0, 1.0]`` as returned by
            `_calculate_density`.

    Returns:
        Corresponding :class:`~._DensityType` member.

    Raises:
        ValueError: If *density* falls outside ``[0.0, 1.0]`` (e.g. negative
            values or values greater than 1).
    """
    if _SPARSE_THRESHOLD[0] <= density < _SPARSE_THRESHOLD[1]:
        return _DensityType.SPARSE
    elif _MEDIUM_THRESHOLD[0] <= density < _MEDIUM_THRESHOLD[1]:
        return _DensityType.MEDIUM
    elif _HIGH_THRESHOLD[0] <= density <= _HIGH_THRESHOLD[1]:
        return _DensityType.HIGH
    else:
        raise ValueError(f"Density {density} is outside the defined thresholds.")


def _calculate_density(m: Matrix) -> float:
    """Compute the fraction of non-zero entries in a coefficient matrix.

    Args:
        m: QUBO coefficient matrix of any shape.

    Returns:
        Value in ``[0.0, 1.0]`` where ``0.0`` means all entries are zero
            and ``1.0`` means no entry is zero. Returns ``0.0`` for empty
            matrices (``m.numel() == 0``) to avoid division by zero.
    """
    if m.numel() == 0:
        return 0.0
    return torch.count_nonzero(m).item() / m.numel()
