from __future__ import annotations

import builtins
import io
import logging
import os
import struct
from contextlib import nullcontext
from importlib.metadata import version

from typing import overload, Sized
from qubosolver.types._checks import _RUNTIME_TYPE_CHECKING, TYPE_CHECKING

logger = logging.getLogger(__name__)

_MAGIC = b"QUBOSLVR"
"""Identifies a stream as written by this library, so a foreign or corrupt file
fails immediately with a clear error instead of being misread as valid data."""

_MAX_BUFFER_SIZE = 1 << 30
"""Ceiling (1 GiB) on a single length-prefixed payload, checked before allocating.
Keeps a corrupt or hostile size prefix from triggering a huge allocation."""

_MAX_STRING_SIZE = 1 << 10
"""Ceiling (1 KiB) on a length-prefixed string: ~1000 ASCII characters, fewer
if multibyte. Strings in these formats are type tags and version numbers, the
longest around 50 bytes."""

if TYPE_CHECKING:
    from typing import Any, Union, IO, Literal, TypeVar
    from typing_extensions import Buffer
    import typing
    from contextlib import AbstractContextManager

    _T = TypeVar("_T", bytes, str)
    if _RUNTIME_TYPE_CHECKING and not typing.TYPE_CHECKING:
        FileLike = Union[_T, Any]
    else:
        FileLike = Union[str, os.PathLike[str], IO[_T]]


def read_exact(src: IO[bytes], length: int) -> bytes:
    """Read exactly `length` bytes from a binary stream.

    Args:
        src: A binary input stream to read from.
        length: The exact number of bytes to read. Must be non-negative.

    Returns:
        bytes: The exact number of bytes read from the stream.

    Raises:
        ValueError: If length is negative.
        EOFError: If fewer bytes are available than requested.
    """
    if length < 0:
        raise ValueError("Length must be non-negative")
    data = src.read(length)
    actual_length = len(data)
    if actual_length != length:
        raise EOFError(f"Expected {length} bytes, got {actual_length}")
    return data


def save(output: IO[bytes], format: str, data: Any) -> None:
    """Pack `data` with `struct.pack` and write it to a binary stream.

    Args:
        output: A binary output stream to write the packed data to.
        format: A struct format string specifying how to pack the data
               (e.g., '>I' for big-endian unsigned int, 'f' for float).
        data: The data value to be packed and written. Must be compatible
              with the specified format.
    """
    output.write(struct.pack(format, data))


def load(src: IO[bytes], format: str) -> Any:
    """Read and unpack a single value from a binary stream with `struct`.

    Reads exactly the number of bytes required by `format` and returns the
    first (and typically only) unpacked value.

    Args:
        src: A binary input stream to read the packed data from.
        format: A struct format string specifying how to interpret the binary
               data (e.g., '>I' for big-endian unsigned int, 'f' for float).
               The format should typically specify a single value.

    Returns:
        Any: The unpacked value from the binary stream. The type depends on
             the struct format used (int, float, etc.).

    Raises:
        EOFError: If fewer bytes are available than required by the format.
        struct.error: If the format string is invalid or the data cannot
                     be unpacked according to the specified format.
    """
    return struct.unpack(format, read_exact(src, struct.calcsize(format)))[0]


def save_sized_buffer(output: IO[bytes], buffer: Buffer) -> None:
    """Write a buffer to a binary stream, prefixed with its length.

    The length is written as a 4-byte big-endian unsigned integer, followed
    by the buffer's contents, so it can be read back with `load_sized_buffer`.

    Args:
        output: A binary output stream to write the sized buffer to.
        buffer: A buffer object (bytes-like) that supports len() to be written.
               Must implement the Buffer protocol and be Sized.

    Raises:
        AssertionError: If the buffer is not an instance of Sized.
        struct.error: If the buffer length cannot be packed as an unsigned int.
    """
    assert isinstance(buffer, Sized)
    output.write(struct.pack(">I", len(buffer)))
    output.write(buffer)


def load_sized_buffer(src: IO[bytes], *, max_size: int = _MAX_BUFFER_SIZE) -> bytes:
    """Read a length-prefixed buffer previously written by `save_sized_buffer`.

    Args:
        src: A binary input stream to read the sized buffer from.
        max_size: Reject a size prefix larger than this, before reading. Guards
            against a corrupt or hostile prefix causing a huge allocation.

    Returns:
        bytes: The buffer data that was read from the stream.

    Raises:
        ValueError: If the size prefix exceeds `max_size`.
        EOFError: If fewer bytes are available than required by the size prefix
                 or if the stream ends before the complete buffer is read.
        struct.error: If the size prefix cannot be unpacked as an unsigned int.
    """
    size_fmt = ">I"
    length = struct.unpack(size_fmt, read_exact(src, struct.calcsize(size_fmt)))[0]
    # Checked before `read_exact` so an implausible prefix is rejected rather
    # than allocated.
    if length > max_size:
        raise ValueError(
            f"Refusing to read a {length}-byte payload: exceeds the {max_size}-byte limit."
        )
    return read_exact(src, length)


def save_string(output: IO[bytes], string: str, *, encoding: str = "utf-8") -> None:
    """Encode a string and write it to a binary stream with a length prefix.

    Uses `save_sized_buffer` internally, so it can be read back with
    `load_string`.

    Args:
        output: A binary output stream to write the encoded string to.
        string: The string to be encoded and written to the stream.
        encoding: The character encoding to use when converting the string
                 to bytes. Defaults to "utf-8".

    Raises:
        UnicodeEncodeError: If the string cannot be encoded using the
                           specified encoding.
        struct.error: If the encoded string length cannot be packed as
                     an unsigned int.
    """
    save_sized_buffer(output, string.encode(encoding))


def load_string(
    src: IO[bytes], *, encoding: str = "utf-8", max_size: int = _MAX_STRING_SIZE
) -> str:
    """Read a length-prefixed, encoded string previously written by `save_string`.

    Args:
        src: A binary input stream to read the encoded string from.
        encoding: The character encoding to use when converting the bytes
                 back to a string. Defaults to "utf-8". Must match the
                 encoding used when the string was saved.
        max_size: Reject a size prefix larger than this, before reading.

    Returns:
        str: The decoded string that was read from the stream.

    Raises:
        ValueError: If the size prefix exceeds `max_size`.
        EOFError: If fewer bytes are available than required by the size prefix
                 or if the stream ends before the complete string is read.
        UnicodeDecodeError: If the bytes cannot be decoded using the
                           specified encoding.
        struct.error: If the size prefix cannot be unpacked as an unsigned int.
    """
    return load_sized_buffer(src, max_size=max_size).decode(encoding)


def _package_version() -> str:
    """Version of this package, as written into file headers by `save_header`.

    Read from the installed distribution metadata rather than
    `qubosolver.__version__`, because `qubosolver/__init__.py` imports this
    module transitively and importing it back would be circular.
    """
    return version("qubo-solver")


def save_header(output: IO[bytes]) -> None:
    """Write the format header: magic bytes followed by this package's version.

    Written once at the start of a file by the outermost `save`. Nested
    objects (e.g. a `Solution` inside a `Dataset`) do not carry their own
    header.

    Args:
        output: A binary output stream to write the header to.
    """
    output.write(_MAGIC)
    save_string(output, _package_version())


def load_header(src: IO[bytes]) -> str:
    """Read and validate a header written by `save_header`.

    A wrong magic is a hard error: the stream is not one of ours, so reading
    on would misinterpret arbitrary bytes. A version mismatch is only a
    warning — these formats are deliberately not versioned, so the writer's
    version is recorded purely to make a later failure easier to diagnose.

    Args:
        src: A binary input stream positioned at the start of the header.

    Returns:
        str: The version of the package that wrote the stream.

    Raises:
        ValueError: If the magic bytes are missing or do not match.
        EOFError: If the stream ends inside the header.
    """
    magic = read_exact(src, len(_MAGIC))
    if magic != _MAGIC:
        raise ValueError(f"Not a qubosolver file: expected magic {_MAGIC!r}, got {magic!r}.")
    writer_version = load_string(src)
    current_version = _package_version()
    # Compared by equality only. Ordering PEP 440 strings would imply a
    # compatibility guarantee that these formats do not offer.
    if writer_version != current_version:
        logger.warning(
            "Reading a file written by qubosolver %s using qubosolver %s. "
            "This format is not versioned, so a mismatch may surface as a "
            "confusing error or as wrong values.",
            writer_version,
            current_version,
        )
    return writer_version


@overload
def open(file_like: FileLike[bytes]) -> AbstractContextManager[IO[bytes]]:
    """Open a binary file-like object, defaulting to write ("wb") mode."""
    ...  # pragma: no cover # fmt: skip


@overload
def open(
    file_like: FileLike[bytes], mode: Literal["rb", "wb"]
) -> AbstractContextManager[IO[bytes]]:
    """Open a binary file-like object with an explicit "rb" or "wb" mode."""
    ...  # pragma: no cover # fmt: skip


@overload
def open(file_like: FileLike[str], mode: Literal["r", "w"]) -> AbstractContextManager[IO[str]]:
    """Open a text file-like object with an explicit "r" or "w" mode."""
    ...  # pragma: no cover # fmt: skip


def open(
    file_like: Union[FileLike[bytes], FileLike[str]],
    mode: Literal["rb", "wb", "r", "w"] = "wb",
) -> AbstractContextManager[IO[bytes]] | AbstractContextManager[IO[str]]:
    """Open a file path or wrap an existing file-like object as a context manager.

    When given a path (str or PathLike), opens the file with the specified mode
    and returns the resulting file object as a context manager. When given an
    already-open IO object, wraps it in a nullcontext so it can be used in a
    ``with`` statement without closing it on exit.

    Args:
        file_like: A file path (str or PathLike) or an existing IO object.
            Binary modes (``"rb"``, ``"wb"``) require a bytes-mode IO object;
            text modes (``"r"``, ``"w"``) require a text-mode IO object.
        mode: The file access mode. One of ``"rb"``, ``"wb"``, ``"r"``, or
            ``"w"``. Defaults to ``"wb"`` (write binary).

    Returns:
        AbstractContextManager[IO[bytes]] | AbstractContextManager[IO[str]]:
        A context manager yielding the appropriate IO object. Paths opened here
        are closed on context exit; pre-existing IO objects are left open.

    Raises:
        TypeError: If ``file_like`` is an IO object whose type does not match
            the requested mode (e.g. a text stream passed with ``"rb"``).
        OSError: If a file path cannot be opened (e.g. permission denied,
            file not found).
    """
    if "b" in mode:
        if isinstance(file_like, (str, os.PathLike)):
            return builtins.open(file_like, mode)
        if not isinstance(file_like, (io.RawIOBase, io.BufferedIOBase)):
            raise TypeError("Expected a binary file-like object.")
        return nullcontext(file_like)
    else:
        if isinstance(file_like, (str, os.PathLike)):
            return builtins.open(file_like, mode, encoding="utf-8")
        if not isinstance(file_like, io.TextIOBase):
            raise TypeError("Expected a text file-like object.")
        return nullcontext(file_like)
