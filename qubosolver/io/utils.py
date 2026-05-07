from __future__ import annotations

import builtins
import io
import os
import struct
from contextlib import nullcontext

from typing import TYPE_CHECKING, overload, Sized

if TYPE_CHECKING:
    from typing import Any, Union, IO, Literal, TypeVar
    from typing_extensions import Buffer
    from contextlib import AbstractContextManager

    _T = TypeVar("_T", bytes, str)
    FileLike = Union[str, os.PathLike[str], IO[_T]]


def read_exact(src: IO[bytes], length: int) -> bytes:
    """Read exactly the specified number of bytes from a binary stream.

    This function ensures that exactly the requested number of bytes are read
    from the source stream. If fewer bytes are available than requested, an
    EOFError is raised.

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
    """Save data to a binary stream using struct format.

    This function packs data according to the specified struct format and writes
    it to the output stream. The format string follows Python's struct module
    conventions for binary data serialization.

    Args:
        output: A binary output stream to write the packed data to.
        format: A struct format string specifying how to pack the data
               (e.g., '>I' for big-endian unsigned int, 'f' for float).
        data: The data value to be packed and written. Must be compatible
              with the specified format.

    Returns:
        None
    """
    output.write(struct.pack(format, data))


def load(src: IO[bytes], format: str) -> Any:
    """Load and unpack a single value from a binary stream using struct format.

    This function reads the exact number of bytes required by the specified
    struct format from the source stream, unpacks the binary data according
    to the format, and returns the first (and typically only) unpacked value.

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
    """Save a buffer to a binary stream with its size prefix.

    This function writes a buffer to the output stream by first writing the
    buffer's length as a 4-byte big-endian unsigned integer, followed by the
    buffer's contents. This format allows the buffer to be read back later
    using load_sized_buffer().

    Args:
        output: A binary output stream to write the sized buffer to.
        buffer: A buffer object (bytes-like) that supports len() to be written.
               Must implement the Buffer protocol and be Sized.

    Returns:
        None

    Raises:
        AssertionError: If the buffer is not an instance of Sized.
        struct.error: If the buffer length cannot be packed as an unsigned int.
    """
    assert isinstance(buffer, Sized)
    output.write(struct.pack(">I", len(buffer)))
    output.write(buffer)


def load_sized_buffer(src: IO[bytes]) -> bytes:
    """Load a buffer from a binary stream that was saved with a size prefix.

    This function reads a buffer from the source stream that was previously
    written using save_sized_buffer(). It first reads a 4-byte big-endian
    unsigned integer representing the buffer length, then reads exactly that
    many bytes and returns them.

    Args:
        src: A binary input stream to read the sized buffer from.

    Returns:
        bytes: The buffer data that was read from the stream.

    Raises:
        EOFError: If fewer bytes are available than required by the size prefix
                 or if the stream ends before the complete buffer is read.
        struct.error: If the size prefix cannot be unpacked as an unsigned int.
    """
    size_fmt = ">I"
    length = struct.unpack(size_fmt, read_exact(src, struct.calcsize(size_fmt)))[0]
    return read_exact(src, length)


def save_string(output: IO[bytes], string: str, encoding: str = "utf-8") -> None:
    """Save a string to a binary stream with size prefix and encoding.

    This function encodes a string using the specified encoding and saves it
    to the output stream using save_sized_buffer(), which prefixes the encoded
    string with its byte length. This allows the string to be read back later
    using load_string().

    Args:
        output: A binary output stream to write the encoded string to.
        string: The string to be encoded and written to the stream.
        encoding: The character encoding to use when converting the string
                 to bytes. Defaults to "utf-8".

    Returns:
        None

    Raises:
        UnicodeEncodeError: If the string cannot be encoded using the
                           specified encoding.
        struct.error: If the encoded string length cannot be packed as
                     an unsigned int.
    """
    save_sized_buffer(output, string.encode(encoding))


def load_string(src: IO[bytes], encoding: str = "utf-8") -> str:
    """Load a string from a binary stream that was saved with size prefix and encoding.

    This function reads a string from the source stream that was previously
    written using save_string(). It first loads the sized buffer containing
    the encoded string bytes, then decodes those bytes back to a string using
    the specified encoding.

    Args:
        src: A binary input stream to read the encoded string from.
        encoding: The character encoding to use when converting the bytes
                 back to a string. Defaults to "utf-8". Must match the
                 encoding used when the string was saved.

    Returns:
        str: The decoded string that was read from the stream.

    Raises:
        EOFError: If fewer bytes are available than required by the size prefix
                 or if the stream ends before the complete string is read.
        UnicodeDecodeError: If the bytes cannot be decoded using the
                           specified encoding.
        struct.error: If the size prefix cannot be unpacked as an unsigned int.
    """
    return load_sized_buffer(src).decode(encoding)


@overload
def open(file_like: FileLike[bytes]) -> AbstractContextManager[IO[bytes]]:
    """Open a binary file-like object with default write mode.

    This overload handles binary file operations for file-like objects that work
    with bytes, using the default "wb" (write binary) mode. It provides a context
    manager for safe file handling with automatic resource cleanup.

    Args:
        file_like: A file-like object that works with bytes. Can be a file path
                  (str or PathLike) or an existing binary IO object.

    Returns:
        AbstractContextManager[IO[bytes]]: A context manager that yields a binary
        IO object when entered. The context manager handles proper resource cleanup.

    Raises:
        TypeError: If file_like is not a valid binary file-like object when it's
                  not a path.
        OSError: If the file cannot be opened (when file_like is a path).
    """
    ...  # pragma: no cover # fmt: skip


@overload
def open(
    file_like: FileLike[bytes], mode: Literal["rb", "wb"]
) -> AbstractContextManager[IO[bytes]]:
    """Open a binary file-like object with specified mode.

    This overload handles binary file operations for file-like objects that work
    with bytes. It provides a context manager for safe file handling with automatic
    resource cleanup.

    Args:
        file_like: A file-like object that works with bytes. Can be a file path
                  (str or PathLike) or an existing binary IO object.
        mode: The file access mode. Must be either "rb" for reading binary data
              or "wb" for writing binary data.

    Returns:
        AbstractContextManager[IO[bytes]]: A context manager that yields a binary
        IO object when entered. The context manager handles proper resource cleanup.

    Raises:
        TypeError: If file_like is not a valid binary file-like object when it's
                  not a path.
        OSError: If the file cannot be opened (when file_like is a path).
    """
    ...  # pragma: no cover # fmt: skip


@overload
def open(file_like: FileLike[str], mode: Literal["r", "w"]) -> AbstractContextManager[IO[str]]:
    """Open a text file-like object with specified mode.

    This overload handles text file operations for file-like objects that work
    with strings. It provides a context manager for safe file handling with automatic
    resource cleanup.

    Args:
        file_like: A file-like object that works with strings. Can be a file path
                  (str or PathLike) or an existing text IO object.
        mode: The file access mode. Must be either "r" for reading text data
              or "w" for writing text data.

    Returns:
        AbstractContextManager[IO[str]]: A context manager that yields a text
        IO object when entered. The context manager handles proper resource cleanup.

    Raises:
        TypeError: If file_like is not a valid text file-like object when it's
                  not a path.
        OSError: If the file cannot be opened (when file_like is a path).
    """
    ...  # pragma: no cover # fmt: skip


def open(
    file_like: Union[FileLike[bytes], FileLike[str]],
    mode: Literal["rb", "wb", "r", "w"] = "wb",
) -> AbstractContextManager[IO[bytes]] | AbstractContextManager[IO[str]]:

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
