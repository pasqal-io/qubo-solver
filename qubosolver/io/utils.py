from __future__ import annotations

import builtins
import io
import os
import struct

from typing import Any, Union, IO, Literal, overload, TypeVar, Sized
from typing_extensions import Buffer

from contextlib import nullcontext, AbstractContextManager

_T = TypeVar("_T", bytes, str)
FileLike = Union[str, os.PathLike[str], IO[_T]]


def read_exact(src: IO[bytes], length: int) -> bytes:
    if length < 0:
        raise ValueError("Length must be non-negative")
    data = src.read(length)
    actual_length = len(data)
    if actual_length != length:
        raise EOFError(f"Expected {length} bytes, got {actual_length}")
    return data


def save(output: IO[bytes], format: str, data: Any) -> None:
    output.write(struct.pack(format, data))


def load(src: IO[bytes], format: str) -> Any:
    return struct.unpack(format, read_exact(src, struct.calcsize(format)))[0]


def save_sized_buffer(output: IO[bytes], buffer: Buffer) -> None:
    assert isinstance(buffer, Sized)
    output.write(struct.pack(">I", len(buffer)))
    output.write(buffer)


def load_sized_buffer(src: IO[bytes]) -> bytes:
    size_fmt = ">I"
    length = struct.unpack(size_fmt, read_exact(src, struct.calcsize(size_fmt)))[0]
    return read_exact(src, length)


def save_string(output: IO[bytes], string: str, encoding: str = "utf-8") -> None:
    save_sized_buffer(output, string.encode(encoding))


def load_string(src: IO[bytes], encoding: str = "utf-8") -> str:
    return load_sized_buffer(src).decode(encoding)


@overload
def open(
    file_like: FileLike[bytes], mode: Literal["rb", "wb"]
) -> AbstractContextManager[IO[bytes]]:
    ... # pragma: no cover
@overload
def open(file_like: FileLike[str], mode: Literal["r", "w"]) -> AbstractContextManager[IO[str]]:
    ... # pragma: no cover

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
