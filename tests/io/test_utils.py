from __future__ import annotations

import io
import logging
import struct
from pathlib import Path

import pytest
import pytest_check as check

from qubosolver._io.utils import (
    _MAGIC,
    _MAX_STRING_SIZE,
    _package_version,
    read_exact,
    save,
    load,
    save_header,
    load_header,
    save_sized_buffer,
    load_sized_buffer,
    save_string,
    load_string,
)
from qubosolver._io.utils import open as io_utils_open


class TestReadExact:
    def test_read_exact_success(self) -> None:
        data = b"hello world"
        stream = io.BytesIO(data)
        result = read_exact(stream, 5)
        check.equal(result, b"hello")

    def test_read_exact_full_data(self) -> None:
        data = b"test"
        stream = io.BytesIO(data)
        result = read_exact(stream, 4)
        check.equal(result, b"test")

    def test_read_exact_zero_length(self) -> None:
        stream = io.BytesIO(b"test")
        result = read_exact(stream, 0)
        check.equal(result, b"")

    def test_read_exact_negative_length(self) -> None:
        stream = io.BytesIO(b"test")
        with pytest.raises(ValueError, match="Length must be non-negative"):
            read_exact(stream, -1)

    def test_read_exact_insufficient_data(self) -> None:
        stream = io.BytesIO(b"test")
        with pytest.raises(EOFError, match="Expected 10 bytes, got 4"):
            read_exact(stream, 10)

    def test_read_exact_empty_stream(self) -> None:
        stream = io.BytesIO(b"")
        with pytest.raises(EOFError, match="Expected 1 bytes, got 0"):
            read_exact(stream, 1)


class TestSaveLoad:
    def test_save_load_integer(self) -> None:
        stream = io.BytesIO()
        value = 42
        save(stream, ">I", value)

        stream.seek(0)
        result = load(stream, ">I")
        check.equal(result, 42)

    def test_save_load_float(self) -> None:
        stream = io.BytesIO()
        value = 3.14159
        save(stream, ">f", value)

        stream.seek(0)
        result = load(stream, ">f")
        check.almost_equal(result, 3.14159)

    def test_save_load_multiple_values(self) -> None:
        stream = io.BytesIO()
        values = [123, 456, 789]

        for value in values:
            save(stream, ">I", value)

        stream.seek(0)
        results = []
        for _ in values:
            results.append(load(stream, ">I"))

        check.equal(results, values)

    def test_load_insufficient_data(self) -> None:
        stream = io.BytesIO(b"abc")  # Only 3 bytes, but integer needs 4
        with pytest.raises(EOFError):
            load(stream, ">I")


class TestSizedBuffer:
    def test_save_load_sized_buffer(self) -> None:
        stream = io.BytesIO()
        data = b"hello world"

        save_sized_buffer(stream, data)
        stream.seek(0)
        result = load_sized_buffer(stream)

        check.equal(result, data)

    def test_save_load_empty_buffer(self) -> None:
        stream = io.BytesIO()
        data = b""

        save_sized_buffer(stream, data)
        stream.seek(0)
        result = load_sized_buffer(stream)

        check.equal(result, data)

    def test_save_load_large_buffer(self) -> None:
        stream = io.BytesIO()
        data = b"x" * 10000

        save_sized_buffer(stream, data)
        stream.seek(0)
        result = load_sized_buffer(stream)

        check.equal(result, data)

    def test_load_sized_buffer_corrupted_size(self) -> None:
        stream = io.BytesIO(b"\x00\x00\x00\x05abc")  # Says 5 bytes but only has 3
        with pytest.raises(EOFError):
            load_sized_buffer(stream)


class TestString:
    def test_save_load_string_utf8(self) -> None:
        stream = io.BytesIO()
        text = "Hello, 世界!"

        save_string(stream, text)
        stream.seek(0)
        result = load_string(stream)

        check.equal(result, text)

    def test_save_load_string_custom_encoding(self) -> None:
        stream = io.BytesIO()
        text = "Hello, world!"
        encoding = "ascii"

        save_string(stream, text, encoding=encoding)
        stream.seek(0)
        result = load_string(stream, encoding=encoding)

        check.equal(result, text)

    def test_save_load_empty_string(self) -> None:
        stream = io.BytesIO()
        text = ""

        save_string(stream, text)
        stream.seek(0)
        result = load_string(stream)

        check.equal(result, text)

    def test_save_load_string_with_newlines(self) -> None:
        stream = io.BytesIO()
        text = "Line 1\nLine 2\r\nLine 3"

        save_string(stream, text)
        stream.seek(0)
        result = load_string(stream)

        check.equal(result, text)


class TestHeader:
    def test_save_load_header_roundtrip(self) -> None:
        stream = io.BytesIO()
        save_header(stream)
        stream.seek(0)

        check.equal(load_header(stream), _package_version())

    def test_header_precedes_the_payload(self) -> None:
        # load_header must leave the stream positioned exactly at the payload,
        # so a header can be prepended to any existing format.
        stream = io.BytesIO()
        save_header(stream)
        save_string(stream, "payload")
        stream.seek(0)

        load_header(stream)
        check.equal(load_string(stream), "payload")

    def test_load_header_rejects_wrong_magic(self) -> None:
        stream = io.BytesIO(b"NOTMAGIC" + b"whatever follows")

        with pytest.raises(ValueError, match="Not a qubosolver file"):
            load_header(stream)

    def test_load_header_rejects_a_truncated_header(self) -> None:
        stream = io.BytesIO(b"QUBO")

        with pytest.raises(EOFError):
            load_header(stream)

    def test_load_header_warns_on_a_version_mismatch_but_still_reads(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A mismatch is a debugging breadcrumb, not an error: the format is
        # deliberately unversioned, so loading must proceed.
        stream = io.BytesIO()
        stream.write(_MAGIC)
        save_string(stream, "0.0.1-from-the-past")
        save_string(stream, "payload")
        stream.seek(0)

        with caplog.at_level(logging.WARNING):
            writer_version = load_header(stream)

        check.equal(writer_version, "0.0.1-from-the-past")
        check.is_in("0.0.1-from-the-past", caplog.text)
        check.is_in(_package_version(), caplog.text)
        # The payload is still readable after the warning.
        check.equal(load_string(stream), "payload")

    def test_load_header_does_not_warn_on_a_matching_version(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        stream = io.BytesIO()
        save_header(stream)
        stream.seek(0)

        with caplog.at_level(logging.WARNING):
            load_header(stream)

        check.equal(caplog.text, "")


class TestSizeLimits:
    def test_load_sized_buffer_rejects_an_oversized_prefix(self) -> None:
        # The size prefix must be rejected before it is used to allocate, so a
        # corrupt or hostile prefix cannot request an enormous read.
        stream = io.BytesIO(struct.pack(">I", 4096))

        with pytest.raises(ValueError, match="exceeds the 1024-byte limit"):
            load_sized_buffer(stream, max_size=1024)

    def test_load_sized_buffer_accepts_a_payload_at_the_limit(self) -> None:
        stream = io.BytesIO()
        save_sized_buffer(stream, b"x" * 64)
        stream.seek(0)

        check.equal(load_sized_buffer(stream, max_size=64), b"x" * 64)

    def test_load_string_rejects_an_oversized_prefix(self) -> None:
        stream = io.BytesIO(struct.pack(">I", _MAX_STRING_SIZE + 1))

        with pytest.raises(ValueError, match="exceeds"):
            load_string(stream)


class TestOpen:
    def test_open_binary_file_path(self, tmp_path: Path) -> None:
        file = str(tmp_path / "test.bin")

        # Test writing
        with io_utils_open(file, "wb") as f:
            f.write(b"test data")

        # Test reading
        with io_utils_open(file, "rb") as f:
            data = f.read()
            check.equal(data, b"test data")

    def test_open_text_file_path(self, tmp_path: Path) -> None:
        file = str(tmp_path / "test.txt")

        # Test writing
        with io_utils_open(file, "w") as f:
            f.write("test data")

        # Test reading
        with io_utils_open(file, "r") as f:
            data = f.read()
            check.equal(data, "test data")

    def test_open_pathlib_path(self, tmp_path: Path) -> None:
        file = tmp_path / "test.txt"

        with io_utils_open(file, "wb") as f:
            f.write(b"pathlib test")

        with io_utils_open(file, "rb") as f:
            data = f.read()
            check.equal(data, b"pathlib test")

    def test_open_binary_io_object(self) -> None:
        stream = io.BytesIO(b"existing data")

        with io_utils_open(stream, "rb") as f:
            check.is_(f, stream)
            data = f.read()
            check.equal(data, b"existing data")

    def test_open_text_io_object(self) -> None:
        stream = io.StringIO("existing text")

        with io_utils_open(stream, "r") as f:
            check.is_(f, stream)
            data = f.read()
            check.equal(data, "existing text")

    def test_open_invalid_binary_io_type(self) -> None:
        text_stream = io.StringIO("text")
        # Type-checking also catches this error
        with pytest.raises(TypeError, match="Expected a binary file-like object"):
            with io_utils_open(text_stream, "rb"):  # type: ignore[call-overload]
                pass

    def test_open_invalid_text_io_type(self) -> None:
        binary_stream = io.BytesIO(b"binary")
        # Type-checking also catches this error
        with pytest.raises(TypeError, match="Expected a text file-like object"):
            with io_utils_open(binary_stream, "r"):  # type: ignore[call-overload]
                pass

    def test_open_default_mode(self, tmp_path: Path) -> None:
        file = tmp_path / "test.bin"

        # Default mode should be "wb"
        with io_utils_open(file) as f:
            f.write(b"default mode test")

        with io_utils_open(file, "rb") as f:
            data = f.read()
            check.equal(data, b"default mode test")


class TestIntegration:
    def test_complete_workflow(self) -> None:
        """Test a complete save/load workflow with mixed data types."""
        stream = io.BytesIO()

        # Save various data types
        save(stream, ">I", 42)
        save(stream, ">f", 3.14)
        save_string(stream, "Hello, world!")
        save_sized_buffer(stream, b"binary data")

        # Load everything back
        stream.seek(0)
        int_val = load(stream, ">I")
        float_val = load(stream, ">f")
        str_val = load_string(stream)
        bin_val = load_sized_buffer(stream)

        check.equal(int_val, 42)
        check.almost_equal(float_val, 3.14)
        check.equal(str_val, "Hello, world!")
        check.equal(bin_val, b"binary data")

    def test_file_roundtrip(self, tmp_path: Path) -> None:
        """Test saving to and loading from an actual file."""
        file = tmp_path / "test.bin"

        # Save data
        with file.open("wb") as f:
            save_string(f, "File test")
            save(f, ">Q", 9876543210)

        # Load data
        with file.open("rb") as f:
            text = load_string(f)
            number = load(f, ">Q")

        check.equal(text, "File test")
        check.equal(number, 9876543210)
