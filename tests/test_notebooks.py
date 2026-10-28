# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Test examples scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

expected_fail: dict[str, str] = {}
skip: dict[str, str] = {
    "00-a-tour-of-qubo.ipynb": "Requires qubovert",
    "01-dataset-generation-and-loading.ipynb": "Must manually save data",
    "03-prepostprocessing.ipynb": "Must manually save data from notebook 01",
    "05-blade.ipynb": "Blade moved to Qoolqit",
    "08-qubo_analyzer.ipynb": "TODO: update with revamp",
    "09-decomposition.ipynb": "Flaky: Need a device with DMM other than DigitalAnalogDevice",
}


def get_ipynb_files(directory: Path) -> list[Path]:
    files = []

    for it in directory.iterdir():
        if it.suffix == ".ipynb" and not it.match("*.ipynb_checkpoints*"):
            files.append(it)
        elif it.is_dir() and it.name != "retired":
            files.extend(get_ipynb_files(it))
    return files


notebooks_dir = Path(__file__).parent.parent.joinpath("docs").joinpath("tutorials").resolve()
assert notebooks_dir.exists()
notebooks_files = get_ipynb_files(notebooks_dir)


def notebook_name(notebook: Path) -> str:
    return f"{notebook.relative_to(notebooks_dir)}"


notebooks_names = [notebook_name(example) for example in notebooks_files]

notebooks = []
for file in notebooks_files:
    filename = notebook_name(file)
    reason = expected_fail.get(filename)
    if reason is not None:  # no cov
        notebooks.append(pytest.param(file, marks=pytest.mark.xfail(reason=reason)))
        continue
    reason = skip.get(filename)
    if reason is not None:  # no cov
        notebooks.append(pytest.param(file, marks=pytest.mark.skip(reason=reason)))
        continue
    notebooks.append(pytest.param(file))


@pytest.mark.duration(160)
@pytest.mark.parametrize("notebook", notebooks, ids=notebooks_names)
def test_notebooks(notebook: Path) -> None:
    """Execute docs notebooks as a test, passes if it returns 0."""
    jupyter_cmd = ["-m", "jupyter", "nbconvert", "--to", "python", "--execute"]
    cmd: list[str | Path] = [sys.executable, *jupyter_cmd, notebook]
    py_file = notebook.with_suffix(".py")
    try:
        with subprocess.Popen(  # noqa: S603 (runs nbconvert on the repo's own notebooks)
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ},
        ) as run_example:
            stdout, stderr = run_example.communicate()
            error_string = (
                f"Notebook {notebook.name} failed\n"
                f"stdout:{stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )

        if run_example.returncode != 0:  # no cov
            pytest.fail(error_string)

    finally:
        # Cleanup always runs, even if errors happen
        py_file.unlink(missing_ok=True)
