import dataclasses

from qubosolver.config import SolverConfig, max_duration_ratio
from qoolqit.execution.compilation_functions import CompilerProfile
from qoolqit import Device, QuantumProgram, Drive, Register


def create_compiled_program(
    device: Device, config: SolverConfig, drive: Drive, embedding: Register
) -> QuantumProgram:
    """
    Create a compiled QuantumProgram from the drive and embedding.

    Args:
        drive (Drive): Drive schedule containing the quantum operations.
        embedding (Register): Register defining the qubit layout.

    Returns:
        QuantumProgram: Compiled quantum program.
    """
    program = QuantumProgram(
        register=embedding,
        drive=drive,
    )

    if device.specs["max_duration"] is None:
        device_with_duration = dataclasses.replace(
            device._device,
            max_sequence_duration=config.drive_shaping.default_sequence_duration,
        )
        device = Device(
            pulser_device=device_with_duration,
            default_converter=device.converter,
        )

    program.compile_to(
        device,
        profile=CompilerProfile.MAX_ENERGY,
        device_max_duration_ratio=max_duration_ratio(device),
    )
    return program
