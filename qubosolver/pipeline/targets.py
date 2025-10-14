"""
Code emitted by compilation.

In practice, this code is a very thin layer around Pulser's representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pulser
from qoolqit._solvers import Detuning


@dataclass
class Pulse:
    """
    Specification of a laser pulse to be executed on a quantum device

    Attributes:
        pulse (pulser.Pulse): The low-level Pulser pulse.
        norm_weights (list): List with normalized weights for applying DMM.
        duration (int, optional): Pulse duration in ns.
        final_detuning (float | None, optional): Final detuning parameter.
            Defaults to None, so no DMM is applied.
    """

    pulse: pulser.Pulse

    norm_weights: list = field(
        default_factory=list
    )  # List with normalized weights for applying DMM at the pulse

    duration: int = 4000
    final_detuning: float | None = None

    def draw(self) -> None:
        """
        Draw the shape of this laser pulse.
        """
        self.pulse.draw()

    def detuning(self, embedding: pulser.Register) -> list[Detuning]:
        if self.final_detuning is not None:
            waveform = pulser.ConstantWaveform(self.duration, self.final_detuning)
            return [
                Detuning(
                    weights={embedding.qubit_ids[i]: w for i, w in enumerate(self.norm_weights)},
                    waveform=waveform,
                )
            ]

        return list()


@dataclass
class Register:
    """
    Specification of a geometry of atoms to be executed on a quantum device

    Attributes:
        device: The quantum device targeted.
        register: The low-level Pulser register.
    """

    device: pulser.devices.Device
    register: pulser.Register

    def __len__(self) -> int:
        """
        The number of qubits in this register.
        """
        return len(self.register.qubits)

    def draw(self) -> None:
        """
        Draw the geometry of this register.
        """
        self.register.draw(blockade_radius=self.device.min_atom_distance + 0.01)
