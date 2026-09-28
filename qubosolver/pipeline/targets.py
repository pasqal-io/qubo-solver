"""
Code emitted by compilation.

In practice, this code is a very thin layer around Pulser's representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pulser


@dataclass
class Pulse:
    """
    Specification of a laser pulse to be executed on a quantum device

    Attributes:
        pulse: The low-level Pulser pulse.
    """

    pulse: pulser.Pulse

    norm_weights: list = field(
        default_factory=list
    )  # List with normalized weights for applying DMM at the pulse

    duration: int = 4000

    def draw(self) -> None:
        """
        Draw the shape of this laser pulse.
        """
        self.pulse.draw()


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
