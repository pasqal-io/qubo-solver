from __future__ import annotations

from dataclasses import dataclass, field
import torch

from qoolqit import Device, AnalogDeviceWithDMM
from qoolqit.execution import QPU

from qubosolver.types import LocalEmulator, RemoteEmulator
from qubosolver import embedding, drive_shaping


@dataclass
class Config():
    """A `quantum.Config` instance defines the quantum part of a `SolverConfig`.

    Attributes:
        embedding (embedding.Config, optional): Embedding part configuration of the solver.
        drive_shaping (drive_shaping.Config, optional): Drive-shaping part configuration
            of the solver.
        backend (LocalEmulator | RemoteEmulator | QPU, optional): backend
            for running quantum programs. Note that parameters
            such as `dt` are directly set when creating LocalEmulator | RemoteEmulator | QPU,
            hence they are deprecated compared to previous qubo-solver versions.
            Also the number of shots is set there as well.
            Defaults to a LocalEmulator using qutip.
        device (Device, optional): The quantum device specification. Defaults to `AnalogDeviceWithDMM`.
    """

    embedding: embedding.Config = field(default_factory=embedding.Config)
    drive_shaping: drive_shaping.Config = field(default_factory=drive_shaping.Config)
    backend: LocalEmulator | RemoteEmulator | QPU = field(default_factory=LocalEmulator)
    device: Device = field(default_factory=AnalogDeviceWithDMM)

    @property
    def max_min_dist_ratio(self) -> float:
        """Maximum allowed ratio between the largest and smallest inter-atom distance.

        Resolves ``embedding.max_min_dist_ratio``: returns it directly unless it is
        the sentinel ``"device"``, in which case the ratio is derived from the
        configured device's ``max_radial_distance`` / ``min_distance`` specs
        (or ``inf`` when the device imposes no such limits).

        Returns:
            float: The resolved maximum min/max distance ratio.
        """
        if self.embedding.max_min_dist_ratio != "device":
            return self.embedding.max_min_dist_ratio
        specs = self.device.specs
        min_distance = specs["min_distance"]
        max_radial_distance = specs["max_radial_distance"]
        if min_distance is not None and min_distance > 0 and max_radial_distance is not None:
            return max_radial_distance / min_distance
        return torch.inf
