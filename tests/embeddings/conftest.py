import pytest

from qoolqit.devices.device import AnalogDeviceWithDMM
from pulser.devices._device_datacls import BaseDevice

@pytest.fixture
def device() -> BaseDevice:
    return AnalogDeviceWithDMM()._device


@pytest.fixture
def max_min_dist_ratio(device: BaseDevice) -> float:
    assert isinstance(device.max_radial_distance, int)
    assert isinstance(device.min_atom_distance, int)
    return device.max_radial_distance / device.min_atom_distance