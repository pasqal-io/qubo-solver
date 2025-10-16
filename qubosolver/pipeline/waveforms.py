from __future__ import annotations

from typing import Any, Optional, cast
from numpy.typing import ArrayLike
import numpy as np
from qoolqit.register import Register
from qoolqit.drive import WeightedDetuning
from qoolqit.waveforms import Waveform, Constant
import scipy.interpolate as interpolate


class InterpolatedWaveform(Waveform):
    """A waveform created from interpolation of a set of data points.

    Arguments:
        duration: The waveform duration (in ns).
        values: Values of the interpolation points. Must be a list of castable
            to float or a parametrized object.
        times: Fractions of the total duration (between 0
            and 1), indicating where to place each value on the time axis. Must
            be a list of castable to float or a parametrized object. If
            not given, the values are spread evenly throughout the full
            duration of the waveform.
        interpolator: The SciPy interpolation class
            to use. Supports "PchipInterpolator" and "interp1d".
    """

    def __init__(
        self,
        duration: float,
        values: ArrayLike,
        times: Optional[ArrayLike] = None,
        interpolator: str = "PchipInterpolator",
        **interpolator_kwargs: Any,
    ):
        """Initializes a new InterpolatedWaveform."""
        super().__init__(duration, values=values)

        self._values = np.array(values, dtype=float)
        if times is not None:
            times = cast(ArrayLike, times)
            times_ = np.array(times, dtype=float)
            self._times = times_
        else:
            self._times = np.linspace(0, 1, num=len(self._values))

        valid_interpolators = ("PchipInterpolator", "interp1d")
        if interpolator not in valid_interpolators:
            raise ValueError(
                f"Invalid interpolator '{interpolator}', only "
                "accepts: " + ", ".join(valid_interpolators)
            )
        interp_cls = getattr(interpolate, interpolator)
        self._data_pts = np.array(
            [(round(t), v) for t, v in zip(self._times * (self._duration - 1), self._values)]
        )
        self._interp_func = interp_cls(
            self._data_pts[:, 0], self._data_pts[:, 1], **interpolator_kwargs
        )
        self._kwargs: dict[str, Any] = {
            "times": times,
            "interpolator": interpolator,
            **interpolator_kwargs,
        }

    def function(self, t: float) -> float:
        return float(self._interp_func(t))


def weighted_detunings(
    embedding: Register,
    duration: float,
    norm_weights: list[float],
    final_detuning: float | None = None,
) -> list[WeightedDetuning]:
    if final_detuning is not None:
        waveform = Constant(duration, final_detuning)
        return [
            WeightedDetuning(
                weights={embedding.qubit_ids[i]: w for i, w in enumerate(norm_weights)},
                waveform=waveform,
            )
        ]

    return list()
