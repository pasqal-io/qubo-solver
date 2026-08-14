## Local-Energy-Scale Drive Shaper

`LocalEnergyScaleDriveShaper` generates a fixed analog drive from:

- the diagonal of the QUBO matrix;
- the interactions returned by the embedded register;
- a scaling parameter \(\kappa\).

It does not run a pulse-parameter optimization loop.

The shaper returns the generated `Drive` and an empty `Solution`. Samples and
QUBO costs are produced when the drive is executed.

---

### Target Detunings

For a QUBO matrix \(Q\), the target detuning associated with variable \(i\) is

$$
d_i = -\frac{Q_{ii}}{2}.
$$

These values are taken directly from the matrix received by the drive shaper.

---

### DMM Encoding

The DMM is used only when:

- `dmm=True`;
- the selected device exposes a DMM detuning range;
- the target detunings are not all equal.

Let

$$
d_{\min}=\min_i d_i,
\qquad
d_{\max}=\max_i d_i,
\qquad
\Delta d=d_{\max}-d_{\min}.
$$

When \(\Delta d>10^{-15}\), the code sets

$$
\delta_g(T)=d_{\max},
$$

$$
\delta_{\mathrm{dmm}}(T)=-\Delta d,
$$

and

$$
w_i=
\frac{d_{\max}-d_i}{\Delta d}.
$$

The weights are clipped to \([0,1]\). The resulting local final detuning is

$$
\delta_i(T)
=
\delta_g(T)
+
\delta_{\mathrm{dmm}}(T)w_i.
$$

With the unscaled target detunings, this gives

$$
\delta_i(T)=d_i.
$$

If DMM is not used, the code applies one global final detuning:

$$
\delta_g(T)
=
\frac{1}{N}
\sum_{i=1}^{N}d_i.
$$

In that case,

$$
\delta_i(T)=\delta_g(T)
$$

for every qubit.

---

### Local Energy Scale

Let \(V_{ij}\) denote the interaction value returned by

```python
register.interactions()
```

for qubits \(i\) and \(j\).

For each qubit, the code accumulates

$$
I_i
=
\sum_{j\neq i}|V_{ij}|.
$$

The local energy scale is

$$
E_i
=
|\delta_i(T)|
+
I_i.
$$

The mean local energy scale is

$$
\overline{E}
=
\frac{1}{N}
\sum_{i=1}^{N}E_i.
$$

The raw peak Rabi frequency is

$$
\Omega_{\max}^{\mathrm{raw}}
=
\kappa\overline{E}.
$$

The default value is

```python
local_energy_scale_kappa = 0.25
```

---

### Hardware Adjustments

#### Amplitude

The code obtains the maximum compilable amplitude for the selected device and
register through `max_virtual_amplitude(...)`.

If

$$
\Omega_{\max}^{\mathrm{raw}}
>
\Omega_{\max}^{\mathrm{device}},
$$

the amplitude is clamped:

$$
\Omega_{\max}
=
\Omega_{\max}^{\mathrm{device}}.
$$

Otherwise,

$$
\Omega_{\max}
=
\Omega_{\max}^{\mathrm{raw}}.
$$

A warning is emitted when clamping occurs.

#### Detunings

The code defines the allowed detuning magnitude as

$$
d_{\mathrm{allowed}}
=
\rho\,\Omega_{\max}(1-10^{-3}),
$$

where \(\rho\) is returned by `detuning_amplitude_ratio(device)`.

If

$$
\max_i |d_i|
>
d_{\mathrm{allowed}},
$$

all target detunings are multiplied by

$$
\frac{d_{\mathrm{allowed}}}
     {\max_i |d_i|}.
$$

The global detuning and DMM encoding are then recomputed from the scaled target
detunings.

The local energy scale and \(\Omega_{\max}\) are not recomputed after this
detuning rescaling.

---

### Waveforms

The sequence duration is

```python
device.specs["max_duration"] or 1000.0
```

The amplitude waveform is

$$
\left[
\varepsilon,
\Omega_{\max},
\Omega_{\max},
\varepsilon
\right],
\qquad
\varepsilon=10^{-9}.
$$

The initial detuning is

$$
\delta_0
=
-\max_i |d_i|,
$$

using the possibly rescaled target detunings.

The global detuning waveform is

$$
\left[
\delta_0,
\delta_0,
\delta_g(T),
\delta_g(T)
\right].
$$

Both waveforms are created with `qoolqit.InterpolatedWaveform`.

When DMM is active, the weighted detuning waveform is created with
`constant_weighted_dmm(...)`.

---

### Configuration

| Field | Type | Description |
|---|---|---|
| `drive_shaping_method` | `DriveType \| str` | `DriveType.LOCAL_ENERGY_SCALE` or `"local_energy_scale"` |
| `dmm` | `bool` | Requests DMM encoding when supported by the device |
| `local_energy_scale_kappa` | `float` | Multiplies the mean local energy scale; default: `0.25` |

---

### Example

```python
from qubosolver import (
    DriveShapingConfig,
    DriveType,
    Instance,
    Solver,
    SolverConfig,
    matrix,
)

qubo = matrix.tensor(
    [
        [-6.0, 2.0, 2.0, 2.0],
        [2.0, -7.5, 2.0, 2.0],
        [2.0, 2.0, -7.5, 2.0],
        [2.0, 2.0, 2.0, -7.0],
    ]
)

instance = Instance(matrix=qubo)

config = SolverConfig(
    use_quantum=True,
    drive_shaping=DriveShapingConfig(
        drive_shaping_method=DriveType.LOCAL_ENERGY_SCALE,
        dmm=True,
        local_energy_scale_kappa=0.25,
    ),
)

solver = Solver(instance, config)
solution = solver.solve()

print(solution)
```