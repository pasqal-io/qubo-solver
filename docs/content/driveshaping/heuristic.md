## Heuristic Drive Shaper

`HeuristicDriveShaper` constructs a fixed, closed-form analog drive directly from
the diagonal of the QUBO matrix, without running any classical pulse
optimization loop.

Compared with [`OptimizedDriveShaper`](./optimized.md), it is lighter and faster:
it builds the drive analytically from the problem structure and the hardware
limits. It is especially useful when the diagonal coefficients carry meaningful
site-dependent bias information.

The shaper returns:

- a generated `Drive`, ready to be simulated by the solver;
- an empty `Solution` placeholder. The actual bitstrings, probabilities, and
  costs are produced later, when the quantum simulation is executed.

---

### Principle

The shaper first normalizes the QUBO matrix:
$$
Q_{\mathrm{eff}} = \frac{Q}{Q_{\mathrm{offdiag, max}}}
$$
where $Q_{\mathrm{offdiag, max}}$ is the maximal off-diagonal coefficient of $Q$, and extracts its diagonal:
$$
q_i = (Q_{\mathrm{eff}})_{ii}.
$$

It then defines target final local detunings from these diagonal terms:
$$
d_i = -\frac{q_i}{2},
$$
Intuitively, the diagonal coefficients act as local biases, and the shaper
translates them into a final detuning landscape.

---

### Final Detuning Encoding

#### With DMM enabled

When a Detuning Map Modulator (DMM) is available, the shaper encodes the final
local targets exactly in the intended operating regime.

Let
$$
d_{\min} = \min_i d_i,
\qquad
d_{\max} = \max_i d_i.
$$

The final detuning is decomposed into:

- a **global detuning** $\delta_g(T) = d_{\max}$,
- a **DMM detuning amplitude** $\delta_{\mathrm{dmm}}(T) = -(d_{\max} - d_{\min}) \le 0$,
- and site-dependent weights $w_i = \frac{d_{\max} - d_i}{d_{\max} - d_{\min}} \in [0,1]$.

This gives the final local detuning
$$
\delta_i(T) = \delta_g(T) + \delta_{\mathrm{dmm}}(T)\, w_i = d_i.
$$

This sign convention is important: in this stack, `DMM` waveforms
must be non-positive, so the DMM contribution is encoded as a negative quantity.

If the DMM spread is negligible, or if no effective DMM range is available, the
DMM contribution is omitted.

#### Without DMM

If `dmm=False`, only a global detuning can be applied. In that case the shaper
cannot realize each $d_i$ individually, so it uses a single global final
value:
$$
\delta_g(T) = \mathrm{mean}_i(d_i),
$$
and no weighted detunings are declared.

---

### How $\Omega_{\max}$ is chosen

The plateau amplitude is derived from the detuning energy scale:
$$
\Omega_{\max}^{(\mathrm{raw})} = \kappa \max_i |d_i|.
$$

The current fallback default is:

- `heuristic_kappa = 0.25`

---

### Schedule Shape

The shaper uses the full sequence duration allowed by the device and constructs
simple interpolated waveforms.

#### Amplitude

The amplitude follows a flat-top profile:
$$
\Omega(t) = [\varepsilon,\ \Omega_{\max},\ \Omega_{\max},\ \varepsilon]
$$
with a very small $\varepsilon > 0$ at the endpoints rather than a strict zero.

#### Global detuning

The global detuning starts from the most negative hardware-allowed value and
then ramps to the final encoded value:
$$
\delta_g(t) = [\delta_0,\ \delta_0,\ \delta_g(T),\ \delta_g(T)],
\qquad
\delta_0 = -|\delta_{\max}|.
$$

This creates a simple three-stage pattern:

1. initial negative detuning,
2. drive-on plateau,
3. sweep toward the encoded final bias.

#### DMM weighted detuning

When DMM is active and the final DMM amplitude is strictly negative, the shaper
declares a weighted detuning map through `constant_weighted_dmm(...)`, using the
weights $w_i$ and final detuning $\delta_{\mathrm{dmm}}(T)$.

---

### Hardware Bounds Used

Hardware bounds are enforced at compilation-time by QoolQit (TODO: link reference)

---

### Register

The register should be normalized to have a minimal inter-atomic distance equal to 1 (plus a margin, e.g. 1.001). This can be done using the parameter `min_distance` from `EmbeddingConfig`.

---


### Configuration Parameters

| Field | Type | Description |
|-------|------|-------------|
| `drive_shaping_method` | `DriveType \| str` | Must be set to `DriveType.HEURISTIC` or `"heuristic"` |
| `dmm` | `bool` | Enables site-dependent final detuning through weighted DMM detunings |
| `heuristic_kappa` | `float` | Proportionality factor used to derive $\Omega_{\max}$ from the detuning scale; current fallback default: `0.25` |

These parameters are provided through `DriveShapingConfig` and read at drive
generation time.

---

### Notes and Limitations

- The method uses **only the diagonal** of the normalized QUBO matrix.
  Off-diagonal couplings are not used directly to shape the schedule.
- With DMM enabled, the final local targets are encoded analytically through the
  global detuning plus a weighted negative DMM correction.
- Without DMM, the method can only apply a single global final detuning, so the
  site-dependent targets are approximated by their mean.

---

### Example

```python
import torch
from qubosolver import Instance, SolverConfig, DriveShapingConfig, Solver, DriveType

Q = torch.tensor([
    [-1.0, 0.5, 0.2],
    [0.5, -2.0, 0.3],
    [0.2, 0.3, -3.0],
])

instance = Instance(Q)

config = SolverConfig(
    use_quantum=True,
    drive_shaping=DriveShapingConfig(
        drive_shaping_method=DriveType.HEURISTIC,
        dmm=True,
        heuristic_kappa=0.25,
    ),
)

solver = Solver(instance, config)
solution = solver.solve()

print(solution)
```
