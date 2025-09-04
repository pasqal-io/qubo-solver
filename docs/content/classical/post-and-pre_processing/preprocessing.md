## QUBO Preprocessing

QUBO preprocessing is an optional step that attempts to reduce the size of the problem before solving, by deterministically fixing variables to 0 or 1 when possible.

### Activation

Preprocessing is enabled by setting `do_preprocessing=True` in the solver configuration (`SolverConfig`). No additional setup is required. This mechanism is supported by all solvers.

---

### Method

Two deterministic rules are applied iteratively until no further variables can be fixed:

- **Hansen Fixing Rule**
  A rule based on the diagonal and off-diagonal entries of the QUBO matrix. It fixes variables whose contribution to the objective function can be bounded independently of the rest of the problem.

- **Roof Duality**
  A technique based on duality theory that provides provably optimal variable fixations. It is implemented using the `roof_duality` function from the D-Wave Ocean SDK.

These rules are applied in sequence until convergence, reducing the QUBO instance before it is passed to the solver.

---

### Fixation Restoration

After solving the reduced QUBO, fixed variables are automatically reinserted into the solution bitstrings to restore their original size and order.

---

## Fields

| Field              | Type    | Description |
|--------------------|---------|-------------|
| `do_preprocessing`  | `bool`  | If `True`, activates preprocessing before solving. The solver will attempt to fix variables and reduce the QUBO size. |

---

### Example

```python exec="on" source="material-block" html="1"
from qubosolver import QUBOInstance
from qubosolver.solver import QuboSolver
from qubosolver.config import SolverConfig, ClassicalConfig

qubo = QUBOInstance(coefficients=[[-2.0, 1.0], [1.0, -2.0]])

# Create a SolverConfig object with classical solver options.
config = SolverConfig(
    use_quantum=False,
    classical=ClassicalConfig(classical_solver_type="dwave_tabu"),
    do_preprocessing=True
)

solver = QuboSolver(qubo, config)
solution = solver.solve()
print(solution)
```

---

### Notes

- Preprocessing does not introduce approximation or randomness. All variable fixations are guaranteed to be optimal with respect to the original QUBO.
- This step is particularly effective for sparse or structured QUBO matrices, where many variables can often be fixed early.

---

### References

- Hansen, P. (1979). *Method of non-linear 0-1 programming*. Annals of Discrete Mathematics, 5:53–70.
- D-Wave Systems Inc., *Ocean SDK — dwave.preprocessing.roof_duality*
  [doc](https://docs.dwavequantum.com/en/latest/ocean/api_ref_preprocessing/api_ref.html)
