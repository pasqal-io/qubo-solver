# Quadratic Unconstrained Binary Optimization
Solving combinatorial optimization (CO) problems using quantum computing is one of those promising applications for the near term. The Quadratic Unconstrained Binary Optimization (QUBO) (also known as unconstrained binary quadratic programming) [^1] [^2] model enables to formulate many CO problems that can be tackled using quantum hardware. QUBO offers a wide range of applications from finance and economics to machine learning.

Given a QUBO problem made of $n$ binary variables, its formulation is:

$$\min_{x \in \{0,1\}^n} x^T Q x$$

where $Q \in \mathbb{R}^{n \times n}$ is a matrix of coefficients (generally an upper triangular matrix).

Expanding the formulation:

$$\min_{x \in \{0,1\}^n} \sum_{i=1}^{n} Q_{ii} x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} Q_{ij} x_i x_j$$


## QUBOInstance

`QUBOInstance` represents a single Quadratic Unconstrained Binary Optimization (QUBO) problem. It encapsulates the QUBO matrix, solution, and relevant metrics of interest.

### Features:
- Store the QUBO coefficient matrix (`coefficients`).
- Evaluate solutions to compute their cost.
- Automatically compute metrics:
  - **Density**: Fraction of non-zero elements in the matrix.
- Dynamically update the QUBO coefficients.

### Code Example:
```python exec="on" source="material-block" html="1"
from qubosolver import QUBOInstance

# Define a QUBO coefficient matrix
coefficients = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
instance = QUBOInstance(coefficients=coefficients)
print(instance)

solution = [1, 0, 1]
cost = instance.evaluate_solution(solution)
print(f"\nSolution Cost: {cost}")

# Save load
from qubosolver.saveload import save_qubo_instance, load_qubo_instance

save_qubo_instance(instance, "/tmp/qubo_instance.pt")
loaded_instance = load_qubo_instance("/tmp/qubo_instance.pt")
print(f"Loaded QUBOInstance: {loaded_instance}")
```



## QUBODataset

`QUBODataset` represents a collection of QUBO problems. It is designed to store coefficients in a matrix, and solutions for multiple qubo problems, allowing for efficient batch operations and random dataset generation.

### Features:
- Store a batch of QUBO coefficient matrices (`coefficients`).
- Optionally include solutions for each instance.
- Generate datasets with random matrices, configurable by:
    - Size of the QUBO matrix.
    - Density of non-zero elements.
    - Value bounds range.
- Access individual instances using indexing.

### Code Example:
```python exec="on" source="material-block" html="1"
from qubosolver.data import QUBODataset

# Generate a random dataset
dataset = QUBODataset.from_random(
    n_matrices=5, matrix_dim=4, densities=[0.3, 0.7], coefficient_bounds=(-10, 10), device="cpu"
)

# Access the first instance
coeffs, solution = dataset[0]
print(f"Coefficients: {coeffs}")
# Get the dataset size
print(f"\nDataset size: {len(dataset)}")

from qubosolver.saveload import save_qubo_dataset, load_qubo_dataset

# Save load
save_qubo_dataset(dataset, "/tmp/qubo_dataset.pt")
loaded_dataset = load_qubo_dataset("/tmp/qubo_dataset.pt")
print(f"\nLoaded QUBODataset size: {len(loaded_dataset)}")
```

---

# References

[^1]: [Glover et al., A Tutorial on Formulating and Using QUBO Models (2018)](https://arxiv.org/abs/1811.11538)
[^2]: [Glover et al., Quantum bridge analytics I: a tutorial on formulating and using QUBO models (2022)](https://link.springer.com/article/10.1007/s10479-022-04634-2)
