## Utilities

Utilities provide helper functions to calculate and classify key metrics, as well as to save and load QUBO instances and datasets.

### Density
- **Calculate Density**: Compute the fraction of non-zero elements in the QUBO matrix.
- **Classify Density**: Categorize density as SPARSE, MEDIUM, or HIGH.

```python
from qubosolver.utils.density import calculate_density, classify_density

coefficients = torch.tensor([[0, 1, -2], [1, 0, 3], [-2, 3, 0]])
size = 3

# Calculate density
density = calculate_density(coefficients, size)

# Classify density
density_type = classify_density(density)

print(f"Density: {density}, Classified as: {density_type}")
```


### Save and Load
- **Save a QUBOInstance**: Save a `QUBOInstance` to a file.
- **Load a QUBOInstance**: Load a `QUBOInstance` from a file.
- **Save a QUBODataset**: Save a `QUBODataset` to a file.
- **Load a QUBODataset**: Load a `QUBODataset` from a file.

```python
from qubosolver.utils import save_qubo_instance, load_qubo_instance
from qubosolver import QUBOInstance

# Define a QUBO coefficient matrix
coefficients = [[0, 1, -2], [1, 0, 3], [-2, 3, 0]]
instance = QUBOInstance(coefficients=coefficients)

# Save and load a QUBOInstance
save_qubo_instance(instance, "qubo_instance.pt")
loaded_instance = load_qubo_instance("qubo_instance.pt")
print(loaded_instance)
```
```python
from qubosolver import QUBODataset
from qubosolver.utils import save_qubo_dataset, load_qubo_dataset

dataset = QUBODataset.from_random(n=5, size=4, densities=[0.3, 0.7], coefficient_bounds=(-10,10), device="cpu")

# Save and load a QUBODataset
save_qubo_dataset(dataset, "qubo_dataset.pt")
loaded_dataset = load_qubo_dataset("qubo_dataset.pt")
print(f"Loaded dataset size: {len(loaded_dataset)}")
```
---
