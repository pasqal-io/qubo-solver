### Running on the real QPU

To run the QUBO solver on the real QPU, you need to connect to the Pasqal Cloud. You can do this by following these steps:

1. Import the necessary libraries:

```python
import pasqal.cloud as pc
from qubo_solvers import SolverConfig, QPU
```

2. Connect to the Pasqal Cloud:

```python
connection = pc.Connection('your_email', 'your_password')
```

3. Fetch the available devices:

```python
device = pc.fetch_available_devices()[0]
```

4. Create a QPU instance with the correct number of shots:

```python
qpu = QPU(connection=connection, num_shots=100)
```

5. Create a SolverConfig instance with the QPU backend:

```python
config = SolverConfig(use_quantum=True, backend=qpu)
```

6. Run the QUBO solver:

```python
solver = Solver(config)
solver.solve()
```

### Example

Here is an example of how to run the QUBO solver on the real QPU:

```python
import pasqal.cloud as pc
from qubo_solvers import SolverConfig, QPU

connection = pc.Connection('your_email', 'your_password')

device = pc.fetch_available_devices()[0]

qpu = QPU(connection=connection, num_shots=100)

config = SolverConfig(use_quantum=True, backend=qpu)

solver = Solver(config)
solver.solve()
```

### Results

The results of the QUBO solver can be accessed through the `results` attribute of the solver instance:

```python
results = solver.results
```
