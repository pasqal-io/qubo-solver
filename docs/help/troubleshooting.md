# Troubleshooting

## Enabling logging

Qubo Solver uses the standard Python [`logging`][] module, under the `"qubosolver"` logger. By default, it stays silent — you need to configure logging yourself to see its output:

```python
import logging

logging.basicConfig()
logging.getLogger("qubosolver").setLevel(logging.DEBUG)
```

[`WARNING`][logging.WARNING] messages are actionable: they flag something you can likely fix yourself, such as an unusual or inconsistent input. [`INFO`][logging.INFO] and [`DEBUG`][logging.DEBUG] messages are lower-level diagnostic detail — they are less useful on their own, but can help when reporting an issue to support.
