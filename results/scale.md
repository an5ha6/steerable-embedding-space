# CPU scale benchmark

- CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- Python: 3.12.10
- NumPy: 2.5.5
- Data: synthetic float32, 384 dimensions; 1M measured in 100k chunks.

## apply()

| Vectors | Median total (s) | µs / vector | Vectors / s |
|---:|---:|---:|---:|
| 10,000 | 0.0197 | 1.965 | 508,872 |
| 100,000 | 0.3428 | 3.428 | 291,697 |
| 1,000,000 | 2.8726 | 2.873 | 348,115 |

## Nullspace direction

Steerer initialization and one direction lookup (5 runs, median): **0.0157 s**.
