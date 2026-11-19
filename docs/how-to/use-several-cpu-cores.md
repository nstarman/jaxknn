# Use several CPU cores

On CPU, JAX gives you one device by default, and jaxknn's query only uses part of the machine. If you ask XLA for several host devices, jaxknn splits the queries between them. Each device answers its own slice of the queries against a copy of the tree. The results are bit-identical to one device: the same neighbours, ties going to the lower index.

Set the flag before JAX is imported, for example in the shell:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python my_script.py
```

or at the top of the script, before any `import jax`:

<!-- skip: next -->

```python
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jaxknn
```

Nothing else changes in your code. jaxknn uses every host device when the default backend is the CPU; on a GPU it ignores them.

## How many devices

Measured speed-up against one device on 16-CPU Slurm allocations, k = 10, 3-D float32, under `jit` (query on a prebuilt tree / [`jaxknn.tree.all_knn`][]):

| cluster | n | N = 2 | N = 4 | N = 8 | N = 16 |
|---|---|---|---|---|---|
| MIT subMIT | 100k | 1.24 / 1.24 | 1.44 / 1.38 | 1.95 / 1.59 | 1.82 / 1.19 |
| MIT subMIT | 1M | 1.69 / 1.61 | 2.44 / 2.05 | 2.76 / 2.25 | 2.69 / 1.75 |
| MIT Engaging, AMD EPYC 7542 | 100k | 1.20 / 1.32 | 1.62 / 1.65 | 2.49 / 2.05 | 2.77 / 1.86 |
| MIT Engaging, AMD EPYC 7542 | 1M | 1.49 / 1.27 | 1.67 / 1.41 | 1.62 / 1.27 | 1.35 / 0.86 |

Half as many devices as cores, or a quarter, works best. With one device per core, `all_knn` can end up slower than with one device, probably because the tree build still runs once on every device. The nodes were shared with other jobs, so treat these as rough. [Benchmark script](https://github.com/nstarman/jaxknn/blob/e6e9533/tests/benchmarks/sharding.py).

That is less than SciPy's 3–8× lead over one device in the CPU benchmark; see [when to use SciPy instead](../when-to-use-scipy-instead.md).
