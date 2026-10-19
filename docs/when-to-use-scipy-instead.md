# When to use SciPy instead

jaxknn makes the kNN *traceable*; on a CPU it doesn't make it faster. SciPy's `cKDTree` queries with every core, while jaxknn's query is only partly parallel on CPU.

In the [CPU benchmark](comparison.md#benchmarks-on-cpu) (all-points kNN, k = 10, 3-D float32, n from 1,000 to 1,000,000), eager SciPy is 3–8× faster than jaxknn's steady state under `jit`, on a stream, a Gaussian blob and a stream with interlopers. jaxknn's first call also compiles, which takes seconds per input shape and `k` (see [compile time](comparison.md#compile-time)).

So:

- **Inside `jit`, `vmap` or `grad`, or on an accelerator:** use jaxknn. SciPy can't run on traced values.
- **Eager CPU code:** SciPy is faster and compiles nothing.
- **A GPU is available:** jaxknn is faster, even counting SciPy's CPU cores.

On a GPU it's the other way round. In the [GPU benchmark](comparison.md#benchmarks-on-gpu), on an NVIDIA L40S, jaxknn is 5–8× faster than SciPy on 16 CPU cores at 1M points (55 ms against 455 ms on the stream).
