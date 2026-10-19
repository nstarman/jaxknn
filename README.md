# jaxknn

Exact k-nearest neighbours in [JAX](https://github.com/jax-ml/jax): a kd-tree that traces under `jit`, `vmap` and `grad`.

```bash
pip install jaxknn
```

```python
>>> import jax
>>> import jax.numpy as jnp
>>> import jaxknn

>>> points = jax.random.normal(jax.random.key(0), (1000, 3))
>>> idx, dist = jaxknn.knn(points, 5)  # each point's 5 nearest *other* points
>>> idx.shape, dist.shape
((1000, 5), (1000, 5))

>>> queries = jnp.zeros((2, 3))
>>> idx, dist = jax.jit(lambda p, q: jaxknn.knn(p, 1, queries=q))(points, queries)
>>> idx.shape
(2, 1)

```

## What it gives you

- **Exact results.** No approximation. Ties go to the lower index, identically eagerly and under `jit`.
- **Fully traceable.** `knn` works inside `jit`, `vmap` and `grad`. Distances are differentiable with respect to the coordinates, with the neighbour selection held fixed, and the gradient is 0 rather than NaN at coincident points.
- **Robust inputs.**
  - NaN or inf input is an error, not a silently wrong answer: `ValueError` eagerly, a runtime error under `jit`.
  - Integers and bfloat16 are promoted to float32.
  - Coordinates are divided by an exact power of two when float32 would overflow or underflow, so values from about 1e-38 to 3e38 keep their neighbours. Typical cases are metres at kiloparsec scale, or tiny lengths.
- **Compiles once per size bucket.** Eager calls pad `n` to a bucket (2ʲ or 1.5·2ʲ), so inputs of different sizes share compiled code.
- **A raw tree for custom queries.** `jaxknn.tree` exposes `build_tree`, `knn`, `all_knn` and label-excluding queries (`knn(tree, q, k, exclude=(point_labels, query_labels))`: the nearest point with a *different* label, with whole same-label subtrees pruned).

`jaxknn.brute_knn` has the same contract by brute force. It's faster for small inputs.

## When to use it

Use jaxknn when the kNN has to live inside a traced JAX program: under `jit`, `vmap` or `grad`, or on an accelerator.

On a CPU, eagerly, SciPy's `cKDTree` is faster, because it queries with every core. Here's all-points kNN (k=10, a 3-D stream with 2% interlopers), comparing jaxknn compiled under `jit` with SciPy called eagerly. It was measured on an Apple M2 Pro under load, so the ratios matter more than the absolute times.

| n | `jit` compile (first call) | jaxknn under `jit`, steady | SciPy, eager | jaxknn / SciPy |
|---|---|---|---|---|
| 2,000 | 9.4 s | 17.8 ms | 6.0 ms | 2.9× |
| 10,000 | 11.6 s | 106 ms | 16.4 ms | 6.5× |
| 100,000 | 13.2 s | 1,261 ms | 148 ms | 8.5× |
| 1,000,000 | 26.8 s | 11,719 ms | 1,990 ms | 5.9× |

The first call compiles, which takes seconds per input shape and `k`. To pay that once per machine rather than once per process, turn on JAX's [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html):

```bash
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

Or, in Python, `jax.config.update("jax_compilation_cache_dir", ...)` before the first call. Later processes then load the compiled kNN from disk; tracing still runs each time. At n = 100,000, the first call in a new process fell from 6.8 s to 2.8 s.

Against [jaxkd](https://github.com/dodgebc/jaxkd) at n = 100,000 (all-points, k = 10, CPU):

| data | jaxknn | jaxkd | SciPy |
|---|---|---|---|
| stream | 307 ms | 815 ms | 73 ms |
| blob | 355 ms | 1,161 ms | 80 ms |
| stream + interlopers | 513 ms | ~140 s | 78 ms |

## Origin

jaxknn was extracted from [phasecurvefit](https://github.com/GalacticDynamics/phasecurvefit), where it is the traceable kNN behind `MSTOrderer`, with its history. See [phasecurvefit#197](https://github.com/GalacticDynamics/phasecurvefit/issues/197).

## License

MIT
