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

On a CPU, eagerly, SciPy's `cKDTree` is 3–8× faster, and the first call under `jit` compiles, which takes seconds (once per program, not once per tree). On a GPU (an NVIDIA L40S), jaxknn is the fastest: 5–8× faster than SciPy on 16 CPU cores at 1M points. Against [jaxkd](https://github.com/dodgebc/jaxkd), jaxknn is 1.9–4.8× faster from 3,000 points up, and stays fast on data with scattered outliers. See [how jaxknn compares](https://nstarkman.space/jaxknn/comparison/) for the benchmarks and trade-offs.

## Origin

jaxknn was extracted from [phasecurvefit](https://github.com/GalacticDynamics/phasecurvefit), where it is the traceable kNN behind `MSTOrderer`, with its history. See [phasecurvefit#197](https://github.com/GalacticDynamics/phasecurvefit/issues/197).

## License

MIT
