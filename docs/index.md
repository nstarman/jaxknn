# jaxknn

Exact k-nearest neighbours in [JAX](https://github.com/jax-ml/jax): a kd-tree that traces under `jit`, `vmap` and `grad`.

```bash
pip install jaxknn
```

```python
import jax
import jax.numpy as jnp
import jaxknn

points = jax.random.normal(jax.random.key(0), (1000, 3))

# Each point's 5 nearest *other* points (self excluded by index).
idx, dist = jaxknn.knn(points, 5)
assert idx.shape == dist.shape == (1000, 5)

# The 3 nearest points to each query.
queries = jnp.zeros((2, 3))
idx, dist = jaxknn.knn(points, 3, queries=queries)
assert idx.shape == (2, 3)
```

Both return `(indices, distances)`:
- distances are Euclidean, with each row sorted ascending;
- equidistant neighbours go to the lower index;
- a missing neighbour (fewer than `k` candidates) is index `len(points)` with distance `inf`.

## Where to go next

- **[Use it under `jit`, `vmap` and `grad`](how-to/use-under-jit-vmap-grad.md):** the reason jaxknn exists.
- **[Find the nearest point with another label](how-to/find-the-nearest-point-with-another-label.md):** for example, bridging the components of a graph.
- **[Tune it](how-to/tune-for-large-k-or-high-dimension.md)** for large `k` or high dimension.
- **[How jaxknn compares](comparison.md)** with jaxkd and SciPy: what it does differently, CPU benchmarks, and the trade-offs.
- **[How it works](how-it-works.md)** and **[when to use SciPy instead](when-to-use-scipy-instead.md)**.
- **Reference:** [`jaxknn`](api/jaxknn.md) (the safe entry points) and [`jaxknn.tree`](api/tree.md) (the raw tree).
