# How jaxknn compares

jaxknn is an exact kd-tree kNN that lives inside traced JAX programs. The closest alternatives are [jaxkd](https://github.com/dodgebc/jaxkd), another JAX kd-tree, and SciPy's [`cKDTree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html), which can't be traced but is fast on CPU. This page sets out what jaxknn does differently, how fast it is, and what it costs.

## What jaxknn does differently

- **Exact, deterministic ties.** Equidistant neighbours go to the lower index, identically eagerly and under `jit`, independent of tree layout and padding. The tests check this against a float64 brute-force oracle on Gaussian, stream, integer-grid and duplicate-clump data.
- **Robust to scattered outliers.** On a stream with 1% uniformly scattered interlopers, jaxkd's query time grows steeply: ~40 s at n = 100k and ~275 s at 300k, against 0.5 s and 1.8 s for jaxknn (third panel below).
- **No float32 overflow or underflow.** Coordinates are divided by an exact power of two when float32 squared distances would overflow or underflow. For example, metres at kiloparsec scale (~3e19) keep their neighbours. Dividing by a power of two is exact, so neighbour selection, ties included, is unchanged.
- **Traceable.** Works under `jit`, `vmap` and `grad`. Gradients go through a plain gather, and are 0 rather than NaN at coincident points. See [use it under `jit`, `vmap` and `grad`](how-to/use-under-jit-vmap-grad.md).
- **Eager size buckets.** Eager calls pad n to a bucket (2ʲ or 1.5·2ʲ), so differing sizes share compiled code.
- **Label-excluding queries.** `knn(tree, q, k, exclude=(point_labels, query_labels))` returns the nearest point with a *different* label and prunes same-label subtrees. See [find the nearest point with another label](how-to/find-the-nearest-point-with-another-label.md).

[How it works](how-it-works.md) explains the mechanics behind each of these.

## Benchmarks on CPU

![All-points kNN time against n for jaxknn, jaxkd and SciPy on three datasets, and first-call time on the stream](_static/knn_scaling.png)

All-points kNN, k = 10, 3-D float32, n from 1,000 to 1,000,000. jaxknn and jaxkd ran under `jit`, steady state (best of 3); SciPy's `cKDTree` ran eagerly on all cores. The machine was an Apple M2 Pro (10 cores) under load (load average ~10), so the ratios matter more than the absolute times. Versions: jaxknn `main` at f093366, jax 0.8.2, jaxkd 0.1.2, SciPy 1.18.1. [Script](https://github.com/nstarman/jaxknn/blob/7f53c418cc8434cd075c5ed2d0a08de634b2e284/bench.py), [plotting script](https://github.com/nstarman/jaxknn/blob/7f53c418cc8434cd075c5ed2d0a08de634b2e284/plot.py) and [raw numbers](https://github.com/nstarman/jaxknn/blob/7f53c418cc8434cd075c5ed2d0a08de634b2e284/results_f093366.json).

Reading the panels, left to right:

1. **Stream** (a thin, curved 3-D stream). jaxknn and jaxkd tie at n = 1,000 (4.4 ms against 4.3 ms); from 3,000 up, jaxknn is 1.9–4.5× faster.
2. **Gaussian blob.** jaxknn is 2.8–4.8× faster than jaxkd at every size.
3. **Stream + 1% interlopers** (the stream, with 1% of points scattered uniformly around it). jaxknn slows modestly against the clean stream (0.52 s against 0.31 s at 100k). jaxkd's time grows steeply, to 41 s at 100k and 274 s at 300k; its curve stops there because that point alone took 4.5 minutes.
4. **First call** (compile and run) on the stream. This is jaxknn's weak spot, covered below. The dotted line is jaxknn v0.1.0, for reference.

In all three timing panels, SciPy is below both JAX libraries.

| n = 100,000, steady state | jaxknn | jaxkd | SciPy |
|---|---|---|---|
| stream | 0.31 s | 0.81 s | 0.07 s |
| Gaussian blob | 0.35 s | 1.05 s | 0.08 s |
| stream + interlopers | 0.52 s | 41 s | 0.08 s |

## Trade-offs

### Compile time

The first call compiles. On the stream it takes 1.7–3.8 s up to 300k points, against about a second or less for jaxkd up to 100k; at 300k the two are level. At 1M points the run time dominates and jaxknn's first call is faster (6.1 s against 15.6 s).

Compiling happens once per input shape and `k`. New data of the same shape reuses the compiled program, and eagerly so does any n in the same size bucket. To keep it across processes too, turn on JAX's [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html):

```bash
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

Or, in Python, `jax.config.update("jax_compilation_cache_dir", ...)` before the first call. Later processes then load the compiled kNN from disk; tracing still runs each time.

### SciPy is faster on CPU

When you don't need tracing, SciPy's `cKDTree` is 3–8× faster than jaxknn here, and compiles nothing. See [when to use SciPy instead](when-to-use-scipy-instead.md).

### GPU

Not yet measured. jaxkd is designed to be GPU-friendly and has a CUDA path; jaxknn was developed and tuned on CPU/XLA.

<!-- GPU results (H200, L40S) go here once measured. -->
