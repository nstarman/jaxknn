"""All-points kNN scaling on GPU: jaxknn (tree, brute) vs jaxkd (pure JAX, CUDA) vs SciPy.

Same data and sizes as the CPU bench.py on assets/upstream-possibilities.
Writes results JSON incrementally (one row per measurement).
"""

import json
import platform
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import jaxkd
import jaxknn
import numpy as np
from scipy.spatial import cKDTree

K = 10
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
NAMES = ["stream", "blob", "interlopers"]
GIVE_UP = 60.0  # s; skip larger n for a (lib, data) once one steady call exceeds this
OUT = sys.argv[1]

dev = jax.devices()[0]
meta = dict(
    device=str(dev), device_kind=dev.device_kind, platform=dev.platform,
    jax=jax.__version__, jaxkd=getattr(jaxkd, "__version__", "0.1.2"),
    jaxknn=jaxknn.__version__, host=platform.node(),
)
print(meta, flush=True)
assert dev.platform == "gpu", dev


def data(name, n):  # same generator as the CPU benchmark
    rng = np.random.default_rng(0)
    if name == "blob":
        return rng.normal(size=(n, 3)).astype(np.float32)
    t = np.linspace(0, 1, n)
    p = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)] + rng.normal(0, 0.02, (n, 3))
    if name == "interlopers":
        i = rng.choice(n, n // 100, replace=False)
        pad = np.array([0.0, 3.0, 3.0])
        p[i] = rng.uniform(p.min(0) - pad, p.max(0) + pad, (len(i), 3))
    return p[rng.permutation(n)].astype(np.float32)


libs = {
    "jaxknn": jax.jit(lambda x: jaxknn.knn(x, K)),
    "jaxknn_brute": jax.jit(lambda x: jaxknn.brute_knn(x, K)),
    "jaxkd": jax.jit(lambda x: jaxkd.query_neighbors(jaxkd.build_tree(x), x, k=K + 1)),
}
try:  # jaxkd's CUDA extension (optional; a source build on this node)
    f = jax.jit(lambda x: jaxkd.query_neighbors(jaxkd.build_tree(x, cuda=True), x, k=K + 1, cuda=True))
    jax.block_until_ready(f(jnp.ones((64, 3), jnp.float32)))
    libs["jaxkd_cuda"] = f
    print("jaxkd cuda: available", flush=True)
except Exception:  # noqa: BLE001
    print("jaxkd cuda: unavailable\n" + traceback.format_exc(limit=2), flush=True)


def scipy_knn(x):
    return cKDTree(x).query(x, k=K + 1, workers=-1)


libs["scipy"] = scipy_knn


def timed(f, x, reps):
    t0 = time.perf_counter()
    jax.block_until_ready(f(x))
    first = time.perf_counter() - t0
    best = first
    for _ in range(reps if first < 20 else 1):
        t0 = time.perf_counter()
        jax.block_until_ready(f(x))
        best = min(best, time.perf_counter() - t0)
    return first, best


dead = set()
rows = []
for n in SIZES:
    for name in NAMES:
        x_np = data(name, n)
        x = jax.device_put(jnp.asarray(x_np), dev)
        ref = np.sort(scipy_knn(x_np)[0][:, 1:], axis=1)
        for lib, f in libs.items():
            if (lib, name) in dead:
                continue
            arg = x_np if lib == "scipy" else x
            reps = 3 if n <= 100_000 else 1
            try:
                first, best = timed(f, arg, reps)
            except Exception as e:  # noqa: BLE001 (e.g. out of memory)
                print(f"FAILED {lib} {name} {n}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                dead.add((lib, name))
                continue
            ok = None
            if lib.startswith("jaxknn"):  # neighbour distances match SciPy's
                d = np.sort(np.asarray(f(x)[1]), axis=1)
                ok = bool(np.allclose(d, ref, rtol=1e-4, atol=1e-6))
            row = dict(lib=lib, data=name, n=n, first_s=first, steady_s=best, matches_scipy=ok)
            rows.append(row)
            print(row, flush=True)
            if best > GIVE_UP:
                dead.add((lib, name))
            with open(OUT, "w") as fh:
                json.dump(dict(meta=meta, rows=rows), fh, indent=1)
