"""Query speed-up from sharding across N host CPU devices (nstarman/jaxknn#16).

``XLA_FLAGS=--xla_force_host_platform_device_count=N`` must be set before JAX
is imported, so each N runs in a fresh subprocess. Times are steady state under
``jit``, best of ``--reps``: the raw query ``tree.knn(tree, points, k)`` on a
prebuilt tree, and the whole ``tree.all_knn``. Each N's neighbours are checked
bit-for-bit against N = 1.

    uv run --group bench python tests/benchmarks/sharding.py --sizes 100000 1000000
"""

import argparse
import functools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _child(ndev: int, n: int, reps: int, out: str) -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from jaxknn import tree as kd  # noqa: PLC0415

    assert len(jax.devices()) == ndev
    rng = np.random.default_rng(0)  # the stream of tests/benchmarks/test_tree.py
    t = np.linspace(0, 1, n)
    p = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)] + rng.normal(0, 0.02, (n, 3))
    p = jnp.asarray(p[rng.permutation(n)].astype(np.float32))

    def best(f: object, *a: object) -> float:
        jax.block_until_ready(f(*a))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(f(*a))
            ts.append(time.perf_counter() - t0)
        return min(ts)

    tree = jax.jit(kd.build_tree)(p)
    query = jax.jit(functools.partial(kd.knn, k=10))
    all_knn = jax.jit(functools.partial(kd.all_knn, k=10))
    tq, ta = best(query, tree, p), best(all_knn, p)
    np.savez(out, *map(np.asarray, (*query(tree, p), *all_knn(p))))
    print(json.dumps({"devices": ndev, "n": n, "query": tq, "all_knn": ta}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    ap.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4, 8, 10])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json", type=Path, help="also write the results here")
    a = ap.parse_args()
    import numpy as np  # noqa: PLC0415

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in a.sizes:
            base = None
            for ndev in a.devices:
                out = f"{tmp}/{ndev}_{n}.npz"
                env = dict(
                    os.environ,
                    JAX_PLATFORMS="cpu",
                    XLA_FLAGS=f"--xla_force_host_platform_device_count={ndev}",
                )
                cmd = [
                    sys.executable,
                    __file__,
                    "--child",
                    str(ndev),
                    str(n),
                    str(a.reps),
                    out,
                ]
                res = subprocess.run(  # noqa: S603
                    cmd, env=env, check=True, capture_output=True, text=True
                )
                row = json.loads(res.stdout.splitlines()[-1])
                got = np.load(out)
                base = base or {k: got[k] for k in got.files}
                row["identical"] = all(
                    np.array_equal(got[k], base[k]) for k in got.files
                )
                row["load"] = os.getloadavg()[0]
                rows.append(row)
                one = next(r for r in rows if r["n"] == n)
                print(
                    f"n={n:>8} N={ndev:>2}  query {row['query']:.3f} s "
                    f"({one['query'] / row['query']:.2f}x)  all_knn "
                    f"{row['all_knn']:.3f} s ({one['all_knn'] / row['all_knn']:.2f}x)  "
                    f"identical={row['identical']}  load={row['load']:.0f}",
                    flush=True,
                )
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        _child(*map(int, sys.argv[2:5]), sys.argv[5])
    else:
        main()
