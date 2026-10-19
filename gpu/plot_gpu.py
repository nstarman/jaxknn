"""Plot GPU all-points kNN scaling (results JSON from bench_gpu.py)."""

import json
import sys

import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
rows, meta = d["rows"], d["meta"]
TITLES = {"stream": "stream", "blob": "Gaussian blob", "interlopers": "stream + 1% interlopers"}
STYLE = {
    "jaxknn": dict(color="#2a6fdb", label="jaxknn (jit)"),
    "jaxknn_brute": dict(color="#2a6fdb", ls=":", label="jaxknn.brute_knn (jit)"),
    "jaxkd": dict(color="#d9822b", label="jaxkd, pure JAX (jit)"),
    "scipy": dict(color="#7a7a7a", ls="--", label="SciPy cKDTree (eager, 16 CPU cores)"),
}

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax in axes[1:3]:
    ax.sharey(axes[0])
for ax, name in zip(axes, TITLES):
    for lib, st in STYLE.items():
        pts = sorted((r["n"], r["steady_s"]) for r in rows if r["lib"] == lib and r["data"] == name)
        if pts:
            ax.loglog(*zip(*pts), marker="o", ms=4, **st)
    ax.set_title(TITLES[name])
    ax.set_xlabel("number of points n")
    ax.grid(True, which="major", alpha=0.3)
axes[0].set_ylabel("all-points kNN, k = 10 (s, steady state)")
axes[0].legend(frameon=False, fontsize=8)
ax = axes[3]
for lib, st in STYLE.items():
    if lib == "scipy":
        continue
    pts = sorted((r["n"], r["first_s"]) for r in rows if r["lib"] == lib and r["data"] == "stream")
    ax.semilogx(*zip(*pts), marker="o", ms=4, **st)
ax.set_title("first call (compile + run), stream")
ax.set_xlabel("number of points n")
ax.set_ylabel("s")
ax.set_ylim(bottom=0)
ax.grid(True, which="major", alpha=0.3)
fig.suptitle(f"Exact all-points kNN on GPU (3-D, float32; {meta['device_kind']}, jax {meta['jax']})", fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150)
