"""Balanced, pointer-free kd-tree build with static shapes.

Node ``i`` at level ``l`` is the contiguous slice ``[i*m, (i+1)*m)`` of the
permuted points, ``m = n_pad >> l``. Each node splits at the exact median of its
widest-extent dimension. Padding rows (fewer than one per leaf) are +inf in every
dimension, so they sort last within a node and never set a split; per-node valid
counts are static, from :func:`layout`.

Build: one stable argsort per dimension up front, then per level a stable
partition of every dimension's order by cumsum + scatter (2.6x faster than a
batched argsort per level).
"""

__all__: tuple[str, ...] = ("Tree", "build_tree")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from ._layout import layout


class Tree(eqx.Module):
    """A built tree. Arrays are in tree (permuted) order unless noted."""

    points: Float[Array, "n_pad d"]  # padding rows are 0
    perm: Int[Array, " n_pad"]  # original index per row; ``n`` on padding
    valid: Bool[Array, " n_pad"]
    cell_lo: tuple[Float[Array, "_nodes d"], ...]  # per level 0..depth
    cell_hi: tuple[Float[Array, "_nodes d"], ...]
    split_dim: tuple[Int[Array, " _nodes"], ...]  # per level 0..depth-1
    split_val: tuple[Float[Array, " _nodes"], ...]
    n: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)  # effective, B_eff
    n_pad: int = eqx.field(static=True)

    @property
    def n_leaves(self) -> int:
        """Number of leaves, ``2**depth``."""
        return 2**self.depth


def build_tree(points: Float[Array, "n d"], /, *, leaf_size: int = 16) -> Tree:
    """Build a tree over ``points`` (``n >= 0`` rows, any dimension ``d >= 1``).

    The levels run in a ``lax.fori_loop`` over flat ``(n_pad,)`` arrays, so the
    compiled program does not grow with the depth: node ``i`` of the current
    level is the segment ``[i*m, (i+1)*m)``, and per-node results go to
    heap-ordered arrays (level ``l`` at rows ``2**l - 1 ...``), sliced back into
    per-level tuples at the end. Node-wise work runs on ``2**(depth-1)`` slots
    every level, masked past the level's ``2**l`` nodes.
    """
    n, d = points.shape
    lay = layout(n, leaf_size)
    n_pad, depth, inf = lay.n_pad, lay.depth, jnp.inf
    dt = points.dtype
    pp = jnp.concat([points, jnp.full((n_pad - n, d), inf, dt)])
    ords = jnp.stack(
        [jnp.argsort(pp[:, j], stable=True).astype(jnp.int32) for j in range(d)]
    )
    n_nodes = 2 ** (depth + 1) - 1  # heap rows, levels 0..depth
    slots = max(1, 2 ** (depth - 1))  # node slots per level (the widest level)
    # Static valid counts per node and per left child, heap-ordered (levels
    # 0..depth-1), padded so a dynamic slice of ``slots`` rows stays in bounds.
    v_heap = np.zeros(n_nodes + slots, np.int32)
    vl_heap = np.zeros(n_nodes + slots, np.int32)
    for lvl in range(depth):
        nn = 2**lvl
        v_heap[nn - 1 : 2 * nn - 1] = lay.leaf_valid.reshape(nn, -1).sum(1)
        vl_heap[nn - 1 : 2 * nn - 1] = lay.leaf_valid.reshape(2 * nn, -1).sum(1)[0::2]
    v_heap, vl_heap = jnp.asarray(v_heap), jnp.asarray(vl_heap)
    lo_heap = jnp.full((n_nodes + 2 * slots, d), -inf, dt)
    hi_heap = jnp.full((n_nodes + 2 * slots, d), inf, dt)
    dim_heap = jnp.zeros(n_nodes + slots, jnp.int32)
    val_heap = jnp.zeros(n_nodes + slots, dt)
    pos = jnp.arange(n_pad, dtype=jnp.int32)
    slot = jnp.arange(slots, dtype=jnp.int32)
    dims = jnp.arange(d)

    def level(lvl: Array, state: tuple) -> tuple:
        ords, lo_heap, hi_heap, dim_heap, val_heap = state
        nn = jnp.left_shift(1, lvl)
        m = jnp.right_shift(n_pad, lvl)
        h = m // 2
        off = nn - 1  # this level's first heap row
        live = slot < nn
        v = jax.lax.dynamic_slice(v_heap, (off,), (slots,))
        vl = jax.lax.dynamic_slice(vl_heap, (off,), (slots,))
        start = jnp.where(live, slot * m, 0)
        first = pp[ords[:, start], dims[:, None]]  # (d, slots)
        last = pp[ords[:, start + jnp.maximum(v - 1, 0)], dims[:, None]]
        dim = jnp.argmax(jnp.where((v > 0)[None], last - first, -inf), 0)
        node = pos // m
        loc = pos - node * m
        dim_p, v_p, vl_p = dim[node], v[node], vl[node]
        os_ = ords[dim_p, pos]  # each node's order on its split dimension
        left = (loc < vl_p) | ((loc >= v_p) & (loc < v_p + (h - vl_p)))
        side = jnp.zeros(n_pad, bool).at[os_].set(left)
        right_first = ords[dim, start + jnp.where(live, vl, 0)]
        split = pp[right_first, dim]
        s = side[ords]  # (d, n_pad)
        before = jnp.cumsum(s, 1) - s  # exclusive count of left rows
        n_left = before - before[:, node * m]  # ... within the node
        tgt = node * m + jnp.where(s, n_left, h + loc - n_left)
        rows = jnp.broadcast_to(dims[:, None], (d, n_pad))
        ords = (
            jnp.zeros((d, n_pad), jnp.int32)
            .at[rows, tgt]
            .set(ords, unique_indices=True)
        )
        onehot = jax.nn.one_hot(dim, d, dtype=bool)
        lo = jax.lax.dynamic_slice(lo_heap, (off, 0), (slots, d))
        hi = jax.lax.dynamic_slice(hi_heap, (off, 0), (slots, d))
        kids_lo = jnp.stack([lo, jnp.where(onehot, split[:, None], lo)], 1)
        kids_hi = jnp.stack([jnp.where(onehot, split[:, None], hi), hi], 1)
        # Rows past this level's nodes land in later levels' rows, which those
        # levels overwrite (and the last level's fit exactly).
        kid0 = 2 * nn - 1
        lo_heap = jax.lax.dynamic_update_slice(
            lo_heap, kids_lo.reshape(2 * slots, d), (kid0, 0)
        )
        hi_heap = jax.lax.dynamic_update_slice(
            hi_heap, kids_hi.reshape(2 * slots, d), (kid0, 0)
        )
        dim_heap = jax.lax.dynamic_update_slice(dim_heap, dim.astype(jnp.int32), (off,))
        val_heap = jax.lax.dynamic_update_slice(val_heap, split, (off,))
        return ords, lo_heap, hi_heap, dim_heap, val_heap

    ords, lo_heap, hi_heap, dim_heap, val_heap = jax.lax.fori_loop(
        0, depth, level, (ords, lo_heap, hi_heap, dim_heap, val_heap)
    )

    def per_level(heap: Array, levels: int) -> tuple[Array, ...]:
        return tuple(heap[2**lvl - 1 : 2 ** (lvl + 1) - 1] for lvl in range(levels))

    o = ords[0]
    real = o < n
    return Tree(
        points=jnp.where(real[:, None], pp[o], 0.0).astype(dt),
        perm=jnp.where(real, o, n).astype(jnp.int32),
        valid=real,
        cell_lo=per_level(lo_heap, depth + 1),
        cell_hi=per_level(hi_heap, depth + 1),
        split_dim=per_level(dim_heap, depth),
        split_val=per_level(val_heap, depth),
        n=n,
        depth=depth,
        leaf_size=lay.leaf_size,
        n_pad=n_pad,
    )
