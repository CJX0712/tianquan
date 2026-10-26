"""
通信层不变量验证 (L1)

五条硬不变量
------------
I1 正确性: ring/tree allreduce 结果与 NumPy 全局归约**逐位一致**
I2 最优性: ring 每 rank 发送量 == 2(N-1)/N * S  (Patarasuk & Yuan 下界)
I3 等价性: ring 结果 == tree 结果
I4 可分解: reduce_scatter 后 all_gather == allreduce
I5 语义:   all_to_all 的 (i,j) 路由、broadcast 的根传播
"""

from __future__ import annotations

import numpy as np
import pytest

from tianquan.comm import (
    all_gather,
    all_to_all,
    broadcast,
    reduce_scatter,
    ring_allreduce,
    tree_allreduce,
)
from tianquan.comm.launch import run_threads

WS = [1, 2, 3, 4, 5, 8]


def _mk(rank: int, size: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(1234 + rank)
    return rng.standard_normal(size)


# --------------------------------------------------------------------------- #
# I1 正确性
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("world_size", WS)
def test_ring_allreduce_exact(world_size: int) -> None:
    """数值正确性: 与 NumPy 全局归约一致到 1e-12。

    注: float64 归约不满足结合律, 环式归约的顺序与 NumPy 不同,
    因此只要求 1e-12 量级一致 (实测偏差 ~1e-15, 属机器精度)。
    """
    size = 1024
    expected = sum(_mk(r, size) for r in range(world_size))

    def fn(rank, pg):
        return ring_allreduce(pg, _mk(rank, size))

    got = run_threads(fn, world_size)
    for g in got:
        np.testing.assert_allclose(g, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.invariant
@pytest.mark.parametrize("world_size", WS)
def test_ring_allreduce_bitwise_on_exact_values(world_size: int) -> None:
    """逻辑正确性: 当数值可精确表示时, 结果必须**逐位**等于全局归约。

    这条排除了"算法索引写错但被浮点误差掩盖"的可能 —— 与上面的数值测试互补。
    """
    size = 1024
    exact = lambda r: np.arange(size, dtype=np.float64) * (r + 1)  # noqa: E731
    expected = sum(exact(r) for r in range(world_size))

    def fn(rank, pg):
        return ring_allreduce(pg, exact(rank))

    got = run_threads(fn, world_size)
    for g in got:
        np.testing.assert_array_equal(g, expected)


@pytest.mark.invariant
@pytest.mark.parametrize("world_size", WS)
def test_tree_allreduce_exact(world_size: int) -> None:
    size = 1024
    exact = lambda r: np.arange(size, dtype=np.float64) * (r + 1)  # noqa: E731
    expected = sum(exact(r) for r in range(world_size))

    def fn(rank, pg):
        return tree_allreduce(pg, exact(rank))

    got = run_threads(fn, world_size)
    for g in got:
        np.testing.assert_array_equal(g, expected)


# --------------------------------------------------------------------------- #
# I2 通信量最优性 —— 这是本项目最硬的一条证据
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ring_allreduce_bandwidth_optimal(world_size: int) -> None:
    size = 1024
    elem_bytes = 8  # float64
    S = size * elem_bytes
    optimal = 2 * (world_size - 1) * (S // world_size)

    def fn(rank, pg):
        out = ring_allreduce(pg, _mk(rank, size))
        return {"bytes": pg.bytes_moved.value, "steps": pg.n_steps, "sum": float(out.sum())}

    res = run_threads(fn, world_size)
    for r in res:
        assert r["bytes"] == optimal, (
            f"N={world_size}: 实际发送 {r['bytes']}B, 理论下界 {optimal}B"
        )
        assert r["steps"] == 2 * (world_size - 1), f"步数应为 2(N-1)={2 * (world_size - 1)}"


@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_reduce_scatter_bandwidth_optimal(world_size: int) -> None:
    size = 1024
    optimal = (world_size - 1) * (size * 8 // world_size)

    def fn(rank, pg):
        reduce_scatter(pg, _mk(rank, size))
        return pg.bytes_moved.value

    got = run_threads(fn, world_size)
    for b in got:
        assert b == optimal, f"reduce_scatter 发送 {b}B, 理论 {optimal}B"


# --------------------------------------------------------------------------- #
# I3 算法等价性
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ring_equals_tree(world_size: int) -> None:
    size = 512

    def fn_ring(rank, pg):
        return ring_allreduce(pg, _mk(rank, size))

    def fn_tree(rank, pg):
        return tree_allreduce(pg, _mk(rank, size))

    a = run_threads(fn_ring, world_size)
    b = run_threads(fn_tree, world_size)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# I4 reduce_scatter + all_gather == allreduce
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_reduce_scatter_then_allgather(world_size: int) -> None:
    size = 1024
    expected = sum(_mk(r, size) for r in range(world_size))

    def fn(rank, pg):
        shard = reduce_scatter(pg, _mk(rank, size))
        return all_gather(pg, shard)

    got = run_threads(fn, world_size)
    for g in got:
        np.testing.assert_allclose(g, expected, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# I5 all_to_all 路由语义 / broadcast 根传播
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_all_to_all_routing(world_size: int) -> None:
    """rank i 的第 j 块应到达 rank j 的第 i 个位置。"""

    def fn(rank, pg):
        chunks = [np.full(4, rank * 100 + j, dtype=np.float64) for j in range(world_size)]
        return all_to_all(pg, chunks)

    got = run_threads(fn, world_size)
    for j, row in enumerate(got):
        for i, blk in enumerate(row):
            np.testing.assert_array_equal(blk, np.full(4, i * 100 + j))


@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("root", [0, 1])
def test_broadcast(world_size: int, root: int) -> None:
    if root >= world_size:
        pytest.skip("root 超出 world_size")
    payload = np.arange(16, dtype=np.float64) + 7.0

    def fn(rank, pg):
        my = payload.copy() if rank == root else np.zeros(16)
        return broadcast(pg, my, root=root)

    got = run_threads(fn, world_size)
    for g in got:
        np.testing.assert_array_equal(g, payload)
