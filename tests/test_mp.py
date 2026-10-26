"""
真实多进程 (spawn) 测试

标记 ``mp``: 需要启动子进程, 较慢, CI 单独跑一档。

worker 函数**必须在模块顶层** —— Windows/macOS 的 spawn 会 pickle 函数,
嵌套函数/lambda 会失败。这是写多进程测试时最容易踩的坑。

历史上这里抓到过一个真实 bug: 共享内存后端 ``np.frombuffer`` 返回只读数组,
而 DDP 会对 allreduce 结果做原地 ``out /= world_size``。线程后端永远复现不了,
只有真实跨进程才暴露 —— 所以这一档测试不可省略。
"""

from __future__ import annotations

import numpy as np
import pytest

from tianquan.comm.launch import spawn_world
from tianquan.comm import ring_allreduce, broadcast, reduce_scatter, all_gather


def _ring_worker(rank, pg, size: int):
    x = np.arange(size, dtype=np.float64) * (rank + 1)
    out = ring_allreduce(pg, x)
    return {"sum": float(out.sum()), "bytes": pg.bytes_moved.value}


def _bcast_worker(rank, pg):
    payload = np.arange(16, dtype=np.float64) + 5.0
    mine = payload.copy() if rank == 0 else np.zeros(16)
    return {"sum": float(broadcast(pg, mine, root=0).sum())}


def _rs_ag_worker(rank, pg, size: int):
    """reduce_scatter → all_gather 必须还原成完整归约。"""
    g = np.arange(size, dtype=np.float64) * (rank + 1)
    shard = reduce_scatter(pg, g)
    full = all_gather(pg, shard)
    return {"sum": float(full.sum())}


@pytest.mark.mp
@pytest.mark.parametrize("world_size", [2, 4])
def test_spawn_ring_allreduce(world_size: int) -> None:
    size = 512
    expected = sum(np.arange(size, dtype=np.float64) * (r + 1) for r in range(world_size)).sum()
    res = spawn_world(_ring_worker, world_size, size)
    for r in res:
        assert abs(r["sum"] - expected) < 1e-6
        optimal = 2 * (world_size - 1) * (size * 8 // world_size)
        assert r["bytes"] == optimal, f"跨进程通信量 {r['bytes']} != 理论下界 {optimal}"


@pytest.mark.mp
def test_spawn_broadcast() -> None:
    res = spawn_world(_bcast_worker, 3)
    expected = (np.arange(16, dtype=np.float64) + 5.0).sum()
    for r in res:
        assert abs(r["sum"] - expected) < 1e-9


@pytest.mark.mp
@pytest.mark.parametrize("world_size", [2, 4])
def test_spawn_reduce_scatter_allgather(world_size: int) -> None:
    size = 512
    expected = sum(np.arange(size, dtype=np.float64) * (r + 1) for r in range(world_size)).sum()
    res = spawn_world(_rs_ag_worker, world_size, size)
    for r in res:
        assert abs(r["sum"] - expected) < 1e-6
