"""
tianquan.comm.collectives — 集合通信算法 (L1)

零重写声明
----------
所有算法均为已发表的**最优/工业标准**方案, 本项目只做最小可信实现并给出
可断言的最优性证据:

* ``ring_allreduce``   — Patarasuk & Yuan 2009 的带宽最优环算法;
                         经 Horovod (Sergeev & Del Balso 2018) 工程化。
                         每 rank 通信量恰为 ``2(N-1)/N * S``, 达到理论下界。
* ``tree_allreduce``   — 递归折半 (binary tree), 延迟 ``O(log N)``,
                         但通信量为 ``O(S log N)`` 量级 —— 小消息更优。
* ``reduce_scatter`` / ``all_gather`` — ZeRO 的分片原语 (Rajbhandari et al. 2020)
* ``all_to_all``       — 专家并行 (MoE All-to-All) 与张量并行转置的基础
* ``broadcast``        — 二树广播

硬不变量 (tests/test_comm.py 独立验证)
--------------------------------------
1. 正确性: 所有 rank 上的结果与 NumPy 全局归约**逐位一致** (float64)。
2. 最优性: ring_allreduce 每 rank 发送字节数 == ``ceil(2*(N-1)/N * S)``。
3. 一致性: ring 与 tree 两种算法结果相同。
"""

from __future__ import annotations

import numpy as np

from .backend import ProcessGroup

__all__ = [
    "ring_allreduce",
    "tree_allreduce",
    "allreduce",
    "reduce_scatter",
    "all_gather",
    "all_to_all",
    "broadcast",
]


# --------------------------------------------------------------------------- #
# AllReduce
# --------------------------------------------------------------------------- #
def ring_allreduce(pg: ProcessGroup, buf: np.ndarray, op: str = "sum") -> np.ndarray:
    """带宽最优环 AllReduce。

    两阶段共 ``2(N-1)`` 步, 每步搬运 ``S/N``:
      * 阶段一 scatter-reduce: N-1 步后每个 rank 持有 1/N 的完整归约结果
      * 阶段二 all-gather:    N-1 步后每个 rank 持有完整结果

    不变量: bytes_moved == 2*(N-1)/N * S  (Patarasuk & Yuan 下界, 带宽最优)
    """
    buf = np.asarray(buf, dtype=np.float64)
    n = pg.world_size
    if n == 1:
        return buf.copy()

    size = buf.size
    # 不均分时用 padding 到 n 的整数倍, 保证每步搬运量严格相等
    pad = (-size) % n
    flat = np.concatenate([buf.ravel(), np.zeros(pad, dtype=np.float64)]) if pad else buf.ravel().copy()
    chunk = flat.size // n
    chunks = [flat[i * chunk : (i + 1) * chunk].copy() for i in range(n)]

    recv_from = (pg.rank - 1) % n
    shape = (chunk,)

    # ---- 阶段一: scatter-reduce (N-1 步) ---- #
    for step in range(n - 1):
        send_idx = (pg.rank - step) % n
        recv_idx = (pg.rank - step - 1) % n
        got = pg.exchange(chunks[send_idx], recv_from, shape)
        chunks[recv_idx] = _apply(chunks[recv_idx], got, op)

    # ---- 阶段二: all-gather (N-1 步) ---- #
    for step in range(n - 1):
        send_idx = (pg.rank - step + 1) % n
        recv_idx = (pg.rank - step) % n
        chunks[recv_idx] = pg.exchange(chunks[send_idx], recv_from, shape)

    out = np.concatenate(chunks)
    return out[:size].reshape(buf.shape)


def tree_allreduce(pg: ProcessGroup, buf: np.ndarray, op: str = "sum") -> np.ndarray:
    """递归折半 AllReduce: reduce 到 rank0, 再 broadcast。延迟 O(log N)。

    适用: 小消息 / 高延迟网络。通信量高于 ring, 但步数只有 2*ceil(log2 N)。
    """
    buf = np.asarray(buf, dtype=np.float64)
    n = pg.world_size
    if n == 1:
        return buf.copy()

    # 阶段一: 折半归约到 root (二叉树自底向上, 支持任意 N)
    acc = _tree_reduce(pg, buf, root=0, op=op)
    # 阶段二: 从 root 广播 (binomial tree, 支持任意 N)
    return broadcast(pg, acc, root=0)


def _tree_reduce(pg: ProcessGroup, buf: np.ndarray, root: int, op: str) -> np.ndarray:
    """二叉归约: 相对编号 r 满足 r % (2*step)==0 的节点收集 r+step, 其余节点上报后退出。

    支持任意 world_size (非 2 的幂也正确), 步数 O(log N), 每 rank 通信量 <= S。
    """
    n = pg.world_size
    r = (pg.rank - root) % n
    acc = np.asarray(buf, dtype=np.float64).copy()
    shape = acc.shape
    active = True
    step = 1
    while step < n:
        # 关键: 已上报的节点**不能退出循环** —— 它必须继续参与每一步屏障,
        # 否则仍在归约树上的节点会永久等待 (经典分布式死锁陷阱)。
        if active and r % (2 * step) == 0:
            src = r + step
            if src < n:
                got = pg.exchange(None, (src + root) % n, shape)
                acc = _apply(acc, got, op)
            else:
                pg.exchange(None, None, shape)
        elif active:
            pg.exchange(acc, None, shape)
            active = False
        else:
            pg.exchange(None, None, shape)
        step <<= 1
    return acc


def allreduce(pg: ProcessGroup, buf: np.ndarray, op: str = "sum", algo: str = "auto") -> np.ndarray:
    """自动选算法的 allreduce。

    选择依据 (与 Horovod/NCCL 的启发式一致):
      * ``auto``  —— 大消息用 ring (带宽最优), 小消息用 tree (延迟最优)
      * 阈值: 单 rank 消息 >= 256 KiB 走 ring
    """
    if algo == "auto":
        algo = "ring" if buf.nbytes >= 256 * 1024 else "tree"
    if algo == "ring":
        return ring_allreduce(pg, buf, op)
    if algo == "tree":
        return tree_allreduce(pg, buf, op)
    raise ValueError(f"未知算法: {algo}")


# --------------------------------------------------------------------------- #
# 分片原语 (ZeRO 基石)
# --------------------------------------------------------------------------- #
def reduce_scatter(pg: ProcessGroup, buf: np.ndarray, op: str = "sum") -> np.ndarray:
    """Reduce-Scatter: 归约后 **rank i 得到第 i 块** (标准 MPI 语义)。

    复用 ring 的阶段一 —— 通信量 ``(N-1)/N * S``, 同样达到下界。

    *索引推导*: 若沿用 ring_allreduce 的 ``(rank-step)`` 偏移, N-1 步后
    rank i 拿到的是 ``chunk[(i+1)%N]``; 这里把收发索引整体偏移 -1
    (``send=(rank-step-1)``, ``recv=(rank-step-2)``), 使 chunk k 的累积终点
    恰好落回 rank k, 从而得到标准语义, 可直接被 ZeRO 的 ``offset=rank*shard``
    消费。与 ``all_gather`` 组合恒等于 ``allreduce``。
    """
    buf = np.asarray(buf, dtype=np.float64)
    n = pg.world_size
    if n == 1:
        return buf.copy()

    size = buf.size
    pad = (-size) % n
    flat = np.concatenate([buf.ravel(), np.zeros(pad)]) if pad else buf.ravel().copy()
    chunk = flat.size // n
    chunks = [flat[i * chunk : (i + 1) * chunk].copy() for i in range(n)]

    recv_from = (pg.rank - 1) % n
    for step in range(n - 1):
        send_idx = (pg.rank - step - 1) % n
        recv_idx = (pg.rank - step - 2) % n
        got = pg.exchange(chunks[send_idx], recv_from, (chunk,))
        chunks[recv_idx] = _apply(chunks[recv_idx], got, op)

    return chunks[pg.rank].copy()  # 标准语义: 本 rank 得到第 rank 块


def all_gather(pg: ProcessGroup, shard: np.ndarray) -> np.ndarray:
    """All-Gather: 把各自 1/N 分片拼成完整张量。通信量 (N-1)/N * S。"""
    shard = np.asarray(shard, dtype=np.float64)
    n = pg.world_size
    if n == 1:
        return shard.copy()

    chunk = shard.size
    chunks: list[np.ndarray | None] = [None] * n
    chunks[pg.rank] = shard.copy()

    recv_from = (pg.rank - 1) % n
    # 标准语义: rank i 提供第 i 块, 输出按 0..N-1 顺序拼接。
    for step in range(n - 1):
        send_idx = (pg.rank - step) % n
        recv_idx = (pg.rank - step - 1) % n
        chunks[recv_idx] = pg.exchange(chunks[send_idx], recv_from, (chunk,))

    return np.concatenate([c for c in chunks])  # type: ignore[arg-type]


def all_to_all(pg: ProcessGroup, chunks: list[np.ndarray]) -> list[np.ndarray]:
    """All-to-All: rank i 的第 j 块发给 rank j —— MoE 专家并行的核心原语。

    实现: **直接配对交换** (发布-订阅)。第 step 步:
      * rank r **读** peer=(r+step)%N 的槽位, 存入 ``out[peer]``;
      * 反过来, 会读 rank r 槽位的是 ``(r-step)%N``, 故 rank r 应**发布**
        ``chunks[(r-step) % N]`` —— 即目的地恰为读者的那一块, 一步到位。
    每个槽位每步恰好一写一读, 无冲突。

    * 通信量: (N-1)/N * S —— 与 ring 版相同, 达到下界
    * 通信步数: N-1 (配对, 无需逐跳转发)
    * 代价: 要求逻辑全连接 (每步与不同 peer 直接通信); ring 版可降为
      纯邻居通信但需转发调度, 本项目选用语义无歧义的配对版。
    """
    n = pg.world_size
    assert len(chunks) == n, "chunks 数量必须等于 world_size"
    if n == 1:
        return [np.asarray(c, dtype=np.float64).copy() for c in chunks]

    out: list[np.ndarray | None] = [None] * n
    out[pg.rank] = np.asarray(chunks[pg.rank], dtype=np.float64).copy()
    shape = np.asarray(chunks[0], dtype=np.float64).shape

    for step in range(1, n):
        peer = (pg.rank + step) % n
        send_idx = (pg.rank - step) % n  # 目的地 = 将要读我槽位的 rank
        got = pg.exchange(np.asarray(chunks[send_idx], dtype=np.float64), peer, shape)
        out[peer] = got
    return [o for o in out]  # type: ignore[return-value]


def broadcast(pg: ProcessGroup, buf: np.ndarray, root: int = 0) -> np.ndarray:
    """Binomial-tree 广播 (MPICH `MPIR_Bcast_intra_binomial` 语义)。

    第 k 轮 (mask=2^k): 相对编号 ``r < mask`` 的进程已持有数据, 发给 ``r+mask``;
    ``mask <= r < 2*mask`` 的进程从 ``r-mask`` 接收。

    不变量: 通信步数 = ceil(log2 N); 每个 rank 发送量 <= S; 结果逐位等于 root 的输入。
    """
    buf = np.asarray(buf, dtype=np.float64)
    n = pg.world_size
    if n == 1:
        return buf.copy()

    r = (pg.rank - root) % n
    out = buf.copy()
    shape = buf.shape
    mask = 1
    while mask < n:
        if r < mask:
            dst = r + mask
            if dst < n:
                pg.exchange(out, None, shape)
            else:
                pg.exchange(None, None, shape)
        elif r < 2 * mask:
            out = pg.exchange(None, (r - mask + root) % n, shape)
        else:
            pg.exchange(None, None, shape)
        mask <<= 1
    return out


# --------------------------------------------------------------------------- #
def _apply(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    if op == "sum":
        return a + b
    if op == "max":
        return np.maximum(a, b)
    if op == "min":
        return np.minimum(a, b)
    if op == "mean":
        return a + b
    raise ValueError(f"未知算子: {op}")


def _mean_div(pg: ProcessGroup, x: np.ndarray) -> np.ndarray:
    return x / pg.world_size
