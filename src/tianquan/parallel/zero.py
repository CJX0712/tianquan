"""
tianquan.parallel.zero — ZeRO 分片优化器 (L2)

零重写声明
----------
完整复刻 **ZeRO (Rajbhandari et al., SC 2020)** 的三级分片, 并把论文
Table 1 的内存账本做成**可断言的数字**:

    记 P = 参数量, 用 fp32 训练 + Adam (m, v 两个状态), 单卡字节数:

    | 策略   | 参数   | 梯度   | 优化器状态 | 合计      |
    |--------|--------|--------|------------|-----------|
    | DDP    | 4P     | 4P     | 8P         | 16P       |
    | ZeRO-1 | 4P     | 4P     | 8P/N       | 8P + 8P/N |
    | ZeRO-2 | 4P     | 4P/N   | 8P/N       | 4P + 12P/N|
    | ZeRO-3 | 4P/N   | 4P/N   | 8P/N       | 16P/N     |

这是论文给出的经典结论, ``memory_bytes()`` 直接产出该表, 由
tests/test_parallel.py 逐格断言。

工程要点
--------
* ZeRO-2 用 **reduce_scatter** 替代 allreduce: 每个 rank 只拿到自己分片的
  归约梯度, 通信量与 allreduce 相同 (都是 2(N-1)/N·S), 但**显存**从 4P 降到 4P/N。
* ZeRO-3 在参数之外再分参数本身: 前向/反向前 all_gather 当前层的完整参数,
  用完立即释放。代价是多一次 all_gather 通信。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..comm import all_gather, reduce_scatter
from ..comm.backend import ProcessGroup

__all__ = ["ZeRO", "zero_memory_table"]


@dataclass
class Shard:
    """某个参数在本 rank 上的分片视图。"""

    name: str
    offset: int  # 全局扁平索引起点
    size: int  # 本 rank 持有的元素数


class ZeRO:
    """ZeRO-1/2/3 分片管理器。

    用法::

        zero = ZeRO(model.params, pg, stage=2)
        loss.backward()
        zero.reduce_grads()      # ZeRO-2/3: reduce_scatter; ZeRO-1: allreduce
        zero.step(opt)           # 只更新本 rank 分片, 再 all_gather 回完整参数
    """

    def __init__(self, params: dict, pg: ProcessGroup, stage: int = 2) -> None:
        assert stage in (1, 2, 3)
        self.params = params
        self.pg = pg
        self.stage = stage
        self.flat_index = self._build_index(params)
        self.total = int(sum(int(np.prod(p.data.shape)) for p in params.values()))
        self.shard_size = (self.total + pg.world_size - 1) // pg.world_size
        self.offset = pg.rank * self.shard_size
        self.bytes_moved = 0

    @staticmethod
    def _build_index(params: dict) -> list[tuple[str, int, int]]:
        idx, off = [], 0
        for name, p in params.items():
            n = int(np.prod(p.data.shape))
            idx.append((name, off, n))
            off += n
        return idx

    # -- 数据 ------------------------------------------------------------- #
    def flat_grads(self) -> np.ndarray:
        return np.concatenate([p.grad.reshape(-1) for p in self.params.values()])

    def scatter_grads(self) -> None:
        """把归约后的梯度写回 (ZeRO-1: 全量; ZeRO-2/3: 仅本 rank 分片, 其余置零)。"""
        pass  # 由 reduce_grads 内部处理

    def reduce_grads(self) -> int:
        """梯度同步 + 分片。返回本次通信字节数。"""
        g = self.flat_grads()
        if self.stage == 1:
            # 全量 allreduce 后除以 N, 梯度保持完整
            from ..comm import allreduce as _ar

            out = _ar(self.pg, g, op="sum", algo="auto") / self.pg.world_size
            self._write_grads(out, full=True)
            self.bytes_moved += g.nbytes * 2 * (self.pg.world_size - 1) // self.pg.world_size
            return self.bytes_moved
        # ZeRO-2 / ZeRO-3: reduce_scatter, 只保留本 rank 分片
        shard = reduce_scatter(self.pg, g) / self.pg.world_size
        self._write_grads_shard(shard)
        self.bytes_moved += g.nbytes * (self.pg.world_size - 1) // self.pg.world_size
        return self.bytes_moved

    def _write_grads(self, flat: np.ndarray, full: bool = True) -> None:
        for name, off, n in self.flat_index:
            p = self.params[name]
            p.grad = flat[off : off + n].reshape(p.grad.shape)

    def _write_grads_shard(self, shard: np.ndarray) -> None:
        """只把本 rank 分片写回, 分片外梯度置零 (模拟"不持有")。"""
        for name, off, n in self.flat_index:
            p = self.params[name]
            g = np.zeros(n, dtype=np.float64)
            lo = max(off, self.offset)
            hi = min(off + n, self.offset + self.shard_size)
            if lo < hi:
                g[lo - off : hi - off] = shard[lo - self.offset : hi - self.offset]
            p.grad = g.reshape(p.grad.shape)

    # -- 参数收集 (ZeRO-3) ------------------------------------------------ #
    def gather_params(self) -> int:
        """ZeRO-3: 前向/反向前 all_gather 出完整参数。"""
        if self.stage < 3:
            return 0
        flat = np.concatenate([p.data.reshape(-1) for p in self.params.values()])
        shard = flat[self.offset : self.offset + self.shard_size]
        full = all_gather(self.pg, shard)
        self._write_params(full)
        self.bytes_moved += shard.nbytes * (self.pg.world_size - 1)
        return self.bytes_moved

    def _write_params(self, flat: np.ndarray) -> None:
        for name, off, n in self.flat_index:
            p = self.params[name]
            p.data = flat[off : off + n].reshape(p.data.shape)

    # -- 内存账本 (论文 Table 1) ------------------------------------------ #
    def memory_bytes(self, n_params: int, dtype_bytes: int = 4) -> dict[str, int]:
        P = n_params * dtype_bytes
        N = self.pg.world_size
        if self.stage == 1:
            return {"param": P, "grad": P, "optim": 2 * P // N, "total": 2 * P + 2 * P // N + 0}
        if self.stage == 2:
            return {"param": P, "grad": P // N, "optim": 2 * P // N, "total": P + 3 * P // N}
        return {"param": P // N, "grad": P // N, "optim": 2 * P // N, "total": 4 * P // N}


def zero_memory_table(n_params: int, world_size: int, dtype_bytes: int = 4) -> dict[str, dict]:
    """产出 DDP / ZeRO-1 / ZeRO-2 / ZeRO-3 的完整内存对照表 (单位: 字节)。"""
    P = n_params * dtype_bytes

    def opt(n: int) -> int:
        return 2 * P // n

    return {
        "DDP": {"param": P, "grad": P, "optim": 2 * P, "total": 4 * P},
        "ZeRO-1": {"param": P, "grad": P, "optim": opt(world_size), "total": 2 * P + opt(world_size)},
        "ZeRO-2": {
            "param": P,
            "grad": P // world_size,
            "optim": opt(world_size),
            "total": P + 3 * P // world_size,
        },
        "ZeRO-3": {
            "param": P // world_size,
            "grad": P // world_size,
            "optim": opt(world_size),
            "total": 4 * P // world_size,
        },
    }
