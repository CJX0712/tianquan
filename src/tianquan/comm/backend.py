"""
tianquan.comm.backend — 通信后端 (L1)

设计原则
--------
集合通信算法 (collectives.py) **只依赖** 三个原语: send / recv / barrier。
后端负责提供这三个原语的**真实**实现, 从而把"算法正确性"与"传输介质"解耦:

* ``ThreadGroup``        — 同进程多线程。零 spawn 开销, 用于单测与计算-通信重叠实验。
* ``SharedMemoryGroup``  — 跨进程 (spawn), 通过 ``multiprocessing.shared_memory``
                           真实搬运字节。用于真实的多进程并行基准。

两种后端共享同一套 ``bytes_moved`` 计费口径, 因此可以**实测并断言**
Ring AllReduce 的带宽最优性 (Patarasuk & Yuan 2009 下界)。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from multiprocessing import shared_memory

import numpy as np

__all__ = ["ProcessGroup", "ThreadGroup", "SharedMemoryGroup", "Counter"]


class Counter:
    """线程/进程安全的字节计数器 —— 通信量最优性的度量尺。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._n = 0

    def add(self, n: int) -> None:
        with self._lock:
            self._n += int(n)

    @property
    def value(self) -> int:
        with self._lock:
            return self._n

    def reset(self) -> None:
        with self._lock:
            self._n = 0


class ProcessGroup(ABC):
    """通信域抽象。所有并行策略都面向此接口编程, 不绑定具体传输介质。"""

    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.bytes_moved = Counter()
        self.n_steps = 0  # 通信步数 (延迟代理指标)

    # -- 点对点 ----------------------------------------------------------- #
    # 统一语义: **发布-订阅**。
    #   send(buf)      —— 把 buf 发布到**本 rank 自己的输出槽位** (每步只能发一块)
    #   recv(src)      —— 读取 src rank 的槽位
    # 该语义对环 (读 rank-1)、双向交换 (读 peer)、配对 all-to-all (读 rank+step)
    # 全部成立, 且每个槽位每步恰好一写一读, 无冲突。
    @abstractmethod
    def send(self, buf: np.ndarray, dst: int | None = None) -> None: ...

    @abstractmethod
    def recv(self, src: int, shape: tuple[int, ...] | None = None, dtype: np.dtype | None = None) -> np.ndarray: ...

    @abstractmethod
    def barrier(self) -> None: ...

    def exchange(
        self,
        send_buf: np.ndarray | None,
        src: int | None,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype = np.dtype(np.float64),
    ) -> np.ndarray | None:
        """一次"交换"= 可选发送 → 屏障 → 可选接收 → 屏障。

        这是所有集合通信算法的**唯一同步原语**: 双屏障保证
        (a) 接收方读到的一定是本步新写入的数据 (不会读到上一步的陈旧值);
        (b) 下一步的发送不会覆盖本步尚未读完的槽位。

        返回接收到的数据; 若 ``src is None`` 返回 None。
        """
        if send_buf is not None:
            self.send(np.asarray(send_buf, dtype=np.float64))
        self.barrier()
        got = self.recv(src, shape, dtype) if src is not None else None  # type: ignore[arg-type]
        self.barrier()
        return got

    # -- 便捷 ------------------------------------------------------------- #
    @property
    def is_single(self) -> bool:
        return self.world_size == 1

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rank={self.rank}/{self.world_size})"


# --------------------------------------------------------------------------- #
# 线程后端
# --------------------------------------------------------------------------- #
class ThreadGroup(ProcessGroup):
    """单进程多线程后端。每个 rank 拥有一个"邮箱", send 写入、recv 读取。

    同步使用 ``threading.Barrier``: 每个通信步之后全体对齐, 从而让
    Ring AllReduce 的 2(N-1) 步语义与真实分布式完全一致。
    """

    def __init__(self, rank: int, world_size: int, barrier: threading.Barrier, slots: list) -> None:
        super().__init__(rank, world_size)
        self._barrier = barrier
        self._slots = slots  # list[np.ndarray | None], 每 rank 一个输出槽位

    @classmethod
    def create(cls, world_size: int) -> list["ThreadGroup"]:
        barrier = threading.Barrier(world_size)
        slots: list = [None] * world_size
        return [cls(r, world_size, barrier, slots) for r in range(world_size)]

    def send(self, buf: np.ndarray, dst: int | None = None) -> None:
        buf = np.ascontiguousarray(buf)
        self._slots[self.rank] = buf.copy()
        self.bytes_moved.add(buf.nbytes)
        self.n_steps += 1

    def recv(
        self, src: int, shape: tuple[int, ...] | None = None, dtype: np.dtype | None = None
    ) -> np.ndarray:
        """读取 ``src`` rank 的输出槽位 (发布-订阅语义)。"""
        v = self._slots[src]
        if v is None:  # pragma: no cover
            raise RuntimeError(f"rank {self.rank} 读取 rank {src} 的槽位为空")
        return v

    def barrier(self) -> None:
        self._barrier.wait()


# --------------------------------------------------------------------------- #
# 共享内存后端 (跨进程)
# --------------------------------------------------------------------------- #
class SharedMemoryGroup(ProcessGroup):
    """跨进程后端 (spawn 兼容)。

    每个 rank 拥有一个以自身 rank 命名的输出槽位 (SharedMemory block)。
    ``send`` 把数据写入本 rank 的槽位, ``recv(src)`` 从 src 的槽位读取。
    ``barrier`` 使用 multiprocessing.Barrier, 保证 ring 的分步语义。
    """

    SLOT_BYTES = 64 * 1024 * 1024  # 单槽位容量上限 64 MiB

    def __init__(
        self,
        rank: int,
        world_size: int,
        barrier,
        slot_names: list[str],
        header_locks: list,
    ) -> None:
        super().__init__(rank, world_size)
        self._barrier = barrier
        self._slot_names = slot_names
        self._locks = header_locks
        self._shm = {s: shared_memory.SharedMemory(name=s) for s in slot_names}

    @classmethod
    def create(cls, world_size: int, ctx) -> tuple[list[str], object]:
        """在**父进程**中创建槽位, 返回 (slot_names, barrier) 以便传给子进程。"""
        slots = []
        for _ in range(world_size):
            shm = shared_memory.SharedMemory(create=True, size=cls.SLOT_BYTES)
            shm.buf[:8] = np.zeros(8, dtype=np.uint8)  # 触页
            slots.append(shm)
        names = [s.name for s in slots]
        barrier = ctx.Barrier(world_size)
        return names, barrier, slots

    @staticmethod
    def build(rank: int, world_size: int, barrier, slot_names: list[str]) -> "SharedMemoryGroup":
        locks = [None] * world_size
        return SharedMemoryGroup(rank, world_size, barrier, slot_names, locks)

    def send(self, buf: np.ndarray, dst: int | None = None) -> None:
        buf = np.ascontiguousarray(buf)
        raw = buf.tobytes()
        n = len(raw)
        shm = self._shm[self._slot_names[self.rank]]
        shm.buf[0:8] = np.array([n], dtype=np.int64).tobytes()
        shm.buf[8 : 8 + n] = raw
        self.bytes_moved.add(n)
        self.n_steps += 1

    def recv(
        self, src: int, shape: tuple[int, ...] | None = None, dtype: np.dtype | None = None
    ) -> np.ndarray:
        shm = self._shm[self._slot_names[src]]
        n = int(np.frombuffer(shm.buf[0:8], dtype=np.int64, count=1)[0])
        raw = bytes(shm.buf[8 : 8 + n])
        dtype = dtype or np.dtype(np.float64)
        # np.frombuffer 返回**只读**视图; 必须 copy, 否则调用方一旦原地修改
        # (如 DDP 的 ``out /= world_size``) 就会抛 "output array is read-only"。
        arr = np.frombuffer(raw, dtype=dtype)
        return arr.reshape(shape).copy() if shape else arr.copy()

    def barrier(self) -> None:
        self._barrier.wait()

    def close(self) -> None:
        for s in self._shm.values():
            try:
                s.close()
            except Exception:  # pragma: no cover
                pass
