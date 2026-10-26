"""
tianquan.serve.kv_cache — PagedAttention 式分页 KV 缓存

零重写声明
----------
复刻 **vLLM 的 PagedAttention (Kwon et al., SOSP 2023)** 的核心机制:

* 把 KV 缓存切成固定大小的 **block**, 用一张 block table 做逻辑→物理映射,
  从而消灭"按 max_seq_len 预分配"造成的内部碎片与外部碎片。
* **写时复制 (copy-on-write)**: 共享同一前缀的序列 (beam search / 并行采样 /
  系统提示) 共享物理块, 只在写入分叉时复制 —— 这正是 vLLM 把 beam search
  内存降到 1/beam_width 的手段。

可测指标
--------
* 块利用率 = 已用 slot / 已分配 slot  (朴素预分配的浪费率可量化对比)
* 共享率 = 因前缀共享而省下的块数
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BlockAllocator", "PagedKVCache", "NaiveKVCache"]


@dataclass
class SeqBlocks:
    seq_id: int
    blocks: list[int] = field(default_factory=list)
    length: int = 0


class BlockAllocator:
    """物理块池: 分配 / 释放 / 引用计数 / 写时复制。"""

    def __init__(self, n_blocks: int, block_size: int) -> None:
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.free: list[int] = list(range(n_blocks))
        self.refcount: dict[int, int] = {}

    @property
    def n_free(self) -> int:
        return len(self.free)

    @property
    def n_used(self) -> int:
        return self.n_blocks - len(self.free)

    def alloc(self) -> int:
        if not self.free:
            raise MemoryError("KV 块池耗尽")
        b = self.free.pop()
        self.refcount[b] = 1
        return b

    def free_block(self, b: int) -> None:
        self.refcount[b] = self.refcount.get(b, 1) - 1
        if self.refcount[b] <= 0:
            self.refcount.pop(b, None)
            self.free.append(b)

    def share(self, b: int) -> None:
        self.refcount[b] = self.refcount.get(b, 0) + 1

    def usage(self) -> float:
        return self.n_used / self.n_blocks


class PagedKVCache:
    """分页 KV 缓存。接口: put / get / fork(共享) / release。"""

    def __init__(
        self,
        n_blocks: int,
        block_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.allocator = BlockAllocator(n_blocks, block_size)
        self.block_size = block_size
        self.n_layers = n_layers
        shape = (n_blocks, block_size, n_kv_heads, head_dim)
        self.k = [np.zeros(shape) for _ in range(n_layers)]
        self.v = [np.zeros(shape) for _ in range(n_layers)]
        self.seqs: dict[int, SeqBlocks] = {}
        self.cow_copies = 0  # 写时复制次数 (共享率的反向指标)
        self.shared_blocks = 0  # 通过共享省下的块数

    # -- 序列管理 --------------------------------------------------------- #
    def create(self, seq_id: int) -> None:
        self.seqs[seq_id] = SeqBlocks(seq_id)

    def _ensure(self, seq_id: int, need_len: int) -> None:
        s = self.seqs[seq_id]
        while len(s.blocks) * self.block_size < need_len:
            s.blocks.append(self.allocator.alloc())

    def _loc(self, seq_id: int, pos: int) -> tuple[int, int]:
        s = self.seqs[seq_id]
        return s.blocks[pos // self.block_size], pos % self.block_size

    # -- 读写 ------------------------------------------------------------- #
    def put(self, seq_id: int, layer: int, pos: int, k: np.ndarray, v: np.ndarray) -> None:
        s = self.seqs[seq_id]
        if pos >= len(s.blocks) * self.block_size:
            self._ensure(seq_id, pos + 1)
        b, off = self._loc(seq_id, pos)
        if self.allocator.refcount.get(b, 1) > 1:  # 共享块被写 → 写时复制
            nb = self.allocator.alloc()
            self.k[layer][nb] = self.k[layer][b]
            self.v[layer][nb] = self.v[layer][b]
            self.allocator.free_block(b)
            s.blocks[pos // self.block_size] = nb
            b, off = nb, off
            self.cow_copies += 1
        self.k[layer][b, off] = k
        self.v[layer][b, off] = v
        s.length = max(s.length, pos + 1)

    def get(self, seq_id: int, layer: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """重组出该序列前 length 个位置的 K/V, 形状 (length, n_kv, hd)。"""
        s = self.seqs[seq_id]
        assert length <= s.length, f"请求 {length} > 已缓存 {s.length}"
        nb = (length + self.block_size - 1) // self.block_size
        blocks = s.blocks[:nb]
        kk = np.concatenate([self.k[layer][b] for b in blocks], axis=0)[:length]
        vv = np.concatenate([self.v[layer][b] for b in blocks], axis=0)[:length]
        return kk, vv

    # -- 共享 / 释放 ------------------------------------------------------ #
    def fork(self, src_id: int, dst_id: int) -> None:
        """dst 共享 src 的全部块 (写时复制保证后续分叉不互相污染)。"""
        src, dst = self.seqs[src_id], self.seqs[dst_id]
        dst.blocks = list(src.blocks)
        dst.length = src.length
        n_shared = src.length // self.block_size
        for b in dst.blocks[:n_shared]:
            self.allocator.share(b)
        self.shared_blocks += n_shared

    def release(self, seq_id: int) -> None:
        s = self.seqs.pop(seq_id, None)
        if s is None:
            return
        for b in s.blocks:
            self.allocator.free_block(b)

    # -- 指标 ------------------------------------------------------------- #
    def stats(self) -> dict:
        used_slots = 0
        for b in range(self.n_blocks_()):
            if self.allocator.refcount.get(b, 0) > 0:
                used_slots += self.block_size
        total_slots = sum(len(s.blocks) for s in self.seqs.values()) * self.block_size
        real_tokens = sum(s.length for s in self.seqs.values())
        return {
            "n_blocks_used": self.allocator.n_used,
            "block_utilization": real_tokens / max(used_slots, 1),
            "logical_slot_utilization": real_tokens / max(total_slots, 1),
            "cow_copies": self.cow_copies,
            "shared_blocks": self.shared_blocks,
        }

    def n_blocks_(self) -> int:
        return self.allocator.n_blocks


class NaiveKVCache:
    """对照基线: 按 max_seq_len 预分配连续显存 —— 用来量化 PagedAttention 的收益。"""

    def __init__(self, max_seqs: int, max_len: int, n_layers: int, n_kv: int, hd: int) -> None:
        self.max_len = max_len
        self.n_layers = n_layers
        self.k = [np.zeros((max_seqs, max_len, n_kv, hd)) for _ in range(n_layers)]
        self.v = [np.zeros((max_seqs, max_len, n_kv, hd)) for _ in range(n_layers)]
        self.lens: dict[int, int] = {}
        self.next_slot = 0

    def create(self, seq_id: int) -> None:
        self.lens[seq_id] = 0
        self.slot_of = getattr(self, "slot_of", {})
        self.slot_of[seq_id] = self.next_slot
        self.next_slot += 1

    def put(self, seq_id: int, layer: int, pos: int, k: np.ndarray, v: np.ndarray) -> None:
        s = self.slot_of[seq_id]
        self.k[layer][s, pos] = k
        self.v[layer][s, pos] = v
        self.lens[seq_id] = max(self.lens[seq_id], pos + 1)

    def get(self, seq_id: int, layer: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.slot_of[seq_id]
        return self.k[layer][s, :length], self.v[layer][s, :length]

    def release(self, seq_id: int) -> None:
        self.lens.pop(seq_id, None)

    def stats(self) -> dict:
        real = sum(self.lens.values())
        allocated = max(self.next_slot, 1) * self.max_len
        return {
            "allocated_slots": allocated,
            "used_slots": real,
            "utilization": real / max(allocated, 1),
        }
