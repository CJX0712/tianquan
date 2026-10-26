"""
多进程基准的 worker 函数。

**必须位于模块顶层且可通过 pickle**: Windows/macOS 的 spawn 启动方式
会把函数序列化到子进程, 嵌套函数或 lambda 会失败。
"""

from __future__ import annotations

import time

import numpy as np

from tianquan.comm import allreduce, ring_allreduce, tree_allreduce
from tianquan.parallel import DDP
from tianquan.train import GPT, AdamW, GPTConfig
from tianquan.train.trainer import make_synthetic_batches, train_step


def ddp_train_worker(rank, pg, cfg_dict, n_steps, batch_per_rank, seq_len, lr, bucket_bytes=None):
    # bucket_bytes=None ⇒ 交给 DDP 的默认策略 (4 MiB, 实测最优)
    """真实的多进程 DDP 训练: 每个 rank 吃自己的数据分片, 梯度 AllReduce 后取平均。

    ``bucket_bytes`` 是 DDP 最关键的调优旋钮: 桶越小 ⇒ allreduce 次数越多 ⇒
    屏障开销主导; 桶越大 ⇒ 次数少但丧失与反向计算重叠的机会。
    """
    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg)  # 同 seed ⇒ 各 rank 初始参数一致
    opt = AdamW(model.params, lr=lr)
    ddp = (
        DDP(model.params, pg)
        if bucket_bytes is None
        else DDP(model.params, pg, bucket_bytes=bucket_bytes)
    )
    idx, tgt = make_synthetic_batches(cfg.vocab_size, seq_len, n_steps, batch_per_rank, seed=rank)

    def sync():
        ddp.mark_all_ready()
        return ddp.sync()

    losses = []
    t0 = time.perf_counter()
    for s in range(n_steps):
        losses.append(train_step(model, opt, idx[s], tgt[s], sync=sync))
    wall = time.perf_counter() - t0

    tokens = n_steps * batch_per_rank * seq_len
    return {
        "rank": rank,
        "wall_s": round(wall, 4),
        "tokens_per_s": round(tokens / max(wall, 1e-9), 2),
        "final_loss": round(float(losses[-1]), 5),
        "first_loss": round(float(losses[0]), 5),
        "comm_bytes": pg.bytes_moved.value,
        "comm_steps": pg.n_steps,
    }


def allreduce_worker(rank, pg, size, algo):
    """测量一次 AllReduce 的墙钟与实测通信量。"""
    x = np.random.default_rng(rank).standard_normal(size)
    fn = ring_allreduce if algo == "ring" else tree_allreduce
    # 计时只覆盖集合通信本身
    pg.barrier()
    t0 = time.perf_counter()
    out = fn(pg, x)
    wall = time.perf_counter() - t0
    return {
        "rank": rank,
        "wall_s": wall,
        "bytes": pg.bytes_moved.value,
        "checksum": float(out.sum()),
    }


def tp_worker(rank, pg, cfg_dict, seq_len, batch_size, n_repeat):
    """张量并行前向耗时 (每 rank 持 1/N 权重, 每层 2 次 AllReduce)。"""
    from tianquan.parallel.tensor import TensorParallelGPT

    cfg = GPTConfig(**cfg_dict)
    model = TensorParallelGPT(cfg, pg, seed=cfg.seed)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, cfg.vocab_size, size=(batch_size, seq_len))
    tgt = rng.integers(0, cfg.vocab_size, size=(batch_size, seq_len))

    model.forward(idx, tgt)  # warmup
    pg.bytes_moved.reset()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        _, loss = model.forward(idx, tgt)
        loss.backward()
    wall = time.perf_counter() - t0
    return {
        "rank": rank,
        "wall_s": round(wall / n_repeat, 6),
        "comm_bytes": pg.bytes_moved.value // n_repeat,
        "loss": round(float(loss.data), 6),
    }


def zero_worker(rank, pg, cfg_dict, stage, seq_len, batch_per_rank):
    """ZeRO: 测量梯度同步的通信量与分片后的内存占用。"""
    from tianquan.parallel import ZeRO

    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg)
    opt = AdamW(model.params, lr=1e-3)
    z = ZeRO(model.params, pg, stage=stage)
    rng = np.random.default_rng(rank)
    idx = rng.integers(0, cfg.vocab_size, size=(batch_per_rank, seq_len + 1))
    x, y = idx[:, :-1], idx[:, 1:]

    model.zero_grad()
    model.forward(x, y)[1].backward()
    pg.bytes_moved.reset()
    t0 = time.perf_counter()
    z.reduce_grads()
    opt.step()
    wall = time.perf_counter() - t0

    n_params = model.n_params()
    return {
        "rank": rank,
        "wall_s": round(wall, 6),
        "comm_bytes": pg.bytes_moved.value,
        "memory": z.memory_bytes(n_params),
        "n_params": n_params,
    }
