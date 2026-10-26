"""
tianquan.train.trainer — 训练循环与合成数据 (L3)

刻意保持"薄": 训练循环本身不是本仓的创新点, 它的职责只是把
模型 / 优化器 / 并行策略接起来, 让基准可以稳定地测到**通信与并行**的开销。
"""

from __future__ import annotations

import numpy as np

from .model import GPT


__all__ = ["make_synthetic_batches", "train_step", "evaluate"]


def make_synthetic_batches(
    vocab_size: int, seq_len: int, n_batch: int, batch_size: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """确定性合成序列: 一段随机游走的 token, 供端到端训练/基准使用。"""
    rng = np.random.default_rng(seed)
    total = n_batch * batch_size * (seq_len + 1)
    walk = rng.integers(0, vocab_size, size=total)
    data = walk.reshape(n_batch * batch_size, seq_len + 1)
    return data[:, :-1].reshape(n_batch, batch_size, seq_len), data[:, 1:].reshape(
        n_batch, batch_size, seq_len
    )


def train_step(model: GPT, opt, idx: np.ndarray, tgt: np.ndarray, sync=None) -> float:
    """一步训练。``sync`` 为 DDP/ZeRO 的梯度同步回调。"""
    model.zero_grad()
    _, loss = model.forward(idx, tgt)
    loss.backward()
    if sync is not None:
        sync()
    opt.step()
    return float(loss.data)


def evaluate(model: GPT, idx: np.ndarray, tgt: np.ndarray) -> float:
    _, loss = model.forward(idx, tgt)
    return float(loss.data)
