"""
天权 TIANQUAN — 分布式训练 x 高效推理服务统一内核
=================================================

作者: 晨星 (Chenxing)  |  License: Apache-2.0

分层
----
    L0 core.numerics      数值稳定原语 (logsumexp/RoPE/RMSNorm...)
    L1 comm               通信后端 + 带宽最优集合通信 (ring/tree/ZeRO 原语)
    L2 parallel           DDP / ZeRO-1,2,3 / 张量并行 / 流水并行 (GPipe, 1F1B)
    L3 train              小 GPT (RMSNorm+RoPE+GQA+SwiGLU) + AdamW/Muon + Trainer
    L4 serve              PagedAttention / 连续批处理 / Radix 前缀缓存 / 投机解码
    L5 eval               确定性基准 harness 与报告

设计信条
--------
1. **零重写**: 不发明算法, 只把世界顶级成果做成最小可信实现, 并给出可断言的证据。
2. **可验证**: 每个模块都配一条"另一条路径"的硬不变量, 而不是靠肉眼。
3. **可复现**: 纯 NumPy + CPU, 固定种子, 无网络、无 GPU 也能全绿。
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "晨星 (Chenxing)"

from . import comm, core  # noqa: F401

__all__ = ["core", "comm", "__version__", "__author__"]
