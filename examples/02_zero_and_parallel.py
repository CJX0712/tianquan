"""
示例 2: ZeRO 内存账本 + 张量并行 + 流水并行

运行:  python examples/02_zero_and_parallel.py

展示三件事
----------
1. ZeRO-1/2/3 相对 DDP 的单卡内存下降 (Rajbhandari et al. 2020 Table 1)
2. Megatron 式张量并行: 切分后的输出与未切分模型逐位一致
3. GPipe vs 1F1B: 峰值激活与 bubble 的实测对比
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tianquan.comm.launch import run_threads  # noqa: E402
from tianquan.parallel import (  # noqa: E402
    PipelineConfig,
    TensorParallelGPT,
    simulate,
    zero_memory_table,
)
from tianquan.train import GPT, GPTConfig  # noqa: E402


def main() -> int:
    # ---- 1. ZeRO 内存 ---- #
    print("== ZeRO 单卡内存 (1M 参数, fp32, Adam, N=8) ==")
    t = zero_memory_table(1_000_000, 8)
    base = t["DDP"]["total"]
    for k, v in t.items():
        print(f"  {k:8s} 参数 {v['param']/1e6:6.2f}MB  梯度 {v['grad']/1e6:6.2f}MB  "
              f"优化器 {v['optim']/1e6:6.2f}MB  合计 {v['total']/1e6:6.2f}MB  "
              f"({v['total']/base*100:5.1f}% of DDP)")

    # ---- 2. 张量并行等价性 ---- #
    print("\n== 张量并行 (Megatron 切分) 等价性 ==")
    cfg = GPTConfig(vocab_size=16, d_model=32, n_heads=4, n_kv_heads=4, n_layers=2, d_ff=64, seed=13)
    rng = np.random.default_rng(9)
    idx, tgt = rng.integers(0, 16, size=(2, 6)), rng.integers(0, 16, size=(2, 6))
    ref = GPT(cfg).forward(idx, tgt)

    def tp_fn(rank, pg):
        lg, ls = TensorParallelGPT(cfg, pg, seed=13).forward(idx, tgt)
        return lg.data, float(ls.data)

    for n in (2, 4):
        outs = run_threads(tp_fn, n)
        err = max(np.abs(o[0] - ref[0].data).max() for o in outs)
        print(f"  TP={n}: 与单进程最大偏差 {err:.3e}  ({'✅ 一致' if err < 1e-10 else '❌'})")

    # ---- 3. 流水并行 ---- #
    print("\n== 流水并行: GPipe vs 1F1B (p=8 stages) ==")
    print(f"  {'micro':>6} {'GPipe峰值':>10} {'1F1B峰值':>10} {'节省':>8} {'bubble':>9}")
    for m in (8, 16, 32, 64):
        c = PipelineConfig(n_stages=8, n_micro=m)
        g, o = simulate(c, "gpipe"), simulate(c, "1f1b")
        print(f"  {m:>6} {g.peak_per_stage:>10} {o.peak_per_stage:>10} "
              f"{(1 - o.peak_per_stage / g.peak_per_stage) * 100:>7.0f}% {o.bubble_fraction * 100:>8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
