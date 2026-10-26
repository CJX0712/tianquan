"""
示例 1: 真实多进程 DDP 训练 (数据并行)

运行:  python examples/01_ddp_training.py

要点
----
* 每个进程持有一份完整模型副本, 吃掉**不同的数据分片**
* 反向后把梯度 AllReduce 求平均 ⇒ 等价于单进程吃下整个 global batch
* 打印聚合吞吐与扩展效率
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from tianquan.comm.launch import spawn_world  # noqa: E402
from mp_workers import ddp_train_worker  # noqa: E402

CFG = dict(vocab_size=32, d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=128, seed=42)


def main() -> int:
    base = None
    print(f"{'进程数':>6} {'墙钟(s)':>10} {'聚合吞吐(tok/s)':>16} {'扩展效率':>10} {'loss':>18}")
    print("-" * 66)
    for n in (1, 2, 4, 8):
        res = spawn_world(ddp_train_worker, n, CFG, 8, 4, 16, 1e-3)
        wall = statistics.median(r["wall_s"] for r in res)
        tps = sum(r["tokens_per_s"] for r in res)
        base = tps if base is None else base
        eff = tps / base / n * 100
        print(
            f"{n:>6} {wall:>10.3f} {tps:>16.1f} {eff:>9.1f}% "
            f"{res[0]['first_loss']:>8.4f}→{res[0]['final_loss']:<8.4f}"
        )
    print("\n注: 扩展效率 = 聚合吞吐 / (单进程吞吐 × 进程数)。")
    print("    小模型下通信占比高, 效率下降属正常现象 —— 这正是本基准想量化暴露的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
