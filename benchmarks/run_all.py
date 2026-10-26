"""
天权 TIANQUAN 基准总入口
========================

    python benchmarks/run_all.py              # 完整基准 (数分钟)
    python benchmarks/run_all.py --smoke      # CI 冒烟 (秒级)
    python benchmarks/run_all.py --out r.json # 落盘 JSON

覆盖 L1 通信 → L2 并行 → L3 训练 → L4 服务 全链路, 所有数字实测落盘,
README 中的表格由本脚本生成, 杜绝手抄。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from tianquan.comm import ThreadGroup  # noqa: E402
from tianquan.comm.launch import run_threads, spawn_world  # noqa: E402
from tianquan.eval import Suite, bench, env_fingerprint  # noqa: E402
from tianquan.parallel import (  # noqa: E402
    DDP,
    PipelineConfig,
    TensorParallelGPT,
    zero_memory_table,
    simulate,
)
from tianquan.serve import (  # noqa: E402
    ContinuousScheduler,
    InferenceEngine,
    PagedKVCache,
    NaiveKVCache,
    RadixTree,
    Request,
    StaticScheduler,
    speculative_decode,
    standard_decode,
)
from tianquan.train import GPT, GPTConfig  # noqa: E402

from mp_workers import allreduce_worker, ddp_train_worker, tp_worker, zero_worker  # noqa: E402

SMOKE = "--smoke" in sys.argv


def _bucket_count(cfg_dict: dict, bucket_bytes: int) -> int:
    """给定 bucket 大小下, 一次梯度同步会被切成多少次 AllReduce (仅依赖参数形状)。"""
    return len(DDP(GPT(GPTConfig(**cfg_dict)).params, ThreadGroup.create(1)[0], bucket_bytes).buckets)


# --------------------------------------------------------------------------- #
# B1 集合通信
# --------------------------------------------------------------------------- #
def bench_collectives(suite: Suite, smoke: bool) -> None:
    sizes = [1024] if smoke else [1024, 65_536, 1_048_576]
    for n in ([2, 4] if smoke else [2, 4, 8]):
        for size in sizes:
            for algo in ("ring", "tree"):
                res = spawn_world(allreduce_worker, n, size, algo)
                wall = statistics.median([r["wall_s"] for r in res])
                byts = res[0]["bytes"]
                S = size * 8
                lower = 2 * (n - 1) * (S // n) if algo == "ring" else None
                suite.record(
                    **{
                        "基准": f"AllReduce-{algo}",
                        "world_size": n,
                        "消息": f"{size * 8 / 1024:.0f} KiB",
                        "墙钟(中位)": f"{wall * 1000:.2f} ms",
                        "实测通信量": f"{byts} B",
                        "理论下界": f"{lower} B" if lower else "— (tree 非带宽最优)",
                    }
                )


# --------------------------------------------------------------------------- #
# B2 多进程 DDP 训练扩展率 (核心)
# --------------------------------------------------------------------------- #
def bench_ddp_scaling(suite: Suite, smoke: bool) -> None:
    cfg = dict(
        vocab_size=32,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        seed=42,
    )
    n_steps = 2 if smoke else 8
    bpr = 4
    seq = 16
    worlds = [1, 2] if smoke else [1, 2, 4, 8]

    base = None
    for n in worlds:
        t0 = time.perf_counter()
        res = spawn_world(ddp_train_worker, n, cfg, n_steps, bpr, seq, 1e-3)
        spawn_overhead = time.perf_counter() - t0
        wall = statistics.median([r["wall_s"] for r in res])
        tps = sum(r["tokens_per_s"] for r in res)
        comm = res[0]["comm_bytes"]
        if base is None:
            base = tps
        suite.record(
            **{
                "基准": "DDP 训练 (弱扩展)",
                "world_size": n,
                "墙钟": f"{wall:.3f} s",
                "聚合吞吐": f"{tps:.1f} tok/s",
                "扩展效率": f"{tps / base / n * 100:.1f}%",
                "总通信量": f"{comm / 1024:.1f} KiB",
                "loss": f"{res[0]['first_loss']} → {res[0]['final_loss']}",
                "进程启动开销": f"{spawn_overhead:.2f} s (不计入墙钟)",
            }
        )

    # ---- 分桶调优: 同一并行度下, 桶大小对效率的影响 ---- #
    if not smoke:
        for bucket in (16 * 1024, 64 * 1024, 4 * 1024 * 1024):
            res = spawn_world(ddp_train_worker, 8, cfg, n_steps, bpr, seq, 1e-3, bucket)
            tps = sum(r["tokens_per_s"] for r in res)
            suite.record(
                **{
                    "基准": "DDP 分桶调优 (N=8)",
                    "bucket_bytes": f"{bucket / 1024:.0f} KiB",
                    "allreduce 次数/步": _bucket_count(cfg, bucket),
                    "聚合吞吐": f"{tps:.1f} tok/s",
                    "扩展效率": f"{tps / base / 8 * 100:.1f}%",
                    "总通信量": f"{res[0]['comm_bytes'] / 1024:.1f} KiB",
                }
            )


# --------------------------------------------------------------------------- #
# B3 ZeRO 内存与通信
# --------------------------------------------------------------------------- #
def bench_zero(suite: Suite, smoke: bool) -> None:
    n = 1_000_000
    table = zero_memory_table(n, 8)
    for k, v in table.items():
        suite.record(
            **{
                "策略": k,
                "参数": f"{v['param'] / 1e6:.1f} MB",
                "梯度": f"{v['grad'] / 1e6:.1f} MB",
                "优化器状态": f"{v['optim'] / 1e6:.1f} MB",
                "单卡合计": f"{v['total'] / 1e6:.1f} MB",
                "相对 DDP": f"{v['total'] / table['DDP']['total'] * 100:.1f}%",
            }
        )

    cfg = dict(vocab_size=32, d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=128, seed=1)
    for stage in ([2] if smoke else [1, 2, 3]):
        res = spawn_world(zero_worker, 4, cfg, stage, 16, 4)
        suite.record(
            **{
                "基准": f"ZeRO-{stage} (N=4)",
                "梯度同步耗时": f"{statistics.median([r['wall_s'] for r in res]) * 1000:.2f} ms",
                "通信量": f"{res[0]['comm_bytes'] / 1024:.1f} KiB",
                "单卡内存账本": f"{res[0]['memory']['total'] / 1024:.0f} KiB",
                "参数量": res[0]["n_params"],
            }
        )


# --------------------------------------------------------------------------- #
# B4 张量并行
# --------------------------------------------------------------------------- #
def bench_tensor_parallel(suite: Suite, smoke: bool) -> None:
    cfg = dict(vocab_size=32, d_model=64, n_heads=4, n_kv_heads=4, n_layers=2, d_ff=128, seed=5)
    for n in ([2] if smoke else [1, 2, 4]):
        res = spawn_world(tp_worker, n, cfg, 16, 4, 3)
        wall = statistics.median([r["wall_s"] for r in res])
        suite.record(
            **{
                "基准": "张量并行前向+反向",
                "TP 度数": n,
                "单次耗时": f"{wall * 1000:.2f} ms",
                "通信量/次": f"{res[0]['comm_bytes'] / 1024:.1f} KiB",
                "loss": res[0]["loss"],
            }
        )


# --------------------------------------------------------------------------- #
# B5 流水并行
# --------------------------------------------------------------------------- #
def bench_pipeline(suite: Suite, smoke: bool) -> None:
    for p, m in ([(4, 8)] if smoke else [(4, 8), (8, 32), (4, 32)]):
        cfg = PipelineConfig(n_stages=p, n_micro=m)
        g = simulate(cfg, "gpipe")
        o = simulate(cfg, "1f1b")
        suite.record(
            **{
                "基准": f"流水并行 p={p}, m={m}",
                "GPipe 峰值激活/stage": g.peak_per_stage,
                "1F1B 峰值激活/stage": o.peak_per_stage,
                "激活节省": f"{(1 - o.peak_per_stage / g.peak_per_stage) * 100:.0f}%",
                "bubble(实测)": f"{o.bubble_fraction * 100:.1f}%",
                "bubble(理论 (p-1)/(m+p-1))": f"{(p - 1) / (m + p - 1) * 100:.1f}%",
            }
        )


# --------------------------------------------------------------------------- #
# B6 PagedAttention 内存
# --------------------------------------------------------------------------- #
def bench_paged_attention(suite: Suite, smoke: bool) -> None:
    cfg = GPTConfig(vocab_size=64, d_model=64, n_heads=4, n_kv_heads=2, n_layers=4, d_ff=128)
    rng = np.random.default_rng(0)
    n_seqs = 12 if smoke else 48
    lengths = rng.integers(8, 64, size=n_seqs)
    block_size = 8

    paged = PagedKVCache(
        n_blocks=int(lengths.sum() * 1.2) // block_size + n_seqs,
        block_size=block_size,
        n_layers=cfg.n_layers,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
    )
    for i, L in enumerate(lengths):
        paged.create(i)
        for t in range(int(L)):
            z = np.zeros((cfg.n_kv_heads, cfg.head_dim))
            paged.put(i, 0, t, z, z)

    naive = NaiveKVCache(
        max_seqs=n_seqs + 8,
        max_len=int(lengths.max()),
        n_layers=cfg.n_layers,
        n_kv=cfg.n_kv_heads,
        hd=cfg.head_dim,
    )
    for i, L in enumerate(lengths):
        naive.create(i)
        for t in range(int(L)):
            z = np.zeros((cfg.n_kv_heads, cfg.head_dim))
            naive.put(i, 0, t, z, z)

    ps, ns = paged.stats(), naive.stats()
    suite.record(
        **{
            "基准": "KV 缓存内存效率",
            "请求数": n_seqs,
            "PagedAttention 利用率": f"{ps['logical_slot_utilization'] * 100:.1f}%",
            "预分配 利用率": f"{ns['utilization'] * 100:.1f}%",
            "节省倍率": f"{ns['utilization'] and ps['logical_slot_utilization'] / ns['utilization']:.2f}x",
            "写时复制次数": ps["cow_copies"],
        }
    )


# --------------------------------------------------------------------------- #
# B7 连续批处理 vs 静态批处理
# --------------------------------------------------------------------------- #
def bench_batching(suite: Suite, smoke: bool) -> None:
    cfg = GPTConfig(vocab_size=32, d_model=48, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=96, seed=3)
    model = GPT(cfg)
    eng = InferenceEngine(cfg, source=model)
    rng = np.random.default_rng(1)
    n_req = 8 if smoke else 24
    lens = rng.integers(4, 24, size=n_req)
    reqs = [
        Request(id=i, prompt=list(rng.integers(0, cfg.vocab_size, size=8)), max_new=int(L))
        for i, L in enumerate(lens)
    ]

    def _cache():
        return PagedKVCache(
            n_blocks=2048, block_size=4, n_layers=cfg.n_layers, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim
        )

    _, mc = ContinuousScheduler(eng, _cache(), max_batch=6, seed=2).run([Request(**r.__dict__) for r in reqs])
    _, ms = StaticScheduler(eng, _cache(), max_batch=6, seed=2).run([Request(**r.__dict__) for r in reqs])

    suite.record(
        **{
            "基准": "批处理调度",
            "请求数": n_req,
            "连续批处理 吞吐": f"{mc.throughput:.3f} tok/step",
            "静态批处理 吞吐": f"{ms.throughput:.3f} tok/step",
            "吞吐提升": f"{(mc.throughput / max(ms.throughput, 1e-9) - 1) * 100:.1f}%",
            "槽位利用率": f"{mc.slot_utilization * 100:.1f}% vs {ms.slot_utilization * 100:.1f}%",
            "平均延迟": f"{mc.total_latency / max(mc.n_completed, 1):.1f} vs {ms.total_latency / max(ms.n_completed, 1):.1f} step",
        }
    )


# --------------------------------------------------------------------------- #
# B8 投机解码
# --------------------------------------------------------------------------- #
def bench_speculative(suite: Suite, smoke: bool) -> None:
    V = 64
    rng = np.random.default_rng(11)
    p_t = rng.dirichlet(np.ones(V) * 0.6)
    p_d = p_t**0.6
    p_d /= p_d.sum()
    lt, ld = np.log(p_t), np.log(p_d)
    n = 400 if smoke else 4000

    for gamma in ([4] if smoke else [2, 4, 8]):
        _, st = speculative_decode(
            lambda pre: lt, lambda pre: ld, prefix=[0], n_tokens=n, gamma=gamma,
            rng=np.random.default_rng(7),
        )
        suite.record(
            **{
                "基准": "投机解码",
                "γ": gamma,
                "接受率": f"{st.acceptance_rate * 100:.1f}%",
                "每轮接受 token": f"{st.n_tokens / max(st.n_rounds, 1):.2f}",
                "target 前向减少": f"{st.target_step_reduction:.2f}x",
                "draft 调用": st.n_draft_calls,
            }
        )


# --------------------------------------------------------------------------- #
# B9 Radix 前缀缓存
# --------------------------------------------------------------------------- #
def bench_radix(suite: Suite, smoke: bool) -> None:
    tree = RadixTree()
    rng = np.random.default_rng(0)
    sys_prompt = list(rng.integers(100, 200, size=24))
    n_req = 10 if smoke else 40
    saved = 0
    for i in range(n_req):
        toks = sys_prompt + list(rng.integers(0, 100, size=8))
        n, _ = tree.match(toks)
        saved += n
        tree.insert(toks, [i])
    st = tree.stats()
    total_tokens = n_req * len(sys_prompt)
    suite.record(
        **{
            "基准": "RadixAttention 前缀复用",
            "请求数": n_req,
            "命中率": f"{st['hit_rate'] * 100:.1f}%",
            "省下 prefill token": st["tokens_saved"],
            "占全部 prompt token": f"{st['tokens_saved'] / total_tokens * 100:.1f}%",
        }
    )


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="快速冒烟模式 (CI)")
    ap.add_argument("--out", default="benchmarks/results/report.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    suite = Suite("天权 TIANQUAN 实测基准")
    print(f"[env] {env_fingerprint()}")

    steps = [
        ("B1 集合通信", bench_collectives),
        ("B2 DDP 多进程扩展", bench_ddp_scaling),
        ("B3 ZeRO 内存/通信", bench_zero),
        ("B4 张量并行", bench_tensor_parallel),
        ("B5 流水并行", bench_pipeline),
        ("B6 PagedAttention", bench_paged_attention),
        ("B7 批处理调度", bench_batching),
        ("B8 投机解码", bench_speculative),
        ("B9 Radix 前缀缓存", bench_radix),
    ]
    for name, fn in steps:
        print(f"  → {name} ...", flush=True)
        try:
            fn(suite, a.smoke)
        except Exception as e:  # noqa: BLE001
            print(f"     !! {name} 失败: {type(e).__name__}: {e}")
            suite.record(基准=name, 状态=f"失败: {e}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    suite.to_json(a.out)
    print(f"\n总耗时 {time.perf_counter() - t0:.1f}s → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
