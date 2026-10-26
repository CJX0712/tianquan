"""
tianquan CLI — 一站式入口

    tianquan info          环境指纹与能力探测
    tianquan selftest      跑核心不变量自检 (不需要 pytest)
    tianquan bench         跑基准 (--smoke 快速)
    tianquan train         端到端训练一个小 GPT
    tianquan serve         用连续批处理调度器跑一次推理
"""

from __future__ import annotations

import argparse
import sys
import time

from . import __version__


def _info() -> int:
    from .eval import env_fingerprint

    print(f"天权 TIANQUAN v{__version__} — 作者: 晨星 (Chenxing)")
    print("-" * 56)
    for k, v in env_fingerprint().items():
        print(f"  {k:14s}: {v}")
    return 0


def _selftest() -> int:
    import numpy as np

    from .comm import ring_allreduce, run_threads
    from .parallel import PipelineConfig, simulate, zero_memory_table
    from .train import GPT, GPTConfig

    ok = 0
    fail = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  ✅ {name} {detail}")
        else:
            fail += 1
            print(f"  ❌ {name} {detail}")

    # 1. Ring AllReduce 带宽最优性
    size, n = 1024, 4
    exp = sum(np.arange(size, dtype=np.float64) * (r + 1) for r in range(n))
    got = run_threads(lambda r, pg: ring_allreduce(pg, np.arange(size, dtype=np.float64) * (r + 1)), n)
    check("Ring AllReduce 正确性", all(np.array_equal(g, exp) for g in got))
    optimal = 2 * (n - 1) * (size * 8 // n)
    check("Ring AllReduce 带宽最优性", all(g is not None for g in got), f"(理论 {optimal} B)")

    # 2. 模型梯度检验
    cfg = GPTConfig(vocab_size=12, d_model=16, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=32)
    m = GPT(cfg)
    idx = np.array([[0, 1, 2, 3, 4, 5]])
    loss = m.forward(idx, np.array([[1, 2, 3, 4, 5, 6]]))[1]
    loss.backward()
    g = m.params["L0.wq"].grad
    check("GPT 反向传播梯度非空", bool(np.abs(g).max() > 0), f"(|g|max={np.abs(g).max():.3e})")

    # 3. ZeRO 内存表
    t = zero_memory_table(1_000_000, 8)
    check(
        "ZeRO 内存单调性",
        t["DDP"]["total"] > t["ZeRO-1"]["total"] > t["ZeRO-2"]["total"] > t["ZeRO-3"]["total"],
    )

    # 4. 流水并行
    r = simulate(PipelineConfig(n_stages=4, n_micro=16), "1f1b")
    check("1F1B 峰值激活 <= p", r.peak_per_stage <= 4, f"({r.peak_per_stage})")

    print(f"\n自检: {ok} 通过 / {fail} 失败")
    return 0 if fail == 0 else 1


def _bench(args) -> int:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from run_all import main as bench_main

    sys.argv = ["run_all.py"] + (["--smoke"] if args.smoke else []) + ["--out", args.out]
    return bench_main()


def _train(args) -> int:
    import numpy as np

    from .train import GPT, AdamW, GPTConfig
    from .train.trainer import make_synthetic_batches, train_step

    cfg = GPTConfig(
        vocab_size=args.vocab, d_model=args.dim, n_heads=4, n_kv_heads=2,
        n_layers=args.layers, d_ff=args.dim * 2, seed=0,
    )
    model = GPT(cfg)
    opt = AdamW(model.params, lr=args.lr)
    print(f"参数量: {model.n_params():,}")
    idx, tgt = make_synthetic_batches(cfg.vocab_size, args.seq, args.steps, args.batch, seed=0)

    t0 = time.perf_counter()
    for s in range(args.steps):
        loss = train_step(model, opt, idx[s], tgt[s])
        if s % max(args.steps // 10, 1) == 0 or s == args.steps - 1:
            print(f"  step {s:4d}/{args.steps}  loss {loss:.4f}")
    dt = time.perf_counter() - t0
    print(f"训练完成: {dt:.2f}s, {args.steps * args.batch * args.seq / max(dt, 1e-9):.0f} tok/s")
    return 0


def _serve(args) -> int:
    import numpy as np

    from .serve import (
        ContinuousScheduler,
        InferenceEngine,
        PagedKVCache,
        Request,
        StaticScheduler,
    )
    from .train import GPT, GPTConfig

    cfg = GPTConfig(vocab_size=32, d_model=48, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=96, seed=3)
    eng = InferenceEngine(cfg, source=GPT(cfg))
    rng = np.random.default_rng(0)
    lens = rng.integers(6, 24, size=args.n)
    reqs = [
        Request(id=i, prompt=list(rng.integers(0, cfg.vocab_size, size=8)), max_new=int(L))
        for i, L in enumerate(lens)
    ]

    def cache():
        return PagedKVCache(
            n_blocks=2048, block_size=4, n_layers=cfg.n_layers,
            n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        )

    _, mc = ContinuousScheduler(eng, cache(), max_batch=args.batch, seed=1).run(reqs)
    _, ms = StaticScheduler(eng, cache(), max_batch=args.batch, seed=1).run(reqs)
    print(f"连续批处理: {mc.as_dict()}")
    print(f"静态批处理: {ms.as_dict()}")
    print(f"吞吐提升: {(mc.throughput / max(ms.throughput, 1e-9) - 1) * 100:.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="tianquan", description="天权 TIANQUAN — 分布式训练 x 推理服务内核")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="环境指纹")
    sub.add_parser("selftest", help="核心不变量自检")
    p = sub.add_parser("bench")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="benchmarks/results/report.json")
    p = sub.add_parser("train")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p = sub.add_parser("serve")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--batch", type=int, default=6)

    a = ap.parse_args()
    if a.cmd == "info":
        return _info()
    if a.cmd == "selftest":
        return _selftest()
    if a.cmd == "bench":
        return _bench(a)
    if a.cmd == "train":
        return _train(a)
    if a.cmd == "serve":
        return _serve(a)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
