"""
示例 3: 推理服务内核 —— 连续批处理 / PagedAttention / 前缀复用 / 投机解码

运行:  python examples/03_serving.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tianquan.serve import (  # noqa: E402
    ContinuousScheduler,
    InferenceEngine,
    NaiveKVCache,
    PagedKVCache,
    RadixTree,
    Request,
    StaticScheduler,
    speculative_decode,
    standard_decode,
)
from tianquan.train import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(vocab_size=32, d_model=48, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=96, seed=3)


def main() -> int:
    eng = InferenceEngine(CFG, source=GPT(CFG))
    rng = np.random.default_rng(0)

    # ---- 1. 连续批处理 vs 静态批处理 ---- #
    print("== 批处理调度 (24 个请求, 长度 4~24 参差) ==")
    lens = rng.integers(4, 24, size=24)
    reqs = [
        Request(id=i, prompt=list(rng.integers(0, CFG.vocab_size, size=8)), max_new=int(L))
        for i, L in enumerate(lens)
    ]

    def cache():
        return PagedKVCache(
            n_blocks=2048, block_size=4, n_layers=CFG.n_layers,
            n_kv_heads=CFG.n_kv_heads, head_dim=CFG.head_dim,
        )

    _, mc = ContinuousScheduler(eng, cache(), max_batch=6, seed=1).run(reqs)
    _, ms = StaticScheduler(eng, cache(), max_batch=6, seed=1).run(reqs)
    print(f"  连续批处理: 吞吐 {mc.throughput:.3f} tok/step, 槽位利用率 {mc.slot_utilization*100:.1f}%")
    print(f"  静态批处理: 吞吐 {ms.throughput:.3f} tok/step, 槽位利用率 {ms.slot_utilization*100:.1f}%")
    print(f"  → 吞吐提升 {(mc.throughput/max(ms.throughput,1e-9)-1)*100:.1f}%")

    # ---- 2. PagedAttention vs 预分配 ---- #
    print("\n== KV 缓存内存效率 (48 条长度 8~64 的请求) ==")
    lens2 = rng.integers(8, 64, size=48)
    pg = PagedKVCache(n_blocks=int(lens2.sum() * 1.2) // 8 + 48, block_size=8,
                      n_layers=CFG.n_layers, n_kv_heads=CFG.n_kv_heads, head_dim=CFG.head_dim)
    nv = NaiveKVCache(max_seqs=56, max_len=int(lens2.max()), n_layers=CFG.n_layers,
                      n_kv=CFG.n_kv_heads, hd=CFG.head_dim)
    for i, L in enumerate(lens2):
        for c in (pg, nv):
            c.create(i)
            for t in range(int(L)):
                z = np.zeros((CFG.n_kv_heads, CFG.head_dim))
                c.put(i, 0, t, z, z)
    ps, ns = pg.stats(), nv.stats()
    print(f"  PagedAttention 利用率 {ps['logical_slot_utilization']*100:.1f}%")
    print(f"  预分配     利用率 {ns['utilization']*100:.1f}%")
    print(f"  → 节省 {ps['logical_slot_utilization']/max(ns['utilization'],1e-9):.2f}x")

    # ---- 3. Radix 前缀复用 ---- #
    print("\n== RadixAttention 前缀复用 (共享 24-token 系统提示) ==")
    tree = RadixTree()
    sp = list(rng.integers(100, 200, size=24))
    for i in range(40):
        toks = sp + list(rng.integers(0, 100, size=8))
        n, _ = tree.match(toks)
        tree.insert(toks, [i])
    st = tree.stats()
    print(f"  命中率 {st['hit_rate']*100:.1f}%, 省下 prefill token {st['tokens_saved']} "
          f"(占 {st['tokens_saved']/(40*24)*100:.1f}%)")

    # ---- 4. 投机解码 ---- #
    print("\n== 投机解码 (Leviathan et al. 2023) ==")
    V = 64
    p_t = rng.dirichlet(np.ones(V) * 0.6)
    p_d = p_t**0.6
    p_d /= p_d.sum()
    lt, ld = np.log(p_t), np.log(p_d)
    for gamma in (2, 4, 8):
        _, s = speculative_decode(lambda p: lt, lambda p: ld, [0], 2000, gamma=gamma,
                                  rng=np.random.default_rng(7))
        print(f"  γ={gamma}: 接受率 {s.acceptance_rate*100:5.1f}%  "
              f"每轮 {s.n_tokens/max(s.n_rounds,1):.2f} token  "
              f"target 前向减少 {s.target_step_reduction:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
