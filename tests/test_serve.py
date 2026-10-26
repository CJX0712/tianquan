"""
推理服务内核的不变量验证 (L4)

硬不变量
--------
I1 KV 缓存等价: 增量 decode 的结果 == 整段重算 (1e-10)
I2 推理引擎与训练模型权重一致: 同一输入的 logits 逐位相同
I3 PagedAttention: 块利用率显著高于"按 max_len 预分配"; 写时复制保证分叉隔离
I4 连续批处理: 吞吐 >= 静态批处理, 槽位利用率更高
I5 Radix 前缀缓存: 相同前缀命中并省下 prefill token
I6 投机解码: 输出分布与 target 自回归**统计一致** (TVD 检验), 且减少 target 前向次数
"""

from __future__ import annotations

import numpy as np
import pytest

from tianquan.serve import (
    ContinuousScheduler,
    InferenceEngine,
    NaiveKVCache,
    PagedKVCache,
    RadixTree,
    Request,
    StaticScheduler,
    sample,
    speculative_decode,
    standard_decode,
)
from tianquan.train import GPT, GPTConfig


CFG = GPTConfig(vocab_size=24, d_model=32, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=64, seed=21)


def _cache(cfg=CFG, n_blocks=64, block_size=4):
    return PagedKVCache(
        n_blocks=n_blocks,
        block_size=block_size,
        n_layers=cfg.n_layers,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
    )


# --------------------------------------------------------------------------- #
# I1 / I2 推理正确性
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
def test_engine_matches_training_model() -> None:
    """推理引擎 (纯 NumPy) 与训练模型 (autograd) 对同一输入必须给出相同 logits。"""
    model = GPT(CFG)
    eng = InferenceEngine(CFG, source=model)
    tokens = np.array([1, 5, 9, 3, 7])
    cache = _cache()

    # 训练模型: 全量前向 (teacher forcing, 取最后一个位置)
    logits_train, _ = model.forward(tokens[None, :])
    expected = logits_train.data[0, -1]

    cache.create(0)
    got = eng.prefill(tokens, cache, 0)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.invariant
def test_kv_cache_incremental_equals_recompute() -> None:
    """核心: 逐步 decode == 从头重算。这是 KV cache 能用的前提。"""
    model = GPT(CFG)
    eng = InferenceEngine(CFG, source=model)
    prompt = [1, 5, 9, 3, 7]
    extra = [2, 6]

    # 参照: 把 prompt+extra 整段 prefill
    ref_cache = _cache()
    ref_cache.create(99)
    seq = np.array(prompt + extra)
    ref_last = eng.prefill(seq, ref_cache, 99)

    # 增量: prefill prompt, 然后逐步 decode
    cache = _cache()
    cache.create(0)
    nxt = sample(eng.prefill(np.array(prompt), cache, 0), temperature=0)
    # temperature=0 ⇒ 贪心; 这里直接走 decode 路径喂入 extra 的真实 token
    pos = len(prompt)
    for t in extra:
        logits = eng.decode(t, pos, cache, 0)
        pos += 1
    np.testing.assert_allclose(logits, ref_last, rtol=1e-9, atol=1e-10)


# --------------------------------------------------------------------------- #
# I3 PagedAttention
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
def test_paged_attention_beats_preallocation() -> None:
    """同样服务 N 条长度不一的请求, 分页缓存的 token 利用率必须显著更高。"""
    rng = np.random.default_rng(0)
    lengths = rng.integers(4, 20, size=12)  # 长度参差 ⇒ 预分配浪费巨大

    paged = _cache(n_blocks=200, block_size=4)
    for i, L in enumerate(lengths):
        paged.create(i)
        for t in range(int(L)):
            z = np.zeros((CFG.n_kv_heads, CFG.head_dim))
            paged.put(i, 0, t, z, z)

    naive_max_len = int(lengths.max())
    naive = NaiveKVCache(max_seqs=32, max_len=naive_max_len, n_layers=CFG.n_layers, n_kv=CFG.n_kv_heads, hd=CFG.head_dim)
    for i, L in enumerate(lengths):
        naive.create(i)
        for t in range(int(L)):
            z = np.zeros((CFG.n_kv_heads, CFG.head_dim))
            naive.put(i, 0, t, z, z)

    paged_util = paged.stats()["logical_slot_utilization"]
    naive_util = naive.stats()["utilization"]
    assert paged_util > naive_util, f"分页 {paged_util:.3f} 应优于预分配 {naive_util:.3f}"
    assert paged_util > 0.75, f"分页利用率偏低: {paged_util:.3f}"


@pytest.mark.invariant
def test_copy_on_write_isolates_forked_sequences() -> None:
    """写时复制: fork 出的两条序列共享前缀, 但各自写入后互不污染。"""
    cache = _cache(n_blocks=64, block_size=4)
    cache.create(0)
    for t in range(6):
        v = np.full((CFG.n_kv_heads, CFG.head_dim), float(t))
        cache.put(0, 0, t, v, v)

    cache.create(1)
    cache.fork(0, 1)
    assert cache.shared_blocks >= 1, "fork 应产生共享块"

    # 子序列写入分叉点
    v = np.full((CFG.n_kv_heads, CFG.head_dim), 999.0)
    cache.put(1, 0, 6, v, v)

    k0, _ = cache.get(0, 0, 6)
    k1, _ = cache.get(1, 0, 6)
    np.testing.assert_allclose(k0, k1)  # 分叉前的部分仍然一致
    k1_full, _ = cache.get(1, 0, 7)
    assert np.allclose(k1_full[6], 999.0)
    k0_full, _ = cache.get(0, 0, 6)
    assert not np.allclose(k0_full[-1], 999.0), "父序列不应被子序列写入影响"


# --------------------------------------------------------------------------- #
# I4 连续批处理 vs 静态批处理
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
def test_continuous_batching_higher_throughput() -> None:
    model = GPT(CFG)
    eng = InferenceEngine(CFG, source=model)
    rng = np.random.default_rng(3)
    # 长度差异越大, 静态批处理的空转浪费越严重
    reqs = [
        Request(id=i, prompt=list(rng.integers(0, CFG.vocab_size, size=6)), max_new=int(L))
        for i, L in enumerate([4, 12, 6, 20, 8, 16, 5, 18])
    ]

    c_cache = _cache(n_blocks=512, block_size=4)
    _, mc = ContinuousScheduler(eng, c_cache, max_batch=4, seed=1).run([Request(**r.__dict__) for r in reqs])

    s_cache = _cache(n_blocks=512, block_size=4)
    _, ms = StaticScheduler(eng, s_cache, max_batch=4, seed=1).run([Request(**r.__dict__) for r in reqs])

    assert mc.n_completed == ms.n_completed == len(reqs)
    assert mc.throughput >= ms.throughput, (
        f"连续批处理吞吐 {mc.throughput:.3f} 应 >= 静态 {ms.throughput:.3f}"
    )
    assert mc.slot_utilization > ms.slot_utilization, "连续批处理槽位利用率应更高"


# --------------------------------------------------------------------------- #
# I5 Radix 前缀缓存
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
def test_radix_prefix_cache_hit() -> None:
    tree = RadixTree()
    sys_prompt = [1, 2, 3, 4, 5, 6]

    tree.insert(sys_prompt + [10], [100])
    n, _ = tree.match(sys_prompt + [10, 11])
    assert n == len(sys_prompt) + 1, f"应命中完整前缀, 实际 {n}"

    n2, _ = tree.match([9, 9, 9])
    assert n2 == 0, "完全不同的前缀不应命中"

    st = tree.stats()
    assert st["hit_rate"] > 0
    assert st["tokens_saved"] >= len(sys_prompt)


# --------------------------------------------------------------------------- #
# I6 投机解码分布一致性 (核心定理)
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
def test_speculative_decoding_distribution_matches_target() -> None:
    """Leviathan et al. 定理: 投机解码输出分布 == target 自回归分布。

    用固定的 target / draft 分布做大样本频率检验 (TVD)。
    """
    V = 8
    rng_target = np.random.default_rng(11)
    p_target = rng_target.dirichlet(np.ones(V) * 0.8)
    p_draft = p_target ** 0.7  # draft 与 target 相似但不相同
    p_draft /= p_draft.sum()

    log_p_t = np.log(p_target)
    log_p_d = np.log(p_draft)

    def target_next(prefix, _k=None):
        return log_p_t.copy()

    def draft_next(prefix):
        return log_p_d.copy()

    N = 4000
    rng = np.random.default_rng(7)
    spec_tokens, st = speculative_decode(
        target_next, draft_next, prefix=[0], n_tokens=N, gamma=4, rng=rng
    )
    std_tokens = standard_decode(target_next, prefix=[0], n_tokens=N, rng=np.random.default_rng(8))

    def freq(tokens):
        c = np.bincount(np.array(tokens), minlength=V).astype(np.float64)
        return c / c.sum()

    f_spec, f_std = freq(spec_tokens), freq(std_tokens)
    tvd_spec = 0.5 * np.abs(f_spec - p_target).sum()
    tvd_std = 0.5 * np.abs(f_std - p_target).sum()

    # 投机解码与标准解码应同样接近 target 分布 (互为对照)
    assert tvd_spec < 0.03, f"投机解码分布偏离 target: TVD={tvd_spec:.4f}"
    assert tvd_spec <= tvd_std * 1.6 + 0.01, f"投机 {tvd_spec:.4f} 显著差于标准 {tvd_std:.4f}"
    assert st.acceptance_rate > 0.3, f"接受率过低: {st.acceptance_rate}"
    assert st.target_step_reduction > 1.5, f"未减少 target 前向: {st.target_step_reduction}"


@pytest.mark.invariant
def test_speculative_exact_when_draft_equals_target() -> None:
    """极端情形: draft == target ⇒ 接受率应为 100%, 每轮产出 γ+1 个 token。"""
    V = 6
    p = np.full(V, 1.0 / V)
    lp = np.log(p)
    rng = np.random.default_rng(5)
    tokens, st = speculative_decode(
        lambda pre: lp, lambda pre: lp, prefix=[0], n_tokens=20, gamma=4, rng=rng
    )
    assert st.n_accepted == 20
    assert st.acceptance_rate == 1.0
    assert st.n_rounds <= 5  # 20 / (4+1) = 4 轮


@pytest.mark.invariant
def test_sampler_top_p_and_greedy() -> None:
    logits = np.array([1.0, 5.0, 2.0, 0.1])
    assert sample(logits, temperature=0) == 1  # 贪心
    rng = np.random.default_rng(0)
    counts = np.bincount([sample(logits, temperature=1.0, rng=rng) for _ in range(500)], minlength=4)
    assert counts[1] == counts.max(), "高 logits 的 token 应最常被采到"
