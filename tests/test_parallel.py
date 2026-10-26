"""
并行策略层的不变量验证 (L2)

硬不变量
--------
I1 DDP 等价性:  N 个 rank 各自 micro-batch 的梯度经 AllReduce 平均后,
                必须等于单进程用 global batch 算出的梯度 (1e-10)
I2 ZeRO 内存表: 与 Rajbhandari et al. 2020 Table 1 逐格一致
I3 ZeRO-2 分片: reduce_scatter 后本 rank 分片 == 全局平均梯度的对应片段
I4 张量并行:    Megatron 切分后的前向输出 == 未切分单进程 (1e-10)
I5 流水并行:    1F1B 峰值激活 <= p 而 GPipe == m; bubble ≈ (p-1)/m
"""

from __future__ import annotations

import numpy as np
import pytest

from tianquan.comm.launch import run_threads
from tianquan.parallel import (
    DDP,
    PipelineConfig,
    TensorParallelGPT,
    ZeRO,
    gpipe_schedule,
    onef1b_schedule,
    simulate,
    zero_memory_table,
)
from tianquan.train import GPT, AdamW, GPTConfig


def _mk_data(vocab: int, seq: int, bs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, vocab, size=(bs, seq + 1))
    return x[:, :-1], x[:, 1:]


# --------------------------------------------------------------------------- #
# I1 DDP == 单进程 global batch
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_ddp_equals_global_batch(world_size: int) -> None:
    cfg = GPTConfig(vocab_size=16, d_model=32, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=64, seed=7)
    idx_all, tgt_all = _mk_data(cfg.vocab_size, 8, batch := 2 * world_size, seed=3)

    # 参照: 单进程吃下整个 global batch
    ref = GPT(cfg)
    ref.zero_grad()
    _, loss_ref = ref.forward(idx_all, tgt_all)
    loss_ref.backward()
    ref_grads = {k: v.grad.copy() for k, v in ref.params.items()}

    shards = [(idx_all[i * 2 : i * 2 + 2], tgt_all[i * 2 : i * 2 + 2]) for i in range(world_size)]

    def fn(rank, pg):
        m = GPT(cfg)  # 同种子 ⇒ 同初始参数
        m.zero_grad()
        _, loss = m.forward(shards[rank][0], shards[rank][1])
        loss.backward()
        ddp = DDP(m.params, pg, bucket_bytes=1024)
        ddp.mark_all_ready()
        ddp.sync()
        return {k: v.grad.copy() for k, v in m.params.items()}

    got = run_threads(fn, world_size)
    for g in got:
        for k in ref_grads:
            np.testing.assert_allclose(
                g[k], ref_grads[k], rtol=1e-9, atol=1e-10, err_msg=f"DDP 梯度不等价: {k}"
            )


# --------------------------------------------------------------------------- #
# I2 ZeRO 内存表 (论文 Table 1)
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [1, 2, 4, 8, 16])
def test_zero_memory_table(world_size: int) -> None:
    P = 1_000_000 * 4  # 1M 参数, fp32
    t = zero_memory_table(1_000_000, world_size)
    assert t["DDP"]["total"] == 4 * P
    assert t["ZeRO-1"]["total"] == 2 * P + 2 * P // world_size
    assert t["ZeRO-2"]["total"] == P + 3 * P // world_size
    assert t["ZeRO-3"]["total"] == 4 * P // world_size
    # 单调性: 分片度越高越省
    assert t["DDP"]["total"] >= t["ZeRO-1"]["total"] >= t["ZeRO-2"]["total"] >= t["ZeRO-3"]["total"]
    # 极限: N→∞ 时 ZeRO-3 总内存 → 0
    t_big = zero_memory_table(1_000_000, 1024)
    assert t_big["ZeRO-3"]["total"] < t["ZeRO-3"]["total"] or world_size >= 1024


# --------------------------------------------------------------------------- #
# I3 ZeRO-2 reduce_scatter 分片正确
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [2, 4])
def test_zero2_shard_matches_global_mean(world_size: int) -> None:
    cfg = GPTConfig(vocab_size=16, d_model=32, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, seed=11)
    idx_all, tgt_all = _mk_data(cfg.vocab_size, 8, 2 * world_size, seed=5)
    shards = [(idx_all[i * 2 : i * 2 + 2], tgt_all[i * 2 : i * 2 + 2]) for i in range(world_size)]

    ref = GPT(cfg)
    ref.zero_grad()
    ref.forward(idx_all, tgt_all)[1].backward()
    flat_ref = np.concatenate([v.grad.reshape(-1) for v in ref.params.values()])

    def fn(rank, pg):
        m = GPT(cfg)
        m.zero_grad()
        m.forward(shards[rank][0], shards[rank][1])[1].backward()
        z = ZeRO(m.params, pg, stage=2)
        z.reduce_grads()
        flat = np.concatenate([v.grad.reshape(-1) for v in m.params.values()])
        lo, hi = z.offset, min(z.offset + z.shard_size, z.total)
        return {"flat": flat, "offset": z.offset, "hi": hi}

    got = run_threads(fn, world_size)
    for g in got:
        lo, hi = g["offset"], g["hi"]
        if lo < hi:
            np.testing.assert_allclose(g["flat"][lo:hi], flat_ref[lo:hi], rtol=1e-9, atol=1e-10)
        # 分片外必须为零 (不持有)
        if hi < len(flat_ref):
            np.testing.assert_allclose(g["flat"][hi:], 0.0, atol=0)


# --------------------------------------------------------------------------- #
# I4 张量并行 == 单进程
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_tensor_parallel_matches_single(world_size: int) -> None:
    cfg = GPTConfig(
        vocab_size=16, d_model=32, n_heads=4, n_kv_heads=4, n_layers=2, d_ff=64, seed=13
    )
    # 注意: TP 要求 n_kv_heads % world_size == 0, 故这里用 MHA (n_kv_heads == n_heads)
    idx, tgt = _mk_data(cfg.vocab_size, 6, 2, seed=9)

    ref = GPT(cfg)
    ref_logits, ref_loss = ref.forward(idx, tgt)

    def fn(rank, pg):
        m = TensorParallelGPT(cfg, pg, seed=13)
        lg, ls = m.forward(idx, tgt)
        return {"logits": lg.data, "loss": float(ls.data)}

    got = run_threads(fn, world_size)
    for g in got:
        np.testing.assert_allclose(g["logits"], ref_logits.data, rtol=1e-9, atol=1e-10)
        assert abs(g["loss"] - float(ref_loss.data)) < 1e-10


# --------------------------------------------------------------------------- #
# I5 流水并行: 峰值激活与 bubble
# --------------------------------------------------------------------------- #
@pytest.mark.invariant
@pytest.mark.parametrize("p,m", [(4, 8), (4, 16), (8, 16), (2, 8)])
def test_pipeline_1f1b_lowers_peak_activation(p: int, m: int) -> None:
    cfg = PipelineConfig(n_stages=p, n_micro=m)
    g = simulate(cfg, "gpipe")
    o = simulate(cfg, "1f1b")

    # 论文口径: GPipe 每 stage 峰值 = m; 1F1B <= p
    assert g.peak_per_stage == m, f"GPipe 峰值应为 m={m}, 实际 {g.peak_per_stage}"
    assert o.peak_per_stage <= p, f"1F1B 峰值应 <= p={p}, 实际 {o.peak_per_stage}"
    assert o.peak_activations < g.peak_activations, "1F1B 总激活必须低于 GPipe"

    # bubble ≈ (p-1)/m
    theory = (p - 1) / (m + p - 1)
    assert abs(o.bubble_fraction - theory) < 0.02, f"1F1B bubble {o.bubble_fraction} vs 理论 {theory}"
    assert abs(g.bubble_fraction - theory) < 0.02, f"GPipe bubble {g.bubble_fraction} vs 理论 {theory}"


@pytest.mark.invariant
def test_pipeline_schedule_shapes() -> None:
    p, m = 4, 8
    for sched in (gpipe_schedule(p, m), onef1b_schedule(p, m)):
        assert len(sched) == p
        for s in range(p):
            nf = sum(1 for _, k in sched[s] if k == "F")
            nb = sum(1 for _, k in sched[s] if k == "B")
            assert nf == m and nb == m, "每个 stage 必须执行 m 次前向和 m 次反向"


@pytest.mark.invariant
def test_more_microbatches_reduces_bubble() -> None:
    """microbatch 越多, bubble 越低 —— 这是流水并行的根本权衡。"""
    bubbles = [simulate(PipelineConfig(n_stages=4, n_micro=m), "1f1b").bubble_fraction for m in (4, 8, 32)]
    assert bubbles[0] > bubbles[1] > bubbles[2]
