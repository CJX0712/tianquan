"""
tianquan.parallel.tensor — 张量并行 (Megatron-LM 方案)

零重写声明
----------
严格按 **Megatron-LM (Shoeybi et al. 2019)** 的切分规则:

* ``ColumnParallelLinear``: 权重按**输出维**切。每个 rank 只算自己那几列,
  输出天然是"按列分片"的, 可直接喂给下游的 RowParallel, **无需通信**。
  → 用于 wq / wk / wv / w_gate / w_up
* ``RowParallelLinear``: 权重按**输入维**切。每个 rank 算出部分和,
  必须 **AllReduce** 才能得到完整输出。
  → 用于 wo / w_down

因此每个 Transformer 层恰好 **2 次 AllReduce** (attention 输出 + FFN 输出),
这正是 Megatron 的设计 —— 把通信压到最低。

注意力天然按 head 切: wq 按列切 ⇒ 每个 rank 持有 n_heads/N 个注意力头,
KV 同理 (GQA 下要求 n_kv_heads 能被 N 整除)。

正确性
------
TP 的前向输出 / 反向梯度必须与**未切分的单进程模型**逐位一致 (1e-10),
由 tests/test_parallel.py::test_tensor_parallel_matches_single 验证。
AllReduce 的反向是恒等映射 (y = Σxᵢ ⇒ ∂L/∂xᵢ = ∂L/∂y)。
"""

from __future__ import annotations

import numpy as np

from ..comm import allreduce
from ..comm.backend import ProcessGroup
from ..train import autograd as ag
from ..train.model import GPT, GPTConfig, causal_mask

__all__ = ["TensorParallelGPT", "COLUMN_PARALLEL", "ROW_PARALLEL"]


# 按输出维切 (无需通信) / 按输入维切 (需要 allreduce)
COLUMN_PARALLEL = ("wq", "wk", "wv", "w_gate", "w_up")
ROW_PARALLEL = ("wo", "w_down")


def _all_reduce_op(a: ag.Tensor, pg: ProcessGroup) -> ag.Tensor:
    """前向 AllReduce, 反向恒等 (sum 的雅可比每块都是 I)。"""
    out_data = allreduce(pg, a.data, op="sum", algo="auto")
    out = ag.Tensor(out_data, requires_grad=a.requires_grad)
    ag._CHILD_REG[id(out)] = (a,)

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad

    out._backward = _bw
    return out


class TensorParallelGPT:
    """对 GPT 做 Megatron 式张量并行。每个 rank 持有切分后的权重。"""

    def __init__(self, cfg: GPTConfig, pg: ProcessGroup, seed: int = 0) -> None:
        self.cfg, self.pg = cfg, pg
        self.world_size = pg.world_size
        self.rank = pg.rank
        self.p: dict[str, ag.Tensor] = {}

        full = GPT(GPTConfig(**{**cfg.__dict__, "seed": seed}))
        for name, t in full.params.items():
            self.p[name] = ag.Tensor(self._shard(name, t.data), requires_grad=True, name=name)
        self._mask_cache: dict[int, np.ndarray] = {}
        self.n_comm = 0

    def _shard(self, name: str, w: np.ndarray) -> np.ndarray:
        N = self.world_size
        suffix = name.split(".")[-1]
        if w.ndim >= 2 and suffix in COLUMN_PARALLEL:
            assert w.shape[1] % N == 0, f"{name} 输出维 {w.shape[1]} 不能被 {N} 整除"
            return np.ascontiguousarray(np.split(w, N, axis=1)[self.rank])
        if w.ndim >= 2 and suffix in ROW_PARALLEL:
            assert w.shape[0] % N == 0, f"{name} 输入维 {w.shape[0]} 不能被 {N} 整除"
            return np.ascontiguousarray(np.split(w, N, axis=0)[self.rank])
        return w.copy()  # 归一化参数 / 嵌入: 各 rank 复制

    @property
    def params(self) -> dict[str, ag.Tensor]:
        return self.p

    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None):
        cfg = self.cfg
        B, T = idx.shape
        N = self.world_size
        nh = cfg.n_heads // N  # 本 rank 持有的 Q 头数
        nkv = cfg.n_kv_heads // N
        hd = cfg.head_dim
        assert cfg.n_heads % N == 0 and cfg.n_kv_heads % N == 0, "头数必须能被 TP 度数整除"

        if T not in self._mask_cache:
            self._mask_cache[T] = causal_mask(T).reshape(1, 1, T, T)
        mask = self._mask_cache[T]

        x = ag.embedding(self.p["tok_emb"], idx)
        for i in range(cfg.n_layers):
            pf = f"L{i}."
            h = ag.rms_norm(x, self.p[pf + "attn_norm"])
            flat = h.reshape(B * T, cfg.d_model)
            q = ag.matmul(flat, self.p[pf + "wq"]).reshape(B, T, nh, hd)
            k = ag.matmul(flat, self.p[pf + "wk"]).reshape(B, T, nkv, hd)
            v = ag.matmul(flat, self.p[pf + "wv"]).reshape(B, T, nkv, hd)

            # 与 train/model.py 一致: 先置换成 (B,nh,T,hd) 再做 RoPE
            qh = ag.rope(q.transpose(0, 2, 1, 3), cfg.rope_theta)
            kh = ag.rope(k.transpose(0, 2, 1, 3), cfg.rope_theta)
            vh = v.transpose(0, 2, 1, 3)

            rep = nh // nkv
            if rep > 1:
                kh = _repeat_kv(kh, rep)
                vh = _repeat_kv(vh, rep)
            scores = ag.add(ag.bmm(qh, kh.transpose(0, 1, 3, 2)) / np.sqrt(hd), ag.Tensor(mask))
            o = ag.bmm(ag.softmax(scores, axis=-1), vh)
            o = o.transpose(0, 2, 1, 3).reshape(B * T, nh * hd)
            o = ag.matmul(o, self.p[pf + "wo"]).reshape(B, T, cfg.d_model)
            o = _all_reduce_op(o, self.pg)  # ← 通信点 1
            x = ag.add(x, o)

            h2 = ag.rms_norm(x, self.p[pf + "ffn_norm"]).reshape(B * T, cfg.d_model)
            g = ag.matmul(h2, self.p[pf + "w_gate"])
            u = ag.matmul(h2, self.p[pf + "w_up"])
            ff = ag.matmul(ag.mul(ag.silu(g), u), self.p[pf + "w_down"]).reshape(B, T, cfg.d_model)
            ff = _all_reduce_op(ff, self.pg)  # ← 通信点 2
            x = ag.add(x, ff)

        xn = ag.rms_norm(x, self.p["final_norm"]).reshape(B * T, cfg.d_model)
        logits = ag.matmul(xn, ag.transpose2d(self.p["tok_emb"])).reshape(B, T, cfg.vocab_size)
        loss = (
            ag.cross_entropy(logits.reshape(B * T, cfg.vocab_size), targets.reshape(-1))
            if targets is not None
            else None
        )
        self.n_comm += 2 * cfg.n_layers
        return logits, loss


def _repeat_kv(kv: ag.Tensor, rep: int) -> ag.Tensor:
    """(B,nkv,T,hd) → 沿 head 维 (axis=1) 重复 → (B,nkv*rep,T,hd)。"""
    data = np.repeat(kv.data, rep, axis=1)
    out = ag.Tensor(data, requires_grad=kv.requires_grad)
    ag._CHILD_REG[id(out)] = (kv,)

    def _bw() -> None:
        if kv.requires_grad:
            B, H, T, hd = out.grad.shape
            kv.grad += out.grad.reshape(B, H // rep, rep, T, hd).sum(axis=2)

    out._backward = _bw
    return out
