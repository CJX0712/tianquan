"""
tianquan.train.model — 小 GPT (L3)

架构选型 (全部为当前 SOTA 解码器的标准件, 不 invent):
* Pre-RMSNorm          — Zhang & Sennrich 2019 / LLaMA (Touvron et al. 2023)
* RoPE                 — Su et al. 2021 (RoFormer)
* GQA                  — Ainslie et al. 2023 (分组查询注意力, LLaMA-3/Qwen2 标配)
* SwiGLU FFN           — Shazeer 2020
* 权重绑定 (tie emb)    — Press & Wolf 2017

之所以自己写而不用 transformers 库: 让并行策略层 (DDP/ZeRO/TP/PP) 能
**直接操作参数分片**, 并让每条梯度都能被有限差分独立验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import autograd as ag

__all__ = ["GPTConfig", "GPT", "causal_mask"]


def causal_mask(t: int) -> np.ndarray:
    """因果掩码: 下三角 (含对角) 为 0, 上三角为 -inf。"""
    m = np.triu(np.ones((t, t), dtype=np.float64), k=1)
    return np.where(m > 0, -np.inf, 0.0)


@dataclass
class GPTConfig:
    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_kv_heads: int = 2  # GQA: KV 头数 < Q 头数
    n_layers: int = 2
    d_ff: int = 128
    max_seq: int = 64
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    seed: int = 0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def kv_rep(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads


class GPT:
    """解码器-only Transformer。参数以 dict[str, Tensor] 暴露, 便于并行策略分片。"""

    def __init__(self, cfg: GPTConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        d, hd, nh, nkv, V = cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size
        self.p: dict[str, ag.Tensor] = {}

        def w(*shape: int, std: float | None = None) -> ag.Tensor:
            s = std if std is not None else np.sqrt(2.0 / shape[0])
            return ag.Tensor(rng.standard_normal(shape) * s, requires_grad=True)

        self.p["tok_emb"] = ag.Tensor(
            rng.standard_normal((V, d)) * 0.02, requires_grad=True, name="tok_emb"
        )
        for i in range(cfg.n_layers):
            pf = f"L{i}."
            self.p[pf + "attn_norm"] = ag.Tensor(np.ones(d), requires_grad=True)
            self.p[pf + "wq"] = w(d, nh * hd)
            self.p[pf + "wk"] = w(d, nkv * hd)
            self.p[pf + "wv"] = w(d, nkv * hd)
            self.p[pf + "wo"] = w(nh * hd, d, std=np.sqrt(2.0 / (nh * hd)))
            self.p[pf + "ffn_norm"] = ag.Tensor(np.ones(d), requires_grad=True)
            self.p[pf + "w_gate"] = w(d, cfg.d_ff)
            self.p[pf + "w_up"] = w(d, cfg.d_ff)
            self.p[pf + "w_down"] = w(cfg.d_ff, d, std=np.sqrt(2.0 / cfg.d_ff))
        self.p["final_norm"] = ag.Tensor(np.ones(d), requires_grad=True)
        if not cfg.tie_embeddings:
            self.p["lm_head"] = w(d, V)
        self._mask_cache: dict[int, np.ndarray] = {}

    # -- 参数 ------------------------------------------------------------- #
    @property
    def params(self) -> dict[str, ag.Tensor]:
        return self.p

    def n_params(self) -> int:
        return int(sum(t.data.size for t in self.p.values()))

    def zero_grad(self) -> None:
        for t in self.p.values():
            t.zero_grad()

    # -- 前向 ------------------------------------------------------------- #
    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None):
        """idx: (B, T) int; 返回 (logits, loss|None)。"""
        cfg = self.cfg
        B, T = idx.shape
        hd, nh, nkv, rep = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.kv_rep

        if T not in self._mask_cache:
            self._mask_cache[T] = causal_mask(T).reshape(1, 1, T, T)  # 广播到 (B,nh,T,T)
        mask = self._mask_cache[T]

        x = ag.embedding(self.p["tok_emb"], idx)  # (B,T,d)
        for i in range(cfg.n_layers):
            pf = f"L{i}."
            # ---- 注意力 (Pre-RMSNorm + RoPE + GQA) ---- #
            h = ag.rms_norm(x, self.p[pf + "attn_norm"])
            q = ag.matmul(h.reshape(B * T, cfg.d_model), self.p[pf + "wq"]).reshape(B, T, nh, hd)
            k = ag.matmul(h.reshape(B * T, cfg.d_model), self.p[pf + "wk"]).reshape(B, T, nkv, hd)
            v = ag.matmul(h.reshape(B * T, cfg.d_model), self.p[pf + "wv"]).reshape(B, T, nkv, hd)

            # 先置换到 (B,nh,T,hd) 再做 RoPE —— rope 把**倒数第二维**当序列长度,
            # 若直接在 (B,T,nh,hd) 上做会错误地把 head 数当成 T (曾真实踩坑)。
            qh = ag.rope(q.transpose(0, 2, 1, 3), cfg.rope_theta)
            kh = ag.rope(k.transpose(0, 2, 1, 3), cfg.rope_theta)
            vh = v.transpose(0, 2, 1, 3)

            if rep > 1:  # GQA: 把 KV 头复制到 Q 的头数 (head 维现在是 axis=1)
                kh = _repeat_kv(kh, rep)
                vh = _repeat_kv(vh, rep)

            scores = ag.bmm(qh, kh.transpose(0, 1, 3, 2)) / np.sqrt(hd)  # (B,nh,T,T)
            scores = ag.add(scores, ag.Tensor(mask))
            att = ag.softmax(scores, axis=-1)
            o = ag.bmm(att, vh)  # (B,nh,T,hd)
            o = o.transpose(0, 2, 1, 3).reshape(B * T, nh * hd)  # numpy transpose on Tensor?
            o = ag.matmul(o, self.p[pf + "wo"]).reshape(B, T, cfg.d_model)
            x = ag.add(x, o)

            # ---- FFN (SwiGLU) ---- #
            h2 = ag.rms_norm(x, self.p[pf + "ffn_norm"]).reshape(B * T, cfg.d_model)
            g = ag.matmul(h2, self.p[pf + "w_gate"])
            u = ag.matmul(h2, self.p[pf + "w_up"])
            ff = ag.matmul(ag.mul(ag.silu(g), u), self.p[pf + "w_down"]).reshape(B, T, cfg.d_model)
            x = ag.add(x, ff)

        xn = ag.rms_norm(x, self.p["final_norm"]).reshape(B * T, cfg.d_model)
        head = self.p["tok_emb"] if self.cfg.tie_embeddings else self.p["lm_head"]
        logits = ag.matmul(xn, ag.transpose2d(head)).reshape(B, T, cfg.vocab_size)

        loss = ag.cross_entropy(logits.reshape(B * T, cfg.vocab_size), targets.reshape(-1)) if targets is not None else None
        return logits, loss


def _repeat_kv(kv, rep: int):
    """把 (B,nkv,T,hd) 的 KV 沿 head 维 (axis=1) 重复 rep 次 → (B,nkv*rep,T,hd)。"""
    data = np.repeat(kv.data, rep, axis=1)
    out = ag.Tensor(data, requires_grad=kv.requires_grad)
    ag._CHILD_REG[id(out)] = (kv,)

    def _bw() -> None:
        if kv.requires_grad:
            B, H, T, hd = out.grad.shape
            kv.grad += out.grad.reshape(B, H // rep, rep, T, hd).sum(axis=2)

    out._backward = _bw
    return out
