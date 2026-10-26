"""
tianquan.serve.engine — 带 KV 缓存的增量推理引擎

与训练用的 ``train.GPT`` 分离: 推理不需要计算图, 直接用 NumPy 走
**prefill / decode 两段式**, 这是所有生产级推理引擎 (vLLM / TGI / TensorRT-LLM)
的标准形态:

* prefill: 一次性算完 prompt 所有位置的 K/V 并写入缓存, 得到首个 logits
* decode : 每步只用最后 1 个 token 做 Q, 从缓存里读 K/V 做注意力
           ⇒ 单步复杂度从 O(T²) 降到 O(T)

架构件: RMSNorm / RoPE / GQA / SwiGLU —— 与训练模型逐位一致 (可交叉验证)。
"""

from __future__ import annotations

import numpy as np

from ..train.model import GPT, GPTConfig

__all__ = ["InferenceEngine", "sample"]


def _rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def _rope(x: np.ndarray, theta: float = 10000.0, start: int = 0) -> np.ndarray:
    """x: (..., T, hd), 最后一维为偶数。"""
    hd = x.shape[-1]
    half = hd // 2
    T = x.shape[-2]
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    ang = np.outer(np.arange(start, start + T, dtype=np.float64), inv)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


class InferenceEngine:
    """纯推理引擎。权重从训练模型复制而来, 保证与训练端一致。"""

    def __init__(self, cfg: GPTConfig, seed: int = 0, source: GPT | None = None) -> None:
        self.cfg = cfg
        model = source if source is not None else GPT(GPTConfig(**{**cfg.__dict__, "seed": seed}))
        self.w = {k: v.data.copy() for k, v in model.params.items()}
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_rep = cfg.kv_rep

    # ------------------------------------------------------------------ #
    def prefill(self, tokens: np.ndarray, cache, seq_id: int) -> np.ndarray:
        """写入 prompt 的全部 K/V, 返回最后一个位置的 logits。"""
        cfg = self.cfg
        T = len(tokens)
        x = self.w["tok_emb"][tokens]  # (T, d)
        for li in range(cfg.n_layers):
            pf = f"L{li}."
            h = _rmsnorm(x, self.w[pf + "attn_norm"])
            q = (h @ self.w[pf + "wq"]).reshape(T, cfg.n_heads, self.head_dim)
            k = (h @ self.w[pf + "wk"]).reshape(T, self.n_kv_heads, self.head_dim)
            v = (h @ self.w[pf + "wv"]).reshape(T, self.n_kv_heads, self.head_dim)
            # _rope 把倒数第二维当序列长度 ⇒ 先换成 (H,T,hd) 再换回来
            q = _rope(q.transpose(1, 0, 2), cfg.rope_theta).transpose(1, 0, 2)
            k = _rope(k.transpose(1, 0, 2), cfg.rope_theta).transpose(1, 0, 2)
            for t in range(T):
                cache.put(seq_id, li, t, k[t], v[t])
            x = x + self._attend(q, k, v, causal=True) @ self.w[pf + "wo"]

            h2 = _rmsnorm(x, self.w[pf + "ffn_norm"])
            ff = ((h2 @ self.w[pf + "w_gate"]) * (1.0 / (1.0 + np.exp(-(h2 @ self.w[pf + "w_gate"])))))
            ff = (ff * (h2 @ self.w[pf + "w_up"])) @ self.w[pf + "w_down"]
            x = x + ff
        xn = _rmsnorm(x, self.w["final_norm"])
        return xn[-1] @ self.w["tok_emb"].T

    def decode(self, token: int, pos: int, cache, seq_id: int) -> np.ndarray:
        """单步解码: 只算当前位置, K/V 从缓存读取。"""
        cfg = self.cfg
        x = self.w["tok_emb"][token]  # (d,)
        for li in range(cfg.n_layers):
            pf = f"L{li}."
            h = _rmsnorm(x, self.w[pf + "attn_norm"])
            q = (h @ self.w[pf + "wq"]).reshape(cfg.n_heads, self.head_dim)
            k = (h @ self.w[pf + "wk"]).reshape(self.n_kv_heads, self.head_dim)
            v = (h @ self.w[pf + "wv"]).reshape(self.n_kv_heads, self.head_dim)
            # decode: 单步, 形状 (H,1,hd) —— 倒数第二维为 T=1
            q = _rope(q[:, None, :], cfg.rope_theta, start=pos)[:, 0]
            k = _rope(k[:, None, :], cfg.rope_theta, start=pos)[:, 0]
            cache.put(seq_id, li, pos, k, v)

            K, V = cache.get(seq_id, li, pos + 1)  # (pos+1, n_kv, hd)
            K = np.repeat(K, self.kv_rep, axis=1) if self.kv_rep > 1 else K
            V = np.repeat(V, self.kv_rep, axis=1) if self.kv_rep > 1 else V
            scale = 1.0 / np.sqrt(self.head_dim)
            o = np.empty((cfg.n_heads, self.head_dim))
            for hi in range(cfg.n_heads):
                s = K[:, hi] @ q[hi] * scale  # (pos+1,)
                s = s - s.max()
                e = np.exp(s)
                p = e / e.sum()
                o[hi] = p @ V[:, hi]
            x = x + o.reshape(-1) @ self.w[pf + "wo"]

            h2 = _rmsnorm(x, self.w[pf + "ffn_norm"])
            g = h2 @ self.w[pf + "w_gate"]
            ff = ((g * (1.0 / (1.0 + np.exp(-g)))) * (h2 @ self.w[pf + "w_up"])) @ self.w[pf + "w_down"]
            x = x + ff
        xn = _rmsnorm(x, self.w["final_norm"])
        return xn @ self.w["tok_emb"].T

    # ------------------------------------------------------------------ #
    def _attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
        """prefill 用的多头因果注意力, 返回 (T, n_heads*head_dim)。"""
        cfg = self.cfg
        T = q.shape[0]
        rep = self.kv_rep
        K = np.repeat(k, rep, axis=1)
        V = np.repeat(v, rep, axis=1)
        scale = 1.0 / np.sqrt(self.head_dim)
        # 注意: 不能用 triu(...)*(-inf) —— 0 * (-inf) = NaN, 会污染整行 softmax
        mask = np.where(np.triu(np.ones((T, T)), k=1) > 0, -np.inf, 0.0)
        out = np.empty((T, cfg.n_heads, self.head_dim))
        for hi in range(cfg.n_heads):
            s = q[:, hi] @ K[:, hi].T * scale + mask
            s = s - s.max(axis=-1, keepdims=True)
            e = np.exp(s)
            out[:, hi] = (e / e.sum(axis=-1, keepdims=True)) @ V[:, hi]
        return out.reshape(T, cfg.n_heads * self.head_dim)


# --------------------------------------------------------------------------- #
# 采样
# --------------------------------------------------------------------------- #
def sample(logits: np.ndarray, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, rng=None) -> int:
    """temperature + top-k (Fan et al. 2018) + top-p/nucleus (Holtzman et al. 2020)。

    top-p 的关键在于**先排序再按累积概率截断**, 而非按概率阈值直接过滤。
    """
    rng = rng or np.random.default_rng()
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits / temperature
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()

    idx = np.argsort(-p)
    if top_k > 0:
        idx = idx[:top_k]
        p = p[idx] / p[idx].sum()
        return int(idx[rng.choice(len(idx), p=p)])

    sp = p[idx]
    c = np.cumsum(sp)
    keep = np.searchsorted(c, top_p) + 1
    idx = idx[: max(keep, 1)]
    q = p[idx]
    q = q / q.sum()
    return int(idx[rng.choice(len(idx), p=q)])
