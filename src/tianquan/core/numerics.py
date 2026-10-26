"""
tianquan.core.numerics — 数值稳定原语层 (L0)

零重写声明
----------
本模块不发明任何新数值方法, 只把已被工业界验证的**稳定写法**固化为单一入口,
并给每一条附上"可被独立复现的硬不变量":

* logsumexp        — 经典 max-shift 技巧, 避免 exp 上溢/下溢
* softmax / log_softmax — 基于 logsumexp, 保证行和恒为 1 (机器精度)
* sigmoid          — 分段稳定写法, 避免 x<<0 时 exp(-x) 上溢
* rms_norm / layer_norm — LLaMA / Pre-LN Transformer 标配
* rope             — Su et al. 2021 (RoFormer); 旋转矩阵保范数
* swiglu / gelu    — Shazeer 2020 (SwiGLU) / Hendrycks & Gimpel 2016 (GELU)

每条不变量都在 tests/test_numerics.py 中被独立实现的"另一条路径"交叉验证。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "logsumexp",
    "softmax",
    "log_softmax",
    "sigmoid",
    "gelu",
    "swiglu",
    "rms_norm",
    "layer_norm",
    "rope",
    "causal_mask",
    "set_determinism",
]


# --------------------------------------------------------------------------- #
# 稳定性原语
# --------------------------------------------------------------------------- #
def logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """数值稳定的 log(sum(exp(a)))。

    恒等式: logsumexp(a) = m + log(sum(exp(a - m))),  m = max(a)
    该写法对任意有限 a 都不上溢 (因为 exp(a-m) <= 1)。
    """
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis) if axis is not None else out
    return out


def softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """稳定 softmax。不变量: sum(softmax(a), axis) == 1 (误差 < 1e-12)。"""
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(a - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """log-softmax。不变量: exp(log_softmax(a)) == softmax(a)。"""
    a = np.asarray(a, dtype=np.float64)
    return a - logsumexp(a, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """分段稳定 sigmoid: x>=0 用 1/(1+e^-x), x<0 用 e^x/(1+e^x)。

    不变量: sigmoid(x) + sigmoid(-x) == 1 (机器精度); 值域 (0,1)。
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU (tanh 近似, Hendrycks & Gimpel 2016)。不变量: 单调; GELU(0)=0。"""
    x = np.asarray(x, dtype=np.float64)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def swiglu(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """SwiGLU (Shazeer 2020): silu(w) * v, silu(x)=x*sigmoid(x)。"""
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return (w * sigmoid(w)) * v


# --------------------------------------------------------------------------- #
# 归一化
# --------------------------------------------------------------------------- #
def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """RMSNorm (Zhang & Sennrich 2019), LLaMA 系列标配。

    不变量: rms_norm(a*x) == rms_norm(x) 对 a>0 成立 (尺度不变性)。
    """
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * np.asarray(weight, dtype=np.float64)


def layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None, eps: float = 1e-8
) -> np.ndarray:
    """Pre-LN LayerNorm。不变量: 输出沿最后一维均值≈0、方差≈1 (加权前)。"""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    y = y * np.asarray(weight, dtype=np.float64)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y


def rope(x: np.ndarray, theta: float = 10000.0, start: int = 0) -> np.ndarray:
    """旋转位置编码 RoPE (Su et al. 2021)。

    把相邻两维视为复数 (x_{2i}, x_{2i+1}), 乘以 e^{i * pos * theta^{-2i/d}}。

    不变量 (硬): 旋转是正交变换 ⇒ 每个 pair 的 L2 范数逐位不变。
    """
    x = np.asarray(x, dtype=np.float64)
    *lead, d = x.shape
    assert d % 2 == 0, "RoPE 要求最后一维为偶数"
    half = d // 2
    seq = x.shape[-2] if len(lead) > 0 else 1

    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(start, start + seq, dtype=np.float64)
    angles = np.outer(pos, inv_freq)  # (seq, half)
    cos, sin = np.cos(angles), np.sin(angles)

    x_r = x.reshape(-1, seq, half, 2)
    x1, x2 = x_r[..., 0], x_r[..., 1]
    out = np.empty_like(x_r)
    out[..., 0] = x1 * cos - x2 * sin
    out[..., 1] = x2 * cos + x1 * sin
    return out.reshape(x.shape)


def causal_mask(n: int) -> np.ndarray:
    """下三角因果掩码 (1=可见)。不变量: 矩阵下三角全 1, 严格上三角全 0。"""
    return np.tril(np.ones((n, n), dtype=np.float64))


def set_determinism(seed: int = 0) -> None:
    """固定随机种子, 保证 bit 级可复现。"""
    np.random.seed(seed)
