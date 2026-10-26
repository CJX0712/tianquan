"""
tianquan.train.autograd — 精简反向模式自动微分 (L3)

零重写声明
----------
这是 **micrograd (Karpathy) 架构** 的 NumPy 张量化版本: 动态计算图 +
拓扑逆序遍历 + 链式法则。之所以不引入 PyTorch, 是为了让整仓在
"无 GPU、无重型依赖" 的前提下仍可复现, 且每个算子的梯度都能被
**中心差分**独立验证 (tests/test_autograd.py, 目标 ~1e-7)。

设计取舍
--------
* 只实现 GPT 真正用到的算子, 不为通用性牺牲可读性
* 矩阵乘统一走 2D (batch 维度靠 reshape 折叠) —— 反向公式简单且不易错
* 梯度累加用 ``+=``, 支持同一张量被多次复用 (如 KV cache 共享权重)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Tensor",
    "matmul",
    "bmm",
    "add",
    "sub",
    "mul",
    "scale",
    "sum_all",
    "exp",
    "log",
    "silu",
    "gelu",
    "softmax",
    "rms_norm",
    "rope",
    "embedding",
    "cross_entropy",
    "reshape",
    "transpose",
    "transpose2d",
    "concat",
]


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "name")

    def __init__(
        self, data, requires_grad: bool = False, _children: tuple = (), name: str = ""
    ) -> None:
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64, copy=False)
        else:
            self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(id(c) for c in _children)
        self.name = name

    # -- 反向入口 --------------------------------------------------------- #
    def backward(self) -> None:
        topo: list[Tensor] = []
        seen: set[int] = set()

        def build(v: Tensor) -> None:
            if id(v) not in seen:
                seen.add(id(v))
                for c in _children_of(v):
                    build(c)
                topo.append(v)

        build(self)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    # -- 便捷 ------------------------------------------------------------- #
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def item(self) -> float:
        return float(self.data)

    def reshape(self, *shape) -> "Tensor":
        return reshape(self, shape)

    def transpose(self, *axes) -> "Tensor":
        return transpose(self, axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes)

    def __add__(self, other):
        return add(self, other if isinstance(other, Tensor) else Tensor(other))

    def __mul__(self, other):
        return mul(self, other if isinstance(other, Tensor) else Tensor(other))

    def __sub__(self, other):
        return sub(self, other if isinstance(other, Tensor) else Tensor(other))

    def __truediv__(self, k):
        return scale(self, 1.0 / float(k))

    def __neg__(self):
        return scale(self, -1.0)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return sub(other if isinstance(other, Tensor) else Tensor(other), self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __repr__(self) -> str:
        return f"Tensor({self.shape}, name={self.name or '-'})"


# 由于 __slots__ 不含弱引用集合, 用辅助字典保存真实子节点引用
_CHILD_REG: dict[int, tuple] = {}


def _children_of(v: Tensor) -> tuple:
    return _CHILD_REG.get(id(v), ())


def _wrap(out_data: np.ndarray, a: Tensor, b: Tensor | None = None, name: str = "") -> Tensor:
    kids = (a,) if b is None else (a, b)
    t = Tensor(out_data, requires_grad=a.requires_grad or (b is not None and b.requires_grad))
    _CHILD_REG[id(t)] = kids
    t.name = name
    return t


class _NoGrad:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *exc) -> None:
        pass


def no_grad() -> _NoGrad:
    return _NoGrad()


# --------------------------------------------------------------------------- #
# 算子
# --------------------------------------------------------------------------- #
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """2D 矩阵乘。batch 维度请在外层 reshape 折叠。"""
    out = _wrap(a.data @ b.data, a, b, "matmul")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad @ b.data.T
        if b.requires_grad:
            b.grad += a.data.T @ out.grad

    out._backward = _bw
    return out


def bmm(a: Tensor, b: Tensor) -> Tensor:
    """批量矩阵乘: (..., M, K) @ (..., K, N) -> (..., M, N)。

    反向用 ``swapaxes(-1,-2)`` 而非 ``.T`` —— 后者对 3D+ 会反转**所有**轴, 是错的。
    """
    out = _wrap(a.data @ b.data, a, b, "bmm")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad @ b.data.swapaxes(-1, -2)
        if b.requires_grad:
            b.grad += a.data.swapaxes(-1, -2) @ out.grad

    out._backward = _bw
    return out


def rope(a: Tensor, theta: float = 10000.0) -> Tensor:
    """旋转位置编码 RoPE (Su et al. 2021)。输入最后一维为 head_dim (偶数)。

    旋转是正交变换 ⇒ **反向就是逆旋转** (把角度取负), 无需存储中间量。
    """
    d = a.data.shape[-1]
    half = d // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    T = a.data.shape[-2]
    ang = np.outer(np.arange(T, dtype=np.float64), inv)  # (T, half)
    cos, sin = np.cos(ang), np.sin(ang)

    x1, x2 = a.data[..., 0::2], a.data[..., 1::2]
    out_data = np.empty_like(a.data)
    out_data[..., 0::2] = x1 * cos - x2 * sin
    out_data[..., 1::2] = x2 * cos + x1 * sin
    out = _wrap(out_data, a, None, "rope")

    def _bw() -> None:
        if a.requires_grad:
            g1, g2 = out.grad[..., 0::2], out.grad[..., 1::2]
            a.grad[..., 0::2] += g1 * cos + g2 * sin
            a.grad[..., 1::2] += g2 * cos - g1 * sin

    out._backward = _bw
    return out


def add(a: Tensor, b: Tensor) -> Tensor:
    out = _wrap(a.data + b.data, a, b, "add")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += _reduce_to(out.grad, a.data.shape)
        if b.requires_grad:
            b.grad += _reduce_to(out.grad, b.data.shape)

    out._backward = _bw
    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    out = _wrap(a.data - b.data, a, b, "sub")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += _reduce_to(out.grad, a.data.shape)
        if b.requires_grad:
            b.grad -= _reduce_to(out.grad, b.data.shape)

    out._backward = _bw
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = _wrap(a.data * b.data, a, b, "mul")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += _reduce_to(out.grad * b.data, a.data.shape)
        if b.requires_grad:
            b.grad += _reduce_to(out.grad * a.data, b.data.shape)

    out._backward = _bw
    return out


def sum_all(a: Tensor) -> Tensor:
    """全元素求和 → 标量。反向为全 1 广播。"""
    out = _wrap(np.array(a.data.sum()), a, None, "sum_all")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += np.full(a.data.shape, out.grad.item())

    out._backward = _bw
    return out


def scale(a: Tensor, k: float) -> Tensor:
    out = _wrap(a.data * k, a, None, "scale")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad * k

    out._backward = _bw
    return out


def exp(a: Tensor) -> Tensor:
    out = _wrap(np.exp(a.data), a, None, "exp")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad * out.data

    out._backward = _bw
    return out


def log(a: Tensor) -> Tensor:
    out = _wrap(np.log(a.data), a, None, "log")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad / a.data

    out._backward = _bw
    return out


def silu(a: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = _wrap(a.data * s, a, None, "silu")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad * (s + a.data * s * (1.0 - s))

    out._backward = _bw
    return out


def gelu(a: Tensor) -> Tensor:
    c = np.sqrt(2.0 / np.pi)
    t = np.tanh(c * (a.data + 0.044715 * a.data**3))
    out = _wrap(0.5 * a.data * (1.0 + t), a, None, "gelu")

    def _bw() -> None:
        if a.requires_grad:
            dt = (1 - t * t) * c * (1 + 3 * 0.044715 * a.data**2)
            a.grad += out.grad * (0.5 * (1 + t) + 0.5 * a.data * dt)

    out._backward = _bw
    return out


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    m = np.max(a.data, axis=axis, keepdims=True)
    e = np.exp(a.data - m)
    p = e / np.sum(e, axis=axis, keepdims=True)
    out = _wrap(p, a, None, "softmax")

    def _bw() -> None:
        if a.requires_grad:
            g = out.grad
            s = (g * p).sum(axis=axis, keepdims=True)
            a.grad += p * (g - s)

    out._backward = _bw
    return out


def rms_norm(a: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    rms = np.sqrt(np.mean(a.data * a.data, axis=-1, keepdims=True) + eps)
    xn = a.data / rms
    out = _wrap(xn * weight.data, a, weight, "rms_norm")

    def _bw() -> None:
        g = out.grad
        if weight.requires_grad:
            weight.grad += _reduce_to((g * xn).sum(axis=0), weight.data.shape)
        if a.requires_grad:
            # y = (x / rms) * w, rms = sqrt(mean(x^2) + eps)
            # ∂L/∂x = (1/rms) * [gx - xn * mean(gx * xn)],  其中 gx = g * w
            gx = g * weight.data
            n = a.data.shape[-1]
            a.grad += (gx - xn * (gx * xn).sum(axis=-1, keepdims=True) / n) / rms

    out._backward = _bw
    return out


def embedding(w: Tensor, ids: np.ndarray) -> Tensor:
    """查表: out[i] = w[ids[i]]。反向为 scatter-add。"""
    out = _wrap(w.data[ids], w, None, "embedding")

    def _bw() -> None:
        if w.requires_grad:
            np.add.at(w.grad, ids, out.grad)

    out._backward = _bw
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """融合 softmax + NLL: 数值稳定, 且反向就是 (p - onehot), 无需存中间概率。"""
    B, V = logits.data.shape
    m = np.max(logits.data, axis=-1, keepdims=True)
    logsum = m + np.log(np.sum(np.exp(logits.data - m), axis=-1, keepdims=True))
    logp = logits.data - logsum
    loss = -logp[np.arange(B), targets].mean()
    out = _wrap(np.array(loss), logits, None, "cross_entropy")

    def _bw() -> None:
        if logits.requires_grad:
            p = np.exp(logp)
            p[np.arange(B), targets] -= 1.0
            # 必须乘上游梯度 out.grad —— 否则当 loss 非根节点 (如被加权、被复用)
            # 时梯度会静默错误。这是链式法则最容易漏写的一处。
            logits.grad += (p / B) * out.grad.item()

    out._backward = _bw
    return out


def reshape(a: Tensor, shape) -> Tensor:
    shape = tuple(shape)
    src = a.data.shape
    out = _wrap(a.data.reshape(shape), a, None, "reshape")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad.reshape(src)

    out._backward = _bw
    return out


def transpose(a: Tensor, axes) -> Tensor:
    """多维轴置换。反向 = 用逆置换 ``argsort(axes)`` 把梯度换回去。"""
    axes = tuple(axes)
    inv = tuple(int(i) for i in np.argsort(axes))
    out = _wrap(a.data.transpose(axes), a, None, "transpose")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad.transpose(inv)

    out._backward = _bw
    return out


def transpose2d(a: Tensor) -> Tensor:
    out = _wrap(a.data.T, a, None, "transpose")

    def _bw() -> None:
        if a.requires_grad:
            a.grad += out.grad.T

    out._backward = _bw
    return out


def concat(parts: list[Tensor], axis: int = -1) -> Tensor:
    out_data = np.concatenate([p.data for p in parts], axis=axis)
    out = Tensor(out_data, requires_grad=any(p.requires_grad for p in parts))
    _CHILD_REG[id(out)] = tuple(parts)
    sizes = [p.data.shape[axis] for p in parts]

    def _bw() -> None:
        idx = 0
        for p, s in zip(parts, sizes):
            if p.requires_grad:
                sl = [slice(None)] * out.grad.ndim
                sl[axis] = slice(idx, idx + s)
                p.grad += out.grad[tuple(sl)]
            idx += s

    out._backward = _bw
    return out


def _reduce_to(g: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """把广播后的梯度还原到 shape (对多余维度/被广播维度求和)。"""
    if g.shape == shape:
        return g
    ndim_extra = g.ndim - len(shape)
    if ndim_extra > 0:
        g = g.sum(axis=tuple(range(ndim_extra)))
    axis = tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1)
    if axis:
        g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)
