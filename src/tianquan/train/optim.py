"""
tianquan.train.optim — 优化器 (L3)

* ``AdamW``  — Loshchilov & Hutter 2019 (解耦权重衰减), 工业默认
* ``Muon``   — Jordan et al. 2024: 对矩阵参数用 Newton-Schulz 迭代做
               近似正交化后再更新, 在 Transformer 上普遍优于 AdamW。
               本项目保留其**数学核心** (NS 迭代的系数即为原论文推荐值),
               并让 Adam 负责标量/向量参数与偏置。
* ``SGD``    — 带动量, 作为对照基线
"""

from __future__ import annotations

import numpy as np

from . import autograd as ag

__all__ = ["AdamW", "Muon", "SGD"]


class _Base:
    def __init__(self, params: dict[str, ag.Tensor]) -> None:
        self.params = params
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params.values():
            p.zero_grad()

    def step(self) -> None:  # pragma: no cover
        raise NotImplementedError


class AdamW(_Base):
    """解耦权重衰减的 Adam。不变量: 无梯度时步长≈0; lr=0 时参数不变。"""

    def __init__(
        self,
        params: dict[str, ag.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in params.items()}

    def step(self) -> None:
        self.t += 1
        for k, p in self.params.items():
            if p.grad is None:
                continue
            g = p.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1**self.t)
            vhat = self.v[k] / (1 - self.b2**self.t)
            p.data -= self.lr * (mhat / (np.sqrt(vhat) + self.eps) + self.wd * p.data)


class SGD(_Base):
    """带动量的 SGD (对照基线)。"""

    def __init__(self, params: dict[str, ag.Tensor], lr: float = 0.1, momentum: float = 0.9) -> None:
        super().__init__(params)
        self.lr, self.mu = lr, momentum
        self.buf = {k: np.zeros_like(p.data) for k, p in params.items()}

    def step(self) -> None:
        for k, p in self.params.items():
            if p.grad is None:
                continue
            self.buf[k] = self.mu * self.buf[k] + p.grad
            p.data -= self.lr * self.buf[k]


class Muon(_Base):
    """Muon (Jordan et al. 2024): 矩阵参数走 Newton-Schulz 正交化动量。

    NS 迭代 (系数取原论文推荐 a=3.4445, b=-4.7750, c=2.0315):
        X <- aX + b(XXᵀ)X + c(XXᵀ)²X
    该迭代在奇异值 ∈ (0,1] 时收敛到最近的半正交矩阵。
    """

    NS_COEFFS = (3.4445, -4.7750, 2.0315)
    NS_STEPS = 5

    def __init__(
        self,
        params: dict[str, ag.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        adam_lr: float = 1e-3,
    ) -> None:
        super().__init__(params)
        self.lr, self.mu, self.wd, self.adam_lr = lr, momentum, weight_decay, adam_lr
        self.buf = {k: np.zeros_like(p.data) for k, p in params.items()}
        self._adam = AdamW(params, lr=adam_lr, weight_decay=weight_decay)

    def step(self) -> None:
        for k, p in self.params.items():
            if p.grad is None:
                continue
            if p.data.ndim >= 2 and min(p.data.shape) > 1:
                self.buf[k] = self.mu * self.buf[k] + p.grad
                upd = _newton_schulz(self.buf[k], self.NS_COEFFS, self.NS_STEPS)
                # 尺度对齐: 保持与 Adam 相近的更新量级 (原论文的经验缩放)
                scale = max(1.0, p.data.shape[0] / max(p.data.shape[1], 1)) ** 0.5
                p.data -= self.lr * upd * scale
                p.data -= self.lr * self.wd * p.data
            else:
                # 标量/向量参数交给 Adam
                self._adam.m[k] = 0.9 * self._adam.m[k] + 0.1 * p.grad
                self._adam.v[k] = 0.999 * self._adam.v[k] + 0.001 * (p.grad * p.grad)
                p.data -= self.adam_lr * self._adam.m[k] / (np.sqrt(self._adam.v[k]) + 1e-8)


def _newton_schulz(g: np.ndarray, coeffs: tuple[float, float, float], steps: int) -> np.ndarray:
    """Newton-Schulz 正交化 (对梯度矩阵), 使更新方向近似半正交。"""
    a, b, c = coeffs
    x = g / (np.linalg.norm(g) + 1e-12)
    # 保证行数 <= 列数, 便于 X Xᵀ 计算
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_ = x @ x.T
        b_ = b * a_ + c * (a_ @ a_)
        x = a * x + b_ @ x
    if transposed:
        x = x.T
    return x
