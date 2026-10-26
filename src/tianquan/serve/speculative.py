"""
tianquan.serve.speculative — 投机解码 (Speculative Decoding)

零重写声明
----------
严格实现 **Leviathan et al., ICML 2023** 的算法与其核心定理:

    用小 draft 模型先猜 γ 个 token, 再用大 target 模型**一次并行前向**
    验证这 γ 个位置, 按接受-拒绝采样逐个决定是否采纳:

        accept  x_i ~ min(1, p_target(x_i) / p_draft(x_i))
        reject  从 (p_target - p_draft)_+ 归一化后重采样, 并终止本轮

**定理 (分布一致性)**: 上述过程输出的 token 分布与 target 模型
自回归采样**完全相同** —— 这是投机解码能保证"无损加速"的根本原因。
本项目把它做成可统计验证的门禁 (tests/test_serve.py, 大样本 TVD 检验)。

加速机理
--------
target 模型每轮只做 1 次前向 (验证 γ+1 个位置, 可并行), 而标准解码
每生成 1 个 token 就要 1 次 target 前向。故 target 前向次数从 N 降到
N / E[每轮接受数]。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SpecStats", "speculative_decode", "standard_decode"]


@dataclass
class SpecStats:
    n_tokens: int = 0
    n_rounds: int = 0  # target 模型的前向轮数 (每轮 1 次并行验证)
    n_draft_calls: int = 0
    n_accepted: int = 0
    acceptance_rate: float = 0.0
    target_step_reduction: float = 0.0  # 标准解码步数 / 投机解码轮数

    def as_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "n_rounds": self.n_rounds,
            "n_draft_calls": self.n_draft_calls,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "target_step_reduction": round(self.target_step_reduction, 4),
        }


def _probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        p = np.zeros_like(logits)
        p[int(np.argmax(logits))] = 1.0
        return p
    z = logits / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def speculative_decode(
    target_next,
    draft_next,
    prefix: list[int],
    n_tokens: int,
    gamma: int = 4,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], SpecStats]:
    """投机解码。``*_next(prefix_tokens) -> logits``。

    ``target_next`` 每轮被调用一次, 但一次要能给出 γ+1 个位置的条件分布
    (真实引擎里靠一次并行前向实现; 这里用回调 ``target_next(prefix, k)``
    表示"已知 prefix 与已追加的 k 个 draft token 后的分布")。
    """
    rng = rng or np.random.default_rng(0)
    st = SpecStats()
    out: list[int] = []
    cur = list(prefix)

    while len(out) < n_tokens:
        want = min(gamma, n_tokens - len(out))
        # ---- 阶段 1: draft 自回归猜 want 个 token ---- #
        drafts: list[int] = []
        q_probs: list[np.ndarray] = []
        for _ in range(want):
            q = _probs(draft_next(cur + drafts), temperature)
            x = int(rng.choice(len(q), p=q))
            drafts.append(x)
            q_probs.append(q)
            st.n_draft_calls += 1

        # ---- 阶段 2: target 并行验证 want+1 个位置 ---- #
        tgts = [_probs(target_next(cur + drafts[:i]), temperature) for i in range(want + 1)]
        st.n_rounds += 1

        # ---- 阶段 3: 接受-拒绝 ---- #
        n_acc = 0
        rejected = False
        for i in range(want):
            x = drafts[i]
            p, q = tgts[i], q_probs[i]
            if rng.random() < min(1.0, p[x] / max(q[x], 1e-12)):
                out.append(x)
                cur.append(x)
                n_acc += 1
            else:
                # 从 (p - q)_+ 重采样 —— 这一步保证了分布与目标完全一致
                diff = np.maximum(0.0, p - q)
                if diff.sum() <= 1e-12:
                    x_new = int(rng.choice(len(p), p=p))
                else:
                    x_new = int(rng.choice(len(diff), p=diff / diff.sum()))
                out.append(x_new)
                cur.append(x_new)
                rejected = True
                break

        if not rejected:
            # 全部接受 ⇒ 额外采样第 γ+1 个位置
            x = int(rng.choice(len(tgts[want]), p=tgts[want]))
            out.append(x)
            cur.append(x)
            n_acc += 1

        st.n_accepted += n_acc

    st.n_tokens = len(out)
    st.acceptance_rate = st.n_accepted / max(len(out), 1)
    st.target_step_reduction = len(out) / max(st.n_rounds, 1)
    return out, st


def standard_decode(
    target_next, prefix: list[int], n_tokens: int, temperature: float = 1.0, rng=None
) -> list[int]:
    """对照: 标准自回归解码 (每 token 一次 target 前向)。"""
    rng = rng or np.random.default_rng(0)
    out: list[int] = []
    cur = list(prefix)
    for _ in range(n_tokens):
        p = _probs(target_next(cur), temperature)
        x = int(rng.choice(len(p), p=p))
        out.append(x)
        cur.append(x)
    return out
