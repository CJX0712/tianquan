"""
tianquan.parallel.pipeline — 流水并行调度 (GPipe vs 1F1B)

零重写声明
----------
复刻两类经典调度, 并用**离散事件模拟**实测它们的 makespan 与峰值激活:

* **GPipe** (Huang et al., NeurIPS 2019)
  先把 m 个 microbatch 的前向全部流水做完, 再统一反向。
  代价: 峰值激活 O(m) —— microbatch 越多越吃显存。

* **1F1B / PipeDream-Flush** (Narayanan et al., ICML 2021)
  warmup 若干 F 后进入"一个前向、一个反向"的稳态, 把激活数压到 O(p)。
  Megatron-LM / DeepSpeed 的默认选择。

理论对照 (两种调度的 bubble 时间相同, 差别全在**内存**):

    makespan ≈ (m + p - 1) · (t_f + t_b)
    bubble fraction ≈ (p - 1) / (m + p - 1)   ← 与 m 成反比, 故 m 要取大
    peak activations:  GPipe ≈ m ,  1F1B ≈ p

本项目把这些"教科书结论"变成**可运行的模拟 + 可断言的数字**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PipelineConfig", "gpipe_schedule", "onef1b_schedule", "simulate", "ScheduleResult"]


@dataclass
class PipelineConfig:
    n_stages: int = 4
    n_micro: int = 8
    t_forward: float = 1.0
    t_backward: float = 2.0  # 反向通常约为前向的 2 倍


@dataclass
class ScheduleResult:
    makespan: float = 0.0
    bubble_fraction: float = 0.0
    peak_activations: int = 0  # 全流水线同时存活的激活总数
    peak_per_stage: int = 0  # 单个 stage 的峰值激活 (论文口径: GPipe=m, 1F1B=p)
    stage_busy: list[float] = field(default_factory=list)
    n_events: int = 0

    def as_dict(self) -> dict:
        return {
            "makespan": round(self.makespan, 4),
            "bubble_fraction": round(self.bubble_fraction, 6),
            "peak_activations": self.peak_activations,
            "peak_per_stage": self.peak_per_stage,
            "n_events": self.n_events,
        }


# --------------------------------------------------------------------------- #
# 调度生成: 返回 per-stage 的本地事件序列 [(micro, kind), ...]
# --------------------------------------------------------------------------- #
def gpipe_schedule(p: int, m: int) -> list[list[tuple[int, str]]]:
    """GPipe: 先全部前向, 再全部反向 (反向按 micro 逆序)。"""
    per_stage = []
    for _s in range(p):
        events = [(i, "F") for i in range(m)]
        events += [(i, "B") for i in reversed(range(m))]
        per_stage.append(events)
    return per_stage


def onef1b_schedule(p: int, m: int) -> list[list[tuple[int, str]]]:
    """1F1B (PipeDream-Flush): warmup → 稳态 1F1B → cooldown。

    stage s 的 warmup 数 = min(m, p - 1 - s): 越靠后的 stage 需要的前置越少。
    """
    per_stage = []
    for s in range(p):
        w = min(m, p - 1 - s)
        events: list[tuple[int, str]] = []
        for i in range(w):  # warmup: 只做前向
            events.append((i, "F"))
        for i in range(m - w):  # steady: 1F1B
            events.append((i + w, "F"))
            events.append((i, "B"))
        for i in range(m - w, m):  # cooldown: 只做反向
            events.append((i, "B"))
        per_stage.append(events)
    return per_stage


# --------------------------------------------------------------------------- #
# 离散事件模拟
# --------------------------------------------------------------------------- #
def simulate(cfg: PipelineConfig, kind: str = "1f1b") -> ScheduleResult:
    """按依赖约束模拟调度, 得出 makespan / bubble / 峰值激活。"""
    p, m = cfg.n_stages, cfg.n_micro
    tf, tb = cfg.t_forward, cfg.t_backward
    per_stage = onef1b_schedule(p, m) if kind == "1f1b" else gpipe_schedule(p, m)

    finish: dict[tuple[int, int, str], float] = {}
    stage_free = [0.0] * p
    ptr = [0] * p
    live: list[set[int]] = [set() for _ in range(p)]
    peak = 0
    stage_peak = [0] * p
    total_events = sum(len(e) for e in per_stage)

    while any(ptr[s] < len(per_stage[s]) for s in range(p)):
        progressed = False
        for s in range(p):
            if ptr[s] >= len(per_stage[s]):
                continue
            micro, k = per_stage[s][ptr[s]]
            deps: list[float] = []
            if k == "F":
                if s > 0:
                    t = finish.get((s - 1, micro, "F"))
                    if t is None:
                        continue
                    deps.append(t)
            else:
                t = finish.get((s, micro, "F"))
                if t is None:
                    continue
                deps.append(t)
                if s < p - 1:
                    t2 = finish.get((s + 1, micro, "B"))
                    if t2 is None:
                        continue
                    deps.append(t2)

            start = max([stage_free[s]] + deps)
            dur = tf if k == "F" else tb
            end = start + dur
            finish[(s, micro, k)] = end
            stage_free[s] = end
            if k == "F":
                live[s].add(micro)
                peak = max(peak, sum(len(x) for x in live))
                stage_peak[s] = max(stage_peak[s], len(live[s]))
            else:
                live[s].discard(micro)
            ptr[s] += 1
            progressed = True
        if not progressed:  # pragma: no cover
            raise RuntimeError("流水调度出现死锁 (依赖不成环检查失败)")

    makespan = max(stage_free)
    # 理想: 每个 stage 满负荷工作 m*(tf+tb), 全部并行 ⇒ 下界为 m*(tf+tb)
    ideal = m * (tf + tb)
    # bubble 口径取论文定义 (Narayanan et al. 2021): 空闲时间占 **总 makespan** 的比例
    #   bubble = (makespan - ideal) / makespan ≈ (p-1) / (m + p - 1)
    bubble = max(0.0, (makespan - ideal) / max(makespan, 1e-12))
    return ScheduleResult(
        makespan=makespan,
        bubble_fraction=bubble,
        peak_activations=peak,
        peak_per_stage=max(stage_peak),
        stage_busy=stage_free,
        n_events=total_events,
    )
