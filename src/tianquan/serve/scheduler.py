"""
tianquan.serve.scheduler — 连续批处理 vs 静态批处理

零重写声明
----------
复刻两类调度并给出**实测吞吐对比**:

* **Static batching (静态批处理)**
  组一批请求一起跑, 批次里只要还有一个请求没生成完, 整批就占着槽位
  (先完成的请求空转等待)。这是早期 TF Serving / Triton 的默认形态。

* **Continuous batching (迭代级调度)**
  **Orca (Yu et al., OSDI 2022)** 提出、vLLM 发扬: 以"一次 decode 迭代"为
  调度粒度 —— 每步都把已完成的请求移出、把新请求补进来, 槽位零空转。

衡量口径
--------
以"迭代步数"为时间单位 (一次 batch decode ≈ 一个时间步, 与 batch 大小弱相关),
从而把**调度策略本身**的收益隔离出来, 不受硬件波动干扰。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engine import InferenceEngine, sample
from .kv_cache import PagedKVCache

__all__ = ["Request", "ServeMetrics", "ContinuousScheduler", "StaticScheduler"]


@dataclass
class Request:
    id: int
    prompt: list[int]
    max_new: int
    arrival: int = 0


@dataclass
class ServeMetrics:
    steps: int = 0
    n_completed: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0  # Σ(完成步 - 到达步)
    ttft: list[int] = field(default_factory=list)  # 首 token 延迟 (步)
    throughput: float = 0.0  # tokens / step
    slot_utilization: float = 0.0

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "n_completed": self.n_completed,
            "total_tokens": self.total_tokens,
            "throughput": round(self.throughput, 4),
            "avg_latency": round(self.total_latency / max(self.n_completed, 1), 3),
            "avg_ttft": round(float(np.mean(self.ttft)) if self.ttft else 0.0, 3),
            "slot_utilization": round(self.slot_utilization, 4),
        }


class _BaseScheduler:
    def __init__(self, engine: InferenceEngine, cache: PagedKVCache, max_batch: int, seed: int = 0):
        self.engine = engine
        self.cache = cache
        self.max_batch = max_batch
        self.rng = np.random.default_rng(seed)

    def _prefill(self, req: Request) -> tuple[int, list[int]]:
        self.cache.create(req.id)
        tokens = np.array(req.prompt, dtype=np.int64)
        logits = self.engine.prefill(tokens, self.cache, req.id)
        nxt = sample(logits, rng=self.rng)
        return nxt, [nxt]


class ContinuousScheduler(_BaseScheduler):
    """迭代级调度: 每步重排 batch, 完成即出、新请求即入。"""

    def run(self, requests: list[Request]) -> tuple[dict[int, list[int]], ServeMetrics]:
        m = ServeMetrics()
        pending = list(requests)
        running: dict[int, dict] = {}
        results: dict[int, list[int]] = {}
        slot_used = 0
        step = 0

        while pending or running:
            # ---- admission: 只要有空槽就补新请求 ---- #
            while pending and len(running) < self.max_batch:
                req = pending.pop(0)
                if req.arrival > step:
                    pending.insert(0, req)
                    break
                nxt, out = self._prefill(req)
                running[req.id] = {"req": req, "pos": len(req.prompt), "tok": nxt, "out": out}
                m.ttft.append(step - req.arrival + 1)

            if not running:  # 所有请求都还没到
                step += 1
                continue

            # ---- 一步 decode: batch 内所有序列并行 ---- #
            done = []
            for sid, st in list(running.items()):
                req = st["req"]
                logits = self.engine.decode(st["tok"], st["pos"], self.cache, sid)
                nxt = sample(logits, rng=self.rng)
                st["out"].append(nxt)
                st["tok"] = nxt
                st["pos"] += 1
                if len(st["out"]) >= req.max_new:
                    done.append(sid)
            for sid in done:
                st = running.pop(sid)
                results[sid] = st["out"]
                self.cache.release(sid)
                m.n_completed += 1
                m.total_tokens += len(st["out"])
                m.total_latency += step - st["req"].arrival

            slot_used += len(running)
            step += 1
            if step > 100_000:  # pragma: no cover
                break

        m.steps = step
        m.throughput = m.total_tokens / max(step, 1)
        m.slot_utilization = slot_used / max(step * self.max_batch, 1)
        return results, m


class StaticScheduler(_BaseScheduler):
    """静态批处理: 批次凑满即启动, 整批全部完成才释放槽位。"""

    def run(self, requests: list[Request]) -> tuple[dict[int, list[int]], ServeMetrics]:
        m = ServeMetrics()
        pending = list(requests)
        results: dict[int, list[int]] = {}
        slot_used = 0
        step = 0

        while pending:
            batch = pending[: self.max_batch]
            pending = pending[self.max_batch :]
            running: dict[int, dict] = {}
            for req in batch:
                nxt, out = self._prefill(req)
                running[req.id] = {"req": req, "pos": len(req.prompt), "tok": nxt, "out": out}
                m.ttft.append(1)

            while running:
                done = []
                for sid, st in list(running.items()):
                    logits = self.engine.decode(st["tok"], st["pos"], self.cache, sid)
                    nxt = sample(logits, rng=self.rng)
                    st["out"].append(nxt)
                    st["tok"] = nxt
                    st["pos"] += 1
                    if len(st["out"]) >= st["req"].max_new:
                        done.append(sid)
                for sid in done:
                    st = running.pop(sid)
                    results[sid] = st["out"]
                    self.cache.release(sid)
                    m.n_completed += 1
                    m.total_tokens += len(st["out"])
                    # 延迟 = 完成步 - 到达步 (含排队等待: 后批次要等前批次整体跑完)
                    m.total_latency += step - st["req"].arrival
                slot_used += len(running)
                step += 1
                if step > 100_000:  # pragma: no cover
                    break

        m.steps = step
        m.throughput = m.total_tokens / max(step, 1)
        m.slot_utilization = slot_used / max(step * self.max_batch, 1)
        return results, m
