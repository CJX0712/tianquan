"""
tianquan.eval.bench — 确定性基准 harness

设计信条
--------
性能数字如果不可复现, 就只是广告。本 harness 强制:

* 固定种子 (``set_determinism``) —— 同一 commit 多次运行结果一致
* warmup + 多次重复, 报告 **中位数** (对异常值稳健) 与 p95
* 记录环境指纹 (CPU/内存/Python/NumPy 版本), 写进报告
* 所有结果落成 JSON, 供 README 直接引用 (杜绝手抄数字)
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = ["BenchResult", "bench", "Suite", "env_fingerprint"]


def env_fingerprint() -> dict:
    import os
    import sys

    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        info["mem_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        info["cpu_freq_mhz"] = (
            round(psutil.cpu_freq().current, 0) if psutil.cpu_freq() else None
        )
    except Exception:  # noqa: BLE001
        pass
    return info


@dataclass
class BenchResult:
    name: str
    unit: str = "s"
    samples: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else float("nan")

    @property
    def p95(self) -> float:
        if not self.samples:
            return float("nan")
        s = sorted(self.samples)
        return s[min(len(s) - 1, int(0.95 * len(s)))]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "unit": self.unit,
            "median": round(self.median, 6),
            "p95": round(self.p95, 6),
            "min": round(min(self.samples), 6) if self.samples else None,
            "n": len(self.samples),
            **self.extra,
        }


def bench(
    name: str, fn: Callable[[], float], n_repeat: int = 5, warmup: int = 1, unit: str = "s", **extra
) -> BenchResult:
    """跑 ``fn`` (返回耗时或自定义标量) 若干次, 取中位数。"""
    for _ in range(warmup):
        fn()
    r = BenchResult(name=name, unit=unit, extra=dict(extra))
    for _ in range(n_repeat):
        r.samples.append(float(fn()))
    return r


class Suite:
    """收集一组基准, 产出 JSON + Markdown 表。"""

    def __init__(self, title: str = "天权基准") -> None:
        self.title = title
        self.results: list[BenchResult] = []
        self.records: list[dict] = []

    def add(self, r: BenchResult) -> BenchResult:
        self.results.append(r)
        return r

    def record(self, **kv) -> None:
        """非计时类结果 (如内存表、吞吐比) 直接记录。"""
        self.records.append(kv)

    def to_json(self, path: str | None = None) -> dict:
        data = {
            "title": self.title,
            "env": env_fingerprint(),
            "results": [r.as_dict() for r in self.results],
            "records": self.records,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    def to_markdown(self) -> str:
        lines = [f"### {self.title}", ""]
        if self.results:
            lines.append("| 基准 | 中位数 | p95 | 最小 | 次数 | 单位 |")
            lines.append("|---|---:|---:|---:|---:|---|")
            for r in self.results:
                lines.append(
                    f"| {r.name} | {r.median:.4f} | {r.p95:.4f} | {min(r.samples):.4f} "
                    f"| {len(r.samples)} | {r.unit} |"
                )
            lines.append("")
        for rec in self.records:
            lines.append("| " + " | ".join(f"**{k}**" for k in rec) + " |")
            lines.append("|" + "---|" * len(rec))
            lines.append("| " + " | ".join(str(v) for v in rec.values()) + " |")
            lines.append("")
        return "\n".join(lines)
