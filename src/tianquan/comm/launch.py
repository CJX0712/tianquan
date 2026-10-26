"""
tianquan.comm.launch — 世界 (world) 启动器

提供两种真实并行执行方式, 供基准与测试使用:

* ``run_threads``  — 同进程多线程, 零 spawn 开销, 用于功能测试与重叠实验
* ``spawn_world``  — 跨进程 (spawn, Windows/macOS 安全), 共享内存搬运真实字节

Windows 注意: spawn 启动方式要求 target 函数是模块顶层可 pickle 对象;
``multiprocessing.Barrier`` 不可 pickle, 故同步用 ``Manager().Barrier`` 代理。
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable

from .backend import SharedMemoryGroup, ThreadGroup

__all__ = ["run_threads", "spawn_world"]


def run_threads(fn: Callable, world_size: int, *args: Any) -> list[Any]:
    """用线程后端并行执行 ``fn(rank, pg, *args)``, 返回按 rank 排序的结果。"""
    import threading

    groups = ThreadGroup.create(world_size)
    results: list[Any] = [None] * world_size
    errors: list[BaseException] = []

    def _run(rank: int) -> None:
        try:
            results[rank] = fn(rank, groups[rank], *args)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_run, args=(r,), daemon=True) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    if errors:
        raise errors[0]
    if any(t.is_alive() for t in threads):  # pragma: no cover
        raise RuntimeError("线程超时未完成 (可能存在死锁)")
    return results


def _worker(rank: int, world_size: int, slot_names: list[str], barrier, queue, fn, args) -> None:
    pg = SharedMemoryGroup.build(rank, world_size, barrier, slot_names)
    try:
        res = fn(rank, pg, *args)
        queue.put((rank, _safe(res)))
    except BaseException as e:  # noqa: BLE001
        import traceback

        queue.put((rank, {"__error__": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}))
    finally:
        pg.close()


def _safe(res: Any) -> Any:
    """只回传可 pickle 的小结果 (标量/小数组/字符串), 避免队列拥塞。"""
    if res is None:
        return None
    if isinstance(res, dict):
        return {k: _safe(v) for k, v in res.items()}
    if isinstance(res, (int, float, str, bool, bytes)):
        return res
    if isinstance(res, tuple):
        return tuple(_safe(x) for x in res)
    if isinstance(res, list):
        return [_safe(x) for x in res]
    return str(res)


def spawn_world(fn: Callable, world_size: int, *args: Any, timeout: float = 600.0) -> list[Any]:
    """跨进程 (spawn) 执行 ``fn(rank, pg, *args)``, 返回按 rank 排序的结果。

    ``fn`` 必须是模块顶层函数 (spawn pickling 要求)。
    """
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    barrier = manager.Barrier(world_size)
    queue = ctx.Queue()

    slot_names, _b, slot_objs = SharedMemoryGroup.create(world_size, ctx)
    procs = [
        ctx.Process(
            target=_worker,
            args=(r, world_size, slot_names, barrier, queue, fn, args),
            daemon=True,
        )
        for r in range(world_size)
    ]
    try:
        for p in procs:
            p.start()
        collected: dict[int, Any] = {}
        for _ in range(world_size):
            rank, payload = queue.get(timeout=timeout)
            collected[rank] = payload
        for p in procs:
            p.join(timeout=30)
        for r, payload in collected.items():
            if isinstance(payload, dict) and "__error__" in payload:
                raise RuntimeError(f"rank {r} 失败: {payload['__error__']}\n{payload.get('tb', '')}")
        return [collected[r] for r in range(world_size)]
    finally:
        for p in procs:
            if p.is_alive():  # pragma: no cover
                p.terminate()
        for s in slot_objs:
            try:
                s.close()
                s.unlink()
            except Exception:  # noqa: BLE001, S112
                pass
        manager.shutdown()
