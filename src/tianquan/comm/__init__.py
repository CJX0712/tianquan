"""天权通信层 (L1): 通信后端 + 集合通信算法 + 世界启动器。"""

from .backend import ProcessGroup, SharedMemoryGroup, ThreadGroup
from .collectives import (
    all_gather,
    all_to_all,
    allreduce,
    broadcast,
    reduce_scatter,
    ring_allreduce,
    tree_allreduce,
)
from .launch import run_threads, spawn_world

__all__ = [
    "ProcessGroup",
    "ThreadGroup",
    "SharedMemoryGroup",
    "ring_allreduce",
    "tree_allreduce",
    "allreduce",
    "reduce_scatter",
    "all_gather",
    "all_to_all",
    "broadcast",
    "run_threads",
    "spawn_world",
]
