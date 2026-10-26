"""
tianquan.serve.radix — RadixAttention 前缀树缓存

零重写声明
----------
复刻 **SGLang 的 RadixAttention (Zheng et al., 2024)**: 用一棵基数树
(radix tree) 把所有请求的 KV 缓存按 token 前缀组织起来, 使得
多轮对话 / 相同系统提示 / few-shot 模板之间能**自动复用前缀**,
配合 LRU 淘汰在有限显存下维持高命中率。

与 vLLM 的 automatic prefix caching 是同一思想; 这里把命中率做成可测指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RadixNode", "RadixTree"]


@dataclass
class RadixNode:
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    parent: "RadixNode | None" = None
    token: int | None = None
    n_ref: int = 0  # 引用计数 (正在被多少个活跃序列使用)
    last_access: int = 0
    kv_blocks: list[int] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class RadixTree:
    """前缀树: insert / match / evict(LRU)。"""

    def __init__(self) -> None:
        self.root = RadixNode()
        self.clock = 0
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.n_insert = 0

    # ------------------------------------------------------------------ #
    def match(self, tokens: list[int]) -> tuple[int, RadixNode]:
        """返回 (匹配到的最长前缀长度, 对应节点)。"""
        node = self.root
        n = 0
        for t in tokens:
            c = node.children.get(t)
            if c is None or not c.kv_blocks and not c.children:
                break
            node = c
            n += 1
        self.clock += 1
        if n > 0:
            node.last_access = self.clock
            self.hits += 1
            self.tokens_saved += n
        else:
            self.misses += 1
        return n, node

    def insert(self, tokens: list[int], kv_blocks: list[int]) -> None:
        node = self.root
        path = [node]
        for t in tokens:
            c = node.children.get(t)
            if c is None:
                c = RadixNode(parent=node, token=t)
                node.children[t] = c
            node = c
            path.append(node)
        # 只在最后一个节点挂块 (内部节点的块由路径上最先出现的持有者代表)
        if kv_blocks:
            node.kv_blocks = list(kv_blocks)
        self.clock += 1
        node.last_access = self.clock
        self.n_insert += 1

    # ------------------------------------------------------------------ #
    def evict(self, n_blocks_needed: int, total_blocks: int, used_blocks: int) -> int:
        """LRU 淘汰: 优先淘汰 n_ref==0 的叶子, 直到腾出所需块数。"""
        freed = 0
        while used_blocks - freed > total_blocks - n_blocks_needed:
            victim = self._find_lru_leaf()
            if victim is None:
                break
            freed += len(victim.kv_blocks)
            victim.kv_blocks = []
            # 从父节点摘除
            if victim.parent is not None:
                victim.parent.children.pop(victim.token, None)
        return freed

    def _find_lru_leaf(self) -> RadixNode | None:
        best: RadixNode | None = None
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root and n.n_ref == 0 and not n.children and n.kv_blocks:
                if best is None or n.last_access < best.last_access:
                    best = n
            stack.extend(n.children.values())
        return best

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "lookups": total,
            "hit_rate": self.hits / max(total, 1),
            "tokens_saved": self.tokens_saved,
            "n_insert": self.n_insert,
        }
