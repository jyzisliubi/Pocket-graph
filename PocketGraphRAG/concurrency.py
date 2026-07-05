"""
实体级锁管理器（v0.3.7：对标 LightRAG v1.4.3 entity-keyed locks）

防止并发 KG 抽取/增量索引时多个线程同时修改同一实体导致图数据竞态。

问题场景：
  - 多个文档同时抽取三元组，都涉及"水稻"实体
  - 不加锁：两个线程同时 add_entity("水稻") + add_edge，可能重复创建或覆盖
  - 加锁：按实体名加锁，同一实体的操作串行化，不同实体并行

设计：
  - 每个实体名对应一个 threading.Lock（惰性创建）
  - 提供 context manager：with lock_manager.lock("水稻"): ...
  - 全局锁保护 _locks dict 本身（防止并发创建同一实体的锁）
  - 可重入：同一线程可多次 acquire 同一实体的锁（RLock）
  - 自动清理：长时间不用的锁会被 GC（可选，默认关闭避免复杂度）

使用方式：
    from .concurrency import get_entity_lock_manager

    mgr = get_entity_lock_manager()
    with mgr.lock("水稻"):
        graph_store.add_entity("水稻")
        graph_store.add_edge("水稻", "种植于", "稻田")
"""

import threading
from contextlib import contextmanager
from typing import Dict, Optional


class EntityLockManager:
    """实体级锁管理器

    为每个实体名维护一个独立的 RLock（可重入锁），
    保证同一实体的图操作串行化，不同实体可并行。

    线程安全：_locks dict 的访问由 _global_lock 保护。
    """

    def __init__(self):
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, entity: str) -> threading.RLock:
        """获取或创建实体对应的锁（线程安全）"""
        # 先不加锁读一次（快路径），命中则直接返回
        lock = self._locks.get(entity)
        if lock is not None:
            return lock
        # 慢路径：加全局锁创建
        with self._global_lock:
            # double-check（可能其他线程已经创建了）
            lock = self._locks.get(entity)
            if lock is None:
                lock = threading.RLock()
                self._locks[entity] = lock
            return lock

    @contextmanager
    def lock(self, entity: str):
        """获取实体锁的上下文管理器

        用法:
            with mgr.lock("水稻"):
                # 对"水稻"实体的图操作
                graph_store.add_entity("水稻")

        同一实体可重入（RLock），同一线程多次 acquire 不会死锁。
        """
        lk = self._get_lock(entity)
        lk.acquire()
        try:
            yield
        finally:
            lk.release()

    @contextmanager
    def lock_multiple(self, entities: list):
        """同时获取多个实体的锁（按排序避免死锁）

        用法:
            with mgr.lock_multiple(["水稻", "稻田"]):
                # 涉及多个实体的操作

        按实体名排序后依次加锁，保证不同线程以相同顺序获取锁，避免死锁。
        """
        # 去重 + 排序（避免死锁：所有线程以相同顺序获取锁）
        unique = sorted(set(entities))
        acquired = []
        try:
            for e in unique:
                lk = self._get_lock(e)
                lk.acquire()
                acquired.append(lk)
            yield
        finally:
            for lk in reversed(acquired):
                lk.release()

    def cleanup_unused(self, max_keep: int = 10000) -> int:
        """清理未被持有的锁（防止内存泄漏）

        遍历所有锁，删除未被持有的（RLock.acquire() 返回 True 表示当前未被持有）。
        注意：此方法应仅在无并发操作时调用（如索引构建完成后）。

        Returns:
            清理的锁数量
        """
        with self._global_lock:
            to_remove = []
            for entity, lk in list(self._locks.items()):
                # 尝试 acquire（non-blocking），成功表示锁未被持有
                if lk.acquire(blocking=False):
                    lk.release()
                    to_remove.append(entity)
            for e in to_remove:
                del self._locks[e]
            return len(to_remove)

    def stats(self) -> dict:
        """返回锁管理器统计信息"""
        with self._global_lock:
            return {
                "total_locks": len(self._locks),
                "entities": list(self._locks.keys())[:20],  # 仅前20个用于展示
            }


# 全局单例
_lock_manager: Optional[EntityLockManager] = None
_singleton_lock = threading.Lock()


def get_entity_lock_manager() -> EntityLockManager:
    """获取全局 EntityLockManager 单例"""
    global _lock_manager
    if _lock_manager is not None:
        return _lock_manager
    with _singleton_lock:
        if _lock_manager is None:
            _lock_manager = EntityLockManager()
        return _lock_manager
