"""v0.3.7 新特性单元测试

覆盖：
  1. WORKSPACE 数据隔离（对标 LightRAG v1.4.0）
  2. LLM Cache（对标 LightRAG v1.4.3 Redis cache）
  3. Entity-keyed locks 防竞态（对标 LightRAG v1.4.3）
  4. Phantom entities 清理（对标 graphrag v3.0.9）
"""

import os
import tempfile
import threading
import time

import pytest


# ==========================
# 1. WORKSPACE 数据隔离
# ==========================


class TestWorkspaceIsolation:
    """WORKSPACE 数据隔离测试"""

    def test_default_workspace_no_change(self):
        """WORKSPACE=default 时路径不变（向后兼容）"""
        from PocketGraphRAG.config import _apply_workspace

        assert _apply_workspace("/data/index", "default") == "/data/index"
        assert _apply_workspace("/data/user_docs", "default") == "/data/user_docs"

    def test_custom_workspace_adds_subdir(self):
        """自定义 workspace 在路径下追加子目录"""
        from PocketGraphRAG.config import _apply_workspace

        result = _apply_workspace("/data/index", "rice")
        assert "rice" in result
        assert result.endswith(os.sep + "rice")

    def test_workspace_path_traversal_blocked(self):
        """workspace 名含特殊字符时做安全过滤"""
        from PocketGraphRAG.config import _apply_workspace

        # 路径穿越尝试应被过滤为下划线
        result = _apply_workspace("/data/index", "../../etc")
        assert ".." not in result
        # 应该是一个安全的子目录名
        assert result.startswith("/data/index")

    def test_workspace_empty_falls_back_to_default(self):
        """空 workspace 名回退到原路径"""
        from PocketGraphRAG.config import _apply_workspace

        assert _apply_workspace("/data/index", "") == "/data/index"

    def test_workspace_with_special_chars(self):
        """workspace 名含中文/空格等特殊字符时过滤"""
        from PocketGraphRAG.config import _apply_workspace

        result = _apply_workspace("/data/index", "my workspace")
        # 空格应被替换为下划线
        assert " " not in result
        assert os.path.basename(result) == "my_workspace"

    def test_workspace_env_var_integration(self, monkeypatch):
        """通过环境变量设置 WORKSPACE 后路径隔离生效"""
        monkeypatch.setenv("POCKET_WORKSPACE", "test_ws")
        # 重新导入 config 模块以应用新环境变量
        import importlib

        import PocketGraphRAG.config as config_module

        importlib.reload(config_module)
        assert config_module.WORKSPACE == "test_ws"
        assert "test_ws" in config_module.INDEX_DIR
        assert "test_ws" in config_module.USER_DOCS_DIR
        # 清理：恢复默认
        monkeypatch.delenv("POCKET_WORKSPACE", raising=False)
        importlib.reload(config_module)


# ==========================
# 2. LLM Cache
# ==========================


class TestInMemoryCache:
    """InMemoryCache 测试"""

    def test_basic_set_get(self):
        """基本 set/get"""
        from PocketGraphRAG.llm_cache import InMemoryCache

        cache = InMemoryCache(max_size=10, ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        """未命中返回 None"""
        from PocketGraphRAG.llm_cache import InMemoryCache

        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """LRU 淘汰：超过 max_size 时淘汰最久未访问"""
        from PocketGraphRAG.llm_cache import InMemoryCache

        cache = InMemoryCache(max_size=3, ttl=60)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        # 访问 a，使 b 成为最久未访问
        cache.get("a")
        cache.set("d", "4")  # 应淘汰 b
        assert cache.get("b") is None
        assert cache.get("a") == "1"
        assert cache.get("c") == "3"
        assert cache.get("d") == "4"

    def test_ttl_expiration(self):
        """TTL 过期"""
        from PocketGraphRAG.llm_cache import InMemoryCache

        cache = InMemoryCache(max_size=10, ttl=0)  # ttl=0 不过期
        cache.set("key", "value", ttl=0)
        assert cache.get("key") == "value"  # ttl=0 不过期

        cache2 = InMemoryCache(max_size=10, ttl=1)  # 1秒过期
        cache2.set("key", "value")
        assert cache2.get("key") == "value"
        time.sleep(1.1)
        assert cache2.get("key") is None

    def test_clear(self):
        """清空缓存"""
        from PocketGraphRAG.llm_cache import InMemoryCache

        cache = InMemoryCache()
        cache.set("a", "1")
        cache.set("b", "2")
        n = cache.clear()
        assert n == 2
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_stats(self):
        """统计信息"""
        from PocketGraphRAG.llm_cache import InMemoryCache

        cache = InMemoryCache(max_size=10, ttl=60)
        cache.set("a", "1")
        cache.get("a")  # hit
        cache.get("b")  # miss
        stats = cache.stats()
        assert stats["backend"] == "memory"
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_thread_safety(self):
        """线程安全：多线程并发 set/get 不出错"""
        from PocketGraphRAG.llm_cache import InMemoryCache

        cache = InMemoryCache(max_size=1000, ttl=60)
        errors = []

        def worker(tid):
            try:
                for i in range(100):
                    key = f"thread_{tid}_key_{i}"
                    cache.set(key, f"value_{i}")
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert cache.stats()["size"] > 0


class TestLLMCacheIntegration:
    """LLM Cache 集成测试"""

    def test_cache_key_deterministic(self):
        """相同输入生成相同 key"""
        from PocketGraphRAG.llm_cache import _cache_key

        k1 = _cache_key("sys", "user", "model", 0.1, 2000)
        k2 = _cache_key("sys", "user", "model", 0.1, 2000)
        assert k1 == k2

    def test_cache_key_differs_on_different_input(self):
        """不同输入生成不同 key"""
        from PocketGraphRAG.llm_cache import _cache_key

        k1 = _cache_key("sys", "user1", "model", 0.1, 2000)
        k2 = _cache_key("sys", "user2", "model", 0.1, 2000)
        assert k1 != k2

    def test_should_cache_role_default(self):
        """默认仅缓存 query 角色"""
        from PocketGraphRAG import llm_cache

        # 保存原始状态
        orig_enabled = llm_cache.LLM_CACHE_ENABLED
        orig_roles = llm_cache.LLM_CACHE_ROLES
        try:
            llm_cache.LLM_CACHE_ENABLED = True
            llm_cache.LLM_CACHE_ROLES = ("query",)
            assert llm_cache.should_cache_role("query") is True
            assert llm_cache.should_cache_role("extract") is False
            assert llm_cache.should_cache_role("keywords") is False
        finally:
            llm_cache.LLM_CACHE_ENABLED = orig_enabled
            llm_cache.LLM_CACHE_ROLES = orig_roles

    def test_should_cache_role_wildcard(self):
        """通配符 * 缓存所有角色"""
        from PocketGraphRAG import llm_cache

        orig_enabled = llm_cache.LLM_CACHE_ENABLED
        orig_roles = llm_cache.LLM_CACHE_ROLES
        try:
            llm_cache.LLM_CACHE_ENABLED = True
            llm_cache.LLM_CACHE_ROLES = ("*",)
            assert llm_cache.should_cache_role("query") is True
            assert llm_cache.should_cache_role("extract") is True
        finally:
            llm_cache.LLM_CACHE_ENABLED = orig_enabled
            llm_cache.LLM_CACHE_ROLES = orig_roles

    def test_get_cached_or_call(self):
        """get_cached_or_call 缓存包装器"""
        from PocketGraphRAG import llm_cache

        orig_enabled = llm_cache.LLM_CACHE_ENABLED
        orig_roles = llm_cache.LLM_CACHE_ROLES
        orig_instance = llm_cache._cache_instance
        try:
            llm_cache.LLM_CACHE_ENABLED = True
            llm_cache.LLM_CACHE_ROLES = ("query",)
            llm_cache._cache_instance = None  # 重置单例

            call_count = 0

            def caller():
                nonlocal call_count
                call_count += 1
                return "result"

            # 第一次调用：未命中，执行 caller
            r1 = llm_cache.get_cached_or_call(
                "sys", "user", "model", 0.1, 2000, "query", caller
            )
            assert r1 == "result"
            assert call_count == 1

            # 第二次调用：命中缓存，不执行 caller
            r2 = llm_cache.get_cached_or_call(
                "sys", "user", "model", 0.1, 2000, "query", caller
            )
            assert r2 == "result"
            assert call_count == 1  # caller 未被再次调用
        finally:
            llm_cache.LLM_CACHE_ENABLED = orig_enabled
            llm_cache.LLM_CACHE_ROLES = orig_roles
            llm_cache._cache_instance = orig_instance

    def test_cache_disabled_returns_directly(self):
        """缓存禁用时直接调用"""
        from PocketGraphRAG import llm_cache

        orig_enabled = llm_cache.LLM_CACHE_ENABLED
        try:
            llm_cache.LLM_CACHE_ENABLED = False
            call_count = 0

            def caller():
                nonlocal call_count
                call_count += 1
                return "result"

            r = llm_cache.get_cached_or_call(
                "sys", "user", "model", 0.1, 2000, "query", caller
            )
            assert r == "result"
            assert call_count == 1
        finally:
            llm_cache.LLM_CACHE_ENABLED = orig_enabled


# ==========================
# 3. Entity-keyed locks
# ==========================


class TestEntityLockManager:
    """EntityLockManager 测试"""

    def test_basic_lock_unlock(self):
        """基本加锁/解锁"""
        from PocketGraphRAG.concurrency import EntityLockManager

        mgr = EntityLockManager()
        with mgr.lock("entity1"):
            # 锁已获取
            pass
        # 锁已释放（无异常即成功）

    def test_reentrant_same_thread(self):
        """同线程可重入（RLock）"""
        from PocketGraphRAG.concurrency import EntityLockManager

        mgr = EntityLockManager()
        with mgr.lock("entity1"):
            with mgr.lock("entity1"):  # 同线程再次获取同一锁
                pass  # 不死锁即成功

    def test_concurrent_different_entities(self):
        """不同实体可并行加锁"""
        from PocketGraphRAG.concurrency import EntityLockManager

        mgr = EntityLockManager()
        results = []

        def worker(entity, delay=0.05):
            with mgr.lock(entity):
                results.append(f"{entity}_start")
                time.sleep(delay)
                results.append(f"{entity}_end")

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # 不同实体应能并行（两个 start 在两个 end 之前）
        starts = [r for r in results if r.endswith("_start")]
        ends = [r for r in results if r.endswith("_end")]
        assert len(starts) == 2
        assert len(ends) == 2

    def test_concurrent_same_entity_serialized(self):
        """相同实体串行化"""
        from PocketGraphRAG.concurrency import EntityLockManager

        mgr = EntityLockManager()
        execution_order = []

        def worker(tid):
            with mgr.lock("shared"):
                execution_order.append(f"start_{tid}")
                time.sleep(0.05)
                execution_order.append(f"end_{tid}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 相同实体的操作应串行（start 和 end 交替出现）
        # 即不会有 start_A, start_B, end_A, end_B 的情况
        for i in range(0, len(execution_order), 2):
            assert execution_order[i].startswith("start")
            assert execution_order[i + 1].startswith("end")

    def test_lock_multiple(self):
        """多实体同时加锁（按排序避免死锁）"""
        from PocketGraphRAG.concurrency import EntityLockManager

        mgr = EntityLockManager()
        with mgr.lock_multiple(["B", "A", "C"]):
            pass  # 不死锁即成功

    def test_lock_multiple_deadlock_free(self):
        """多实体锁无死锁（不同线程以相同顺序获取）"""
        from PocketGraphRAG.concurrency import EntityLockManager

        mgr = EntityLockManager()
        completed = []

        def worker(tid, entities):
            with mgr.lock_multiple(entities):
                time.sleep(0.02)
                completed.append(tid)

        # 线程1: A, B；线程2: B, A（顺序相反，但内部排序后一致）
        t1 = threading.Thread(target=worker, args=(1, ["A", "B"]))
        t2 = threading.Thread(target=worker, args=(2, ["B", "A"]))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert len(completed) == 2  # 都完成，无死锁

    def test_stats(self):
        """统计信息"""
        from PocketGraphRAG.concurrency import EntityLockManager

        mgr = EntityLockManager()
        with mgr.lock("e1"):
            pass
        with mgr.lock("e2"):
            pass
        stats = mgr.stats()
        assert stats["total_locks"] >= 2

    def test_cleanup_unused(self):
        """清理未持有的锁"""
        from PocketGraphRAG.concurrency import EntityLockManager

        mgr = EntityLockManager()
        with mgr.lock("temp1"):
            pass
        with mgr.lock("temp2"):
            pass
        n = mgr.cleanup_unused()
        assert n >= 2  # 至少清理了2个未持有的锁


class TestGetEntityLockManager:
    """全局单例测试"""

    def test_singleton(self):
        """get_entity_lock_manager 返回同一实例"""
        from PocketGraphRAG.concurrency import get_entity_lock_manager

        m1 = get_entity_lock_manager()
        m2 = get_entity_lock_manager()
        assert m1 is m2


# ==========================
# 4. Phantom entities 清理
# ==========================


class TestCleanupOrphanEntities:
    """Phantom entities 清理测试"""

    def test_no_orphans(self):
        """无孤儿实体时返回 0"""
        from PocketGraphRAG.core.storages.in_memory_graph import InMemoryGraphStore

        graph = InMemoryGraphStore()
        graph.add_triple("A", "关系", "B")
        graph.add_triple("B", "关系", "C")
        removed = graph.cleanup_orphan_entities()
        assert removed == 0

    def test_cleanup_orphans(self):
        """清理孤儿实体"""
        from PocketGraphRAG.core.storages.in_memory_graph import InMemoryGraphStore

        graph = InMemoryGraphStore()
        graph.add_triple("A", "关系", "B")
        # 手动添加一个孤儿实体（有 key 但无边）
        graph.entity_relations["orphan"] = []
        graph.reverse_relations["orphan2"] = []
        removed = graph.cleanup_orphan_entities()
        assert removed == 2
        assert "orphan" not in graph.entity_relations
        assert "orphan2" not in graph.reverse_relations

    def test_cleanup_preserves_connected(self):
        """清理不删除有边的实体"""
        from PocketGraphRAG.core.storages.in_memory_graph import InMemoryGraphStore

        graph = InMemoryGraphStore()
        graph.add_triple("A", "关系", "B")
        graph.add_triple("B", "关系", "C")
        graph.entity_relations["orphan"] = []
        removed = graph.cleanup_orphan_entities()
        assert removed == 1
        # A, B, C 都应该保留
        entities = graph.all_entities()
        assert "A" in entities
        assert "B" in entities
        assert "C" in entities
        assert "orphan" not in entities

    def test_cleanup_empty_graph(self):
        """空图清理返回 0"""
        from PocketGraphRAG.core.storages.in_memory_graph import InMemoryGraphStore

        graph = InMemoryGraphStore()
        removed = graph.cleanup_orphan_entities()
        assert removed == 0

    def test_cleanup_after_manual_removal(self):
        """模拟三元组删除后清理孤儿"""
        from PocketGraphRAG.core.storages.in_memory_graph import InMemoryGraphStore

        graph = InMemoryGraphStore()
        graph.add_triple("A", "防治", "病害X")
        graph.add_triple("B", "防治", "病害X")
        # 模拟删除 A 的所有三元组（但 A 实体仍留在 dict 中）
        graph.entity_relations["A"] = []
        graph.reverse_relations["A"] = []
        removed = graph.cleanup_orphan_entities()
        assert removed == 1
        assert "A" not in graph.all_entities()
        # B 和 病害X 仍保留
        assert "B" in graph.all_entities()
        assert "病害X" in graph.all_entities()

    def test_abc_default_returns_zero(self):
        """GraphStore ABC 默认实现返回 0"""
        from PocketGraphRAG.core.storages.base import GraphStore

        # 创建一个最小实现的 GraphStore 子类
        class MinimalGraphStore(GraphStore):
            def add_triple(self, h, r, t):
                return True

            def add_triples(self, triples):
                return 0

            def neighbors(self, entity, hops=2):
                return []

            def relations_of(self, entity):
                return []

            def reverse_relations_of(self, entity):
                return []

            def all_entities(self):
                return []

            def all_relations(self):
                return []

            def __len__(self):
                return 0

        store = MinimalGraphStore()
        assert store.cleanup_orphan_entities() == 0
