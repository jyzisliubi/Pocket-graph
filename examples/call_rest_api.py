"""调用 PocketGraphRAG 的 REST API

这个示例展示如何通过 HTTP 调用 PocketGraphRAG 的 REST API，
适合需要从其他语言/服务调用的场景。

启动 API 服务：
    python -m PocketGraphRAG.api_server --host 0.0.0.0 --port 8000

然后运行本脚本：
    python examples/call_rest_api.py

API 文档：http://localhost:8000/docs
"""

import json
import os
import sys
import urllib.request
import urllib.error


API_BASE = "http://localhost:8000"


def _post(path: str, data: dict) -> dict:
    """发送 POST 请求"""
    url = f"{API_BASE}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"  请求失败: {e}")
        print(f"  请确认 API 服务已启动: python -m PocketGraphRAG.api_server")
        sys.exit(1)


def _get(path: str) -> dict:
    """发送 GET 请求"""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"  请求失败: {e}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  PocketGraphRAG REST API 调用示例")
    print("=" * 60)

    # 1. 检查服务状态
    print("\n[1/4] 检查 API 服务...")
    try:
        stats = _get("/api/graph/stats")
        print(f"  图谱统计：{stats.get('total_entities', 0)} 实体, "
              f"{stats.get('total_relations', 0)} 关系")
    except Exception:
        return

    # 2. 只检索不生成（不调用 LLM，速度快）
    print("\n[2/4] 只检索不生成（/api/retrieve）...")
    retrieve_resp = _post("/api/retrieve", {
        "query": "稻瘟病症状",
        "top_k": 3,
        "search_mode": "kg_only",
    })
    sources = retrieve_resp.get("sources", [])
    print(f"  检索到 {len(sources)} 条来源")
    for i, s in enumerate(sources[:2], 1):
        print(f"  [{i}] {s.get('text', '')[:70]}...")

    # 3. 完整问答（调用 LLM 生成）
    print("\n[3/4] 完整问答（/api/qa，调用 LLM）...")
    qa_resp = _post("/api/qa", {
        "query": "三环唑防治稻瘟病的用量是多少？",
        "top_k": 5,
        "search_mode": "kg_only",
    })
    print(f"  答案：{qa_resp.get('answer', '')[:100]}...")
    print(f"  来源数：{len(qa_resp.get('sources', []))}")

    # 4. 图谱查询
    print("\n[4/4] 图谱查询...")
    try:
        pagerank = _get("/api/graph/pagerank")
        # /api/graph/pagerank 返回 List[{entity, score}]，兼容 list 与 dict 包装
        if isinstance(pagerank, list):
            top_entities = pagerank[:3]
        else:
            top_entities = pagerank.get("entities", [])[:3]
        print(f"  PageRank Top 3：")
        for e in top_entities:
            if isinstance(e, dict):
                print(f"    {e.get('entity', '')}: score={e.get('score', 0):.4f}")
            else:
                print(f"    {e}")
    except Exception as e:
        print(f"  PageRank 查询失败: {e}")

    print(f"\n{'=' * 60}")
    print(f"  完成！完整 API 文档见：{API_BASE}/docs")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
