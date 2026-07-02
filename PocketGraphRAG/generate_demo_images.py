"""
PocketGraphRAG Demo 可视化图片生成脚本

使用 networkx + matplotlib 生成知识图谱示意图。
生成的图片保存在 assets/ 目录下。
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use("Agg")  # 无头模式


def setup_chinese_font():
    """设置中文字体"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def create_movie_kg_graph():
    """创建电影知识图谱示例

    选用 4 部经典电影 + 3 位导演 + 5 位演员 + 3 个类型，
    诺兰作为连接《盗梦空间》与《星际穿越》的枢纽节点，体现 KG 多跳检索能力。
    """
    G = nx.DiGraph()

    # 电影节点
    movies = ["肖申克的救赎", "盗梦空间", "星际穿越", "霸王别姬"]
    # 导演节点
    directors = ["弗兰克·德拉邦特", "克里斯托弗·诺兰", "陈凯歌"]
    # 演员节点
    actors = ["蒂姆·罗宾斯", "摩根·弗里曼", "莱昂纳多·迪卡普里奥", "马修·麦康纳", "张国荣"]
    # 类型节点
    genres = ["剧情", "科幻", "犯罪"]

    # 添加节点
    for m in movies:
        G.add_node(m, type="movie")
    for d in directors:
        G.add_node(d, type="director")
    for a in actors:
        G.add_node(a, type="actor")
    for g in genres:
        G.add_node(g, type="genre")

    # 添加边（导演/主演/类型关系）
    edges = [
        ("肖申克的救赎", "弗兰克·德拉邦特", "导演"),
        ("肖申克的救赎", "蒂姆·罗宾斯", "主演"),
        ("肖申克的救赎", "摩根·弗里曼", "主演"),
        ("肖申克的救赎", "剧情", "类型"),
        ("肖申克的救赎", "犯罪", "类型"),
        ("盗梦空间", "克里斯托弗·诺兰", "导演"),
        ("盗梦空间", "莱昂纳多·迪卡普里奥", "主演"),
        ("盗梦空间", "科幻", "类型"),
        ("星际穿越", "克里斯托弗·诺兰", "导演"),
        ("星际穿越", "马修·麦康纳", "主演"),
        ("星际穿越", "科幻", "类型"),
        ("霸王别姬", "陈凯歌", "导演"),
        ("霸王别姬", "张国荣", "主演"),
        ("霸王别姬", "剧情", "类型"),
    ]
    for u, v, _ in edges:
        G.add_edge(u, v)

    return G


def draw_kg_overview(output_path: str):
    """绘制知识图谱概览图"""
    setup_chinese_font()
    G = create_movie_kg_graph()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 节点颜色
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get("type", "other")
        if node_type == "movie":
            node_colors.append("#ef4444")  # 红色 - 电影
        elif node_type == "director":
            node_colors.append("#22c55e")  # 绿色 - 导演
        elif node_type == "actor":
            node_colors.append("#3b82f6")  # 蓝色 - 演员
        else:
            node_colors.append("#f59e0b")  # 橙色 - 类型

    # 节点大小
    node_sizes = [
        2800 if G.nodes[n].get("type") == "movie" else 2200
        for n in G.nodes()
    ]

    # 布局
    pos = nx.spring_layout(G, k=1.5, seed=42)

    # 画边
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#94a3b8",
        width=1.5,
        alpha=0.7,
        arrows=True,
        arrowsize=15,
    )

    # 画节点
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="#ffffff",
        linewidths=2,
    )

    # 画标签
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=10,
        font_weight="bold",
        font_color="#1e293b",
    )

    # 图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#ef4444", edgecolor="#fff", label="电影实体"),
        Patch(facecolor="#22c55e", edgecolor="#fff", label="导演实体"),
        Patch(facecolor="#3b82f6", edgecolor="#fff", label="演员实体"),
        Patch(facecolor="#f59e0b", edgecolor="#fff", label="类型实体"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=11,
        framealpha=0.9,
        fancybox=True,
    )

    ax.set_title(
        "PocketGraphRAG - 电影知识图谱",
        fontsize=16,
        fontweight="bold",
        pad=20,
        color="#0f172a",
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="#f8fafc", edgecolor="none"
    )
    plt.close()
    print(f"[OK] 生成: {output_path}")


def draw_pagerank_demo(output_path: str):
    """绘制 Pagerank 重要性排序示意图"""
    setup_chinese_font()
    G = nx.karate_club_graph()

    # 计算 Pagerank
    pr = nx.pagerank(G, alpha=0.85)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    pos = nx.spring_layout(G, seed=42)

    # 节点大小和颜色由 Pagerank 分数决定
    node_sizes = [300 + 3000 * pr[n] for n in G.nodes()]
    node_colors = [pr[n] for n in G.nodes()]

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        alpha=0.9,
        edgecolors="#ffffff",
        linewidths=2,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#94a3b8",
        width=0.8,
        alpha=0.5,
    )

    # 颜色条
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pagerank 重要性分数", fontsize=11)

    ax.set_title(
        "Pagerank 实体重要性排序示例",
        fontsize=16,
        fontweight="bold",
        pad=20,
        color="#0f172a",
    )
    ax.text(
        0.5,
        -0.05,
        "节点越大、颜色越红 → 实体越重要",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#64748b",
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="#f8fafc", edgecolor="none"
    )
    plt.close()
    print(f"[OK] 生成: {output_path}")


def draw_community_detection_demo(output_path: str):
    """绘制社区发现示意图"""
    setup_chinese_font()
    G = nx.karate_club_graph()

    # 简单社区发现：基于标签传播思路，用 connected components 分组模拟
    import random

    random.seed(42)
    communities = {}
    colors = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#8b5cf6"]

    # 用贪心方式模拟社区
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for i, node in enumerate(nodes):
        neighbors = list(G.neighbors(node))
        neighbor_communities = [communities[n] for n in neighbors if n in communities]
        if neighbor_communities:
            # 选邻居最多的社区
            from collections import Counter

            c = Counter(neighbor_communities).most_common(1)[0][0]
            communities[node] = c
        else:
            communities[node] = colors[i % len(colors)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    pos = nx.spring_layout(G, seed=42)

    node_colors = [communities[n] for n in G.nodes()]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#cbd5e1",
        width=0.8,
        alpha=0.5,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=500,
        alpha=0.9,
        edgecolors="#ffffff",
        linewidths=2,
    )

    from matplotlib.patches import Patch

    unique_colors = list(set(communities.values()))
    legend_elements = [
        Patch(facecolor=c, edgecolor="#fff", label=f"社区 {i + 1}")
        for i, c in enumerate(unique_colors)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=11,
        framealpha=0.9,
        fancybox=True,
    )

    ax.set_title(
        "社区发现 (Community Detection) 示例",
        fontsize=16,
        fontweight="bold",
        pad=20,
        color="#0f172a",
    )
    ax.text(
        0.5,
        -0.05,
        "同一颜色的实体连接更紧密，属于同一社区",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#64748b",
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="#f8fafc", edgecolor="none"
    )
    plt.close()
    print(f"[OK] 生成: {output_path}")


def draw_architecture_diagram(output_path: str):
    """绘制系统架构图"""
    setup_chinese_font()

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    # 定义框
    boxes = [
        # (x, y, w, h, text, color, fontsize)
        (1, 7.5, 3, 1, "用户输入\n(问题)", "#e0f2fe", 12),
        (5.5, 7.5, 3, 1, "查询理解\n(Multi-hop 分解)", "#fef3c7", 12),
        (10, 7.5, 3, 1, "实体/关系\nEmbedding 匹配", "#fce7f3", 12),
        (1, 5, 3, 1.5, "向量检索\n(FAISS + BGE)", "#bbf7d0", 12),
        (5.5, 5, 3, 1.5, "KG 双层检索\n(Local + Global)", "#fecaca", 12),
        (10, 5, 3, 1.5, "Pagerank 加权\n+ RRF 融合", "#ddd6fe", 12),
        (5.5, 2.5, 3, 1.2, "重排序 & Top-K\n(知识来源)", "#fed7aa", 12),
        (5.5, 0.5, 3, 1.2, "LLM 生成回答\n(流式输出)", "#a5f3fc", 12),
    ]

    for x, y, w, h, text, color, fs in boxes:
        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            facecolor=color,
            edgecolor="#475569",
            linewidth=2,
            transform=ax.transData,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color="#0f172a",
        )

    # 箭头
    arrows = [
        (4, 8, 5.5, 8),  # 用户输入 → 查询理解
        (8.5, 8, 10, 8),  # 查询理解 → Embedding
        (2.5, 7.5, 2.5, 6.5),  # 用户输入 → 向量检索
        (7, 7.5, 7, 6.5),  # 查询理解 → KG 检索
        (11.5, 7.5, 11.5, 6.5),  # Embedding → Pagerank+RRF
        (4, 5.75, 5.5, 5.75),  # 向量检索 → 重排序
        (8.5, 5.75, 10, 5.75),  # KG检索 → Pagerank+RRF
        (11.5, 5, 8.5, 3.5),  # RRF → 重排序
        (7, 2.5, 7, 1.7),  # 重排序 → LLM
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#475569", lw=2),
        )

    ax.set_title(
        "PocketGraphRAG 系统架构图",
        fontsize=20,
        fontweight="bold",
        pad=20,
        color="#0f172a",
    )

    # 底部标注
    ax.text(
        7,
        -0.3,
        "五大核心模块：实体级切块 · 双层 KG 检索 · Pagerank 排序 · RRF 融合 · 流式生成",
        ha="center",
        fontsize=12,
        color="#64748b",
        transform=ax.transData,
    )

    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="#f8fafc", edgecolor="none"
    )
    plt.close()
    print(f"[OK] 生成: {output_path}")


def main():
    assets_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "assets"
    )
    os.makedirs(assets_dir, exist_ok=True)

    print("生成 Demo 可视化图片...")
    print()

    draw_kg_overview(os.path.join(assets_dir, "kg_overview.png"))
    draw_pagerank_demo(os.path.join(assets_dir, "pagerank_demo.png"))
    draw_community_detection_demo(os.path.join(assets_dir, "community_demo.png"))
    draw_architecture_diagram(os.path.join(assets_dir, "architecture.png"))

    print()
    print(f"[完成] 所有图片已保存至: {assets_dir}")


if __name__ == "__main__":
    main()
