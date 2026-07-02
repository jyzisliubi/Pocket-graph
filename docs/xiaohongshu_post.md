# 🔥 一个人两周写了个 GraphRAG 框架，同事都在用

> 从 0 到 1 造轮子的快乐谁懂啊！

---

## 📌 背景

做 RAG 做久了，总觉得差点意思：

😩 微软 GraphRAG 太重了，Neo4j 配置半天搞不定
😩 LightRAG 命令行用起来有点原始
😩 中文支持差，全是英文文档
😩 跑出来结果看不见摸不着

**于是决定自己写一个！**

两周后，它长这样 👇

---

## ✨ PocketGraphRAG 是什么？

一个**装在口袋里的 GraphRAG 框架

### 💡 核心特点：
- 🚀 **pip install 即用**，不依赖任何外部数据库（纯 FAISS
- 🇨🇳 **中文友好**，BGE 中文 Embedding + 中文 Prompt
- 🎨 **看得见**，交互式图谱可视化
- 💻 **跑得动**，笔记本 CPU 就能跑
- 🔧 **可扩展**，接口都留好了

---

## 🧠 核心思路：实体级切块

传统 RAG 是「按字数切文本」→ 上下文稀碎 💔

我们是「按实体聚合」→ 一个实体的所有信息都在一起 ✅

```
传统：[文本块1][文本块2][文本块3]  ← 切得稀碎

我们：
实体A → 所有和A有关的内容 🎯
实体B → 所有和B有关的内容 🎯
实体C → 所有和C有关的内容 🎯
```

---

## 🔍 检索怎么玩：双层检索 + RRF 融合

### 第一层：Local Search（局部检索）
从实体出发 → BFS 扩展邻居 → 找到相关实体
适合问具体问题

### 第二层：Global Search（全局检索）
从关系出发 → 找所有满足关系的实体对 → 适合问类型性问题

### 融合：RRF 算法
不需要归一化，直接按排名融合，效果稳得一批

### 再加 Pagerank 加权
重要节点（连接多的实体）排前面
回答质量直接 up up 📈

---

## 🎯 踩过的坑（血泪教训

### 坑1：实体名称不一样怎么办？
「苏云金杆菌（Bt）制剂」和「苏云金杆菌」
子串匹配：根本对不上 ❌

✅ 解决方案：Entity Embedding Matching
把所有实体名转成向量，语义相似度匹配
不一样的名字也能找到 ✨

### 坑2：反向关系怎么搞？
「诺兰导演盗梦空间」但用户问「盗梦空间谁导的？」
只有正向关系就答不好

✅ 解决方案：自动推断反向链接
关系名里有「防治/包含属于」→ 自动建反向
也支持手动配置

### 坑3：Gradio 里 ECharts 不执行？
Gradio 的 HTML 把 script 全禁了 😤

✅ 解决方案：base64 + iframe
把 HTML 转 base64 塞 iframe 里
完美解决 ✨

---

## ⚡ 性能优化：从 1 秒 → 1 毫秒

Pagerank 从 Python 循环改成 NumPy 向量化
500 节点：1秒 → 1毫秒 🚀
提升了 1000 倍！

```python
# 优化前：Python 循环
for i in range(n):
    for j in adj[i]:
        new_pr[j] += pr[i] / len(adj[i])

# 优化后：NumPy 向量化
np.add.at(new_pr, dst, pr[src] / out_degree[src])
```

---

## 📊 现在有啥功能？

数一下：

✅ 131 个单元测试
✅ 11 种图算法（Pagerank/社区发现/最短路径/各种中心性...）
✅ 5 种检索模式
✅ REST API（13 个端点
✅ Web UI（交互式图谱 + 数据管理）
✅ Docker 一键部署
✅ Benchmark 评测框架
✅ 多模态基础框架
✅ 内置示例数据集（电影知识图谱

---

## 🎨 可视化长啥样？

### 知识图谱可视化
ECharts 力导向图
节点大小 = 连接多少
颜色区分类型
鼠标悬停看详情
搜索实体秒定位

### 检索路径透明
匹配了哪些实体？
扩展了哪些邻居？
用了哪些关系？
全给你看明明白白 ✨

---

## 🚀 怎么用？

```bash
# 安装
pip install pocketgraphrag

# 启动 Web UI
python -m PocketGraphRAG.webapp

# 启动 API
python -m PocketGraphRAG.api_server
```

Python API 也很简单：

```python
from PocketGraphRAG import PocketGraphRAG

rag = PocketGraphRAG(search_mode="mix")
result = rag.answer("盗梦空间讲了什么？")
print(result["answer"])
```

---

## 📝 接下来打算做什么？

🔜 支持 PDF/网页/Notion 一键导入
🔜 更多图算法
🔜 可视化编辑器
🔜 多模态增强
🔜 分布式支持

---

## 💬 最后想说的话

做这个项目最大的感受：
**GraphRAG 不一定非要「大而全」**

很多垂直领域小项目
一个「轻量快速看得见」的方案
可能比复杂的企业级框架好用多了

---

🙋‍♂️ 你们做 RAG 都遇到过啥坑？
评论区聊聊～

---

#GraphRAG #RAG #知识图谱 #大模型 #AI #开源项目 #程序员 #编程 #NLP #知识图谱可视化 #技术分享
