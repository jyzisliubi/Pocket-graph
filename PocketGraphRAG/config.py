"""
配置文件 - 所有可调参数集中管理

支持通过环境变量覆盖默认配置：
  POCKET_DATA_PATH        - 知识图谱数据文件路径
  POCKET_INDEX_DIR        - 向量索引目录
  POCKET_EMBEDDING_MODEL  - Embedding 模型路径或名称
"""

import os
import warnings


def _env(new_name: str, old_name: str | None = None, default: str = "") -> str:
    """读取环境变量。若提供 old_name 且 new_name 未设置，回退到 old_name 并发 DeprecationWarning。"""
    new_val = os.environ.get(new_name, "")
    if new_val:
        return new_val
    if old_name:
        old_val = os.environ.get(old_name, "")
        if old_val:
            warnings.warn(
                f"环境变量 {old_name} 已弃用，请改用 {new_name}",
                DeprecationWarning,
                stacklevel=2,
            )
            return old_val
    return default


def _env_int(new_name: str, old_name: str | None, default: int) -> int:
    val = _env(new_name, old_name, str(default))
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _env_float(new_name: str, old_name: str | None, default: float) -> float:
    val = _env(new_name, old_name, str(default))
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _env_bool(new_name: str, old_name: str | None, default: bool) -> bool:
    val = _env(new_name, old_name, "true" if default else "false").lower()
    return val in ("1", "true", "yes", "on")

# ========================
# 加载 .env 文件（python-dotenv）
# ========================
# 在读取任何环境变量之前先加载 .env，让 .env 里的 API Key / 模型配置生效。
# 找不到 .env 时静默跳过（生产环境通常用真实环境变量）。
try:
    from dotenv import load_dotenv

    _PROJECT_ROOT_TMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # override=True：让 Web UI 修改后写入 .env 的配置在进程重启后仍能生效，
    # 而不被进程启动时的旧环境变量覆盖。
    load_dotenv(os.path.join(_PROJECT_ROOT_TMP, ".env"), override=True)
except ImportError:
    pass

# ========================
# 项目根目录（自动推断）
# ========================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ========================
# 数据路径
# ========================
# 默认数据路径：项目根目录下的 examples/movie_kg/data/movie_knowledge_graph_triples.txt
# v0.3.3 起默认数据集为 movie_kg（通用 GraphRAG 定位，不绑死领域）
_DEFAULT_DATA_PATH = os.path.join(
    _PROJECT_ROOT,
    "examples",
    "movie_kg",
    "data",
    "movie_knowledge_graph_triples.txt",
)
DATA_PATH = _env("POCKET_DATA_PATH", default=_DEFAULT_DATA_PATH)

# 索引目录
_DEFAULT_INDEX_DIR = os.path.join(_PROJECT_ROOT, "index")
INDEX_DIR = _env("POCKET_INDEX_DIR", default=_DEFAULT_INDEX_DIR)

# 用户上传文档目录（Web UI 上传的原始文档存放处）
_DEFAULT_USER_DOCS_DIR = os.path.join(_PROJECT_ROOT, "user_docs")
USER_DOCS_DIR = _env("POCKET_USER_DOCS_DIR", default=_DEFAULT_USER_DOCS_DIR)

# 用户上传文档抽取出的三元组存放路径（build-index 会读取此文件）
USER_TRIPLES_PATH = os.path.join(USER_DOCS_DIR, "triples.txt")

# ========================
# Embedding 模型
# ========================
# BAAI/bge-small-zh-v1.5: 中文轻量级 embedding 模型，512 维，效果好且速度快
# 优先使用本地已下载的模型，若不存在则从 HuggingFace 下载
_LOCAL_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "BAAI", "bge-small-zh-v1___5")
_DEFAULT_EMBEDDING_MODEL = (
    _LOCAL_MODEL_PATH if os.path.exists(_LOCAL_MODEL_PATH) else "BAAI/bge-small-zh-v1.5"
)
EMBEDDING_MODEL = _env("POCKET_EMBEDDING_MODEL", default=_DEFAULT_EMBEDDING_MODEL)
# 运行时由 SentenceTransformer.get_sentence_embedding_dimension() 推断；
# 此处留 None 以避免与实际加载的模型维度不一致（如换用其他模型时 512 维硬编码会出错）。
EMBEDDING_DIM = None

# ========================
# 检索参数
# ========================
TOP_K = 5  # 默认检索 top-k 条最相关知识

# 重排序模型（开启 reranker 时使用）
# 默认 bge-reranker-v2-m3（多语言、效果优于 base）；国内首次使用需预先下载缓存，
# 或设 HF_ENDPOINT=https://hf-mirror.com 走镜像。设 HF_HUB_OFFLINE=1 强制只用本地缓存。
RERANKER_MODEL = _env("POCKET_RERANKER_MODEL", "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# 检索模式: vector / local / global / mix / kg_only / global_summary
# vector           - 纯向量检索
# local            - 基于实体匹配的局部检索（BFS 扩展邻域）
# global           - 基于关系关键词的全局检索
# mix              - 向量检索 + KG 检索融合
# kg_only          - 仅 KG 检索（不使用向量检索）
# global_summary   - 社区摘要检索（Louvain 分社区 + LLM 摘要 + 摘要向量召回，对标 MS GraphRAG Global Search）
# 默认 mix：实测 mix 检索质量优于纯向量，新用户首次体验应是最强模式。
SEARCH_MODE = os.environ.get("POCKET_SEARCH_MODE", "mix")

# 实体嵌入匹配相似度阈值
ENTITY_SIMILARITY_THRESHOLD = float(os.environ.get("POCKET_ENTITY_THRESHOLD", "0.5"))

# KG 局部检索 BFS 跳数
KG_SEARCH_HOPS = 2

# 融合策略: "weighted" (简单加权) 或 "rrf" (Reciprocal Rank Fusion)
FUSION_STRATEGY = os.environ.get("POCKET_FUSION_STRATEGY", "rrf").lower()

# RRF 融合的 k 参数，默认 60 (业界常用值)
RRF_K = int(os.environ.get("POCKET_RRF_K", "60"))

# 混合检索时向量结果权重 (0.0~1.0)，KG 结果权重 = 1 - VECTOR_WEIGHT
# 默认 0.3 偏向 KG：实测纯 KG 检索质量通常优于纯向量，
# VW=0.3 时 mix 接近 kg_only 效果，VW 越高越偏向向量。
# 同时影响 weighted 与 rrf 两种融合策略。
VECTOR_WEIGHT = float(os.environ.get("POCKET_VECTOR_WEIGHT", "0.3"))

# HyDE (假设性文档嵌入) 查询改写：用 LLM 生成假设性答案文档做向量检索，
# 提升短查询对长文档的召回率。需要 LLM；无 LLM 自动回退。默认关闭，按需开启。
HYDE_ENABLED = os.environ.get("POCKET_HYDE", "").lower() in ("1", "true", "yes", "on")

# 查询路由器：用 LLM 自动判断问题类型，选择最优检索模式（vector/local/global/mix/global_summary）。
# 用户不再需要手动选模式。需要 LLM；无 LLM 回退到 search_mode。默认关闭，按需开启。
QUERY_ROUTER_ENABLED = os.environ.get("POCKET_QUERY_ROUTER", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# 答案自检：LLM 生成后用一次调用校验答案是否被检索到的上下文支持。
# 发现幻觉(partial/none)时在答案前加警告前缀，并最多重试 1 次。需要 LLM；失败默认通过。默认关闭。
SELF_CHECK_ENABLED = os.environ.get("POCKET_SELF_CHECK", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Schema 驱动抽取：加载三元组时归一化碎片化关系名（"2018年亩产" → "产量"），
# LLM 新抽取时注入关系白名单约束。解决 match_relations 关键词匹配召回率低的问题。
# 默认关闭，按需开启。可用 POCKET_SCHEMA_PATH 指定自定义 schema JSON。
SCHEMA_ENABLED = os.environ.get("POCKET_SCHEMA_ENABLED", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SCHEMA_PATH = os.environ.get("POCKET_SCHEMA_PATH", "")

# Pagerank 在检索排序中的权重 (0.0-1.0)
PAGERANK_WEIGHT = float(os.environ.get("POCKET_PAGERANK_WEIGHT", "0.3"))

# ========================
# 社区层次摘要配置（P3：对标 MS GraphRAG Hierarchical Summaries）
# ========================
# 社区发现算法: "auto" 优先 Leiden（需 pip install leidenalg igraph）回退 Louvain
# 设置 "leiden" 强制使用 Leiden（未安装时报错），"louvain" 强制 Louvain
COMMUNITY_ALGORITHM = os.environ.get("POCKET_COMMUNITY_ALGORITHM", "auto")

# 是否启用层次社区摘要（global_summary 模式）
# 启用后会构建多层级社区树（默认 3 层），检索时支持 roll-up
# 关闭则用单层社区摘要（向后兼容）
COMMUNITY_HIERARCHICAL = os.environ.get(
    "POCKET_COMMUNITY_HIERARCHICAL", "true"
).lower() in ("1", "true", "yes", "on")

# 层次社区分辨率列表（升序：由粗到细），逗号分隔
# 默认 [0.5, 1.0, 2.0]：
#   0.5 → 粗粒度（少数大社区，适合全局归纳）
#   1.0 → 中粒度（Louvain 默认）
#   2.0 → 细粒度（多数小社区，适合具体细节）
COMMUNITY_RESOLUTIONS = [
    float(x.strip())
    for x in os.environ.get(
        "POCKET_COMMUNITY_RESOLUTIONS", "0.5,1.0,2.0"
    ).split(",")
    if x.strip()
]

# 单层社区摘要分辨率（COMMUNITY_HIERARCHICAL=false 时用）
COMMUNITY_RESOLUTION = float(os.environ.get("POCKET_COMMUNITY_RESOLUTION", "1.0"))

# 单社区喂给 LLM 的最大三元组数（超长截断）
# 40 条三元组 ~ 1500 token prompt，LLM 响应 5-10s
# 80 条会让 prompt 翻倍，大社区 LLM 调用 20-40s
COMMUNITY_MAX_TRIPLES = int(os.environ.get("POCKET_COMMUNITY_MAX_TRIPLES", "40"))

# 小社区阈值：实体数 < 此值的社区跳过 LLM 摘要（用实体列表兜底）
# 稀疏 KG 上很多 1-2 实体的小社区，调 LLM 既慢又没意义
COMMUNITY_MIN_SIZE = int(os.environ.get("POCKET_COMMUNITY_MIN_SIZE", "5"))

# 每层最多对 N 个大社区生成 LLM 摘要（按大小降序取 top-N）
# 防止稀疏图上 500+ 社区导致 LLM 调用爆炸
COMMUNITY_MAX_PER_LEVEL = int(os.environ.get("POCKET_COMMUNITY_MAX_PER_LEVEL", "50"))

# Roll-up 阈值：当所有子社区得分 ≥ 此值时，替换为父社区摘要
COMMUNITY_ROLLUP_THRESHOLD = float(
    os.environ.get("POCKET_COMMUNITY_ROLLUP_THRESHOLD", "0.7")
)

# ========================
# LLM 配置（按优先级依次尝试，配置任一即可）
# ========================

# --- 选项 1: SiliconFlow（硅基流动，默认推荐）---
# 申请地址: https://siliconflow.cn/
# 设置环境变量: set POCKET_SILICONFLOW_API_KEY="your-key"
SILICONFLOW_API_KEY = _env("POCKET_SILICONFLOW_API_KEY", "SILICONFLOW_API_KEY", "")
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = _env("POCKET_SILICONFLOW_MODEL", "SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# --- 选项 2: DeepSeek（推理能力强，有免费额度）---
# 申请地址: https://platform.deepseek.com/
# 设置环境变量: set POCKET_DEEPSEEK_API_KEY="your-key"
DEEPSEEK_API_KEY = _env("POCKET_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = _env("POCKET_DEEPSEEK_MODEL", "DEEPSEEK_MODEL", "deepseek-chat")

# --- 选项 3: DashScope (通义千问) ---
# 申请地址: https://dashscope.console.aliyun.com/
# 设置环境变量: set POCKET_DASHSCOPE_API_KEY="your-key"
DASHSCOPE_API_KEY = _env("POCKET_DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY", "")
DASHSCOPE_MODEL = _env("POCKET_DASHSCOPE_MODEL", "DASHSCOPE_MODEL", "qwen-plus")
DASHSCOPE_VLM_MODEL = _env("POCKET_DASHSCOPE_VLM_MODEL", "DASHSCOPE_VLM_MODEL", "qwen-vl-plus")
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# --- 选项 4: 通用 OpenAI 兼容 API ---
# 设置环境变量: set POCKET_OPENAI_API_KEY="your-key"
OPENAI_API_KEY = _env("POCKET_OPENAI_API_KEY", "OPENAI_API_KEY", "")
OPENAI_API_BASE = _env("POCKET_OPENAI_API_BASE", "OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = _env("POCKET_OPENAI_MODEL", "OPENAI_MODEL", "gpt-3.5-turbo")

# --- 选项 5: Ollama 本地模型 ---
# 设置环境变量: set POCKET_OLLAMA_MODEL="qwen2:7b"
OLLAMA_MODEL = _env("POCKET_OLLAMA_MODEL", "OLLAMA_MODEL", "")
OLLAMA_API_BASE = _env("POCKET_OLLAMA_API_BASE", "OLLAMA_API_BASE", "http://localhost:11434/v1")

# --- 选项 6: freellm-cn 本地大模型服务 ---
# freellm-cn 是 OpenAI 兼容的本地大模型服务网关
# 项目地址: https://github.com/your-org/freellm-cn
# 启动服务后，设置环境变量: set POCKET_FREELM_CN_API_KEY="your-key"
FREELM_CN_API_KEY = _env("POCKET_FREELM_CN_API_KEY", "FREELM_CN_API_KEY", "")
FREELM_CN_API_BASE = _env("POCKET_FREELM_CN_API_BASE", "FREELM_CN_API_BASE", "http://localhost:8000/v1")
FREELM_CN_MODEL = _env("POCKET_FREELM_CN_MODEL", "FREELM_CN_MODEL", "auto")

# LLM 请求超时（秒）。Ollama 本地推理较慢（7B 模型首 token ~10-30s），
# 旧默认 60s 在弱机器上频繁超时。默认 120s，可通过环境变量调整。
# 云端 API（SiliconFlow/DeepSeek/DashScope/OpenAI）通常 30s 足够，
# 但为统一配置先用一个值；用户可按需调小。
LLM_TIMEOUT = int(os.environ.get("POCKET_LLM_TIMEOUT", "120"))

# 多角色 LLM 配置：允许为抽取和查询分别指定不同的模型
# 如果未设置，则使用上方各 provider 的统一模型配置（向后兼容）
# 例如：用便宜的模型做抽取，用强的模型做问答
# POCKET_EXTRACT_MODEL=Qwen/Qwen2.5-14B-Instruct  # 抽取用大模型
# POCKET_QUERY_MODEL=Qwen/Qwen2.5-7B-Instruct     # 问答用小模型（更快）
EXTRACT_MODEL = os.environ.get("POCKET_EXTRACT_MODEL", "")
QUERY_MODEL = os.environ.get("POCKET_QUERY_MODEL", "")

# Gleaning 轮数（参考 microsoft/graphrag）
# 0 = 单轮抽取（向后兼容）
# 1 = 首轮 + 1 轮追问补漏（默认，开箱可用，召回率提升约 15-25%）
# 2+ = 多轮追问（收益递减，LLM 成本线性增加）
# 推荐值：1-2（超过 2 收益递减）
GLEANING_STEPS = int(os.environ.get("POCKET_GLEANING_STEPS", "1"))

# 实体语义别名字典路径
# 领域别名 → 规范名映射，align_entities 在 embedding 匹配前做 pre-normalize
# 解决中文缩写/同义词 embedding 相似度不足的问题（如 Bt vs 苏云金杆菌）
# 通用默认留空：不加载任何领域别名，避免对用户数据施加领域归一化。
# 需要时通过 POCKET_ENTITY_ALIASES_PATH 指向自定义 JSON（参考 data/entity_aliases.json 格式）
ENTITY_ALIASES_PATH = os.environ.get("POCKET_ENTITY_ALIASES_PATH", "")

# ========================
# 知识图谱处理配置
# ========================
# 反向链接关系集合：指定哪些关系需要建立反向链接
# 例如 "化学防治" — 当遇到 [病害|化学防治|药剂] 时，会在药剂的文本块中补充"可用于防治：病害"
# 支持通过环境变量配置，用逗号分隔：set POCKET_REVERSE_LINK_RELATIONS="化学防治,生物防治,农业防治"
_env_rels = os.environ.get("POCKET_REVERSE_LINK_RELATIONS", "")
REVERSE_LINK_RELATIONS = (
    {r.strip() for r in _env_rels.split(",") if r.strip()} if _env_rels else set()
)

# 是否自动推断反向链接关系（关系名中包含"防治/治疗/包含/属于/用于/引起/导致/产生"等关键词时自动加入）
AUTO_REVERSE_LINK = os.environ.get("POCKET_AUTO_REVERSE_LINK", "true").lower() == "true"

# 关系到自然语言的映射模板（可选，不配置则使用通用格式 "关系：尾实体"）
# 示例：{"症状表现": "症状表现为{tail}"}
# 配置方式：通过 JSON 文件路径或环境变量 JSON 字符串
RELATION_TEMPLATES = {}

# ========================
# RAG Prompt 模板
# ========================
# Prompt 外部化：优先从 YAML 文件加载，支持 POCKET_PROMPTS_PATH 自定义路径
_DEFAULT_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.yaml")

def _load_prompts() -> dict:
    """从 YAML 文件加载 prompt 模板，失败时回退到硬编码默认值。"""
    prompts_path = os.environ.get("POCKET_PROMPTS_PATH", _DEFAULT_PROMPTS_FILE)
    try:
        import yaml
        with open(prompts_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        # YAML 文件不可用时回退到硬编码默认值
        return {}

_PROMPTS = _load_prompts()

SYSTEM_PROMPT = _PROMPTS.get("system_prompt", """你是一个专业的知识问答助手。你的任务是基于提供的检索信息，给用户一个自然、流畅且准确的回答。

回答规则：
1. **硬事实约束**：所有涉及具体领域知识（如专有名词、核心数据、具体步骤、专业结论等）的结论，必须严格来自"检索到的知识信息"。
2. **允许软补白**：允许并鼓励使用自然的连接词（如"因此"、"需要注意"）、常识性解释和句式重组，把离散的检索结果揉成一段通顺的专家回答。
3. **严禁新增事实**：绝对不允许编造检索信息中没有的具体参数、新药剂、新方案。如果知识库没提具体细节（如用量），你可以用自然语言坦诚说明"现有资料未提及具体数值"，但不要生硬地拒绝。
4. **强制引用**：你陈述的每一个核心事实（非软补白），必须在句末或短语后用方括号标注来源，如 [1][2]。
5. 不要使用"根据提供的信息"、"知识库显示"这类机械的资料整理式开头，直接以专家的口吻回答即可。""")

USER_PROMPT_TEMPLATE = _PROMPTS.get("user_prompt_template", """请基于以下知识信息，回答用户问题。

## 检索到的知识信息：
{context}

## 用户问题：
{question}

请给出自然、专业的回答，并在关键事实后标注 [编号]：""")

MULTIHOP_SYSTEM_PROMPT = _PROMPTS.get("multihop_system_prompt", "")
CONVERSATION_REWRITE_PROMPT = _PROMPTS.get("conversation_rewrite_prompt", "")
HYDE_PROMPT = _PROMPTS.get("hyde_prompt", "")

# ========================
# 硬约束防幻觉：检索为空/低分时强制拒答
# ========================
# 总开关：检索为空或低质量时跳过 LLM，直接返回"知识库中未找到相关信息"
REFUSE_ON_EMPTY_RETRIEVAL = (
    os.environ.get("POCKET_REFUSE_ON_EMPTY", "true").lower() == "true"
)
# 纯向量模式拒答阈值：top1 余弦相似度低于此值视为检索不相关
VECTOR_REFUSE_THRESHOLD = float(os.environ.get("POCKET_VECTOR_REFUSE_THRESHOLD", "0.3"))
# 拒答时的固定回复
REFUSE_ANSWER_TEXT = "知识库中未找到相关信息。"
