"""
PocketGraphRAG 自动知识图谱抽取工具 (升级版)

功能：
从 TXT / Markdown / PDF / 网页等多源文本中，
使用 LLM 自动抽取高质量知识图谱三元组。

核心优化：
1. 高质量 Prompt - few-shot + 思维链 + 格式强化
2. 语义切分 - 按段落/句子切分，保留完整语义
3. 实体对齐 - 合并相似实体，统一命名
4. 置信度评分 - 给每条三元组打质量分
5. 后处理 - 去重 + 矛盾检测 + 质量过滤
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

EXTRACT_PROMPT_V2 = """你是一个专业的知识图谱抽取专家。请从给定文本中抽取实体和关系，构建高质量知识图谱三元组。

## 抽取原则

1. **完整性**：抽取文本中所有重要的实体及其关系，不要遗漏关键信息
2. **准确性**：严格基于文本内容，不要添加文本中没有的信息
3. **规范性**：
   - 实体名称简洁完整，使用最常用的称呼
   - 关系名称标准化（动宾结构或名词短语），方向正确
   - 同一实体在全文中使用统一名称

4. **粒度适中**：
   - 实体粒度：概念、事物、属性值等
   - 关系粒度：直接关联，避免过度细化

## 输出格式（严格 JSON）

```json
{
  "triples": [
    {
      "head": "头实体",
      "relation": "关系",
      "tail": "尾实体",
      "confidence": 0.95,
      "evidence": "文本中支持该三元组的原文片段"
    }
  ]
}
```

### 置信度评分标准
- **0.9-1.0**：文本明确陈述，信息完整准确
- **0.7-0.89**：文本有提及但不够直接，或部分信息需推断
- **0.5-0.69**：间接暗示或上下文推断，有一定不确定性

## 示例

输入文本：
"盗梦空间是克里斯托弗·诺兰执导的科幻电影，由莱昂纳多·迪卡普里奥主演。
该片于2010年上映，全球票房超过8亿美元。盗梦空间的核心概念是进入他人梦境窃取秘密。"

输出：
```json
{
  "triples": [
    {
      "head": "盗梦空间",
      "relation": "导演",
      "tail": "克里斯托弗·诺兰",
      "confidence": 0.98,
      "evidence": "盗梦空间是克里斯托弗·诺兰执导的科幻电影"
    },
    {
      "head": "盗梦空间",
      "relation": "主演",
      "tail": "莱昂纳多·迪卡普里奥",
      "confidence": 0.95,
      "evidence": "由莱昂纳多·迪卡普里奥主演"
    },
    {
      "head": "盗梦空间",
      "relation": "属于",
      "tail": "科幻电影",
      "confidence": 0.96,
      "evidence": "盗梦空间是克里斯托弗·诺兰执导的科幻电影"
    },
    {
      "head": "盗梦空间",
      "relation": "上映年份",
      "tail": "2010年",
      "confidence": 0.97,
      "evidence": "该片于2010年上映"
    },
    {
      "head": "盗梦空间",
      "relation": "票房",
      "tail": "超过8亿美元",
      "confidence": 0.95,
      "evidence": "全球票房超过8亿美元"
    },
    {
      "head": "盗梦空间",
      "relation": "核心概念",
      "tail": "进入他人梦境窃取秘密",
      "confidence": 0.96,
      "evidence": "盗梦空间的核心概念是进入他人梦境窃取秘密"
    }
  ]
}
```

## 注意事项

- 优先输出 JSON 格式；若 JSON 解析失败，可改用 delimiter 格式（见下）
- 每个三元组必须有 evidence（原文依据）
- 置信度要客观反映信息的确定性
- 关系方向要正确（A 防治 B ≠ B 防治 A）
- 实体名用最规范的全称，避免简称

## 备选输出格式（delimiter，仅在 JSON 输出困难时使用）

如果因为嵌套引号、特殊字符等原因无法输出合法 JSON，可改用以下 delimiter 格式，
每行一条三元组，字段用 `|` 分隔：

```
<|#|>头实体|关系|尾实体|置信度|原文依据<|#|>
```

示例：
```
<|#|>盗梦空间|上映年份|2010年|0.95|该片于2010年上映<|#|>
<|#|>盗梦空间|核心概念|进入他人梦境窃取秘密|0.90|盗梦空间的核心概念是进入他人梦境<|#|>
```
"""


# ========================
# 实体名质量清洗
# ========================

# 句内停顿标点：出现这些几乎必是句子片段而非单个实体
_SENTENCE_PUNCT = "、，,。；;！？!?"
# 用法/施药方法动词：与数字同时出现时，说明是"用量+方法描述"被当成了实体
_METHOD_VERBS = ("喷雾", "浸种", "稀释", "拌种", "喷洒", "灌根", "涂抹", "熏蒸")
# 单位词（数字+单位组合识别用，如 "0.1kg/亩"、"1%"、"0.5天"）
_UNITS_PATTERN = (
    r"(kg|克|公斤|千克|亩|公顷|ml|毫升|升|%|MPa|度|天|日|小时|分钟|次|倍|"
    r"cm|米|mm|ppm|g|L|个|株|穗|叶|粒|kg/亩|ml/亩|公斤/亩|千克/公顷|g/m²)"
)
# 日期关键词（纯日期识别用）
_DATE_KEYWORDS = ("年", "月", "日", "上旬", "中旬", "下旬", "初", "末")


def is_low_quality_entity(name: str) -> Optional[str]:
    """判断实体名是否为低质量（句子片段 / 用法描述 / LLM 占位符 / 纯数值单位日期公式）。

    返回:
        命中原因字符串（如 'sent_punct'）；None 表示是合格实体。

    设计依据：对真实知识图谱数据集的实测 + HotpotQA/MuSiQue 风格实体的通用性验证。
    - 原 4 类规则可剔除约 9% 的噪声三元组（句子型实体、LLM 占位符、用量描述当实体）
    - v0.3.3 新增 4 类规则（pure_digit/digit_with_unit/pure_date/formula）：
      在真实数据集上额外过滤 14.5% 的漏网垃圾实体（纯数字、数字+单位、纯日期、公式），
      在 HotpotQA 英文实体上误伤率 <10%（仅 4 位年份/书名 1984 边缘案例）。
    - 不误伤 "浸种"/"拌种"（不含数字）、"80%乙蒜素乳油"（含 % 但非纯单位）、
      "Xanthomonas oryzae"（拉丁学名）、"NY/T 391-2013"（标准号）。
    """
    if not name or not name.strip():
        return "empty"
    name = name.strip()
    # 1. 含句内停顿标点 → 句子片段
    if any(p in name for p in _SENTENCE_PUNCT):
        return "sent_punct"
    # 2. LLM 占位符 / 含下划线（中文 KG 里几乎不会出现 _）
    if "_" in name or name.lower().startswith("llm"):
        return "placeholder"
    # 3. "…等" 结尾的列举性短语
    if name.endswith("等") and len(name) > 4:
        return "enum"
    # 4. 用法描述当实体：含方法动词 + 含数字（如 "80%乙蒜素2000倍液浸种48小时"）
    #    合法的短方法实体（"浸种"/"拌种"）不含数字，不会被误伤
    if any(v in name for v in _METHOD_VERBS) and any(c.isdigit() for c in name):
        return "method_desc"
    # 5. 纯数字串：过滤非 4 位年份整数的纯数字
    #    4 位整数 (1900-2099) 保留，可能是年份/书名《1984》/编号；其他纯数字（1.2/110.56）是垃圾
    if re.fullmatch(r"[\d.]+", name):
        if not (re.fullmatch(r"\d{4}", name) and 1900 <= int(name) <= 2099):
            return "pure_digit"
    # 6. 数字+单位（如 "0.1kg/亩"、"1%"、"0.5天"、"3次"）
    #    含 % 但有非数字字符的（如 "80%乙蒜素乳油"）不会被误伤
    if re.fullmatch(
        r"[\d.]+\s*" + _UNITS_PATTERN + r"(/" + _UNITS_PATTERN + r")?", name
    ):
        return "digit_with_unit"
    # 7. 纯日期/时间：整串都是日期字符（数字+年月日上中下旬初末 + 连接符）
    #    避免误伤 "2018年亩产500公斤" 这种含日期但非纯日期的描述
    if (
        re.fullmatch(r"[\d年月日时分秒上中下旬初末/\-至~\s]+", name)
        and any(c.isdigit() for c in name)
        and any(kw in name for kw in _DATE_KEYWORDS)
    ):
        return "pure_date"
    # 8. 公式类（含 = 且含数字，如 "理论产量 = 每亩穗数 × ..."）
    if "=" in name and any(c.isdigit() for c in name):
        return "formula"
    return None


@dataclass
class Triple:
    """三元组数据结构"""

    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    evidence: str = ""
    source_chunk: int = 0

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.head, self.relation, self.tail)

    def key(self) -> Tuple[str, str, str]:
        return (
            self.head.strip().lower(),
            self.relation.strip().lower(),
            self.tail.strip().lower(),
        )


@dataclass
class ExtractionResult:
    """抽取结果"""

    triples: List[Triple] = field(default_factory=list)
    raw_triples_count: int = 0
    aligned_count: int = 0
    removed_duplicates: int = 0
    removed_low_quality: int = 0

    @property
    def avg_confidence(self) -> float:
        if not self.triples:
            return 0.0
        return sum(t.confidence for t in self.triples) / len(self.triples)

    @property
    def high_quality_count(self) -> int:
        return sum(1 for t in self.triples if t.confidence >= 0.8)

    @property
    def entities(self) -> Set[str]:
        ents = set()
        for t in self.triples:
            ents.add(t.head)
            ents.add(t.tail)
        return ents


# ========================
# 1. 语义切分
# ========================


def semantic_chunk_text(
    text: str, max_chunk_size: int = 1200, min_chunk_size: int = 200
) -> List[str]:
    """语义感知的文本切分

    优先按段落切分，段落太大再按句子切分，
    避免从句子中间切断。

    Args:
        text: 输入文本
        max_chunk_size: 每个块的最大字符数
        min_chunk_size: 每个块的最小字符数（太小的块会合并）

    Returns:
        切分后的文本块列表
    """
    if not text or not text.strip():
        return []

    # 规范化换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 先按段落切分（空行分隔）
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_len = len(para)

        # 单个段落已经超过 max_chunk_size，需要按句子切分
        if para_len > max_chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            # 按句子切分
            sentences = _split_sentences(para)
            sent_chunk = []
            sent_size = 0

            for sent in sentences:
                sent_len = len(sent)
                if sent_size + sent_len > max_chunk_size and sent_chunk:
                    chunks.append("".join(sent_chunk))
                    sent_chunk = []
                    sent_size = 0
                sent_chunk.append(sent)
                sent_size += sent_len

            if sent_chunk:
                chunks.append("".join(sent_chunk))

            continue

        # 普通段落，看看能不能加入当前块
        if current_size + para_len + 2 > max_chunk_size and current_chunk:
            # 当前块满了，先保存
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(para)
        current_size += para_len + 2  # +2 for \n\n

    # 处理最后一块
    if current_chunk:
        last_chunk = "\n\n".join(current_chunk)
        # 如果最后一块太小，且前面有块，尝试合并到前一块
        if len(chunks) > 0 and len(last_chunk) < min_chunk_size:
            prev = chunks[-1]
            if len(prev) + len(last_chunk) + 2 <= max_chunk_size * 1.2:
                chunks[-1] = prev + "\n\n" + last_chunk
            else:
                chunks.append(last_chunk)
        else:
            chunks.append(last_chunk)

    return chunks


def _split_sentences(text: str) -> List[str]:
    """按句子切分中文/英文文本"""
    # 中文句子分隔符：。！？；
    # 英文句子分隔符：. ! ? ;
    sentences = re.split(r"(?<=[。！？；.!?;])", text)
    # 去掉空字符串
    return [s for s in sentences if s.strip()]


# ========================
# 2. 三元组抽取
# ========================


# P0-3: delimiter 格式解析器，作为 JSON 解析失败的 fallback
# 格式: <|#|>头实体|关系|尾实体|置信度|原文依据<|#|>
_DELIM_MARKER = "<|#|>"


def _parse_delimiter_format(
    result: str, schema=None, chunk_index: int = 0
) -> list:
    """解析 delimiter 格式的 LLM 输出

    格式: <|#|>头实体|关系|尾实体|置信度|原文依据<|#|>
    每行一条，字段用 | 分隔。置信度和原文依据可省略。

    Returns:
        Triple 列表；解析失败返回空列表
    """
    if not result or _DELIM_MARKER not in result:
        return []

    triples = []
    # 按行扫描，提取每行中 <|#|>...<|#|> 包裹的内容
    for line in result.splitlines():
        if _DELIM_MARKER not in line:
            continue
        # 提取 <|#|> 和 <|#|> 之间的内容
        start = line.find(_DELIM_MARKER)
        end = line.rfind(_DELIM_MARKER)
        if start == end:
            continue
        content = line[start + len(_DELIM_MARKER) : end].strip()
        if not content:
            continue

        parts = content.split("|")
        if len(parts) < 3:
            continue

        head = parts[0].strip()
        relation = parts[1].strip()
        tail = parts[2].strip()
        confidence = 0.7
        evidence = ""

        if len(parts) >= 4:
            try:
                confidence = float(parts[3].strip())
            except ValueError:
                pass
        if len(parts) >= 5:
            evidence = parts[4].strip()

        # 复用 JSON 解析路径的有效性和质量检查
        if not head or not relation or not tail:
            continue
        if len(head) > 100 or len(tail) > 200:
            continue
        if is_low_quality_entity(head) or is_low_quality_entity(tail):
            continue

        confidence = max(0.0, min(1.0, confidence))

        if schema:
            relation = schema.normalize_relation(relation)

        triples.append(
            Triple(
                head=head,
                relation=relation,
                tail=tail,
                confidence=confidence,
                evidence=evidence,
                source_chunk=chunk_index,
            )
        )

    return triples


def _parse_triples_result(
    result: str, schema=None, chunk_index: int = 0
) -> List[Triple]:
    """解析 LLM 返回的三元组结果（JSON 优先，delimiter 兜底）

    Args:
        result: LLM 原始返回文本
        schema: RelationSchema 实例（可选，用于归一化关系名）
        chunk_index: 文本块索引

    Returns:
        Triple 列表；解析失败返回空列表
    """
    if not result:
        return []

    # 清洗 markdown 代码块包裹
    result = result.strip()
    if result.startswith("```json"):
        result = result[7:].strip()
        if result.endswith("```"):
            result = result[:-3].strip()
    elif result.startswith("```"):
        result = result[3:].strip()
        if result.endswith("```"):
            result = result[:-3].strip()

    # 尝试提取 JSON 对象（处理 LLM 输出多余文字的情况）
    json_match = re.search(r'\{[\s\S]*"triples"[\s\S]*\}', result)
    if json_match:
        result = json_match.group(0)

    try:
        data = json.loads(result)
        raw_triples = data.get("triples", [])

        triples = []
        for item in raw_triples:
            if not isinstance(item, dict):
                continue

            head = str(item.get("head", "")).strip()
            relation = str(item.get("relation", "")).strip()
            tail = str(item.get("tail", "")).strip()
            confidence = float(item.get("confidence", 0.7))
            evidence = str(item.get("evidence", "")).strip()

            # 基本有效性检查
            if not head or not relation or not tail:
                continue
            if len(head) > 100 or len(tail) > 200:
                continue  # 太长的实体/属性值可能有问题

            # 实体名质量检查：拦截句子片段 / 用法描述 / LLM 占位符
            if is_low_quality_entity(head) or is_low_quality_entity(tail):
                continue

            # 置信度范围限制
            confidence = max(0.0, min(1.0, confidence))

            # Schema 驱动：归一化关系名（碎片化 → 标准名）
            if schema:
                relation = schema.normalize_relation(relation)

            triples.append(
                Triple(
                    head=head,
                    relation=relation,
                    tail=tail,
                    confidence=confidence,
                    evidence=evidence,
                    source_chunk=chunk_index,
                )
            )

        return triples

    except json.JSONDecodeError as e:
        # P0-3: JSON 解析失败时尝试 delimiter 格式 fallback
        delim_triples = _parse_delimiter_format(result, schema, chunk_index)
        if delim_triples:
            print(f"[警告] JSON 解析失败，改用 delimiter 格式解析到 {len(delim_triples)} 条三元组")
            return delim_triples
        print(f"[错误] JSON 解析失败: {e}")
        print(f"LLM 原始返回（前500字符）:\n{result[:500]}")
        return []


# Gleaning 追问 prompt（参考 microsoft/graphrag 的 gleaning 设计）
_GLEANING_PROMPT = """你已经从文本中抽取了以下三元组：

已抽取实体：{entities}
已抽取三元组数：{n_triples}

请重新审视原文，检查是否遗漏了任何重要的实体或关系。仅输出**遗漏的新增**三元组（不要重复已抽取的），使用相同的 JSON 格式：

```json
{{"triples": [{{"head": "...", "relation": "...", "tail": "...", "confidence": 0.9, "evidence": "..."}}]}}
```

如果确认没有遗漏，输出：{{"triples": []}}

待复查原文：
\"\"\"{text}\"\"\"
"""


def extract_triples_from_text(
    text: str,
    chunk_index: int = 0,
    temperature: float = 0.1,
    schema=None,
    gleaning_steps: int = 0,
) -> List[Triple]:
    """调用 LLM 从文本中抽取三元组（新版，带置信度和证据）

    Args:
        text: 待抽取文本
        chunk_index: 文本块索引（用于追踪来源）
        temperature: LLM 温度
        schema: RelationSchema 实例，注入 prompt 约束 + 抽取后归一化关系名。
            None 则不约束（保持向后兼容）
        gleaning_steps: Gleaning 轮数（参考 microsoft/graphrag）。
            0 = 单轮抽取（默认，向后兼容）；
            1 = 首轮 + 1 轮追问补漏；
            N = 首轮 + N 轮追问。每轮会问 LLM "是否遗漏"，合并去重。

    Returns:
        Triple 列表
    """
    from .llm import call_llm, has_llm

    if not has_llm():
        print("[错误] 未配置 LLM API Key 或 Ollama，无法进行自动抽取。")
        return []

    # Schema 驱动：注入关系白名单约束到 prompt
    schema_constraint = schema.build_prompt_constraint() if schema else ""
    prompt = (
        EXTRACT_PROMPT_V2 + schema_constraint + '\n\n待抽取文本:\n"""' + text + '"""'
    )

    result = call_llm(
        "你是一个知识图谱抽取专家，只输出 JSON 格式的三元组。",
        prompt,
        temperature=temperature,
        max_tokens=3000,
        stream=False,
        role="extract",
    )

    if not result:
        print("[错误] LLM 调用失败或返回为空。")
        return []

    triples = _parse_triples_result(result, schema, chunk_index)

    # ====== Gleaning：多轮追问补漏（参考 microsoft/graphrag） ======
    if gleaning_steps > 0 and triples:
        seen_keys = {t.key() for t in triples}
        for round_i in range(1, gleaning_steps + 1):
            entities_summary = "、".join(sorted({t.head for t in triples} | {t.tail for t in triples})[:30])
            gleaning_prompt = _GLEANING_PROMPT.format(
                entities=entities_summary,
                n_triples=len(triples),
                text=text,
            )
            gleaning_result = call_llm(
                "你是知识图谱抽取质检员，只输出 JSON 格式的遗漏三元组。",
                gleaning_prompt,
                temperature=temperature,
                max_tokens=2000,
                stream=False,
                role="extract",
            )

            if not gleaning_result:
                print(f"[gleaning] 第 {round_i} 轮无返回，结束补漏")
                break

            new_triples = _parse_triples_result(gleaning_result, schema, chunk_index)
            # 合并去重
            added = 0
            for t in new_triples:
                if t.key() not in seen_keys:
                    seen_keys.add(t.key())
                    triples.append(t)
                    added += 1

            print(f"[gleaning] 第 {round_i} 轮新增 {added} 条三元组（累计 {len(triples)} 条）")
            if added == 0:
                # 本轮无新增，提前结束
                break

    return triples


# ========================
# 异步并发抽取（P1-3）
# ========================

async def extract_triples_from_text_async(
    text: str,
    chunk_index: int = 0,
    temperature: float = 0.1,
    schema=None,
    gleaning_steps: int = 0,
) -> List[Triple]:
    """extract_triples_from_text 的异步版本

    内部用 acall_llm（thread-based），不阻塞事件循环。
    gleaning 循环也改为 await。
    """
    from .llm import acall_llm, has_llm

    if not has_llm():
        print("[错误] 未配置 LLM API Key 或 Ollama，无法进行自动抽取。")
        return []

    schema_constraint = schema.build_prompt_constraint() if schema else ""
    prompt = (
        EXTRACT_PROMPT_V2 + schema_constraint + '\n\n待抽取文本:\n"""' + text + '"""'
    )

    result = await acall_llm(
        "你是一个知识图谱抽取专家，只输出 JSON 格式的三元组。",
        prompt,
        temperature=temperature,
        max_tokens=3000,
    )

    if not result:
        print("[错误] LLM 调用失败或返回为空。")
        return []

    triples = _parse_triples_result(result, schema, chunk_index)

    # Gleaning 循环（异步）
    if gleaning_steps > 0 and triples:
        seen_keys = {t.key() for t in triples}
        for round_i in range(1, gleaning_steps + 1):
            entities_summary = "、".join(sorted({t.head for t in triples} | {t.tail for t in triples})[:30])
            gleaning_prompt = _GLEANING_PROMPT.format(
                entities=entities_summary,
                n_triples=len(triples),
                text=text,
            )
            gleaning_result = await acall_llm(
                "你是知识图谱抽取质检员，只输出 JSON 格式的遗漏三元组。",
                gleaning_prompt,
                temperature=temperature,
                max_tokens=2000,
            )

            if not gleaning_result:
                print(f"[gleaning] 第 {round_i} 轮无返回，结束补漏")
                break

            new_triples = _parse_triples_result(gleaning_result, schema, chunk_index)
            added = 0
            for t in new_triples:
                if t.key() not in seen_keys:
                    seen_keys.add(t.key())
                    triples.append(t)
                    added += 1

            print(f"[gleaning] 第 {round_i} 轮新增 {added} 条三元组（累计 {len(triples)} 条）")
            if added == 0:
                break

    return triples


async def extract_knowledge_graph_async(
    text: str,
    min_confidence: float = 0.6,
    align_threshold: float = 0.88,
    chunk_size: int = 1200,
    temperature: float = 0.1,
    verbose: bool = True,
    schema=None,
    gleaning_steps: int = 0,
    concurrency: int = 4,
) -> ExtractionResult:
    """extract_knowledge_graph 的异步并发版本

    Args:
        concurrency: 并发抽取的 chunk 数（Semaphore 控制）。默认 4。
            1 = 串行（等价于同步版本）；>1 = 并发，显著加速大文档抽取。
            注意：并发过高可能触发 LLM API 限流，建议 2-8。
    """
    import asyncio

    result = ExtractionResult()

    if not text or not text.strip():
        return result

    chunks = semantic_chunk_text(text, max_chunk_size=chunk_size)
    if verbose:
        print(f"[1/5] 文本已切分为 {len(chunks)} 个语义块（并发度 {concurrency}）")

    sem = asyncio.Semaphore(max(1, concurrency))

    async def extract_one(idx: int, chunk: str):
        async with sem:
            if verbose:
                print(f"[2/5] 正在抽取第 {idx + 1}/{len(chunks)} 块...")
            return await extract_triples_from_text_async(
                chunk,
                chunk_index=idx,
                temperature=temperature,
                schema=schema,
                gleaning_steps=gleaning_steps,
            )

    tasks = [extract_one(i, c) for i, c in enumerate(chunks)]
    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_triples = []
    for i, r in enumerate(chunk_results):
        if isinstance(r, Exception):
            print(f"[错误] 第 {i + 1} 块抽取失败: {r}")
            continue
        all_triples.extend(r)
        if verbose:
            print(f"       第 {i + 1} 块抽取到 {len(r)} 条三元组")

    result.raw_triples_count = len(all_triples)
    if verbose:
        print(f"[2/5] 原始抽取完成，共 {len(all_triples)} 条三元组")

    if not all_triples:
        return result

    if verbose:
        print("[3/5] 正在进行实体对齐...")
    aligned_triples, mapping = align_entities(
        all_triples, threshold=align_threshold, aliases=_get_default_aliases()
    )
    if verbose:
        print(f"       合并了 {len(mapping)} 个实体别名")

    if verbose:
        print("[4/5] 正在去重...")
    deduped, removed_dup = deduplicate_triples(aligned_triples)
    result.removed_duplicates = removed_dup
    if verbose:
        print(f"       去除了 {removed_dup} 条重复三元组")

    if verbose:
        print("[5/5] 质量过滤...")
    filtered, removed_low = filter_low_quality(deduped, min_confidence=min_confidence)
    result.removed_low_quality = removed_low
    result.triples = filtered

    if verbose:
        high_qual = sum(1 for t in filtered if t.confidence >= 0.8)
        avg_conf = (
            sum(t.confidence for t in filtered) / len(filtered) if filtered else 0
        )
        print(f"\n{'=' * 50}")
        print(f"抽取完成！最终 {len(filtered)} 条三元组")
        print(f"  - 高质量 (≥0.8): {high_qual} 条")
        print(f"  - 平均置信度: {avg_conf:.3f}")
        print(f"{'=' * 50}")

    return result
# ========================


def _load_entity_aliases(path: str = "") -> Dict[str, str]:
    """加载实体语义别名字典

    通用 GraphRAG 工具默认不加载任何领域别名字典（避免领域泄露）。
    仅当 path 显式指定自定义文件时才加载。内置的
    PocketGraphRAG/data/entity_aliases.json 仅作为参考示例（领域别名格式示例），
    不会自动应用。

    Returns:
        {别名: 规范名} 字典；path 为空或加载失败返回空字典（通用默认）
    """
    import json
    import os

    if not path:
        # 通用默认：不加载任何领域别名，避免对用户数据施加领域归一化
        return {}

    candidate = path

    if not os.path.exists(candidate):
        return {}

    try:
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
        aliases = {}
        for _group, mapping in data.get("aliases", {}).items():
            for alias, canonical in mapping.items():
                # alias == canonical 也保留，确保 canonical 自身在字典里
                aliases[alias] = canonical
        return aliases
    except Exception as e:
        print(f"[警告] 加载实体别名字典失败: {e}，跳过别名 pre-normalize")
        return {}


# 模块级别名缓存（避免每次抽取都读文件）
_aliases_cache: Optional[Dict[str, str]] = None


def _get_default_aliases() -> Dict[str, str]:
    """获取默认别名字典（带模块级缓存）"""
    global _aliases_cache
    if _aliases_cache is None:
        from .config import ENTITY_ALIASES_PATH

        _aliases_cache = _load_entity_aliases(ENTITY_ALIASES_PATH)
    return _aliases_cache


def align_entities(
    triples: List[Triple],
    threshold: float = 0.88,
    aliases: Optional[Dict[str, str]] = None,
) -> Tuple[List[Triple], Dict[str, str]]:
    """实体对齐 - 合并相似的实体名

    策略（三层级联）：
    0. 语义别名 pre-normalize（领域别名表，解决中文缩写 embedding 匹配不足）
    1. 规则匹配（去除括号注释、全半角统一、首尾标点清理）
    2. Embedding 相似度匹配（≥threshold 视为同一实体）

    Canonical 选择：按出现频率最高 → 频率相同取最长名（更完整）
    旧实现只取最长名，导致 "Bt"(高频) vs "苏云金杆菌"(低频) 选错 canonical。

    Args:
        triples: 三元组列表
        threshold: embedding 相似度阈值（0-1）
        aliases: 语义别名字典 {别名: 规范名}。None 时不启用别名 pre-normalize

    Returns:
        (对齐后的三元组, 实体映射字典 {旧名: 规范名})
    """
    if not triples:
        return triples, {}

    # 统计实体出现频率（用于 canonical 选择）
    freq: Dict[str, int] = {}
    for t in triples:
        freq[t.head] = freq.get(t.head, 0) + 1
        freq[t.tail] = freq.get(t.tail, 0) + 1

    # 第零步：语义别名 pre-normalize（领域别名表）
    alias_map: Dict[str, str] = {}
    if aliases:
        for t in triples:
            for name in (t.head, t.tail):
                if name in aliases and name != aliases[name]:
                    alias_map[name] = aliases[name]

    # 第一步：规则标准化（对未命中别名的实体）
    entities_to_normalize = set()
    for t in triples:
        for name in (t.head, t.tail):
            if name not in alias_map:
                entities_to_normalize.add(name)
    normalized_map = _rule_based_normalize(entities_to_normalize)

    # 合并别名映射 + 规则映射
    rule_map: Dict[str, str] = {}
    rule_map.update(alias_map)
    rule_map.update(normalized_map)

    # 第二步：Embedding 相似度匹配（传入 frequency 用于 canonical 选择）
    embedding_map = _embedding_based_match(
        set(rule_map.values()), threshold=threshold, freq=freq
    )

    # 合并映射（过滤 self-mapping，避免 mapping 里出现 a→a 的噪声）
    full_map = {}
    for orig, norm in rule_map.items():
        final = embedding_map.get(norm, norm)
        if final != orig:  # self-mapping 不进 full_map
            full_map[orig] = final

    # 应用映射到三元组
    aligned_triples = []
    for t in triples:
        new_head = full_map.get(t.head, t.head)
        new_tail = full_map.get(t.tail, t.tail)

        # 如果头尾变成同一个实体，跳过（自环通常没意义）
        if new_head == new_tail:
            continue

        aligned_triples.append(
            Triple(
                head=new_head,
                relation=t.relation,
                tail=new_tail,
                confidence=t.confidence,
                evidence=t.evidence,
                source_chunk=t.source_chunk,
            )
        )

    return aligned_triples, full_map


def _rule_based_normalize(entities: Set[str]) -> Dict[str, str]:
    """基于规则的实体名标准化

    处理：
    - 去除括号注释："盗梦空间(电影)" -> "盗梦空间"
    - 全半角统一
    - 去除首尾标点
    - 别称合并（如 "Bt" -> "苏云金杆菌" 需要领域知识，这里先做简单处理）
    """
    norm_map = {}

    for ent in entities:
        original = ent
        text = ent.strip()

        # 全角转半角
        text = _fullwidth_to_halfwidth(text)

        # 去除括号及内容（常见格式：全称(简称) 或 简称(全称)）
        # 保留更常用的那个——通常括号外的更常用
        text = re.sub(r"[（(][^）)]*[）)]", "", text).strip()

        # 去除首尾标点
        text = text.strip('，。、；：""（）【】《》!！?？,.')

        # 统一某些常见格式
        # 比如 "XX剂" 和 "XX农药" 不完全相同，不强制合并

        if text:
            norm_map[original] = text
        else:
            norm_map[original] = original  # 退化情况

    return norm_map


def _embedding_based_match(
    entities: Set[str],
    threshold: float = 0.88,
    freq: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    """基于 Embedding 相似度的实体匹配

    将非常相似的实体合并为同一个。Canonical 选择策略：
    1. 频率最高（出现次数多的实体名更可能是规范名）
    2. 频率相同时取最长名（更完整）

    Args:
        entities: 待匹配的实体集合（通常已过规则标准化）
        threshold: 相似度阈值
        freq: 实体出现频率字典 {实体名: 次数}。None 时退化为按最长名选
    """
    # 尝试加载 Embedding 模型
    try:
        from sentence_transformers import SentenceTransformer

        from .config import EMBEDDING_MODEL

        if len(entities) <= 5:
            return {}  # 实体太少，没必要做

        model = SentenceTransformer(EMBEDDING_MODEL)
        entity_list = sorted(entities)
        embeddings = model.encode(entity_list, normalize_embeddings=True)

        # 计算相似度矩阵
        sim_matrix = embeddings @ embeddings.T

        # 构建合并映射
        merged = {}
        visited = set()

        def _canonical_key(name: str):
            """canonical 选择 key: (-频率, -长度, 名称) 频率高优先，相同则长优先"""
            f = freq.get(name, 0) if freq else 0
            return (-f, -len(name), name)

        for i in range(len(entity_list)):
            if entity_list[i] in visited:
                continue

            # 找所有与 i 高度相似的实体
            similar = [entity_list[i]]
            for j in range(i + 1, len(entity_list)):
                if entity_list[j] in visited:
                    continue
                if sim_matrix[i][j] >= threshold:
                    similar.append(entity_list[j])
                    visited.add(entity_list[j])

            visited.add(entity_list[i])

            if len(similar) > 1:
                # 按频率选 canonical：频率高优先，频率相同取最长名
                canonical = min(similar, key=_canonical_key)
                for s in similar:
                    if s != canonical:
                        merged[s] = canonical

        return merged

    except Exception:
        # 模型不可用，返回空映射
        return {}


def _fullwidth_to_halfwidth(text: str) -> str:
    """全角转半角"""
    result = []
    for char in text:
        code = ord(char)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    return "".join(result)


# ========================
# 4. 后处理 & 质量控制
# ========================


def deduplicate_triples(triples: List[Triple]) -> Tuple[List[Triple], int]:
    """三元组去重，重复的取置信度最高的

    去重规则：
    - 头尾关系完全相同（不区分大小写）
    - 保留置信度最高的，合并 evidence
    """
    best: Dict[Tuple[str, str, str], Triple] = {}
    removed = 0

    for t in triples:
        key = (
            t.head.strip().lower(),
            t.relation.strip().lower(),
            t.tail.strip().lower(),
        )

        if key in best:
            existing = best[key]
            # 决定保留哪个（置信度高的）
            if t.confidence > existing.confidence:
                keeper, other = t, existing
            else:
                keeper, other = existing, t
            # 合并 evidence 到 keeper（先合并再替换，避免丢失）
            if other.evidence and other.evidence not in keeper.evidence:
                if keeper.evidence:
                    keeper.evidence = keeper.evidence + "; " + other.evidence
                else:
                    keeper.evidence = other.evidence
            best[key] = keeper
            removed += 1
        else:
            best[key] = t

    return list(best.values()), removed


def filter_low_quality(
    triples: List[Triple],
    min_confidence: float = 0.6,
    min_head_len: int = 1,
    min_tail_len: int = 1,
) -> Tuple[List[Triple], int]:
    """过滤低质量三元组

    过滤条件：
    - 置信度低于阈值
    - 实体名太短（可能没意义）
    - 关系名太泛（如 "有", "是" 等过于宽泛的关系）
    """
    generic_relations = {
        "有",
        "是",
        "为",
        "属于一种",
        "是一种",
        "have",
        "is",
        "are",
        "be",
    }

    filtered = []
    removed = 0

    for t in triples:
        # 置信度过滤
        if t.confidence < min_confidence:
            removed += 1
            continue

        # 长度过滤
        if len(t.head) < min_head_len or len(t.tail) < min_tail_len:
            removed += 1
            continue

        # 泛关系过滤（只过滤完全没意义的）
        if t.relation.strip().lower() in generic_relations:
            removed += 1
            continue

        filtered.append(t)

    return filtered, removed


# ========================
# 完整抽取流程
# ========================


def extract_knowledge_graph(
    text: str,
    min_confidence: float = 0.6,
    align_threshold: float = 0.88,
    chunk_size: int = 1200,
    temperature: float = 0.1,
    verbose: bool = True,
    schema=None,
    gleaning_steps: int = 0,
) -> ExtractionResult:
    """完整的知识图谱抽取流程

    Args:
        text: 输入文本
        min_confidence: 最低置信度阈值
        align_threshold: 实体对齐相似度阈值
        chunk_size: 文本块大小
        temperature: LLM 温度
        verbose: 是否打印过程信息
        schema: RelationSchema 实例，约束 LLM 抽取 + 归一化关系名。None 不约束
        gleaning_steps: Gleaning 轮数（参考 microsoft/graphrag）。0=单轮（默认），
            N=首轮+N轮追问补漏。提升召回率但增加 LLM 调用次数。

    Returns:
        ExtractionResult 抽取结果
    """
    result = ExtractionResult()

    if not text or not text.strip():
        return result

    # Step 1: 语义切分
    chunks = semantic_chunk_text(text, max_chunk_size=chunk_size)
    if verbose:
        print(f"[1/5] 文本已切分为 {len(chunks)} 个语义块")

    # Step 2: 逐块抽取
    all_triples = []
    for i, chunk in enumerate(chunks, 1):
        if verbose:
            print(f"[2/5] 正在抽取第 {i}/{len(chunks)} 块...")
        triples = extract_triples_from_text(
            chunk, chunk_index=i - 1, temperature=temperature, schema=schema,
            gleaning_steps=gleaning_steps,
        )
        all_triples.extend(triples)
        if verbose:
            print(f"       本块抽取到 {len(triples)} 条三元组")

    result.raw_triples_count = len(all_triples)
    if verbose:
        print(f"[2/5] 原始抽取完成，共 {len(all_triples)} 条三元组")

    if not all_triples:
        return result

    # Step 3: 实体对齐
    if verbose:
        print("[3/5] 正在进行实体对齐...")
    aligned_triples, mapping = align_entities(
        all_triples, threshold=align_threshold, aliases=_get_default_aliases()
    )
    # 注意：实体对齐不会减少三元组数量，这里统计的是被合并的实体对数
    if verbose:
        print(f"       合并了 {len(mapping)} 个实体别名")

    # Step 4: 去重
    if verbose:
        print("[4/5] 正在去重...")
    deduped, removed_dup = deduplicate_triples(aligned_triples)
    result.removed_duplicates = removed_dup
    if verbose:
        print(f"       去除了 {removed_dup} 条重复三元组")

    # Step 5: 质量过滤
    if verbose:
        print("[5/5] 质量过滤...")
    filtered, removed_low = filter_low_quality(deduped, min_confidence=min_confidence)
    result.removed_low_quality = removed_low
    result.triples = filtered

    if verbose:
        high_qual = sum(1 for t in filtered if t.confidence >= 0.8)
        avg_conf = (
            sum(t.confidence for t in filtered) / len(filtered) if filtered else 0
        )
        print(f"\n{'=' * 50}")
        print(f"抽取完成！最终 {len(filtered)} 条三元组")
        print(
            f"  - 高质量 (≥0.8): {high_qual} 条 ({high_qual / len(filtered) * 100:.1f}%)"
        )
        print(f"  - 平均置信度: {avg_conf:.3f}")
        print(f"  - 去重移除: {removed_dup} 条")
        print(f"  - 低质量移除: {removed_low} 条")
        print(f"{'=' * 50}")

    return result


# ========================
# 流式抽取（用于 Web UI 实时进度显示）
# ========================


def extract_knowledge_graph_stream(
    text: str,
    min_confidence: float = 0.6,
    align_threshold: float = 0.88,
    chunk_size: int = 1200,
    temperature: float = 0.1,
    schema=None,
):
    """知识图谱抽取流式生成器（Web UI 进度显示用）。

    每个进度点 yield 一个 dict：
        {
            "phase": str,          # chunk/extract/extract_done/align/dedup/filter/done/empty
            "message": str,        # 给用户看的进度文案（Markdown）
            "triples": list,       # 累积的三元组 tuple 列表（部分→完整）
            "result": ExtractionResult | None,  # 仅 done 时非空
            "done": bool,
        }

    与 extract_knowledge_graph 等价的最终结果放在最后一个 yield 的 result 字段。
    """
    result = ExtractionResult()

    if not text or not text.strip():
        yield {
            "phase": "empty",
            "message": "文本为空，无内容可抽取",
            "triples": [],
            "result": None,
            "done": True,
        }
        return

    # Step 1: 语义切分
    chunks = semantic_chunk_text(text, max_chunk_size=chunk_size)
    yield {
        "phase": "chunk",
        "message": (
            f"### ⏳ 抽取中…\n\n"
            f"**[1/5] 文本切分完成**\n\n"
            f"共切分为 **{len(chunks)}** 个语义块，即将调用 LLM 逐块抽取…"
        ),
        "triples": [],
        "result": None,
        "done": False,
    }

    if not chunks:
        yield {
            "phase": "empty",
            "message": "切分后无有效内容",
            "triples": [],
            "result": None,
            "done": True,
        }
        return

    # Step 2: 逐块抽取（最慢，逐块报进度）
    all_triples = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        yield {
            "phase": "extract",
            "message": (
                f"### ⏳ 抽取中…\n\n"
                f"**[2/5] 正在抽取第 {i}/{total_chunks} 块**\n\n"
                f"当前已累计 **{len(all_triples)}** 条原始三元组"
            ),
            "triples": [t.to_tuple() for t in all_triples],
            "result": None,
            "done": False,
        }
        triples = extract_triples_from_text(
            chunk, chunk_index=i - 1, temperature=temperature, schema=schema
        )
        all_triples.extend(triples)

    result.raw_triples_count = len(all_triples)
    yield {
        "phase": "extract_done",
        "message": (
            f"### ⏳ 抽取中…\n\n"
            f"**[2/5] 原始抽取完成**\n\n"
            f"共抽取到 **{len(all_triples)}** 条原始三元组，进入后处理…"
        ),
        "triples": [t.to_tuple() for t in all_triples],
        "result": None,
        "done": False,
    }

    if not all_triples:
        yield {
            "phase": "empty",
            "message": "未能抽取到任何三元组，请检查文档内容或 LLM 配置。",
            "triples": [],
            "result": None,
            "done": True,
        }
        return

    # Step 3: 实体对齐
    yield {
        "phase": "align",
        "message": (
            "### ⏳ 抽取中…\n\n"
            "**[3/5] 实体对齐中**\n\n"
            "加载 Embedding 模型合并相似实体名，稍候…"
        ),
        "triples": [t.to_tuple() for t in all_triples],
        "result": None,
        "done": False,
    }
    aligned_triples, mapping = align_entities(
        all_triples, threshold=align_threshold, aliases=_get_default_aliases()
    )

    # Step 4: 去重
    yield {
        "phase": "dedup",
        "message": (
            f"### ⏳ 抽取中…\n\n"
            f"**[4/5] 去重中**\n\n"
            f"实体对齐合并 {len(mapping)} 个别名，正在去重…"
        ),
        "triples": [t.to_tuple() for t in aligned_triples],
        "result": None,
        "done": False,
    }
    deduped, removed_dup = deduplicate_triples(aligned_triples)
    result.removed_duplicates = removed_dup

    # Step 5: 质量过滤
    yield {
        "phase": "filter",
        "message": (
            f"### ⏳ 抽取中…\n\n"
            f"**[5/5] 质量过滤中**\n\n"
            f"去重移除 {removed_dup} 条，正在过滤低质量三元组…"
        ),
        "triples": [t.to_tuple() for t in deduped],
        "result": None,
        "done": False,
    }
    filtered, removed_low = filter_low_quality(deduped, min_confidence=min_confidence)
    result.removed_low_quality = removed_low
    result.triples = filtered

    yield {
        "phase": "done",
        "message": "",  # 由调用方格式化
        "triples": [t.to_tuple() for t in result.triples],
        "result": result,
        "done": True,
    }


# ========================
# CLI 入口
# ========================


def main():
    parser = argparse.ArgumentParser(
        description="PocketGraphRAG 高质量知识图谱抽取工具 v2"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="输入文本文件路径 (.txt, .md, .pdf 等)"
    )
    parser.add_argument(
        "--output", type=str, default="triples.txt", help="输出三元组文件路径"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="最低置信度阈值 (0-1, 默认 0.6)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1200, help="文本块大小（字符数，默认 1200）"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="LLM 温度 (默认 0.1)"
    )
    parser.add_argument(
        "--save-json", action="store_true", help="同时保存带置信度和证据的 JSON 格式"
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Schema JSON 文件路径，约束 LLM 抽取并归一化关系名。"
        "不传则用默认 schema（需 POCKET_SCHEMA_ENABLED=1）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[错误] 找不到输入文件: {args.input}")
        return

    # 读取文件内容（支持多种格式）
    from .data_importer import DataImporter

    importer = DataImporter()
    doc = importer.import_file(args.input)

    if doc is None:
        print(f"[错误] 无法读取文件: {args.input}")
        return

    text = doc.content
    if not text.strip():
        print("[错误] 文档内容为空。")
        return

    print(f"文件: {doc.source}")
    print(f"类型: {doc.source_type}")
    print(f"字符数: {len(text)}")
    print()

    # 执行抽取
    # Schema 驱动：--schema 指定 JSON 文件，或 POCKET_SCHEMA_ENABLED=1 用默认 schema
    schema = None
    from .config import SCHEMA_ENABLED, SCHEMA_PATH

    if args.schema:
        from .schema import RelationSchema

        schema = RelationSchema(schema_path=args.schema)
        print(f"[Schema] 已加载自定义 schema: {args.schema}")
    elif SCHEMA_ENABLED:
        from .schema import RelationSchema

        schema = RelationSchema(schema_path=SCHEMA_PATH or None)
        print("[Schema] 已启用默认 schema 归一化")

    result = extract_knowledge_graph(
        text,
        min_confidence=args.min_confidence,
        chunk_size=args.chunk_size,
        temperature=args.temperature,
        verbose=True,
        schema=schema,
    )

    if not result.triples:
        print("\n[警告] 未能抽取到任何有效三元组。")
        return

    # 保存三元组（传统格式）
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for t in result.triples:
            head = t.head.replace("|", "").replace("\n", " ")
            rel = t.relation.replace("|", "").replace("\n", " ")
            tail = t.tail.replace("|", "").replace("\n", " ")
            f.write(f"{head} | {rel} | {tail}\n")

    print(f"\n三元组已保存至: {args.output}")

    # 可选：保存 JSON 格式（含置信度和证据）
    if args.save_json:
        json_path = args.output.rsplit(".", 1)[0] + ".json"
        json_data = {
            "stats": {
                "total": len(result.triples),
                "raw_count": result.raw_triples_count,
                "avg_confidence": result.avg_confidence,
                "high_quality_count": result.high_quality_count,
                "removed_duplicates": result.removed_duplicates,
                "removed_low_quality": result.removed_low_quality,
                "source_file": doc.source,
                "source_type": doc.source_type,
            },
            "triples": [
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "confidence": t.confidence,
                    "evidence": t.evidence,
                    "source_chunk": t.source_chunk,
                }
                for t in result.triples
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"详细结果（含置信度+证据）已保存至: {json_path}")

    print(
        "\n你可以将此文件路径配置到 POCKET_DATA_PATH 环境变量中，然后运行 build_index 构建索引。"
    )


# ========================
# 向后兼容函数
# ========================


# 旧版 chunk_text（保持向后兼容）
def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """旧版简单按长度切分（保持向后兼容）

    新代码请使用 semantic_chunk_text() 获得更好的切分效果。
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# 旧版 extract_triples_from_text（保持向后兼容）
def extract_triples_from_text_legacy(text: str) -> List[Tuple[str, str, str]]:
    """旧版抽取函数（保持向后兼容，返回 tuple 列表）

    新代码请使用 extract_knowledge_graph() 获得完整质量控制。
    """
    triples = extract_triples_from_text(text)
    return [t.to_tuple() for t in triples]


if __name__ == "__main__":
    main()
