"""
VLM 多模态抽取模块
支持图片、扫描件的 OCR 和知识抽取
"""

import base64
import os
import re
from typing import List, Optional, Tuple

from .config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_API_URL,
    DASHSCOPE_VLM_MODEL,
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SILICONFLOW_API_BASE,
    SILICONFLOW_API_KEY,
    SILICONFLOW_MODEL,
)
from .logging_config import get_logger

logger = get_logger(__name__)

# ========================
# 图片处理工具
# ========================


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """将图片文件编码为 base64

    Args:
        image_path: 图片文件路径

    Returns:
        base64 编码字符串，失败返回 None
    """
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def get_image_mime_type(image_path: str) -> str:
    """根据文件扩展名判断 MIME 类型"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


def is_image_file(file_path: str) -> bool:
    """判断是否为图片文件"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


# ========================
# VLM 调用 - OpenAI 兼容格式
# ========================


def _call_vlm_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    image_base64: str,
    image_mime: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    label: str = "VLM",
) -> Optional[str]:
    """调用 OpenAI 兼容格式的 VLM API（支持图片输入）

    Args:
        api_base: API 基础地址
        api_key: API Key
        model: 模型名
        image_base64: 图片 base64 编码
        image_mime: 图片 MIME 类型
        prompt: 文本提示词
        temperature: 温度
        max_tokens: 最大 token 数
        label: 标签（用于日志）

    Returns:
        生成的文本，失败返回 None
    """
    try:
        import requests

        url = api_base.rstrip("/") + "/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_mime};base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(url, json=data, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()

        if result.get("choices") and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            return content.strip()

        return None

    except Exception as e:
        print(f"  [{label}] VLM 调用失败: {e}")
        return None


# ========================
# VLM 调用 - DashScope 特殊格式
# ========================


def _call_vlm_dashscope(
    image_base64: str,
    image_mime: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> Optional[str]:
    """调用 DashScope 通义千问 VLM（Qwen-VL）

    DashScope 的 OpenAI 兼容格式也支持图片输入，但部分模型需要特殊格式。
    这里直接用 OpenAI 兼容格式。
    """
    return _call_vlm_openai_compatible(
        DASHSCOPE_API_URL.rsplit("/chat/completions", 1)[0],
        DASHSCOPE_API_KEY,
        DASHSCOPE_VLM_MODEL,
        image_base64,
        image_mime,
        prompt,
        temperature,
        max_tokens,
        label="DashScope-VLM",
    )


# ========================
# 统一 VLM 调用入口
# ========================


def call_vlm(
    image_path: str = None,
    image_base64: str = None,
    prompt: str = "请详细描述这张图片的内容，包括所有文字、图表和关键信息。",
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> Optional[str]:
    """统一 VLM 调用入口，按优先级尝试不同后端

    优先级：DashScope → Ollama → SiliconFlow → OpenAI

    Args:
        image_path: 图片文件路径（和 image_base64 二选一）
        image_base64: 图片 base64 编码（和 image_path 二选一）
        prompt: 提示词
        temperature: 温度
        max_tokens: 最大 token 数

    Returns:
        VLM 输出的文本描述，失败返回 None
    """
    # 参数校验
    if not image_base64 and not image_path:
        return None

    if image_path and not image_base64:
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return None

    if image_path:
        image_mime = get_image_mime_type(image_path)
    else:
        image_mime = "image/jpeg"  # 默认

    # 1. DashScope（Qwen-VL，国内速度快，有免费额度）
    if DASHSCOPE_API_KEY:
        result = _call_vlm_dashscope(
            image_base64, image_mime, prompt, temperature, max_tokens
        )
        if result:
            return result

    # 2. SiliconFlow（可能有 VLM 模型）
    if SILICONFLOW_API_KEY:
        result = _call_vlm_openai_compatible(
            SILICONFLOW_API_BASE,
            SILICONFLOW_API_KEY,
            SILICONFLOW_MODEL,
            image_base64,
            image_mime,
            prompt,
            temperature,
            max_tokens,
            label="SiliconFlow-VLM",
        )
        if result:
            return result

    # 3. Ollama 本地 VLM（llava, bakllava 等）
    if OLLAMA_MODEL:
        # 检查模型名是否像 VLM 模型
        vlm_keywords = [
            "llava",
            "bakllava",
            "moondream",
            "minicpm-v",
            "qwen-vl",
            "qwen2-vl",
        ]
        is_vlm_model = any(kw in OLLAMA_MODEL.lower() for kw in vlm_keywords)

        if is_vlm_model:
            result = _call_vlm_openai_compatible(
                OLLAMA_API_BASE,
                "",
                OLLAMA_MODEL,
                image_base64,
                image_mime,
                prompt,
                temperature,
                max_tokens,
                label="Ollama-VLM",
            )
            if result:
                return result

    # 4. OpenAI 兼容 API
    if OPENAI_API_KEY:
        result = _call_vlm_openai_compatible(
            OPENAI_API_BASE,
            OPENAI_API_KEY,
            OPENAI_MODEL,
            image_base64,
            image_mime,
            prompt,
            temperature,
            max_tokens,
            label="OpenAI-VLM",
        )
        if result:
            return result

    return None


# ========================
# 图片 OCR（文字提取）
# ========================

OCR_PROMPT = """请仔细识别图片中的所有文字内容，按原文排版输出。
要求：
1. 完整提取所有可见文字，包括标题、正文、表格、标注等
2. 保留段落结构和换行
3. 如果有表格，用 Markdown 表格格式输出
4. 如果有公式，用文字描述或 LaTeX 格式输出
5. 只输出识别到的文字内容，不要添加其他解释"""


def ocr_image(image_path: str = None, image_base64: str = None) -> Optional[str]:
    """对图片进行 OCR 文字识别

    Args:
        image_path: 图片文件路径
        image_base64: 图片 base64 编码

    Returns:
        识别到的文字内容，失败返回 None
    """
    return call_vlm(
        image_path=image_path,
        image_base64=image_base64,
        prompt=OCR_PROMPT,
        temperature=0.1,
        max_tokens=4000,
    )


# ========================
# 图片知识抽取（直接从图片抽三元组）
# ========================

IMAGE_KG_PROMPT = """你是一个专业的知识图谱抽取专家。请仔细分析这张图片的内容，从中提取结构化的知识三元组。

要求：
1. 从图片中的文字、图表、示意图中提取实体和关系
2. 只提取明确、客观、可验证的知识
3. 三元组格式：(头实体, 关系, 尾实体)
4. 为每条三元组给出置信度评分（0-1，越接近1越确定）
5. 如果图片中包含图表或数据，请特别关注数据关系

请以 JSON 格式输出，结构如下：
{
    "triples": [
        {"head": "头实体", "relation": "关系", "tail": "尾实体", "confidence": 0.95},
        ...
    ],
    "summary": "图片内容简要描述"
}

只输出 JSON，不要其他文字。"""


def extract_kg_from_image(
    image_path: str = None, image_base64: str = None
) -> Tuple[List[dict], str]:
    """从图片中直接抽取知识图谱三元组

    Args:
        image_path: 图片文件路径
        image_base64: 图片 base64 编码

    Returns:
        (三元组列表, 图片摘要)，失败返回 ([], "")
    """
    result = call_vlm(
        image_path=image_path,
        image_base64=image_base64,
        prompt=IMAGE_KG_PROMPT,
        temperature=0.1,
        max_tokens=4000,
    )

    if not result:
        return [], ""

    # 尝试解析 JSON
    try:
        import json

        # 提取 JSON 对象
        json_match = re.search(r'\{[\s\S]*"triples"[\s\S]*\}', result)
        if json_match:
            data = json.loads(json_match.group())
            triples = data.get("triples", [])
            summary = data.get("summary", "")
            return triples, summary
    except Exception as e:
        logger.warning("VLM 抽取结果 JSON 解析失败: %s", e)

    return [], result or ""


# ========================
# 扫描版 PDF 处理
# ========================


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[str]:
    """将 PDF 每页转为图片（用于扫描版 PDF）

    Args:
        pdf_path: PDF 文件路径
        dpi: 输出图片 DPI（越高越清晰，越慢）

    Returns:
        图片文件路径列表（临时文件），失败返回空列表
    """
    try:
        # 优先用 pdf2image
        import tempfile

        from pdf2image import convert_from_path

        temp_dir = tempfile.mkdtemp(prefix="pocket_rag_pdf_")
        images = convert_from_path(pdf_path, dpi=dpi)

        image_paths = []
        for i, img in enumerate(images):
            img_path = os.path.join(temp_dir, f"page_{i + 1:04d}.png")
            img.save(img_path, "PNG")
            image_paths.append(img_path)

        return image_paths

    except ImportError:
        print("  [提示] 未安装 pdf2image，无法处理扫描版 PDF。")
        print("  安装方式: pip install pdf2image，并安装 poppler 工具")
        return []
    except Exception as e:
        print(f"  [PDF转图片] 失败: {e}")
        return []


def ocr_scanned_pdf(pdf_path: str, dpi: int = 200, max_pages: int = 50) -> str:
    """对扫描版 PDF 进行 OCR 识别

    Args:
        pdf_path: PDF 文件路径
        dpi: 图片 DPI
        max_pages: 最大处理页数（避免超大 PDF）

    Returns:
        识别后的全文文本
    """
    image_paths = pdf_to_images(pdf_path, dpi=dpi)
    if not image_paths:
        return ""

    all_text = []
    total = min(len(image_paths), max_pages)

    print(f"  [扫描版PDF] 共 {len(image_paths)} 页，处理前 {total} 页...")

    for i, img_path in enumerate(image_paths[:max_pages]):
        print(f"  [扫描版PDF] 处理第 {i + 1}/{total} 页...")
        text = ocr_image(image_path=img_path)
        if text:
            all_text.append(f"--- 第 {i + 1} 页 ---\n{text}")

    # 清理临时图片
    try:
        import shutil

        temp_dir = os.path.dirname(image_paths[0])
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logger.debug("清理临时图片失败（可忽略）: %s", e)

    return "\n\n".join(all_text)


# ========================
# 检查 VLM 是否可用
# ========================


def has_vlm() -> bool:
    """检查是否配置了至少一个可用的 VLM 后端"""
    # DashScope 一定支持（Qwen-VL 系列）
    if DASHSCOPE_API_KEY:
        return True

    # Ollama VLM 模型
    if OLLAMA_MODEL:
        vlm_keywords = [
            "llava",
            "bakllava",
            "moondream",
            "minicpm-v",
            "qwen-vl",
            "qwen2-vl",
        ]
        if any(kw in OLLAMA_MODEL.lower() for kw in vlm_keywords):
            return True

    # SiliconFlow / OpenAI 可能支持，假设配置了就可能用
    if SILICONFLOW_API_KEY or OPENAI_API_KEY:
        return True

    return False


def get_vlm_provider() -> str:
    """获取当前 VLM 提供商"""
    if DASHSCOPE_API_KEY:
        return f"DashScope ({DASHSCOPE_VLM_MODEL})"
    if OLLAMA_MODEL:
        return f"Ollama ({OLLAMA_MODEL})"
    if SILICONFLOW_API_KEY:
        return f"SiliconFlow ({SILICONFLOW_MODEL})"
    if OPENAI_API_KEY:
        return f"OpenAI ({OPENAI_MODEL})"
    return "未配置 VLM"
