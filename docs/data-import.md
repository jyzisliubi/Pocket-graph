# Multi-Source Data Import

> Moved out of the main README. PocketGraphRAG imports knowledge from various
> sources, automatically extracts triples, and builds the knowledge graph.

## Supported Formats

| Source | Format | Requirements | Quality |
|--------|--------|-------------|---------|
| **Plain Text** | `.txt` | - | ⭐⭐⭐⭐⭐ |
| **Markdown** | `.md` / `.markdown` | - | ⭐⭐⭐⭐⭐ |
| **PDF (text)** | `.pdf` | `pip install pdfplumber` | ⭐⭐⭐⭐ |
| **PDF (scanned)** | `.pdf` | `pip install pdfplumber pdf2image` + VLM | ⭐⭐⭐⭐ |
| **Word** | `.doc` / `.docx` | `pip install python-docx` | ⭐⭐⭐⭐ |
| **Images** | `.jpg` / `.png` / `.webp` | VLM model (DashScope Qwen-VL recommended) | ⭐⭐⭐⭐ |
| **Web Pages** | URLs | `pip install requests beautifulsoup4` | ⭐⭐⭐⭐ |
| **Dynamic Web** | URLs (JS-heavy) | `pip install playwright && playwright install chromium` | ⭐⭐⭐⭐ |

## Extraction Quality v2

The KG extractor has been upgraded to v2 with a 5-stage pipeline:

```
Input → ① Semantic Chunking → ② LLM Extraction → ③ Entity Alignment → ④ Deduplication → ⑤ Quality Filter → High-Quality KG
```

| Stage | Description |
|-------|-------------|
| **Semantic Chunking** | Split by paragraph/sentence boundaries, avoid cutting mid-sentence |
| **LLM Extraction** | Few-shot prompting + confidence scoring + evidence tracking |
| **Entity Alignment** | Rule-based (bracket removal, full-width normalization) + embedding similarity matching |
| **Deduplication** | Merge duplicate triples, keep the one with highest confidence |
| **Quality Filter** | Filter low-confidence triples (default threshold: 0.6) |

### Benchmark Results

Tested on movie knowledge domain text (~150 chars) with DashScope Qwen-Plus +
Qwen-VL-Plus.

> Qualitative observations on a small in-domain sample, not a third-party
> benchmark. Confidence scores are self-reported by the extractor LLM; treat as
> indicative, not ground truth.

| Data Source | Triples Extracted | High Quality (≥0.8) | Avg Confidence |
|-------------|-------------------|---------------------|----------------|
| **TXT Plain Text** | 14 | 14/14 | 0.948 |
| **Word (.docx)** | 17 | 17/17 | 0.955 |
| **PDF (text)** | 12 | 12/12 | 0.944 |
| **Image (OCR)** | 4 | 4/4 | 0.943 |
| **Web Page** | 14-46 | all retained | 0.95-0.97 |

## Extraction Examples by Source

### Example 1: TXT / Word / PDF Input → Output

```
Input text (~150 chars):
  《盗梦空间》是诺兰执导的科幻悬疑电影，由莱昂纳多主演。
  诺兰是英国裔美国导演，以复杂叙事结构著称。
  盗梦空间的剧情围绕梦境与现实的层层嵌套展开...

Output (14 triples, avg confidence 0.948, all ≥0.8):
  [0.98] 诺兰 --[职业]--> 导演
  [0.97] 盗梦空间 --[导演]--> 诺兰
  [0.96] 盗梦空间 --[类型]--> 科幻悬疑片
  [0.96] 莱昂纳多 --[主演]--> 盗梦空间
  [0.95] 诺兰 --[国籍]--> 英国裔美国
  [0.95] 盗梦空间 --[主题]--> 梦境与现实嵌套
  [0.95] 诺兰 --[执导风格]--> 复杂叙事结构
  [0.95] 盗梦空间 --[上映年份]--> 2010年
  ... (6 more triples)
```

### Example 2: Image (OCR) → Output

```
Input: PNG image with Chinese text (movie knowledge summary)

OCR Result:
  电影知识图谱要点
  盗梦空间由诺兰执导
  莱昂纳多主演盗梦空间
  影片于2010年上映
  星际穿越也是诺兰作品
  诺兰擅长科幻题材

Output (4 triples, avg confidence 0.943, all ≥0.8):
  [0.97] 诺兰 --[执导]--> 盗梦空间
  [0.95] 莱昂纳多 --[主演]--> 盗梦空间
  [0.95] 盗梦空间 --[上映年份]--> 2010年
  [0.90] 诺兰 --[执导]--> 星际穿越
```

### Example 3: Web Page → Output

```
Input: http://www.moa.gov.cn/ (Ministry of Agriculture homepage)
Content length: ~5,570 chars

Output (34 triples, avg confidence 0.951, all retained):
  [0.99] 农业农村部办公厅 --[发布]--> 中华人民共和国农业农村部公告 第1013号
  [0.98] 农业农村数据分类分级指南 --[属于]--> 农业行业标准
  [0.98] 全国"三夏"小麦大规模机收 --[状态]--> 基本结束
  [0.98] 农业农村部办公厅 --[发布]--> 关于加强农业科普工作的意见
  [0.97] 农业农村部办公厅 --[组织]--> 2026年度农业"火花技术"征集工作
  ... (29 more triples)
```

## Python API

```python
from PocketGraphRAG.data_importer import DataImporter
from PocketGraphRAG.kg_extractor import extract_knowledge_graph

# Initialize importer
importer = DataImporter()

# Import from various sources
doc = importer.import_file("document.pdf")
# doc = importer.import_file("document.docx")
# doc = importer.import_file("image.png", image_mode="ocr")
# doc = importer.import_url("https://example.com/article")

# Extract knowledge graph
result = extract_knowledge_graph(doc.content, min_confidence=0.6)
print(f"Extracted {len(result.triples)} triples")
print(f"Avg confidence: {result.avg_confidence:.3f}")
print(f"High quality (>=0.8): {result.high_quality_count}")
```

## CLI

```bash
# Extract triples from a file
python -m PocketGraphRAG.kg_extractor --input your_document.txt --output my_triples.txt

# Or via the modern CLI
pocketgraphrag extract -i document.txt -o triples.txt --min-confidence 0.6
```
