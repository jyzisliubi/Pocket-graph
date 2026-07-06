# 贡献指南

感谢你对 PocketGraphRAG 的兴趣！欢迎提交 Issue、PR 和建议。

## 🚀 快速开始

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/Pocket-graph.git
cd Pocket-graph
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装开发依赖

```bash
pip install -e ".[web,docs,dev]"
pre-commit install
```

### 4. 运行测试

```bash
# 全量测试
python -m pytest PocketGraphRAG/tests/ -v

# 带覆盖率
python -m pytest PocketGraphRAG/tests/ --cov=PocketGraphRAG --cov-report=html

# 仅运行特定模块
python -m pytest PocketGraphRAG/tests/test_v037_features.py -v
```

## 📝 代码规范

### Python 代码

- **格式化**：使用 [ruff](https://docs.astral.sh/ruff/) 格式化
  ```bash
  ruff format PocketGraphRAG/
  ruff check PocketGraphRAG/ --select E,F
  ```

- **类型检查**：使用 [mypy](https://mypy-lang.org/)
  ```bash
  mypy PocketGraphRAG/rag_system.py
  ```

- **命名约定**：
  - 类：`PascalCase`（如 `PocketGraphRAG`）
  - 函数/变量：`snake_case`（如 `extract_knowledge_graph`）
  - 常量：`UPPER_SNAKE_CASE`（如 `SEARCH_MODE`）
  - 私有：前缀 `_`（如 `_should_refuse`）

### 前端代码

- **TypeScript**：必须类型安全，禁止 `any`
- **格式化**：Prettier + ESLint
- **组件**：函数式组件 + Hooks

### 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>(<scope>): <description>

<body>

<footer>
```

**类型**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档
- `refactor`: 重构
- `test`: 测试
- `chore`: 杂项

**示例**：
```
feat(v0.3.7): add WORKSPACE isolation

- config.py: POCKET_WORKSPACE env var
- _apply_workspace() with path traversal protection
- default workspace keeps backward compatibility
```

## 🧪 测试要求

### 新功能必须包含测试

- 每个新功能至少 5 个单元测试
- 覆盖正常路径 + 边界情况 + 错误处理
- 使用 `unittest.mock.patch` 做 mock

### 测试命名

```python
class TestFeatureName:
    def test_basic_functionality(self):
        """基本功能测试"""
        ...

    def test_edge_case_empty_input(self):
        """边界情况：空输入"""
        ...

    def test_error_handling_invalid_input(self):
        """错误处理：无效输入"""
        ...
```

### 运行测试

```bash
# 提交前必须全量通过
python -m pytest PocketGraphRAG/tests/ -v --tb=short

# 当前基线：830+ 测试
```

## 🔄 PR 流程

### 1. 创建分支

```bash
git checkout -b feat/your-feature-name
# 或
git checkout -b fix/issue-123
```

### 2. 提交更改

```bash
git add <files>
git commit -m "feat(scope): description"
```

### 3. 推送并创建 PR

```bash
git push origin feat/your-feature-name
```

在 GitHub 上创建 PR，描述：
- **What**: 做了什么
- **Why**: 为什么做
- **How**: 怎么做的
- **Testing**: 如何测试的

### 4. Code Review

- 至少需要 1 个 reviewer 批准
- CI 必须通过（4 版本 Python 矩阵测试）
- 测试覆盖率不下降

## 🐛 报告 Bug

使用 [Bug Report 模板](.github/ISSUE_TEMPLATE/bug_report.md) 提交 Issue，包含：
- 环境信息（OS / Python / PocketGraphRAG 版本）
- 复现步骤
- 期望行为
- 实际行为
- 错误日志

## 💡 建议新功能

使用 [Feature Request 模板](.github/ISSUE_TEMPLATE/feature_request.md) 提交 Issue，说明：
- 使用场景
- 期望的行为
- 替代方案
- 是否愿意贡献代码

## 📖 文档贡献

- 修复文档错误：直接 PR
- 新增文档：放在 `docs/` 目录下，mkdocs Material 主题
- 翻译：欢迎添加新语言

## 🏷️ 版本发布

版本号遵循 [Semantic Versioning](https://semver.org/)：
- MAJOR：不兼容的 API 变更
- MINOR：向后兼容的新功能
- PATCH：向后兼容的 Bug 修复

发布流程：
1. 更新 `pyproject.toml` 版本号
2. 更新 `CHANGELOG.md`
3. 创建 GitHub Release
4. CI 自动发布到 PyPI + Docker Hub

## ❓ 问题？

- [GitHub Issues](https://github.com/jyzisliubi/Pocket-graph/issues)
- Email: 3364415961@qq.com

感谢你的贡献！🎉
