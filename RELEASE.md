# Release Checklist

本文件列出发布新版本到 PyPI 前的检查项，避免常见翻车。
参考 microsoft/graphrag 的 `RELEASE.md` 做法。

---

## 发版前（每次）

### 代码与版本
- [ ] 所有改动已合并到 `main` 分支
- [ ] 更新 `PocketGraphRAG/__init__.py` 中的 `__version__`
- [ ] 更新 `pyproject.toml` 中的 `version`
- [ ] 更新 `CHANGELOG.md`：把 `[Unreleased]` 改为 `[x.y.z] - YYYY-MM-DD`，新增空的 `[Unreleased]` 段
- [ ] 确认 `pyproject.toml` 的 `project.urls` 指向真实仓库（当前为 `JayZ/PocketGraphRAG`）

### 质量门禁
- [ ] 本地 `pytest` 全绿（`pytest PocketGraphRAG/tests/`）
- [ ] 本地 `ruff check .` 无报错
- [ ] 本地 `ruff format --check .` 无报错
- [ ] CI（GitHub Actions）在 `main` 上是绿的状态
- [ ] 覆盖率未明显下降（codecov）

### 元数据
- [ ] `README.md` 顶部徽章链接有效（CI / codecov / License）
- [ ] `README.md` 的安装命令与发布状态一致（发布后把 "coming soon" 改回正常 pip install）
- [ ] `LICENSE` 文件存在且年份正确
- [ ] `pyproject.toml` 的 `classifiers` 与 Python 版本支持一致

### 打包验证
- [ ] `python -m build` 成功生成 `dist/`
- [ ] `twine check dist/*` 无报错
- [ ] 本地干净环境验证安装：
  ```bash
  python -m venv /tmp/release-test
  /tmp/release-test/Scripts/activate    # Windows
  pip install dist/pocketgraphrag-*.whl
  python -c "from PocketGraphRAG import PocketGraphRAG; print('import ok')"
  ```

---

## 发布到 PyPI（首次需配置 Trusted Publishing）

本项目使用 **Trusted Publishing (OIDC)**，无需长期 API token。

### 一次性配置（首次发布）
1. 在 PyPI 创建项目：https://pypi.org/manage/projects/
2. 配置 Trusted Publisher：
   - PyPI Project name: `pocketgraphrag`
   - Owner: `JayZ`
   - Repo: `PocketGraphRAG`
   - Workflow: `publish.yml`
   - Environment name: `pypi`（须与 `.github/workflows/publish.yml` 的 `environment:` 一致）
3. 详见：https://docs.pypi.org/trusted-publishers/

### 每次发版
1. 在 GitHub 创建 Release，打 `v*` 格式的 tag（如 `v0.3.0`）
2. `publish.yml` workflow 会自动触发：
   - build → twine check → publish to PyPI
3. 发布后验证：
   - https://pypi.org/project/pocketgraphrag/
   - `pip install pocketgraphrag` 在干净环境能装上

### 发布后
- [ ] 把 README 的 "coming soon" pip install 段改回正式推荐
- [ ] 把 PyPI 徽章从 "coming soon" 改回正式版本徽章：
      `[![PyPI](https://img.shields.io/pypi/v/pocketgraphrag.svg)](https://pypi.org/project/pocketgraphrag/)`
- [ ] 在 GitHub Release 页面填写 Release Notes（可从 CHANGELOG 复制）
- [ ] 如有 breaking change，更新 `docs/` 里的迁移说明

---

## 紧急回滚

若发布后发现严重 bug：
1. **不要** 在 PyPI 上删除/覆盖已发版本（PyPI 不允许覆盖）
2. 发 `vX.Y.Z+1` hotfix 版本
3. 在 README/CHANGELOG 标注受影响版本

---

## 版本号约定（Semantic Versioning）

- `0.x.y` — Alpha 阶段，API 可能变动
- `1.0.0` — 首个稳定版，API 冻结
- `MAJOR` — 不兼容改动
- `MINOR` — 向下兼容的新功能
- `PATCH` — 向下兼容的 bug 修复
