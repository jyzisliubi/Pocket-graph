"""Capture PocketGraphRAG Web UI screenshots and assemble a demo GIF.

This fills the "no demo GIF" gap called out in the README. Run it against a
live Web UI to produce real screenshots of the Q&A / Graph / Data-Management
tabs, then stitch them into a single `assets/demo.gif`.

Prereqs:
    1. Start the Web UI in another terminal:
            python -m PocketGraphRAG.webapp
       and wait for "Running on local URL: http://127.0.0.1:7860".
    2. Install Playwright (already an optional extra):
            pip install "PocketGraphRAG[playwright]"
            playwright install chromium

Usage:
    python -m PocketGraphRAG.tools.capture_webui_demo
    python -m PocketGraphRAG.tools.capture_webui_demo --url http://127.0.0.1:7860 --query "这部电影讲了什么？"

Output:
    assets/demo_qa.png
    assets/demo_graph.png
    assets/demo_data.png
    assets/demo.gif          # vertical stack of the three tabs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


def capture(url: str, query: str, out_dir: Path) -> list[Path]:
    """Capture three Web UI tabs. Returns list of saved screenshot paths."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(
            "Playwright 未安装。请运行:\n"
            '    pip install "PocketGraphRAG[playwright]"\n'
            "    playwright install chromium"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    shots: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(url, wait_until="load", timeout=15000)

        # --- Tab 1: Q&A ---
        # Gradio tabs are <button> elements with role="tab".
        qa_tab = page.get_by_role("tab", name="问答")
        if qa_tab.count():
            qa_tab.first.click()
            time.sleep(0.6)
        # Type a sample query and submit if an input box is present.
        try:
            textarea = page.locator("textarea").first
            if textarea.count():
                textarea.fill(query)
                time.sleep(0.3)
                page.keyboard.press("Enter")
                time.sleep(4.0)  # let the streaming answer settle
        except Exception:
            pass
        qa_path = out_dir / "demo_qa.png"
        page.screenshot(path=str(qa_path), full_page=False)
        shots.append(qa_path)
        print(f"[ok] {qa_path}")

        # --- Tab 2: Knowledge Graph ---
        kg_tab = page.get_by_role("tab", name="知识图谱")
        if kg_tab.count():
            kg_tab.first.click()
            time.sleep(1.2)  # let ECharts render
        kg_path = out_dir / "demo_graph.png"
        page.screenshot(path=str(kg_path), full_page=False)
        shots.append(kg_path)
        print(f"[ok] {kg_path}")

        # --- Tab 3: Data Management ---
        dm_tab = page.get_by_role("tab", name="数据管理")
        if dm_tab.count():
            dm_tab.first.click()
            time.sleep(0.8)
        dm_path = out_dir / "demo_data.png"
        page.screenshot(path=str(dm_path), full_page=False)
        shots.append(dm_path)
        print(f"[ok] {dm_path}")

        browser.close()
    return shots


def assemble_gif(shots: list[Path], out_path: Path) -> None:
    """Stack screenshots vertically into a single GIF."""
    try:
        from PIL import Image
    except ImportError:
        print("[skip] Pillow 未安装，跳过 GIF 合成（截图已保存）")
        return
    if not shots:
        return
    images = [Image.open(s).convert("RGB") for s in shots]
    w = max(im.width for im in images)
    h = sum(im.height for im in images)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(out_path, format="GIF", duration=2500, loop=0, optimize=True)
    print(f"[ok] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture Web UI demo screenshots / GIF."
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:7860", help="Web UI base URL"
    )
    parser.add_argument("--query", default="这部电影讲了什么？", help="sample Q&A query")
    parser.add_argument("--out", default=str(ASSETS_DIR), help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    print(f"[*] Capturing {args.url} → {out_dir}")
    shots = capture(args.url, args.query, out_dir)
    if shots:
        assemble_gif(shots, out_dir / "demo.gif")
    print("[done] 将 demo.gif 嵌入 README 顶部 hero 区即可大幅提升首屏观感。")


if __name__ == "__main__":
    main()
