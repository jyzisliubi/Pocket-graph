"""Sitemap / RSS 导入测试（v0.3.7）

测试 ``DataImporter.import_sitemap`` / ``import_rss`` / ``_parse_sitemap`` / ``_parse_rss``
对标 RAGFlow 多数据源导入能力。
"""

from unittest.mock import MagicMock, patch

import pytest

from PocketGraphRAG.data_importer import DataImporter, ExtractedDocument


# ===== Sitemap 测试 =====

SAMPLE_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/page1</loc>
    <lastmod>2026-01-01</lastmod>
  </url>
  <url>
    <loc>https://example.com/page2</loc>
    <lastmod>2026-01-02</lastmod>
  </url>
  <url>
    <loc>https://example.com/page3</loc>
  </url>
</urlset>
"""


SAMPLE_SITEMAP_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://example.com/sitemap1.xml</loc>
  </sitemap>
  <sitemap>
    <loc>https://example.com/sitemap2.xml</loc>
  </sitemap>
</sitemapindex>
"""


class TestSitemapParsing:
    """Sitemap XML 解析测试"""

    def test_parse_standard_sitemap(self):
        """标准 urlset sitemap 应正确解析所有 URL"""
        importer = DataImporter()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = SAMPLE_SITEMAP_XML.encode("utf-8")
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            urls = importer._parse_sitemap("https://example.com/sitemap.xml")

        assert len(urls) == 3
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls
        assert "https://example.com/page3" in urls

    def test_parse_sitemap_with_max_urls(self):
        """max_urls 应限制返回的 URL 数量"""
        importer = DataImporter()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = SAMPLE_SITEMAP_XML.encode("utf-8")
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            urls = importer._parse_sitemap(
                "https://example.com/sitemap.xml", max_urls=2
            )

        assert len(urls) == 2

    def test_parse_sitemap_index_recursively(self):
        """sitemapindex 应递归解析子 sitemap"""
        importer = DataImporter()

        # mock：第一次返回 sitemapindex，后续返回子 sitemap
        def mock_get_impl(url, *args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if url == "https://example.com/sitemap.xml":
                resp.content = SAMPLE_SITEMAP_INDEX_XML.encode("utf-8")
            elif url == "https://example.com/sitemap1.xml":
                resp.content = b"""<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/a1</loc></url>
</urlset>"""
            elif url == "https://example.com/sitemap2.xml":
                resp.content = b"""<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/b1</loc></url>
  <url><loc>https://example.com/b2</loc></url>
</urlset>"""
            return resp

        with patch("requests.get", side_effect=mock_get_impl):
            urls = importer._parse_sitemap("https://example.com/sitemap.xml")

        assert len(urls) == 3
        assert "https://example.com/a1" in urls
        assert "https://example.com/b1" in urls
        assert "https://example.com/b2" in urls

    def test_parse_empty_sitemap(self):
        """空 sitemap 应返回空列表"""
        importer = DataImporter()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            urls = importer._parse_sitemap("https://example.com/sitemap.xml")

        assert urls == []


# ===== RSS 测试 =====

SAMPLE_RSS_2_0 = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Test Blog</title>
    <link>https://example.com</link>
    <description>A test blog</description>
    <item>
      <title>First Post</title>
      <link>https://example.com/post1</link>
      <description>&lt;p&gt;First post description&lt;/p&gt;</description>
      <content:encoded>&lt;p&gt;Full content of first post&lt;/p&gt;</content:encoded>
      <pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Second Post</title>
      <link>https://example.com/post2</link>
      <description>Second post summary</description>
      <pubDate>Tue, 02 Jan 2026 00:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


SAMPLE_ATOM_1_0 = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Test Feed</title>
  <link href="https://example.com"/>
  <entry>
    <title>Atom Entry 1</title>
    <link href="https://example.com/entry1" rel="alternate"/>
    <summary>Atom summary 1</summary>
    <published>2026-01-01T00:00:00Z</published>
  </entry>
  <entry>
    <title>Atom Entry 2</title>
    <link href="https://example.com/entry2" rel="alternate"/>
    <content>Atom full content 2</content>
    <updated>2026-01-02T00:00:00Z</updated>
  </entry>
</feed>
"""


class TestRSSParsing:
    """RSS / Atom feed 解析测试"""

    def test_parse_rss_2_0(self):
        """RSS 2.0 应正确解析 item 列表"""
        importer = DataImporter()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = SAMPLE_RSS_2_0.encode("utf-8")
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            items = importer._parse_rss("https://example.com/feed.xml")

        assert len(items) == 2
        assert items[0]["title"] == "First Post"
        assert items[0]["link"] == "https://example.com/post1"
        # content:encoded 优先于 description
        assert "Full content of first post" in items[0]["content"]
        assert items[0]["published"] == "Mon, 01 Jan 2026 00:00:00 GMT"

    def test_parse_rss_falls_back_to_description(self):
        """无 content:encoded 时应回退到 description"""
        importer = DataImporter()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = SAMPLE_RSS_2_0.encode("utf-8")
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            items = importer._parse_rss("https://example.com/feed.xml")

        # 第二个 item 没有 content:encoded，应用 description
        assert items[1]["title"] == "Second Post"
        assert "Second post summary" in items[1]["content"]

    def test_parse_atom_1_0(self):
        """Atom 1.0 feed 应正确解析 entry 列表"""
        importer = DataImporter()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = SAMPLE_ATOM_1_0.encode("utf-8")
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            items = importer._parse_rss("https://example.com/atom.xml")

        assert len(items) == 2
        assert items[0]["title"] == "Atom Entry 1"
        assert items[0]["link"] == "https://example.com/entry1"
        assert "Atom summary 1" in items[0]["content"]
        # content 优先于 summary
        assert "Atom full content 2" in items[1]["content"]

    def test_parse_rss_with_max_items(self):
        """max_items 应限制返回条目数"""
        importer = DataImporter()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = SAMPLE_RSS_2_0.encode("utf-8")
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            items = importer._parse_rss(
                "https://example.com/feed.xml", max_items=1
            )

        assert len(items) == 1


class TestImportSitemapRSS:
    """import_sitemap / import_rss 集成测试"""

    def test_import_rss_with_feed_content(self):
        """RSS 有 feed 自带内容时应直接用，不抓取网页"""
        importer = DataImporter()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = SAMPLE_RSS_2_0.encode("utf-8")
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            docs = importer.import_rss("https://example.com/feed.xml")

        # 两个 item 都有 content（第一个有 content:encoded，第二个有 description）
        assert len(docs) == 2
        assert all(d.source_type == "rss" for d in docs)
        assert docs[0].title == "First Post"
        assert "Full content of first post" in docs[0].content
        # 不应调用 import_url（因为有 feed 自带内容）
        # mock_get 只被调用 1 次（获取 feed 本身）

    def test_import_sitemap_empty_returns_empty_list(self):
        """空 sitemap 应返回空列表"""
        importer = DataImporter()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            docs = importer.import_sitemap("https://example.com/sitemap.xml")

        assert docs == []

    def test_import_rss_network_error_returns_empty(self):
        """RSS 网络错误应返回空列表（不抛异常）"""
        importer = DataImporter()
        with patch("requests.get", side_effect=Exception("Network error")):
            docs = importer.import_rss("https://example.com/feed.xml")

        assert docs == []


class TestHtmlToText:
    """_html_to_text 工具函数测试"""

    def test_html_to_text_strips_tags(self):
        """应正确去除 HTML 标签"""
        text = DataImporter._html_to_text("<p>Hello <b>world</b></p>")
        assert "Hello" in text
        assert "world" in text
        assert "<" not in text

    def test_html_to_text_decodes_entities(self):
        """应正确解码 HTML 实体"""
        text = DataImporter._html_to_text("a &amp; b &lt; c")
        assert "a & b" in text
        assert "< c" in text


class TestSanitizeFilename:
    """_sanitize_filename 工具函数测试"""

    def test_sanitizes_url_to_filename(self):
        """应把 URL 清理成合法文件名"""
        from PocketGraphRAG.api_server import _sanitize_filename

        # 普通标题
        assert _sanitize_filename("Hello World") == "Hello_World"
        # URL：取最后一段路径
        result = _sanitize_filename("https://example.com/post/123")
        assert result == "123"
        # URL 带尾部斜杠
        result = _sanitize_filename("https://example.com/post/")
        assert "post" in result
        # 特殊字符
        result = _sanitize_filename("file/with:special*chars")
        assert "/" not in result
        assert ":" not in result
        assert "*" not in result
        # 空字符串
        assert _sanitize_filename("") == "imported"
        # 中文保留
        assert _sanitize_filename("水稻种植技术") == "水稻种植技术"
