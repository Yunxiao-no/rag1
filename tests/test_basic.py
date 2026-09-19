"""
ResearchAgent 基础测试
运行方式：pytest tests/ -v
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pdf_parser import PDFParser
from core.citation_manager import CitationManager, Citation


class TestCitationManager:
    """引用管理器测试"""

    def test_add_citation(self):
        mgr = CitationManager()
        cite = Citation(key="test2024", title="Test Paper", authors=["Test Author"], year="2024")
        key = mgr.add(cite)
        assert key == "test2024"
        assert mgr.count() == 1

    def test_duplicate_key(self):
        mgr = CitationManager()
        cite1 = Citation(key="test2024", title="Paper 1")
        cite2 = Citation(key="test2024", title="Paper 2")
        mgr.add(cite1)
        key2 = mgr.add(cite2)
        assert key2 == "test2024_1"
        assert mgr.count() == 2

    def test_generate_key(self):
        mgr = CitationManager()
        key = mgr.add_from_paper_info(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            year="2017",
            venue="NeurIPS",
        )
        assert key  # 确保生成了非空键
        assert "2017" in key or "17" in key

    def test_bibtex_export(self):
        mgr = CitationManager()
        mgr.add_from_paper_info(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            year="2024",
            venue="Test Conference",
        )
        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False) as f:
            path = f.name
        try:
            content = mgr.export_bibtex(path)
            assert "@article" in content
            assert "Test Paper" in content
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_reference_list(self):
        mgr = CitationManager()
        mgr.add_from_paper_info(title="Paper A", authors=["Author A"], year="2023")
        mgr.add_from_paper_info(title="Paper B", authors=["Author B"], year="2024")
        ref_list = mgr.generate_reference_list("apa")
        assert "参考文献" in ref_list
        assert "Paper A" in ref_list
        assert "Paper B" in ref_list

    def test_extract_citations(self):
        mgr = CitationManager()
        text = "这是一个引用 [1]，这是另一个 [2]，还有 [1] 重复"
        citations = mgr.extract_citations_from_text(text)
        assert citations == ["1", "2"]


class TestPDFParser:
    """PDF 解析器测试（不依赖真实 PDF）"""

    def test_section_detection(self):
        parser = PDFParser()

        # 测试数字编号章节
        is_sec, title, level = parser._is_section_title("1 Introduction")
        assert is_sec
        assert "Introduction" in title
        assert level == 1

        # 测试二级章节
        is_sec, title, level = parser._is_section_title("2.1 Related Work")
        assert is_sec
        assert level == 2

        # 测试常见章节名
        is_sec, title, level = parser._is_section_title("Conclusion")
        assert is_sec

        # 测试非章节
        is_sec, _, _ = parser._is_section_title("This is a normal sentence.")
        assert not is_sec

        is_sec, _, _ = parser._is_section_title("References")
        assert not is_sec  # References 单独处理

    def test_extract_abstract(self):
        parser = PDFParser()
        text = """Some header text

Abstract
This paper proposes a novel method for natural language processing.
Our approach achieves state-of-the-art results on multiple benchmarks.

1 Introduction
The rest of the paper...
"""
        abstract = parser._extract_abstract(text)
        assert "novel method" in abstract
        assert "Introduction" not in abstract

    def test_split_sections(self):
        parser = PDFParser()
        text = """Front matter content

1 Introduction
This is the introduction section.
It has multiple lines of content.

2 Method
This is the method section.
Describes the proposed approach.

3 Experiments
Experimental results go here.
"""
        sections = parser._split_sections(text)
        assert len(sections) >= 3
        titles = [s.title for s in sections]
        assert any("Introduction" in t for t in titles)
        assert any("Method" in t for t in titles)


if __name__ == "__main__":
    # 直接运行测试
    import pytest
    pytest.main([__file__, "-v"])
