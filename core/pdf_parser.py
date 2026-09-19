"""
PDF 解析模块
专门针对学术论文 PDF 的解析，提取文本、标题、章节、公式位置等
支持 pypdf（快速）和 pdfplumber（精确，能处理表格）
"""
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import pdfplumber
from pypdf import PdfReader


@dataclass
class PaperSection:
    """论文章节"""
    title: str                    # 章节标题
    content: str                  # 章节内容
    page: int = 0                 # 起始页码
    level: int = 1                # 标题级别（1=一级标题，2=二级标题）


@dataclass
class ParsedPaper:
    """解析后的论文结构"""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    sections: List[PaperSection] = field(default_factory=list)
    references: str = ""
    total_pages: int = 0
    full_text: str = ""
    metadata: Dict = field(default_factory=dict)

    def get_section(self, title_keyword: str) -> Optional[PaperSection]:
        """根据关键词查找章节"""
        for sec in self.sections:
            if title_keyword.lower() in sec.title.lower():
                return sec
        return None


class PDFParser:
    """
    学术论文 PDF 解析器

    设计思路：
    1. 用 pypdf 快速提取全文（速度快）
    2. 用 pdfplumber 提取表格和更精确的布局信息
    3. 基于正则和启发式规则识别章节标题
    4. 提取摘要、参考文献等特殊部分
    """

    # 常见的章节标题模式（学术论文通用）
    SECTION_PATTERNS = [
        # 数字编号的章节，如 "1 Introduction", "2. Related Work"
        r"^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s\-:,]+)$",
        # 全大写的章节标题，如 "INTRODUCTION", "RELATED WORK"
        r"^([A-Z][A-Z\s\-]{2,})$",
        # 常见章节名（不带编号）
        r"^(Abstract|Introduction|Related Work|Background|Methodology|Methods|"
        r"Approach|Experiments|Results|Discussion|Conclusion|Conclusions|"
        r"References|Acknowledgements|Acknowledgments|Appendix)$",
    ]

    def __init__(self, use_pdfplumber: bool = True):
        """
        Args:
            use_pdfplumber: 是否使用 pdfplumber（更精确但更慢）
        """
        self.use_pdfplumber = use_pdfplumber

    def parse(self, pdf_path: str) -> ParsedPaper:
        """
        解析 PDF 文件

        Args:
            pdf_path: PDF 文件路径

        Returns:
            解析后的论文结构
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        result = ParsedPaper()
        result.total_pages = self._get_page_count(pdf_path)
        result.metadata = self._get_metadata(pdf_path)

        # 提取全文
        full_text = self._extract_full_text(pdf_path)
        result.full_text = full_text

        # 提取标题和作者（从第一页）
        first_page_text = self._extract_first_page(pdf_path)
        result.title, result.authors = self._extract_title_and_authors(first_page_text)

        # 提取摘要
        result.abstract = self._extract_abstract(full_text)

        # 提取参考文献
        result.references = self._extract_references(full_text)

        # 切分章节
        result.sections = self._split_sections(full_text)

        return result

    def _get_page_count(self, pdf_path: str) -> int:
        """获取页数"""
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception:
            return 0

    def _get_metadata(self, pdf_path: str) -> Dict:
        """获取 PDF 元数据"""
        try:
            reader = PdfReader(pdf_path)
            meta = reader.metadata or {}
            return {
                "title": getattr(meta, "title", ""),
                "author": getattr(meta, "author", ""),
                "subject": getattr(meta, "subject", ""),
                "creator": getattr(meta, "creator", ""),
            }
        except Exception:
            return {}

    def _extract_full_text(self, pdf_path: str) -> str:
        """提取全文文本"""
        text_parts = []

        if self.use_pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        text_parts.append(page_text)
                return "\n\n".join(text_parts)
            except Exception:
                pass  # pdfplumber 失败则降级到 pypdf

        # 降级方案：pypdf
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"[PDF 提取失败] {e}")
            return ""

    def _extract_first_page(self, pdf_path: str) -> str:
        """提取第一页文本（用于识别标题和作者）"""
        if self.use_pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    if pdf.pages:
                        return pdf.pages[0].extract_text() or ""
            except Exception:
                pass

        try:
            reader = PdfReader(pdf_path)
            if reader.pages:
                return reader.pages[0].extract_text() or ""
        except Exception:
            pass
        return ""

    def _extract_title_and_authors(self, first_page_text: str) -> Tuple[str, List[str]]:
        """
        从第一页提取标题和作者
        启发式：标题通常在最前面，字号最大；作者在标题下方
        """
        lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
        if not lines:
            return "", []

        # 标题：通常是前几行中最长的、不含邮箱的行
        title = ""
        title_idx = 0
        for i, line in enumerate(lines[:10]):
            # 跳过明显不是标题的行
            if any(kw in line.lower() for kw in ["@", "arxiv", "abstract", "http", "www."]):
                continue
            if len(line) > 15 and not line.endswith("."):
                title = line
                title_idx = i
                break

        # 作者：标题后面的几行，通常包含逗号或 "and"
        authors = []
        for line in lines[title_idx + 1: title_idx + 5]:
            if "@" in line or "abstract" in line.lower():
                break
            # 提取作者名（简单处理：按逗号或 and 分割）
            if any(c.isalpha() for c in line) and len(line) < 200:
                # 清理上标数字、星号等
                clean = re.sub(r"[\d\*†‡§¶#]+", "", line).strip()
                if clean and len(clean) > 2:
                    # 分割多个作者
                    parts = re.split(r",| and |&", clean)
                    for p in parts:
                        p = p.strip()
                        if p and len(p) > 2 and not any(kw in p.lower() for kw in ["university", "institute", "department", "school", "lab"]):
                            authors.append(p)

        return title, authors[:10]  # 最多取10个作者

    def _extract_abstract(self, full_text: str) -> str:
        """提取摘要"""
        # 匹配 "Abstract" 后面到下一个章节标题的内容
        pattern = r"(?:^|\n)Abstract\s*[\n:](.*?)(?=\n\s*(?:\d+\.?\s+[A-Z]|Introduction|Related Work|Keywords|1\s))"
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # 清理
            abstract = re.sub(r"\s+", " ", abstract)
            return abstract[:2000]  # 限制长度

        # 降级：找 "Abstract" 关键字后面的 500 字符
        idx = full_text.lower().find("abstract")
        if idx >= 0:
            snippet = full_text[idx:idx + 1000]
            return snippet[:500]

        return ""

    def _extract_references(self, full_text: str) -> str:
        """提取参考文献部分"""
        # 匹配 References 到文末
        pattern = r"(?:^|\n)(References|Bibliography)\s*[\n:]?(.*)$"
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(2).strip()[:5000]
        return ""

    def _split_sections(self, full_text: str) -> List[PaperSection]:
        """
        将全文切分为章节
        基于正则匹配章节标题，然后切分
        """
        lines = full_text.split("\n")
        sections = []
        current_title = "Front Matter"
        current_content = []
        current_page = 0

        for line in lines:
            stripped = line.strip()

            # 检测是否是章节标题
            is_section, title, level = self._is_section_title(stripped)

            if is_section:
                # 保存上一个章节
                if current_content:
                    sections.append(PaperSection(
                        title=current_title,
                        content="\n".join(current_content).strip(),
                        page=current_page,
                        level=level,
                    ))
                current_title = title
                current_content = []
            else:
                current_content.append(stripped)

        # 保存最后一个章节
        if current_content:
            sections.append(PaperSection(
                title=current_title,
                content="\n".join(current_content).strip(),
                page=current_page,
            ))

        # 过滤掉内容太少的章节（可能是误识别）
        sections = [s for s in sections if len(s.content) > 50]

        return sections

    def _is_section_title(self, line: str) -> Tuple[bool, str, int]:
        """
        判断一行是否是章节标题

        Returns:
            (是否是标题, 标题文本, 级别)
        """
        if not line or len(line) > 100:
            return False, "", 0

        # 模式1：数字编号 "1 Introduction" / "2.1 Related Work"
        m = re.match(r"^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s\-:,]+)$", line)
        if m:
            number = m.group(1)
            title = m.group(2).strip()
            level = number.count(".") + 1
            # 排除参考文献等已单独处理的
            if title.lower() in ["references", "bibliography"]:
                return False, "", 0
            return True, f"{number} {title}", level

        # 模式2：常见章节名（不带编号）
        common_sections = [
            "Abstract", "Introduction", "Related Work", "Background",
            "Preliminaries", "Methodology", "Methods", "Approach",
            "Experiments", "Results", "Evaluation", "Discussion",
            "Conclusion", "Conclusions", "Future Work",
            "Acknowledgements", "Acknowledgments", "Appendix",
        ]
        for sec in common_sections:
            if line.lower() == sec.lower():
                level = 1
                if sec.lower() in ["abstract", "references"]:
                    return False, "", 0  # 这些单独处理
                return True, sec, level

        # 模式3：全大写短标题（如 "RELATED WORK"）
        if re.match(r"^[A-Z][A-Z\s\-]{3,40}$", line) and " " in line:
            # 排除全是缩写的情况
            words = line.split()
            if all(len(w) > 1 for w in words):
                return True, line.title(), 1

        return False, "", 0

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        提取 PDF 中的表格（用 pdfplumber）

        Returns:
            表格列表，每个包含页码和表格数据
        """
        if not self.use_pdfplumber:
            return []

        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:
                            tables.append({
                                "page": page_num,
                                "data": table,
                            })
        except Exception as e:
            print(f"[表格提取出错] {e}")

        return tables
