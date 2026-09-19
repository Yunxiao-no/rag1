"""
引用管理模块
管理论文引用信息，支持生成 BibTeX、格式化引用列表
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Citation:
    """一条引用"""
    key: str                    # 引用键，如 "zhang2024agent"
    title: str                  # 论文标题
    authors: List[str] = field(default_factory=list)
    year: str = ""
    venue: str = ""             # 会议/期刊
    url: str = ""
    arxiv_id: str = ""

    def to_bibtex(self) -> str:
        """生成 BibTeX 格式"""
        author_str = " and ".join(self.authors)
        return f"""@article{{{self.key},
  title={{{self.title}}},
  author={{{author_str}}},
  year={{{self.year}}},
  journal={{{self.venue}}},
  url={{{self.url}}}
}}"""

    def to_text(self, style: str = "apa") -> str:
        """
        生成文本格式引用

        Args:
            style: 引用格式，apa / ieee / simple
        """
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."

        if style == "apa":
            return f"{authors_str} ({self.year}). {self.title}. {self.venue}."
        elif style == "ieee":
            return f"{authors_str}, \"{self.title},\" {self.venue}, {self.year}."
        else:  # simple
            return f"{authors_str}. {self.title}. {self.venue}, {self.year}."


class CitationManager:
    """
    引用管理器

    功能：
    1. 从论文信息自动生成引用键
    2. 管理引用集合（去重、查找）
    3. 导出 BibTeX 文件
    4. 生成格式化的参考文献列表
    5. 从文本中提取引用（解析 [1] 格式）
    """

    def __init__(self):
        self._citations: Dict[str, Citation] = {}

    def add(self, citation: Citation) -> str:
        """
        添加一条引用

        Returns:
            引用键
        """
        # 如果键已存在，自动加后缀
        key = citation.key
        suffix = 1
        while key in self._citations:
            key = f"{citation.key}_{suffix}"
            suffix += 1
        citation.key = key
        self._citations[key] = citation
        return key

    def add_from_paper_info(self, title: str, authors: List[str], year: str = "",
                             venue: str = "", url: str = "", arxiv_id: str = "") -> str:
        """
        从论文信息自动生成引用并添加

        Returns:
            引用键
        """
        key = self._generate_key(title, authors, year)
        citation = Citation(
            key=key,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            url=url,
            arxiv_id=arxiv_id,
        )
        return self.add(citation)

    def get(self, key: str) -> Optional[Citation]:
        """根据键获取引用"""
        return self._citations.get(key)

    def find_by_title(self, title_keyword: str) -> List[Citation]:
        """根据标题关键词查找引用"""
        results = []
        for cite in self._citations.values():
            if title_keyword.lower() in cite.title.lower():
                results.append(cite)
        return results

    def remove(self, key: str) -> bool:
        """删除一条引用"""
        if key in self._citations:
            del self._citations[key]
            return True
        return False

    def list_all(self) -> List[Citation]:
        """列出所有引用"""
        return list(self._citations.values())

    def count(self) -> int:
        """引用数量"""
        return len(self._citations)

    def export_bibtex(self, filepath: str) -> str:
        """
        导出为 BibTeX 文件

        Returns:
            BibTeX 内容
        """
        entries = [cite.to_bibtex() for cite in self._citations.values()]
        content = "\n\n".join(entries)

        if filepath:
            import os
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def generate_reference_list(self, style: str = "apa") -> str:
        """
        生成格式化的参考文献列表

        Args:
            style: 引用格式

        Returns:
            参考文献列表文本
        """
        citations = sorted(self._citations.values(), key=lambda c: (c.year, c.title))
        lines = ["## 参考文献\n"]
        for i, cite in enumerate(citations, 1):
            lines.append(f"{i}. {cite.to_text(style)}")
        return "\n".join(lines)

    def extract_citations_from_text(self, text: str) -> List[str]:
        """
        从文本中提取 [数字] 格式的引用标记

        Returns:
            引用编号列表，如 ["1", "2", "3"]
        """
        matches = re.findall(r'\[(\d+)\]', text)
        return sorted(set(matches), key=int)

    def _generate_key(self, title: str, authors: List[str], year: str) -> str:
        """
        自动生成引用键
        格式：第一作者姓 + 年份 + 标题第一个实词
        如：zhang2024agent
        """
        # 第一作者姓
        first_author = authors[0] if authors else "unknown"
        # 取姓（简单处理：取最后一个词，中文取第一个字）
        parts = first_author.split()
        if len(parts) > 1:
            last_name = parts[-1]
        else:
            last_name = first_author[:1] if any('\u4e00' <= c <= '\u9fff' for c in first_author) else first_author

        # 年份
        year_short = year[-2:] if len(year) >= 2 else "xx"

        # 标题第一个实词（跳过冠词介词）
        stop_words = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with"}
        title_words = re.findall(r'[a-zA-Z]+', title.lower())
        first_word = "paper"
        for w in title_words:
            if w not in stop_words and len(w) > 2:
                first_word = w
                break

        key = f"{last_name.lower()}{year_short}{first_word}"
        # 清理非字母数字字符
        key = re.sub(r'[^a-z0-9]', '', key)
        return key
