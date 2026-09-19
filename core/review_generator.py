"""
综述生成模块
基于多篇论文自动生成文献综述，支持多维度对比分析
"""
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from .rag_engine import RAGEngine
from .citation_manager import CitationManager


@dataclass
class ReviewSection:
    """综述的一个章节"""
    title: str
    content: str
    citations: List[str] = field(default_factory=list)


@dataclass
class LiteratureReview:
    """完整的文献综述"""
    topic: str
    summary: str = ""
    sections: List[ReviewSection] = field(default_factory=list)
    comparison_table: str = ""
    research_gaps: str = ""
    future_directions: str = ""
    references: str = ""

    def to_markdown(self) -> str:
        """导出为 Markdown"""
        lines = [f"# 文献综述：{self.topic}\n"]

        if self.summary:
            lines.append("## 摘要\n")
            lines.append(self.summary)
            lines.append("")

        for sec in self.sections:
            lines.append(f"## {sec.title}\n")
            lines.append(sec.content)
            lines.append("")

        if self.comparison_table:
            lines.append("## 研究方法对比\n")
            lines.append(self.comparison_table)
            lines.append("")

        if self.research_gaps:
            lines.append("## 研究空白\n")
            lines.append(self.research_gaps)
            lines.append("")

        if self.future_directions:
            lines.append("## 未来研究方向\n")
            lines.append(self.future_directions)
            lines.append("")

        if self.references:
            lines.append(self.references)

        return "\n".join(lines)


class ReviewGenerator:
    """
    文献综述生成器

    工作流程：
    1. 基于研究主题检索相关论文
    2. 对论文进行多维度分析（方法、数据集、实验结果、贡献）
    3. 生成对比表格
    4. 识别研究空白和未来方向
    5. 组装成完整的文献综述，带引用
    """

    def __init__(self, rag_engine: RAGEngine, citation_manager: CitationManager = None):
        """
        Args:
            rag_engine: RAG 引擎（已索引论文）
            citation_manager: 引用管理器
        """
        self.rag = rag_engine
        self.citations = citation_manager or CitationManager()

        self._llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.4,  # 综述需要一定的创造性，但不能太高
        )

    def generate(self, topic: str, focus_areas: List[str] = None) -> LiteratureReview:
        """
        生成完整的文献综述

        Args:
            topic: 研究主题
            focus_areas: 关注的重点方向，如 ["方法", "应用", "评估"]

        Returns:
            LiteratureReview 对象
        """
        focus_areas = focus_areas or ["研究方法", "核心贡献", "实验评估", "应用场景"]

        review = LiteratureReview(topic=topic)

        # 1. 检索相关内容
        print(f"[综述生成] 正在检索主题: {topic}")
        rag_result = self.rag.query(topic, top_k=10)
        if not rag_result.chunks:
            review.summary = "未找到相关文献，无法生成综述。请先索引相关论文。"
            return review

        # 收集所有检索到的论文信息
        papers_info = self._collect_papers_info(rag_result)

        # 2. 生成摘要
        print("[综述生成] 正在生成摘要...")
        review.summary = self._generate_summary(topic, rag_result)

        # 3. 按关注方向生成各章节
        print("[综述生成] 正在生成各章节...")
        for area in focus_areas:
            section = self._generate_section(topic, area, rag_result)
            if section:
                review.sections.append(section)

        # 4. 生成对比表格
        print("[综述生成] 正在生成对比表格...")
        review.comparison_table = self._generate_comparison_table(topic, papers_info)

        # 5. 识别研究空白
        print("[综述生成] 正在识别研究空白...")
        review.research_gaps = self._identify_gaps(topic, rag_result)

        # 6. 未来研究方向
        print("[综述生成] 正在生成未来方向...")
        review.future_directions = self._generate_future_directions(topic, rag_result)

        # 7. 参考文献
        review.references = self.citations.generate_reference_list("apa")

        print("[综述生成] 完成！")
        return review

    def _collect_papers_info(self, rag_result) -> List[Dict]:
        """从检索结果中收集论文信息"""
        papers = []
        seen_titles = set()
        for chunk in rag_result.chunks:
            title = chunk.metadata.get("paper_title", "Unknown")
            if title not in seen_titles:
                seen_titles.add(title)
                papers.append({
                    "title": title,
                    "authors": chunk.metadata.get("authors", ""),
                    "section": chunk.metadata.get("section_title", ""),
                })
        return papers

    def _generate_summary(self, topic: str, rag_result) -> str:
        """生成综述摘要"""
        context = self._build_context(rag_result)

        prompt = f"""你是一个资深的学术研究者。请基于以下文献内容，为"{topic}"这个研究主题写一段200-300字的综述摘要。

要求：
1. 概括该领域的研究现状和主要进展
2. 提到关键的研究方法和代表性工作
3. 指出当前的主要挑战
4. 使用 [数字] 标注引用来源
5. 语言要学术化、精炼

文献内容：
{context}

请写摘要："""

        response = self._llm.invoke(prompt)
        return response.content

    def _generate_section(self, topic: str, area: str, rag_result) -> Optional[ReviewSection]:
        """生成一个章节"""
        context = self._build_context(rag_result)

        prompt = f"""你是一个资深的学术研究者。请基于以下文献内容，围绕"{topic}"的"{area}"方面，写一段300-500字的分析。

要求：
1. 总结该方面的主要研究方法和技术路线
2. 对比不同工作的优缺点
3. 分析发展趋势
4. 使用 [数字] 标注引用来源
5. 语言要学术化、有条理

文献内容：
{context}

请写"{area}"部分的分析："""

        try:
            response = self._llm.invoke(prompt)
            content = response.content
            if len(content) > 50:
                return ReviewSection(title=area, content=content)
        except Exception as e:
            print(f"[章节生成失败] {area}: {e}")

        return None

    def _generate_comparison_table(self, topic: str, papers: List[Dict]) -> str:
        """生成研究方法对比表格"""
        if not papers:
            return ""

        # 取前8篇论文做对比
        papers = papers[:8]
        paper_titles = [p["title"][:40] + "..." if len(p["title"]) > 40 else p["title"] for p in papers]

        prompt = f"""请基于以下论文列表，生成一个研究方法对比表格的 Markdown。

论文列表：
{chr(10).join(f'- {p["title"]} (作者: {p["authors"]})' for p in papers)}

表格要求：
1. 列包括：论文、研究方法、核心技术、主要贡献、局限性
2. 如果某些信息不确定，填"未明确提及"
3. 表格要简洁，每格不超过30字
4. 只输出 Markdown 表格，不要其他解释

请生成表格："""

        try:
            response = self._llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"[对比表格生成失败] {e}")
            return ""

    def _identify_gaps(self, topic: str, rag_result) -> str:
        """识别研究空白"""
        context = self._build_context(rag_result)

        prompt = f"""你是一个资深的学术研究者。请基于以下文献内容，分析"{topic}"领域存在的研究空白和未解决的问题。

要求：
1. 列出3-5个主要的研究空白
2. 每个空白说明为什么重要、为什么还没解决
3. 使用 [数字] 标注引用
4. 语言要学术化

文献内容：
{context}

请分析研究空白："""

        try:
            response = self._llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"[研究空白识别失败] {e}")
            return ""

    def _generate_future_directions(self, topic: str, rag_result) -> str:
        """生成未来研究方向"""
        context = self._build_context(rag_result)

        prompt = f"""你是一个资深的学术研究者。请基于以下文献内容和已识别的研究空白，提出"{topic}"领域未来3-5年的研究方向。

要求：
1. 提出3-5个具体的研究方向
2. 每个方向说明研究内容、预期成果、可行性
3. 结合当前技术发展趋势
4. 语言要学术化、有前瞻性

文献内容：
{context}

请提出未来研究方向："""

        try:
            response = self._llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"[未来方向生成失败] {e}")
            return ""

    def _build_context(self, rag_result, max_chunks: int = 8) -> str:
        """从 RAG 结果构造上下文"""
        parts = []
        for i, chunk in enumerate(rag_result.chunks[:max_chunks]):
            paper_title = chunk.metadata.get("paper_title", "Unknown")
            section = chunk.metadata.get("section_title", "")
            label = f"[{i+1}] {paper_title}"
            if section:
                label += f" ({section})"
            parts.append(f"{label}:\n{chunk.content[:800]}")  # 限制每块长度
        return "\n\n---\n\n".join(parts)

    def save_review(self, review: LiteratureReview, output_dir: str = None) -> str:
        """
        保存综述到文件

        Returns:
            文件路径
        """
        output_dir = output_dir or settings.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        safe_topic = "".join(c if c.isalnum() or c in " -_" else "_" for c in review.topic)
        filename = f"review_{safe_topic[:50]}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(review.to_markdown())

        # 同时导出 BibTeX
        bib_path = os.path.join(output_dir, f"references_{safe_topic[:30]}.bib")
        self.citations.export_bibtex(bib_path)

        print(f"[综述已保存] {filepath}")
        return filepath
