"""
论文检索模块
通过 arXiv API 检索学术论文，支持关键词搜索、分类筛选、排序
"""
import arxiv
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os
import time

from config import settings


@dataclass
class Paper:
    """论文数据结构"""
    title: str                           # 论文标题
    authors: List[str] = field(default_factory=list)  # 作者列表
    abstract: str = ""                  # 摘要
    url: str = ""                       # 论文页面 URL
    pdf_url: str = ""                   # PDF 下载链接
    published: Optional[datetime] = None  # 发布时间
    updated: Optional[datetime] = None    # 更新时间
    categories: List[str] = field(default_factory=list)  # 分类（如 cs.AI）
    comment: str = ""                   # 作者评论
    journal_ref: str = ""               # 期刊引用
    doi: str = ""                        # DOI
    local_path: str = ""                 # 本地下载路径

    def to_dict(self) -> dict:
        """转字典，方便序列化"""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "published": self.published.isoformat() if self.published else None,
            "categories": self.categories,
            "local_path": self.local_path,
        }

    def __str__(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        date_str = self.published.strftime("%Y-%m-%d") if self.published else "Unknown"
        return f"[{date_str}] {self.title} ({authors_str})"


class PaperSearcher:
    """
    论文检索器
    封装 arXiv API，提供更友好的检索接口
    """

    def __init__(self, max_results: int = None, sort_by: str = None):
        """
        初始化检索器

        Args:
            max_results: 每次检索最大返回数，默认用配置文件的值
            sort_by: 排序方式，relevance（相关度）或 submittedDate（提交时间）
        """
        self.max_results = max_results or settings.ARXIV_MAX_RESULTS
        self.sort_by = sort_by or settings.ARXIV_SORT_BY
        self._client = arxiv.Client()

    def search(self, query: str, max_results: int = None) -> List[Paper]:
        """
        按关键词检索论文

        Args:
            query: 检索关键词，支持 arXiv 高级检索语法
                   例如："agent"、"ti:agent"、"cat:cs.AI AND agent"
            max_results: 覆盖默认的最大返回数

        Returns:
            论文列表
        """
        max_results = max_results or self.max_results

        # 构造检索
        sort = arxiv.SortCriterion.Relevance
        if self.sort_by == "submittedDate":
            sort = arxiv.SortCriterion.SubmittedDate

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort,
            sort_order=arxiv.SortOrder.Descending,
        )

        # 执行检索并转换为 Paper 对象
        papers = []
        try:
            for result in self._client.results(search):
                paper = Paper(
                    title=result.title.strip(),
                    authors=[a.name for a in result.authors],
                    abstract=result.summary.strip(),
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    published=result.published,
                    updated=result.updated,
                    categories=result.categories,
                    comment=result.comment or "",
                    journal_ref=result.journal_ref or "",
                    doi=result.doi or "",
                )
                papers.append(paper)
        except Exception as e:
            print(f"[检索出错] {e}")

        return papers

    def search_by_category(self, category: str, query: str = "", max_results: int = None) -> List[Paper]:
        """
        按分类检索论文

        Args:
            category: arXiv 分类，如 cs.AI、cs.CL、stat.ML
            query: 附加关键词，可为空
            max_results: 最大返回数

        Returns:
            论文列表
        """
        full_query = f"cat:{category}"
        if query:
            full_query += f" AND {query}"
        return self.search(full_query, max_results)

    def search_by_author(self, author: str, max_results: int = None) -> List[Paper]:
        """
        按作者检索论文

        Args:
            author: 作者姓名
            max_results: 最大返回数

        Returns:
            论文列表
        """
        return self.search(f'au:"{author}"', max_results)

    def download_paper(self, paper: Paper, save_dir: str = None) -> str:
        """
        下载论文 PDF

        Args:
            paper: 论文对象
            save_dir: 保存目录，默认用配置文件的 papers_dir

        Returns:
            本地文件路径
        """
        save_dir = save_dir or settings.PAPERS_DIR
        os.makedirs(save_dir, exist_ok=True)

        # 生成安全的文件名
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title)
        safe_title = safe_title[:80]  # 限制文件名长度
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(save_dir, filename)

        # 已下载就跳过
        if os.path.exists(filepath):
            paper.local_path = filepath
            return filepath

        # 下载
        try:
            # 使用 arxiv 库的下载方法
            search = arxiv.Search(id_list=[paper.url.split("/abs/")[-1]])
            result = next(self._client.results(search))
            result.download_pdf(dirpath=save_dir, filename=filename)
            paper.local_path = filepath
            print(f"[下载完成] {paper.title[:50]}...")
        except Exception as e:
            print(f"[下载失败] {paper.title[:50]}... 错误: {e}")
            # 备用方案：直接用 requests 下载
            try:
                import requests
                resp = requests.get(paper.pdf_url, timeout=30)
                resp.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                paper.local_path = filepath
                print(f"[备用下载完成] {paper.title[:50]}...")
            except Exception as e2:
                print(f"[备用下载也失败] {e2}")
                return ""

        time.sleep(1)  # 礼貌性延迟，避免请求过快
        return filepath

    def download_papers(self, papers: List[Paper], save_dir: str = None) -> List[str]:
        """
        批量下载论文

        Args:
            papers: 论文列表
            save_dir: 保存目录

        Returns:
            本地路径列表
        """
        paths = []
        for i, paper in enumerate(papers):
            print(f"[下载进度] {i+1}/{len(papers)}")
            path = self.download_paper(paper, save_dir)
            if path:
                paths.append(path)
        return paths
