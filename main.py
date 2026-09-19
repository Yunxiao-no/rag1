#!/usr/bin/env python3
"""
ResearchAgent 命令行入口
支持：检索论文、下载论文、索引论文、问答、生成综述
"""
import os
import sys
import argparse

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from core.paper_searcher import PaperSearcher
from core.pdf_parser import PDFParser
from core.rag_engine import RAGEngine
from core.review_generator import ReviewGenerator
from core.citation_manager import CitationManager


def ensure_dirs():
    """确保必要目录存在"""
    for d in [settings.DATA_DIR, settings.PAPERS_DIR, settings.VECTOR_DB_DIR, settings.OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)


def cmd_search(args):
    """检索论文"""
    print(f"\n🔍 正在检索: {args.query}")
    searcher = PaperSearcher(max_results=args.limit)
    papers = searcher.search(args.query)

    if not papers:
        print("未找到相关论文")
        return

    print(f"\n找到 {len(papers)} 篇论文：\n")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper}")
        print(f"   摘要: {paper.abstract[:100]}...")
        print(f"   链接: {paper.url}")
        print()

    # 询问是否下载
    if args.download:
        print("开始下载论文...")
        searcher.download_papers(papers)


def cmd_index(args):
    """索引本地 PDF 论文到向量数据库"""
    ensure_dirs()
    pdf_dir = args.dir or settings.PAPERS_DIR

    if not os.path.exists(pdf_dir):
        print(f"目录不存在: {pdf_dir}")
        return

    # 查找所有 PDF
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"目录中没有 PDF 文件: {pdf_dir}")
        return

    print(f"找到 {len(pdf_files)} 个 PDF 文件，开始解析和索引...")

    parser = PDFParser()
    rag = RAGEngine()

    total_chunks = 0
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n[{i}/{len(pdf_files)}] 解析: {pdf_file}")

        try:
            parsed = parser.parse(pdf_path)
            print(f"  标题: {parsed.title or '未知'}")
            print(f"  页数: {parsed.total_pages}")
            print(f"  章节数: {len(parsed.sections)}")

            # 索引章节
            chunks = rag.index_paper_sections(
                paper_title=parsed.title or pdf_file,
                sections=parsed.sections,
                authors=parsed.authors,
            )
            total_chunks += chunks
            print(f"  索引了 {chunks} 个文本块")

        except Exception as e:
            print(f"  解析失败: {e}")

    print(f"\n✅ 索引完成！共索引 {total_chunks} 个文本块")
    stats = rag.get_stats()
    print(f"向量数据库统计: {stats}")


def cmd_ask(args):
    """基于已索引论文问答"""
    rag = RAGEngine()
    stats = rag.get_stats()

    if stats["total_chunks"] == 0:
        print("向量数据库为空，请先运行 index 命令索引论文")
        return

    print(f"已索引 {stats['total_chunks']} 个文本块")
    print(f"已索引论文: {rag.list_papers()}\n")

    if args.question:
        # 单次提问
        result = rag.query(args.question, use_mixed=args.mixed)
        print(f"问: {args.question}\n")
        print(f"答: {result.answer}\n")
        if result.sources:
            print("参考来源:")
            for src in result.sources:
                print(f"  [{src['index']}] {src['paper_title']} - {src['section_title']} (相似度: {src['score']:.3f})")
    else:
        # 交互模式
        print("进入问答模式，输入 exit 退出\n")
        while True:
            try:
                question = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if question.lower() in ("exit", "quit", "q"):
                print("再见！")
                break
            if not question:
                continue

            result = rag.query(question, use_mixed=args.mixed)
            print(f"\n助手: {result.answer}\n")
            if result.sources:
                print("参考来源:")
                for src in result.sources[:3]:
                    print(f"  [{src['index']}] {src['paper_title']}")
            print()


def cmd_review(args):
    """生成文献综述"""
    ensure_dirs()
    rag = RAGEngine()
    stats = rag.get_stats()

    if stats["total_chunks"] == 0:
        print("向量数据库为空，请先运行 index 命令索引论文")
        return

    print(f"基于 {stats['total_chunks']} 个文本块生成综述...")

    citation_mgr = CitationManager()
    # 把已索引的论文加入引用管理
    for title in rag.list_papers():
        citation_mgr.add_from_paper_info(title=title, authors=[], year="2024")

    generator = ReviewGenerator(rag, citation_mgr)
    review = generator.generate(args.topic)

    # 保存
    filepath = generator.save_review(review)
    print(f"\n✅ 综述已生成: {filepath}")
    print("\n预览（前500字）:")
    print(review.to_markdown()[:500])


def cmd_stats(args):
    """查看向量数据库统计"""
    rag = RAGEngine()
    stats = rag.get_stats()
    print("向量数据库统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n已索引论文 ({len(rag.list_papers())}):")
    for title in rag.list_papers():
        print(f"  - {title}")


def main():
    parser = argparse.ArgumentParser(
        description="ResearchAgent - 智能学术研究助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py search "large language model agent" --limit 5
  python main.py search "reinforcement learning" --download
  python main.py index --dir data/papers
  python main.py ask "什么是RAG？"
  python main.py ask --mixed
  python main.py review "大语言模型Agent的研究进展"
  python main.py stats
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # search 命令
    search_parser = subparsers.add_parser("search", help="检索论文")
    search_parser.add_argument("query", help="检索关键词")
    search_parser.add_argument("--limit", type=int, default=10, help="返回数量")
    search_parser.add_argument("--download", action="store_true", help="检索后自动下载")

    # index 命令
    index_parser = subparsers.add_parser("index", help="索引本地 PDF 论文")
    index_parser.add_argument("--dir", help="PDF 目录，默认 data/papers")

    # ask 命令
    ask_parser = subparsers.add_parser("ask", help="基于论文问答")
    ask_parser.add_argument("question", nargs="?", help="问题（不填则进入交互模式）")
    ask_parser.add_argument("--mixed", action="store_true", help="使用混合检索（向量+BM25）")

    # review 命令
    review_parser = subparsers.add_parser("review", help="生成文献综述")
    review_parser.add_argument("topic", help="综述主题")

    # stats 命令
    subparsers.add_parser("stats", help="查看索引统计")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "search": cmd_search,
        "index": cmd_index,
        "ask": cmd_ask,
        "review": cmd_review,
        "stats": cmd_stats,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
