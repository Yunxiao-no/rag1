#!/usr/bin/env python3
"""
ResearchAgent Streamlit Web 界面
运行方式：streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import settings
from core.paper_searcher import PaperSearcher
from core.pdf_parser import PDFParser
from core.rag_engine import RAGEngine
from core.review_generator import ReviewGenerator
from core.citation_manager import CitationManager

# 页面配置
st.set_page_config(
    page_title="ResearchAgent - 智能学术研究助手",
    page_icon="📚",
    layout="wide",
)

# 初始化 session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()
if "searcher" not in st.session_state:
    st.session_state.searcher = PaperSearcher()
if "parser" not in st.session_state:
    st.session_state.parser = PDFParser()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def main():
    st.title("📚 ResearchAgent - 智能学术研究助手")
    st.markdown("基于 RAG 的学术论文研究助手，支持论文检索、PDF 解析、智能问答、文献综述生成")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 功能导航")
        page = st.radio(
            "选择功能",
            ["🔍 论文检索", "📄 PDF 解析与索引", "💬 智能问答", "📝 文献综述", "📊 索引统计"],
        )

        st.divider()
        stats = st.session_state.rag.get_stats()
        st.metric("已索引文本块", stats["total_chunks"])
        st.metric("已索引论文", len(st.session_state.rag.list_papers()))

    if page == "🔍 论文检索":
        render_search()
    elif page == "📄 PDF 解析与索引":
        render_index()
    elif page == "💬 智能问答":
        render_qa()
    elif page == "📝 文献综述":
        render_review()
    elif page == "📊 索引统计":
        render_stats()


def render_search():
    st.header("🔍 论文检索")
    st.markdown("从 arXiv 检索学术论文")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("检索关键词", placeholder="例如：large language model agent")
    with col2:
        max_results = st.number_input("返回数量", min_value=1, max_value=50, value=10)
    with col3:
        category = st.text_input("分类（可选）", placeholder="如 cs.AI")

    if st.button("🔍 检索", type="primary"):
        if not query:
            st.warning("请输入检索关键词")
            return

        with st.spinner("正在检索论文..."):
            searcher = st.session_state.searcher
            searcher.max_results = max_results
            if category:
                papers = searcher.search_by_category(category, query)
            else:
                papers = searcher.search(query)

        if not papers:
            st.info("未找到相关论文")
            return

        st.success(f"找到 {len(papers)} 篇论文")

        for i, paper in enumerate(papers, 1):
            with st.expander(f"{i}. {paper.title}", expanded=(i == 1)):
                st.markdown(f"**作者**: {', '.join(paper.authors[:5])}")
                st.markdown(f"**发布时间**: {paper.published.strftime('%Y-%m-%d') if paper.published else '未知'}")
                st.markdown(f"**分类**: {', '.join(paper.categories)}")
                st.markdown(f"**摘要**: {paper.abstract}")
                st.markdown(f"**链接**: [{paper.url}]({paper.url})")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⬇️ 下载 PDF", key=f"dl_{i}"):
                        with st.spinner("正在下载..."):
                            path = searcher.download_paper(paper)
                            if path:
                                st.success(f"已下载: {path}")
                            else:
                                st.error("下载失败")
                with col2:
                    st.markdown(f"[📄 打开原文]({paper.pdf_url})")


def render_index():
    st.header("📄 PDF 解析与索引")
    st.markdown("上传 PDF 文件，自动解析并索引到向量数据库")

    # 上传文件
    uploaded_files = st.file_uploader(
        "上传论文 PDF（可多选）",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        os.makedirs(settings.PAPERS_DIR, exist_ok=True)
        saved_paths = []

        for uploaded_file in uploaded_files:
            save_path = os.path.join(settings.PAPERS_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(save_path)
            st.success(f"已保存: {uploaded_file.name}")

        if st.button("🚀 开始解析和索引", type="primary"):
            rag = st.session_state.rag
            parser = st.session_state.parser

            progress_bar = st.progress(0)
            total_chunks = 0

            for i, path in enumerate(saved_paths):
                with st.spinner(f"正在解析 {os.path.basename(path)}..."):
                    try:
                        parsed = parser.parse(path)
                        chunks = rag.index_paper_sections(
                            paper_title=parsed.title or os.path.basename(path),
                            sections=parsed.sections,
                            authors=parsed.authors,
                        )
                        total_chunks += chunks
                        st.write(f"✅ {parsed.title or os.path.basename(path)}: {chunks} 个文本块")
                    except Exception as e:
                        st.error(f"解析失败 {os.path.basename(path)}: {e}")

                progress_bar.progress((i + 1) / len(saved_paths))

            st.success(f"索引完成！共索引 {total_chunks} 个文本块")
            st.rerun()

    # 显示已索引论文
    st.divider()
    st.subheader("📚 已索引论文")
    papers = st.session_state.rag.list_papers()
    if papers:
        for title in papers:
            st.text(f"  • {title}")
    else:
        st.info("还没有索引任何论文")


def render_qa():
    st.header("💬 智能问答")
    st.markdown("基于已索引论文进行问答，回答带引用来源")

    # 显示聊天历史
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    # 输入框
    if question := st.chat_input("输入你的问题..."):
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索和生成回答..."):
                result = st.session_state.rag.query(question)

            st.markdown(result.answer)

            if result.sources:
                st.markdown("**📖 参考来源:**")
                for src in result.sources:
                    st.markdown(f"- [{src['index']}] **{src['paper_title']}** - {src['section_title']} (相似度: {src['score']:.3f})")

            st.session_state.chat_history.append(("assistant", result.answer))

    # 清空对话
    if st.session_state.chat_history:
        if st.button("🗑️ 清空对话"):
            st.session_state.chat_history = []
            st.rerun()


def render_review():
    st.header("📝 文献综述生成")
    st.markdown("基于已索引论文自动生成文献综述")

    topic = st.text_input("研究主题", placeholder="例如：大语言模型Agent的研究进展")
    focus = st.text_input(
        "关注方向（逗号分隔，可选）",
        placeholder="例如：研究方法,核心贡献,实验评估,应用场景",
    )

    if st.button("✨ 生成综述", type="primary"):
        if not topic:
            st.warning("请输入研究主题")
            return

        stats = st.session_state.rag.get_stats()
        if stats["total_chunks"] == 0:
            st.error("向量数据库为空，请先索引论文")
            return

        with st.spinner("正在生成文献综述，这可能需要几分钟..."):
            citation_mgr = CitationManager()
            for title in st.session_state.rag.list_papers():
                citation_mgr.add_from_paper_info(title=title, authors=[], year="2024")

            generator = ReviewGenerator(st.session_state.rag, citation_mgr)
            focus_areas = [f.strip() for f in focus.split(",") if f.strip()] if focus else None
            review = generator.generate(topic, focus_areas)
            filepath = generator.save_review(review)

        st.success(f"综述已生成并保存到: {filepath}")

        # 显示综述
        st.markdown("---")
        st.markdown(review.to_markdown())

        # 下载按钮
        st.download_button(
            "⬇️ 下载综述 (Markdown)",
            data=review.to_markdown(),
            file_name=f"review_{topic}.md",
            mime="text/markdown",
        )


def render_stats():
    st.header("📊 索引统计")

    rag = st.session_state.rag
    stats = rag.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("集合名称", stats["collection_name"])
        st.metric("总文本块数", stats["total_chunks"])
    with col2:
        st.metric("持久化目录", stats["persist_dir"])
        st.metric("已索引论文数", len(rag.list_papers()))

    st.divider()
    st.subheader("📚 已索引论文列表")
    papers = rag.list_papers()
    if papers:
        for i, title in enumerate(papers, 1):
            st.text(f"{i}. {title}")
    else:
        st.info("还没有索引任何论文")

    st.divider()
    if st.button("🗑️ 清空索引（慎用）", type="secondary"):
        rag.clear()
        st.success("索引已清空")
        st.rerun()


if __name__ == "__main__":
    main()
