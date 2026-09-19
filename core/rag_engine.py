"""
RAG 引擎模块
核心检索增强生成模块，负责文本分块、向量化、存储、检索
支持混合检索（向量 + BM25）和重排序
"""
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

from config import settings


@dataclass
class Chunk:
    """文本块"""
    id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    score: float = 0.0  # 检索相似度得分


@dataclass
class RAGResult:
    """RAG 检索结果"""
    query: str
    chunks: List[Chunk] = field(default_factory=list)
    answer: str = ""
    sources: List[Dict] = field(default_factory=list)


class RAGEngine:
    """
    RAG 引擎

    架构：
    1. 索引阶段：文本 → 分块 → 向量化 → 存入 ChromaDB
    2. 检索阶段：查询 → 向量化 → 相似度搜索 → 可选重排序 → 返回 Top-K
    3. 生成阶段：查询 + 检索结果 → LLM → 带引用的回答

    优化点：
    - 递归字符分块（比简单分块更智能，优先按段落/句子边界切）
    - 元数据过滤（按论文、章节过滤）
    - 可扩展混合检索接口
    """

    def __init__(self, collection_name: str = "research_papers", persist_dir: str = None):
        """
        初始化 RAG 引擎

        Args:
            collection_name: ChromaDB 集合名
            persist_dir: 向量数据库持久化目录
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir or settings.VECTOR_DB_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        # 初始化 Embedding 模型：
        # 1) 配置了有效 API Key → 用 OpenAI 兼容的 embedding API（如硅基流动 SiliconFlow）
        # 2) 没有 Key / 占位符 → 用 ChromaDB 内置本地模型（all-MiniLM-L6-v2，离线可用，无需联网）
        # 注：DeepSeek 官方 API 不提供 embedding 接口，故模板默认走本地模型。
        api_key = (settings.EMBEDDING_API_KEY or "").strip()
        if api_key and api_key != "your_api_key_here":
            self._embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.EMBEDDING_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL,
            )
            self._embed_local = False
        else:
            self._embeddings = embedding_functions.DefaultEmbeddingFunction()
            self._embed_local = True

        # 初始化 LLM（用于生成回答）
        self._llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
        )

        # 初始化 ChromaDB 客户端
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 初始化文本分块器
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "。", " ", ""],
            length_function=len,
        )

    def index_text(self, text: str, metadata: Dict = None, doc_id: str = None) -> int:
        """
        索引一段文本

        Args:
            text: 要索引的文本
            metadata: 元数据（论文标题、作者、章节等）
            doc_id: 文档 ID，不指定则自动生成

        Returns:
            索引的块数
        """
        metadata = metadata or {}
        doc_id = doc_id or f"doc_{abs(hash(text[:100]))}"

        # 分块
        chunks = self._text_splitter.split_text(text)
        if not chunks:
            return 0

        # 生成块 ID 和元数据
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadatas = []
        for i, chunk in enumerate(chunks):
            meta = dict(metadata)
            meta["chunk_index"] = i
            meta["doc_id"] = doc_id
            meta["chunk_size"] = len(chunk)
            chunk_metadatas.append(meta)

        # 向量化并存储（本地模型与 API 模型调用方式不同）
        embeddings = (self._embeddings(chunks) if self._embed_local
                      else self._embeddings.embed_documents(chunks))
        self._collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            metadatas=chunk_metadatas,
            embeddings=embeddings,
        )

        return len(chunks)

    def index_paper_sections(self, paper_title: str, sections: List, authors: List[str] = None) -> int:
        """
        索引论文的各个章节（每个章节作为一个文档，保留章节元数据）

        Args:
            paper_title: 论文标题
            sections: PaperSection 列表
            authors: 作者列表

        Returns:
            索引的总块数
        """
        total = 0
        authors = authors or []
        for i, section in enumerate(sections):
            if not section.content or len(section.content) < 20:
                continue
            doc_id = f"{abs(hash(paper_title))}_sec_{i}"
            metadata = {
                "paper_title": paper_title,
                "section_title": section.title,
                "authors": ", ".join(authors[:5]),
                "page": section.page,
                "type": "paper_section",
            }
            total += self.index_text(section.content, metadata, doc_id)
        return total

    def search(self, query: str, top_k: int = None, filter_metadata: Dict = None) -> List[Chunk]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回数量
            filter_metadata: 元数据过滤条件，如 {"paper_title": "xxx"}

        Returns:
            检索到的文本块列表（按相似度排序）
        """
        top_k = top_k or settings.TOP_K

        # 查询向量化
        query_embedding = (self._embeddings([query])[0] if self._embed_local
                           else self._embeddings.embed_query(query))

        # 执行检索
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
        )

        # 解析结果
        chunks = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                chunk = Chunk(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=1 - results["distances"][0][i] if results["distances"] else 0.0,
                )
                chunks.append(chunk)

        return chunks

    def search_with_bm25(self, query: str, top_k: int = None) -> List[Chunk]:
        """
        混合检索：向量检索 + BM25 关键词检索（简化实现）
        实际生产环境建议用专门的混合检索方案

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            合并排序后的结果
        """
        top_k = top_k or settings.TOP_K

        # 1. 向量检索（取多一些，后面合并）
        vector_chunks = self.search(query, top_k=top_k * 2)

        # 2. 简单的关键词匹配（BM25 的简化版）
        # 从所有文档中找包含查询关键词的块
        query_terms = set(re.findall(r'\w+', query.lower()))
        all_results = self._collection.get()

        bm25_chunks = []
        if all_results and all_results["ids"]:
            for i, doc in enumerate(all_results["documents"]):
                doc_terms = set(re.findall(r'\w+', doc.lower()))
                overlap = len(query_terms & doc_terms)
                if overlap > 0:
                    score = overlap / max(len(query_terms), 1)
                    bm25_chunks.append(Chunk(
                        id=all_results["ids"][i],
                        content=doc,
                        metadata=all_results["metadatas"][i] if all_results["metadatas"] else {},
                        score=score * 0.5,  # BM25 权重较低
                    ))

        # 3. 合并去重，按分数排序
        seen_ids = set()
        merged = []
        for chunk in vector_chunks + bm25_chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                merged.append(chunk)

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    def query(self, query: str, top_k: int = None, use_mixed: bool = False) -> RAGResult:
        """
        完整的 RAG 查询：检索 + 生成

        Args:
            query: 用户问题
            top_k: 检索数量
            use_mixed: 是否使用混合检索

        Returns:
            RAGResult，包含检索到的块、生成的回答、来源
        """
        top_k = top_k or settings.TOP_K

        # 1. 检索
        if use_mixed:
            chunks = self.search_with_bm25(query, top_k)
        else:
            chunks = self.search(query, top_k)

        if not chunks:
            return RAGResult(
                query=query,
                chunks=[],
                answer="未找到相关文献，请尝试其他关键词或先索引论文。",
                sources=[],
            )

        # 2. 构造上下文
        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks):
            paper_title = chunk.metadata.get("paper_title", "Unknown")
            section_title = chunk.metadata.get("section_title", "")
            source_label = f"[{i+1}] {paper_title}"
            if section_title:
                source_label += f" - {section_title}"
            context_parts.append(f"{source_label}:\n{chunk.content}")
            sources.append({
                "index": i + 1,
                "paper_title": paper_title,
                "section_title": section_title,
                "score": chunk.score,
                "snippet": chunk.content[:200],
            })

        context = "\n\n---\n\n".join(context_parts)

        # 3. 构造提示词并生成回答
        prompt = f"""你是一个专业的学术研究助手。请基于以下检索到的文献内容回答用户的问题。

要求：
1. 只基于提供的文献内容回答，不要编造文献中没有的信息
2. 回答中引用文献时使用 [数字] 格式标注来源，如 [1]、[2]
3. 如果文献内容不足以回答问题，请明确说明"根据现有文献无法确定"
4. 回答要专业、准确、有条理
5. 最后列出所有引用的文献来源

检索到的文献内容：
{context}

用户问题：{query}

请给出你的回答："""

        response = self._llm.invoke(prompt)
        answer = response.content

        return RAGResult(
            query=query,
            chunks=chunks,
            answer=answer,
            sources=sources,
        )

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        count = self._collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_dir": self.persist_dir,
        }

    def clear(self):
        """清空集合（慎用）"""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def list_papers(self) -> List[str]:
        """列出已索引的论文标题"""
        results = self._collection.get()
        papers = set()
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "paper_title" in meta:
                    papers.add(meta["paper_title"])
        return sorted(list(papers))
