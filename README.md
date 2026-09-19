# 📚 ResearchAgent - 智能学术研究助手

> 基于 RAG（检索增强生成）的学术论文研究助手，支持论文检索、PDF 解析、智能问答、文献综述自动生成。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ 功能特性

### 🔍 论文检索
- 基于 arXiv API 的学术论文检索
- 支持关键词、分类、作者多维度检索
- 一键批量下载论文 PDF

### 📄 PDF 智能解析
- 专门针对学术论文的 PDF 解析
- 自动识别标题、作者、摘要、章节、参考文献
- 支持表格提取（pdfplumber）
- 双引擎解析（pypdf 快速 + pdfplumber 精确）

### 🧠 RAG 智能问答
- 基于向量检索的论文问答
- 递归字符分块 + 重叠窗口
- 混合检索（向量相似度 + BM25 关键词）
- 回答带引用来源标注，可溯源
- 支持元数据过滤（按论文、章节筛选）

### 📝 文献综述自动生成
- 基于多篇论文自动生成结构化文献综述
- 多维度分析（研究方法、核心贡献、实验评估、应用场景）
- 自动生成研究方法对比表格
- 智能识别研究空白和未来方向
- 自动管理引用，导出 BibTeX

### 🖥️ 双端交互
- 命令行工具（适合脚本化和批量处理）
- Streamlit Web 界面（可视化操作，适合演示）

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户交互层                            │
│  ┌──────────────┐          ┌──────────────────────┐     │
│  │  CLI (main.py)│          │  Web UI (app.py)     │     │
│  └──────┬───────┘          └──────────┬───────────┘     │
└─────────┼───────────────────────────────┼─────────────────┘
          │                               │
┌─────────▼───────────────────────────────▼─────────────────┐
│                     核心业务层                               │
│  ┌────────────┐ ┌──────────┐ ┌────────┐ ┌──────────────┐ │
│  │PaperSearcher│ │PDFParser │ │RAGEngine│ │ReviewGenerator│ │
│  │  论文检索    │ │ PDF解析   │ │ 检索问答 │ │  综述生成      │ │
│  └────────────┘ └──────────┘ └───┬────┘ └──────┬───────┘ │
│                                     │               │         │
│                              ┌──────▼───────────────▼─────┐ │
│                              │    CitationManager          │ │
│                              │      引用管理                 │ │
│                              └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│                     数据存储层                                 │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  arXiv API    │  │  ChromaDB      │  │  本地文件系统   │  │
│  │  论文数据源    │  │  向量数据库     │  │  PDF/报告/引用  │  │
│  └──────────────┘  └───────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- 大模型 API Key（推荐 DeepSeek，也支持 OpenAI、通义千问等）

### 2. 安装

```bash
# 克隆仓库
git clone https://github.com/yourname/ResearchAgent.git
cd ResearchAgent

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
# LLM_API_KEY=your_api_key_here
# LLM_BASE_URL=https://api.deepseek.com
# LLM_MODEL=deepseek-chat
```

### 4. 运行

#### 命令行方式

```bash
# 检索论文
python main.py search "large language model agent" --limit 5

# 检索并下载
python main.py search "reinforcement learning" --download

# 索引本地 PDF
python main.py index --dir data/papers

# 问答（单次）
python main.py ask "什么是RAG？"

# 问答（交互模式）
python main.py ask

# 生成文献综述
python main.py review "大语言模型Agent的研究进展"

# 查看索引统计
python main.py stats
```

#### Web 界面方式

```bash
streamlit run app.py
```

然后在浏览器打开 http://localhost:8501

---

## 📖 使用示例

### 示例1：检索并索引论文

```bash
# 1. 检索 "agent" 相关论文并下载
python main.py search "llm agent" --limit 5 --download

# 2. 索引下载的 PDF
python main.py index

# 3. 查看索引状态
python main.py stats
```

### 示例2：智能问答

```bash
python main.py ask
```

```
你: 什么是ReAct框架？
助手: ReAct（Reasoning + Acting）是一种让大语言模型结合推理和行动的框架...
参考来源:
  [1] ReAct: Synergizing Reasoning and Acting in Language Models
  [2] A Survey on Large Language Model based Autonomous Agents
```

### 示例3：生成文献综述

```bash
python main.py review "大语言模型Agent的研究进展"
```

输出会保存在 `output/review_大语言模型Agent的研究进展.md`，包含：
- 摘要
- 各维度分析章节
- 研究方法对比表格
- 研究空白
- 未来研究方向
- 参考文献列表

---

## 📁 项目结构

```
ResearchAgent/
├── README.md                    # 项目说明（本文件）
├── 学习文档.md                   # 详细学习文档（原理+代码讲解）
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── .gitignore
├── config.py                    # 配置管理
├── main.py                      # 命令行入口
├── app.py                       # Streamlit Web 界面
├── core/
│   ├── __init__.py
│   ├── paper_searcher.py        # 论文检索（arXiv API）
│   ├── pdf_parser.py            # PDF 解析（学术论文专用）
│   ├── rag_engine.py            # RAG 引擎（分块+检索+生成）
│   ├── review_generator.py      # 文献综述生成
│   └── citation_manager.py      # 引用管理（BibTeX 导出）
├── data/
│   ├── papers/                  # 下载的论文 PDF
│   └── vector_db/               # ChromaDB 向量数据库
├── output/                      # 生成的综述和报告
└── tests/
    └── test_basic.py            # 单元测试
```

---

## 🔧 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 大模型 | DeepSeek / OpenAI / 通义千问 | 文本生成、问答 |
| Embedding | OpenAI text-embedding / BGE | 文本向量化 |
| RAG 框架 | LangChain | 分块、链编排 |
| 向量数据库 | ChromaDB | 向量存储和检索 |
| PDF 解析 | pypdf + pdfplumber | 论文 PDF 解析 |
| 论文检索 | arxiv API | 学术论文检索 |
| Web 界面 | Streamlit | 可视化交互 |
| 配置管理 | Pydantic Settings | 环境变量管理 |

---

## 📈 性能优化

### 已实现的优化
1. **递归字符分块**：优先按段落、句子边界切分，比固定大小分块更合理
2. **重叠窗口**：相邻块重叠 10%，避免上下文断裂
3. **混合检索**：向量检索 + BM25 关键词检索，兼顾语义和精确匹配
4. **元数据过滤**：支持按论文、章节过滤检索范围
5. **双 PDF 引擎**：pypdf 快速提取 + pdfplumber 精确布局，自动降级

### 可扩展的优化方向
- [ ] 重排序（Rerank）：用 Cross-Encoder 对检索结果精排
- [ ] 查询改写（Query Rewriting）：用 LLM 优化查询
- [ ] 分层索引：摘要索引 + 全文索引，先粗筛再精检
- [ ] GraphRAG：构建知识图谱，支持关系推理
- [ ] 多模态支持：解析论文中的图表
- [ ] 批量索引加速：并行处理和增量索引

---

## 🧪 测试

```bash
# 运行单元测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_basic.py::TestCitationManager -v
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [LangChain](https://python.langchain.com/) - RAG 开发框架
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [arXiv](https://arxiv.org/) - 学术论文开放获取平台
- [Streamlit](https://streamlit.io/) - 数据应用框架

---

## 📞 联系方式

- GitHub: [@yourname](https://github.com/yourname)
- Email: your.email@example.com

---

**如果这个项目对你有帮助，欢迎给个 Star ⭐**
