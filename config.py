"""
ResearchAgent 配置文件
从 .env 文件读取配置，也支持环境变量覆盖
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """项目配置，自动从 .env 文件读取"""

    # 大模型配置
    LLM_API_KEY: str = Field(default="", description="大模型 API Key")
    LLM_BASE_URL: str = Field(default="https://api.deepseek.com", description="大模型 API 地址")
    LLM_MODEL: str = Field(default="deepseek-chat", description="大模型名称")
    LLM_TEMPERATURE: float = Field(default=0.3, description="生成温度，越低越确定")

    # Embedding 配置
    EMBEDDING_API_KEY: str = Field(default="", description="Embedding API Key")
    EMBEDDING_BASE_URL: str = Field(default="https://api.deepseek.com")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")

    # arXiv 配置
    ARXIV_MAX_RESULTS: int = Field(default=10, description="每次检索最多返回论文数")
    ARXIV_SORT_BY: str = Field(default="relevance", description="排序方式：relevance / submittedDate")

    # RAG 配置
    CHUNK_SIZE: int = Field(default=500, description="文本分块大小（字符数）")
    CHUNK_OVERLAP: int = Field(default=50, description="相邻块重叠字符数")
    TOP_K: int = Field(default=5, description="检索时返回最相似的块数")

    # 路径配置
    DATA_DIR: str = Field(default="data", description="数据目录")
    PAPERS_DIR: str = Field(default="data/papers", description="论文 PDF 存放目录")
    VECTOR_DB_DIR: str = Field(default="data/vector_db", description="向量数据库存放目录")
    OUTPUT_DIR: str = Field(default="output", description="输出目录")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# 全局单例
settings = Settings()
