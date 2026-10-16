# settings.py
"""
环境变量配置（pydantic-settings 强类型）
- 所有路径自动 mkdir
- 派生路径统一在此定义，避免散落
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========== 配置优化开关 ==========
    config_optimize_enabled: bool = True
    config_min_quality_score: float = 0.6

    # ========== 缓存控制 ==========
    clear_cache_on_start: bool = False

    # ========== LLM ==========
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_timeout: int = 60
    llm_max_retries: int = 3
    llm_max_concurrency: int = 16
    llm_batch_size: int = 500

    # ========== Embedding ==========
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_batch_size: int = 256
    embedding_device: str = "cpu"
    embedding_max_length: int = 128

    # ========== BERTopic ==========
    bertopic_min_topic_size: int = 20
    bertopic_zeroshot_similarity: float = 0.7
    bertopic_seed_multiplier: int = 2
    bertopic_umap_neighbors: int = 15
    bertopic_umap_components: int = 5
    bertopic_hdbscan_min_size: int = 20

    # ========== 路径 ==========
    data_dir: str = "./data"
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    log_dir: str = "./log"  # ★ 注意：单数 log
    calls_file: str = "./data/calls.jsonl"
    prompt_file: str = "./prompts.yaml"  # ★ 新增：prompt 文件
    parsed_calls_file: str = "./output/parsed_calls.jsonl"

    # ========== 日志 ==========
    log_level: str = "INFO"
    log_rotation: str = "100 MB"
    log_retention: str = "7 days"
    log_format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    )
    log_enable_file: bool = True
    log_enable_console: bool = True

    # ========== 在线 ==========
    online_top_k: int = 3
    online_timeout_ms: int = 500
    online_embedding_cache_size: int = 1000
    online_recall_top_k: int = 20
    online_rrf_k: int = 60

    # ========== 配置优化 ==========
    config_optimize_enabled: bool = True
    config_min_quality_score: float = 0.6

    # ========== 并发 ==========
    concurrency_labeler: int = 16
    concurrency_embedding: int = 4
    concurrency_retrieval: int = 4

    # ========== 派生路径 ==========
    @property
    def session_labels_file(self) -> Path:
        return self.output_path / "session_labels.jsonl"

    @property
    def pair_labels_file(self) -> Path:
        return self.output_path / "pair_labels.jsonl"

    @property
    def labels_file(self) -> Path:
        """兼容旧调用，指向 session_labels"""
        return self.session_labels_file

    @property
    def parsed_calls_path(self) -> Path:
        """解析后会话缓存路径（自动创建父目录）"""
        p = Path(self.parsed_calls_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_path(self) -> Path:
        p = Path(self.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def labels_file(self) -> Path:
        return self.output_path / "labels.jsonl"

    @property
    def optimized_config_file(self) -> Path:
        return self.output_path / "optimized_config.json"

    @property
    def bertopic_model_dir(self) -> Path:
        return self.output_path / "bertopic_model"

    @property
    def qa_pairs_file(self) -> Path:
        return self.output_path / "qa_pairs.json"

    @property
    def sop_graph_file(self) -> Path:
        return self.output_path / "sop_graph.json"

    @property
    def faiss_dir(self) -> Path:
        return self.output_path / "faiss"

    @property
    def testset_file(self) -> Path:
        return self.output_path / "testset.jsonl"


_settings = None


def get_settings() -> Settings:
    """全局单例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
