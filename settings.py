from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ==================== LLM ====================
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_timeout: int = 60
    llm_max_retries: int = 3
    llm_max_concurrency: int = 16
    llm_batch_size: int = 500

    # ==================== Embedding ====================
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_batch_size: int = 256
    embedding_device: str = "cpu"
    embedding_max_length: int = 128

    # ==================== BERTopic ====================
    bertopic_min_topic_size: int = 20
    bertopic_zeroshot_similarity: float = 0.7
    bertopic_seed_multiplier: int = 2
    bertopic_umap_neighbors: int = 15
    bertopic_umap_components: int = 5
    bertopic_hdbscan_min_size: int = 20

    # ==================== 路径 ====================
    data_dir: str = "./data"
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    log_dir: str = "./log"
    calls_file: str = "./data/calls.jsonl"
    prompt_file: str = "./prompts.yaml"
    save_parsed_calls: bool = True

    # ==================== 日志 ====================
    log_level: str = "INFO"
    log_rotation: str = "100 MB"
    log_retention: str = "7 days"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    log_enable_file: bool = True
    log_enable_console: bool = True

    # ==================== 在线检索 ====================
    online_top_k: int = 3
    online_timeout_ms: int = 500
    online_embedding_cache_size: int = 1000
    online_recall_top_k: int = 20
    online_rrf_k: int = 60
    online_rule_cutoff: float = 80.0
    online_intent_min: float = 0.65

    # ==================== 配置与并发 ====================
    clear_cache_on_start: bool = False
    config_optimize_enabled: bool = True
    config_min_quality_score: float = 0.6

    concurrency_labeler: int = 16
    concurrency_embedding: int = 4
    concurrency_retrieval: int = 4

    # ==================== 派生路径 ====================
    def _mkdir(self, path_str: str) -> Path:
        p = Path(path_str)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_path(self) -> Path: return self._mkdir(self.output_dir)
    @property
    def cache_path(self) -> Path: return self._mkdir(self.cache_dir)
    @property
    def log_path(self) -> Path: return self._mkdir(self.log_dir)
    @property
    def parsed_calls_path(self) -> Path: return self.output_path / "parsed_calls.jsonl"
    @property
    def session_labels_file(self) -> Path: return self.output_path / "session_labels.jsonl"
    @property
    def pair_labels_file(self) -> Path: return self.output_path / "pair_labels.jsonl"
    @property
    def qa_pairs_file(self) -> Path: return self.output_path / "qa_pairs.json"
    @property
    def sop_graph_file(self) -> Path: return self.output_path / "sop_graph.json"
    @property
    def faiss_dir(self) -> Path: return self.output_path / "faiss"
    @property
    def stage_emb_file(self) -> Path: return self.output_path / "stage_emb.npy"
    @property
    def intent_emb_file(self) -> Path: return self.output_path / "intent_emb.npy"
    @property
    def intent_labels_file(self) -> Path: return self.output_path / "intent_labels.json"

_settings = None
def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings