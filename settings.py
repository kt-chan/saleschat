from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_max_concurrency: int = 16

    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 256

    use_embedded_db: bool = True
    use_embedded_redis: bool = True
    clear_cache_on_start: bool = False
    enable_state_machine: bool = False

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "cca_db"
    db_user: str = "cca_user"
    db_password: str = "cca_pass"
    db_pool_min: int = 2
    db_pool_max: int = 50

    redis_url: str = ""
    session_ttl: int = 1800

    data_dir: str = "./data"
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    log_dir: str = "./log"
    calls_file: str = "./data/calls.jsonl"
    prompt_file: str = "./prompts.yaml"

    log_level: str = "INFO"
    online_top_k: int = 3
    online_recall_top_k: int = 10
    online_sim_threshold: float = 0.50
    online_context_window: int = 2

    def _mkdir(self, path_str: str) -> Path:
        p = Path(path_str)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_path(self) -> Path:
        return self._mkdir(self.output_dir)

    @property
    def cache_path(self) -> Path:
        return self._mkdir(self.cache_dir)

    @property
    def log_path(self) -> Path:
        return self._mkdir(self.log_dir)

    # === 核心落盘路径 ===
    @property
    def sop_graph_file(self) -> Path:
        return self.output_path / "sop_graph.json"

    @property
    def session_labels_file(self) -> Path:
        return self.output_path / "session_labels.jsonl"

    @property
    def pair_labels_file(self) -> Path:
        return self.output_path / "pair_labels.jsonl"


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
