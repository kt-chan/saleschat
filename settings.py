from pathlib import Path
from typing import Dict, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_max_concurrency: int = 16
    llm_timeout: int = 60
    llm_retry_max: int = 3
    llm_retry_backoff: float = 2.0
    llm_temperature_labeling: float = 0.1
    llm_temperature_synthesis: float = 0.3

    # Embedding
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 256
    embed_cache_size: int = 2000

    # Dialog parsing
    dialog_parse_mode: str = "auto"
    dialog_turn_separators: List[str] = Field(default_factory=lambda: ["；", ";"])
    dialog_role_prefixes: List[str] = Field(
        default_factory=lambda: ["催收员", "客户", "客服", "坐席", "用户"]
    )
    dialog_role_aliases: Dict[str, str] = Field(
        default_factory=lambda: {"客服": "催收员", "坐席": "催收员", "用户": "客户"}
    )
    dialog_colon_pattern: str = "[：:]"
    dialog_customer_role: str = "客户"
    dialog_agent_role: str = "催收员"

    # Rewrite prefixes
    prompt_customer_prefix: str = "客户："
    prompt_agent_prefix: str = "催收员："

    # Runtime
    use_embedded_db: bool = True
    use_embedded_redis: bool = True
    clear_cache_on_start: bool = False
    state_machine_mode: str = "guide"

    # DB / Redis
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "cca_db"
    db_user: str = "cca_user"
    db_password: str = "cca_pass"
    db_pool_min: int = 2
    db_pool_max: int = 50
    db_write_batch_size: int = 500
    redis_url: str = ""
    session_ttl: int = 1800

    # Paths
    data_dir: str = "./data"
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    log_dir: str = "./log"
    calls_file: str = "./data/calls.jsonl"
    prompt_file: str = "./prompts.yaml"
    demo_text: str = ""
    demo_text_file: str = "./data/demo_text.txt"

    # Retrieval
    log_level: str = "INFO"
    online_top_k: int = 3
    online_sim_threshold: float = 0.50

    hybrid_max_hops: int = 2
    hybrid_golden_weight: float = 1.0
    hybrid_pg_weight: float = 0.9
    hybrid_pg_top_k: int = 20
    hybrid_golden_top_k_per_node: int = 2
    hybrid_hop_decay: List[float] = Field(default_factory=lambda: [1.0, 0.95, 0.85])
    hybrid_vote_top_n: int = 5

    # Suggest
    suggest_context_customer_turns: int = 3
    suggest_max_query_chars: int = 1000
    suggest_fallback_script: str = "抱歉，我这边需要先核实一下信息，请您稍等片刻。"

    # Golden synthesis
    golden_exact_per_tag: int = 3
    golden_exact_limit: int = 20
    golden_fuzzy_per_intent: int = 2
    golden_fuzzy_limit: int = 10

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