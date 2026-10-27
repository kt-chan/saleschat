from pathlib import Path
from typing import Dict, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

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

    # Embedding (common)
    # 注意：本版本 embedding 仅通过远程 TEI 服务提供，进程内不加载任何本地模型。
    # embedding_model 仅作为“与 TEI 侧 --model-id 保持一致”的参考值，
    # 两边向量维度必须相同（BAAI/bge-base-zh-v1.5 → 768 维）。
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embed_cache_size: int = 2000
    embedding_sort_by_length: bool = True

    # Embedding — 远程 TEI
    embedding_tei_url: str = "http://127.0.0.1:8080"
    embedding_tei_timeout: int = 30
    embedding_tei_batch_size: int = 16
    embedding_tei_concurrency: int = 8
    embedding_tei_retry_max: int = 3
    embedding_tei_retry_backoff: float = 2.0
    embedding_tei_truncate: bool = True

    @field_validator("embedding_tei_url")
    @classmethod
    def _validate_tei_url(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError(
                "EMBEDDING_TEI_URL 为必填项（本版本只支持远程 TEI），"
                "请设置为本地 TEI 服务地址，例如 http://127.0.0.1:8080"
            )
        return v.rstrip("/")

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

    prompt_customer_prefix: str = "客户："
    prompt_agent_prefix: str = "催收员："

    # Runtime
    clear_cache_on_start: bool = False
    state_machine_mode: str = "guide"

    # DB (standalone Docker PostgreSQL, no pgvector)
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "cca_db"
    db_user: str = "cca_user"
    db_password: str = "cca_pass"
    db_pool_min: int = 2
    db_pool_max: int = 50
    db_write_batch_size: int = 500
    session_ttl: int = 1800

    # Paths
    data_dir: str = "./data"
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    log_dir: str = "./log"
    calls_file_pattern: str = "*.jsonl"
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
