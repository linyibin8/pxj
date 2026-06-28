from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "知进拍学"
    public_base_url: str = "https://pxj.evowit.com"
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/pxj.sqlite3"
    redis_url: str = ""
    llm_base_url: str = "http://100.64.0.5:39000/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "evowit-agent27b"
    llm_provider: str = "evowit"
    llm_gateway_url: str = ""
    llm_gateway_key: str = ""
    # 高质量「精批/精准分题」后台路径：前沿大模型(GPT-5.5) Responses API 网关。
    # 有 url+key 才启用；批改/分题路由到它(外部，不占本地 27B GPU gate)，实时问答仍用 27B。
    grading_llm_url: str = "http://100.64.0.13:8080"
    grading_llm_key: str = ""
    grading_llm_model: str = "gpt-5.5"
    grading_llm_seg_effort: str = "low"     # 分题 bbox：low 已够准(~35s)
    grading_llm_grade_effort: str = "medium"  # 批改判分：medium 平衡质量/延迟(~60-105s, <iOS 180s 超时)
    grading_llm_max_concurrency: int = 3
    grading_llm_timeout_seconds: float = 170.0
    embed_url: str = ""
    embed_dim: int = 512
    llm_max_concurrency: int = 1
    llm_min_interval_seconds: float = 8.0
    llm_queue_warn_size: int = 3
    # Background jobs (visualization/report) yield the model to realtime voice/QA:
    # they only run once the realtime lane has been idle this long, and defer at
    # most this long before forcing through (so they never starve).
    llm_background_idle_seconds: float = 12.0
    llm_background_max_defer_seconds: float = 240.0
    llm_realtime_retry: int = 1
    # Free-tier cap: max realtime calls per account per UTC day on the DEFAULT (our)
    # model. 0 = unlimited. Users who configure their own model key are never capped.
    free_daily_quota: int = 0
    # When a realtime QA follow-up frame is essentially the same page as the
    # previous QA image, answer on carried context (text-only) instead of a
    # redundant vision call. Set PXJ_QA_SKIP_DUPLICATE_FRAME_VISION=false to disable.
    qa_skip_duplicate_frame_vision: bool = True
    max_upload_mb: int = 20
    control_token: str = ""
    auth_required: bool = False
    registration_enabled: bool = True
    observe_demo_public: bool = False
    auth_secret_key: str = ""
    auth_token_ttl_minutes: int = 43200
    default_account_id: str = "local"

    model_config = SettingsConfigDict(env_prefix="PXJ_")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "images").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    return settings
