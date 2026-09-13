from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTOS_", env_file=".env", extra="ignore")
    env: str = "development"
    database_url: str = "sqlite+aiosqlite:///./agentos.db"
    workspace: Path = Path(".")
    default_model: str = "openai/gpt-4o-mini"
    fallback_models: list[str] = Field(
        default_factory=lambda: [
            "anthropic/claude-3-5-haiku-latest",
            "gemini/gemini-2.0-flash",
        ]
    )
    max_steps: int = 50
    terminal_timeout: int = 30
    log_level: str = "INFO"
    allow_network: bool = True

    # ---- model policy / budget ------------------------------------------
    model_policy: str = "local-first"
    ollama_url: str = "http://127.0.0.1:11434"
    free_models: list[str] = Field(
        default_factory=lambda: [
            "groq/llama-3.3-70b-versatile",
            "openrouter/deepseek/deepseek-chat-v3-2:free",
            "openrouter/nvidia/nemotron-3-super:free",
            "nvidia/deepseek-r1",
            "deepseek/deepseek-chat",
            "mistral/mistral-small-latest",
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ]
    )
    local_models: list[str] = Field(
        default_factory=lambda: [
            "ollama/llama3.2:3b",
            "ollama/llama3.1:8b",
            "ollama/qwen3:8b",
            "ollama/qwen2.5:14b",
            "ollama/qwen2.5-coder:7b",
            "ollama/mistral:7b",
            "ollama/mistral-nemo:12b",
            "ollama/gemma3:12b",
            "ollama/gemma3:4b",
            "ollama/phi4-mini",
            "ollama/deepseek-r1:8b",
        ]
    )
    task_token_budget: int = 200_000
    max_attempts: int = 3
    max_tool_steps: int = 8
    verify_llm_review: bool = True
    verify_with_tools: bool = True
    verify_on_error: str = "fail_closed"  # fail_closed | fail_open
    model_cooldown_seconds: int = 60
    model_max_consecutive_failures: int = 3

    # ---- memory ----------------------------------------------------------
    memory_enabled: bool = True
    memory_max_short_term: int = 40
    memory_top_k: int = 5

    # ---- book / slides ---------------------------------------------------
    book_output_dir: Path = Path("books")
    book_max_chapter_attempts: int = 3
    book_require_approval: bool = False
    slides_output_dir: Path = Path("slides")
    slides_max_section_attempts: int = 3
    scrape_max_bytes: int = 300_000
    scrape_timeout: float = 20.0
    scrape_allow_private: bool = False

    @property
    def database_path(self) -> Path:
        if self.database_url.startswith("sqlite"):
            return Path(self.database_url.rsplit("/", 1)[-1])
        return Path("agentos.db")
