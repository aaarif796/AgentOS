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

    @property
    def database_path(self) -> Path:
        if self.database_url.startswith("sqlite"):
            return Path(self.database_url.rsplit("/", 1)[-1])
        return Path("agentos.db")
