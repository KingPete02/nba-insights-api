from __future__ import annotations

import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("PYDANTIC_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENV: str = Field(default="development")

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/nba_insights"
    )

    JWT_SECRET_KEY: str = Field(default="CHANGE_ME_SUPER_SECRET")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)

    CORS_ALLOW_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


settings = Settings()
