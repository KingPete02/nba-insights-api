from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Force the app to read test settings BEFORE importing app.*
os.environ["PYDANTIC_ENV_FILE"] = ".env.test"
os.environ["ENV"] = "test"

from app.core.config import settings  # noqa: E402
from app.db.base import Base  # noqa: E402


@pytest.fixture(autouse=True, scope="function")
async def _reset_db() -> None:
    engine = create_async_engine(settings.DATABASE_URL, future=True)

    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE;"))
        await conn.execute(text("CREATE SCHEMA public;"))
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
