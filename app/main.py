from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router as api_router
from app.core.config import settings
from app.core.errors import install_error_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestIdLoggingMiddleware

setup_logging()

allow_docs = os.getenv("ALLOW_DOCS", "false").lower() == "true"
is_prod = settings.ENV == "production"

app = FastAPI(
    title="NBA Insights API",
    version="0.1.0",
    docs_url="/docs" if (allow_docs or not is_prod) else None,
    redoc_url="/redoc" if (allow_docs or not is_prod) else None,
    openapi_url="/openapi.json" if (allow_docs or not is_prod) else None,
)

install_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestIdLoggingMiddleware)

app.include_router(api_router, prefix="/v1")
