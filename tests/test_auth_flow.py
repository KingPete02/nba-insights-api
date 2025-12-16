from __future__ import annotations

import uuid

import pytest
import httpx

from app.main import app


@pytest.mark.asyncio
async def test_signup_login_me_flow() -> None:
    email = f"test_{uuid.uuid4()}@example.com"
    password = "SuperSecure123"

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/v1/auth/signup", json={"email": email, "password": password})
        assert r.status_code == 201, r.text

        r = await client.post(
            "/v1/auth/login",
            data={"username": email, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code == 200, r.text
        token = r.json()["access_token"]

        r = await client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        assert r.json()["email"] == email
