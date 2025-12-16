from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter

router = APIRouter()

# Public NBA scoreboard JSON (date-based)
# Example format:
# https://data.nba.net/prod/v2/YYYYMMDD/scoreboard.json   [oai_citation:1‡nbasense.com](https://nbasense.com/nba-api/Data/Prod/Scores/Scoreboard?utm_source=chatgpt.com)
NBA_SCOREBOARD_URL = "https://data.nba.net/prod/v2/{date}/scoreboard.json"


@router.get("/scoreboard/today")
async def scoreboard_today() -> dict:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    url = NBA_SCOREBOARD_URL.format(date=today)

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers={"User-Agent": "nba-insights-api/1.0"})
        r.raise_for_status()
        data = r.json()

    games = data.get("games", []) or []
    simplified = []
    for g in games:
        h = g.get("hTeam", {}) or {}
        v = g.get("vTeam", {}) or {}
        simplified.append(
            {
                "gameId": g.get("gameId"),
                "startTimeUTC": g.get("startTimeUTC"),
                "statusNum": g.get("statusNum"),
                "period": g.get("period", {}),
                "home": {
                    "teamId": h.get("teamId"),
                    "triCode": h.get("triCode"),
                    "score": h.get("score"),
                },
                "away": {
                    "teamId": v.get("teamId"),
                    "triCode": v.get("triCode"),
                    "score": v.get("score"),
                },
            }
        )

    return {"date": today, "games": simplified}
