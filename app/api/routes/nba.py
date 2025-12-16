from __future__ import annotations

import httpx
from fastapi import APIRouter

router = APIRouter()

# NBA Live Data scoreboard (no date required)
# https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json  
NBA_TODAY_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"


@router.get("/scoreboard/today")
async def scoreboard_today() -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(NBA_TODAY_SCOREBOARD_URL, headers={"User-Agent": "nba-insights-api/1.0"})
        r.raise_for_status()
        data = r.json()
    except httpx.RequestError as e:
        return {"games": [], "upstream_error": str(e), "upstream_status": 502}
    except httpx.HTTPStatusError as e:
        return {"games": [], "upstream_error": str(e), "upstream_status": e.response.status_code}

    # This endpoint shape:
    # { "scoreboard": { "games": [ ... ] } }
    scoreboard = data.get("scoreboard", {}) or {}
    games = scoreboard.get("games", []) or []

    simplified = []
    for g in games:
        home = g.get("homeTeam", {}) or {}
        away = g.get("awayTeam", {}) or {}
        simplified.append(
            {
                "gameId": g.get("gameId"),
                "gameStatus": g.get("gameStatus"),
                "gameStatusText": g.get("gameStatusText"),
                "gameTimeUTC": g.get("gameTimeUTC"),
                "home": {
                    "teamId": home.get("teamId"),
                    "teamTricode": home.get("teamTricode"),
                    "score": home.get("score"),
                },
                "away": {
                    "teamId": away.get("teamId"),
                    "teamTricode": away.get("teamTricode"),
                    "score": away.get("score"),
                },
            }
        )

    return {"games": simplified}
