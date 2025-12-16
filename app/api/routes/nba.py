from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter

router = APIRouter()

NBA_SCOREBOARD_URL = "https://data.nba.net/prod/v2/{date}/scoreboard.json"


def _ny_date_yyyymmdd() -> str:
    # NBA schedule/scoreboard aligns best with US Eastern date
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")


@router.get("/scoreboard/today")
async def scoreboard_today() -> dict:
    date = _ny_date_yyyymmdd()
    url = NBA_SCOREBOARD_URL.format(date=date)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers={"User-Agent": "nba-insights-api/1.0"})

        # If NBA hasn't posted a scoreboard for the date yet, return empty safely
        if r.status_code == 404:
            return {"date": date, "games": [], "upstream_status": 404}

        r.raise_for_status()
        data = r.json()

    except httpx.RequestError as e:
        # Network/DNS/timeouts etc.
        return {"date": date, "games": [], "upstream_error": str(e), "upstream_status": 502}
    except httpx.HTTPStatusError as e:
        # Any other HTTP error
        return {"date": date, "games": [], "upstream_error": str(e), "upstream_status": e.response.status_code}

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
                "home": {"triCode": h.get("triCode"), "score": h.get("score")},
                "away": {"triCode": v.get("triCode"), "score": v.get("score")},
            }
        )

    return {"date": date, "games": simplified}
