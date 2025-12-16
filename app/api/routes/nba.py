from __future__ import annotations

import math
import httpx
from fastapi import APIRouter

router = APIRouter()

NBA_TODAY_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"


async def _fetch_today_games() -> list[dict]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(NBA_TODAY_SCOREBOARD_URL, headers={"User-Agent": "nba-insights-api/1.0"})
    r.raise_for_status()
    data = r.json()
    scoreboard = data.get("scoreboard", {}) or {}
    return scoreboard.get("games", []) or []


@router.get("/scoreboard/today")
async def scoreboard_today() -> dict:
    try:
        games = await _fetch_today_games()
    except httpx.RequestError as e:
        return {"games": [], "upstream_error": str(e), "upstream_status": 502}
    except httpx.HTTPStatusError as e:
        return {"games": [], "upstream_error": str(e), "upstream_status": e.response.status_code}

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


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@router.get("/projections/today")
async def projections_today() -> dict:
    """
    Simple baseline win probabilities for today's games.
    Not a true model yet—just a stable placeholder we can improve.
    """
    try:
        games = await _fetch_today_games()
    except httpx.RequestError as e:
        return {"items": [], "upstream_error": str(e), "upstream_status": 502}
    except httpx.HTTPStatusError as e:
        return {"items": [], "upstream_error": str(e), "upstream_status": e.response.status_code}

    items = []
    for g in games:
        game_id = g.get("gameId")
        home = g.get("homeTeam", {}) or {}
        away = g.get("awayTeam", {}) or {}

        home_score = float(home.get("score") or 0)
        away_score = float(away.get("score") or 0)

        # Very simple components:
        # - home court advantage: +0.20 logits
        # - live score margin adjustment: +/-0.02 logits per point (capped)
        home_adv = 0.20
        margin = max(min(home_score - away_score, 15.0), -15.0)
        margin_adj = 0.02 * margin

        # Combine into a probability via logistic
        home_prob = _logistic(home_adv + margin_adj)
        away_prob = 1.0 - home_prob

        items.append(
            {
                "gameId": game_id,
                "homeWinProb": home_prob,
                "awayWinProb": away_prob,
            }
        )

    return {"items": items}
