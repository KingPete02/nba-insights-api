from __future__ import annotations

import math
import httpx
from fastapi import APIRouter

router = APIRouter()

NBA_TODAY_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NBA_PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"


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
                "home": {"teamId": home.get("teamId"), "teamTricode": home.get("teamTricode"), "score": home.get("score")},
                "away": {"teamId": away.get("teamId"), "teamTricode": away.get("teamTricode"), "score": away.get("score")},
            }
        )

    return {"games": simplified}


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@router.get("/projections/today")
async def projections_today() -> dict:
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

        home_adv = 0.20
        margin = max(min(home_score - away_score, 15.0), -15.0)
        margin_adj = 0.02 * margin

        home_prob = _logistic(home_adv + margin_adj)
        away_prob = 1.0 - home_prob

        items.append({"gameId": game_id, "homeWinProb": home_prob, "awayWinProb": away_prob})

    return {"items": items}


async def _infer_possession_for_game(game_id: str) -> dict:
    """
    Best-effort possession inference from latest play-by-play action.
    If we can't infer, return possessionTeamId=None.
    """
    url = NBA_PBP_URL.format(game_id=game_id)
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers={"User-Agent": "nba-insights-api/1.0"})
    if r.status_code == 404:
        return {"gameId": game_id, "possessionTeamId": None}
    r.raise_for_status()
    data = r.json()

    game = data.get("game", {}) or {}
    actions = game.get("actions", []) or []

    # Walk backwards to find any action that declares possession.
    # Many actions include a "possession" field (teamId) or equivalent.
    for a in reversed(actions[-50:]):  # last ~50 events
        poss = a.get("possession", None)
        if isinstance(poss, int) and poss != 0:
            return {"gameId": game_id, "possessionTeamId": poss}

        # Fallback: some events encode teamId; use as weak signal
        team_id = a.get("teamId", None)
        if isinstance(team_id, int) and team_id != 0:
            # Not perfect, but better than nothing
            return {"gameId": game_id, "possessionTeamId": team_id}

    return {"gameId": game_id, "possessionTeamId": None}


@router.get("/possession/today")
async def possession_today() -> dict:
    """
    Returns possession info for games in progress.
    Output items: { gameId, possession: "HOME"|"AWAY"|None, possessionTricode: str|None }
    """
    try:
        games = await _fetch_today_games()
    except Exception:
        return {"items": []}

    items = []
    for g in games:
        # gameStatus 2 = in progress (usually)
        if g.get("gameStatus") != 2:
            continue

        game_id = g.get("gameId")
        home = g.get("homeTeam", {}) or {}
        away = g.get("awayTeam", {}) or {}
        home_id = home.get("teamId")
        away_id = away.get("teamId")

        try:
            poss = await _infer_possession_for_game(str(game_id))
        except Exception:
            items.append({"gameId": game_id, "possession": None, "possessionTricode": None})
            continue

        team_id = poss.get("possessionTeamId")
        if team_id == home_id:
            items.append({"gameId": game_id, "possession": "HOME", "possessionTricode": home.get("teamTricode")})
        elif team_id == away_id:
            items.append({"gameId": game_id, "possession": "AWAY", "possessionTricode": away.get("teamTricode")})
        else:
            items.append({"gameId": game_id, "possession": None, "possessionTricode": None})

    return {"items": items}
