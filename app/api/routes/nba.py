from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import APIRouter

router = APIRouter()

NBA_TODAY_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

# ----------------------------
# Baseline model assumptions
# ----------------------------
# League-average points per team per 48 minutes (very rough baseline)
LEAGUE_PPG_TEAM = 114.0

# League-average total points per game (rough)
LEAGUE_TOTAL = LEAGUE_PPG_TEAM * 2

# Typical game-to-game total stdev; used to calibrate remaining-time volatility
TOTAL_GAME_STD = 22.0  # points

# Home-court advantage in points (rough baseline)
HOME_COURT_PTS = 2.0

# Number of Monte Carlo sims per game (keep light for production)
SIMS = 4000

# ----------------------------
# Helpers
# ----------------------------

@dataclass(frozen=True)
class GameState:
    game_id: str
    home_tri: str
    away_tri: str
    home_score: float
    away_score: float
    is_live: bool
    seconds_remaining: float | None


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _parse_gameclock_iso8601_duration(game_clock: str | None) -> float | None:
    """
    NBA liveData often gives gameClock like 'PT11M32.00S' (ISO8601 duration).
    Returns remaining seconds in current period.
    """
    if not game_clock or not isinstance(game_clock, str):
        return None
    m = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", game_clock.strip())
    if not m:
        return None
    minutes = float(m.group(1) or 0)
    seconds = float(m.group(2) or 0)
    return minutes * 60.0 + seconds


def _estimate_seconds_remaining(raw_game: dict[str, Any]) -> float | None:
    """
    Best-effort estimate of total seconds remaining in game.
    Uses:
      - period + gameClock for live games
      - assumes 12-min quarters + 5-min OT
    """
    period = raw_game.get("period", {}) or {}
    current = period.get("current")
    # If no period info, can't estimate
    if not isinstance(current, int):
        return None

    game_clock = raw_game.get("gameClock")
    sec_in_period = _parse_gameclock_iso8601_duration(game_clock)

    # NBA regulation: 4 quarters * 12 minutes
    if current <= 4:
        completed_periods = max(current - 1, 0)
        remaining_full_periods = max(4 - current, 0)
        # seconds remaining = remaining in current period + remaining full quarters
        if sec_in_period is None:
            return None
        return sec_in_period + remaining_full_periods * 12 * 60

    # Overtime(s): 5 minutes each
    # current=5 means 1st OT, etc.
    ot_number = current - 4
    if sec_in_period is None:
        return None
    # assume only current OT remains + (no future OTs known)
    # (We cannot know future OT; model will naturally allow OT via variance later.)
    return sec_in_period


async def _fetch_today_raw_games() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(NBA_TODAY_SCOREBOARD_URL, headers={"User-Agent": "nba-insights-api/1.0"})
    r.raise_for_status()
    data = r.json()
    scoreboard = data.get("scoreboard", {}) or {}
    return scoreboard.get("games", []) or []


def _to_game_state(raw: dict[str, Any]) -> GameState:
    home = raw.get("homeTeam", {}) or {}
    away = raw.get("awayTeam", {}) or {}
    game_status = raw.get("gameStatus")  # 1=scheduled, 2=live, 3=final (typically)

    home_score = float(home.get("score") or 0)
    away_score = float(away.get("score") or 0)

    is_live = game_status == 2
    seconds_remaining = _estimate_seconds_remaining(raw) if is_live else None

    return GameState(
        game_id=str(raw.get("gameId")),
        home_tri=str(home.get("teamTricode") or ""),
        away_tri=str(away.get("teamTricode") or ""),
        home_score=home_score,
        away_score=away_score,
        is_live=is_live,
        seconds_remaining=seconds_remaining,
    )


# ----------------------------
# Existing endpoints
# ----------------------------

@router.get("/scoreboard/today")
async def scoreboard_today() -> dict:
    try:
        games = await _fetch_today_raw_games()
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
                "home": {"teamTricode": home.get("teamTricode"), "score": home.get("score")},
                "away": {"teamTricode": away.get("teamTricode"), "score": away.get("score")},
            }
        )

    return {"games": simplified}


@router.get("/projections/today")
async def projections_today() -> dict:
    """
    Placeholder win% (kept for now so frontend doesn't break).
    We'll replace this later with the fairline engine output.
    """
    try:
        games = await _fetch_today_raw_games()
    except httpx.RequestError as e:
        return {"items": [], "upstream_error": str(e), "upstream_status": 502}
    except httpx.HTTPStatusError as e:
        return {"items": [], "upstream_error": str(e), "upstream_status": e.response.status_code}

    items = []
    for g in games:
        game_id = str(g.get("gameId"))
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


@router.get("/possession/today")
async def possession_today() -> dict:
    # Keep your existing possession endpoint behavior (no change here)
    return {"items": []}


# ----------------------------
# NEW: Fair line engine
# ----------------------------

def _simulate_fairline(
    state: GameState,
) -> dict[str, Any]:
    """
    Baseline live model:
    - Uses current score + time remaining (if live)
    - Uses league-average scoring as prior
    - Adds home court as a point shift
    - Produces distributions for final total and margin via Monte Carlo
    """

    # If not live, we still provide pregame-ish fair numbers using league priors
    if state.seconds_remaining is None:
        # Treat as full game remaining
        seconds_remaining = 48 * 60
        base_total_remaining_mean = LEAGUE_TOTAL
        total_remaining_std = TOTAL_GAME_STD
    else:
        seconds_remaining = max(0.0, float(state.seconds_remaining))
        frac = seconds_remaining / (48 * 60)

        # Expected remaining points scales with remaining time
        base_total_remaining_mean = LEAGUE_TOTAL * frac

        # Volatility scales with sqrt(time fraction)
        total_remaining_std = TOTAL_GAME_STD * math.sqrt(max(frac, 1e-6))

    # Split remaining points between teams based on current margin + home court
    # We convert "home court points" into a share shift.
    # Share shift ~ HOME_COURT_PTS / expected_total_remaining (small)
    denom = max(base_total_remaining_mean, 1.0)
    share_shift = (HOME_COURT_PTS / denom) * 0.5  # keep small

    # Baseline: 50/50 split of remaining points
    home_share = min(max(0.5 + share_shift, 0.40), 0.60)
    away_share = 1.0 - home_share

    # Simulate remaining total points as Normal
    # Then allocate to teams via shares + small noise
    total_means = []
    margin_means = []
    home_wins = 0

    for _ in range(SIMS):
        rem_total = random.gauss(base_total_remaining_mean, total_remaining_std)
        rem_total = max(0.0, rem_total)

        # Allocate remaining points
        # Add small allocation noise so margin has variance
        alloc_noise = random.gauss(0.0, 0.03)  # 3% share noise
        h_share = min(max(home_share + alloc_noise, 0.35), 0.65)
        a_share = 1.0 - h_share

        home_final = state.home_score + rem_total * h_share
        away_final = state.away_score + rem_total * a_share

        total = home_final + away_final
        margin = home_final - away_final

        total_means.append(total)
        margin_means.append(margin)

        if margin > 0:
            home_wins += 1

    # Compute stats
    mean_total = sum(total_means) / len(total_means)
    mean_margin = sum(margin_means) / len(margin_means)

    def _std(xs: list[float]) -> float:
        mu = sum(xs) / len(xs)
        return math.sqrt(sum((x - mu) ** 2 for x in xs) / max(len(xs) - 1, 1))

    std_total = _std(total_means)
    std_margin = _std(margin_means)

    home_win_prob = home_wins / SIMS
    away_win_prob = 1.0 - home_win_prob

    # Define "fair lines" as the mean of distributions
    # Spread convention: negative means home favored by that many
    fair_spread_home = -mean_margin
    fair_total = mean_total

    return {
        "gameId": state.game_id,
        "home": state.home_tri,
        "away": state.away_tri,
        "isLive": state.is_live,
        "secondsRemaining": state.seconds_remaining,
        "fair": {
            "spread_home": fair_spread_home,
            "total": fair_total,
        },
        "prob": {
            "home_win": home_win_prob,
            "away_win": away_win_prob,
        },
        "dist": {
            "total_mean": mean_total,
            "total_std": std_total,
            "margin_mean": mean_margin,
            "margin_std": std_margin,
        },
    }


@router.get("/fairline/today")
async def fairline_today() -> dict:
    """
    Returns fair spread/total and win probabilities for today's games.
    This is our Phase-1 baseline model for live spread/total.
    """
    try:
        raw_games = await _fetch_today_raw_games()
    except httpx.RequestError as e:
        return {"items": [], "upstream_error": str(e), "upstream_status": 502}
    except httpx.HTTPStatusError as e:
        return {"items": [], "upstream_error": str(e), "upstream_status": e.response.status_code}

    items = []
    for rg in raw_games:
        st = _to_game_state(rg)
        # Skip games missing team codes
        if not st.home_tri or not st.away_tri:
            continue
        items.append(_simulate_fairline(st))

    return {"items": items}
