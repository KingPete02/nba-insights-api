from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Query

router = APIRouter()

# Live NBA scoreboard + odds feeds
NBA_TODAY_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NBA_TODAY_ODDS_URL = "https://cdn.nba.com/static/json/liveData/odds/odds_todaysGames.json"

# Optional: The Odds API (multi-bookmaker aggregator)
# Docs: sport key basketball_nba, markets=spreads,totals, oddsFormat=decimal  [oai_citation:2‡The Odds API](https://the-odds-api.com/sports-odds-data/nba-odds.html?utm_source=chatgpt.com)
THE_ODDS_API_SPORT = "basketball_nba"
THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

# ----------------------------
# Baseline model parameters
# ----------------------------
LEAGUE_PPG_TEAM = 114.0
LEAGUE_TOTAL = LEAGUE_PPG_TEAM * 2
TOTAL_GAME_STD = 22.0
HOME_COURT_PTS = 2.0
SIMS = 3000  # keep light for production

# ----------------------------
# Team name mapping for The Odds API (home/away names -> tricodes)
# ----------------------------
TEAM_NAME_TO_TRICODE = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

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


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _normal_cdf(z: float) -> float:
    # Standard normal CDF
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _parse_gameclock_iso8601_duration(game_clock: str | None) -> float | None:
    # e.g. "PT11M32.00S"
    if not game_clock or not isinstance(game_clock, str):
        return None
    m = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", game_clock.strip())
    if not m:
        return None
    minutes = float(m.group(1) or 0)
    seconds = float(m.group(2) or 0)
    return minutes * 60.0 + seconds


def _estimate_seconds_remaining(raw_game: dict[str, Any]) -> float | None:
    period = raw_game.get("period", {}) or {}
    current = period.get("current")
    if not isinstance(current, int):
        return None

    sec_in_period = _parse_gameclock_iso8601_duration(raw_game.get("gameClock"))

    if current <= 4:
        if sec_in_period is None:
            return None
        remaining_full_periods = max(4 - current, 0)
        return sec_in_period + remaining_full_periods * 12 * 60

    # OT: 5 minutes each; we only know remaining in current OT
    if sec_in_period is None:
        return None
    return sec_in_period


async def _fetch_today_scoreboard_raw() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(NBA_TODAY_SCOREBOARD_URL, headers={"User-Agent": "nba-insights-api/1.0"})
    r.raise_for_status()
    data = r.json()
    scoreboard = data.get("scoreboard", {}) or {}
    return scoreboard.get("games", []) or []


def _to_game_state(raw: dict[str, Any]) -> GameState:
    home = raw.get("homeTeam", {}) or {}
    away = raw.get("awayTeam", {}) or {}
    status = raw.get("gameStatus")  # 1 scheduled, 2 live, 3 final

    home_score = float(home.get("score") or 0)
    away_score = float(away.get("score") or 0)

    is_live = status == 2
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


def _simulate_fairline(state: GameState) -> dict[str, Any]:
    if state.seconds_remaining is None:
        seconds_remaining = 48 * 60
        base_total_remaining_mean = LEAGUE_TOTAL
        total_remaining_std = TOTAL_GAME_STD
    else:
        seconds_remaining = max(0.0, float(state.seconds_remaining))
        frac = seconds_remaining / (48 * 60)
        base_total_remaining_mean = LEAGUE_TOTAL * frac
        total_remaining_std = TOTAL_GAME_STD * math.sqrt(max(frac, 1e-6))

    denom = max(base_total_remaining_mean, 1.0)
    share_shift = (HOME_COURT_PTS / denom) * 0.5
    home_share = min(max(0.5 + share_shift, 0.40), 0.60)

    totals: list[float] = []
    margins: list[float] = []
    home_wins = 0

    for _ in range(SIMS):
        rem_total = random.gauss(base_total_remaining_mean, total_remaining_std)
        rem_total = max(0.0, rem_total)

        alloc_noise = random.gauss(0.0, 0.03)
        h_share = min(max(home_share + alloc_noise, 0.35), 0.65)
        a_share = 1.0 - h_share

        home_final = state.home_score + rem_total * h_share
        away_final = state.away_score + rem_total * a_share

        total = home_final + away_final
        margin = home_final - away_final

        totals.append(total)
        margins.append(margin)
        if margin > 0:
            home_wins += 1

    mean_total = sum(totals) / len(totals)
    mean_margin = sum(margins) / len(margins)
    std_total = _std(totals)
    std_margin = _std(margins)

    home_win_prob = home_wins / SIMS
    away_win_prob = 1.0 - home_win_prob

    fair_spread_home = -mean_margin  # negative => home favored

    return {
        "gameId": state.game_id,
        "home": state.home_tri,
        "away": state.away_tri,
        "isLive": state.is_live,
        "secondsRemaining": state.seconds_remaining,
        "fair": {"spread_home": fair_spread_home, "total": mean_total},
        "prob": {"home_win": home_win_prob, "away_win": away_win_prob},
        "dist": {"total_mean": mean_total, "total_std": std_total, "margin_mean": mean_margin, "margin_std": std_margin},
    }


@router.get("/fairline/today")
async def fairline_today() -> dict:
    raw = await _fetch_today_scoreboard_raw()
    items = []
    for rg in raw:
        st = _to_game_state(rg)
        if st.home_tri and st.away_tri:
            items.append(_simulate_fairline(st))
    return {"items": items}


# ----------------------------
# Odds normalization
# ----------------------------

Market = Literal["spreads", "totals"]


def _decimal_from_str(x: Any) -> float | None:
    try:
        v = float(x)
        if v <= 1.0:
            return None
        return v
    except Exception:
        return None


def _implied_prob_from_decimal(odds: float) -> float:
    return 1.0 / odds


async def _fetch_nba_odds() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(NBA_TODAY_ODDS_URL, headers={"User-Agent": "nba-insights-api/1.0"})
    r.raise_for_status()
    data = r.json()
    return data.get("games", []) or []


async def _fetch_theoddsapi_odds() -> list[dict[str, Any]] | None:
    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        return None

    params = {
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "decimal",
        "apiKey": api_key,
    }
    url = THE_ODDS_API_URL.format(sport=THE_ODDS_API_SPORT)

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
    r.raise_for_status()
    return r.json()


def _normalize_nba_odds(nba_games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []

    for g in nba_games:
        game_id = str(g.get("gameId"))
        home_id = str(g.get("homeTeamId"))
        away_id = str(g.get("awayTeamId"))
        markets = g.get("markets", []) or []

        for m in markets:
            name = (m.get("name") or "").lower()
            books = m.get("books", []) or []

            if name == "spread":
                for b in books:
                    book = b.get("name") or "Unknown"
                    outcomes = b.get("outcomes", []) or []
                    # outcomes: home/away with spread and odds
                    for o in outcomes:
                        side = (o.get("type") or "").lower()  # home/away
                        odds = _decimal_from_str(o.get("odds"))
                        spread = o.get("spread")
                        try:
                            line = float(spread)
                        except Exception:
                            continue
                        if odds is None:
                            continue

                        offers.append(
                            {
                                "source": "nba_odds",
                                "book": book,
                                "market": "spreads",
                                "gameId": game_id,
                                "side": "home" if side == "home" else "away",
                                "line": line,  # home line usually negative
                                "odds_decimal": odds,
                                "homeTeamId": home_id,
                                "awayTeamId": away_id,
                            }
                        )

            # Totals market name varies; try a few common keys
            if name in {"total", "totals", "overunder", "over_under", "ou"}:
                for b in books:
                    book = b.get("name") or "Unknown"
                    outcomes = b.get("outcomes", []) or []
                    for o in outcomes:
                        side = (o.get("type") or "").lower()  # over/under
                        odds = _decimal_from_str(o.get("odds"))
                        total = o.get("total") or o.get("points") or o.get("line")
                        try:
                            line = float(total)
                        except Exception:
                            continue
                        if odds is None:
                            continue
                        offers.append(
                            {
                                "source": "nba_odds",
                                "book": book,
                                "market": "totals",
                                "gameId": game_id,
                                "side": "over" if side == "over" else "under",
                                "line": line,
                                "odds_decimal": odds,
                            }
                        )

    return offers


def _normalize_theoddsapi(odds_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []

    for ev in odds_events:
        home_name = ev.get("home_team")
        away_name = ev.get("away_team")
        home_tri = TEAM_NAME_TO_TRICODE.get(str(home_name), "")
        away_tri = TEAM_NAME_TO_TRICODE.get(str(away_name), "")
        if not home_tri or not away_tri:
            continue

        bookmakers = ev.get("bookmakers", []) or []
        for bk in bookmakers:
            book = bk.get("title") or bk.get("key") or "Unknown"
            markets = bk.get("markets", []) or []
            for m in markets:
                mk = (m.get("key") or "").lower()
                outcomes = m.get("outcomes", []) or []

                if mk == "spreads":
                    # outcomes have name and point + price
                    for o in outcomes:
                        team_name = o.get("name")
                        point = o.get("point")
                        price = o.get("price")
                        odds = _decimal_from_str(price)
                        try:
                            line = float(point)
                        except Exception:
                            continue
                        if odds is None:
                            continue

                        side = None
                        if str(team_name) == str(home_name):
                            side = "home"
                        elif str(team_name) == str(away_name):
                            side = "away"
                        else:
                            continue

                        offers.append(
                            {
                                "source": "the_odds_api",
                                "book": book,
                                "market": "spreads",
                                "gameHome": home_tri,
                                "gameAway": away_tri,
                                "side": side,
                                "line": line,  # line for the listed team
                                "odds_decimal": odds,
                            }
                        )

                if mk == "totals":
                    for o in outcomes:
                        name = (o.get("name") or "").lower()  # over/under
                        point = o.get("point")
                        price = o.get("price")
                        odds = _decimal_from_str(price)
                        try:
                            line = float(point)
                        except Exception:
                            continue
                        if odds is None:
                            continue
                        if name not in {"over", "under"}:
                            continue
                        offers.append(
                            {
                                "source": "the_odds_api",
                                "book": book,
                                "market": "totals",
                                "gameHome": home_tri,
                                "gameAway": away_tri,
                                "side": name,
                                "line": line,
                                "odds_decimal": odds,
                            }
                        )

    return offers


@router.get("/odds/today")
async def odds_today() -> dict:
    # Primary: NBA odds feed (multi-book)
    nba_games = await _fetch_nba_odds()
    offers = _normalize_nba_odds(nba_games)

    # Optional: The Odds API (if key set)
    try:
        o = await _fetch_theoddsapi_odds()
        if o:
            offers.extend(_normalize_theoddsapi(o))
    except Exception:
        # don't fail the endpoint if optional feed fails
        pass

    return {"offers": offers}


# ----------------------------
# Edge computation
# ----------------------------

def _prob_cover_spread(margin_mean: float, margin_std: float, home_spread: float) -> float:
    """
    home_spread: sportsbook line for HOME (e.g., -2.5).
    Home covers if margin > abs(line) when line negative.
    Equivalent: margin + home_spread > 0
    """
    if margin_std <= 1e-9:
        return 1.0 if (margin_mean + home_spread) > 0 else 0.0

    # Need P(margin > -home_spread)
    threshold = -home_spread
    z = (threshold - margin_mean) / margin_std
    return 1.0 - _normal_cdf(z)


def _prob_over_total(total_mean: float, total_std: float, line: float) -> float:
    if total_std <= 1e-9:
        return 1.0 if total_mean > line else 0.0
    z = (line - total_mean) / total_std
    return 1.0 - _normal_cdf(z)


def _ev_decimal(p: float, odds_decimal: float) -> float:
    # EV per $1 stake (profit expectation)
    return p * odds_decimal - 1.0


@router.get("/edges/today")
async def edges_today(
    market: Literal["spreads", "totals"] = Query(default="spreads"),
    min_ev: float = Query(default=0.02, ge=-1.0, le=10.0),
    max_results: int = Query(default=25, ge=1, le=200),
) -> dict:
    """
    Compute top positive-EV edges by comparing:
      sportsbook prices -> implied payout
      vs our model distribution -> probability

    Output: sorted by EV descending.
    """
    # 1) fairlines by gameId (tricodes)
    fair = (await fairline_today()).get("items", []) or []
    fair_by_game: dict[str, dict[str, Any]] = {it["gameId"]: it for it in fair}

    # 2) odds offers
    odds = (await odds_today()).get("offers", []) or []

    edges: list[dict[str, Any]] = []

    for offer in odds:
        if offer.get("market") != market:
            continue

        odds_dec = float(offer.get("odds_decimal") or 0)
        if odds_dec <= 1.0:
            continue

        game_id = offer.get("gameId")
        # The Odds API offers may not have gameId; match by tricodes if present
        fair_item = None
        if game_id and game_id in fair_by_game:
            fair_item = fair_by_game[game_id]
        else:
            home = offer.get("gameHome")
            away = offer.get("gameAway")
            if home and away:
                for it in fair:
                    if it.get("home") == home and it.get("away") == away:
                        fair_item = it
                        game_id = it.get("gameId")
                        break

        if not fair_item:
            continue

        dist = fair_item["dist"]
        margin_mean = float(dist["margin_mean"])
        margin_std = float(dist["margin_std"])
        total_mean = float(dist["total_mean"])
        total_std = float(dist["total_std"])

        side = offer.get("side")
        line = float(offer.get("line"))

        if market == "spreads":
            # Normalize to HOME spread for probability
            # If offer is for away with +2.5, the home spread is -2.5
            if side == "home":
                home_spread = line
                p = _prob_cover_spread(margin_mean, margin_std, home_spread)
                pick = f"{fair_item['home']} {fmt_spread(home_spread)}"
            else:
                # away spread: line is typically positive; home spread is -line
                home_spread = -line
                p_home_cover = _prob_cover_spread(margin_mean, margin_std, home_spread)
                p = 1.0 - p_home_cover
                pick = f"{fair_item['away']} {fmt_spread(line)}"

        else:  # totals
            if side == "over":
                p = _prob_over_total(total_mean, total_std, line)
                pick = f"OVER {line}"
            else:
                p_over = _prob_over_total(total_mean, total_std, line)
                p = 1.0 - p_over
                pick = f"UNDER {line}"

        ev = _ev_decimal(p, odds_dec)
        if ev < min_ev:
            continue

        edges.append(
            {
                "gameId": game_id,
                "book": offer.get("book"),
                "source": offer.get("source"),
                "market": market,
                "pick": pick,
                "line": line,
                "odds_decimal": odds_dec,
                "model_prob": round(p, 4),
                "ev": round(ev, 4),
            }
        )

    edges.sort(key=lambda x: x["ev"], reverse=True)
    return {"items": edges[:max_results]}


def fmt_spread(x: float) -> str:
    v = round(float(x) * 10) / 10
    return f"{v:+g}"
