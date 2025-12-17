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

NBA_TODAY_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NBA_TODAY_ODDS_URL = "https://cdn.nba.com/static/json/liveData/odds/odds_todaysGames.json"

THE_ODDS_API_SPORT = "basketball_nba"
THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

LEAGUE_PPG_TEAM = 114.0
LEAGUE_TOTAL = LEAGUE_PPG_TEAM * 2
TOTAL_GAME_STD = 22.0
HOME_COURT_PTS = 2.0
SIMS = 2500

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

Market = Literal["spreads", "totals"]


@dataclass(frozen=True)
class GameState:
    game_id: str
    home_tri: str
    away_tri: str
    home_score: float
    away_score: float
    is_live: bool
    seconds_remaining: float | None


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _parse_gameclock_iso8601_duration(game_clock: str | None) -> float | None:
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
    if sec_in_period is None:
        return None

    if current <= 4:
        remaining_full_periods = max(4 - current, 0)
        return sec_in_period + remaining_full_periods * 12 * 60

    # overtime current period only
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


# -----------------------------
# SCOREBOARD (RESTORED)
# -----------------------------
@router.get("/scoreboard/today")
async def scoreboard_today() -> dict:
    try:
        games = await _fetch_today_scoreboard_raw()
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
                "gameId": str(g.get("gameId")),
                "gameStatus": g.get("gameStatus"),
                "gameStatusText": g.get("gameStatusText"),
                "gameTimeUTC": g.get("gameTimeUTC"),
                "home": {
                    "teamTricode": home.get("teamTricode"),
                    "score": home.get("score"),
                },
                "away": {
                    "teamTricode": away.get("teamTricode"),
                    "score": away.get("score"),
                },
            }
        )
    return {"games": simplified}


# -----------------------------
# FAIR LINE
# -----------------------------
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

    fair_spread_home = -mean_margin

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


# -----------------------------
# PROJECTIONS (kept for UI compatibility)
# -----------------------------
@router.get("/projections/today")
async def projections_today() -> dict:
    raw = await _fetch_today_scoreboard_raw()
    items = []
    for rg in raw:
        st = _to_game_state(rg)
        if not st.home_tri:
            continue
        # fallback from fairline prob
        fl = _simulate_fairline(st)
        items.append({"gameId": st.game_id, "homeWinProb": fl["prob"]["home_win"], "awayWinProb": fl["prob"]["away_win"]})
    return {"items": items}


# -----------------------------
# POSSESSION (safe empty for now)
# -----------------------------
@router.get("/possession/today")
async def possession_today() -> dict:
    return {"items": []}


# -----------------------------
# ODDS ingestion
# -----------------------------
def _decimal_from_str(x: Any) -> float | None:
    try:
        v = float(x)
        if v <= 1.0:
            return None
        return v
    except Exception:
        return None


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

    params = {"regions": "us", "markets": "spreads,totals", "oddsFormat": "decimal", "apiKey": api_key}
    url = THE_ODDS_API_URL.format(sport=THE_ODDS_API_SPORT)
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
    r.raise_for_status()
    return r.json()


def _normalize_nba_odds(nba_games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []
    for g in nba_games:
        game_id = str(g.get("gameId"))
        markets = g.get("markets", []) or []
        for m in markets:
            name = (m.get("name") or "").lower()
            books = m.get("books", []) or []

            if name == "spread":
                for b in books:
                    book = b.get("name") or "Unknown"
                    for o in b.get("outcomes", []) or []:
                        side = (o.get("type") or "").lower()
                        odds = _decimal_from_str(o.get("odds"))
                        try:
                            line = float(o.get("spread"))
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
                                "line": line,
                                "odds_decimal": odds,
                            }
                        )

            if name in {"total", "totals", "overunder", "over_under", "ou"}:
                for b in books:
                    book = b.get("name") or "Unknown"
                    for o in b.get("outcomes", []) or []:
                        side = (o.get("type") or "").lower()
                        odds = _decimal_from_str(o.get("odds"))
                        try:
                            line = float(o.get("total") or o.get("points") or o.get("line"))
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


def _normalize_theoddsapi(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []
    for ev in events:
        home_name = ev.get("home_team")
        away_name = ev.get("away_team")
        home_tri = TEAM_NAME_TO_TRICODE.get(str(home_name), "")
        away_tri = TEAM_NAME_TO_TRICODE.get(str(away_name), "")
        if not home_tri or not away_tri:
            continue

        for bk in ev.get("bookmakers", []) or []:
            book = bk.get("title") or bk.get("key") or "Unknown"
            for m in bk.get("markets", []) or []:
                mk = (m.get("key") or "").lower()
                outs = m.get("outcomes", []) or []

                if mk == "spreads":
                    for o in outs:
                        team_name = o.get("name")
                        odds = _decimal_from_str(o.get("price"))
                        try:
                            line = float(o.get("point"))
                        except Exception:
                            continue
                        if odds is None:
                            continue
                        side = "home" if str(team_name) == str(home_name) else "away" if str(team_name) == str(away_name) else None
                        if not side:
                            continue
                        offers.append(
                            {
                                "source": "the_odds_api",
                                "book": book,
                                "market": "spreads",
                                "gameHome": home_tri,
                                "gameAway": away_tri,
                                "side": side,
                                "line": line,
                                "odds_decimal": odds,
                            }
                        )

                if mk == "totals":
                    for o in outs:
                        name = (o.get("name") or "").lower()
                        if name not in {"over", "under"}:
                            continue
                        odds = _decimal_from_str(o.get("price"))
                        try:
                            line = float(o.get("point"))
                        except Exception:
                            continue
                        if odds is None:
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
    nba_games = await _fetch_nba_odds()
    offers = _normalize_nba_odds(nba_games)

    try:
        o = await _fetch_theoddsapi_odds()
        if o:
            offers.extend(_normalize_theoddsapi(o))
    except Exception:
        pass

    return {"offers": offers}


# -----------------------------
# EDGE computation
# -----------------------------
def _prob_cover_spread(margin_mean: float, margin_std: float, home_spread: float) -> float:
    if margin_std <= 1e-9:
        return 1.0 if (margin_mean + home_spread) > 0 else 0.0
    threshold = -home_spread
    z = (threshold - margin_mean) / margin_std
    return 1.0 - _normal_cdf(z)


def _prob_over_total(total_mean: float, total_std: float, line: float) -> float:
    if total_std <= 1e-9:
        return 1.0 if total_mean > line else 0.0
    z = (line - total_mean) / total_std
    return 1.0 - _normal_cdf(z)


def _ev_decimal(p: float, odds_decimal: float) -> float:
    return p * odds_decimal - 1.0


def fmt_spread(x: float) -> str:
    v = round(float(x) * 10) / 10
    return f"{v:+g}"



@router.get("/edges/today")
async def edges_today(
    market: Literal["spreads", "totals"] = Query(default="spreads"),
    min_ev: float = Query(default=0.02, ge=-1.0, le=10.0),
    max_results: int = Query(default=50, ge=1, le=500),
) -> dict:
    """
    Positive-EV feed: edges vs multi-book odds (spread/total).
    Returns UI-ready rows: game info + fair line + book line + implied/model probs + EV.
    """
    # Scoreboard map for game metadata (time/status/teams)
    try:
        sb_raw = await _fetch_today_scoreboard_raw()
    except Exception:
        sb_raw = []

    sb_by_game: dict[str, dict[str, Any]] = {}
    for g in sb_raw:
        home = (g.get("homeTeam") or {}) if isinstance(g.get("homeTeam"), dict) else {}
        away = (g.get("awayTeam") or {}) if isinstance(g.get("awayTeam"), dict) else {}
        gid = str(g.get("gameId"))
        sb_by_game[gid] = {
            "gameId": gid,
            "gameStatus": g.get("gameStatus"),
            "gameStatusText": g.get("gameStatusText"),
            "gameTimeUTC": g.get("gameTimeUTC"),
            "home": {"teamTricode": home.get("teamTricode"), "score": home.get("score")},
            "away": {"teamTricode": away.get("teamTricode"), "score": away.get("score")},
        }

    fair = (await fairline_today()).get("items", []) or []
    fair_by_game: dict[str, dict[str, Any]] = {it["gameId"]: it for it in fair}

    odds_payload = await odds_today()
    odds = (odds_payload.get("offers") or [])
    fetched_at = odds_payload.get("fetched_at")

    def implied_prob(odds_decimal: float) -> float:
        return 1.0 / odds_decimal

    edges: list[dict[str, Any]] = []

    for offer in odds:
        if offer.get("market") != market:
            continue

        odds_dec = float(offer.get("odds_decimal") or 0)
        if odds_dec <= 1.0:
            continue

        game_id = offer.get("gameId")
        fair_item = None

        if game_id and game_id in fair_by_game:
            fair_item = fair_by_game[game_id]
        else:
            # Match by tricodes if present (The Odds API)
            home = offer.get("gameHome")
            away = offer.get("gameAway")
            if home and away:
                for it in fair:
                    if it.get("home") == home and it.get("away") == away:
                        fair_item = it
                        game_id = it.get("gameId")
                        break

        if not fair_item or not game_id:
            continue

        dist = fair_item["dist"]
        margin_mean = float(dist["margin_mean"])
        margin_std = float(dist["margin_std"])
        total_mean = float(dist["total_mean"])
        total_std = float(dist["total_std"])

        side = offer.get("side")
        line = float(offer.get("line"))

        if market == "spreads":
            # Home spread normalization
            if side == "home":
                home_spread = line
                p = _prob_cover_spread(margin_mean, margin_std, home_spread)
                pick = f"{fair_item['home']} {fmt_spread(home_spread)}"
                fair_line = float(fair_item["fair"]["spread_home"])
            else:
                home_spread = -line
                p_home = _prob_cover_spread(margin_mean, margin_std, home_spread)
                p = 1.0 - p_home
                pick = f"{fair_item['away']} {fmt_spread(line)}"
                fair_line = float(fair_item["fair"]["spread_home"])
        else:
            if side == "over":
                p = _prob_over_total(total_mean, total_std, line)
                pick = f"OVER {line}"
                fair_line = float(fair_item["fair"]["total"])
            else:
                p_over = _prob_over_total(total_mean, total_std, line)
                p = 1.0 - p_over
                pick = f"UNDER {line}"
                fair_line = float(fair_item["fair"]["total"])

        ev = _ev_decimal(p, odds_dec)
        if ev < min_ev:
            continue

        sb = sb_by_game.get(game_id, {})
        edges.append(
            {
                "gameId": game_id,
                "home": fair_item.get("home"),
                "away": fair_item.get("away"),
                "gameTimeUTC": sb.get("gameTimeUTC"),
                "gameStatus": sb.get("gameStatus"),
                "gameStatusText": sb.get("gameStatusText"),
                "book": offer.get("book"),
                "source": offer.get("source"),
                "market": market,
                "pick": pick,
                "book_line": line,
                "fair_line": round(fair_line, 3),
                "odds_decimal": odds_dec,
                "implied_prob": round(implied_prob(odds_dec), 4),
                "model_prob": round(float(p), 4),
                "ev": round(float(ev), 4),
                "fetched_at": fetched_at,
            }
        )

    edges.sort(key=lambda x: x["ev"], reverse=True)
    return {"items": edges[:max_results]}
