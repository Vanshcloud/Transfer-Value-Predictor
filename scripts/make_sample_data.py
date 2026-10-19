"""Write data/sample/: a synthetic football world in the Kaggle mirror's schema.

The unit suite runs against data/sample/ so CI needs no download. It used to
hold rows copied from the Kaggle mirror, whose rights are uncertain
(docs/DATASET_CARD.md, "Data rights"). Everything written here is invented:
fictional leagues, clubs, players and results drawn from a seeded generator,
so no record describes a real person, match or valuation.

What the fixtures must still do is exercise every code path. So the world is
small but coherent: three leagues of four clubs play double round-robins with
a domestic and a continental cup on top; squads age, retire, get replaced and
transfer; every match has lineups, appearances and a result; valuations follow
a latent quality, age and output, are published twice a year and so can be a
row's prior or its label. Each league-season has more than the twenty
labelled rows a competition needs for a strength value.

Columns match the mirror's files exactly, including ones the code never reads,
so ingestion and validation see the real shapes. Text the code never reads
(agents, stadiums, URLs) is left empty.

    python scripts/make_sample_data.py            # rewrite data/sample/
    python scripts/make_sample_data.py --check    # fail if it would change

Deterministic: the same seed writes byte-identical files, and a test holds the
committed files to that.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
SEED = 20261019
SEASONS = range(2013, 2024)
LAST_VALUATION = date(2024, 12, 31)

# Identifiers start far above the mirror's largest (players 1.6M, clubs 135k,
# games 4.9M), so a synthetic row can never be joined to a real one.
FIRST_PLAYER = 90_000_001
FIRST_CLUB = 9_000_001
FIRST_GAME = 90_000_001

COLUMNS = {
    "appearances": [
        "appearance_id", "game_id", "player_id", "player_club_id", "player_current_club_id",
        "date", "player_name", "competition_id", "yellow_cards", "red_cards", "goals",
        "assists", "minutes_played",
    ],
    "club_games": [
        "game_id", "club_id", "own_goals", "own_position", "own_manager_name", "opponent_id",
        "opponent_goals", "opponent_position", "opponent_manager_name", "hosting", "is_win",
    ],
    "competitions": [
        "competition_id", "competition_code", "name", "sub_type", "type", "country_id",
        "country_name", "domestic_league_code", "confederation", "total_clubs", "url",
    ],
    "game_lineups": [
        "game_lineups_id", "date", "game_id", "player_id", "club_id", "player_name", "type",
        "position", "number", "team_captain",
    ],
    "games": [
        "game_id", "competition_id", "season", "round", "date", "home_club_id", "away_club_id",
        "home_club_goals", "away_club_goals", "home_club_position", "away_club_position",
        "home_club_manager_name", "away_club_manager_name", "stadium", "attendance", "referee",
        "url", "home_club_formation", "away_club_formation", "home_club_name",
        "away_club_name", "aggregate", "competition_type",
    ],
    "player_valuations": [
        "player_id", "date", "market_value_in_eur", "current_club_name", "current_club_id",
        "player_club_domestic_competition_id",
    ],
    "players": [
        "player_id", "first_name", "last_name", "name", "last_season", "current_club_id",
        "player_code", "country_of_birth", "city_of_birth", "country_of_citizenship",
        "date_of_birth", "sub_position", "position", "foot", "height_in_cm",
        "contract_expiration_date", "agent_name", "image_url", "international_caps",
        "international_goals", "current_national_team_id", "url",
        "current_club_domestic_competition_id", "current_club_name", "market_value_in_eur",
        "highest_market_value_in_eur",
    ],
}  # fmt: skip

# (id, name, country, confederation, level). Fictional throughout.
LEAGUES = (
    ("SYN1", "synthetic-premier", "Northland", "europa", 0.6),
    ("SYN2", "synthetic-division", "Southland", "europa", 0.0),
    ("SYN3", "synthetic-liga", "Eastland", "amerika", -0.6),
)
CUP = ("SYNC", "synthetic-cup", "Northland", "europa")
CONTINENTAL = ("SYNI", "synthetic-continental-cup", "europa")
COUNTRIES = ("Northland", "Southland", "Eastland", "Westland")

# Squad shape: (position, sub-positions, players per club, starters per match).
SHAPE = (
    ("Goalkeeper", ("Goalkeeper",), 2, 1),
    ("Defender", ("Centre-Back", "Left-Back", "Right-Back"), 5, 4),
    ("Midfield", ("Central Midfield", "Defensive Midfield", "Attacking Midfield"), 5, 4),
    ("Attack", ("Centre-Forward", "Left Winger", "Right Winger", "Second Striker"), 4, 2),
)
SCORING_WEIGHT = {"Goalkeeper": 0.0, "Defender": 0.3, "Midfield": 1.0, "Attack": 2.5}


@dataclass
class Player:
    player_id: int
    name: str
    born: date
    position: str
    sub_position: str
    foot: str | None
    height: float | None
    country: str
    quality: float
    club_id: int
    number: int
    first_season: int
    last_season: int = 0
    output: list[float] = field(default_factory=list)


@dataclass
class World:
    rng: np.random.Generator
    players: dict[int, Player] = field(default_factory=dict)
    rows: dict[str, list[dict[str, object]]] = field(
        default_factory=lambda: {name: [] for name in COLUMNS}
    )
    next_player: int = FIRST_PLAYER
    next_game: int = FIRST_GAME

    def club_league(self, club_id: int) -> tuple[str, str, str, str, float]:
        return LEAGUES[(club_id - FIRST_CLUB) // 4]


def club_name(club_id: int) -> str:
    return f"Synthetic Club {club_id}"


def new_player(world: World, club_id: int, slot: int, season: int, age: int) -> Player:
    position, subs, _, _ = SHAPE[slot]
    rng = world.rng
    player_id = world.next_player
    world.next_player += 1
    foot = rng.choice(["right", "left", "both", ""], p=[0.68, 0.22, 0.06, 0.04])
    player = Player(
        player_id=player_id,
        name=f"Synthetic Player {player_id - FIRST_PLAYER + 1:04d}",
        born=date(season - age, 1, 1) + timedelta(days=int(rng.integers(0, 365))),
        position=position,
        sub_position=str(rng.choice(subs)),
        foot=str(foot) or None,
        height=None if rng.random() < 0.03 else float(np.clip(rng.normal(181, 7), 165, 200)),
        country=str(rng.choice(COUNTRIES)),
        quality=float(rng.normal(0, 1)),
        club_id=club_id,
        number=int(rng.integers(1, 40)),
        first_season=season,
    )
    world.players[player_id] = player
    return player


def squads_for(world: World, season: int, squads: dict[int, list[Player]]) -> None:
    """Age the squads by a season: retire, replace, transfer."""
    rng = world.rng
    for club_id, squad in squads.items():
        for index, player in enumerate(squad):
            age = season - player.born.year
            if age >= 35 or rng.random() < 0.08:
                slot = next(i for i, shape in enumerate(SHAPE) if shape[0] == player.position)
                squad[index] = new_player(world, club_id, slot, season, int(rng.integers(17, 21)))
    # A few transfers between clubs of the same position, so careers cross
    # clubs and leagues and the momentum features see a change.
    clubs = list(squads)
    for _ in range(4):
        a, b = rng.choice(clubs, size=2, replace=False)
        i = int(rng.integers(0, len(squads[a])))
        j = next((k for k, p in enumerate(squads[b]) if p.position == squads[a][i].position), None)
        if j is not None:
            squads[a][i], squads[b][j] = squads[b][j], squads[a][i]
            squads[a][i].club_id, squads[b][j].club_id = int(a), int(b)
    for squad in squads.values():
        for player in squad:
            player.last_season = season


def pick_lineup(world: World, squad: list[Player]) -> tuple[list[Player], list[Player]]:
    """Starters by quality and fitness, then a five-man bench."""
    rng = world.rng
    starters: list[Player] = []
    bench: list[Player] = []
    for position, _, _, starting in SHAPE:
        group = [p for p in squad if p.position == position and rng.random() > 0.1]
        group.sort(key=lambda p: p.quality + rng.normal(0, 0.7), reverse=True)
        starters += group[:starting]
        bench += group[starting:]
    return starters, bench[:5]


def play_match(
    world: World,
    *,
    competition: str,
    competition_type: str,
    season: int,
    day: date,
    round_name: str,
    home: tuple[int, list[Player]],
    away: tuple[int, list[Player]],
    positions: dict[int, int] | None,
) -> tuple[int, int]:
    rng = world.rng
    game_id = world.next_game
    world.next_game += 1
    goals: dict[int, int] = {}
    for (club_id, squad), (_, other) in ((home, away), (away, home)):
        edge = np.mean([p.quality for p in squad]) - np.mean([p.quality for p in other])
        goals[club_id] = int(rng.poisson(max(0.3, 1.35 + 0.5 * edge)))

    for side, (club_id, squad) in enumerate((home, away)):
        opponent = away[0] if side == 0 else home[0]
        starters, bench = pick_lineup(world, squad)
        used = bench[: int(rng.integers(1, 4))]
        minutes = {p.player_id: 90 for p in starters}
        for sub, off in zip(
            used, rng.choice(len(starters), size=len(used), replace=False), strict=True
        ):
            leaving = int(rng.integers(46, 86))
            minutes[starters[int(off)].player_id] = leaving
            minutes[sub.player_id] = 90 - leaving
        on_pitch = starters + used
        weights = np.array([SCORING_WEIGHT[p.position] * np.exp(p.quality) for p in on_pitch])
        scored = np.zeros(len(on_pitch), dtype=int)
        assisted = np.zeros(len(on_pitch), dtype=int)
        for _ in range(goals[club_id]):
            scorer = int(rng.choice(len(on_pitch), p=weights / weights.sum()))
            scored[scorer] += 1
            if rng.random() < 0.7:
                helper = int(rng.choice(len(on_pitch), p=weights / weights.sum()))
                assisted[helper] += 1
        captain = max(starters, key=lambda p: (day.year - p.first_season, p.player_id))

        for player in starters + bench:
            moved = rng.random() < 0.1 and player.position != "Goalkeeper"
            subs = next(s for name, s, _, _ in SHAPE if name == player.position)
            world.rows["game_lineups"].append(
                {
                    "game_lineups_id": hashlib.md5(
                        f"{game_id}-{player.player_id}".encode()
                    ).hexdigest(),
                    "date": day.isoformat(),
                    "game_id": game_id,
                    "player_id": player.player_id,
                    "club_id": club_id,
                    "player_name": player.name,
                    "type": "starting_lineup" if player in starters else "substitutes",
                    "position": str(rng.choice(subs)) if moved else player.sub_position,
                    "number": player.number,
                    "team_captain": int(player is captain),
                }
            )
        for index, player in enumerate(on_pitch):
            world.rows["appearances"].append(
                {
                    "appearance_id": f"{game_id}_{player.player_id}",
                    "game_id": game_id,
                    "player_id": player.player_id,
                    "player_club_id": club_id,
                    "player_current_club_id": club_id,
                    "date": day.isoformat(),
                    "player_name": player.name,
                    "competition_id": competition,
                    "yellow_cards": int(rng.random() < 0.12),
                    "red_cards": int(rng.random() < 0.01),
                    "goals": int(scored[index]),
                    "assists": int(assisted[index]),
                    "minutes_played": minutes[player.player_id],
                }
            )
            player.output[-1] += (
                scored[index] + 0.6 * assisted[index] + minutes[player.player_id] / 900
            )

        own, against = goals[club_id], goals[opponent]
        world.rows["club_games"].append(
            {
                "game_id": game_id,
                "club_id": club_id,
                "own_goals": own,
                "own_position": positions.get(club_id) if positions else None,
                "own_manager_name": None,
                "opponent_id": opponent,
                "opponent_goals": against,
                "opponent_position": positions.get(opponent) if positions else None,
                "opponent_manager_name": None,
                "hosting": "Home" if side == 0 else "Away",
                "is_win": int(own > against),
            }
        )

    world.rows["games"].append(
        {
            "game_id": game_id,
            "competition_id": competition,
            "season": season,
            "round": round_name,
            "date": day.isoformat(),
            "home_club_id": home[0],
            "away_club_id": away[0],
            "home_club_goals": goals[home[0]],
            "away_club_goals": goals[away[0]],
            "home_club_position": positions.get(home[0]) if positions else None,
            "away_club_position": positions.get(away[0]) if positions else None,
            "home_club_manager_name": None,
            "away_club_manager_name": None,
            "stadium": None,
            "attendance": int(rng.integers(2_000, 40_000)),
            "referee": None,
            "url": None,
            "home_club_formation": None,
            "away_club_formation": None,
            "home_club_name": club_name(home[0]),
            "away_club_name": club_name(away[0]),
            "aggregate": f"{goals[home[0]]}:{goals[away[0]]}",
            "competition_type": competition_type,
        }
    )
    return goals[home[0]], goals[away[0]]


def play_season(world: World, season: int, squads: dict[int, list[Player]]) -> None:
    for squad in squads.values():
        for player in squad:
            player.output.append(0.0)
    start = date(season, 8, 15)
    for league_index, (league, *_rest) in enumerate(LEAGUES):
        clubs = [FIRST_CLUB + 4 * league_index + k for k in range(4)]
        points = dict.fromkeys(clubs, 0)
        pairs = [(a, b) for a in clubs for b in clubs if a != b]
        world.rng.shuffle(pairs)
        for matchday, (home, away) in enumerate(pairs):
            table = sorted(clubs, key=lambda c: (-points[c], c))
            positions = {club: rank + 1 for rank, club in enumerate(table)}
            day = start + timedelta(days=24 * matchday + league_index)
            home_goals, away_goals = play_match(
                world,
                competition=league,
                competition_type="domestic_league",
                season=season,
                day=day,
                round_name=f"{matchday + 1}. Matchday",
                home=(home, squads[home]),
                away=(away, squads[away]),
                positions=positions,
            )
            points[home] += 3 if home_goals > away_goals else int(home_goals == away_goals)
            points[away] += 3 if away_goals > home_goals else int(home_goals == away_goals)

    for offset, (home, away) in enumerate(
        [(FIRST_CLUB + h, FIRST_CLUB + a) for h, a in ((0, 2), (1, 3))]
    ):
        play_match(
            world,
            competition=CUP[0],
            competition_type="domestic_cup",
            season=season,
            day=date(season + 1, 2, 10) + timedelta(days=offset),
            round_name="Semi-Finals",
            home=(home, squads[home]),
            away=(away, squads[away]),
            positions=None,
        )
    for offset, (home, away) in enumerate(
        [(FIRST_CLUB + h, FIRST_CLUB + a) for h, a in ((0, 4), (5, 8))]
    ):
        play_match(
            world,
            competition=CONTINENTAL[0],
            competition_type="international_cup",
            season=season,
            day=date(season, 10, 20) + timedelta(days=7 * offset),
            round_name="Group A",
            home=(home, squads[home]),
            away=(away, squads[away]),
            positions=None,
        )


def value_of(world: World, player: Player, on: date) -> int:
    """A valuation: latent quality, league, age curve and recent output."""
    age = (on - player.born).days / 365.25
    league_level = world.club_league(player.club_id)[4]
    recent = np.mean(player.output[-2:]) if player.output else 0.0
    log_value = (
        13.4
        + 0.9 * player.quality
        + league_level
        + 0.08 * recent
        - 0.012 * (age - 26) ** 2
        + world.rng.normal(0, 0.15)
    )
    return max(25_000, int(round(np.exp(log_value) / 25_000) * 25_000))


def valuations(world: World, season: int, squads: dict[int, list[Player]]) -> None:
    """Published in December and June, plus a few in early autumn.

    Half of all debutants go unvalued in their first season, as many real ones
    do, so the variant without a prior value has rows to drop.
    """
    for squad in squads.values():
        for player in squad:
            if player.first_season == season and world.rng.random() < 0.5:
                continue
            league = world.club_league(player.club_id)
            days = [date(season, 12, 15), date(season + 1, 6, 20)]
            if world.rng.random() < 0.3:
                days.append(date(season + 1, 9, 1) + timedelta(days=int(world.rng.integers(0, 40))))
            for day in days:
                if day <= LAST_VALUATION:
                    world.rows["player_valuations"].append(
                        {
                            "player_id": player.player_id,
                            "date": day.isoformat(),
                            "market_value_in_eur": value_of(world, player, day),
                            "current_club_name": club_name(player.club_id),
                            "current_club_id": player.club_id,
                            "player_club_domestic_competition_id": league[0],
                        }
                    )


def build() -> dict[str, pd.DataFrame]:
    world = World(rng=np.random.default_rng(SEED))
    first = SEASONS[0]
    squads: dict[int, list[Player]] = {}
    for club_id in range(FIRST_CLUB, FIRST_CLUB + 12):
        squads[club_id] = [
            new_player(world, club_id, slot, first, int(world.rng.integers(18, 33)))
            for slot, (_, _, size, _) in enumerate(SHAPE)
            for _ in range(size)
        ]
    for season in SEASONS:
        if season != first:
            squads_for(world, season, squads)
        else:
            for squad in squads.values():
                for player in squad:
                    player.last_season = season
        play_season(world, season, squads)
        valuations(world, season, squads)
    # The December after the last season labels it; LAST_VALUATION stops there.
    valuations(world, SEASONS[-1] + 1, squads)

    # Valuations can repeat a (player, date) when a transfer and a scheduled
    # date coincide; the mirror's primary key forbids that.
    frames = {name: pd.DataFrame(rows, columns=COLUMNS[name]) for name, rows in world.rows.items()}
    frames["player_valuations"] = frames["player_valuations"].drop_duplicates(
        ["player_id", "date"], keep="last"
    )
    frames["competitions"] = competitions()
    frames["players"] = players(world, frames["player_valuations"])
    return frames


def competitions() -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "competition_id": league,
            "competition_code": name,
            "name": name,
            "sub_type": "first_tier",
            "type": "domestic_league",
            "country_id": index + 1,
            "country_name": country,
            "domestic_league_code": league,
            "confederation": confederation,
            "total_clubs": 4.0,
            "url": None,
        }
        for index, (league, name, country, confederation, _) in enumerate(LEAGUES)
    ]
    rows.append(
        {
            "competition_id": CUP[0],
            "competition_code": CUP[1],
            "name": CUP[1],
            "sub_type": "domestic_cup",
            "type": "domestic_cup",
            "country_id": 1,
            "country_name": CUP[2],
            "domestic_league_code": LEAGUES[0][0],
            "confederation": CUP[3],
            "total_clubs": 4.0,
            "url": None,
        }
    )
    rows.append(
        {
            "competition_id": CONTINENTAL[0],
            "competition_code": CONTINENTAL[1],
            "name": CONTINENTAL[1],
            "sub_type": "group_stage",
            "type": "international_cup",
            "country_id": -1,
            "country_name": None,
            "domestic_league_code": None,
            "confederation": CONTINENTAL[2],
            "total_clubs": None,
            "url": None,
        }
    )
    return pd.DataFrame(rows, columns=COLUMNS["competitions"])


def players(world: World, valuations_frame: pd.DataFrame) -> pd.DataFrame:
    by_player = valuations_frame.sort_values("date").groupby("player_id")["market_value_in_eur"]
    latest, highest = by_player.last(), by_player.max()
    rows = []
    for player in world.players.values():
        first_name, last_name = player.name.rsplit(" ", 1)
        rows.append(
            {
                "player_id": player.player_id,
                "first_name": first_name,
                "last_name": last_name,
                "name": player.name,
                "last_season": player.last_season,
                "current_club_id": player.club_id,
                "player_code": player.name.lower().replace(" ", "-"),
                "country_of_birth": player.country,
                "city_of_birth": None,
                "country_of_citizenship": player.country,
                "date_of_birth": f"{player.born.isoformat()} 00:00:00",
                "sub_position": player.sub_position,
                "position": player.position,
                "foot": player.foot,
                "height_in_cm": player.height and round(player.height),
                "contract_expiration_date": None,
                "agent_name": None,
                "image_url": None,
                "international_caps": None,
                "international_goals": None,
                "current_national_team_id": None,
                "url": None,
                "current_club_domestic_competition_id": world.club_league(player.club_id)[0],
                "current_club_name": club_name(player.club_id),
                "market_value_in_eur": latest.get(player.player_id),
                "highest_market_value_in_eur": highest.get(player.player_id),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS["players"])


def render(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if any file would change")
    parser.add_argument("--out", type=Path, default=SAMPLE_DIR)
    args = parser.parse_args(argv)

    stale = []
    for name, frame in build().items():
        path = args.out / f"{name}.csv"
        text = render(frame)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(
            f"{path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}: {len(frame):,} rows"
        )
    if stale:
        print(f"out of date: {', '.join(stale)} — run scripts/make_sample_data.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
