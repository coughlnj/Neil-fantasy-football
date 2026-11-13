"""Read a DraftKings entries export (DKEntries.csv) and write lineups back into it.

The export lists your contest entries on the left and, to the right, every player on
the slate with their DraftKings ID, salary and game.
"""
from __future__ import annotations

import csv
import re

import pandas as pd

LINEUP_SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
TEAM_ALIASES = {"JAC": "JAX", "LA": "LAR", "WSH": "WAS", "GNB": "GB", "KAN": "KC", "NWE": "NE",
                "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV", "HST": "HOU", "BLT": "BAL",
                "CLV": "CLE", "ARZ": "ARI"}


def normalise_team(team: str) -> str:
    team = str(team).upper().strip()
    return TEAM_ALIASES.get(team, team)


def normalise_name(name: str) -> str:
    name = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " "))
    return " ".join(w for w in name.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v", "dst", "defense"})


def read_dk_entries(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (entries, players) from a DraftKings entries export."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    entries = pd.DataFrame(
        [r[:4] for r in rows[1:] if r and r[0].strip().isdigit()],
        columns=["entry_id", "contest_name", "contest_id", "entry_fee"],
    )
    header = next((i for i, r in enumerate(rows) if "Name + ID" in r), None)
    if header is None:
        raise ValueError(f"{path}: no player list found; is this a DraftKings entries export?")
    col = rows[header].index("Position")
    players = []
    for r in rows[header + 1 :]:
        if len(r) <= col + 7 or not r[col + 3].strip().isdigit():
            continue
        position, _, name, dk_id, _, salary, game_info, team = (v.strip() for v in r[col : col + 8])
        matchup, _, kickoff = game_info.partition(" ")
        away, _, home = matchup.partition("@")
        team = normalise_team(team)
        opponent = normalise_team(home if team == normalise_team(away) else away)
        players.append({"name": name, "position": position, "team": team, "opponent": opponent,
                        "salary": int(salary), "dk_id": dk_id, "kickoff": kickoff})
    return entries, pd.DataFrame(players)


def attach_dk(projections: pd.DataFrame, dk_players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match projected players to the DraftKings slate. DraftKings' position, team,
    opponent and salary replace the projection's own. Returns (matched, unmatched)."""
    dk = dk_players.copy()
    dk["key"] = dk["name"].map(normalise_name)
    by_name_pos = {(k, p): i for i, (k, p) in enumerate(zip(dk["key"], dk["position"]))}
    by_name = {k: i for i, k in enumerate(dk["key"]) if (dk["key"] == k).sum() == 1}
    dst = dk[dk["position"] == "DST"]
    dst_by_team = dict(zip(dst["team"], dst.index))

    by_id = {str(v): i for i, v in enumerate(dk["dk_id"])}

    def find(row) -> int | None:
        given = str(row.get("dk_id", "")).split(".")[0]
        if given in by_id:
            return by_id[given]
        key = normalise_name(row["name"])
        position = str(row.get("position", "")).upper().strip()
        position = {"DEF": "DST", "D": "DST", "D/ST": "DST"}.get(position, position)
        if position == "DST" or key in set(dst["key"]):
            team = normalise_team(row.get("team", ""))
            if team in dst_by_team:
                return dst_by_team[team]
            hits = [i for i, k in zip(dst.index, dst["key"]) if k and (k in key or key in k)]
            return hits[0] if len(hits) == 1 else None
        return by_name_pos.get((key, position), by_name.get(key))

    found = [find(row) for _, row in projections.iterrows()]
    matched = projections[[f is not None for f in found]].copy()
    hit = dk.iloc[[f for f in found if f is not None]]
    for col in ("position", "team", "opponent", "salary", "dk_id", "kickoff"):
        matched[col] = hit[col].to_numpy()
    matched = matched[~matched["dk_id"].duplicated()]
    return matched, projections[[f is None for f in found]]


def write_upload(path: str, entries: pd.DataFrame, lineup_ids: list[list[str]]) -> int:
    """Write a DraftKings upload file assigning one lineup per entry. Returns the number
    of entries filled."""
    filled = min(len(entries), len(lineup_ids))
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + LINEUP_SLOTS)
        for (_, entry), ids in zip(entries.iterrows(), lineup_ids):
            writer.writerow([entry["entry_id"], entry["contest_name"], entry["contest_id"], entry["entry_fee"]] + list(ids))
    return filled


def read_lineups(path: str, players: pd.DataFrame) -> tuple[list[list[int]], list[str]]:
    """Read lineups from a CSV: a DraftKings upload or entries file, or any file whose
    rows hold nine players. A player may be a DraftKings ID, "Name (ID)" or a name.
    Returns (lineups as row numbers into `players`, problems found)."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    header = [c.strip().upper() for c in rows[0]]
    if "QB" in header and "DST" in header[header.index("QB"):]:
        first = header.index("QB")
        cols, rows = list(range(first, header.index("DST", first) + 1)), rows[1:]
    else:
        cols = None
    by_id = {str(v): i for i, v in enumerate(players["dk_id"])} if "dk_id" in players.columns else {}
    names = players["name"].map(normalise_name)
    by_name = {k: i for i, k in enumerate(names) if (names == k).sum() == 1}

    def find(cell: str) -> int | None:
        cell = cell.strip()
        inside = re.search(r"\((\d+)\)\s*$", cell)
        for key in (cell, inside.group(1) if inside else None):
            if key in by_id:
                return by_id[key]
        return by_name.get(normalise_name(re.sub(r"\s*\([^)]*\)\s*$", "", cell)))

    lineups, problems = [], []
    for number, row in enumerate(rows, start=2 if cols else 1):
        cells = [row[c] for c in cols if c < len(row)] if cols else [c for c in row if c.strip()]
        cells = [c for c in cells if c.strip()]
        if not cells:
            continue  # an entry with no lineup yet
        found = [find(c) for c in cells]
        missing = [c.strip() for c, f in zip(cells, found) if f is None]
        if missing:
            problems.append(f"row {number}: not in the simulation: {', '.join(missing)}")
        elif len(found) != 9 or len(set(found)) != 9:
            problems.append(f"row {number}: expected nine different players, found {len(set(found))}")
        else:
            lineups.append(found)
    return lineups, problems


def check_lineup(lineup: list[int], players: pd.DataFrame, cap: int = 50000) -> str | None:
    """Why a lineup is not a legal DraftKings classic lineup, or None if it is."""
    pos = players["position"].to_numpy()[lineup]
    count = {p: int((pos == p).sum()) for p in ("QB", "RB", "WR", "TE", "DST")}
    if count["QB"] != 1 or count["DST"] != 1 or count["RB"] < 2 or count["WR"] < 3 or count["TE"] < 1:
        return "positions are " + ", ".join(f"{n} {p}" for p, n in count.items())
    salary = int(players["salary"].to_numpy()[lineup].sum())
    return f"salary ${salary:,} is over the cap" if salary > cap else None
