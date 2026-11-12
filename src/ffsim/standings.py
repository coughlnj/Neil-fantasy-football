"""Read a DraftKings contest standings export and measure how the real field built.

The export lists every entry's lineup on the left and, to the right, every player's
actual ownership and score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .draftkings import normalise_name

SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
_LINEUP = r"^DST (.*?) FLEX (.*?) QB (.*?) RB (.*?) RB (.*?) TE (.*?) WR (.*?) WR (.*?) WR (.*)$"
_ORDER = ["DST", "FLEX", "QB", "RB1", "RB2", "TE", "WR1", "WR2", "WR3"]
GROUPS = [("1 entry", 1, 1), ("2-19 entries", 2, 19), ("20-149 entries", 20, 149), ("150 entries", 150, 10**9)]


def read_standings(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (entries, players). entries has one row per entry with a lineup: user,
    the number of entries that user has, points, and a column per roster slot.
    players has actual ownership (percent), score and position."""
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if "Lineup" not in raw.columns or "%Drafted" not in raw.columns:
        raise ValueError(f"{path}: not a DraftKings contest standings export")
    filled = raw[raw["Lineup"].notna() & raw["EntryId"].notna()]
    slots = filled["Lineup"].str.extract(_LINEUP)
    slots.columns = _ORDER
    slots = slots.apply(lambda col: col.str.strip())
    parsed = slots["DST"].notna()
    if not parsed.any():
        raise ValueError(f"{path}: no classic (QB/RB/WR/TE/FLEX/DST) lineups found")
    filled, slots = filled[parsed], slots[parsed]
    user = filled["EntryName"].str.replace(r" \(\d+/\d+\)$", "", regex=True)
    entries = pd.DataFrame(
        {"rank": filled["Rank"].astype(int), "entry_id": filled["EntryId"], "user": user,
         "user_entries": user.map(user.value_counts()), "points": filled["Points"].astype(float)}
    )
    entries = pd.concat([entries, slots[SLOTS]], axis=1).reset_index(drop=True)

    position = {}
    for slot in SLOTS:
        if slot != "FLEX":
            position.update(dict.fromkeys(entries[slot].unique(), slot.rstrip("123")))
    players = raw[["Player", "%Drafted", "FPTS"]].dropna(subset=["Player"])
    players = pd.DataFrame(
        {"name": players["Player"].str.strip(),
         "ownership": players["%Drafted"].str.rstrip("%").astype(float),
         "points": players["FPTS"].astype(float)}
    )
    # A player used both in his own slot and in the flex is listed once for each.
    players = players.groupby("name", as_index=False).agg(ownership=("ownership", "sum"), points=("points", "first"))
    players["position"] = players["name"].map(position)
    return entries, players


def standings_report(entries: pd.DataFrame, players: pd.DataFrame, slate: pd.DataFrame | None = None):
    """Return (overall, by_group, notes). overall and by_group measure how lineups were
    built; by_group splits entries by how many their entrant has. With a slate (name,
    team, and optionally opponent and salary) stacking and salary are measured too."""
    position = {}
    for slot in SLOTS:
        if slot != "FLEX":
            position.update(dict.fromkeys(entries[slot].unique(), slot.rstrip("123")))
    if slate is not None and "position" in slate.columns:  # players only ever used in the flex
        for name, pos in zip(slate["name"].str.strip(), slate["position"]):
            if pd.isna(position.get(name)):
                position[name] = pos
    lineups = entries[SLOTS]
    flex = lineups["FLEX"].map(position)
    key = lineups.apply(lambda row: "|".join(sorted(row)), axis=1)
    copies = key.map(key.value_counts())
    measures = {
        "Flex is a RB": flex == "RB", "Flex is a WR": flex == "WR", "Flex is a TE": flex == "TE",
        "Shares its lineup with another entry": copies > 1,
        "In a group of 5+ identical lineups": copies >= 5,
    }
    numbers = {"Average copies of each entry's lineup": copies.astype(float), "Average points": entries["points"]}
    notes = []

    if slate is not None:
        known = slate.assign(key=slate["name"].map(normalise_name)).drop_duplicates("key").set_index("key")
        look = lambda col: lineups.apply(lambda c: c.map(normalise_name).map(known[col]))
        team = look("team")
        qb_team = team["QB"]
        has_qb = qb_team.notna()
        catcher = pd.DataFrame({s: True for s in ("TE", "WR1", "WR2", "WR3")}, index=lineups.index)
        catcher["FLEX"] = flex.isin(["WR", "TE"])
        cols = list(catcher.columns)
        mates = (team[cols].eq(qb_team, axis=0) & catcher).sum(axis=1)
        notes.append(f"Stacking measured on {has_qb.mean():.0%} of entries (those whose QB is on the slate file).")
        mask = lambda cond: cond.where(has_qb)
        measures.update({
            "QB with no pass catcher": mask(mates == 0),
            "QB with exactly 1 pass catcher": mask(mates == 1),
            "QB with 2 or more pass catchers": mask(mates >= 2),
            "QB with his own RB": mask(team[["RB1", "RB2"]].eq(qb_team, axis=0).any(axis=1) | ((team["FLEX"] == qb_team) & (flex == "RB"))),
        })
        # A slate file from a different week has the right teams for most players but
        # the wrong salaries and opponents; totals over the cap give it away.
        same_week = True
        if "salary" in known.columns:
            salary = look("salary")
            complete = salary.notna().all(axis=1)
            total = salary.sum(axis=1).where(complete)
            same_week = complete.mean() >= 0.5 and total.median() <= 50000
            if not same_week:
                notes.append("The slate file does not look like this contest's week (salaries do not fit the "
                             "cap or most lineups have players missing from it), so salary use and "
                             "opponent-based measures are left out.")
        if "opponent" in known.columns and same_week:
            qb_opp = lineups["QB"].map(normalise_name).map(known["opponent"])
            bring = (team[cols].eq(qb_opp, axis=0) & catcher).any(axis=1)
            measures["Stacks that add an opposing pass catcher"] = bring.where(has_qb & (mates >= 1))
            dst_opp = lineups["DST"].map(normalise_name).map(known["opponent"])
            measures["Defense facing its own QB"] = (dst_opp == qb_team).where(has_qb & dst_opp.notna())
        if "salary" in known.columns and same_week:
            notes.append(f"Salary measured on {complete.mean():.0%} of entries (all nine players on the slate file).")
            measures["Uses $49,500 or more"] = (total >= 49500).where(complete)
            numbers["Average salary used"] = total
            numbers["Median salary used"] = total

    def summarise(rows: pd.Series) -> dict:
        out = {name: round(100 * float(cond[rows].astype(float).mean()), 1) for name, cond in measures.items()}
        for name, value in numbers.items():
            stat = value[rows].median() if name.startswith("Median") else value[rows].mean()
            out[name] = round(float(stat), 2)
        return out

    everyone = pd.Series(True, index=entries.index)
    overall = pd.Series({"Share of entries": 100.0, **summarise(everyone)}, name="all entries")
    groups = {}
    for label, low, high in GROUPS:
        rows = entries["user_entries"].between(low, high)
        groups[label] = {"Share of entries": round(100 * float(rows.mean()), 1), **summarise(rows)}
    return overall, pd.DataFrame(groups), notes
