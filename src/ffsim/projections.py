"""Load a projections file into a clean player table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .draftkings import attach_dk, normalise_team
from .settings import POSITIONS

ALIASES = {
    "name": ["name", "player", "player_name", "playername"],
    "position": ["position", "pos", "roster_position", "dk_pos"],
    "team": ["team", "tm", "teamabbrev"],
    "opponent": ["opponent", "opp", "opp_team"],
    "salary": ["salary", "sal", "dk_salary"],
    "mean": ["mean", "proj", "projection", "fpts", "points", "dk_points", "dk_proj", "median"],
    "floor": ["floor", "flr", "low", "dk_floor"],
    "ceiling": ["ceiling", "ceil", "high", "dk_ceiling"],
    "rush_share": ["rush_share", "rushing_share", "rush_pct", "rush%"],
    "ownership": ["ownership", "own", "pown", "proj_own", "projected_ownership", "own%", "large_field", "small_field"],
    "dk_id": ["dk_id", "id", "draftkings_id"],
}
REQUIRED = ["name", "position", "team", "opponent", "salary", "mean"]


def load_projections(
    path: str, dk_players: pd.DataFrame | None = None, log=lambda message: None, ownership_column: str | None = None
) -> pd.DataFrame:
    """Read a CSV and return columns: name, position, team, opponent, salary, mean,
    floor, ceiling, ownership (percent, 0-100). Floor, ceiling and ownership may be NaN.

    With dk_players (from a DraftKings entries export), players are matched to the
    slate by name: DraftKings supplies position, team, opponent, salary and player ID,
    and projected players who are not on the slate are dropped."""
    raw = pd.read_csv(path)
    required = REQUIRED if dk_players is None else ["name", "mean"]
    lookup = {c.strip().lower().replace(" ", "_"): c for c in raw.columns}
    aliases = dict(ALIASES)
    if ownership_column:
        key = ownership_column.strip().lower().replace(" ", "_")
        if key not in lookup:
            raise ValueError(f"{path}: no '{ownership_column}' column for ownership")
        aliases["ownership"] = [key]
    out = pd.DataFrame()
    for field, names in aliases.items():
        source = next((lookup[n] for n in names if n in lookup), None)
        if source is None:
            if field in required:
                raise ValueError(
                    f"{path}: no '{field}' column. Accepted names: {', '.join(names)}"
                )
            out[field] = np.nan
        else:
            out[field] = raw[source]

    if dk_players is not None:
        out["mean"] = pd.to_numeric(out["mean"], errors="coerce")
        out, missing = attach_dk(out, dk_players)
        missing = missing[missing["mean"] > 0].sort_values("mean", ascending=False)
        log(f"Matched {len(out)} projected players to the DraftKings slate; {len(missing)} not on it.")
        if len(missing):
            log("Highest-projected unmatched: " + ", ".join(
                f"{n} ({m:g})" for n, m in zip(missing["name"].head(8), missing["mean"].head(8))))

    out["position"] = (
        out["position"].astype(str).str.upper().str.strip().replace({"DEF": "DST", "D": "DST"})
    )
    for col in ("team", "opponent"):
        out[col] = out[col].astype(str).str.upper().str.strip().str.replace("@", "").str.replace("VS", "")
        out[col] = out[col].map(normalise_team)
    for col in ("salary", "mean", "floor", "ceiling", "ownership", "rush_share"):
        out[col] = pd.to_numeric(
            out[col].astype(str).str.replace(r"[$,%]", "", regex=True), errors="coerce"
        )

    if out["dk_id"].notna().any():
        out["dk_id"] = out["dk_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    else:
        out = out.drop(columns="dk_id")
    out = out[out["position"].isin(POSITIONS)]
    out = out.dropna(subset=["salary", "mean"])
    out = out[(out["mean"] > 0) | (out["position"] == "DST")]
    if out.empty:
        raise ValueError(f"{path}: no usable players after cleaning")
    # Ownership given as fractions (0-1) rather than percent.
    if out["ownership"].notna().any() and out["ownership"].max() <= 1.5:
        out["ownership"] = out["ownership"] * 100
    # Rushing share given as percent (0-100) rather than a fraction.
    if out["rush_share"].notna().any() and out["rush_share"].max() > 1:
        out["rush_share"] = out["rush_share"] / 100
    out["game"] = [" v ".join(sorted(pair)) for pair in zip(out["team"], out["opponent"])]
    out = out.sort_values(["game", "team", "position", "mean"], ascending=[True, True, True, False])
    return out.reset_index(drop=True)


def diff_projections(old: pd.DataFrame, new: pd.DataFrame, min_change: float = 0.5) -> pd.DataFrame:
    """Players whose mean projection moved by at least min_change, biggest moves first."""
    keys = ["name", "position", "team"]
    merged = old[keys + ["mean", "ownership"]].merge(
        new[keys + ["mean", "ownership"]], on=keys, how="outer", suffixes=("_old", "_new")
    )
    merged["change"] = merged["mean_new"].fillna(0) - merged["mean_old"].fillna(0)
    merged["status"] = np.where(
        merged["mean_old"].isna(), "added", np.where(merged["mean_new"].isna(), "removed", "changed")
    )
    moved = merged[merged["change"].abs() >= min_change]
    return moved.reindex(moved["change"].abs().sort_values(ascending=False).index).reset_index(drop=True)
