"""Load a projections file into a clean player table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .settings import POSITIONS

ALIASES = {
    "name": ["name", "player", "player_name", "playername"],
    "position": ["position", "pos", "roster_position"],
    "team": ["team", "tm", "teamabbrev"],
    "opponent": ["opponent", "opp", "opp_team"],
    "salary": ["salary", "sal", "dk_salary"],
    "mean": ["mean", "proj", "projection", "fpts", "points", "dk_points", "median"],
    "floor": ["floor", "flr", "low"],
    "ceiling": ["ceiling", "ceil", "high"],
    "ownership": ["ownership", "own", "pown", "proj_own", "projected_ownership", "own%"],
}
REQUIRED = ["name", "position", "team", "opponent", "salary", "mean"]


def load_projections(path: str) -> pd.DataFrame:
    """Read a CSV and return columns: name, position, team, opponent, salary, mean,
    floor, ceiling, ownership (percent, 0-100). Floor, ceiling and ownership may be NaN."""
    raw = pd.read_csv(path)
    lookup = {c.strip().lower().replace(" ", "_"): c for c in raw.columns}
    out = pd.DataFrame()
    for field, names in ALIASES.items():
        source = next((lookup[n] for n in names if n in lookup), None)
        if source is None:
            if field in REQUIRED:
                raise ValueError(
                    f"{path}: no '{field}' column. Accepted names: {', '.join(names)}"
                )
            out[field] = np.nan
        else:
            out[field] = raw[source]

    out["position"] = (
        out["position"].astype(str).str.upper().str.strip().replace({"DEF": "DST", "D": "DST"})
    )
    for col in ("team", "opponent"):
        out[col] = out[col].astype(str).str.upper().str.strip().str.replace("@", "").str.replace("VS", "")
    for col in ("salary", "mean", "floor", "ceiling", "ownership"):
        out[col] = pd.to_numeric(
            out[col].astype(str).str.replace(r"[$,%]", "", regex=True), errors="coerce"
        )

    out = out[out["position"].isin(POSITIONS)]
    out = out.dropna(subset=["salary", "mean"])
    out = out[(out["mean"] > 0) | (out["position"] == "DST")]
    if out.empty:
        raise ValueError(f"{path}: no usable players after cleaning")
    # Ownership given as fractions (0-1) rather than percent.
    if out["ownership"].notna().any() and out["ownership"].max() <= 1.5:
        out["ownership"] = out["ownership"] * 100
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
