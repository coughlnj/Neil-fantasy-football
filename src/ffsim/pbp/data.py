"""Load and tidy play-by-play data (see scripts/fetch_pbp.py)."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COLUMNS = [
    "game_id", "season", "week", "season_type", "play_id", "home_team", "away_team", "posteam", "defteam",
    "play_type", "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining", "half_seconds_remaining",
    "game_half", "score_differential", "posteam_score", "defteam_score", "yards_gained", "qb_dropback", "qb_scramble",
    "qb_kneel", "qb_spike", "sack", "complete_pass", "incomplete_pass", "interception", "fumble_lost", "touchdown",
    "td_team", "return_touchdown", "safety", "air_yards", "yards_after_catch", "pass_attempt", "rush_attempt",
    "passer_player_id", "passer_player_name", "receiver_player_id", "receiver_player_name", "rusher_player_id",
    "rusher_player_name", "penalty", "penalty_team", "penalty_yards", "first_down_penalty", "first_down",
    "field_goal_result", "kick_distance", "extra_point_result", "two_point_attempt", "two_point_conv_result",
    "touchback", "out_of_bounds", "timeout", "epa",
]


def load_pbp(directory: str = "data/raw/pbp", seasons: list[int] | None = None) -> pd.DataFrame:
    """All plays for the given seasons (default: every downloaded file), in game order,
    with a `kind` label and what happened next (clock used, who has the ball where)."""
    files = sorted(glob.glob(os.path.join(directory, "play_by_play_*.csv.gz")))
    if seasons:
        files = [f for f in files if int(f.split("_")[-1].split(".")[0]) in seasons]
    if not files:
        raise FileNotFoundError(f"no play-by-play files in {directory}; run scripts/fetch_pbp.py")
    plays = pd.concat([pd.read_csv(f, usecols=COLUMNS, low_memory=False) for f in files], ignore_index=True)
    plays = plays.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    plays = plays[plays["posteam"].notna() & plays["play_type"].notna()].reset_index(drop=True)

    kind = np.full(len(plays), "other", dtype=object)
    t = plays["play_type"]
    live = t.isin(["pass", "run"])
    dropback = live & ((t == "pass") | (plays["qb_scramble"] == 1))
    kind[live & ~dropback] = "rush"
    kind[dropback] = "dropback"
    kind[plays["qb_kneel"] == 1] = "kneel"
    kind[plays["qb_spike"] == 1] = "spike"
    for name in ("punt", "field_goal", "kickoff", "extra_point"):
        kind[t == name] = name
    kind[(t == "no_play") & (plays["penalty"] == 1)] = "penalty"
    kind[(plays["two_point_attempt"] == 1)] = "two_point"
    plays["kind"] = kind

    # What the next play in the same game looked like.
    same = plays["game_id"].eq(plays["game_id"].shift(-1))
    same_half = same & plays["game_half"].eq(plays["game_half"].shift(-1))
    plays["elapsed"] = (plays["game_seconds_remaining"] - plays["game_seconds_remaining"].shift(-1)).where(same_half)
    plays["next_posteam"] = plays["posteam"].shift(-1).where(same)
    plays["next_yardline"] = plays["yardline_100"].shift(-1).where(same)
    plays["next_kind"] = plays["kind"].shift(-1).where(same)
    return plays
