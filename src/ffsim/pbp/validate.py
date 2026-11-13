"""Compare simulated games with real ones."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import simulate_game
from .models import GameModel

BOX = ["points", "plays", "dropbacks", "pass_att", "completions", "pass_yards", "pass_td", "interceptions", "sacks",
       "rush_att", "rush_yards", "rush_td", "fumbles_lost", "punts", "fg_made"]


def real_box_scores(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, with the same statistics the simulator records."""
    live = plays[plays["kind"].isin(["dropback", "rush"]) & plays["down"].between(1, 4)]
    complete = live["complete_pass"] == 1
    ran = (live["kind"] == "rush") | (live["qb_scramble"] == 1)
    scored = (live["touchdown"] == 1) & (live["td_team"] == live["posteam"])
    frame = pd.DataFrame({
        "game_id": live["game_id"], "posteam": live["posteam"], "plays": 1, "dropbacks": live["kind"] == "dropback",
        "pass_att": (live["pass_attempt"] == 1) & (live["sack"] != 1) & (live["qb_scramble"] != 1),
        "completions": complete, "pass_yards": live["yards_gained"].where(complete, 0), "pass_td": scored & complete,
        "interceptions": live["interception"] == 1, "sacks": live["sack"] == 1, "rush_att": ran,
        "rush_yards": live["yards_gained"].where(ran, 0), "rush_td": scored & ran, "fumbles_lost": live["fumble_lost"] == 1,
    })
    key = ["game_id", "posteam"]
    box = frame.groupby(key).sum().astype(float)
    count = lambda mask: plays[mask].groupby(key).size().reindex(box.index).fillna(0.0)
    box["punts"] = count(plays["kind"] == "punt")
    box["fg_made"] = count((plays["kind"] == "field_goal") & (plays["field_goal_result"] == "made"))
    # Final scores: the larger of the scores a team was shown with, as offence or defence.
    as_off = plays.groupby(["game_id", "posteam"])["posteam_score"].max()
    as_def = plays.groupby(["game_id", "defteam"])["defteam_score"].max().rename_axis(key)
    final = pd.concat([as_off, as_def], axis=1).max(axis=1)
    box["points"] = final.reindex(box.index)
    opponent = box.reset_index().groupby("game_id")["points"].transform("sum").to_numpy() - box["points"].to_numpy()
    box["opp_points"] = opponent
    return box.dropna(subset=["points"])


def simulated_box_scores(model: GameModel, n_sims: int, seed: int | None = None, **kwargs) -> pd.DataFrame:
    stats = simulate_game(model, n_sims, np.random.default_rng(seed), **kwargs)
    box = pd.DataFrame({c: np.concatenate([stats[c][0], stats[c][1]]) for c in BOX})
    box["opp_points"] = np.concatenate([stats["points"][1], stats["points"][0]])
    return box


def compare(real: pd.DataFrame, sim: pd.DataFrame):
    """Return (table of per-team-game statistics, dict of game-flow checks)."""
    table = pd.DataFrame({
        "real mean": real[BOX].mean(), "sim mean": sim[BOX].mean(), "real sd": real[BOX].std(), "sim sd": sim[BOX].std(),
        "real 95th": real[BOX].quantile(0.95), "sim 95th": sim[BOX].quantile(0.95),
    }).round(2)
    both = lambda f: (round(float(f(real)), 3), round(float(f(sim)), 3))
    checks = {
        "Game total points, mean": both(lambda d: (d["points"] + d["opp_points"]).mean()),
        "Game total points, spread (sd)": both(lambda d: (d["points"] + d["opp_points"]).std()),
        "Winning margin, spread (sd)": both(lambda d: (d["points"] - d["opp_points"]).std()),
        "Share of teams scoring 35+": both(lambda d: (d["points"] >= 35).mean()),
        "Share of teams scoring 10 or fewer": both(lambda d: (d["points"] <= 10).mean()),
        "Share of teams passing for 350+": both(lambda d: (d["pass_yards"] >= 350).mean()),
        "Correlation: passing yards with opponent's points": both(lambda d: d["pass_yards"].corr(d["opp_points"])),
        "Correlation: dropbacks with opponent's points": both(lambda d: d["dropbacks"].corr(d["opp_points"])),
        "Correlation: rushes with own lead": both(lambda d: d["rush_att"].corr(d["points"] - d["opp_points"])),
    }
    return table, pd.DataFrame(checks, index=["real", "sim"]).T
