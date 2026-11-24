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


def _replayed_games(plays: pd.DataFrame, model: GameModel, rng: np.random.Generator, repeats: int, run_quality: float):
    """Real games with their betting lines, and the quality each offence needs for the
    simulator to average what the line implies. Returns (home rows, away rows, home
    implied, away implied, quality), the last three repeated `repeats` times."""
    need = {"spread_line", "total_line", "home_team"}
    if not need <= set(plays.columns) or plays["total_line"].isna().all():
        raise ValueError("the play-by-play data has no betting lines")
    lines = plays.drop_duplicates("game_id").set_index("game_id")[["home_team", "spread_line", "total_line"]]
    box = real_box_scores(plays).reset_index().join(lines, on="game_id").dropna(subset=["total_line", "spread_line"])
    home = box["posteam"] == box["home_team"]
    box["implied"] = box["total_line"] / 2 + np.where(home, 0.5, -0.5) * box["spread_line"]
    box["opp_implied"] = box["total_line"] - box["implied"]
    box = box[box.groupby("game_id")["posteam"].transform("size") == 2]
    one, two = box[home[box.index]].set_index("game_id"), box[~home[box.index]].set_index("game_id")
    two = two.reindex(one.index)
    grid = np.linspace(-0.8, 0.8, 5)
    own, other = zip(*[[simulate_game(model, 4000, rng, quality=(q, 0.0), run_quality=run_quality)["points"][side].mean()
                        for side in (0, 1)] for q in grid])
    a, b, base = np.polyfit(grid, own, 1)[0], np.polyfit(grid, other, 1)[0], float(np.interp(0.0, grid, own))
    i0, i1 = np.repeat(one["implied"].to_numpy(), repeats), np.repeat(two["implied"].to_numpy(), repeats)
    det = a * a - b * b
    quality = np.vstack([a * (i0 - base) - b * (i1 - base), a * (i1 - base) - b * (i0 - base)]) / det
    return one, two, i0, i1, quality


def by_expected_points(plays: pd.DataFrame, model: GameModel, run_quality: float = 1.0, repeats: int = 10,
                       seed: int | None = None) -> pd.DataFrame:
    """Rushing and passing for teams grouped by the points the betting line expected of
    them, real against the simulator replaying the same games. Strong offences should gain
    a lot more per pass but only a little more per carry."""
    rng = np.random.default_rng(seed)
    one, two, i0, i1, quality = _replayed_games(plays, model, rng, repeats, run_quality)
    real = pd.concat([one.reset_index(), two.reset_index()], ignore_index=True)
    stats = simulate_game(model, len(i0), rng, quality=quality, run_quality=run_quality)
    sim = pd.DataFrame({c: np.concatenate([stats[c][0], stats[c][1]]) for c in BOX})
    sim["implied"] = np.concatenate([i0, i1])
    edges = [0, 18, 21, 24, 27, 30, 60]
    rows = {}
    for label, frame in (("real", real), ("sim", sim)):
        group = frame.groupby(pd.cut(frame["implied"], edges), observed=True)
        rows[label] = pd.DataFrame({
            "points": group["points"].mean(), "yards per carry": group["rush_yards"].sum() / group["rush_att"].sum(),
            "rushing yards": group["rush_yards"].mean(), "rushing TDs": group["rush_td"].mean(),
            "yards per pass": group["pass_yards"].sum() / group["pass_att"].sum(), "passing yards": group["pass_yards"].mean(),
            "passing TDs": group["pass_td"].mean(), "team-games": group.size() / (repeats if label == "sim" else 1),
        })
    order = list(rows["real"].columns)
    table = pd.concat(rows, axis=1).swaplevel(axis=1)
    table = table[[(column, side) for column in order for side in ("real", "sim")]]
    table.index = [f"{int(i.left)}-{int(i.right)}" if i.right < 60 else f"{int(i.left)}+" for i in table.index]
    table.index.name = "expected points"
    return table.round(2)


def around_the_line(plays: pd.DataFrame, model: GameModel, game_plan_sd: float, repeats: int = 30, seed: int | None = None,
                    run_quality: float = 1.0) -> pd.DataFrame:
    """How much real games vary around what the betting line expected, against the
    simulator playing the same games (each one `repeats` times) with and without the
    game-plan variation. This is the check that the simulator's swings are the right size:
    scoring, and how a team splits its plays between pass and run."""
    rng = np.random.default_rng(seed)
    one, two, i0, i1, quality = _replayed_games(plays, model, rng, repeats, run_quality)

    def spreads(frame: pd.DataFrame, team_season: np.ndarray | None) -> dict:
        frame = frame.assign(pass_share=frame["dropbacks"] / frame["plays"].clip(lower=1))
        x = np.column_stack([np.ones(len(frame)), frame["implied"], frame["opp_implied"]])
        out = {}
        for label, column in (("Points", "points"), ("Share of plays that are dropbacks", "pass_share"), ("Dropbacks", "dropbacks"),
                              ("Rushes", "rush_att"), ("Passing yards", "pass_yards"), ("Rushing yards", "rush_yards")):
            y = frame[column].to_numpy(float)
            miss = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
            if team_season is not None:      # a team's own habits are known beforehand; only its game-to-game swing counts
                count = pd.Series(team_season).map(pd.Series(team_season).value_counts()).to_numpy()
                miss = (miss - pd.Series(miss).groupby(team_season).transform("mean").to_numpy()) * np.sqrt(count / np.maximum(count - 1, 1))
            out[label + ": spread (sd)"] = miss.std()
        miss = (frame["points"] - frame["implied"]).to_numpy()
        half = len(frame) // 2
        out["Game total points: spread around the line (sd)"] = (miss[:half] + miss[half:]).std()
        out["Winning margin: spread around the line (sd)"] = (miss[:half] - miss[half:]).std()
        out["Teams 14+ points over the line (%)"] = 100 * (miss >= 14).mean()
        out["Teams 14+ points under the line (%)"] = 100 * (miss <= -14).mean()
        out["Teams passing for 350+ (%)"] = 100 * (frame["pass_yards"] >= 350).mean()
        return out

    real = pd.concat([one.reset_index(), two.reset_index()], ignore_index=True)
    table = {"real": spreads(real, (real["posteam"] + real["game_id"].str[:4]).to_numpy())}
    for label, sd in (("sim, no game plan", 0.0), ("sim, with game plan", game_plan_sd)):
        stats = simulate_game(model, len(i0), rng, quality=quality, pass_shift=sd * rng.standard_normal(quality.shape),
                              run_quality=run_quality)
        sim = pd.DataFrame({c: np.concatenate([stats[c][0], stats[c][1]]) for c in BOX})
        sim["implied"], sim["opp_implied"] = np.concatenate([i0, i1]), np.concatenate([i1, i0])
        table[label] = spreads(sim, None)
    return pd.DataFrame(table).round(3)
