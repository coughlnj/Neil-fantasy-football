"""Simulate a whole slate play by play, anchored to the supplied projections."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..distributions import fit_players, scores_from_normals
from . import models as M
from .players import TEAM_ALIASES, Matchup, usage_history


def team_pass_lean(plays: pd.DataFrame, model: M.GameModel, season: int | None = None, shrink: float = 250.0) -> pd.Series:
    """How much more (or less) each team passes than the league would in the same
    situations this season, as a shift in log-odds. Small samples are pulled toward zero."""
    season = int(plays["season"].max()) if season is None else season
    p = plays[(plays["season"] == season) & plays["kind"].isin(["dropback", "rush"]) & plays["down"].between(1, 4)
              & plays["yardline_100"].between(1, 99)]
    sit = M.situation(p["down"], p["ydstogo"], p["yardline_100"], p["score_differential"].fillna(0),
                      p["game_seconds_remaining"], p["half_seconds_remaining"])
    over = (p["kind"] == "dropback").to_numpy(float) - M.predict(M.pass_features(sit), model.pass_w)
    team = p["posteam"].replace(TEAM_ALIASES).to_numpy()
    lean = pd.Series(over).groupby(team).agg(["mean", "size"])
    return (lean["mean"] * lean["size"] / (lean["size"] + shrink) / 0.24).rename("pass_shift")


def simulate_slate_pbp(players: pd.DataFrame, plays: pd.DataFrame, settings: dict, n_sims: int,
                       seed: int | None = None, log=lambda message: None):
    """Play every game on the slate n_sims times.

    Returns (scores, report). scores has shape (players, n_sims), in the order of
    `players`. Each player's simulated results are scaled so their average equals his
    projection; the spread and how players move together come from the simulated games.
    report compares the simulator's own numbers with the projections."""
    rng = np.random.default_rng(seed)
    model = M.fit_game_model(plays)
    usage = usage_history(plays)
    lean = team_pass_lean(plays, model)
    scores = np.zeros((len(players), n_sims), dtype=np.float32)
    own_mean = np.zeros(len(players))
    implied = {}
    for game, group in players.groupby("game", sort=False):
        idx = group.index.to_numpy()
        matchup = Matchup(group.reset_index(drop=True), usage)
        matchup.pass_shift = np.array([lean.get(team, 0.0) for team in matchup.teams])
        matchup.game_plan_sd = float(settings.get("pbp", {}).get("game_plan_sd", 0.0))
        matchup.exit_rate = dict(settings.get("pbp", {}).get("exit_rate", {}))
        matchup.fit(model, seed=int(rng.integers(1 << 31)))
        points, box, _ = matchup.simulate(model, n_sims, rng)
        scores[idx] = points
        own_mean[idx] = points.mean(axis=1)
        for side, team in enumerate(matchup.teams):
            implied[team] = float(box["points"][side].mean())
        log(f"{game}: " + ", ".join(f"{t} {implied[t]:.1f}" for t in matchup.teams) + " points on average")

    target = players["mean"].to_numpy(float)
    is_dst = (players["position"] == "DST").to_numpy()
    used = own_mean > np.maximum(0.25 * target, 0.05)
    # Anchor: scale skill players to the projected mean, shift defences (their scores can be negative).
    scale = np.where(used & ~is_dst, target / np.maximum(own_mean, 1e-9), 1.0)
    scores *= scale[:, None].astype(np.float32)
    shift = np.where(used & is_dst, target - own_mean, 0.0)
    scores += shift[:, None].astype(np.float32)
    scores[is_dst] = np.maximum(scores[is_dst], settings["lower_bound"]["DST"])
    # Players the games never used (deep reserves) keep the projection-only distribution.
    spare = np.flatnonzero(~used & (target > 0))
    if len(spare):
        fit = fit_players(players.iloc[spare], settings)
        scores[spare] = scores_from_normals(fit, rng.standard_normal((len(spare), n_sims))).astype(np.float32)

    report = players[["name", "position", "team", "salary", "mean", "floor", "ceiling"]].copy()
    report["simulator_own_mean"] = own_mean.round(2)
    report["sim_mean"] = scores.mean(axis=1).round(2)
    lo, hi = 100 * settings["floor_percentile"], 100 * settings["ceiling_percentile"]
    report["sim_floor"] = np.percentile(scores, lo, axis=1).round(2)
    report["sim_ceiling"] = np.percentile(scores, hi, axis=1).round(2)
    report["sim_p99"] = np.percentile(scores, 99, axis=1).round(2)
    report["from_games"] = used
    report["team_points"] = players["team"].map(implied).round(1).to_numpy()
    return scores, report
