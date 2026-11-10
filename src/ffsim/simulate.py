"""Simulate a slate: correlated fantasy scores for every player, game by game."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .distributions import fit_players, scores_from_normals


def pair_correlation(pos_a: str, pos_b: str, table: dict) -> float:
    return table.get(f"{pos_a}-{pos_b}", table.get(f"{pos_b}-{pos_a}", 0.0))


def game_correlation(game: pd.DataFrame, corr: dict) -> np.ndarray:
    """Correlation matrix for the players in one game."""
    pos = game["position"].to_numpy()
    team = game["team"].to_numpy()
    ref = game["position"].map(corr["depth_reference"]).to_numpy(float)
    depth = np.clip(game["mean"].to_numpy(float) / ref, corr["min_depth_factor"], 1.0)
    n = len(game)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            table = corr["same_team"] if team[i] == team[j] else corr["opponent"]
            m[i, j] = m[j, i] = pair_correlation(pos[i], pos[j], table) * np.sqrt(depth[i] * depth[j])
    return _nearest_valid(m)


def _nearest_valid(m: np.ndarray) -> np.ndarray:
    """Nudge a matrix so it is a valid correlation matrix."""
    vals, vecs = np.linalg.eigh(m)
    if vals.min() > 1e-8:
        return m
    fixed = (vecs * np.maximum(vals, 1e-6)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def simulate_slate(players: pd.DataFrame, settings: dict, n_sims: int, seed: int | None = None):
    """Return (scores, fit). scores is a float32 array of shape (players, n_sims)."""
    fit = fit_players(players, settings)
    rng = np.random.default_rng(seed)
    z = np.empty((len(players), n_sims))
    for _, game in players.groupby("game", sort=False):
        chol = np.linalg.cholesky(game_correlation(game, settings["correlation"]))
        z[game.index.to_numpy()] = chol @ rng.standard_normal((len(game), n_sims))
    return scores_from_normals(fit, z).astype(np.float32), fit


def fit_report(players: pd.DataFrame, fit: pd.DataFrame, scores: np.ndarray, settings: dict) -> pd.DataFrame:
    """Compare the simulated mean, floor and ceiling with the inputs."""
    lo, hi = settings["floor_percentile"] * 100, settings["ceiling_percentile"] * 100
    report = players[["name", "position", "team", "salary", "mean", "floor", "ceiling"]].copy()
    report["sim_mean"] = scores.mean(axis=1)
    report["sim_floor"] = np.percentile(scores, lo, axis=1)
    report["sim_ceiling"] = np.percentile(scores, hi, axis=1)
    report["sim_p99"] = np.percentile(scores, 99, axis=1)
    report["used_fallback"] = fit["used_fallback"].to_numpy()
    report["shape_adjusted"] = fit["shape_adjusted"].to_numpy()
    return report.round(2)


def save_simulation(path: str, players: pd.DataFrame, scores: np.ndarray) -> None:
    np.savez_compressed(path, scores=scores, players=np.array(players.to_json(orient="split")))


def load_simulation(path: str):
    data = np.load(path)
    players = pd.read_json(io.StringIO(str(data["players"])), orient="split")
    return players, data["scores"]
