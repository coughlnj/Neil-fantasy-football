"""Build candidate lineups with an optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .contest import Contest
from .field import LineupSampler
from .roi import adjust_roi, evaluate


class LineupOptimizer:
    """Finds the highest-scoring legal DraftKings classic lineup for a set of player values."""

    def __init__(self, players: pd.DataFrame, settings: dict):
        site, cfg = settings["site"], settings["candidates"]
        pos = players["position"].to_numpy()
        team = players["team"].to_numpy()
        opp = players["opponent"].to_numpy()
        n = len(players)
        rows, lo, hi = [], [], []

        def add(row, low, high):
            rows.append(row); lo.append(low); hi.append(high)

        add(players["salary"].to_numpy(float), cfg["min_salary"], site["salary_cap"])
        add(np.ones(n), sum(site["roster"].values()) + 1, sum(site["roster"].values()) + 1)
        for p, count in site["roster"].items():
            extra = 1 if p in site["flex_positions"] else 0
            add((pos == p).astype(float), count, count + extra)
        catchers = np.isin(pos, ["WR", "TE"])
        for q in np.flatnonzero(pos == "QB"):
            if cfg["min_stack"] > 0:
                row = (catchers & (team == team[q])).astype(float)
                row[q] = -cfg["min_stack"]
                add(row, 0, np.inf)
            if not cfg["allow_dst_vs_qb"]:
                row = ((pos == "DST") & (team == opp[q])).astype(float)
                row[q] = 1.0
                add(row, 0, 1)
        self.constraint = LinearConstraint(np.array(rows), lo, hi)
        self.n = n

    def solve(self, values: np.ndarray) -> np.ndarray | None:
        result = milp(
            -values, constraints=self.constraint, integrality=np.ones(self.n), bounds=Bounds(0, 1)
        )
        if result.x is None:
            return None
        return np.flatnonzero(result.x > 0.5)


def build_candidates(
    players: pd.DataFrame, scores: np.ndarray, settings: dict, n: int, seed: int | None = None
) -> np.ndarray:
    """Up to n distinct lineups, each the best lineup for a blend of the mean projection
    and one simulated slate. `scores` should be simulations kept apart from the ones used
    to grade the lineups, so a lineup is never graded on the slate it was built for."""
    rng = np.random.default_rng(seed)
    optimizer = LineupOptimizer(players, settings)
    mean = players["mean"].to_numpy(float)
    lo, hi = settings["candidates"]["blend_range"]
    best = optimizer.solve(mean)
    if best is None:
        raise RuntimeError("No legal lineup exists for these players and salary rules.")
    found = {tuple(best)}
    order = rng.permutation(scores.shape[1])
    for attempt in range(n * 3):
        if len(found) >= n:
            break
        blend = rng.uniform(lo, hi)
        lineup = optimizer.solve((1 - blend) * mean + blend * scores[:, order[attempt % len(order)]])
        if lineup is not None:
            found.add(tuple(lineup))
    return np.array(sorted(found))


def search_candidates(
    players: pd.DataFrame,
    scores: np.ndarray,
    field: np.ndarray,
    contest: Contest,
    settings: dict,
    n_optimizer: int,
    n_sampled: int,
    n_keep: int,
    seed: int | None = None,
    log=lambda message: None,
) -> np.ndarray:
    """A wide search for strong lineups. Returns distinct lineups: the optimizer's, plus
    the best n_keep of n_sampled lineups sampled around the players that good lineups use.

    `scores` must be simulations kept apart from the ones used for final grading: they
    drive the optimizer and the screening here.
    """
    cfg = settings["candidates"]
    rng = np.random.default_rng(seed)
    optimal = build_candidates(players, scores, settings, n_optimizer, seed)
    log(f"Optimizer built {len(optimal):,} lineups.")
    if n_sampled <= 0 or n_keep <= 0:
        return optimal

    sampler_cfg = {
        "min_salary": cfg["min_salary"],
        "stack_mix": cfg["stack_mix"], "bringback_rate": cfg["bringback_rate"],
        "flex_weight": cfg["flex_weight"],
        "dst_vs_qb_penalty": 1.0 if cfg["allow_dst_vs_qb"] else 0.0,
    }
    sampler = LineupSampler(players, settings, sampler_cfg)
    screen_scores = scores[:, : cfg["screen_sims"]]
    screen_field = field[: cfg["screen_field"]]
    rounds = max(1, cfg["search_rounds"])
    best, best_value, favoured = np.empty((0, optimal.shape[1]), dtype=int), np.empty(0), optimal
    for number in range(rounds):
        usage = np.bincount(favoured.ravel(), minlength=len(players)) / len(favoured)
        logw = np.log(usage + 0.002)
        drawn = sampler.sample(sampler.salary_tilt(logw, rng), n_sampled // rounds, rng)
        drawn = np.unique(drawn, axis=0)
        value = adjust_roi(evaluate(drawn, screen_field, screen_scores, contest))
        lineups, first = np.unique(np.vstack([best, drawn]), axis=0, return_index=True)
        value = np.concatenate([best_value, value])[first]
        top = np.argsort(-value)[:n_keep]
        best, best_value = lineups[top], value[top]
        favoured = best
        log(f"Search round {number + 1}: screened {len(drawn):,} sampled lineups, "
            f"best {len(best):,} average {best_value.mean():+.0f}% on the screen.")
    return np.unique(np.vstack([optimal, best]), axis=0)
