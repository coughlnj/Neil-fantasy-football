"""Build candidate lineups with an optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


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
