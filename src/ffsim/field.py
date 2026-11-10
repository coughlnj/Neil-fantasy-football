"""Sample the lineups opponents enter, matched to projected ownership."""
from __future__ import annotations

import numpy as np
import pandas as pd


class LineupSampler:
    """Draws many DraftKings classic lineups at once, weighted per player.

    Lineups favour pass catchers who share a team with the quarterback (stacks), avoid a
    defense facing that quarterback, and must land inside a salary window.
    """

    def __init__(self, players: pd.DataFrame, settings: dict):
        self.cfg = settings["field"]
        self.cap = settings["site"]["salary_cap"]
        self.salary = players["salary"].to_numpy(float)
        teams = {t: i for i, t in enumerate(sorted(set(players["team"]) | set(players["opponent"])))}
        self.team = players["team"].map(teams).to_numpy()
        self.opp = players["opponent"].map(teams).to_numpy()
        pos = players["position"].to_numpy()
        self.idx = {p: np.flatnonzero(pos == p) for p in ("QB", "RB", "WR", "TE", "DST")}
        self.flex_bonus = {p: np.log(self.cfg["flex_weight"][p]) for p in ("RB", "WR", "TE")}

    def _draw(self, logw: np.ndarray, batch: int, rng: np.random.Generator) -> np.ndarray:
        def keys(position: str) -> np.ndarray:
            idx = self.idx[position]
            return logw[idx][None, :] + rng.gumbel(size=(batch, len(idx)))

        qb = self.idx["QB"][keys("QB").argmax(axis=1)]
        q_team, q_opp = self.team[qb][:, None], self.opp[qb][:, None]

        picks, flex_keys, flex_ids = [qb[:, None]], [], []
        for position, count in (("RB", 2), ("WR", 3), ("TE", 1)):
            idx = self.idx[position]
            k = keys(position)
            if position != "RB":
                k += np.log(self.cfg["stack_boost"]) * (self.team[idx][None, :] == q_team)
                k += np.log(self.cfg["bringback_boost"]) * (self.team[idx][None, :] == q_opp)
            order = np.argsort(-k, axis=1)[:, : count + 1]
            picks.append(idx[order[:, :count]])
            flex_ids.append(idx[order[:, count]])
            flex_keys.append(np.take_along_axis(k, order[:, count : count + 1], axis=1)[:, 0] + self.flex_bonus[position])
        best = np.argmax(np.stack(flex_keys, axis=1), axis=1)
        picks.append(np.stack(flex_ids, axis=1)[np.arange(batch), best][:, None])

        dst_idx = self.idx["DST"]
        k = keys("DST") + np.log(self.cfg["dst_vs_qb_penalty"]) * (self.opp[dst_idx][None, :] == q_team)
        picks.append(dst_idx[k.argmax(axis=1)][:, None])

        lineups = np.concatenate(picks, axis=1)
        return lineups, self.salary[lineups].sum(axis=1)

    def salary_tilt(self, logw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Lean the weights toward cheaper or dearer players so that a typical lineup
        lands in the salary window. Returns the adjusted weights."""
        target = (self.cfg["min_salary"] + self.cap) / 2
        lo, hi = -3.0, 3.0  # change in log-weight per $1,000 of salary
        for _ in range(9):
            mid = (lo + hi) / 2
            _, total = self._draw(logw + mid * self.salary / 1000, 3000, rng)
            lo, hi = (mid, hi) if np.median(total) < target else (lo, mid)
        return logw + (lo + hi) / 2 * self.salary / 1000

    def sample(self, logw: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return n lineups as sorted player indices, shape (n, 9)."""
        found, have, tries = [], 0, 0
        while have < n:
            got, total = self._draw(logw, 20000, rng)
            got = got[(total <= self.cap) & (total >= self.cfg["min_salary"])]
            found.append(got)
            have += len(got)
            tries += 1
            if tries >= 25 and have < tries * 20:
                raise RuntimeError(
                    "Could not build lineups inside the salary window. Check salaries and "
                    "ownership, or lower field.min_salary."
                )
        return np.sort(np.concatenate(found)[:n], axis=1)


def target_ownership(players: pd.DataFrame, settings: dict) -> np.ndarray:
    """Ownership as fractions, rescaled so each position fills its roster slots."""
    own = players["ownership"].to_numpy(float)
    if np.isnan(own).all():
        raise ValueError("The projections have no ownership column, which the field needs.")
    own = np.nan_to_num(own, nan=0.0).clip(min=0.0) / 100.0
    pos = players["position"].to_numpy()
    for p in ("QB", "DST"):
        own[pos == p] /= own[pos == p].sum()
    flex = np.isin(pos, settings["site"]["flex_positions"])
    own[flex] *= 7.0 / own[flex].sum()
    return own


def sample_field(players: pd.DataFrame, settings: dict, n: int, seed: int | None = None):
    """Sample n opponent lineups. Returns (lineups, realised ownership fractions)."""
    rng = np.random.default_rng(seed)
    sampler = LineupSampler(players, settings)
    target = target_ownership(players, settings)
    active = target > 0
    logw = np.where(active, np.log(np.where(active, target, 1.0)), -np.inf)
    cfg = settings["field"]
    start = logw.copy()
    best, best_gap = logw.copy(), np.inf
    # Nudge the weights until sampled ownership matches the target. Adjustments are
    # capped, and the best round is kept, so targets that cannot all be met at once
    # (given salaries) settle on the closest achievable field instead of running away.
    for _ in range(cfg["calibration_rounds"]):
        try:
            trial = sampler.sample(sampler.salary_tilt(logw, rng), cfg["calibration_lineups"], rng)
        except RuntimeError:
            break
        seen = np.bincount(trial.ravel(), minlength=len(players)) / len(trial)
        gap = np.abs(seen - target).max()
        if gap < best_gap:
            best, best_gap = logw.copy(), gap
        step = np.log(target[active]) - np.log(np.maximum(seen[active], 0.2 / len(trial)))
        logw[active] = np.clip(logw[active] + 0.8 * np.clip(step, -2.0, 2.0), start[active] - 3.0, start[active] + 3.0)
    lineups = sampler.sample(sampler.salary_tilt(best, rng), n, rng)
    realised = np.bincount(lineups.ravel(), minlength=len(players)) / len(lineups)
    return lineups, realised
