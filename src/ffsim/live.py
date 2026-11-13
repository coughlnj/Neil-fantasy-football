"""In-game analysis: rank your lineups against the real field while games are still to
play, and suggest late swaps.

Works from a contest standings export downloaded mid-slate. Every opponent's actual
lineup and live score is known, so no simulated field is needed: only the points still
to come are simulated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contest import Contest
from .draftkings import normalise_name
from .roi import evaluate
from .simulate import simulate_slate
from .standings import SLOTS

SLOT_POSITIONS = {"QB": ["QB"], "RB1": ["RB"], "RB2": ["RB"], "WR1": ["WR"], "WR2": ["WR"], "WR3": ["WR"],
                  "TE": ["TE"], "FLEX": ["RB", "WR", "TE"], "DST": ["DST"]}


def lineup_indices(entries: pd.DataFrame, slate: pd.DataFrame) -> np.ndarray:
    """Each entry's nine players as row numbers into the slate, in roster-slot order."""
    index = {k: i for i, k in enumerate(slate["name"].map(normalise_name))}
    cols = [entries[s].map(normalise_name).map(index) for s in SLOTS]
    out = np.column_stack(cols)
    if np.isnan(out.astype(float)).any():
        missing = sorted({n for s, c in zip(SLOTS, cols) for n in entries[s][c.isna()]})
        raise ValueError("players in the standings are not on the slate: " + ", ".join(missing[:10]))
    return out.astype(int)


def game_time_left(entries: pd.DataFrame, lineups: np.ndarray, slate: pd.DataFrame) -> pd.Series:
    """Fraction of each game still to play (1 = not started, 0 = finished).

    DraftKings reports each entry's player-minutes remaining: 60 for every player whose
    game has not started. Across all entries that pins down the minutes left in each game."""
    games = slate["game"].to_numpy()
    names = sorted(set(games))
    column = {g: i for i, g in enumerate(names)}
    counts = np.zeros((len(lineups), len(names)))
    game_idx = np.vectorize(column.get)(games)[lineups]
    for slot in range(lineups.shape[1]):
        np.add.at(counts, (np.arange(len(lineups)), game_idx[:, slot]), 1)
    known = entries["minutes_remaining"].notna().to_numpy()
    minutes = np.linalg.lstsq(counts[known], entries["minutes_remaining"].to_numpy()[known], rcond=None)[0]
    left = np.clip(minutes / 60.0, 0.0, 1.0)
    left = np.where(left > 0.99, 1.0, np.where(left < 0.01, 0.0, left))  # snap estimation noise
    return pd.Series(left.round(3), index=names)


def remaining_projections(slate: pd.DataFrame, left: pd.Series) -> pd.DataFrame:
    """Projections for the points still to come. A game in progress keeps its share of
    the mean; its spread shrinks more slowly (by the square root of the time left)."""
    f = slate["game"].map(left).to_numpy(float)
    out = slate.copy()
    out["time_left"] = f
    mean = slate["mean"].fillna(0.0).to_numpy(float)
    out["mean"] = mean * f
    out["floor"] = np.maximum(mean * f - (mean - slate["floor"].to_numpy(float)) * np.sqrt(f), np.minimum(slate["floor"], 0))
    out["ceiling"] = mean * f + (slate["ceiling"].to_numpy(float) - mean) * np.sqrt(f)
    return out


def simulate_finish(slate: pd.DataFrame, live_points: np.ndarray, left: pd.Series, settings: dict,
                    n_sims: int, seed: int | None = None) -> np.ndarray:
    """Simulated final scores for every player on the slate: points so far plus
    simulated points to come. Shape (players, n_sims)."""
    rest = remaining_projections(slate, left)
    playing = (rest["time_left"] > 0) & (rest["mean"] > 0)
    scores = np.repeat(live_points[:, None].astype(np.float32), n_sims, axis=1)
    if playing.any():
        active = rest[playing].reset_index(drop=True)
        sims, _ = simulate_slate(active, settings, n_sims, seed)
        scores[np.flatnonzero(playing.to_numpy())] += sims
    return scores


def grade_live(mine: np.ndarray, field: np.ndarray, scores: np.ndarray, contest: Contest, keep_payouts: bool = False):
    """Expected prize for each of `mine` against the real field's lineups."""
    real = Contest(contest.name, contest.entry_fee, len(field) + 2, contest.payouts)
    return evaluate(np.sort(mine, axis=1), np.sort(field, axis=1), scores, real, keep_payouts=keep_payouts)


def swap_options(lineup: np.ndarray, slate: pd.DataFrame, time_left: np.ndarray, cap: int, min_mean: float) -> list[tuple[int, int]]:
    """Legal single swaps for one lineup: (slot number, incoming player). Only players
    whose game has not started can leave or come in, and only into a slot their position fits."""
    pos, salary, mean = (slate[c].to_numpy() for c in ("position", "salary", "mean"))
    spare = cap - salary[lineup].sum()
    open_players = np.flatnonzero((time_left >= 1.0) & (np.nan_to_num(mean) >= min_mean))
    options = []
    for slot, player in enumerate(lineup):
        if time_left[player] < 1.0:
            continue
        allowed = SLOT_POSITIONS[SLOTS[slot]]
        for candidate in open_players:
            if candidate in lineup or pos[candidate] not in allowed:
                continue
            if salary[candidate] - salary[player] <= spare:
                options.append((slot, int(candidate)))
    return options


def suggest_swaps(mine: np.ndarray, field: np.ndarray, scores: np.ndarray, slate: pd.DataFrame, time_left: np.ndarray,
                  contest: Contest, cap: int = 50000, min_mean: float = 4.0, rounds: int = 2, confidence: float = 2.0):
    """Best late swaps for each lineup, found on half the simulations and confirmed on
    the other half so a swap is not recommended just for getting lucky.

    Returns (new lineups, table of swaps). Each round makes at most one swap per lineup.
    A swap is made only when its gain on the confirming half exceeds `confidence`
    standard errors; smaller positive gains are listed with made=False and left alone."""
    half = scores.shape[1] // 2
    search, confirm = scores[:, :half], scores[:, half:]
    current = mine.copy()
    swaps = []
    for _ in range(rounds):
        owner, trial, detail = [], [], []
        for i, lineup in enumerate(current):
            for slot, candidate in swap_options(lineup, slate, time_left, cap, min_mean):
                changed = lineup.copy()
                changed[slot] = candidate
                owner.append(i); trial.append(changed); detail.append((slot, candidate))
        if not trial:
            break
        owner, trial = np.array(owner), np.array(trial)
        base = grade_live(current, field, search, contest)["roi_pct"].to_numpy()
        gain = grade_live(trial, field, search, contest)["roi_pct"].to_numpy() - base[owner]
        best = pd.Series(gain).groupby(owner).idxmax()
        best = best[gain[best.to_numpy()] > 0]
        if best.empty:
            break
        chosen = best.to_numpy()
        _, before = grade_live(current[owner[chosen]], field, confirm, contest, keep_payouts=True)
        _, after = grade_live(trial[chosen], field, confirm, contest, keep_payouts=True)
        diff = after - before
        made = False
        for k, t in enumerate(chosen):
            mean_gain, se = float(diff[k].mean()), float(diff[k].std() / np.sqrt(diff.shape[1]))
            if mean_gain <= 0:
                continue
            i, (slot, candidate) = owner[t], detail[t]
            sure = mean_gain > confidence * se
            swaps.append({"lineup": int(i), "slot": SLOTS[slot], "out": int(current[i][slot]), "in": candidate,
                          "gain": mean_gain, "gain_se": se, "made": sure})
            if sure:
                current[i] = trial[t]
                made = True
        if not made:
            break
    return current, pd.DataFrame(swaps, columns=["lineup", "slot", "out", "in", "gain", "gain_se", "made"])
