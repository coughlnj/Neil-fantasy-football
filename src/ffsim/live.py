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


HIDDEN = "LOCKED"          # how a mid-game export shows a player whose game has not started
MIN_SALARY = {"QB": 4000, "RB": 4000, "WR": 3000, "TE": 2500, "DST": 2000}


def add_unlisted_players(entries: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """Append any player who appears in the standings but not on the slate (a late
    addition by DraftKings), with no projection and his position taken from his slot."""
    known = set(slate["name"].map(normalise_name))
    extra = {}
    for slot in SLOTS:
        for name in entries[slot].unique():
            if name != HIDDEN and normalise_name(name) not in known and name not in extra:
                extra[name] = SLOT_POSITIONS[slot][0]
    if not extra:
        return slate
    rows = pd.DataFrame({"name": list(extra), "position": list(extra.values()), "team": "UNK", "opponent": "UNK",
                         "salary": [MIN_SALARY[p] for p in extra.values()], "dk_id": "", "kickoff": "", "game": "unlisted"})
    return pd.concat([slate, rows], ignore_index=True)


def lineup_indices(entries: pd.DataFrame, slate: pd.DataFrame) -> np.ndarray:
    """Each entry's nine players as row numbers into the slate, in roster-slot order.
    A hidden player (game not started) is -1."""
    index = {k: i for i, k in enumerate(slate["name"].map(normalise_name))}
    cols = []
    for slot in SLOTS:
        names = entries[slot]
        found = names.map(normalise_name).map(index)
        found = found.where(names != HIDDEN, -1)
        if found.isna().any():
            raise ValueError("players in the standings are not on the slate: " + ", ".join(sorted(set(names[found.isna()]))[:10]))
        cols.append(found.astype(int))
    return np.column_stack(cols)


def game_time_left(entries: pd.DataFrame, lineups: np.ndarray, slate: pd.DataFrame) -> pd.Series:
    """Fraction of each game still to play (1 = not started, 0 = finished).

    DraftKings reports each entry's player-minutes remaining: 60 for every player whose
    game has not started. Across all entries that pins down the minutes left in each
    game. Hidden players are in games that have not started, so count 60 each."""
    games = slate["game"].to_numpy()
    names = sorted(set(games))
    column = {g: i for i, g in enumerate(names)}
    game_of = np.vectorize(column.get)(games)
    counts = np.zeros((len(lineups), len(names)))
    hidden = lineups < 0
    for slot in range(lineups.shape[1]):
        shown = ~hidden[:, slot]
        np.add.at(counts, (np.flatnonzero(shown), game_of[lineups[shown, slot]]), 1)
    minutes = entries["minutes_remaining"].to_numpy(float) - 60.0 * hidden.sum(axis=1)
    known = ~np.isnan(minutes)
    played = counts[known].sum(axis=0) > 0
    fitted = np.full(len(names), 60.0)                      # a game nobody is shown in has not started
    fitted[played] = np.linalg.lstsq(counts[known][:, played], minutes[known], rcond=None)[0]
    left = np.clip(fitted / 60.0, 0.0, 1.0)
    left = np.where(left > 0.99, 1.0, np.where(left < 0.01, 0.0, left))  # snap estimation noise
    return pd.Series(left.round(3), index=names)


def fill_hidden(lineups: np.ndarray, slate: pd.DataFrame, time_left: np.ndarray, weights: np.ndarray,
                rng: np.random.Generator, cap: int = 50000, low: int = 48500, stack_boost: float = 4.0) -> np.ndarray:
    """Guess opponents' hidden players. Each hidden slot gets a not-yet-started player
    of a position that fits, drawn in proportion to `weights` (projected ownership),
    favouring pass catchers who share a team with the lineup's quarterback, with no
    repeats and a total salary between `low` and the cap where that is possible."""
    out = lineups.copy()
    pos, team, salary = (slate[c].to_numpy() for c in ("position", "team", "salary"))
    open_player = (time_left >= 1.0) & (np.nan_to_num(weights) > 0)
    pools = {}
    for slot, name in enumerate(SLOTS):
        cand = np.flatnonzero(open_player & np.isin(pos, SLOT_POSITIONS[name]))
        pools[slot] = (cand, np.log(weights[cand]))
    todo = np.flatnonzero((lineups < 0).any(axis=1))
    for attempt in range(40):
        if len(todo) == 0:
            break
        out[todo] = lineups[todo]
        for slot in [0, 1, 2, 3, 4, 5, 6, 7, 8]:                 # the quarterback first, so stacks can follow
            cand, logw = pools[slot]
            rows = todo[lineups[todo, slot] < 0]
            if len(rows) == 0 or len(cand) == 0:
                continue
            keys = logw[None, :] + rng.gumbel(size=(len(rows), len(cand)))
            qb = out[rows, 0]
            if slot >= 3 and slot <= 7:
                qb_team = np.where(qb >= 0, team[np.maximum(qb, 0)], "")
                mate = (team[cand][None, :] == qb_team[:, None]) & np.isin(pos[cand], ["WR", "TE"])[None, :]
                keys += np.log(stack_boost) * mate
            taken = (out[rows][:, :, None] == cand[None, None, :]).any(axis=1)   # no repeats
            keys[taken] = -np.inf
            out[rows, slot] = cand[keys.argmax(axis=1)]
        total = salary[out[todo]].sum(axis=1)
        floor = low if attempt < 25 else 0                          # give up on using the whole cap eventually
        todo = todo[(total > cap) | (total < floor) | (out[todo] < 0).any(axis=1)]
    return out


def match_my_lineups(shown: np.ndarray, uploaded: np.ndarray) -> np.ndarray:
    """Pair each of my entries in the export (hidden players = -1) with one of the
    lineups I uploaded that agrees on every visible slot, using each uploaded lineup
    once where possible. Returns the full lineups; a row stays as shown if nothing fits."""
    from scipy.optimize import linear_sum_assignment

    fits = np.zeros((len(shown), len(uploaded)), dtype=bool)
    for i, row in enumerate(shown):
        visible = row >= 0
        fits[i] = (uploaded[:, visible] == row[visible]).all(axis=1)
        if not fits[i].any():      # same players, different slots
            fits[i] = [set(row[visible]) <= set(u) for u in uploaded]
    full = shown.copy()
    rows, cols = linear_sum_assignment(~fits)          # as many agreeing pairs as possible
    for i, j in zip(rows, cols):
        if fits[i, j]:
            full[i] = uploaded[j]
    for i in np.flatnonzero((full < 0).any(axis=1) & fits.any(axis=1)):   # more entries than lineups: reuse one
        full[i] = uploaded[np.argmax(fits[i])]
    return full


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
