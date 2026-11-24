"""The common build, played differently.

Find the cores the field crowds onto (the same three to five popular players in many
lineups), then, for each core, the finishes the field underplays relative to how often
they win: which other players go with it, which flex position, which bring-backs. The
idea is leverage without dart throws: keep the strong core everyone else has, and be
different in how you complete it.

Everything is measured on the simulated field. In each simulated slate the field's top 1%
is found; a finish's leverage is how much more often core lineups that use it land there
than core lineups in general, and its share is how often the field's core lineups use it.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

FLEX_COUNTS = {"RB": 3, "WR": 4, "TE": 2}       # a position this many times means it fills the flex


def membership(lineups: np.ndarray, n_players: int) -> np.ndarray:
    inside = np.zeros((len(lineups), n_players), dtype=bool)
    inside[np.arange(len(lineups))[:, None], lineups] = True
    return inside


def find_cores(lineups: np.ndarray, players: pd.DataFrame, count: int = 6, pool: int = 14,
               min_share: dict | None = None) -> list[dict]:
    """The player combinations the field crowds onto, largest first.

    Combinations of 3 to 5 of the `pool` most-owned non-defense players are counted in the
    field. A combination counts as a core when enough of the field holds it (by default
    0.8% for five players, 2% for four, 5% for three). Cores are taken by share, skipping
    any that differ from one already chosen by a single player, so the list is not the same
    core several times over."""
    min_share = min_share or {5: 0.008, 4: 0.02, 3: 0.05}
    inside = membership(lineups, len(players))
    own = players["ownership"].fillna(0.0).to_numpy(float)
    eligible = np.flatnonzero(players["position"].to_numpy() != "DST")
    top = eligible[np.argsort(-own[eligible])][:pool]
    found = []
    for size in (5, 4, 3):
        for combo in itertools.combinations(top, size):
            share = inside[:, list(combo)].all(axis=1).mean()
            if share >= min_share[size]:
                found.append((size, share, tuple(sorted(combo))))
    found.sort(key=lambda item: (-item[0], -item[1]))
    chosen: list[dict] = []
    for size, share, combo in found:
        if any(len(set(combo) & set(c["players"])) >= min(size, len(c["players"])) - 1 for c in chosen):
            continue
        chosen.append({"players": combo, "share": share})
        if len(chosen) == count:
            break
    return chosen


def lineup_features(lineups: np.ndarray, players: pd.DataFrame) -> dict:
    pos = players["position"].to_numpy()[lineups]
    team = players["team"].to_numpy()[lineups]
    opp = players["opponent"].to_numpy()
    flex = np.full(len(lineups), "", dtype=object)
    for p, k in FLEX_COUNTS.items():
        flex[(pos == p).sum(axis=1) == k] = p
    qb_col = (pos == "QB").argmax(axis=1)
    qb = lineups[np.arange(len(lineups)), qb_col]
    skill = np.isin(pos, ["RB", "WR", "TE"])
    bring_backs = (skill & (team == opp[qb][:, None])).sum(axis=1)
    return {"flex": flex, "qb": qb, "bring_backs": bring_backs}


def scan_finishes(lineups: np.ndarray, players: pd.DataFrame, scores: np.ndarray, cores: list[dict],
                  sims: int = 2000, chunk: int = 200, min_projection: float = 6.0, min_lineups: int = 8) -> pd.DataFrame:
    """For each core, every way the field finishes it, with its share and leverage.

    Finishes are: each other player (projected at least `min_projection`, defenses
    always), the flex position, and the number of bring-back players from the
    quarterback's opponent. A finish needs `min_lineups` core lineups to be judged."""
    sims = min(sims, scores.shape[1])
    inside = membership(lineups, len(players))
    feats = lineup_features(lineups, players)
    name = players["name"].str.strip().to_numpy()
    pos, team, opp = (players[c].to_numpy() for c in ("position", "team", "opponent"))
    mean = players["mean"].to_numpy(float)
    own = players["ownership"].fillna(0.0).to_numpy(float)

    plans = []
    for number, core in enumerate(cores):
        rows = np.flatnonzero(inside[:, list(core["players"])].all(axis=1))
        if len(rows) < 5 * min_lineups:
            continue
        qbs = [p for p in core["players"] if pos[p] == "QB"]
        labels, kinds, columns, extra = [], [], [], []
        for p in np.flatnonzero(inside[rows].sum(axis=0) >= min_lineups):
            if p in core["players"] or (pos[p] != "DST" and mean[p] < min_projection):
                continue
            kind = "defense" if pos[p] == "DST" else "player"
            if qbs and team[p] == opp[qbs[0]] and pos[p] != "DST":
                kind = "bring-back"
            labels.append(name[p]); kinds.append(kind); columns.append(inside[rows, p])
            extra.append((mean[p], own[p], int(players["salary"].iloc[p])))
        for p in ("RB", "WR", "TE"):
            labels.append(f"{p} in the flex"); kinds.append("flex"); columns.append(feats["flex"][rows] == p)
            extra.append((np.nan, np.nan, np.nan))
        for k, text in ((0, "no bring-back"), (1, "1 bring-back"), (2, "2+ bring-backs")):
            have = feats["bring_backs"][rows]
            labels.append(text); kinds.append("bring-backs")
            columns.append(have >= 2 if k == 2 else have == k)
            extra.append((np.nan, np.nan, np.nan))
        plans.append((number, core, rows, labels, kinds, np.column_stack(columns), extra))

    hits = [np.zeros(len(plan[3])) for plan in plans]
    core_hits = np.zeros(len(plans))
    for start in range(0, sims, chunk):
        block = scores[:, start: min(start + chunk, sims)]
        total = np.zeros((len(lineups), block.shape[1]), dtype=np.float32)
        for slot in range(lineups.shape[1]):
            total += block[lineups[:, slot]]
        cut = np.quantile(total, 0.99, axis=0)
        top = total >= cut
        for i, (_, _, rows, _, _, has, _) in enumerate(plans):
            wins = top[rows].sum(axis=1).astype(float)
            core_hits[i] += wins.sum()
            hits[i] += has.T.astype(float) @ wins

    out = []
    for i, (number, core, rows, labels, kinds, has, extra) in enumerate(plans):
        used = has.sum(axis=0)
        base = core_hits[i] / (len(rows) * sims)
        rate = np.where(used > 0, hits[i] / np.maximum(used, 1) / sims, np.nan)
        for j, label in enumerate(labels):
            if used[j] < min_lineups:
                continue
            out.append({
                "core": number + 1, "core_players": " + ".join(name[list(core["players"])]),
                "core_share_pct": 100 * core["share"], "finish": label, "kind": kinds[j],
                "share_of_core_pct": 100 * used[j] / len(rows), "top1_pct": 100 * rate[j],
                "core_top1_pct": 100 * base, "leverage": rate[j] / base if base > 0 else np.nan,
                "projection": extra[j][0], "ownership": extra[j][1], "salary": extra[j][2],
            })
    return pd.DataFrame(out)


def uncommon_finishes(table: pd.DataFrame, max_share: float = 15.0, min_leverage: float = 1.15) -> pd.DataFrame:
    """The finishes worth a look: used in few of the field's core lineups, yet winning
    more often than the core does."""
    picked = table[(table["share_of_core_pct"] <= max_share) & (table["leverage"] >= min_leverage)]
    return picked.sort_values(["core", "leverage"], ascending=[True, False])
