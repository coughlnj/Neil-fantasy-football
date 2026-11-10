"""Grade candidate lineups against a simulated field."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contest import Contest

SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]


def lineup_scores(lineups: np.ndarray, scores: np.ndarray) -> np.ndarray:
    total = np.zeros((len(lineups), scores.shape[1]), dtype=np.float32)
    for slot in range(lineups.shape[1]):
        total += scores[lineups[:, slot]]
    return total


def evaluate(
    candidates: np.ndarray, field: np.ndarray, scores: np.ndarray, contest: Contest, chunk: int = 200
) -> pd.DataFrame:
    """Expected ROI and finish rates for each candidate, over every simulated slate."""
    n_field, n_sims = len(field), scores.shape[1]
    scale = (contest.entries - 1) / (n_field + 1)
    paid = len(contest.payouts)
    total = np.zeros(len(candidates))
    total_sq = np.zeros(len(candidates))
    wins = np.zeros(len(candidates))
    top1 = np.zeros(len(candidates))
    top01 = np.zeros(len(candidates))
    cashes = np.zeros(len(candidates))
    dupes = np.zeros(len(candidates))

    for start in range(0, n_sims, chunk):
        block = scores[:, start : start + chunk]
        field_scores = np.sort(lineup_scores(field, block), axis=0)
        cand_scores = lineup_scores(candidates, block)
        for col in range(block.shape[1]):
            left = np.searchsorted(field_scores[:, col], cand_scores[:, col], side="left")
            right = np.searchsorted(field_scores[:, col], cand_scores[:, col], side="right")
            above, tied = n_field - right, right - left
            prize = contest.expected_payout(above, tied, n_field)
            rank = (above + 0.5) * scale + 1
            total += prize
            total_sq += prize**2
            wins += (above == 0) / np.maximum((tied + 1) * scale, 1.0)
            top1 += rank <= 0.01 * contest.entries
            top01 += rank <= 0.001 * contest.entries
            cashes += rank <= paid
            if start == 0 and col == 0:
                dupes = tied * scale

    mean_prize = total / n_sims
    variance = np.maximum(total_sq / n_sims - mean_prize**2, 0)
    return pd.DataFrame(
        {
            "roi_pct": 100 * (mean_prize / contest.entry_fee - 1),
            "roi_se_pct": 100 * np.sqrt(variance / n_sims) / contest.entry_fee,
            "win_pct": 100 * wins / n_sims,
            "top1_pct": 100 * top1 / n_sims,
            "top01_pct": 100 * top01 / n_sims,
            "cash_pct": 100 * cashes / n_sims,
            "expected_dupes": dupes,
        }
    )


def adjust_roi(results: pd.DataFrame, extra: pd.DataFrame | None = None) -> np.ndarray:
    """A steadier ROI estimate for ranking.

    Raw tournament ROI swings on a handful of simulated first places, so the lineups at
    the top of a raw ranking are mostly the lucky ones. Finish rates (top 0.1%, top 1%,
    cash) are far less noisy. This predicts each lineup's ROI from those rates, then
    moves it toward its own raw ROI only as far as the raw figure's precision justifies.
    """
    raw, se = results["roi_pct"].to_numpy(), results["roi_se_pct"].to_numpy()
    if len(raw) < 20:
        return raw.copy()
    cols = [results[c].to_numpy() for c in ("top01_pct", "top1_pct", "cash_pct")]
    if extra is not None:
        cols += [extra[c].to_numpy(float) for c in extra.columns]
    x = np.column_stack([np.ones(len(raw))] + [(c - c.mean()) / (c.std() + 1e-12) for c in cols])
    predicted = x @ np.linalg.lstsq(x, raw, rcond=None)[0]
    residual = raw - predicted
    true_spread = max(residual.var() - np.mean(se**2), 0.0)
    return predicted + residual * true_spread / (true_spread + se**2)


def describe_lineups(lineups: np.ndarray, players: pd.DataFrame, field_own: np.ndarray) -> pd.DataFrame:
    """Lay each lineup out by roster slot with its salary, projection and ownership."""
    pos = players["position"].to_numpy()
    name = (players["name"] + " (" + players["team"] + ")").to_numpy()
    mean = players["mean"].to_numpy(float)
    rows = []
    for lineup in lineups:
        by_pos = {p: sorted(lineup[pos[lineup] == p], key=lambda i: -mean[i]) for p in ("QB", "RB", "WR", "TE", "DST")}
        flex = next(by_pos[p].pop() for p, need in (("RB", 2), ("WR", 3), ("TE", 1)) if len(by_pos[p]) > need)
        ordered = by_pos["QB"] + by_pos["RB"] + by_pos["WR"] + by_pos["TE"] + [flex] + by_pos["DST"]
        rows.append([name[i] for i in ordered])
    out = pd.DataFrame(rows, columns=SLOTS)
    out["salary"] = players["salary"].to_numpy()[lineups].sum(axis=1)
    out["projection"] = mean[lineups].sum(axis=1).round(2)
    out["total_ownership_pct"] = (100 * field_own[lineups].sum(axis=1)).round(1)
    return out


def pick_portfolio(lineups: np.ndarray, roi: np.ndarray, n: int, max_exposure: float, max_overlap: int) -> np.ndarray:
    """Greedily take the best lineups while capping any player's share of the set and
    the number of players two chosen lineups may share."""
    limit = max(1, int(np.floor(max_exposure * n + 1e-9)))
    counts: dict[int, int] = {}
    chosen: list[int] = []
    for i in np.argsort(-roi):
        if len(chosen) >= n:
            break
        if any(counts.get(p, 0) >= limit for p in lineups[i]):
            continue
        if any(len(np.intersect1d(lineups[i], lineups[j])) > max_overlap for j in chosen):
            continue
        chosen.append(int(i))
        for p in lineups[i]:
            counts[p] = counts.get(p, 0) + 1
    return np.array(chosen, dtype=int)
