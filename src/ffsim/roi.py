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
    candidates: np.ndarray, field: np.ndarray, scores: np.ndarray, contest: Contest, chunk: int = 200,
    keep_payouts: bool = False,
):
    """Expected ROI and finish rates for each candidate, over every simulated slate.
    With keep_payouts, also returns every candidate's prize in every slate.

    `field` is one array of opponent lineups, or a list of equally sized ones (ownership
    scenarios); the simulated slates are then shared out between them in turn."""
    fields = [field] if isinstance(field, np.ndarray) else list(field)
    if len({len(f) for f in fields}) != 1:
        raise ValueError("opponent fields must all be the same size")
    field = fields[0]
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
    payouts = np.zeros((len(candidates), n_sims), dtype=np.float32) if keep_payouts else None

    chunk = min(chunk, max(1, -(-n_sims // len(fields))))      # every scenario gets its share of slates
    for turn, start in enumerate(range(0, n_sims, chunk)):
        block = scores[:, start : start + chunk]
        field_scores = np.sort(lineup_scores(fields[turn % len(fields)], block), axis=0)
        cand_scores = lineup_scores(candidates, block)
        for col in range(block.shape[1]):
            left = np.searchsorted(field_scores[:, col], cand_scores[:, col], side="left")
            right = np.searchsorted(field_scores[:, col], cand_scores[:, col], side="right")
            above, tied = n_field - right, right - left
            prize = contest.expected_payout(above, tied, n_field)
            rank = (above + 0.5) * scale + 1
            total += prize
            if keep_payouts:
                payouts[:, start + col] = prize
            total_sq += prize**2
            wins += (above == 0) / np.maximum((tied + 1) * scale, 1.0)
            top1 += rank <= 0.01 * contest.entries
            top01 += rank <= 0.001 * contest.entries
            cashes += rank <= paid
            if start == 0 and col == 0:
                dupes = tied * scale

    mean_prize = total / n_sims
    variance = np.maximum(total_sq / n_sims - mean_prize**2, 0)
    results = pd.DataFrame(
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
    return (results, payouts) if keep_payouts else results


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
    rows, slots = [], []
    for lineup in lineups:
        by_pos = {p: sorted(lineup[pos[lineup] == p], key=lambda i: -mean[i]) for p in ("QB", "RB", "WR", "TE", "DST")}
        flex = next(by_pos[p].pop() for p, need in (("RB", 2), ("WR", 3), ("TE", 1)) if len(by_pos[p]) > need)
        ordered = by_pos["QB"] + by_pos["RB"] + by_pos["WR"] + by_pos["TE"] + [flex] + by_pos["DST"]
        rows.append([name[i] for i in ordered])
        slots.append(ordered)
    out = pd.DataFrame(rows, columns=SLOTS)
    out.attrs["slots"] = np.array(slots)
    out["salary"] = players["salary"].to_numpy()[lineups].sum(axis=1)
    out["projection"] = mean[lineups].sum(axis=1).round(2)
    out["total_ownership_pct"] = (100 * field_own[lineups].sum(axis=1)).round(1)
    return out


def exposure_limits(n: int, max_exposure: float, n_players: int, overrides: dict | None = None) -> np.ndarray:
    """Most lineups each player may appear in: max_exposure of the set, except for
    players given their own share in `overrides` (player row number -> share)."""
    limits = np.full(n_players, max(1, int(np.floor(max_exposure * n + 1e-9))))
    for player, share in (overrides or {}).items():
        limits[player] = int(np.floor(share * n + 1e-9))
    return limits


def pick_portfolio(lineups: np.ndarray, roi: np.ndarray, n: int, max_exposure: float, max_overlap: int,
                   overrides: dict | None = None) -> np.ndarray:
    """Greedily take the best lineups while capping any player's share of the set and
    the number of players two chosen lineups may share."""
    limit = exposure_limits(n, max_exposure, int(lineups.max()) + 1, overrides)
    counts: dict[int, int] = {}
    chosen: list[int] = []
    for i in np.argsort(-roi):
        if len(chosen) >= n:
            break
        if any(counts.get(p, 0) >= limit[p] for p in lineups[i]):
            continue
        if any(len(np.intersect1d(lineups[i], lineups[j])) > max_overlap for j in chosen):
            continue
        chosen.append(int(i))
        for p in lineups[i]:
            counts[p] = counts.get(p, 0) + 1
    return np.array(chosen, dtype=int)


def pick_portfolio_for_growth(
    lineups: np.ndarray,
    payouts: np.ndarray,
    roi_pct: np.ndarray,
    entry_fee: float,
    n: int,
    bankroll: float,
    max_exposure: float,
    max_overlap: int,
    overrides: dict | None = None,
) -> np.ndarray:
    """Choose lineups as a set, to maximise the expected growth of a bankroll.

    Expected winnings simply add up across lineups, so judging lineups one at a time
    cannot tell a diversified set from one that wins or loses all together. Growth of a
    bankroll (log utility) can: each lineup is valued by what it adds in the slates
    where the lineups already chosen are not winning. A larger bankroll moves the choice
    toward plain expected value.

    payouts: each lineup's prize in each simulated slate. They are rescaled so each
    lineup's average matches roi_pct, the steadier estimate.
    """
    raw = payouts.mean(axis=1)
    target = entry_fee * (1 + roi_pct / 100)
    factor = np.clip(np.divide(target, raw, out=np.ones_like(target), where=raw > 0), 0.2, 3.0)
    prizes = payouts * factor[:, None].astype(np.float32)
    base = max(bankroll - n * entry_fee, n * entry_fee)
    winnings = np.zeros(payouts.shape[1], dtype=np.float32)

    limit = exposure_limits(n, max_exposure, int(lineups.max()) + 1, overrides)
    membership = np.zeros((len(lineups), lineups.max() + 1), dtype=np.int16)
    membership[np.arange(len(lineups))[:, None], lineups] = 1
    counts = np.zeros(membership.shape[1], dtype=int)
    eligible = np.ones(len(lineups), dtype=bool)

    def gain(i) -> float:
        return float(np.log1p(prizes[i] / (base + winnings)).mean())

    # Gains only shrink as winnings build up, so a stale gain is an upper bound and
    # only the current leader needs recomputing.
    bound = np.log1p(prizes / base).mean(axis=1)
    chosen: list[int] = []
    while len(chosen) < n and eligible.any():
        i = int(np.argmax(np.where(eligible, bound, -np.inf)))
        bound[i] = gain(i)
        if bound[i] < np.where(eligible, bound, -np.inf).max():
            continue
        chosen.append(i)
        winnings += prizes[i]
        counts += membership[i]
        eligible[i] = False
        eligible &= (membership @ membership[i]) <= max_overlap
        eligible &= ~(membership[:, counts >= limit].any(axis=1))
    return np.array(chosen, dtype=int)


CALIBRATION_LINEUPS = 1500


def calibration_lineups(players: pd.DataFrame, settings: dict, fields: list, seed: int | None) -> np.ndarray | None:
    """Typical opponent lineups, drawn fresh from the projected-ownership field, used to
    measure how far grading against shaken-ownership fields lifts every ROI. None when
    there is only one field (no shaking, nothing to correct)."""
    if not isinstance(fields, list) or len(fields) <= 1:
        return None
    from .field import sample_field

    return sample_field(players, settings, CALIBRATION_LINEUPS, None if seed is None else seed + 5003)[0]


def calibration_factor(calibration: pd.DataFrame | None, contest: Contest) -> float:
    """How much the shaken-ownership fields inflate prizes.

    All entries together win the prize pool, so a typical lineup must average the rake
    (-15.9% in the Flea Flicker). Graded against fields built from shaken ownership, typical
    lineups average about -2% instead: the shaking spreads the field's chalk around, so
    every lineup meets fewer duplicates than it would in a real contest (checked on the
    8, 15 and 22 November contests, where real entries graded at +0.1%, -3.3% and +2.3%).
    Dividing every prize by this factor puts a typical lineup back at the rake. The
    ranking of lineups is unchanged."""
    if calibration is None or len(calibration) == 0:
        return 1.0
    typical = 1.0 + float(calibration["roi_pct"].mean()) / 100.0
    return max(typical / (1.0 - contest.rake), 1e-6)


def calibrate(results: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Results with every prize divided by `factor` (see calibration_factor). Finish
    rates are left as they are."""
    if factor == 1.0:
        return results
    out = results.copy()
    out["roi_pct"] = (100.0 + out["roi_pct"]) / factor - 100.0
    if "roi_se_pct" in out:
        out["roi_se_pct"] = out["roi_se_pct"] / factor
    return out


def calibrated_roi(roi_pct: np.ndarray, factor: float) -> np.ndarray:
    return (100.0 + np.asarray(roi_pct, dtype=float)) / factor - 100.0


def portfolio_summary(payouts: np.ndarray, roi_pct: np.ndarray, entry_fee: float) -> dict:
    """How a set of lineups performs together across the simulated slates."""
    cost = entry_fee * len(payouts)
    total = payouts.sum(axis=0)
    return {
        "lineups": len(payouts),
        "expected_roi_pct": round(float(roi_pct.mean()), 1),
        "profit_chance_pct": round(100 * float((total > cost).mean()), 1),
        "median_return_pct": round(100 * float(np.median(total) / cost - 1), 1),
        "chance_of_5x_pct": round(100 * float((total >= 5 * cost).mean()), 2),
        "chance_of_losing_half_pct": round(100 * float((total < 0.5 * cost).mean()), 1),
    }
