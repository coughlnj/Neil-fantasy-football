"""Contest definitions and payout maths."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Contest:
    name: str
    entry_fee: float
    entries: int
    payouts: np.ndarray  # prize for rank 1, 2, 3, ... (paid places only)

    @property
    def prize_pool(self) -> float:
        return float(self.payouts.sum())

    @property
    def rake(self) -> float:
        return 1.0 - self.prize_pool / (self.entry_fee * self.entries)

    def expected_payout(self, above: np.ndarray, tied: np.ndarray, field_size: int) -> np.ndarray:
        """Prize for a lineup with `above` sampled opponents ahead of it and `tied` level.

        The sampled field stands in for the real one: with k of them ahead, the number
        of real opponents ahead is spread evenly over the k-th of field_size + 1 equal
        slices of the contest. Ties share the prizes for the places they cover.
        """
        scale = (self.entries - 1) / (field_size + 1)
        cum = np.concatenate([[0.0], np.cumsum(self.payouts)])
        places = np.arange(len(cum))
        first = above * scale
        last = (above + tied + 1) * scale
        return (np.interp(last, places, cum) - np.interp(first, places, cum)) / (last - first)


def load_contest(path: str) -> Contest:
    with open(path) as fh:
        spec = json.load(fh)
    fee, entries = float(spec["entry_fee"]), int(spec["entries"])
    if "payouts" in spec:
        prizes: list[float] = []
        for tier in spec["payouts"]:
            prizes += [float(tier["prize"])] * (int(tier["to"]) - int(tier["from"]) + 1)
        if "tail" in spec:
            prizes += list(tail_payouts(len(prizes) + 1, **spec["tail"]))
        payouts = np.array(prizes)
    else:
        payouts = generated_payouts(fee, entries, **spec.get("auto", {}))
    if len(payouts) > entries:
        raise ValueError(f"{path}: more paid places than entries")
    return Contest(spec.get("name", path), fee, entries, payouts)


def generated_payouts(
    entry_fee: float,
    entries: int,
    rake: float = 0.15,
    pct_paid: float = 0.22,
    first_place_share: float = 0.15,
    min_cash_multiple: float = 2.0,
) -> np.ndarray:
    """A typical top-heavy payout curve, for when the real table is not to hand."""
    pool = entry_fee * entries * (1 - rake)
    paid = max(1, int(round(entries * pct_paid)))
    min_cash = entry_fee * min_cash_multiple
    first = max(pool * first_place_share, min_cash)
    ranks = np.arange(1, paid + 1)

    def curve(decay: float) -> np.ndarray:
        return np.maximum(first * ranks ** (-decay), min_cash)

    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if curve(mid).sum() > pool else (lo, mid)
    payouts = curve(hi)
    return payouts * (pool / payouts.sum())


def tail_payouts(first_place: int, to: int, total: float, start: float, min_prize: float) -> np.ndarray:
    """Fill in the places below a known top of the table: prizes fall smoothly from
    `start` to `min_prize` through place `to`, adding up to `total`."""
    ranks = np.arange(first_place, to + 1) / first_place
    if not min_prize * len(ranks) <= total <= start * len(ranks):
        raise ValueError("tail total is not reachable with these start and minimum prizes")

    def curve(decay: float) -> np.ndarray:
        return np.maximum(start * ranks ** (-decay), min_prize)

    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if curve(mid).sum() > total else (lo, mid)
    payouts = curve(hi)
    return payouts + (total - payouts.sum()) / len(payouts)
