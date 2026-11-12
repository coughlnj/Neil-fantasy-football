"""Default settings. Any of these can be overridden with a JSON file (--config)."""
from __future__ import annotations

import copy
import json

POSITIONS = ("QB", "RB", "WR", "TE", "DST")

DEFAULTS: dict = {
    # What the floor and ceiling columns mean, as percentiles of a player's outcomes.
    "floor_percentile": 0.10,
    "ceiling_percentile": 0.90,
    # Lowest score a player can post.
    "lower_bound": {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0, "DST": -4.0},
    # Used when a floor or ceiling is missing or inconsistent (as multiples of the mean).
    "fallback_floor_ratio": 0.45,
    "fallback_ceiling_ratio": 1.65,
    # Correlations between players in the same game. Approximate starting values in
    # line with published DFS correlation studies; tune them to taste.
    "correlation": {
        "same_team": {
            "QB-QB": -0.20, "QB-RB": 0.08, "QB-WR": 0.42, "QB-TE": 0.32, "QB-DST": 0.02,
            "RB-RB": -0.12, "RB-WR": 0.02, "RB-TE": 0.02, "RB-DST": 0.20,
            "WR-WR": 0.03, "WR-TE": 0.00, "WR-DST": 0.00,
            "TE-TE": -0.05, "TE-DST": 0.00,
        },
        "opponent": {
            "QB-QB": 0.20, "QB-RB": 0.05, "QB-WR": 0.12, "QB-TE": 0.08, "QB-DST": -0.40,
            "RB-RB": 0.00, "RB-WR": 0.05, "RB-TE": 0.02, "RB-DST": -0.25,
            "WR-WR": 0.08, "WR-TE": 0.05, "WR-DST": -0.20,
            "TE-TE": 0.03, "TE-DST": -0.15,
            "DST-DST": -0.05,
        },
        # Players projected below these means get proportionally weaker correlations
        # (a third-string receiver is tied less tightly to the quarterback).
        "depth_reference": {"QB": 15.0, "RB": 12.0, "WR": 12.0, "TE": 9.0, "DST": 1.0},
        "min_depth_factor": 0.25,
        # A quarterback who scores more by running is tied less tightly to pass catchers.
        # Links between a QB and WRs/TEs are scaled by his passing share relative to a
        # typical QB's. Per-player values come from an optional rush_share column.
        "typical_qb_rush_share": 0.12,
    },
    # DraftKings classic rules.
    "site": {
        "salary_cap": 50000,
        "roster": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1},
        "flex_positions": ["RB", "WR", "TE"],
    },
    # How opponents build lineups.
    "field": {
        "min_salary": 48000,
        "median_salary": 49700,
        # When ownership is supplied, the average lineup salary it implies is used in
        # place of median_salary, so the field spends what the ownership says it spends.
        "match_salary_to_ownership": True,
        # Share of lineups whose flex is a RB / WR / TE, e.g. {"RB": 0.4, "WR": 0.5, "TE": 0.1}.
        # None takes it from the ownership totals per position. When set, ownership is
        # rescaled within each position to fit.
        "flex_split": None,
        # Share of opponent lineups by how many WRs/TEs share a team with the QB. Measured
        # from the 8 November 2026 Flea Flicker (118,855 entries, on the 76% whose QB's
        # team was known): 16% / 51% / 33%.
        "stack_mix": {"none": 0.16, "single": 0.51, "double": 0.33},
        # Share of stacks that also use an opposing WR/TE. Still an estimate.
        "bringback_rate": 0.40,
        "dst_vs_qb_penalty": 0.05,  # weight multiplier for a DST facing the lineup's QB
        "flex_weight": {"RB": 1.0, "WR": 1.0, "TE": 0.35},
        "calibration_rounds": 8,
        "calibration_lineups": 8000,
    },
    # How candidate lineups are built.
    "candidates": {
        "min_salary": 48500,
        "min_stack": 1,            # pass catchers required alongside the QB
        "allow_dst_vs_qb": False,
        "blend_range": [0.25, 0.75],  # weight on a simulated slate vs the mean projection
        # The wide search: lineups are sampled around the players the optimizer favours,
        # screened on a few simulations against part of the field, then re-sampled
        # around the survivors.
        "search_rounds": 2,
        "screen_sims": 1000,
        "screen_field": 5000,
        "stack_mix": {"none": 0.03, "single": 0.42, "double": 0.55},
        "bringback_rate": 0.5,
        "flex_weight": {"RB": 1.0, "WR": 1.0, "TE": 0.35},
    },
}


def load_settings(path: str | None = None) -> dict:
    settings = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as fh:
            _merge(settings, json.load(fh))
    return settings


def _merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
