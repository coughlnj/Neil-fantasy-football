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
    # Correlations between players in the same game. The skill-position values began as
    # published DFS figures and were then checked against every game from 2024 to
    # mid-2026 (707 games): they were within about 0.05 of the real figures and were left
    # alone. The links involving a defence, and between the two lead running backs, were
    # off by 0.1 or more and are set from that data.
    "correlation": {
        "same_team": {
            "QB-QB": -0.20, "QB-RB": 0.08, "QB-WR": 0.42, "QB-TE": 0.32, "QB-DST": -0.15,
            "RB-RB": -0.12, "RB-WR": 0.02, "RB-TE": 0.02, "RB-DST": 0.06,
            "WR-WR": 0.03, "WR-TE": 0.00, "WR-DST": -0.10,
            "TE-TE": -0.05, "TE-DST": -0.12,
        },
        "opponent": {
            "QB-QB": 0.20, "QB-RB": 0.05, "QB-WR": 0.12, "QB-TE": 0.08, "QB-DST": -0.30,
            "RB-RB": -0.10, "RB-WR": 0.05, "RB-TE": 0.02, "RB-DST": -0.25,
            "WR-WR": 0.08, "WR-TE": 0.05, "WR-DST": -0.20,
            "TE-TE": 0.03, "TE-DST": -0.07,
            "DST-DST": -0.18,
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
        # The figures below are measured from the 8 November 2026 $5 Flea Flicker
        # (118,855 entries).
        "min_salary": 48000,
        # Share of lineups by salary left unspent: $0, $100, $200-400, $500-900,
        # $1,000-1,900, $2,000 or more.
        "unspent_salary": [0.469, 0.229, 0.234, 0.057, 0.010, 0.001],
        # Share of opponent lineups by stack type: how many WRs/TEs share a team with
        # the QB, and whether an opposing WR/TE is added ("bringback").
        "stack_classes": {
            "none": 0.126, "single": 0.351, "single_bringback": 0.168,
            "double": 0.199, "double_bringback": 0.156,
        },
        # Kinds of builder in the field: the four kinds of real entrant, by how many entries
        # they play, with their measured shares and stack types (same order as
        # stack_classes). Each kind mixes two styles: lineups guided by ownership, with a
        # sharpness (below 1 spreads picks toward less popular players), and a smaller share
        # of optimizer-style lineups, each the highest-projected of best_of draws, which make
        # most of the duplicates. Fitted on the 8 and 15 November contests, with each week's
        # field built from its actual ownership, to each kind's average projection,
        # chalkiness, spread and duplicated lineups. Set to None for a single uniform field.
        "groups": [
            {"name": "single entry", "share": 0.1176, "sharpness": 0.65, "best_of": 1,
             "stack_classes": [0.2851, 0.4252, 0.1184, 0.1169, 0.0544]},
            {"name": "single entry, optimizer", "share": 0.0116, "sharpness": 1.0, "best_of": 40,
             "stack_classes": [0.2851, 0.4252, 0.1184, 0.1169, 0.0544]},
            {"name": "2-19 entries", "share": 0.2362, "sharpness": 0.7, "best_of": 1,
             "stack_classes": [0.1752, 0.3634, 0.1633, 0.1684, 0.1297]},
            {"name": "2-19 entries, optimizer", "share": 0.0384, "sharpness": 1.0, "best_of": 40,
             "stack_classes": [0.1752, 0.3634, 0.1633, 0.1684, 0.1297]},
            {"name": "20-149 entries", "share": 0.176, "sharpness": 0.85, "best_of": 1,
             "stack_classes": [0.0912, 0.3181, 0.1922, 0.2054, 0.193]},
            {"name": "20-149 entries, optimizer", "share": 0.0386, "sharpness": 1.0, "best_of": 40,
             "stack_classes": [0.0912, 0.3181, 0.1922, 0.2054, 0.193]},
            {"name": "150 entries", "share": 0.3282, "sharpness": 0.95, "best_of": 1,
             "stack_classes": [0.0566, 0.3356, 0.2016, 0.2297, 0.1765]},
            {"name": "150 entries, optimizer", "share": 0.0534, "sharpness": 1.0, "best_of": 40,
             "stack_classes": [0.0566, 0.3356, 0.2016, 0.2297, 0.1765]},
        ],
        # How entrants build around games, as multipliers on a player's chance of being
        # picked (1 = no effect). Fitted to the same two contests: real lineups put a
        # player against their own defense in 8% of entries (20% before this was added),
        # pair a running back with his own defense in 16%, with his quarterback in 18%,
        # with the opposing quarterback in 19%, and three pass catchers with the
        # quarterback in 2%.
        "pairing": {"player_facing_dst": 0.24, "rb_with_own_dst": 2.0, "rb_with_qb": 1.75,
                    "rb_facing_qb": 2.2, "third_pass_catcher": 2.2},
        "dst_vs_qb_penalty": 0.2,  # weight multiplier for a DST facing the lineup's QB
        "flex_weight": {"RB": 1.0, "WR": 1.0, "TE": 0.35},
        "calibration_rounds": 12,
        "calibration_lineups": 8000,
        # Share of lineups whose flex is a RB / WR / TE, e.g. {"RB": 0.4, "WR": 0.5, "TE": 0.1}.
        # None takes it from the ownership totals per position. When set, ownership is
        # rescaled within each position to fit.
        "flex_split": None,
        # Projected ownership is an estimate. Lineups are graded against this many opponent
        # fields, each built from ownership shaken by the provider's typical miss, with the
        # simulated slates shared out between them. 1 uses the projected ownership as given.
        "ownership_scenarios": 4,
        # Size of that miss: spread (sd) of log(actual / projected) at these projected
        # ownership levels, measured on the 8 and 15 November 2026 Flea Flickers.
        "ownership_error": {"at_pct": [1.7, 4.3, 8.5, 16.0, 35.0], "log_sd": [0.50, 0.40, 0.31, 0.24, 0.17]},
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
    # Play-by-play simulator. game_plan_sd: how much a team's lean toward the pass varies from
    # one game to the next (in log-odds), beyond what the score and clock explain. Fitted so the
    # spread of a team's pass share matches real games (see `ffsim pbp-check`). 0 turns it off.
    # exit_rate: chance per game that a running back, receiver or tight end leaves partway
    # through (injury or benching), his later targets and carries going to teammates. Fitted
    # so starters vanish after halftime, fall below a fifth of their usage after halftime,
    # and score near zero about as often as in real games (2024 to mid-2026).
    "pbp": {"game_plan_sd": 0.25, "exit_rate": {"RB": 0.13, "WR": 0.11, "TE": 0.09}},
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
