import os

import numpy as np
import pandas as pd
import pytest

from ffsim.pbp import models as M
from ffsim.pbp.engine import simulate_game
from ffsim.pbp.validate import compare, real_box_scores, simulated_box_scores

REAL_DATA = "data/raw/pbp"


def made_up_plays(n=9000, seed=0) -> pd.DataFrame:
    """A crude stand-in for real play-by-play data, with every column the models read."""
    rng = np.random.default_rng(seed)
    kind = rng.choice(["dropback", "rush", "punt", "field_goal", "penalty", "kickoff", "extra_point", "two_point"],
                      n, p=[0.42, 0.30, 0.06, 0.04, 0.07, 0.06, 0.04, 0.01])
    yardline = rng.integers(1, 100, n)
    down = rng.integers(1, 5, n)
    kicks = np.isin(kind, ["punt", "field_goal"])
    down[kicks] = 4
    yardline[kind == "field_goal"] = rng.integers(1, 45, (kind == "field_goal").sum())
    yardline[kind == "punt"] = rng.integers(40, 100, (kind == "punt").sum())
    togo = np.minimum(rng.integers(1, 16, n), yardline)
    drop, rush = kind == "dropback", kind == "rush"
    sack = drop & (rng.random(n) < 0.07)
    scramble = drop & ~sack & (rng.random(n) < 0.05)
    complete = drop & ~sack & ~scramble & (rng.random(n) < 0.62)
    gain = np.where(complete, rng.gamma(2.0, 5.5, n), np.where(rush | scramble, rng.normal(4.3, 6, n), 0)).round()
    gain = np.where(sack, -rng.integers(1, 12, n), gain)
    gain = np.minimum(gain, yardline)
    pick = drop & ~sack & ~scramble & ~complete & (rng.random(n) < 0.05)
    fumble = (drop | rush) & ~pick & (rng.random(n) < 0.007)
    live_td = (drop | rush) & (gain >= yardline) & ~pick & ~fumble
    pos = rng.choice(["AAA", "BBB"], n)
    other = np.where(pos == "AAA", "BBB", "AAA")
    flips = pick | fumble | kicks
    return pd.DataFrame({
        "game_id": "g" + pd.Series(np.arange(n) // 150).astype(str), "season": 2026, "kind": kind, "posteam": pos,
        "defteam": other, "down": down.astype(float), "ydstogo": togo, "yardline_100": yardline.astype(float),
        "score_differential": rng.integers(-21, 22, n).astype(float),
        "game_seconds_remaining": rng.integers(1, 3600, n).astype(float),
        "half_seconds_remaining": rng.integers(1, 1800, n).astype(float),
        "yards_gained": gain, "interception": pick.astype(float), "fumble_lost": fumble.astype(float),
        "touchdown": live_td.astype(float), "td_team": np.where(live_td, pos, None), "return_touchdown": 0.0,
        "pass_attempt": (drop & ~scramble).astype(float), "sack": sack.astype(float), "qb_scramble": scramble.astype(float),
        "complete_pass": complete.astype(float), "incomplete_pass": (drop & ~sack & ~scramble & ~complete & ~pick).astype(float),
        "out_of_bounds": (rng.random(n) < 0.1).astype(float), "air_yards": rng.integers(-3, 30, n).astype(float),
        "elapsed": np.where(complete | rush | sack, rng.integers(20, 45, n), rng.integers(3, 10, n)).astype(float),
        "next_posteam": np.where(flips, other, pos), "next_yardline": rng.integers(55, 90, n).astype(float),
        "next_kind": "dropback", "penalty_team": rng.choice(["AAA", "BBB"], n),
        "penalty_yards": rng.choice([5.0, 10.0, 15.0], n), "first_down_penalty": (rng.random(n) < 0.3).astype(float),
        "field_goal_result": np.where(rng.random(n) < 0.85, "made", "missed"), "kick_distance": yardline + 18.0,
        "extra_point_result": np.where(rng.random(n) < 0.95, "good", "failed"),
        "two_point_conv_result": np.where(rng.random(n) < 0.45, "success", "failure"),
        "posteam_score": 0.0, "defteam_score": 0.0, "home_team": "AAA", "away_team": "BBB",
    })


@pytest.fixture(scope="module")
def model():
    return M.fit_game_model(made_up_plays())


def test_situation_buckets_cover_every_combination():
    downs, togo, yards = np.meshgrid([1, 2, 3, 4], [1, 2, 3, 6, 7, 10, 11, 25], [1, 5, 6, 10, 11, 20, 21, 50, 51, 99])
    for kind in (0, 1):
        b = M.bucket_index(np.full(downs.size, kind), downs.ravel(), togo.ravel(), yards.ravel())
        assert b.min() >= kind * 60 and b.max() < (kind + 1) * 60 and len(np.unique(b)) == 60
    assert M.play_bucket_distance(0, 60) >= 1000 and M.play_bucket_distance(7, 7) == 0


def test_logistic_fit_recovers_known_odds():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(40000), rng.normal(size=40000)])
    y = (rng.random(40000) < M.predict(x, np.array([-0.5, 1.2]))).astype(float)
    assert M.fit_logistic(x, y, ridge=0.0) == pytest.approx([-0.5, 1.2], abs=0.05)


def test_game_adds_up(model):
    s = simulate_game(model, 3000, np.random.default_rng(1))
    for name, values in s.items():
        assert np.isfinite(values).all(), name
        assert "yards" in name or (np.asarray(values) >= 0).all(), name      # yardage can be negative
    touchdowns = s["pass_td"] + s["rush_td"] + s["return_td"]
    after = s["points"] - 6 * touchdowns - 3 * s["fg_made"] - 2 * s["safeties"]
    assert (after >= 0).all() and (after <= 2 * touchdowns).all()       # conversions: 0 to 2 per touchdown
    assert (s["completions"] <= s["pass_att"]).all() and (s["fg_made"] <= s["fg_att"]).all()
    assert (s["pass_att"] + s["sacks"] <= s["dropbacks"]).all() and (s["dropbacks"] <= s["plays"]).all()
    assert (s["pass_td"] <= s["completions"]).all()
    assert s["plays"].sum(axis=0).min() > 40       # every game ran a full game's worth of snaps
    tied = s["points"][0] == s["points"][1]
    assert (s["overtime"][tied]).all()             # a tie can only come after overtime


def test_better_offence_scores_more_and_a_trailing_team_passes_more(model):
    s = simulate_game(model, 6000, np.random.default_rng(2), quality=(0.6, -0.6))
    assert s["points"][0].mean() > s["points"][1].mean() + 6
    assert s["pass_yards"][0].mean() > s["pass_yards"][1].mean()
    lean = simulate_game(model, 6000, np.random.default_rng(3), pass_shift=(1.0, -1.0))
    assert lean["dropbacks"][0].mean() > lean["dropbacks"][1].mean() + 8


def test_on_play_sees_every_scrimmage_play(model):
    seen = {"plays": 0, "yards": 0.0}

    def record(games, offence, row, gain, scored, is_run):
        seen["plays"] += len(games)
        complete = model.plays.columns["complete"][row]
        seen["yards"] += float(gain[complete].sum())

    s = simulate_game(model, 500, np.random.default_rng(4), on_play=record)
    assert seen["plays"] == int(s["plays"].sum())
    assert seen["yards"] == pytest.approx(float(s["pass_yards"].sum()))


@pytest.mark.skipif(not os.path.isdir(REAL_DATA), reason="real play-by-play data not downloaded")
def test_simulated_games_match_real_league_averages():
    from ffsim.pbp.data import load_pbp

    plays = load_pbp(REAL_DATA)
    real = real_box_scores(plays)
    sim = simulated_box_scores(M.fit_game_model(plays), 20000, seed=1)
    table, checks = compare(real, sim)
    for stat, tolerance in (("points", 0.05), ("plays", 0.03), ("pass_att", 0.04), ("pass_yards", 0.05),
                            ("rush_att", 0.04), ("rush_yards", 0.05), ("sacks", 0.08), ("punts", 0.08)):
        assert table.loc[stat, "sim mean"] == pytest.approx(table.loc[stat, "real mean"], rel=tolerance), stat
    flow = checks.loc["Correlation: dropbacks with opponent's points"]
    assert flow["sim"] > 0.25 and abs(flow["sim"] - flow["real"]) < 0.1
