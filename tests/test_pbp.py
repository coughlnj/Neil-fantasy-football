import os

import numpy as np
import pandas as pd
import pytest

from ffsim.pbp import models as M
from ffsim.pbp.engine import simulate_game
from ffsim.pbp.players import Matchup, defence_points, draftkings_points, player_key, usage_history
from ffsim.pbp.slate import simulate_slate_pbp, team_pass_lean
from ffsim.pbp.validate import compare, real_box_scores, simulated_box_scores
from ffsim.settings import load_settings

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
        "receiver_player_name": rng.choice(["W.One", "W.Two", "T.End", "R.Back"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "rusher_player_name": rng.choice(["R.Back", "R.Spare", "Q.Arm"], n, p=[0.7, 0.2, 0.1]),
        "passer_player_name": "Q.Arm",
    })


def made_up_slate() -> pd.DataFrame:
    rows = []
    for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
        for name, pos, mean in (("Quin Arm", "QB", 19.0), ("Spare Thrower", "QB", 0.2), ("Rex Back", "RB", 15.0),
                                ("Ron Spare", "RB", 6.0), ("Wes One", "WR", 16.0), ("Will Two", "WR", 10.0),
                                ("Walt Three", "WR", 5.0), ("Tom End", "TE", 8.0), ("Deep Reserve", "WR", 0.3)):
            rows.append({"name": f"{name} {team}", "position": pos, "team": team, "opponent": opp, "salary": 5000,
                         "mean": mean, "floor": mean * 0.4, "ceiling": mean * 1.8})
        rows.append({"name": f"{team} D", "position": "DST", "team": team, "opponent": opp, "salary": 3000,
                     "mean": 7.0, "floor": 1.0, "ceiling": 14.0})
    slate = pd.DataFrame(rows)
    slate["game"] = "AAA v BBB"
    return slate


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
    touchdowns = s["pass_td"] + s["rush_td"] + s["return_td"] + s["defence_td"]
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


def test_run_quality_tempers_rushing_for_strong_offences(model):
    full = simulate_game(model, 6000, np.random.default_rng(4), quality=(0.6, 0.0))
    tempered = simulate_game(model, 6000, np.random.default_rng(4), quality=(0.6, 0.0), run_quality=0.3)
    per_carry = lambda s: s["rush_yards"][0].sum() / s["rush_att"][0].sum()
    per_pass = lambda s: s["pass_yards"][0].sum() / s["pass_att"][0].sum()
    assert per_carry(tempered) < per_carry(full) - 0.2
    assert per_pass(tempered) == pytest.approx(per_pass(full), rel=0.05)        # passing keeps its full edge


def test_each_simulated_game_can_have_its_own_plan(model):
    n = 8000
    rng = np.random.default_rng(4)
    flat = simulate_game(model, n, np.random.default_rng(5))
    varied = simulate_game(model, n, np.random.default_rng(5), pass_shift=0.5 * rng.standard_normal((2, n)))
    share = lambda s: (s["dropbacks"] / np.maximum(s["plays"], 1)).std()
    assert share(varied) > 1.1 * share(flat)                       # the pass-run split swings more...
    assert varied["points"].mean() == pytest.approx(flat["points"].mean(), rel=0.04)     # ...scoring does not move
    assert varied["points"].std() == pytest.approx(flat["points"].std(), rel=0.05)
    good_days = np.vstack([np.full(n, 0.5), np.linspace(-0.9, 0.9, n)])       # quality can differ game by game too
    s = simulate_game(model, n, np.random.default_rng(6), quality=good_days)
    assert s["points"][1][n // 2:].mean() > s["points"][1][: n // 2].mean() + 5


def test_game_plan_variation_reaches_the_players(model):
    slate = made_up_slate()
    matchup = Matchup(slate, usage_history(made_up_plays()))
    spread = lambda: matchup.simulate(model, 6000, np.random.default_rng(3))[1]["dropbacks"].std()
    before = spread()
    matchup.game_plan_sd = 0.6
    assert spread() > 1.05 * before


def test_on_play_sees_every_scrimmage_play(model):
    seen = {"plays": 0, "yards": 0.0}

    def record(games, offence, row, gain, scored, is_run, yardline, half):
        seen["plays"] += len(games)
        complete = model.plays.columns["complete"][row]
        seen["yards"] += float(gain[complete].sum())

    s = simulate_game(model, 500, np.random.default_rng(4), on_play=record)
    assert seen["plays"] == int(s["plays"].sum())
    assert seen["yards"] == pytest.approx(float(s["pass_yards"].sum()))


def test_player_names_match_across_sources():
    pairs = [("Jaxon Smith-Njigba", "J.Smith-Njigba"), ("Kenneth Walker III", "K.Walker"), ("C.J. Stroud", "C.Stroud"),
             ("Michael Penix Jr.", "M.Penix"), ("Amon-Ra St. Brown", "A.St. Brown"), ("DJ Moore", "D.Moore")]
    for full, short in pairs:
        assert player_key(full) == player_key(short), full
    assert player_key("Josh Allen") != player_key("Keenan Allen")


def test_draftkings_scoring():
    line = lambda **kw: {k: np.array([float(kw.get(k, 0))]) for k in
                         ["pass_yards", "pass_td", "interceptions", "rush_yards", "rush_td", "receptions", "rec_yards",
                          "rec_td", "fumbles_lost"]}
    assert draftkings_points(line(pass_yards=300, pass_td=2, interceptions=1))[0] == pytest.approx(12 + 8 - 1 + 3)
    assert draftkings_points(line(rush_yards=100, rush_td=1, fumbles_lost=1))[0] == pytest.approx(10 + 6 + 3 - 1)
    assert draftkings_points(line(receptions=7, rec_yards=99, rec_td=1))[0] == pytest.approx(7 + 9.9 + 6)
    zero = np.zeros((2, 1), dtype=float)
    box = {k: zero.copy() for k in ["points", "defence_td", "return_td", "sacks", "interceptions", "fumbles_lost", "safeties"]}
    box["points"][1], box["sacks"][1], box["interceptions"][1] = 13, 3, 1        # the opponent's offence
    box["defence_td"][0] = 1
    assert defence_points(box, 0)[0] == pytest.approx(3 + 2 + 6 + 4)                # 13 allowed is worth 4
    box["points"][1], box["defence_td"][1] = 7, 1                                   # all seven came off a pick-six
    assert defence_points(box, 0)[0] == pytest.approx(3 + 2 + 6 + 10)


def test_every_passing_yard_is_a_receiving_yard(model):
    slate = made_up_slate()
    matchup = Matchup(slate, usage_history(made_up_plays()))
    points, box, stats = matchup.simulate(model, 2000, np.random.default_rng(1))
    assert np.isfinite(points).all() and points.shape == (len(slate), 2000)
    for side in (0, 1):
        on = matchup.side == side
        qb = matchup.passer[side]
        assert slate["name"][qb].startswith("Quin Arm")
        for thrown, caught in (("pass_yards", "rec_yards"), ("pass_td", "rec_td")):
            assert stats[thrown][qb] == pytest.approx(stats[caught][on].sum(axis=0))
            assert stats[thrown][qb] == pytest.approx(box[thrown][side])
        assert stats["rush_yards"][on].sum(axis=0) == pytest.approx(box["rush_yards"][side], abs=1e-3)
        assert stats["interceptions"][qb] == pytest.approx(box["interceptions"][side])
    assert (stats["receptions"][slate["position"] == "QB"] == 0).all()


def test_players_can_leave_a_game(model):
    slate = made_up_slate()
    matchup = Matchup(slate, usage_history(made_up_plays()))
    starters = np.flatnonzero(slate["position"].isin(["RB", "WR", "TE"]).to_numpy() & (slate["mean"] >= 8).to_numpy())

    def vanish(rate):
        matchup.exit_rate = {"RB": rate, "WR": rate, "TE": rate}
        points, box, stats = matchup.simulate(model, 4000, np.random.default_rng(5))
        h1, h2 = stats["touches_h1"][starters], stats["touches_h2"][starters]
        for side in (0, 1):            # the ball still goes somewhere: every completion has a receiver
            on = matchup.side == side
            assert stats["rec_yards"][on].sum(axis=0) == pytest.approx(box["pass_yards"][side], abs=1e-3)
        return ((h1 >= 2) & (h2 == 0)).mean(), points

    before, steady = vanish(0.0)
    after, shaken = vanish(0.3)
    assert after > before + 0.03
    assert (shaken[starters] <= 0.5).mean() > (steady[starters] <= 0.5).mean()


def test_fitting_moves_players_toward_their_projections(model):
    slate = made_up_slate()
    matchup = Matchup(slate, usage_history(made_up_plays()))
    main = (slate["mean"] >= 5) & (slate["position"] != "DST")
    gap = lambda: np.abs(matchup.simulate(model, 4000, np.random.default_rng(2))[0].mean(axis=1) - slate["mean"])[main].mean()
    before = gap()
    matchup.fit(model, rounds=8, n_sims=2000, seed=1)
    assert gap() < 0.5 * before


def test_slate_is_anchored_to_the_projections():
    slate, settings = made_up_slate(), load_settings()
    scores, report = simulate_slate_pbp(slate, made_up_plays(), settings, 4000, seed=1)
    assert scores.shape == (len(slate), 4000) and np.isfinite(scores).all()
    used = report["from_games"].to_numpy()
    assert used[(slate["mean"] >= 5).to_numpy()].all()
    assert scores[used].mean(axis=1) == pytest.approx(slate["mean"].to_numpy()[used], abs=0.15)
    assert scores[(slate["position"] == "DST").to_numpy()].min() >= settings["lower_bound"]["DST"]
    qb, wr = 0, 4                                    # a quarterback and his top receiver move together
    assert np.corrcoef(scores[qb], scores[wr])[0, 1] > 0.15
    lean = team_pass_lean(made_up_plays(), M.fit_game_model(made_up_plays()))
    assert set(lean.index) == {"AAA", "BBB"} and lean.abs().max() < 1


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


@pytest.mark.skipif(not os.path.isdir(REAL_DATA), reason="real play-by-play data not downloaded")
def test_swings_around_the_betting_line_are_the_right_size():
    """The guard against overdoing it: scoring must not swing more than real games do."""
    from ffsim.pbp.data import load_pbp
    from ffsim.pbp.validate import around_the_line

    plays = load_pbp(REAL_DATA)
    sd = load_settings()["pbp"]["game_plan_sd"]
    table = around_the_line(plays, M.fit_game_model(plays), sd, repeats=20, seed=2, run_quality=load_settings()["pbp"]["run_quality"])
    real, sim = table["real"], table["sim, with game plan"]
    assert sim["Share of plays that are dropbacks: spread (sd)"] == pytest.approx(real["Share of plays that are dropbacks: spread (sd)"], abs=0.006)
    for row in ("Points: spread (sd)", "Game total points: spread around the line (sd)", "Winning margin: spread around the line (sd)"):
        assert sim[row] == pytest.approx(real[row], rel=0.05), row
    for row in ("Dropbacks: spread (sd)", "Rushes: spread (sd)", "Passing yards: spread (sd)", "Rushing yards: spread (sd)"):
        assert sim[row] <= 1.03 * real[row], row            # never wider than real


@pytest.mark.skipif(not os.path.isdir(REAL_DATA), reason="real play-by-play data not downloaded")
def test_rushing_rises_with_expected_points_like_real_games():
    """Teams expected to score more gain a lot more per pass but only a little more per carry."""
    from ffsim.pbp.data import load_pbp
    from ffsim.pbp.validate import by_expected_points

    plays = load_pbp(REAL_DATA)
    table = by_expected_points(plays, M.fit_game_model(plays), load_settings()["pbp"]["run_quality"], repeats=5, seed=3)
    busy = table[table[("team-games", "real")] >= 100]
    ypc = busy["yards per carry"]
    assert ((ypc["sim"] - ypc["real"]).abs() < 0.25).all()
    rise = lambda side: ypc[side].iloc[-1] - ypc[side].iloc[0]
    assert rise("sim") == pytest.approx(rise("real"), abs=0.25)
