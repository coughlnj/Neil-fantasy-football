import numpy as np
import pandas as pd
import pytest

from ffsim.candidates import build_candidates
from ffsim.contest import Contest, generated_payouts, load_contest, tail_payouts
from ffsim.field import field_report, sample_field, target_ownership
from ffsim.projections import diff_projections, load_projections
from ffsim.roi import adjust_roi, describe_lineups, evaluate, lineup_scores, pick_portfolio
from ffsim.sample import sample_slate
from ffsim.settings import load_settings
from ffsim.simulate import load_simulation, save_simulation, simulate_slate


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def players(tmp_path_factory):
    path = tmp_path_factory.mktemp("slate") / "projections.csv"
    sample_slate(games=10, seed=3).to_csv(path, index=False)
    return load_projections(str(path))


@pytest.fixture(scope="module")
def sim(players, settings):
    return simulate_slate(players, settings, 20000, seed=1)


@pytest.fixture(scope="module")
def field(players, settings):
    return sample_field(players, settings, 8000, seed=1)


def check_legal(lineups, players, settings, min_salary):
    pos = players["position"].to_numpy()[lineups]
    assert all(len(set(row)) == 9 for row in lineups)
    assert ((pos == "QB").sum(axis=1) == 1).all() and ((pos == "DST").sum(axis=1) == 1).all()
    assert ((pos == "RB").sum(axis=1) >= 2).all() and ((pos == "WR").sum(axis=1) >= 3).all()
    assert ((pos == "TE").sum(axis=1) >= 1).all()
    salary = players["salary"].to_numpy()[lineups].sum(axis=1)
    assert (salary <= settings["site"]["salary_cap"]).all() and (salary >= min_salary).all()


def test_loader_accepts_other_column_names(tmp_path):
    path = tmp_path / "p.csv"
    pd.DataFrame(
        {"Player": ["A", "B"], "Pos": ["QB", "DEF"], "Team": ["x", "y"], "Opp": ["@y", "x"],
         "Salary": ["$6,000", "3000"], "Proj": [20, 7], "Own%": ["12%", "8%"]}
    ).to_csv(path, index=False)
    df = load_projections(str(path))
    assert list(df["position"]) == ["QB", "DST"] and list(df["salary"]) == [6000, 3000]
    assert df["game"].nunique() == 1 and df["ownership"].tolist() == [12, 8]


def test_loader_names_the_missing_column(tmp_path):
    path = tmp_path / "p.csv"
    pd.DataFrame({"name": ["A"], "position": ["QB"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="team"):
        load_projections(str(path))


def test_simulation_reproduces_projections(players, sim, settings):
    scores, _ = sim
    mean = players["mean"].to_numpy()
    width = (players["ceiling"] - players["floor"]).to_numpy()
    assert np.abs(scores.mean(axis=1) - mean).max() < 0.3
    assert (np.abs(np.percentile(scores, 15, axis=1) - players["floor"]) < 0.05 * width + 0.3).all()
    assert (np.abs(np.percentile(scores, 85, axis=1) - players["ceiling"]) < 0.05 * width + 0.3).all()
    non_dst = (players["position"] != "DST").to_numpy()
    assert scores[non_dst].min() >= 0


def test_missing_floor_and_ceiling_still_simulate(players, settings):
    bare = players.assign(floor=np.nan, ceiling=np.nan)
    scores, fit = simulate_slate(bare, settings, 5000, seed=1)
    assert fit["used_fallback"].all()
    assert np.abs(scores.mean(axis=1) - bare["mean"]).max() < 0.5


def test_game_correlations(players, sim):
    scores, _ = sim
    row = {n: i for i, n in enumerate(players["name"])}
    corr = lambda a, b: np.corrcoef(scores[row[a]], scores[row[b]])[0, 1]
    assert corr("H01 QB1", "H01 WR1") > 0.25
    assert corr("H01 QB1", "A01 QB1") > 0.1
    assert corr("H01 QB1", "A01 DST") < -0.2
    assert corr("H01 RB1", "H01 DST") > 0.1
    assert abs(corr("H01 QB1", "H02 WR1")) < 0.03


def test_simulation_round_trip(tmp_path, players, sim):
    path = str(tmp_path / "s.npz")
    save_simulation(path, players, sim[0][:, :50])
    loaded, scores = load_simulation(path)
    assert list(loaded["name"]) == list(players["name"]) and scores.shape == (len(players), 50)


def test_generated_payouts():
    payouts = generated_payouts(20, 10000, rake=0.15, pct_paid=0.2)
    assert len(payouts) == 2000 and payouts.sum() == pytest.approx(170000)
    assert (np.diff(payouts) <= 1e-9).all() and payouts[-1] >= 20


def test_payouts_add_up_to_the_prize_pool():
    contest = Contest("t", 10, 1001, np.array([500.0, 300, 200, 100, 100]))
    for field_size in (99, 1000, 5000):
        above = np.arange(field_size + 1)
        prizes = contest.expected_payout(above, np.zeros_like(above), field_size)
        assert prizes.mean() == pytest.approx(contest.prize_pool / 1000)
    exact = contest.expected_payout(np.array([0, 1, 5]), np.array([0, 0, 0]), 1000)
    assert exact == pytest.approx([500, 300, 0], abs=1)
    # Two lineups tied for first share first and second prize.
    assert contest.expected_payout(np.array([0]), np.array([1]), 1000)[0] == pytest.approx(400, abs=1)


def test_sample_contests_load():
    for name in ("typical_gpp", "sample_small_gpp", "flea_flicker"):
        contest = load_contest(f"contests/{name}.json")
        assert 0.1 < contest.rake < 0.2


def test_flea_flicker_table():
    contest = load_contest("contests/flea_flicker.json")
    assert contest.entries == 118000 and len(contest.payouts) == 25250
    assert contest.payouts[:10].tolist() == [100000, 40000, 20000, 10000, 7500, 5000, 2500, 2500, 1500, 1500]
    assert contest.prize_pool == pytest.approx(500000)
    assert (np.diff(contest.payouts) <= 0).all() and contest.payouts[-1] == 8
    assert contest.payouts[10] == 1000 and contest.payouts[13499] == 10 and contest.payouts[13500] == 8


def test_tail_fills_in_missing_places():
    tail = tail_payouts(11, to=1000, total=30000, start=500, min_prize=8)
    assert len(tail) == 990 and tail.sum() == pytest.approx(30000)
    assert tail[0] == pytest.approx(500, rel=0.01) and tail[-1] == pytest.approx(8, abs=0.1)


def test_field_is_legal_and_matches_targets(players, settings, field):
    field, realised = field
    check_legal(field, players, settings, settings["field"]["min_salary"])
    assert np.abs(realised - target_ownership(players, settings)).max() < 0.08
    assert np.abs(realised - target_ownership(players, settings)).mean() < 0.005
    assert realised.sum() == pytest.approx(9)
    report = field_report(field, players, settings).set_index("measure")["value"]
    assert report[["QB with no pass catcher", "QB with exactly 1 pass catcher", "QB with 2 or more pass catchers"]].sum() == pytest.approx(100, abs=0.2)
    assert report[["Flex is a RB", "Flex is a WR", "Flex is a TE"]].sum() == pytest.approx(100, abs=0.2)
    assert report["QB with no pass catcher"] == pytest.approx(17, abs=0.5)
    assert report["QB with 2 or more pass catchers"] == pytest.approx(38, abs=0.5)
    assert report["Stacks that add an opposing pass catcher"] == pytest.approx(40, abs=1)
    assert report["Median salary used"] >= 49500 and report["Defense facing its own QB"] < 5


def test_field_survives_ownership_it_cannot_match(players, settings):
    greedy = players.assign(ownership=np.where(players["salary"] > 7000, 60.0, 1.0))
    field, _ = sample_field(greedy, settings, 500, seed=1)
    check_legal(field, players, settings, settings["field"]["min_salary"])


def test_candidates_are_legal_stacked_and_distinct(players, sim, settings):
    lineups = build_candidates(players, sim[0][:, -40:], settings, 40, seed=1)
    check_legal(lineups, players, settings, settings["candidates"]["min_salary"])
    assert len({tuple(l) for l in lineups}) == len(lineups) > 20
    pos, team, opp = (players[c].to_numpy() for c in ("position", "team", "opponent"))
    for lineup in lineups:
        qb = lineup[pos[lineup] == "QB"][0]
        mates = [i for i in lineup if team[i] == team[qb] and pos[i] in ("WR", "TE")]
        dst = lineup[pos[lineup] == "DST"][0]
        assert mates and opp[dst] != team[qb]
    described = describe_lineups(lineups, players, np.full(len(players), 0.1))
    assert described["QB"].str.contains("QB1").all() and described["DST"].str.contains("DST").all()


def test_field_lineups_lose_the_rake(players, sim, field):
    """Lineups drawn like the field should, on average, earn back prizes minus rake."""
    field = field[0]
    contest = Contest("t", 20, 20000, generated_payouts(20, 20000))
    results = evaluate(field[:1500], field[1500:], sim[0], contest)
    assert results["roi_pct"].mean() == pytest.approx(-100 * contest.rake, abs=6)
    assert results["cash_pct"].mean() == pytest.approx(22, abs=1.5)
    assert results["top1_pct"].mean() == pytest.approx(1, abs=0.15)
    adjusted = adjust_roi(results)
    assert adjusted.mean() == pytest.approx(results["roi_pct"].mean(), abs=1e-6)
    assert adjusted.std() < results["roi_pct"].std()


def test_duplicated_lineup_splits_its_prize(sim, field):
    pool = np.unique(field[0], axis=0)[:51]
    best = lineup_scores(pool, sim[0][:, :500]).mean(axis=1).argmax()
    mine, others = pool[best : best + 1], np.delete(pool, best, axis=0)
    contest = Contest("t", 10, 51, np.array([500.0]))
    solo = evaluate(mine, others, sim[0], contest)
    shared = evaluate(mine, np.vstack([others[1:], mine]), sim[0], contest)
    assert solo["win_pct"][0] > 2 and solo["expected_dupes"][0] == 0
    assert shared["expected_dupes"][0] == pytest.approx(1, abs=0.03)
    assert shared["roi_pct"][0] + 100 == pytest.approx((solo["roi_pct"][0] + 100) / 2, rel=0.1)


def test_portfolio_respects_limits():
    lineups = np.array([[0, 1, 2], [0, 1, 3], [0, 4, 5], [6, 7, 8], [0, 7, 9]])
    roi = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    chosen = pick_portfolio(lineups, roi, n=3, max_exposure=0.67, max_overlap=1)
    assert chosen.tolist() == [0, 2, 3]


def test_diff_reports_movers(players):
    new = players.copy()
    new.loc[new["name"] == "H01 WR1", "mean"] += 3
    new = new[new["name"] != "H02 TE1"]
    moved = diff_projections(players, new, min_change=0.5)
    assert moved.iloc[0]["status"] in ("changed", "removed")
    assert set(moved["name"]) == {"H01 WR1", "H02 TE1"}
    assert moved.set_index("name").loc["H01 WR1", "change"] == pytest.approx(3)
