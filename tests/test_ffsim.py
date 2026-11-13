import numpy as np
import pandas as pd
import pytest

from ffsim.candidates import build_candidates, search_candidates
from ffsim.contest import Contest, generated_payouts, load_contest, tail_payouts
from ffsim.draftkings import check_lineup, read_lineups, attach_dk, normalise_name, read_dk_entries, write_upload
from ffsim import live
from ffsim.field import LineupSampler, field_report, sample_field, sample_field_by_group, target_ownership
from ffsim.projections import diff_projections, load_projections
from ffsim.roi import (adjust_roi, describe_lineups, evaluate, lineup_scores, pick_portfolio,
                       pick_portfolio_for_growth, portfolio_summary)
from ffsim.sample import sample_slate
from ffsim.settings import load_settings
from ffsim.standings import read_standings, showdown_report, standings_report
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


def test_loader_reads_provider_columns_and_picks_ownership(tmp_path, settings):
    path = tmp_path / "p.csv"
    pd.DataFrame(
        {"Player": ["A", "B", "C"], "DK Pos": ["QB", "WR", "TE"], "Team": ["LA", "LA", "ARI"],
         "Opp": ["@ARI", "@ARI", "LA"], "DK Salary": ["$6,000", "$5,000", "$3,000"], "DK Proj": [20, 12, 8],
         "Small Field": ["10%", "30%", "50%"], "Large Field": ["12%", "20%", "40%"],
         "DK Floor": [10, 4, 2], "DK Ceiling": [31, 22, 15], "id": [1, 2, 3]}
    ).to_csv(path, index=False)
    df = load_projections(str(path)).set_index("name")
    assert df.loc["A", ["team", "opponent", "ownership", "floor", "ceiling", "dk_id"]].tolist() == ["LAR", "ARI", 12, 10, 31, "1"]
    small = load_projections(str(path), ownership_column="Small Field").set_index("name")
    assert small["ownership"].to_dict() == {"A": 10, "B": 30, "C": 50}
    with pytest.raises(ValueError, match="Medium"):
        load_projections(str(path), ownership_column="Medium")
    # A fixed flex split rescales ownership within each position.
    fixed = load_settings()
    fixed["field"]["flex_split"] = {"RB": 0.4, "WR": 0.5, "TE": 0.1}
    slate = sample_slate(games=4, seed=1)
    pos = slate["position"].to_numpy()
    own = target_ownership(slate, fixed)
    assert own[pos == "TE"].sum() == pytest.approx(1.1) and own[pos == "WR"].sum() == pytest.approx(3.5)


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
    assert (np.abs(np.percentile(scores, 10, axis=1) - players["floor"]) < 0.05 * width + 0.3).all()
    assert (np.abs(np.percentile(scores, 90, axis=1) - players["ceiling"]) < 0.05 * width + 0.3).all()
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


def test_rushing_quarterback_is_tied_less_to_his_receivers(players, settings):
    rusher = players.assign(rush_share=np.where(players["name"] == "H01 QB1", 0.45, np.nan))
    scores, _ = simulate_slate(rusher, settings, 20000, seed=1)
    row = {n: i for i, n in enumerate(players["name"])}
    corr = lambda a, b: np.corrcoef(scores[row[a]], scores[row[b]])[0, 1]
    assert corr("H01 QB1", "H01 WR1") < corr("H02 QB1", "H02 WR1") - 0.08
    assert corr("H01 QB1", "H01 WR1") > 0.1


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
    assert np.abs(realised - target_ownership(players, settings)).max() < 0.12
    assert np.abs(realised - target_ownership(players, settings)).mean() < 0.005
    assert realised.sum() == pytest.approx(9)
    report = field_report(field, players, settings).set_index("measure")["value"]
    assert report[["QB with no pass catcher", "QB with exactly 1 pass catcher", "QB with 2 or more pass catchers"]].sum() == pytest.approx(100, abs=0.2)
    assert report[["Flex is a RB", "Flex is a WR", "Flex is a TE"]].sum() == pytest.approx(100, abs=0.2)
    assert report["QB with no pass catcher"] == pytest.approx(12.6, abs=0.5)
    assert report["QB with 2 or more pass catchers"] == pytest.approx(35.5, abs=0.5)
    assert report["Stacks that add an opposing pass catcher"] == pytest.approx(37.1, abs=1)
    assert report["Median salary used"] >= 49700 and report["Defense facing its own QB"] < 5
    assert 100 < report["Average salary left unspent"] < 180


def test_best_of_selection_raises_lineup_quality(players, settings):
    sampler = LineupSampler(players, settings)
    own = target_ownership(players, settings)
    logw = np.where(own > 0, np.log(np.where(own > 0, own, 1.0)), -np.inf)
    value = players["mean"].to_numpy(float)
    rng = np.random.default_rng(0)
    tilted = sampler.salary_tilt(logw, rng)
    total = lambda best_of: value[sampler.sample(tilted, 1500, rng, best_of=best_of, value=value)].sum(axis=1).mean()
    plain, some, lots = total(1), total(2.5), total(12)
    assert plain < some - 1 < lots - 2
    picked = sampler.sample(tilted, 400, rng, best_of=12, value=value)
    check_legal(picked, players, settings, settings["field"]["min_salary"])


def test_field_groups_differ_in_quality_and_duplicate_more(players, settings):
    lineups, realised, group = sample_field_by_group(players, settings, 6000, seed=1)
    names = [g["name"] for g in settings["field"]["groups"]]
    assert names == ["casual", "typical", "optimizer"]
    shares = np.bincount(group, minlength=3) / len(group)
    assert shares == pytest.approx([0.25, 0.68, 0.07], abs=0.01)
    total = players["mean"].to_numpy(float)[lineups].sum(axis=1)
    casual, typical, optimizer = (total[group == i].mean() for i in range(3))
    assert casual < typical - 1 and optimizer > typical + 5
    # Optimizer-style lineups cluster tightly at the top; casual ones are spread out.
    assert total[group == 2].std() < total[group == 1].std() < total[group == 0].std()
    uniform = load_settings()
    uniform["field"]["groups"] = None
    flat, _, labels = sample_field_by_group(players, uniform, 3000, seed=1)
    assert set(labels) == {0}
    assert players["mean"].to_numpy(float)[flat].sum(axis=1).std() < total.std()


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


def test_wide_search_finds_better_lineups_than_it_started_with(players, sim, settings, field):
    contest = Contest("t", 20, 20000, generated_payouts(20, 20000))
    build, grade = sim[0][:, -4000:], sim[0][:, :6000]
    only_optimizer = search_candidates(players, build, field[0], contest, settings, 30, 0, 0, seed=1)
    found = search_candidates(players, build, field[0], contest, settings, 30, 6000, 200, seed=1)
    check_legal(found, players, settings, settings["candidates"]["min_salary"])
    assert len(only_optimizer) <= 30 < len(found) <= 230
    assert len(np.unique(found, axis=0)) == len(found)
    results, payouts = evaluate(found, field[0], grade, contest, keep_payouts=True)
    assert payouts.shape == (len(found), 6000)
    assert payouts.mean(axis=1) / 20 - 1 == pytest.approx(results["roi_pct"] / 100, abs=1e-4)
    # Graded on slates the search never saw, the field as a whole loses the rake.
    assert results["roi_pct"].mean() > -100 * contest.rake


def test_growth_portfolio_spreads_across_outcomes():
    # A and B win in the same slates; C wins in different ones and is slightly worse.
    payouts = np.zeros((3, 100), dtype=np.float32)
    payouts[0, :10] = 1000
    payouts[1, :10] = 990
    payouts[2, 50:60] = 900
    lineups = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    roi = 100 * (payouts.mean(axis=1) / 10 - 1)
    pick = lambda bankroll: sorted(pick_portfolio_for_growth(lineups, payouts, roi, 10, 2, bankroll, 1.0, 9))
    assert pick(200) == [0, 2]
    assert pick(10_000_000) == [0, 1]
    together = portfolio_summary(payouts[[0, 1]], roi[[0, 1]], 10)
    spread = portfolio_summary(payouts[[0, 2]], roi[[0, 2]], 10)
    assert spread["profit_chance_pct"] == 20 and together["profit_chance_pct"] == 10


def test_growth_portfolio_respects_limits():
    rng = np.random.default_rng(0)
    lineups = np.array([rng.choice(30, 9, replace=False) for _ in range(200)])
    payouts = rng.exponential(5, size=(200, 300)).astype(np.float32)
    roi = 100 * (payouts.mean(axis=1) / 5 - 1)
    chosen = pick_portfolio_for_growth(lineups, payouts, roi, 5, 20, 2000, 0.5, 6)
    assert len(set(chosen)) == len(chosen) <= 20
    assert np.bincount(lineups[chosen].ravel()).max() <= 10
    assert all(len(np.intersect1d(lineups[a], lineups[b])) <= 6 for a in chosen for b in chosen if a != b)


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


DK_EXPORT = """Entry ID,Contest Name,Contest ID,Entry Fee,QB,RB,RB,WR,WR,WR,TE,FLEX,DST,,Instructions
101,Big Contest [$1 to 1st],900,$5,,,,,,,,,,,1. Column A lists entries
102,Big Contest [$1 to 1st],900,$5,,,,,,,,,,,
103,Other,901,$1,,,,,,,,,,,Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame
,,,,,,,,,,,,,,RB,Kenneth Walker III (11),Kenneth Walker III,11,RB/FLEX,9200,KC@ATL 11/15/2026 01:00PM ET,KC,26.4
,,,,,,,,,,,,,,WR,Amon-Ra St. Brown (12),Amon-Ra St. Brown,12,WR/FLEX,8000,KC@ATL 11/15/2026 01:00PM ET,ATL,20
,,,,,,,,,,,,,,DST,Jaguars  (13),Jaguars ,13,DST,4000,JAX@TEN 11/15/2026 01:00PM ET,JAX,9
,,,,,,,,,,,,,,DST,Titans  (14),Titans ,14,DST,2000,JAX@TEN 11/15/2026 01:00PM ET,TEN,6
"""


def test_reads_draftkings_export(tmp_path):
    path = tmp_path / "DKEntries.csv"
    path.write_text(DK_EXPORT)
    entries, slate = read_dk_entries(str(path))
    assert entries["entry_id"].tolist() == ["101", "102", "103"]
    assert slate["dk_id"].tolist() == ["11", "12", "13", "14"]
    assert slate.loc[0, ["team", "opponent", "salary"]].tolist() == ["KC", "ATL", 9200]
    assert slate.loc[3, ["name", "team", "opponent"]].tolist() == ["Titans", "TEN", "JAX"]


def test_projections_match_the_draftkings_slate(tmp_path):
    dk = tmp_path / "DKEntries.csv"
    dk.write_text(DK_EXPORT)
    projections = tmp_path / "p.csv"
    pd.DataFrame(
        {"Player": ["Kenneth Walker", "Amon Ra St Brown", "Jacksonville DST", "Titans", "Not Playing"],
         "Pos": ["RB", "WR", "DEF", "DST", "WR"], "Team": ["KC", "ATL", "JAC", "", "XX"],
         "Proj": [20, 15, 8, 6, 9]}
    ).to_csv(projections, index=False)
    notes = []
    players = load_projections(str(projections), read_dk_entries(str(dk))[1], log=notes.append)
    assert sorted(players["dk_id"]) == ["11", "12", "13", "14"]
    assert players.set_index("dk_id").loc["13", ["team", "opponent", "salary"]].tolist() == ["JAX", "TEN", 4000]
    assert players["game"].nunique() == 2 and "Not Playing" in notes[1]
    assert normalise_name("Amon-Ra St. Brown Jr.") == "amon ra st brown"


def test_upload_file_has_one_lineup_per_entry(tmp_path):
    dk = tmp_path / "DKEntries.csv"
    dk.write_text(DK_EXPORT)
    entries, _ = read_dk_entries(str(dk))
    out = tmp_path / "upload.csv"
    filled = write_upload(str(out), entries[entries["contest_id"] == "900"], [list("abcdefghi"), list("jklmnopqr"), list("stuvwxyz!")])
    written = pd.read_csv(out, dtype=str)
    assert filled == 2 and written["Entry ID"].tolist() == ["101", "102"]
    assert written.columns.tolist()[4:] == ["QB", "RB", "RB.1", "WR", "WR.1", "WR.2", "TE", "FLEX", "DST"]
    assert written.iloc[1, 4:].tolist() == list("jklmnopqr")


STANDINGS = """Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
1,1,ann (1/2),0,200.5,DST Bears  FLEX Tom Tight QB Quinn Back RB Ron One RB Rex Two TE Ted End WR Will A WR Wes B WR Walt C,,Quinn Back,QB,75.00%,30
2,2,ann (2/2),0,180,DST Bears  FLEX Tom Tight QB Quinn Back RB Ron One RB Rex Two TE Ted End WR Will A WR Wes B WR Walt C,,Ted End,TE,100.00%,12
3,3,bob,0,150,DST Lions  FLEX Ray Three QB Quinn Back RB Ron One RB Rex Two TE Ted End WR Will A WR Wes B WR Xavi D,,Tom Tight,FLEX,50.00%,9
4,4,cat,0,120,DST Lions  FLEX Xavi D QB Other Guy RB Ray Three RB Rex Two TE Tom Tight WR Will A WR Wes B WR Walt C,,Will A,WR,100.00%,20
5,5,dan,0,0,,,,
"""


def test_contest_standings_report(tmp_path):
    path = tmp_path / "standings.csv"
    path.write_text(STANDINGS)
    entries, players = read_standings(str(path))
    assert len(entries) == 4 and entries["user_entries"].tolist() == [2, 2, 1, 1]
    assert entries.loc[0, ["QB", "FLEX", "DST"]].tolist() == ["Quinn Back", "Tom Tight", "Bears"]
    assert players.set_index("name").loc["Tom Tight", "position"] == "TE"
    slate = pd.DataFrame(
        {"name": ["Quinn Back", "Will A", "Wes B", "Ted End", "Tom Tight", "Xavi D", "Walt C", "Ron One", "Rex Two",
                  "Bears", "Lions", "Ray Three"],
         "team": ["AAA", "AAA", "BBB", "AAA", "CCC", "BBB", "CCC", "AAA", "CCC", "CCC", "BBB", "DDD"],
         "opponent": ["BBB", "BBB", "AAA", "BBB", "DDD", "AAA", "DDD", "BBB", "DDD", "DDD", "AAA", "CCC"],
         "salary": [7000, 6000, 6000, 4000, 3000, 5000, 5000, 6500, 6000, 3000, 3000, 4500]}
    )
    overall, groups, notes = standings_report(entries, players, slate)
    assert overall["Flex is a TE"] == 50 and overall["Flex is a RB"] == 25
    assert overall["Shares its lineup with another entry"] == 50
    # Quinn Back's lineups all pair him with Will A and Ted End; the fourth QB is unknown.
    assert overall["QB with 2 or more pass catchers"] == 100
    assert overall["Stacks that add an opposing pass catcher"] == 100
    assert overall["QB with his own RB"] == 100 and overall["Defense facing its own QB"] == pytest.approx(33.3)
    assert overall["Median salary used"] == 46500
    assert groups.loc["Share of entries", "1 entry"] == 50 and groups.loc["Average points", "2-19 entries"] == 190.25
    # A slate from another week (salaries over the cap) drops salary and opponent measures.
    overall, _, notes = standings_report(entries, players, slate.assign(salary=slate["salary"] * 2))
    assert "Median salary used" not in overall and "Stacks that add an opposing pass catcher" not in overall
    assert any("does not look like this contest" in n for n in notes)


def test_reads_supplied_lineups_in_several_forms(tmp_path, players, settings, field):
    letters = lambda i: chr(97 + i // 26) + chr(97 + i % 26)  # real names have no digits
    slate = players.assign(
        dk_id=[str(1000 + i) for i in range(len(players))],
        name=[f"Player {letters(i)}son" for i in range(len(players))],
    )
    good = field[0][0].tolist()
    name, ids = slate["name"].to_numpy(), slate["dk_id"].to_numpy()
    upload = tmp_path / "upload.csv"
    upload.write_text(
        "Entry ID,Contest Name,Contest ID,Entry Fee,QB,RB,RB,WR,WR,WR,TE,FLEX,DST\n"
        + "1,C,9,$5," + ",".join(ids[good]) + "\n"
        + "2,C,9,$5," + ",".join(f"{name[i]} ({ids[i]})" for i in good) + "\n"
        + "3,C,9,$5,,,,,,,,,\n"
        + "4,C,9,$5," + ",".join(["Nobody Atall"] + list(ids[good[1:]])) + "\n"
    )
    lineups, problems = read_lineups(str(upload), slate)
    assert [sorted(l) for l in lineups] == [good, good]
    assert len(problems) == 1 and "Nobody Atall" in problems[0] and "row 5" in problems[0]
    plain = tmp_path / "names.csv"
    plain.write_text(",".join(name[good]) + "\n")
    assert sorted(read_lineups(str(plain), slate.drop(columns="dk_id"))[0][0]) == good
    assert check_lineup(good, slate) is None
    qbs = slate.index[slate["position"] == "QB"][:2].tolist()
    assert check_lineup(qbs + good[2:], slate).startswith("positions are")
    assert "over the cap" in check_lineup(good, slate, cap=1000)


SHOWDOWN = """Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
1,1,ann (1/2),0,90,CPT Al Arm FLEX Bo Back FLEX Cy Catch FLEX Di End FLEX Home D  FLEX Ed Kick,,Al Arm,FLEX,25.00%,20
2,2,ann (2/2),0,80,CPT Al Arm FLEX Cy Catch FLEX Bo Back FLEX Di End FLEX Home D  FLEX Ed Kick,,Al Arm,CPT,75.00%,30
3,3,bob,0,70,CPT Bo Back FLEX Al Arm FLEX Cy Catch FLEX Di End FLEX Home D  FLEX Ed Kick,,Bo Back,FLEX,50.00%,10
4,4,cat,0,60,CPT Al Arm FLEX Bo Back FLEX Cy Catch FLEX Fy Far FLEX Gus Go FLEX Ed Kick,,Bo Back,CPT,25.00%,15
5,5,dan,0,0,,,,
"""


def test_captain_mode_standings_report(tmp_path):
    path = tmp_path / "showdown.csv"
    path.write_text(SHOWDOWN)
    entries, players = read_standings(str(path))
    assert entries.attrs["format"] == "showdown" and len(entries) == 4
    assert entries.loc[2, ["CPT", "FLEX1", "FLEX5"]].tolist() == ["Bo Back", "Al Arm", "Ed Kick"]
    arm = players.set_index("name").loc["Al Arm"]
    assert (arm["captain_ownership"], arm["flex_ownership"], arm["ownership"], arm["points"]) == (75, 25, 100, 20)
    slate = pd.DataFrame(
        {"name": ["Al Arm", "Bo Back", "Cy Catch", "Di End", "Home D", "Ed Kick", "Fy Far", "Gus Go"],
         "team": ["HOM", "HOM", "HOM", "AWY", "HOM", "AWY", "AWY", "AWY"],
         "position": ["QB", "RB", "WR", "TE", "DST", "K", "WR", "RB"],
         "salary": [10000, 9000, 8000, 5000, 4000, 4000, 6000, 5000]}
    )
    overall, groups, notes = showdown_report(entries, players, slate)
    # Entries 1 and 2 are the same lineup in a different order; entry 3 has the same six
    # players under a different captain, which is a different lineup.
    assert overall["Shares its lineup with another entry"] == 50
    assert overall["Same six players as another entry (any captain)"] == 75
    assert overall["4 from the captain's team, 2 from the other"] == 75
    assert overall["3 from the captain's team, 3 from the other"] == 25
    assert overall["Captain is a QB"] == 75 and overall["Captain is a RB"] == 25
    assert overall["Average salary left unspent"] == pytest.approx(50000 - (45000 * 2 + 44500 + 47000) / 4)
    assert groups.loc["Share of entries", "2-19 entries"] == 50
    assert "2 times" in notes[0]


def slot_lineup(players, rng, games=None):
    """A lineup in roster-slot order (QB, RB, RB, WR, WR, WR, TE, FLEX, DST)."""
    pool = players if games is None else players[players["game"].isin(games)]
    pick = lambda pos, n: rng.choice(pool.index[pool["position"] == pos], n, replace=False).tolist()
    wr = pick("WR", 4)
    return pick("QB", 1) + pick("RB", 2) + wr[:3] + pick("TE", 1) + wr[3:] + pick("DST", 1)


def test_time_left_in_each_game_is_recovered_from_player_minutes(players):
    rng = np.random.default_rng(0)
    games = sorted(players["game"].unique())
    truth = pd.Series([0.0, 0.0, 0.15, 0.15, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0], index=games)
    lineups = np.array([slot_lineup(players, rng) for _ in range(400)])
    minutes = (60 * players["game"].map(truth).to_numpy()[lineups]).sum(axis=1)
    entries = pd.DataFrame({"minutes_remaining": minutes})
    assert live.game_time_left(entries, lineups, players).to_dict() == pytest.approx(truth.to_dict(), abs=1e-6)


def test_remaining_projection_shrinks_with_time_left(players):
    games = sorted(players["game"].unique())
    left = pd.Series([0.0, 0.25] + [1.0] * 8, index=games)
    rest = live.remaining_projections(players, left)
    done, part, fresh = (players["game"] == g for g in games[:3])
    assert (rest.loc[done, "mean"] == 0).all() and (rest.loc[done, "ceiling"] == 0).all()
    assert rest.loc[part, "mean"].to_numpy() == pytest.approx(players.loc[part, "mean"].to_numpy() / 4)
    spread = lambda df: (df["ceiling"] - df["mean"]).to_numpy()
    assert spread(rest[part]) == pytest.approx(spread(players[part]) / 2)
    cols = ["mean", "floor", "ceiling"]
    assert rest.loc[fresh, cols].to_numpy(float) == pytest.approx(players.loc[fresh, cols].to_numpy(float))


def test_finished_players_keep_their_score(players, settings):
    games = sorted(players["game"].unique())
    left = pd.Series([0.0] * 5 + [1.0] * 5, index=games)
    points = np.arange(len(players), dtype=float)
    scores = live.simulate_finish(players, points, left, settings, 300, seed=1)
    over = players["game"].isin(games[:5]).to_numpy()
    assert (scores[over] == points[over, None]).all()
    star = (~over) & (players["mean"] > 10).to_numpy()
    assert (scores[star].mean(axis=1) > points[star] + 5).all() and scores[star].std(axis=1).min() > 1


def test_swaps_stay_legal_and_find_the_better_player(players, settings):
    rng = np.random.default_rng(1)
    games = sorted(players["game"].unique())
    left = pd.Series([0.0] * 5 + [1.0] * 5, index=games)
    time_left = players["game"].map(left).to_numpy(float)
    slate = players.copy()
    late_wr = slate.index[(slate["position"] == "WR") & slate["game"].isin(games[5:])]
    dud, star = late_wr[0], late_wr[1]
    slate.loc[dud, ["mean", "floor", "ceiling", "salary"]] = [4.0, 1.0, 8.0, 5000]
    slate.loc[star, ["mean", "floor", "ceiling", "salary"]] = [30.0, 20.0, 42.0, 4000]
    mine = np.array(slot_lineup(slate.drop(index=[dud, star]), rng))
    mine[3] = dud  # a weak receiver who has not started, in the first WR slot
    cap = int(slate["salary"].to_numpy()[mine].sum()) + 500
    options = live.swap_options(mine, slate, time_left, cap, min_mean=4.0)
    pos, salary = slate["position"].to_numpy(), slate["salary"].to_numpy()
    assert options and (3, star) in options
    for slot, incoming in options:
        assert time_left[mine[slot]] == 1 and time_left[incoming] == 1 and incoming not in mine
        assert pos[incoming] in live.SLOT_POSITIONS[live.SLOTS[slot]]
        assert salary[incoming] - salary[mine[slot]] <= 500
    field = np.array([slot_lineup(slate.drop(index=[dud, star]), rng) for _ in range(60)])
    scores = live.simulate_finish(slate, np.zeros(len(slate)), left, settings, 600, seed=1)
    contest = Contest("t", 5, 61, np.array([150.0, 60, 30, 20, 10]))
    swapped, table = live.suggest_swaps(mine[None, :], field, scores, slate, time_left, contest, cap, rounds=1)
    assert table.loc[0, ["slot", "out", "in"]].tolist() == ["WR1", dud, star] and bool(table.loc[0, "made"])
    assert swapped[0].tolist() == mine[:3].tolist() + [star] + mine[4:].tolist()
