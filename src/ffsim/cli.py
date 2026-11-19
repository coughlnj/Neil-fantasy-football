"""Simulate a DraftKings slate and rank tournament lineups by expected ROI."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .candidates import search_candidates
from .contest import load_contest
from .draftkings import check_lineup, read_dk_entries, read_lineups, write_upload
from . import live
from .field import field_report, sample_field, sample_field_scenarios, target_ownership
from .projections import diff_projections, load_projections
from .sample import sample_slate
from .roi import (adjust_roi, describe_lineups, evaluate, exposure_limits, pick_portfolio, pick_portfolio_for_growth,
                  portfolio_summary)
from .settings import load_settings
from .standings import read_standings, showdown_report, standings_report
from .simulate import fit_report, load_simulation, save_simulation, simulate_slate


def cmd_simulate(args) -> None:
    settings = load_settings(args.config)
    dk_players = read_dk_entries(args.dk)[1] if args.dk else None
    players = load_projections(args.projections, dk_players, log=print, ownership_column=args.ownership)
    os.makedirs(args.out, exist_ok=True)
    if args.engine == "pbp":
        from .pbp.data import load_pbp
        from .pbp.slate import simulate_slate_pbp

        scores, report = simulate_slate_pbp(players, load_pbp(args.pbp_data), settings, args.sims, args.seed, log=print)
        save_simulation(os.path.join(args.out, "simulation.npz"), players, scores)
        report.to_csv(os.path.join(args.out, "simulation_report.csv"), index=False)
        used = report[report["from_games"] & (report["mean"] >= 5)]
        gap = (used["simulator_own_mean"] - used["mean"]).abs()
        print(f"Played {args.sims:,} slates snap by snap: {len(players)} players, {players['game'].nunique()} games.")
        print(f"Before anchoring, the simulator's own averages for players projected 5+ were within "
              f"{gap.mean():.1f} points of the projections on average (largest gap {gap.max():.1f}).")
        print(f"Saved to {args.out}/simulation.npz and {args.out}/simulation_report.csv")
        return
    scores, fit = simulate_slate(players, settings, args.sims, args.seed)
    save_simulation(os.path.join(args.out, "simulation.npz"), players, scores)
    report = fit_report(players, fit, scores, settings)
    report.to_csv(os.path.join(args.out, "simulation_report.csv"), index=False)

    if players["rush_share"].notna().any():
        print(f"Rushing share supplied for {int(players['rush_share'].notna().sum())} players.")
    else:
        print("No rush_share column: every quarterback is treated as a typical passer.")
    print(f"Simulated {args.sims:,} slates: {len(players)} players, {players['game'].nunique()} games.")
    print(f"Largest gap between simulated and projected mean: "
          f"{(report['sim_mean'] - report['mean']).abs().max():.2f} points")
    for flag, text in (("used_fallback", "had a missing or inconsistent floor/ceiling (typical spread used)"),
                       ("shape_adjusted", "had a mean too far off-centre to match floor and ceiling exactly")):
        if report[flag].any():
            print(f"{int(report[flag].sum())} players {text}.")
    print(f"Saved to {args.out}/simulation.npz and {args.out}/simulation_report.csv")


def cmd_lineups(args) -> None:
    settings = load_settings(args.config)
    players, scores = load_simulation(args.simulation)
    contest = load_contest(args.contest)
    n_sims = scores.shape[1]
    held_out = max(settings["candidates"]["screen_sims"], n_sims // 5)
    if held_out >= n_sims:
        raise SystemExit("Too few simulations; rerun 'ffsim simulate' with more --sims.")
    grade, build = scores[:, : n_sims - held_out], scores[:, n_sims - held_out :]

    print(f"{contest.name}: ${contest.entry_fee:g} entry, {contest.entries:,} entries, "
          f"${contest.prize_pool:,.0f} prizes ({100 * contest.rake:.1f}% rake)")
    field_size = min(args.field or contest.entries - 1, contest.entries - 1)
    field, realised = sample_field(players, settings, field_size, args.seed)
    target = target_ownership(players, settings)
    implied = float((target * players["salary"]).sum())
    if implied < settings["field"]["min_salary"]:
        print(f"Note: your ownership implies an average lineup costing only ${implied:,.0f}, well under the "
              "cap, so the field cannot match it closely. Check that ownership covers the whole slate.")
    if implied > settings["site"]["salary_cap"]:
        print(f"Note: your ownership implies an average lineup costing ${implied:,.0f}, over the cap, "
              "so the field cannot match it exactly.")
    print(f"Sampled {field_size:,} opponent lineups; ownership within "
          f"{100 * np.abs(realised - target).max():.1f} points of your projections.")

    candidates = search_candidates(
        players, build, field, contest, settings, args.optimizer_lineups, args.sampled, args.candidates,
        args.seed, log=print,
    )
    graders = sample_field_scenarios(players, settings, field_size, args.seed, log=print)
    print(f"Grading {len(candidates):,} lineups over {grade.shape[1]:,} simulated slates"
          + (f", shared between {len(graders)} versions of the field's ownership..." if len(graders) > 1 else "..."))
    results = evaluate(candidates, graders, grade, contest)
    overrides = {}
    for item in args.exposure or []:
        name, _, share = item.rpartition("=")
        match = players.index[players["name"].str.strip().str.lower() == name.strip().lower()]
        if len(match) != 1 or not share:
            raise SystemExit(f"--exposure '{item}': expected NAME=SHARE with a player's exact name, e.g. \"Josh Allen=0.8\"")
        overrides[int(match[0])] = float(share)
        print(f"Exposure limit for {players['name'][match[0]].strip()}: {100 * float(share):.0f}%")
    limits = (args.portfolio, args.max_exposure, args.max_overlap, overrides)
    bankroll = args.bankroll if args.bankroll is not None else 20 * args.portfolio * contest.entry_fee

    at_limit = np.array([], dtype=int)
    for top_up in range(5):
        table = describe_lineups(candidates, players, realised)
        slots = table.attrs["slots"]
        adjusted = adjust_roi(results, table[["projection", "total_ownership_pct"]])
        table["adj_roi_pct"] = adjusted.round(2)
        table = pd.concat([table, results.round(3)], axis=1)

        # Choose the portfolio from the strongest lineups, judged together.
        order = np.argsort(-adjusted)
        finalists = order[: max(args.finalists, args.portfolio)]
        if len(at_limit):     # also the best lineups that avoid the players already at their limit
            free = order[~np.isin(candidates[order], at_limit).any(axis=1)]
            finalists = np.concatenate([finalists, free[~np.isin(free, finalists)][: max(args.finalists, args.portfolio)]])
        _, payouts = evaluate(candidates[finalists], graders, grade, contest, keep_payouts=True)
        simple = pick_portfolio(candidates[finalists], adjusted[finalists], *limits)
        if bankroll > 0:
            chosen = pick_portfolio_for_growth(
                candidates[finalists], payouts, adjusted[finalists], contest.entry_fee, args.portfolio, bankroll,
                args.max_exposure, args.max_overlap, overrides,
            )
        else:
            chosen = simple
        if len(chosen) >= args.portfolio or top_up == 4:
            break
        # Short of a full set: the strongest lineups lean on players already at their
        # exposure limit. Search again without those players and add what is found.
        used = np.bincount(candidates[finalists[chosen]].ravel(), minlength=len(players))
        full = np.flatnonzero(used >= exposure_limits(args.portfolio, args.max_exposure, len(players), overrides))
        at_limit = np.union1d(at_limit, full)
        print(f"\nOnly {len(chosen)} lineups fit the exposure and overlap limits: the strongest lineups all use "
              + ", ".join(players["name"].str.strip().to_numpy()[full]) + ", now at their limit. Searching for lineups without them...")
        masked = build.copy()
        masked[full] = -1000.0          # the optimizer and the screening steer clear of them
        extra = search_candidates(players, masked, field, contest, settings, args.optimizer_lineups, args.sampled,
                                  args.candidates, None if args.seed is None else args.seed + 1 + top_up, log=print)
        extra = extra[~np.isin(extra, full).any(axis=1)]
        known = {tuple(row) for row in np.sort(candidates, axis=1)}
        extra = np.array([row for row in extra if tuple(np.sort(row)) not in known], dtype=int).reshape(-1, candidates.shape[1])
        if len(extra) == 0:
            break
        print(f"Grading {len(extra):,} more lineups...")
        candidates = np.vstack([candidates, extra])
        results = pd.concat([results, evaluate(extra, graders, grade, contest)], ignore_index=True)
    if len(chosen) < args.portfolio:
        print(f"Note: only {len(chosen)} lineups fit the exposure and overlap limits. Raise --finalists, "
              "--max-exposure or --max-overlap for a full set.")
    table["in_portfolio"] = False
    table.loc[finalists[chosen], "in_portfolio"] = True
    table = table.sort_values("adj_roi_pct", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False)
    picked = table[table["in_portfolio"]]
    picked.drop(columns=["in_portfolio"]).to_csv(args.portfolio_out, index=False)

    compare = pd.DataFrame(
        {
            "chosen as a set": portfolio_summary(payouts[chosen], adjusted[finalists][chosen], contest.entry_fee),
            "top lineups one by one": portfolio_summary(payouts[simple], adjusted[finalists][simple], contest.entry_fee),
        }
    ).T
    if bankroll > 0:
        print(f"\nPortfolio chosen for a ${bankroll:,.0f} bankroll, compared with simply taking the top lineups:\n")
        print(compare.to_string())
    else:
        print("\nPortfolio (top lineups one by one):\n")
        print(compare.iloc[[1]].to_string())
    if args.entries:
        write_dk_upload(args, players, contest, slots[finalists[chosen]], adjusted[finalists][chosen])
    if args.show:
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print("\n" + picked.head(args.show).drop(columns=["in_portfolio"]).to_string(index=False))
    print(f"\nPortfolio saved to {args.portfolio_out}; all {len(table):,} graded lineups to {args.out}")


def write_dk_upload(args, players, contest, slots, value) -> None:
    """Write the portfolio into a DraftKings upload file, best lineup first."""
    if "dk_id" not in players.columns:
        raise SystemExit("No DraftKings IDs in this simulation; rerun 'ffsim simulate' with --dk.")
    entries, _ = read_dk_entries(args.entries)
    if args.contest_id:
        mine = entries[entries["contest_id"] == str(args.contest_id)]
    else:
        mine = entries[[contest.name.startswith(n) or n in contest.name for n in entries["contest_name"]]]
    if mine.empty:
        raise SystemExit("No entries for this contest in the entries file. Contests in it:\n" + entries.groupby(
            ["contest_id", "contest_name"]).size().rename("entries").reset_index().to_string(index=False))
    ids = players["dk_id"].astype(str).to_numpy()[slots[np.argsort(-value)]]
    filled = write_upload(args.upload_out, mine, ids.tolist())
    print(f"\nDraftKings upload file: {filled} of {len(mine)} entries in '{mine['contest_name'].iloc[0]}' "
          f"filled, saved to {args.upload_out}")


def cmd_slate(args) -> None:
    entries, players = read_dk_entries(args.entries)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    players.to_csv(args.out, index=False)
    games = players.groupby(["team", "opponent"]).ngroups // 2
    print(f"{len(players)} players in {games} games saved to {args.out}\n")
    print(entries.groupby(["contest_id", "contest_name", "entry_fee"]).size().rename("entries").reset_index().to_string(index=False))


def cmd_grade(args) -> None:
    settings = load_settings(args.config)
    players, scores = load_simulation(args.simulation)
    contest = load_contest(args.contest)
    found, problems = read_lineups(args.lineups, players)
    legal = []
    for lineup in found:
        reason = check_lineup(lineup, players, settings["site"]["salary_cap"])
        if reason:
            problems.append("not a legal lineup: " + reason + " (" + ", ".join(players["name"].to_numpy()[lineup]) + ")")
        else:
            legal.append(sorted(lineup))
    for problem in problems[:15]:
        print("Skipped " + problem)
    if len(problems) > 15:
        print(f"...and {len(problems) - 15} more skipped.")
    if not legal:
        raise SystemExit("No gradable lineups found.")
    lineups = np.array(legal)

    print(f"{contest.name}: ${contest.entry_fee:g} entry, {contest.entries:,} entries, "
          f"${contest.prize_pool:,.0f} prizes ({100 * contest.rake:.1f}% rake)")
    field_size = min(args.field or contest.entries - 1, contest.entries - 1)
    field, realised = sample_field(players, settings, field_size, args.seed)
    print(f"Grading {len(lineups):,} lineups against {field_size:,} opponent lineups over "
          f"{scores.shape[1]:,} simulated slates...")
    graders = sample_field_scenarios(players, settings, field_size, args.seed)
    if len(graders) > 1:
        print(f"The slates are shared between {len(graders)} versions of the field's ownership.")
    results, payouts = evaluate(lineups, graders, scores, contest, keep_payouts=True)
    table = describe_lineups(lineups, players, realised)
    adjusted = adjust_roi(results, table[["projection", "total_ownership_pct"]])
    table["adj_roi_pct"] = adjusted.round(2)
    table = pd.concat([table, results.round(3)], axis=1)
    summary = portfolio_summary(payouts, results["roi_pct"].to_numpy(), contest.entry_fee)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"\nAll {summary['lineups']} lineups together:")
    print(f"  Expected ROI             {summary['expected_roi_pct']:+.1f}%")
    print(f"  Chance of profit         {summary['profit_chance_pct']:.1f}%")
    print(f"  Median return            {summary['median_return_pct']:+.1f}%")
    print(f"  Chance of a 5x return    {summary['chance_of_5x_pct']:.2f}%")
    print(f"  Chance of losing half    {summary['chance_of_losing_half_pct']:.1f}%")
    print(f"  Lineups with positive expected ROI: {int((results['roi_pct'] > 0).sum())} of {len(lineups)}")
    counts = pd.Series(players["name"].to_numpy()[lineups].ravel()).value_counts()
    print("\nHighest exposures: " + ", ".join(f"{n} {100 * c / len(lineups):.0f}%" for n, c in counts.head(8).items()))
    show = ["QB", "salary", "projection", "total_ownership_pct", "adj_roi_pct", "roi_pct", "roi_se_pct", "top1_pct",
            "cash_pct", "expected_dupes"]
    ranked = table.sort_values("roi_pct", ascending=False)
    with pd.option_context("display.width", 200):
        print(f"\nBest {args.show} by ROI:\n" + ranked.head(args.show)[show].to_string(index=False))
        print(f"\nWorst {args.show} by ROI:\n" + ranked.tail(args.show)[show].to_string(index=False))
    print(f"\nEvery lineup saved to {args.out}, in the order given.")


def cmd_live(args) -> None:
    settings = load_settings(args.config)
    contest = load_contest(args.contest)
    my_entries, dk_players = read_dk_entries(args.entries)
    projected = load_projections(args.projections, dk_players)
    keep = ["dk_id", "mean", "floor", "ceiling", "ownership"] + (["rush_share"] if "rush_share" in projected.columns else [])
    slate = dk_players.merge(projected[keep], on="dk_id", how="left")
    slate["game"] = [" v ".join(sorted(pair)) for pair in zip(slate["team"], slate["opponent"])]

    entries, scored = read_standings(args.standings)
    if entries.attrs.get("format") != "classic":
        raise SystemExit("The in-game tool handles full-roster contests only.")
    slate = live.add_unlisted_players(entries, slate)
    lineups = live.lineup_indices(entries, slate)
    points = dict(zip(scored["name"].map(live.normalise_name), scored["points"]))
    live_points = slate["name"].map(live.normalise_name).map(points).fillna(0.0).to_numpy(float)
    left = live.game_time_left(entries, lineups, slate)
    if "unlisted" in left.index:        # late additions: treat as a game in progress like the others
        started = left[(left < 1) & (left.index != "unlisted")]
        left["unlisted"] = float(started.mean()) if len(started) else 0.0
    time_left = slate["game"].map(left).to_numpy(float)
    hidden_share = float((lineups < 0).mean())

    kickoff = slate.groupby("game")["kickoff"].first()
    status = pd.DataFrame({"kickoff": kickoff, "time_left_pct": (100 * left).round(0)}).sort_values(["time_left_pct", "kickoff"])
    print(f"{len(entries):,} entries. Games:\n")
    print(status.to_string())
    shown_points = np.where(lineups >= 0, live_points[np.maximum(lineups, 0)], 0.0).sum(axis=1)
    gap = float(np.abs(shown_points - entries["points"]).max())
    if gap > 0.5:
        print(f"\nNote: player scores and entry totals in the file differ by up to {gap:.1f} points.")

    mine = entries["entry_id"].isin(set(my_entries["entry_id"])) if not args.user else entries["user"] == args.user
    if not mine.any():
        raise SystemExit("None of your entries are in this contest's standings. Use --user NAME to pick them by username.")
    unprojected = slate.loc[np.unique(lineups[mine.to_numpy()][lineups[mine.to_numpy()] >= 0]), :]
    unprojected = unprojected[unprojected["mean"].isna() & (unprojected["game"].map(left) > 0)]
    if len(unprojected):
        print("\nNo projection for these players of yours still to play (treated as 0): " + ", ".join(unprojected["name"]))

    my_lineups, field = lineups[mine.to_numpy()], lineups[~mine.to_numpy()]
    rng = np.random.default_rng(args.seed)
    guessed = np.zeros(len(my_lineups), dtype=bool)
    if hidden_share > 0:
        print(f"\n{100 * hidden_share:.0f}% of roster spots are hidden (players whose games have not started).")
        weights = slate["ownership"].fillna(0).to_numpy(float) if "ownership" in slate.columns else np.zeros(len(slate))
        if (my_lineups < 0).any():
            if not args.my_lineups:
                raise SystemExit("Your own late-game players are hidden too. Pass the lineup file you uploaded with --my-lineups.")
            uploaded, problems = read_lineups(args.my_lineups, slate)
            shown = my_lineups.copy()
            my_lineups = live.match_my_lineups(shown, np.array(uploaded))
            guessed = (my_lineups < 0).any(axis=1)
            print(f"Matched {int((~guessed).sum())} of your {len(my_lineups)} entries to the lineups in {args.my_lineups}.")
        if guessed.any():
            ids = entries["entry_id"].to_numpy()[mine.to_numpy()][guessed]
            print(f"{int(guessed.sum())} of your entries match no lineup in that file (edited since?). Their hidden players are "
                  "guessed, so treat their figures as rough and no swaps are suggested for them: " + ", ".join(ids))
            my_lineups[guessed] = live.fill_hidden(my_lineups[guessed], slate, time_left, weights, rng, settings["site"]["salary_cap"])
        field = live.fill_hidden(field, slate, time_left, weights, rng, settings["site"]["salary_cap"])
        still = int((field < 0).any(axis=1).sum())
        if still:
            print(f"Dropped {still} opponent lineups whose hidden players could not be filled.")
            field = field[(field >= 0).all(axis=1)]
        print("Opponents' hidden players are filled in from projected ownership, so the late games are less certain than the early ones.")
    slate["mean"] = slate["mean"].astype(float)
    scores = live.simulate_finish(slate, live_points, left, settings, args.sims, args.seed)
    mine_rows = entries[mine.to_numpy()].reset_index(drop=True)
    results = live.grade_live(my_lineups, field, scores, contest)
    rest = live.remaining_projections(slate, left)
    to_start = time_left[my_lineups] >= 1.0
    salary, names = slate["salary"].to_numpy(), slate["name"].to_numpy()
    table = pd.DataFrame({
        "entry_id": mine_rows["entry_id"], "rank_now": mine_rows["rank"], "points_now": mine_rows["points"],
        "players_to_start": to_start.sum(axis=1),
        "salary_to_start": (salary[my_lineups] * to_start).sum(axis=1),
        "minutes_left": mine_rows["minutes_remaining"].round(0),
        "projected_to_come": rest["mean"].to_numpy()[my_lineups].sum(axis=1).round(1),
    })
    table["projected_final"] = (table["points_now"] + table["projected_to_come"]).round(1)
    table["expected_prize"] = (contest.entry_fee * (1 + results["roi_pct"] / 100)).round(2)
    table["top1_pct"] = results["top1_pct"].round(2)
    table["cash_pct"] = results["cash_pct"].round(1)
    table["still_to_start"] = [", ".join(names[l[t]]) for l, t in zip(my_lineups, to_start)]
    table["late_players_guessed"] = guessed
    os.makedirs(args.out_dir, exist_ok=True)
    table.sort_values("expected_prize", ascending=False).to_csv(os.path.join(args.out_dir, "live_lineups.csv"), index=False)

    cost = contest.entry_fee * len(table)
    print(f"\nYour {len(table)} lineups: expected prizes ${table['expected_prize'].sum():,.0f} on ${cost:,.0f} of entries "
          f"({100 * (table['expected_prize'].sum() / cost - 1):+.0f}%).")
    show = ["entry_id", "rank_now", "points_now", "players_to_start", "salary_to_start", "minutes_left",
            "projected_to_come", "projected_final", "expected_prize", "top1_pct", "cash_pct"]
    with pd.option_context("display.width", 220):
        print(f"\nBest {args.show} by expected prize:\n" + table.sort_values("expected_prize", ascending=False).head(args.show)[show].to_string(index=False))

    if args.max_swaps > 0 and to_start.any():
        sure = np.flatnonzero(~guessed)
        if args.swap_mode == "full":
            swapped, swaps = live.late_swap(my_lineups[sure], field, scores, slate, time_left, contest,
                                            settings["site"]["salary_cap"], args.min_projection, seed=args.seed)
        else:
            swapped, swaps = live.suggest_swaps(my_lineups[sure], field, scores, slate, time_left, contest,
                                                settings["site"]["salary_cap"], args.min_projection, args.max_swaps)
            swaps["changes"] = 1
        swaps["lineup"] = sure[swaps["lineup"].to_numpy(int)] if len(swaps) else swaps["lineup"]
        full = my_lineups.copy()
        full[sure] = swapped
        swapped = full
        if swaps.empty:
            print("\nNo late swap improves any lineup.")
        else:
            swaps["entry_id"] = mine_rows["entry_id"].to_numpy()[swaps["lineup"]]
            swaps["out_name"], swaps["in_name"] = names[swaps["out"]], names[swaps["in"]]
            swaps["salary_change"] = salary[swaps["in"]] - salary[swaps["out"]]
            swaps = swaps.sort_values(["made", "gain"], ascending=False)
            cols = ["entry_id", "slot", "out_name", "in_name", "salary_change", "changes", "gain", "gain_se", "made"]
            swaps[cols].round(2).to_csv(os.path.join(args.out_dir, "live_swaps.csv"), index=False)
            sure, maybe = swaps[swaps["made"]], swaps[~swaps["made"]]
            per_lineup = args.swap_mode == "full"         # full mode: one gain per lineup, repeated on its rows
            worth = sure.drop_duplicates("entry_id")["gain"].sum() if per_lineup else sure["gain"].sum()
            left_alone = maybe["entry_id"].nunique() if per_lineup else len(maybe)
            print(f"\n{len(sure)} player changes recommended across {sure['entry_id'].nunique()} lineups, worth ${worth:,.2f} "
                  f"in expected prizes. {left_alone} more {'lineups' if per_lineup else 'swaps'} looked better but were within "
                  "the noise and are left alone.")
            if per_lineup and len(sure):
                sizes = sure.drop_duplicates("entry_id")["changes"].value_counts().sort_index()
                print("Lineups by number of players changed: " + ", ".join(f"{n} changed: {c}" for n, c in sizes.items()))
            if len(sure):
                with pd.option_context("display.width", 220):
                    print("\nLargest recommended:\n" + sure[cols[:-1]].head(args.show).round(2).to_string(index=False))
                pairs = sure.groupby(["out_name", "in_name"]).agg(lineups=("gain", "size"), total_gain=("gain", "sum"))
                if per_lineup:       # a lineup's gain belongs to all its changes together
                    pairs = pairs[["lineups"]].sort_values("lineups", ascending=False)
                else:
                    pairs = pairs.sort_values("total_gain", ascending=False)
                print("\nMost common recommended changes:\n" + pairs.head(10).round(2).to_string())
                changed = sorted(set(sure["lineup"]))
                rows = my_entries.set_index("entry_id").reindex(mine_rows["entry_id"].to_numpy()[changed]).reset_index()
                rows = rows.fillna({"contest_name": contest.name, "contest_id": "", "entry_fee": f"${contest.entry_fee:g}"})
                ids = slate["dk_id"].astype(str).to_numpy()[swapped[changed]]
                path = os.path.join(args.out_dir, "dk_swap_upload.csv")
                write_upload(path, rows, ids.tolist())
                print(f"\nDraftKings upload file for the {len(changed)} changed lineups: {path}")
    print(f"\nPer-lineup detail: {os.path.join(args.out_dir, 'live_lineups.csv')}")


def cmd_pbp_check(args) -> None:
    from .pbp import data, models, validate

    plays = data.load_pbp(args.data, args.seasons)
    model = models.fit_game_model(plays)
    real = validate.real_box_scores(plays)
    sim = validate.simulated_box_scores(model, args.sims, args.seed)
    table, checks = validate.compare(real, sim)
    print(f"Fitted to {model.summary['plays']:,} plays from {model.summary['games']} games; "
          f"{len(real):,} real team-games against {args.sims:,} simulated league-average games.\n")
    print("Per team per game:\n")
    print(table.to_string())
    print("\nGame flow:\n")
    print(checks.to_string())
    try:
        sd = float(load_settings(None).get("pbp", {}).get("game_plan_sd", 0.0))
        swings = validate.around_the_line(plays, model, sd, seed=args.seed)
    except ValueError as error:
        print(f"\nSkipping the check against betting lines: {error}.")
        return
    print(f"\nHow much games vary around the betting line, with each real game replayed 30 times "
          f"(game-plan variation {sd:g}):\n")
    print(swings.to_string())


def cmd_field(args) -> None:
    settings = load_settings(args.config)
    players, _ = load_simulation(args.simulation)
    field, realised = sample_field(players, settings, args.field, args.seed)
    print(f"How {args.field:,} simulated opponent lineups are built:\n")
    print(field_report(field, players, settings).to_string(index=False))
    own = players[["name", "position", "salary"]].copy()
    own["projected_own_pct"] = (100 * target_ownership(players, settings)).round(1)
    own["field_own_pct"] = (100 * realised).round(1)
    own["gap"] = (own["field_own_pct"] - own["projected_own_pct"]).round(1)
    print("\nLargest ownership gaps (field minus your projection):\n")
    print(own.reindex(own["gap"].abs().sort_values(ascending=False).index).head(args.show).to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        names = (players["name"] + " (" + players["team"] + ")").to_numpy()
        pd.DataFrame(names[field]).to_csv(args.out, index=False, header=False)
        print(f"\nField lineups saved to {args.out}")


def cmd_contest_report(args) -> None:
    entries, players = read_standings(args.standings)
    slate = pd.read_csv(args.slate) if args.slate else None
    showdown = entries.attrs.get("format") == "showdown"
    overall, groups, notes = (showdown_report if showdown else standings_report)(entries, players, slate)
    users = entries["user"].nunique()
    print(f"{len(entries):,} {'captain-mode ' if showdown else ''}entries from {users:,} entrants.\n")
    table = pd.concat([overall, groups], axis=1)
    with pd.option_context("display.width", 200):
        print(table.to_string())
    for note in notes:
        print("\n" + note)
    if not showdown:
        by_pos = players.groupby("position")["ownership"].sum().round(1)
        print("\nActual ownership by position (percent): " + ", ".join(f"{p} {v:g}" for p, v in by_pos.items()))
    print("\nMost-owned players:\n")
    print(players.sort_values("ownership", ascending=False).head(args.show).to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        table.to_csv(args.out)


def archive_projections(path: str, folder: str, slate_date: str) -> tuple[str, str | None]:
    """Keep a dated copy of a projections file. Returns (saved copy, the version before it
    or None). A file identical to the latest saved version is not saved again."""
    import glob
    import hashlib
    import re
    import shutil

    os.makedirs(folder, exist_ok=True)
    version = lambda f: int(re.search(r"_v(\d+)\.csv$", f).group(1))
    earlier = sorted(glob.glob(os.path.join(folder, f"{slate_date}_v*.csv")), key=version)
    digest = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
    if earlier and digest(earlier[-1]) == digest(path):
        return earlier[-1], (earlier[-2] if len(earlier) > 1 else None)
    saved = os.path.join(folder, f"{slate_date}_v{version(earlier[-1]) + 1 if earlier else 1}.csv")
    shutil.copyfile(path, saved)
    return saved, (earlier[-1] if earlier else None)


def cmd_week(args) -> None:
    """The whole weekly run for one projections file."""
    import contextlib
    import io

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, text):
            for stream in self.streams:
                stream.write(text)
                stream.flush()
            return len(text)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run_log.txt")
    summary: list[str] = []
    note = lambda text="": (print(text), summary.append(text))
    with open(log_path, "w") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        step = lambda title: print(f"\n{'=' * 8} {title} {'=' * 8}\n")
        projections = args.projections
        if args.entries:
            kickoffs = read_dk_entries(args.entries)[1]["kickoff"].astype(str).str.split().str[0]
            slate_date = pd.to_datetime(kickoffs.mode().iloc[0]).strftime("%Y-%m-%d")
        else:
            slate_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        previous = args.previous
        if args.archive:
            saved, before = archive_projections(projections, args.archive, slate_date)
            previous = previous or before
            note(f"Projections kept as {saved}" + (f"; previous version {previous}" if previous else "; first version for this slate"))
        header = pd.read_csv(projections, nrows=0).columns
        ownership = args.ownership if any(c.strip().lower() == args.ownership.strip().lower() for c in header) else None
        if previous:
            step("What changed since the previous file")
            moved = diff_projections(load_projections(previous), load_projections(projections), args.min_change)
            print(moved.round(2).to_string(index=False) if len(moved) else f"No projection moved by {args.min_change} points or more.")
            note(f"{len(moved)} players' projections moved by {args.min_change:g}+ points since the previous file.")

        engines = ["classic", "pbp"] if args.engine == "both" else [args.engine]
        if "pbp" in engines and not os.path.isdir(args.pbp_data):
            engines.remove("pbp")
            note(f"Skipping the play-by-play simulator: no data in {args.pbp_data} (run scripts/fetch_pbp.py).")
        common = (["--config", args.config] if args.config else []) + (["--seed", str(args.seed)] if args.seed is not None else [])
        names = {"classic": "default simulator", "pbp": "play-by-play simulator"}
        for engine in engines:
            folder = os.path.join(args.out_dir, engine)
            sim = os.path.join(folder, "simulation.npz")
            step(f"Simulating the slate ({names[engine]})")
            main(["simulate", projections, "--engine", engine, "--pbp-data", args.pbp_data, "--sims", str(args.sims), "--out", folder]
                 + (["--dk", args.entries] if args.entries else []) + (["--ownership", ownership] if ownership else []) + common)
            step(f"Building the portfolio ({names[engine]})")
            build = ["lineups", sim, "--contest", args.contest, "--show", "0", "--out", os.path.join(folder, "lineups.csv"),
                     "--portfolio-out", os.path.join(folder, "portfolio.csv"), "--upload-out", os.path.join(folder, "dk_upload.csv"),
                     "--max-exposure", str(args.max_exposure), "--max-overlap", str(args.max_overlap), "--portfolio", str(args.portfolio)]
            for flag in ("sampled", "candidates", "optimizer_lineups", "finalists", "field", "bankroll", "contest_id"):
                if getattr(args, flag) is not None:
                    build += ["--" + flag.replace("_", "-"), str(getattr(args, flag))]
            for item in args.exposure or []:
                build += ["--exposure", item]
            main(build + (["--entries", args.entries] if args.entries else []) + common)
            picked = pd.read_csv(os.path.join(folder, "portfolio.csv"))
            slots = picked[["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]].to_numpy().ravel()
            top = pd.Series(slots).value_counts().head(6)
            note(f"\n{names[engine].capitalize()}: {len(picked)} lineups, expected ROI {picked['adj_roi_pct'].mean():+.0f}% "
                 f"(top 1% {picked['top1_pct'].mean():.2f}% of the time, cash {picked['cash_pct'].mean():.0f}%).")
            note("  Most used: " + ", ".join(f"{name.split(' (')[0]} {100 * count / len(picked):.0f}%" for name, count in top.items()))
            note(f"  Files: {folder}/portfolio.csv" + (f", {folder}/dk_upload.csv" if args.entries else ""))
            for number, mine in enumerate(args.my_lineups or [], start=1):
                step(f"Grading {mine} ({names[engine]})")
                graded = os.path.join(folder, f"graded_my_lineups_{number}.csv")
                grade = ["grade", mine, sim, "--contest", args.contest, "--show", "0", "--out", graded]
                main(grade + (["--field", str(args.field)] if args.field is not None else []) + common)
                table = pd.read_csv(graded)
                note(f"  Your lineups in {os.path.basename(mine)}: {len(table)} graded, expected ROI {table['adj_roi_pct'].mean():+.0f}% "
                     f"(top 1% {table['top1_pct'].mean():.2f}%, cash {table['cash_pct'].mean():.0f}%); {int((table['adj_roi_pct'] > 0).sum())} "
                     f"positive. Detail: {graded}")
        step("Summary")
        print("\n".join(summary))
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print(f"\nSummary saved to {os.path.join(args.out_dir, 'summary.txt')}; full output in {log_path}")


def cmd_diff(args) -> None:
    moved = diff_projections(load_projections(args.old), load_projections(args.new), args.min_change)
    if moved.empty:
        print(f"No projection moved by {args.min_change} points or more.")
        return
    print(moved.round(2).to_string(index=False))


def cmd_sample(args) -> None:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    slate = sample_slate(args.games, args.seed)
    slate.to_csv(args.out, index=False)
    print(f"Wrote {len(slate)} made-up players to {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="ffsim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="simulate a slate from a projections file")
    p.add_argument("projections")
    p.add_argument("--ownership", help="name of the ownership column to use, if the file has several")
    p.add_argument("--engine", choices=["classic", "pbp"], default="classic",
                   help="classic: projection-based; pbp: games played snap by snap (needs play-by-play data)")
    p.add_argument("--pbp-data", default="data/raw/pbp", help="folder of play-by-play files, for --engine pbp")
    p.add_argument("--dk", help="DraftKings entries export; supplies salaries, games and player IDs")
    p.add_argument("--sims", type=int, default=50000)
    p.add_argument("--out", default="out")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("lineups", help="rank lineups by expected ROI in a contest")
    p.add_argument("simulation", help="simulation.npz written by 'ffsim simulate'")
    p.add_argument("--contest", default="contests/flea_flicker.json")
    p.add_argument("--candidates", type=int, default=5000, help="sampled lineups to keep for full grading")
    p.add_argument("--sampled", type=int, default=200000, help="lineups to sample and screen")
    p.add_argument("--optimizer-lineups", type=int, default=600, help="lineups to build with the optimizer")
    p.add_argument("--finalists", type=int, default=2000, help="top lineups the portfolio is chosen from")
    p.add_argument("--field", type=int, help="opponent lineups to sample (default: the whole contest)")
    p.add_argument("--portfolio", type=int, default=150, help="lineups to select")
    p.add_argument("--bankroll", type=float,
                   help="bankroll in dollars the portfolio is sized for (default 20x the entry cost; 0 ignores risk)")
    p.add_argument("--max-exposure", type=float, default=0.5, help="largest share of the portfolio one player may take")
    p.add_argument("--exposure", action="append", metavar="NAME=SHARE",
                   help="a different exposure limit for one player, e.g. \"Josh Allen=1.0\"; repeatable")
    p.add_argument("--max-overlap", type=int, default=7, help="most players two portfolio lineups may share")
    p.add_argument("--show", type=int, default=10)
    p.add_argument("--out", default="out/lineups.csv")
    p.add_argument("--portfolio-out", default="out/portfolio.csv")
    p.add_argument("--entries", help="DraftKings entries export; writes an upload file for this contest")
    p.add_argument("--contest-id", help="DraftKings contest ID to fill (default: match by contest name)")
    p.add_argument("--upload-out", default="out/dk_upload.csv")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_lineups)

    p = sub.add_parser("grade", help="expected ROI for lineups you supply")
    p.add_argument("lineups", help="CSV of lineups: a DraftKings upload/entries file, or nine players per row")
    p.add_argument("simulation", help="simulation.npz written by 'ffsim simulate'")
    p.add_argument("--contest", default="contests/flea_flicker.json")
    p.add_argument("--field", type=int, help="opponent lineups to sample (default: the whole contest)")
    p.add_argument("--show", type=int, default=5)
    p.add_argument("--out", default="out/graded_lineups.csv")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("live", help="in-game: rank your lineups against the real field and suggest late swaps")
    p.add_argument("standings", help="contest standings export downloaded during the games")
    p.add_argument("--entries", required=True, help="your DraftKings entries export for this slate")
    p.add_argument("--projections", required=True, help="latest projections file")
    p.add_argument("--contest", default="contests/flea_flicker.json")
    p.add_argument("--user", help="pick your lineups by DraftKings username instead of by entry ID")
    p.add_argument("--my-lineups", help="the lineup file you uploaded; needed while your late-game players are hidden")
    p.add_argument("--sims", type=int, default=10000)
    p.add_argument("--swap-mode", choices=["full", "single"], default="full",
                   help="full: rebuild each lineup's not-yet-started players together; single: one-for-one swaps only")
    p.add_argument("--max-swaps", type=int, default=2, help="single mode: most swaps per lineup (0 for no swap suggestions at all)")
    p.add_argument("--min-projection", type=float, default=4.0, help="lowest projection a swap-in may have")
    p.add_argument("--show", type=int, default=10)
    p.add_argument("--out-dir", default="out/live")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("pbp-check", help="compare play-by-play simulated games with real ones")
    p.add_argument("--data", default="data/raw/pbp", help="folder of play-by-play files (scripts/fetch_pbp.py)")
    p.add_argument("--seasons", type=int, nargs="*", help="seasons to use (default: all downloaded)")
    p.add_argument("--sims", type=int, default=20000)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_pbp_check)

    p = sub.add_parser("dk-slate", help="list the slate and your entries from a DraftKings entries export")
    p.add_argument("entries")
    p.add_argument("--out", default="data/slates/dk_slate.csv")
    p.set_defaults(func=cmd_slate)

    p = sub.add_parser("field", help="describe the simulated opponent lineups")
    p.add_argument("simulation")
    p.add_argument("--field", type=int, default=20000)
    p.add_argument("--show", type=int, default=10)
    p.add_argument("--out", help="also save the lineups to this CSV")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_field)

    p = sub.add_parser("contest-report", help="measure how the real field built, from a contest standings export")
    p.add_argument("standings", help="DraftKings contest standings CSV (or the zip it came in)")
    p.add_argument("--slate", help="CSV with name, team, opponent and salary for that slate; enables stacking and salary")
    p.add_argument("--show", type=int, default=15)
    p.add_argument("--out", help="save the report table to this CSV")
    p.set_defaults(func=cmd_contest_report)

    p = sub.add_parser("diff", help="show whose projection changed between two files")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--min-change", type=float, default=0.5)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("week", help="the whole weekly run for one projections file: keep it, show what changed, "
                                    "simulate, build the portfolio and upload file, grade your own lineups")
    p.add_argument("projections")
    p.add_argument("--entries", help="DraftKings entries export for the slate (salaries, player IDs, upload file)")
    p.add_argument("--my-lineups", action="append", help="a lineup file of yours to grade; repeatable")
    p.add_argument("--engine", choices=["both", "classic", "pbp"], default="both")
    p.add_argument("--ownership", default="Large Field", help="ownership column to use when the file has it")
    p.add_argument("--exposure", action="append", metavar="NAME=SHARE", help="exposure limit for one player; repeatable")
    p.add_argument("--max-exposure", type=float, default=0.5)
    p.add_argument("--max-overlap", type=int, default=7)
    p.add_argument("--portfolio", type=int, default=150)
    p.add_argument("--bankroll", type=float)
    p.add_argument("--contest", default="contests/flea_flicker.json")
    p.add_argument("--contest-id")
    p.add_argument("--sims", type=int, default=50000)
    p.add_argument("--archive", default="data/raw/projections",
                   help="folder where a dated copy of each projections file is kept ('' to skip)")
    p.add_argument("--previous", help="projections file to compare with (default: the last one kept for this slate)")
    p.add_argument("--min-change", type=float, default=0.5)
    p.add_argument("--pbp-data", default="data/raw/pbp")
    p.add_argument("--out-dir", default="out/week")
    for flag in ("sampled", "candidates", "optimizer-lineups", "finalists", "field"):
        p.add_argument("--" + flag, type=int, help="as for 'ffsim lineups'")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_week)

    p = sub.add_parser("sample-slate", help="write a made-up slate for trying the tools")
    p.add_argument("out")
    p.add_argument("--games", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_sample)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
