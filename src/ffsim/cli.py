"""Simulate a DraftKings slate and rank tournament lineups by expected ROI."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .candidates import search_candidates
from .contest import load_contest
from .draftkings import read_dk_entries, write_upload
from .field import field_report, sample_field, target_ownership
from .projections import diff_projections, load_projections
from .sample import sample_slate
from .roi import (adjust_roi, describe_lineups, evaluate, pick_portfolio, pick_portfolio_for_growth,
                  portfolio_summary)
from .settings import load_settings
from .standings import read_standings, standings_report
from .simulate import fit_report, load_simulation, save_simulation, simulate_slate


def cmd_simulate(args) -> None:
    settings = load_settings(args.config)
    dk_players = read_dk_entries(args.dk)[1] if args.dk else None
    players = load_projections(args.projections, dk_players, log=print, ownership_column=args.ownership)
    scores, fit = simulate_slate(players, settings, args.sims, args.seed)
    os.makedirs(args.out, exist_ok=True)
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
    field_size = min(args.field, contest.entries - 1)
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
    print(f"Grading {len(candidates):,} lineups over {grade.shape[1]:,} simulated slates...")
    results = evaluate(candidates, field, grade, contest)
    table = describe_lineups(candidates, players, realised)
    slots = table.attrs["slots"]
    adjusted = adjust_roi(results, table[["projection", "total_ownership_pct"]])
    table["adj_roi_pct"] = adjusted.round(2)
    table = pd.concat([table, results.round(3)], axis=1)

    # Choose the portfolio from the strongest lineups, judged together.
    finalists = np.argsort(-adjusted)[: max(args.finalists, args.portfolio)]
    _, payouts = evaluate(candidates[finalists], field, grade, contest, keep_payouts=True)
    limits = (args.portfolio, args.max_exposure, args.max_overlap)
    simple = pick_portfolio(candidates[finalists], adjusted[finalists], *limits)
    bankroll = args.bankroll if args.bankroll is not None else 20 * args.portfolio * contest.entry_fee
    if bankroll > 0:
        chosen = pick_portfolio_for_growth(
            candidates[finalists], payouts, adjusted[finalists], contest.entry_fee, args.portfolio, bankroll,
            args.max_exposure, args.max_overlap,
        )
    else:
        chosen = simple
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
    overall, groups, notes = standings_report(entries, players, slate)
    users = entries["user"].nunique()
    print(f"{len(entries):,} entries from {users:,} entrants.\n")
    table = pd.concat([overall, groups], axis=1)
    with pd.option_context("display.width", 200):
        print(table.to_string())
    for note in notes:
        print("\n" + note)
    by_pos = players.groupby("position")["ownership"].sum().round(1)
    print("\nActual ownership by position (percent): " + ", ".join(f"{p} {v:g}" for p, v in by_pos.items()))
    print("\nMost-owned players:\n")
    print(players.sort_values("ownership", ascending=False).head(args.show).to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        table.to_csv(args.out)


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
    p.add_argument("--field", type=int, default=20000, help="opponent lineups to sample")
    p.add_argument("--portfolio", type=int, default=150, help="lineups to select")
    p.add_argument("--bankroll", type=float,
                   help="bankroll in dollars the portfolio is sized for (default 20x the entry cost; 0 ignores risk)")
    p.add_argument("--max-exposure", type=float, default=0.5, help="largest share of the portfolio one player may take")
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

    p = sub.add_parser("sample-slate", help="write a made-up slate for trying the tools")
    p.add_argument("out")
    p.add_argument("--games", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_sample)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
