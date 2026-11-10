"""Simulate a DraftKings slate and rank tournament lineups by expected ROI."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .candidates import build_candidates
from .contest import load_contest
from .field import field_report, sample_field, target_ownership
from .projections import diff_projections, load_projections
from .sample import sample_slate
from .roi import adjust_roi, describe_lineups, evaluate, pick_portfolio
from .settings import load_settings
from .simulate import fit_report, load_simulation, save_simulation, simulate_slate


def cmd_simulate(args) -> None:
    settings = load_settings(args.config)
    players = load_projections(args.projections)
    scores, fit = simulate_slate(players, settings, args.sims, args.seed)
    os.makedirs(args.out, exist_ok=True)
    save_simulation(os.path.join(args.out, "simulation.npz"), players, scores)
    report = fit_report(players, fit, scores, settings)
    report.to_csv(os.path.join(args.out, "simulation_report.csv"), index=False)

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
    held_out = max(1, min(args.candidates, n_sims // 5))
    grade, build = scores[:, : n_sims - held_out], scores[:, n_sims - held_out :]

    print(f"{contest.name}: ${contest.entry_fee:g} entry, {contest.entries:,} entries, "
          f"${contest.prize_pool:,.0f} prizes ({100 * contest.rake:.1f}% rake)")
    field_size = min(args.field, contest.entries - 1)
    field, realised = sample_field(players, settings, field_size, args.seed)
    target = target_ownership(players, settings)
    implied = float((target * players["salary"]).sum())
    if implied > settings["site"]["salary_cap"]:
        print(f"Note: your ownership implies an average lineup costing ${implied:,.0f}, over the cap, "
              "so the field cannot match it exactly.")
    print(f"Sampled {field_size:,} opponent lineups; ownership within "
          f"{100 * np.abs(realised - target).max():.1f} points of your projections.")

    candidates = build_candidates(players, build, settings, args.candidates, args.seed)
    print(f"Built {len(candidates):,} candidate lineups; grading over {grade.shape[1]:,} simulated slates...")
    results = evaluate(candidates, field, grade, contest)

    table = describe_lineups(candidates, players, realised)
    adjusted = adjust_roi(results, table[["projection", "total_ownership_pct"]])
    table["adj_roi_pct"] = adjusted.round(2)
    table = pd.concat([table, results.round(3)], axis=1)
    chosen = pick_portfolio(candidates, adjusted, args.portfolio, args.max_exposure, args.max_overlap)
    table["in_portfolio"] = False
    table.loc[chosen, "in_portfolio"] = True
    table = table.sort_values("adj_roi_pct", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False)
    picked = table[table["in_portfolio"]]
    print(f"\nPortfolio of {len(picked)} lineups: average adjusted ROI {picked['adj_roi_pct'].mean():+.1f}% "
          f"(all candidates average {table['roi_pct'].mean():+.1f}%)")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(picked.head(args.show).drop(columns=["in_portfolio"]).to_string(index=False))
    print(f"\nAll {len(table):,} lineups saved to {args.out}")


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
    p.add_argument("--sims", type=int, default=20000)
    p.add_argument("--out", default="out")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("lineups", help="rank lineups by expected ROI in a contest")
    p.add_argument("simulation", help="simulation.npz written by 'ffsim simulate'")
    p.add_argument("--contest", default="contests/typical_gpp.json")
    p.add_argument("--candidates", type=int, default=3000, help="lineups to build and grade")
    p.add_argument("--field", type=int, default=20000, help="opponent lineups to sample")
    p.add_argument("--portfolio", type=int, default=150, help="lineups to select")
    p.add_argument("--max-exposure", type=float, default=0.5, help="largest share of the portfolio one player may take")
    p.add_argument("--max-overlap", type=int, default=7, help="most players two portfolio lineups may share")
    p.add_argument("--show", type=int, default=20)
    p.add_argument("--out", default="out/lineups.csv")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_lineups)

    p = sub.add_parser("field", help="describe the simulated opponent lineups")
    p.add_argument("simulation")
    p.add_argument("--field", type=int, default=20000)
    p.add_argument("--show", type=int, default=10)
    p.add_argument("--out", help="also save the lineups to this CSV")
    p.add_argument("--config")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_field)

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
