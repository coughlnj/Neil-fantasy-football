"""Attach players to simulated games: who throws, who is targeted, who carries, and
what that is worth in DraftKings points.

Each team's split of targets and carries starts from what its players have done this
season (recent games counting most) and is then adjusted until the simulated averages
line up with the projections supplied for the slate, so injuries and role changes the
projections know about carry through.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import models as M
from .engine import simulate_game

AIR_EDGES = [2, 12]              # targets: behind or near the line, intermediate, deep
N_AIR = 3
N_ZONE = 2                       # carries: outside the 5, inside the 5
PLAYER_STATS = ["pass_yards", "pass_td", "interceptions", "rush_yards", "rush_td", "receptions", "rec_yards", "rec_td",
                "fumbles_lost"]
TEAM_ALIASES = {"LA": "LAR", "WSH": "WAS", "JAC": "JAX"}
# Starting weights for a player with no history, per game: targets by depth, carries by zone.
DEFAULT_USAGE = {"WR": ([0.3, 1.5, 1.0], [0.05, 0.0]), "TE": ([0.4, 1.2, 0.4], [0.0, 0.0]),
                 "RB": ([1.2, 0.4, 0.05], [5.0, 0.6]), "QB": ([0.0, 0.0, 0.0], [1.5, 0.2])}


SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def player_key(name: str) -> str:
    """First initial and surname, so 'Jaxon Smith-Njigba' and 'J.Smith-Njigba' agree,
    as do 'Kenneth Walker III' and 'K.Walker', and 'C.J. Stroud' and 'C.Stroud'."""
    name = str(name).lower().replace(".", ". ").replace("-", "")
    parts = [re.sub(r"[^a-z]", "", part) for part in name.split()]
    parts = [part for part in parts if part and part not in SUFFIXES]
    if len(parts) < 2:
        return "".join(parts)
    rest = [part for part in parts[1:] if len(part) > 1] or parts[1:]      # drop middle initials
    return parts[0][0] + "." + "".join(rest)


def usage_history(plays: pd.DataFrame, season: int | None = None, half_life: float = 4.0) -> pd.DataFrame:
    """Per-game target and carry weights for every player, by team, from one season.
    A game `half_life` games back counts half as much as the latest one.

    Columns: team, key, name, t0..t2 (targets by depth), r0, r1 (carries by zone), dropbacks."""
    season = int(plays["season"].max()) if season is None else season
    p = plays[(plays["season"] == season) & plays["kind"].isin(["dropback", "rush"]) & plays["down"].between(1, 4)].copy()
    p["team"] = p["posteam"].replace(TEAM_ALIASES)
    order = p.groupby("team")["game_id"].transform(lambda g: pd.factorize(g)[0])
    games_played = order.groupby(p["team"]).transform("max") + 1
    p["w"] = 0.5 ** ((games_played - 1 - order) / half_life)
    total = p.drop_duplicates(["team", "game_id"]).groupby("team")["w"].sum()

    target = (p["pass_attempt"] == 1) & (p["sack"] != 1) & (p["qb_scramble"] != 1) & p["receiver_player_name"].notna()
    carry = (p["kind"] == "rush") & p["rusher_player_name"].notna()
    air = np.searchsorted(AIR_EDGES, p["air_yards"].fillna(0).to_numpy(), side="left")
    zone = (p["yardline_100"] <= 5).to_numpy(int)
    rows = []
    for mask, who, bucket, label in ((target, "receiver_player_name", air, "t"), (carry, "rusher_player_name", zone, "r")):
        part = p[mask].assign(bucket=bucket[mask.to_numpy()], name=p.loc[mask, who])
        rows.append(part.groupby(["team", "name", "bucket"])["w"].sum().unstack(fill_value=0.0).add_prefix(label))
    passer = p[(p["kind"] == "dropback") & p["passer_player_name"].notna()]
    rows.append(passer.groupby(["team", passer["passer_player_name"].rename("name")])["w"].sum().rename("dropbacks").to_frame())
    usage = pd.concat(rows, axis=1).fillna(0.0)
    for col in [f"t{i}" for i in range(N_AIR)] + [f"r{i}" for i in range(N_ZONE)] + ["dropbacks"]:
        if col not in usage:
            usage[col] = 0.0
    usage = usage.div(total, level="team", axis=0).reset_index()
    usage["key"] = usage["name"].map(player_key)
    return usage


def draftkings_points(stats: dict) -> np.ndarray:
    """DraftKings classic scoring for skill players, from per-player stat arrays."""
    s = stats
    return (0.04 * s["pass_yards"] + 4 * s["pass_td"] - s["interceptions"] + 3 * (s["pass_yards"] >= 300)
            + 0.1 * s["rush_yards"] + 6 * s["rush_td"] + 3 * (s["rush_yards"] >= 100)
            + s["receptions"] + 0.1 * s["rec_yards"] + 6 * s["rec_td"] + 3 * (s["rec_yards"] >= 100)
            - s["fumbles_lost"])


def defence_points(box: dict, team: int) -> np.ndarray:
    """DraftKings scoring for a team's defence and special teams, from the box score."""
    other = 1 - team
    allowed = box["points"][other] - 7.0 * box["defence_td"][other]     # points off our offence's turnovers do not count
    tiers = np.select([allowed < 1, allowed < 7, allowed < 14, allowed < 21, allowed < 28, allowed < 35],
                      [10.0, 7.0, 4.0, 1.0, 0.0, -1.0], -4.0)
    return (box["sacks"][other] + 2 * box["interceptions"][other] + 2 * box["fumbles_lost"][other]
            + 6 * (box["defence_td"][team] + box["return_td"][team]) + 2 * box["safeties"][team] + tiers)


class Matchup:
    """One game on the slate: the two teams' players and how the ball is shared."""

    def __init__(self, players: pd.DataFrame, usage: pd.DataFrame, min_projection: float = 0.5):
        teams = list(pd.unique(players["team"]))
        if len(teams) != 2:
            raise ValueError(f"a matchup needs two teams, got {teams}")
        self.teams = teams
        self.players = players
        self.side = players["team"].map({teams[0]: 0, teams[1]: 1}).to_numpy()
        pos = players["position"].to_numpy()
        mean = players["mean"].to_numpy(float)
        self.is_dst = pos == "DST"
        self.target_mean = mean
        hist = usage.set_index(["team", "key"])
        hist = hist[~hist.index.duplicated()]
        n = len(players)
        self.target_w = np.zeros((n, N_AIR))
        self.rush_w = np.zeros((n, N_ZONE))
        self.passer = [-1, -1]
        self.has_history = np.zeros(n, dtype=bool)
        for i, (name, team) in enumerate(zip(players["name"], players["team"])):
            if self.is_dst[i] or mean[i] < min_projection:
                continue
            key = (team, player_key(name))
            if key in hist.index and pos[i] != "QB":
                row = hist.loc[key]
                self.target_w[i] = [row[f"t{b}"] for b in range(N_AIR)]
                self.rush_w[i] = [row[f"r{z}"] for z in range(N_ZONE)]
                self.has_history[i] = self.target_w[i].sum() + self.rush_w[i].sum() > 0.5
            if not self.has_history[i]:
                tgt, run = DEFAULT_USAGE[pos[i]]
                scale = 0.25 if pos[i] != "QB" else 1.0
                self.target_w[i], self.rush_w[i] = np.array(tgt) * scale, np.array(run) * scale
            if pos[i] == "QB" and key in hist.index:
                row = hist.loc[key]
                self.rush_w[i] = [row[f"r{z}"] for z in range(N_ZONE)]
                self.has_history[i] = True
        for side in (0, 1):
            qbs = np.flatnonzero((self.side == side) & (pos == "QB"))
            if len(qbs) == 0:
                raise ValueError(f"{teams[side]} has no quarterback in the projections")
            self.passer[side] = int(qbs[np.argmax(mean[qbs])])
            backups = qbs[qbs != self.passer[side]]
            self.rush_w[backups] = 0.0
        self.factor = np.ones(n)                 # each player's share multiplier, adjusted by fit()
        self.quality = np.zeros(2)
        self.pass_shift = np.zeros(2)

    def _shares(self):
        """Cumulative share tables: who is targeted (by side and depth), who carries (by side and zone)."""
        n = len(self.players)
        tgt = np.zeros((2, N_AIR, n))
        run = np.zeros((2, N_ZONE, n))
        for side in (0, 1):
            on = self.side == side
            w = self.target_w * self.factor[:, None] * on[:, None]
            r = self.rush_w * self.factor[:, None] * on[:, None]
            for b in range(N_AIR):
                col = w[:, b] if w[:, b].sum() > 0 else w.sum(axis=1)
                tgt[side, b] = np.cumsum(col / col.sum())
            for z in range(N_ZONE):
                col = r[:, z] if r[:, z].sum() > 0 else r.sum(axis=1)
                run[side, z] = np.cumsum(col / col.sum())
        return tgt, run

    def simulate(self, model: M.GameModel, n_sims: int, rng: np.random.Generator):
        """Play the game n_sims times. Returns (DraftKings points per player, box score, player stats)."""
        n = len(self.players)
        col = model.plays.columns
        tgt, run = self._shares()
        passer = np.array(self.passer)
        stats = {name: np.zeros((n, n_sims), dtype=np.float32) for name in PLAYER_STATS}

        def pick(table, u):
            return (u[:, None] > table).sum(axis=1).clip(max=n - 1)

        def on_play(games, offence, row, gain, scored, is_run, yardline):
            qb = passer[offence]
            complete, target, scramble = col["complete"][row], col["target"][row], col["scramble"][row]
            lost = col["turnover"][row]
            u = rng.random(len(games))
            carrier = np.full(len(games), -1)
            if target.any():
                t = np.flatnonzero(target)
                depth = np.searchsorted(AIR_EDGES, col["air_yards"][row[t]], side="left")
                receiver = pick(tgt[offence[t], depth], u[t])
                done = complete[t]
                g, r, q = games[t][done], receiver[done], qb[t][done]
                stats["receptions"][r, g] += 1
                stats["rec_yards"][r, g] += gain[t][done]
                stats["rec_td"][r, g] += scored[t][done]
                stats["pass_yards"][q, g] += gain[t][done]
                stats["pass_td"][q, g] += scored[t][done]
                carrier[t[done]] = r
            picked = lost == 1
            stats["interceptions"][qb[picked], games[picked]] += 1
            runs = (is_run == 1) | scramble
            if runs.any():
                r = np.flatnonzero(runs)
                zone = (yardline[r] <= 5).astype(int)
                back = np.where(scramble[r], qb[r], pick(run[offence[r], zone], u[r]))
                stats["rush_yards"][back, games[r]] += gain[r]
                stats["rush_td"][back, games[r]] += scored[r]
                carrier[r] = back
            sacked = col["sack"][row]
            carrier[sacked] = qb[sacked]
            fumbled = (lost == 2) & (carrier >= 0)
            stats["fumbles_lost"][carrier[fumbled], games[fumbled]] += 1

        box = simulate_game(model, n_sims, rng, tuple(self.quality), tuple(self.pass_shift), on_play)
        points = draftkings_points(stats)
        for side in (0, 1):
            for i in np.flatnonzero(self.is_dst & (self.side == side)):
                points[i] = defence_points(box, side)
        return points, box, stats

    def fit(self, model: M.GameModel, rounds: int = 10, n_sims: int = 3000, seed: int = 0) -> None:
        """Adjust each player's share of the ball, and each offence's quality, until the
        simulated averages are close to the projections."""
        rng = np.random.default_rng(seed)
        skill = ~self.is_dst & (self.target_w.sum(axis=1) + self.rush_w.sum(axis=1) > 0)
        qb = np.zeros(len(self.players), dtype=bool)
        qb[self.passer] = True
        for _ in range(rounds):
            points, _, _ = self.simulate(model, n_sims, rng)
            sim = points.mean(axis=1)
            ratio = np.clip(self.target_mean / np.maximum(sim, 0.05), 0.2, 5.0)
            self.factor[skill & ~qb] *= ratio[skill & ~qb] ** 0.7
            self.factor[qb] *= ratio[qb] ** 0.3                 # a QB's share is only his designed runs
            self.factor = np.clip(self.factor, 1e-3, 1e3)
            for side in (0, 1):
                on = (self.side == side) & skill
                gap = np.log(self.target_mean[on].sum() / max(sim[on].sum(), 1.0))
                self.quality[side] = float(np.clip(self.quality[side] + 1.6 * gap, -0.95, 0.95))
