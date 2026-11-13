"""Models fitted from play-by-play data: what gets called in each situation, and what
happens when it is.

Play calling and kicking are logistic regressions on the game situation. Play results
are not modelled by formula: a simulated play draws the result of a real play that was
run in a similar situation, so yardage, sacks, turnovers and big plays come in their
real proportions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Logistic regression by Newton's method with a small ridge penalty."""
    w = np.zeros(x.shape[1])
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(40):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        step = np.linalg.solve((x * (p * (1 - p))[:, None]).T @ x + penalty, x.T @ (y - p) - penalty @ w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))


def situation(down, ydstogo, yardline, score_diff, game_sec, half_sec) -> dict:
    """Shared ingredients for the situational models, from arrays describing each play."""
    down, ydstogo, yardline = (np.asarray(v, dtype=float) for v in (down, ydstogo, yardline))
    sd = np.clip(np.asarray(score_diff, dtype=float), -24, 24) / 10.0
    game_sec, half_sec = np.asarray(game_sec, dtype=float), np.asarray(half_sec, dtype=float)
    late = (1.0 - np.clip(game_sec, 0, 3600) / 3600.0) ** 3
    q4 = game_sec <= 900
    return {
        "down": down, "log_togo": np.log(np.clip(ydstogo, 1, 30)), "togo": np.clip(ydstogo, 1, 30),
        "yl": yardline / 100.0, "sd": sd, "late": late, "q4": q4.astype(float),
        "last5": (game_sec <= 300).astype(float), "two_min": (half_sec <= 120).astype(float),
        "trail": (sd < 0).astype(float), "lead": (sd > 0).astype(float),
    }


def pass_features(s: dict) -> np.ndarray:
    d2, d3, d4 = (s["down"] == 2).astype(float), (s["down"] == 3).astype(float), (s["down"] == 4).astype(float)
    return np.column_stack([
        np.ones_like(s["yl"]), d2, d3, d4, s["log_togo"], d2 * s["log_togo"], (d3 + d4) * s["log_togo"],
        (s["togo"] <= 1) * (d3 + d4), s["yl"], (s["yl"] <= 0.10), (s["yl"] <= 0.03), (s["yl"] >= 0.90),
        s["sd"], s["sd"] * s["late"], s["two_min"], s["two_min"] * s["sd"],
        s["q4"] * s["trail"], s["last5"] * s["trail"], s["q4"] * s["lead"], s["last5"] * s["lead"],
        s["q4"] * s["sd"],
    ])


def go_features(s: dict) -> np.ndarray:
    return np.column_stack([
        np.ones_like(s["yl"]), s["log_togo"], (s["togo"] <= 1), (s["togo"] <= 2), s["yl"], s["yl"] ** 2,
        (s["yl"] <= 0.37), (s["yl"] <= 0.37) * s["log_togo"], (s["yl"] >= 0.60), s["sd"], s["sd"] * s["late"],
        s["q4"] * s["trail"], s["last5"] * s["trail"], s["last5"] * s["trail"] * (s["sd"] < -0.8),
        s["q4"] * s["lead"], s["two_min"],
    ])


def kick_features(s: dict) -> np.ndarray:
    return np.column_stack([
        np.ones_like(s["yl"]), s["yl"], s["yl"] ** 2, (s["yl"] <= 0.30), (s["yl"] <= 0.37), (s["yl"] <= 0.42),
        s["log_togo"], s["sd"] * s["late"], s["two_min"],
    ])


DOWN_GROUP = np.array([0, 0, 1, 2, 2])          # by down 0..4


def bucket_index(kind, down, ydstogo, yardline) -> np.ndarray:
    """Which pool of real plays a situation draws from: play kind (0 dropback, 1 rush),
    down (1st, 2nd, 3rd/4th), distance and field position."""
    dist = np.searchsorted([2, 6, 10], np.asarray(ydstogo), side="left")            # 1-2, 3-6, 7-10, 11+
    zone = np.searchsorted([5, 10, 20, 50], np.asarray(yardline), side="left")      # 1-5, 6-10, 11-20, 21-50, 51+
    return ((np.asarray(kind) * 3 + DOWN_GROUP[np.asarray(down).astype(int)]) * 4 + dist) * 5 + zone


N_BUCKETS = 2 * 3 * 4 * 5


@dataclass
class Pool:
    """Real outcomes grouped by bucket: draw(bucket, rng) returns row numbers into the columns."""
    start: np.ndarray
    size: np.ndarray
    columns: dict = field(default_factory=dict)

    def draw(self, bucket: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.start[bucket] + (u * self.size[bucket]).astype(np.int64)


def build_pool(frame: pd.DataFrame, bucket: np.ndarray, n_buckets: int, columns: dict, min_size: int = 30,
               nearest=None) -> Pool:
    """Group outcome columns by bucket. A bucket with too few real plays borrows the
    plays of its nearest well-stocked neighbour (`nearest(a, b)` gives the distance)."""
    order = np.argsort(bucket, kind="stable")
    counts = np.bincount(bucket, minlength=n_buckets)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    thin = np.flatnonzero(counts < min_size)
    if len(thin):
        if nearest is None:
            raise ValueError(f"buckets {thin.tolist()} have fewer than {min_size} plays")
        full = np.flatnonzero(counts >= min_size)
        start, counts = start.copy(), counts.copy()
        for b in thin:
            donor = full[np.argmin([nearest(b, other) for other in full])]
            start[b], counts[b] = start[donor], counts[donor]
    return Pool(start, counts, {name: np.asarray(values)[order] for name, values in columns.items()})


def play_bucket_distance(a: int, b: int) -> float:
    """How different two play buckets are; a different play kind is never borrowed."""
    part = lambda v: (v // 60, (v // 20) % 3, (v // 5) % 4, v % 5)       # kind, down group, distance, zone
    (ka, da, la, za), (kb, db, lb, zb) = part(a), part(b)
    return 1000 * (ka != kb) + 3 * abs(za - zb) + 2 * abs(la - lb) + 4 * abs(da - db)


@dataclass
class GameModel:
    pass_w: np.ndarray
    go_w: np.ndarray
    kick_w: np.ndarray
    fg_w: np.ndarray
    plays: Pool
    clock: Pool
    punts: Pool
    penalties: Pool
    kickoff_start: np.ndarray
    kickoff_td: float
    penalty_rate: float
    xp_rate: float
    two_point_try: float
    two_point_rate: float
    summary: dict


CLOCK_MODES = 3      # 0 normal, 1 hurrying, 2 running the clock down


def clock_mode(score_diff, game_sec, half_sec) -> np.ndarray:
    score_diff, game_sec, half_sec = (np.asarray(v, dtype=float) for v in (score_diff, game_sec, half_sec))
    hurry = (half_sec <= 120) & ((game_sec > 1800) | (score_diff <= 0)) | ((game_sec <= 300) & (score_diff < 0))
    milk = (game_sec <= 480) & (score_diff > 0)
    return np.where(milk, 2, np.where(hurry, 1, 0))


def fit_game_model(plays: pd.DataFrame) -> GameModel:
    p = plays
    scrimmage = p["kind"].isin(["dropback", "rush"]) & p["down"].between(1, 4) & p["yardline_100"].between(1, 99)
    sit = lambda f: situation(f["down"], f["ydstogo"], f["yardline_100"], f["score_differential"].fillna(0),
                              f["game_seconds_remaining"], f["half_seconds_remaining"])

    # Pass or run, on first to third down and on fourth downs that were gone for.
    calls = p[scrimmage]
    pass_w = fit_logistic(pass_features(sit(calls)), (calls["kind"] == "dropback").to_numpy(float))

    # Fourth down: go for it, kick a field goal, or punt.
    fourth = p[(p["down"] == 4) & p["kind"].isin(["dropback", "rush", "punt", "field_goal"]) & p["yardline_100"].between(1, 99)]
    went = fourth["kind"].isin(["dropback", "rush"]).to_numpy(float)
    go_w = fit_logistic(go_features(sit(fourth)), went)
    kicks = fourth[~fourth["kind"].isin(["dropback", "rush"]) & (fourth["yardline_100"] <= 47)]
    kick_w = fit_logistic(kick_features(sit(kicks)), (kicks["kind"] == "field_goal").to_numpy(float))

    tries = p[(p["kind"] == "field_goal") & p["field_goal_result"].notna() & p["kick_distance"].notna()]
    dist = tries["kick_distance"].to_numpy(float) / 10.0
    fg_w = fit_logistic(np.column_stack([np.ones_like(dist), dist, dist**2]), (tries["field_goal_result"] == "made").to_numpy(float))

    # Results of real dropbacks and runs, by situation.
    live = p[scrimmage].copy()
    kind = (live["kind"] == "rush").to_numpy(int)
    bucket = bucket_index(kind, live["down"].to_numpy(int), live["ydstogo"].to_numpy(), live["yardline_100"].to_numpy())
    gave_up = (live["interception"] == 1) | (live["fumble_lost"] == 1)
    offence_td = (live["touchdown"] == 1) & (live["td_team"] == live["posteam"])
    flipped = live["next_posteam"].eq(live["defteam"])
    after = np.where(gave_up & flipped & live["next_yardline"].notna(), live["next_yardline"], np.nan)
    target = (live["pass_attempt"] == 1) & (live["sack"] != 1) & (live["qb_scramble"] != 1)
    stops = (live["incomplete_pass"] == 1) | (live["out_of_bounds"] == 1) | gave_up | (live["touchdown"] == 1)
    outcome = build_pool(live, bucket, N_BUCKETS, {
        "gain": live["yards_gained"].fillna(0).to_numpy(np.int16),
        "td": offence_td.to_numpy(),
        "turnover": np.where(live["interception"] == 1, 1, np.where(live["fumble_lost"] == 1, 2, 0)).astype(np.int8),
        "defence_td": (gave_up & (live["touchdown"] == 1) & (live["td_team"] == live["defteam"])).to_numpy(),
        "after_turnover": after,
        "sack": (live["sack"] == 1).to_numpy(),
        "scramble": (live["qb_scramble"] == 1).to_numpy(),
        "target": target.to_numpy(),
        "complete": (live["complete_pass"] == 1).to_numpy(),
        "air_yards": live["air_yards"].fillna(0).to_numpy(np.int16),
        "stops_clock": stops.to_numpy(),
    }, nearest=play_bucket_distance)

    # Seconds run off per play: by whether the clock stopped and how the offence is managing it.
    timed = live[live["elapsed"].between(0, 60)]
    stopped = ((timed["incomplete_pass"] == 1) | (timed["out_of_bounds"] == 1) | (timed["interception"] == 1)
               | (timed["fumble_lost"] == 1) | (timed["touchdown"] == 1)).to_numpy(int)
    mode = clock_mode(timed["score_differential"].fillna(0), timed["game_seconds_remaining"], timed["half_seconds_remaining"])
    clock = build_pool(timed, stopped * CLOCK_MODES + mode, 2 * CLOCK_MODES, {"seconds": timed["elapsed"].to_numpy(np.int16)})

    # Punts: where the other team takes over.
    punted = p[(p["kind"] == "punt") & p["yardline_100"].between(1, 99) & p["next_posteam"].eq(p["defteam"]) & p["next_yardline"].notna()]
    returned = (punted["return_touchdown"] == 1) | ((punted["touchdown"] == 1) & (punted["td_team"] == punted["defteam"]))
    zone = np.searchsorted([50, 60, 70, 80], punted["yardline_100"].to_numpy(), side="left")
    punts = build_pool(punted, zone, 5, {"next_yardline": punted["next_yardline"].to_numpy(float), "return_td": returned.to_numpy()})

    # Penalties that wipe out the play: yards for the offence and whether they give a first down.
    flags = p[(p["kind"] == "penalty") & p["down"].between(1, 4) & p["penalty_yards"].notna()]
    signed = np.where(flags["penalty_team"] == flags["posteam"], -1, 1) * flags["penalty_yards"].to_numpy()
    penalties = build_pool(flags, np.zeros(len(flags), int), 1, {
        "yards": signed.astype(np.int16), "first_down": (flags["first_down_penalty"] == 1).to_numpy(),
        "seconds": flags["elapsed"].fillna(0).clip(0, 45).to_numpy(np.int16)})

    # In the data the receiving team is listed as having the ball on a kickoff.
    kickoffs = p[(p["kind"] == "kickoff") & p["next_posteam"].eq(p["posteam"]) & p["next_yardline"].notna()
                 & p["next_kind"].isin(["dropback", "rush", "penalty"]) & (p["season"] >= p["season"].max() - 1)]
    all_kickoffs = p[p["kind"] == "kickoff"]
    extra = p[(p["kind"] == "extra_point") & p["extra_point_result"].notna()]
    twos = p[(p["kind"] == "two_point") & p["two_point_conv_result"].notna()]
    drops, runs = live[live["kind"] == "dropback"], live[live["kind"] == "rush"]
    summary = {
        "games": int(p["game_id"].nunique()), "plays": int(len(live)),
        "pass_rate": float((live["kind"] == "dropback").mean()),
        "yards_per_dropback": float(drops["yards_gained"].mean()), "yards_per_rush": float(runs["yards_gained"].mean()),
    }
    return GameModel(
        pass_w=pass_w, go_w=go_w, kick_w=kick_w, fg_w=fg_w, plays=outcome, clock=clock, punts=punts, penalties=penalties,
        kickoff_start=kickoffs["next_yardline"].to_numpy(float),
        kickoff_td=float((all_kickoffs["return_touchdown"] == 1).mean()),
        penalty_rate=float(len(flags) / (len(flags) + len(live))),
        xp_rate=float((extra["extra_point_result"] == "good").mean()),
        two_point_try=float(len(twos) / (len(twos) + len(extra))),
        two_point_rate=float((twos["two_point_conv_result"] == "success").mean()),
        summary=summary,
    )
