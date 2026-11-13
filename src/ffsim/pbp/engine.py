"""Play games snap by snap.

Every simulated game carries its own state (who has the ball, down, distance, field
position, clock, score). Each step, every game still in progress runs one snap: the
situation decides the call, a real play from a similar situation supplies the result,
and the clock, score and box score are updated. All games advance together.
"""
from __future__ import annotations

import numpy as np

from . import models as M

STATS = ["points", "plays", "dropbacks", "pass_att", "completions", "pass_yards", "pass_td", "interceptions",
         "sacks", "rush_att", "rush_yards", "rush_td", "fumbles_lost", "fg_att", "fg_made", "punts",
         "return_td", "safeties", "penalties", "first_downs"]


def simulate_game(model: M.GameModel, n_sims: int, rng: np.random.Generator,
                  quality: tuple[float, float] = (0.0, 0.0), pass_shift: tuple[float, float] = (0.0, 0.0),
                  on_play=None) -> dict:
    """Simulate one matchup n_sims times. Team 0 and team 1 are the two sides.

    quality: each offence's edge over the defence it faces, from -1 to 1. A positive
      value sometimes draws two real plays and keeps the better one; negative, the worse.
    pass_shift: added to each offence's pass-or-run log-odds (its lean relative to the league).
    on_play: optional callback(games, offence, row, gain, scored, kind) called for every
      batch of scrimmage plays, for attaching player statistics.

    Returns the box score: stats[name] has shape (2, n_sims), plus "overtime"."""
    n = n_sims
    plays, clock = model.plays, model.clock
    col = plays.columns
    quality, pass_shift = np.asarray(quality, dtype=float), np.asarray(pass_shift, dtype=float)
    stats = {name: np.zeros((2, n), dtype=np.float32) for name in STATS}
    score = stats["points"]

    pos = rng.integers(0, 2, n)                    # team with the ball
    opening = pos.copy()
    yl = np.zeros(n)                               # yards from the opponent's goal line
    down = np.ones(n, dtype=np.int64)
    togo = np.full(n, 10.0)
    half = np.ones(n, dtype=np.int64)              # 1, 2, or 3 for overtime
    half_sec = np.full(n, 1800.0)
    active = np.ones(n, dtype=bool)
    ot_possessions = np.zeros(n, dtype=np.int64)

    def add(name, games, team, value=1.0):
        stats[name][team, games] += value

    def new_series(games):
        down[games] = 1
        togo[games] = np.minimum(10.0, yl[games])

    def extra_point(games, team):
        u = rng.random(len(games))
        two = u < model.two_point_try
        made_two = two & (rng.random(len(games)) < model.two_point_rate)
        made_one = ~two & (rng.random(len(games)) < model.xp_rate)
        score[team, games] += 2.0 * made_two + 1.0 * made_one

    def kickoff(games, receiver):
        """`receiver` takes the kickoff; once in a while it comes back for a touchdown."""
        if len(games) == 0:
            return
        back = rng.random(len(games)) < model.kickoff_td
        if back.any():
            g, t = games[back], receiver[back]
            score[t, g] += 6
            add("return_td", g, t)
            extra_point(g, t)
            receiver = np.where(back, 1 - receiver, receiver)
        pos[games] = receiver
        yl[games] = rng.choice(model.kickoff_start, len(games))
        new_series(games)

    def touchdown(games, team):
        if len(games) == 0:
            return
        score[team, games] += 6
        extra_point(games, team)
        kickoff(games, 1 - team)

    def change_of_possession(games, spot):
        pos[games] = 1 - pos[games]
        yl[games] = np.clip(spot, 1, 99)
        new_series(games)

    kickoff(np.arange(n), pos.copy())

    for _ in range(600):
        g = np.flatnonzero(active)
        if len(g) == 0:
            break
        off = pos[g]
        game_sec = np.where(half[g] == 1, half_sec[g] + 1800, np.where(half[g] == 2, half_sec[g], np.minimum(half_sec[g], 300)))
        diff = score[off, g] - score[1 - off, g]
        used = np.zeros(len(g))
        before = pos[g].copy()
        todo = np.ones(len(g), dtype=bool)

        # Kneel out a lead when the clock allows.
        kneel = todo & (half[g] >= 2) & (diff > 0) & (half_sec[g] <= (4 - down[g]) * 38)
        if kneel.any():
            k = g[kneel]
            yl[k] = np.minimum(yl[k] + 1, 99)
            down[k] += 1
            used[kneel] = 40
            todo &= ~kneel

        # A field goal try as a half runs out.
        last_kick = todo & (half_sec[g] <= 10) & (yl[g] <= 42) & ((half[g] == 1) | ((diff >= -3) & (diff <= 0)))

        # Penalties that wipe out the snap.
        flagged = todo & ~last_kick & (rng.random(len(g)) < model.penalty_rate)
        if flagged.any():
            k = g[flagged]
            row = model.penalties.draw(np.zeros(len(k), dtype=np.int64), rng.random(len(k)))
            spot = np.clip(yl[k] - model.penalties.columns["yards"][row], 1, 99)
            togo[k] -= yl[k] - spot
            yl[k] = spot
            fresh = model.penalties.columns["first_down"][row] | (togo[k] <= 0)
            new_series(k[fresh])
            togo[k] = np.clip(togo[k], 1, 40)
            add("penalties", k, off[flagged])
            used[flagged] = model.penalties.columns["seconds"][row]
            todo &= ~flagged

        # Fourth down: go, kick or punt.
        sit = M.situation(down[g], togo[g], yl[g], diff, game_sec, half_sec[g])
        fourth = todo & (down[g] == 4) & ~last_kick
        go = fourth & (rng.random(len(g)) < M.predict(M.go_features(sit), model.go_w))
        in_range = yl[g] <= 47
        kick = last_kick | (fourth & ~go & in_range & (rng.random(len(g)) < M.predict(M.kick_features(sit), model.kick_w)))
        punt = fourth & ~go & ~kick

        if kick.any():
            k, t = g[kick], off[kick]
            distance = (yl[k] + 18) / 10.0
            good = rng.random(len(k)) < M.predict(np.column_stack([np.ones_like(distance), distance, distance**2]), model.fg_w)
            add("fg_att", k, t)
            add("fg_made", k[good], t[good])
            score[t[good], k[good]] += 3
            kickoff(k[good], 1 - t[good])
            change_of_possession(k[~good], np.minimum(93 - yl[k[~good]], 80))
            used[kick] = 5
            todo &= ~kick

        if punt.any():
            k, t = g[punt], off[punt]
            row = model.punts.draw(np.searchsorted([50, 60, 70, 80], yl[k], side="left"), rng.random(len(k)))
            add("punts", k, t)
            back = model.punts.columns["return_td"][row]
            change_of_possession(k[~back], model.punts.columns["next_yardline"][row][~back])
            if back.any():
                add("return_td", k[back], 1 - t[back])
                touchdown(k[back], 1 - t[back])
            used[punt] = rng.integers(6, 13, len(k))
            todo &= ~punt

        # Everything else is a dropback or a run.
        if todo.any():
            k, t = g[todo], off[todo]
            logit = M.pass_features({key: value[todo] for key, value in sit.items()}) @ model.pass_w + pass_shift[t]
            is_run = (rng.random(len(k)) >= 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
            bucket = M.bucket_index(is_run, down[k], togo[k], yl[k])
            row = plays.draw(bucket, rng.random(len(k)))
            q = quality[t]
            if (q != 0).any():      # a second look, kept when it is better (or worse) for the offence
                other = plays.draw(bucket, rng.random(len(k)))
                value = lambda r: col["gain"][r] + 25.0 * col["td"][r] - 20.0 * (col["turnover"][r] > 0)
                swap = (rng.random(len(k)) < np.abs(q)) & (np.sign(q) * (value(other) - value(row)) > 0)
                row = np.where(swap, other, row)

            gain = col["gain"][row].astype(float)
            lost = col["turnover"][row]
            spot = yl[k] - gain
            scored = (col["td"][row] | (spot <= 0)) & (lost == 0)
            gain = np.where(scored, yl[k], gain)
            spot = np.where(scored, 0.0, spot)
            safety = (spot >= 100) & (lost == 0)

            add("plays", k, t)
            dropback = is_run == 0
            add("dropbacks", k[dropback], t[dropback])
            sack, scramble = col["sack"][row], col["scramble"][row]
            target, complete = col["target"][row], col["complete"][row]
            add("sacks", k[sack], t[sack])
            add("pass_att", k[target], t[target])
            add("completions", k[complete], t[complete])
            add("pass_yards", k[complete], t[complete], gain[complete])
            add("pass_td", k[complete & scored], t[complete & scored])
            ran = (is_run == 1) | scramble
            add("rush_att", k[ran], t[ran])
            add("rush_yards", k[ran], t[ran], gain[ran])
            add("rush_td", k[ran & scored], t[ran & scored])
            add("interceptions", k[lost == 1], t[lost == 1])
            add("fumbles_lost", k[lost == 2], t[lost == 2])
            if on_play is not None:
                on_play(k, t, row, gain, scored, is_run)

            mode = M.clock_mode(diff[todo], game_sec[todo], half_sec[k])
            stopped = (col["stops_clock"][row] | scored | (lost > 0)).astype(np.int64)
            used[todo] = clock.columns["seconds"][clock.draw(stopped * M.CLOCK_MODES + mode, rng.random(len(k)))]

            touchdown(k[scored], t[scored])
            if safety.any():
                ks, ts = k[safety], t[safety]
                score[1 - ts, ks] += 2
                add("safeties", ks, 1 - ts)
                kickoff(ks, 1 - ts)
            gave = lost > 0
            if gave.any():
                kg, tg = k[gave], t[gave]
                house = col["defence_td"][row][gave]
                after = col["after_turnover"][row][gave]
                after = np.where(np.isnan(after), 100 - spot[gave], after)
                change_of_possession(kg[~house], after[~house])
                add("return_td", kg[house], 1 - tg[house])
                touchdown(kg[house], 1 - tg[house])
            plain = ~scored & ~safety & ~gave
            kp = k[plain]
            yl[kp] = spot[plain]
            moved = gain[plain] >= togo[kp]
            add("first_downs", kp[moved], t[plain][moved])
            new_series(kp[moved])
            short = kp[~moved]
            togo[short] -= gain[plain][~moved]
            down[short] += 1

        # Out of downs: the other side takes over where the ball is.
        failed = g[(down[g] > 4) & (pos[g] == before)]
        change_of_possession(failed, 100 - yl[failed])

        half_sec[g] -= used
        in_ot = half[g] == 3
        ot_possessions[g[in_ot & (pos[g] != before)]] += 1
        decided = score[0, g] != score[1, g]
        active[g[in_ot & decided & (ot_possessions[g] >= 2)]] = False

        ended = g[half_sec[g] <= 0]
        was = half[ended].copy()
        first_half, second_half = ended[was == 1], ended[was == 2]
        active[ended[was == 3]] = False
        half[first_half], half_sec[first_half] = 2, 1800.0
        kickoff(first_half, 1 - opening[first_half])
        tied = score[0, second_half] == score[1, second_half]
        active[second_half[~tied]] = False
        extra = second_half[tied]
        half[extra], half_sec[extra] = 3, 600.0
        ot_possessions[extra] = 0
        kickoff(extra, rng.integers(0, 2, len(extra)))

    stats["overtime"] = half == 3
    return stats
