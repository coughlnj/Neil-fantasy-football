"""Sample the lineups opponents enter, matched to projected ownership."""
from __future__ import annotations

import numpy as np
import pandas as pd


class LineupSampler:
    """Draws many DraftKings classic lineups at once, weighted per player.

    Lineups favour pass catchers who share a team with the quarterback (stacks), avoid a
    defense facing that quarterback, and must land inside a salary window.
    """

    def __init__(self, players: pd.DataFrame, settings: dict, cfg: dict | None = None):
        self.cfg = cfg or settings["field"]
        self.cap = settings["site"]["salary_cap"]
        self.salary = players["salary"].to_numpy(float)
        teams = {t: i for i, t in enumerate(sorted(set(players["team"]) | set(players["opponent"])))}
        self.team = players["team"].map(teams).to_numpy()
        self.opp = players["opponent"].map(teams).to_numpy()
        pos = players["position"].to_numpy()
        self.idx = {p: np.flatnonzero(pos == p) for p in ("QB", "RB", "WR", "TE", "DST")}
        self.idx["catcher"] = np.flatnonzero(np.isin(pos, ["WR", "TE"]))
        # Target share of lineups by salary left unspent (None: no preference), and how
        # strongly the flex pick is drawn toward using up the cap while sampling.
        unspent = self.cfg.get("unspent_salary")
        self.unspent = np.array(unspent, dtype=float) if unspent else None
        self.fade = 150.0 if unspent else np.inf
        self.catcher = np.isin(pos, ["WR", "TE"])
        if "stack_classes" in self.cfg:
            c = self.cfg["stack_classes"]
            self.class_target = np.array(
                [c["none"], c["single"], c["single_bringback"], c["double"], c["double_bringback"]], dtype=float
            )
        else:
            mix, bb = self.cfg["stack_mix"], self.cfg["bringback_rate"]
            self.class_target = np.array(
                [mix["none"], mix["single"] * (1 - bb), mix["single"] * bb, mix["double"] * (1 - bb), mix["double"] * bb]
            )
        self.class_target = self.class_target / self.class_target.sum()
        self.flex_bonus = {p: np.log(self.cfg["flex_weight"][p]) for p in ("RB", "WR", "TE")}

    def _draw(self, logw: np.ndarray, kind: np.ndarray, rng: np.random.Generator):
        """Draw one lineup per entry of `kind` (the stack type to aim for). Returns the
        lineups, their salaries, and a mask of those that came out as the aimed type."""
        batch = len(kind)

        def keys(position: str) -> np.ndarray:
            idx = self.idx[position]
            return logw[idx][None, :] + rng.gumbel(size=(batch, len(idx)))

        qb = self.idx["QB"][keys("QB").argmax(axis=1)]
        q_team, q_opp = self.team[qb][:, None], self.opp[qb][:, None]
        # Stack types: 0 none, 1 single, 2 single + bring-back, 3 double, 4 double +
        # bring-back. Chosen pass catchers are forced in; others are shut out where the
        # type calls for it (no teammates, exactly one teammate, no opposing catcher).
        c_idx = self.idx["catcher"]
        rows = np.arange(batch)[:, None]
        mate = self.team[c_idx][None, :] == q_team
        foe = self.team[c_idx][None, :] == q_opp
        base = logw[c_idx][None, :] + rng.gumbel(size=(batch, len(c_idx)))
        mate_order = np.argsort(-np.where(mate, base, -np.inf), axis=1)[:, :2]
        foe_pick = np.where(foe, base, -np.inf).argmax(axis=1)[:, None]
        adjust = np.zeros((batch, len(logw)))
        block = (mate & (kind <= 2)[:, None]) | (foe & np.isin(kind, [1, 3])[:, None])
        adjust[rows, c_idx[None, :]] = np.where(block, -np.inf, 0.0)
        force = lambda who, col: adjust.__setitem__((np.flatnonzero(who), c_idx[col[who]]), 1e6)
        force(kind >= 1, mate_order[:, 0])
        force(kind >= 3, mate_order[:, 1])
        force(np.isin(kind, [2, 4]), foe_pick[:, 0])

        # Defense first, so the flex can be chosen to fit the salary that is left.
        dst_idx = self.idx["DST"]
        penalty = np.log(max(self.cfg["dst_vs_qb_penalty"], 1e-12))
        k = keys("DST") + np.where(self.opp[dst_idx][None, :] == q_team, penalty, 0.0)
        dst = dst_idx[k.argmax(axis=1)]

        picks, pools = [qb[:, None]], []
        for position, count in (("RB", 2), ("WR", 3), ("TE", 1)):
            idx = self.idx[position]
            k = keys(position) + adjust[:, idx]
            order = np.argsort(-k, axis=1)[:, :count]
            picks.append(idx[order])
            np.put_along_axis(k, order, -np.inf, axis=1)
            pools.append((idx, k + self.flex_bonus[position]))
        core = np.concatenate(picks + [dst[:, None]], axis=1)
        base = self.salary[core].sum(axis=1)

        # The flex takes whoever fits: players who would break the cap are out, and the
        # more salary a player would leave unspent the less likely he is.
        flex_idx = np.concatenate([idx for idx, _ in pools])
        k = np.concatenate([key for _, key in pools], axis=1)
        left = self.cap - base[:, None] - self.salary[flex_idx][None, :]
        fill = 0.0 if np.isinf(self.fade) else -left / self.fade
        k = np.where(left >= 0, k + fill, -np.inf)
        choice = k.argmax(axis=1)
        fits = np.isfinite(k[np.arange(batch), choice])
        lineups = np.concatenate([core[:, :-1], flex_idx[choice][:, None], core[:, -1:]], axis=1)

        catcher = self.catcher[lineups]
        mates = (catcher & (self.team[lineups] == q_team)).sum(axis=1)
        bring = (catcher & (self.team[lineups] == q_opp)).any(axis=1)
        got = np.where(mates == 0, 0, 2 * np.minimum(mates, 2) - 1 + bring)
        return lineups, self.salary[lineups].sum(axis=1), fits & (got == kind), base

    def salary_tilt(self, logw: np.ndarray, rng: np.random.Generator, classes: np.ndarray | None = None) -> np.ndarray:
        """Lean the weights toward cheaper or dearer players so that, before the flex is
        chosen, a typical lineup has about one typical flex salary left. Returns the weights."""
        flex = np.concatenate([self.idx[p] for p in ("RB", "WR", "TE")])
        weight = np.exp(logw[flex] - logw[flex].max())
        room = float((weight * self.salary[flex]).sum() / weight.sum())
        if not np.isinf(self.fade):
            room += self.fade
        lo, hi = -3.0, 3.0  # change in log-weight per $1,000 of salary
        for _ in range(9):
            mid = (lo + hi) / 2
            kind = rng.choice(5, size=4000, p=self.class_target if classes is None else classes)
            base = self._draw(logw + mid * self.salary / 1000, kind, rng)[3]
            lo, hi = (mid, hi) if np.median(base) < self.cap - room else (lo, mid)
        return logw + (lo + hi) / 2 * self.salary / 1000

    def sample(
        self,
        logw: np.ndarray,
        n: int,
        rng: np.random.Generator,
        classes: np.ndarray | None = None,
        best_of: float = 1.0,
        value: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return n lineups as sorted player indices, shape (n, 9), with stack types in
        the given proportions (default: the configured ones).

        best_of models how hard entrants optimise: each lineup is the highest-valued of
        that many draws (value is per player, e.g. projected points). 2.5 means half the
        lineups are the best of 2 and half the best of 3; 1 means no selection."""
        classes = self.class_target if classes is None else np.asarray(classes, dtype=float) / np.sum(classes)
        if value is None or best_of <= 1:
            best_of = 1.0
        draws = int(np.ceil(best_of))
        want = np.floor(classes * n).astype(int)
        want[np.argmax(classes)] += n - want.sum()
        need = want * draws
        found, empty = [[] for _ in range(5)], 0
        while need.sum() > 0 and empty < 40:
            kind = rng.choice(5, size=20000, p=need / need.sum())
            lineups, total, ok, _ = self._draw(logw, kind, rng)
            ok &= (total <= self.cap) & (total >= self.cfg["min_salary"])
            if self.unspent is not None and ok.any():  # thin to the target spread of unspent salary
                band = np.searchsorted([100, 200, 500, 1000, 2000], self.cap - total, side="right")
                seen = np.bincount(band[ok], minlength=6) / ok.sum()
                keep = np.divide(self.unspent, seen, out=np.zeros(6), where=seen > 0)
                ok &= rng.random(len(kind)) < (keep / keep.max())[band]
            empty = 0 if ok.any() else empty + 1
            for c in np.flatnonzero(need):
                take = lineups[ok & (kind == c)][: need[c]]
                found[c].append(take)
                need[c] -= len(take)
        if need.sum() > 0:
            raise RuntimeError(
                "Could not build enough lineups inside the salary window. Check salaries and "
                "ownership, or lower field.min_salary."
            )
        chosen = []
        for c in np.flatnonzero(want):
            sets = np.concatenate(found[c]).reshape(want[c], draws, -1)
            if draws > 1:
                score = value[sets].sum(axis=2)
                fewer = rng.random(want[c]) >= best_of - (draws - 1)  # these get one draw fewer
                score[fewer, -1] = -np.inf
                sets = sets[np.arange(want[c]), score.argmax(axis=1)]
            else:
                sets = sets[:, 0]
            chosen.append(sets)
        out = np.sort(np.concatenate(chosen), axis=1)
        return out[rng.permutation(len(out))]


def target_ownership(players: pd.DataFrame, settings: dict) -> np.ndarray:
    """Ownership as fractions, rescaled so each position fills its roster slots."""
    own = players["ownership"].to_numpy(float)
    if np.isnan(own).all():
        raise ValueError("The projections have no ownership column, which the field needs.")
    own = np.nan_to_num(own, nan=0.0).clip(min=0.0) / 100.0
    pos = players["position"].to_numpy()
    for p in ("QB", "DST"):
        own[pos == p] /= own[pos == p].sum()
    flex = np.isin(pos, settings["site"]["flex_positions"])
    split = settings["field"].get("flex_split")
    if split:
        total = sum(split.values())
        for p in settings["site"]["flex_positions"]:
            own[pos == p] *= (settings["site"]["roster"][p] + split[p] / total) / own[pos == p].sum()
    own[flex] *= 7.0 / own[flex].sum()
    return own


def entrant_groups(settings: dict) -> list[dict]:
    """The kinds of entrant in the field. Without field.groups, one group using the
    field-wide stack types."""
    cfg = settings["field"]
    groups = cfg.get("groups")
    if not groups:
        c = cfg["stack_classes"]
        return [{"name": "everyone", "share": 1.0, "sharpness": 1.0, "best_of": 1.0,
                 "stack_classes": [c["none"], c["single"], c["single_bringback"], c["double"], c["double_bringback"]]}]
    return groups


def sample_field_by_group(players: pd.DataFrame, settings: dict, n: int, seed: int | None = None):
    """Sample n opponent lineups. Returns (lineups, realised ownership fractions, group
    index of each lineup)."""
    rng = np.random.default_rng(seed)
    sampler = LineupSampler(players, settings)
    target = target_ownership(players, settings)
    groups = entrant_groups(settings)
    shares = np.array([g["share"] for g in groups], dtype=float)
    shares = shares / shares.sum()
    value = players["mean"].to_numpy(float) if "mean" in players.columns else None
    wanted_unspent = sampler.unspent.copy() if sampler.unspent is not None else None
    active = target > 0
    logw = np.where(active, np.log(np.where(active, target, 1.0)), -np.inf)
    cfg = settings["field"]
    pos = players["position"].to_numpy()

    def draw(weights: np.ndarray, count: int):
        sizes = np.floor(shares * count).astype(int)
        sizes[np.argmax(shares)] += count - sizes.sum()
        parts, labels = [], []
        for number, (group, size) in enumerate(zip(groups, sizes)):
            if size == 0:
                continue
            classes = np.asarray(group["stack_classes"], dtype=float)
            classes = classes / classes.sum()
            # Sharpness below 1 flattens the weights: more off-the-board picks.
            own = np.where(active, weights * group.get("sharpness", 1.0), -np.inf)
            tilted = sampler.salary_tilt(own, rng, classes)
            parts.append(sampler.sample(tilted, size, rng, classes, group.get("best_of", 1.0), value))
            labels.append(np.full(size, number))
        order = rng.permutation(count)
        return np.concatenate(parts)[order], np.concatenate(labels)[order]

    start = logw.copy()
    best = (logw.copy(), dict(sampler.flex_bonus), None if wanted_unspent is None else sampler.unspent.copy())
    best_gap = np.inf
    # Nudge the weights until sampled ownership matches the target. Adjustments are
    # capped, and the best round is kept, so targets that cannot all be met at once
    # (given salaries) settle on the closest achievable field instead of running away.
    for _ in range(cfg["calibration_rounds"]):
        try:
            trial, _ = draw(logw, cfg["calibration_lineups"])
        except RuntimeError:
            break
        seen = np.bincount(trial.ravel(), minlength=len(players)) / len(trial)
        gap = np.abs(seen - target).max()
        if gap < best_gap:
            best = (logw.copy(), dict(sampler.flex_bonus), None if wanted_unspent is None else sampler.unspent.copy())
            best_gap = gap
        if wanted_unspent is not None:  # selection shifts salary use; steer it back
            band = np.searchsorted([100, 200, 500, 1000, 2000], sampler.cap - sampler.salary[trial].sum(axis=1), side="right")
            got = np.maximum(np.bincount(band, minlength=6) / len(trial), 1e-4)
            sampler.unspent = sampler.unspent * (np.maximum(wanted_unspent, 1e-4) / got) ** 0.7
            sampler.unspent = sampler.unspent / sampler.unspent.sum()
        for p, slots in settings["site"]["roster"].items():
            if p in sampler.flex_bonus:
                want, have = target[pos == p].sum() - slots, seen[pos == p].sum() - slots
                sampler.flex_bonus[p] += 0.8 * np.clip(np.log(max(want, 0.005) / max(have, 0.005)), -2.0, 2.0)
        step = np.log(target[active]) - np.log(np.maximum(seen[active], 0.2 / len(trial)))
        logw[active] = np.clip(logw[active] + 0.8 * np.clip(step, -2.0, 2.0), start[active] - 3.0, start[active] + 3.0)
    logw, sampler.flex_bonus, unspent = best
    if unspent is not None:
        sampler.unspent = unspent
    lineups, labels = draw(logw, n)
    realised = np.bincount(lineups.ravel(), minlength=len(players)) / len(lineups)
    return lineups, realised, labels


def sample_field(players: pd.DataFrame, settings: dict, n: int, seed: int | None = None):
    """Sample n opponent lineups. Returns (lineups, realised ownership fractions)."""
    lineups, realised, _ = sample_field_by_group(players, settings, n, seed)
    return lineups, realised


def field_report(lineups: np.ndarray, players: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Describe how a set of lineups is built, so it can be compared with real contests."""
    pos, team, opp = (players[c].to_numpy() for c in ("position", "team", "opponent"))
    salary = players["salary"].to_numpy(float)[lineups].sum(axis=1)
    l_pos, l_team = pos[lineups], team[lineups]
    is_qb = l_pos == "QB"
    qb_team = l_team[is_qb][:, None]
    qb_opp = opp[lineups][is_qb][:, None]
    catcher = np.isin(l_pos, ["WR", "TE"])
    mates = (catcher & (l_team == qb_team)).sum(axis=1)
    bring_back = (catcher & (l_team == qb_opp)).any(axis=1)
    dst_faces_qb = ((l_pos == "DST") & (opp[lineups] == qb_team)).any(axis=1)
    flex = {p: ((l_pos == p).sum(axis=1) > n).mean() for p, n in (("RB", 2), ("WR", 3), ("TE", 1))}
    _, counts = np.unique(lineups, axis=0, return_counts=True)
    pct = lambda x: round(100 * float(np.mean(x)), 1)
    rows = [
        ("QB with no pass catcher", pct(mates == 0), "%"),
        ("QB with exactly 1 pass catcher", pct(mates == 1), "%"),
        ("QB with 2 or more pass catchers", pct(mates >= 2), "%"),
        ("Stacks that add an opposing pass catcher", pct(bring_back[mates >= 1]), "%"),
        ("Defense facing its own QB", pct(dst_faces_qb), "%"),
        ("Flex is a RB", pct(flex["RB"]), "%"),
        ("Flex is a WR", pct(flex["WR"]), "%"),
        ("Flex is a TE", pct(flex["TE"]), "%"),
        ("Median salary used", float(np.median(salary)), "$"),
        ("Average salary left unspent", round(float(settings["site"]["salary_cap"] - salary.mean()), 0), "$"),
        ("Lineups using $49,500 or more", pct(salary >= settings["site"]["salary_cap"] - 500), "%"),
        ("Lineups that duplicate another in this sample", pct(np.repeat(counts, counts) > 1), "%"),
    ]
    if "mean" in players.columns:
        totals = players["mean"].to_numpy(float)[lineups].sum(axis=1)
        rows.append(("Average projected lineup total", round(float(totals.mean()), 1), "pts"))
        rows.append(("Spread (sd) of projected lineup totals", round(float(totals.std()), 2), "pts"))
    return pd.DataFrame(rows, columns=["measure", "value", "unit"])
