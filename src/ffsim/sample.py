"""Write a made-up slate (fictional teams and players) for trying the tools out."""
import numpy as np
import pandas as pd

ROLES = [  # position, label, share of team strength, salary base, salary per point
    ("QB", "QB1", 18.0, 4200, 130), ("QB", "QB2", 0.6, 4000, 0),
    ("RB", "RB1", 14.5, 3800, 210), ("RB", "RB2", 7.0, 3600, 190), ("RB", "RB3", 2.5, 3800, 100),
    ("WR", "WR1", 15.0, 3000, 250), ("WR", "WR2", 11.0, 3000, 230), ("WR", "WR3", 7.5, 2900, 200),
    ("WR", "WR4", 3.5, 2800, 120), ("WR", "WR5", 1.5, 3000, 0),
    ("TE", "TE1", 8.5, 2300, 220), ("TE", "TE2", 2.5, 2400, 80),
    ("DST", "DST", 7.0, 1500, 220),
]


def sample_slate(games: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(games):
        home, away = f"H{g + 1:02d}", f"A{g + 1:02d}"
        for team, opp in ((home, away), (away, home)):
            strength = rng.normal(1.0, 0.10)
            for pos, label, base, sal_base, sal_rate in ROLES:
                mean = max(0.3, base * (1 / strength if pos == "DST" else strength) * rng.lognormal(0, 0.12))
                salary = int(np.clip(round(0.9 * (sal_base + sal_rate * mean * rng.normal(1, 0.07)) / 100) * 100, {"QB": 4000, "RB": 4000, "WR": 3000, "TE": 2500, "DST": 2000}[pos], 9500))
                spread = rng.uniform(0.5, 0.65)
                rows.append({
                    "name": f"{team} {label}", "position": pos, "team": team, "opponent": opp,
                    "salary": salary, "mean": round(mean, 2),
                    "floor": round(mean * (1 - spread), 2),
                    "ceiling": round(mean * (1 + spread * rng.uniform(1.0, 1.35)), 2),
                })
    df = pd.DataFrame(rows)
    # Ownership: more for players who project well for their price, leaning cheap or
    # dear by just enough that the average lineup costs a little under the cap.
    value = df["mean"] / df["salary"] * 1000
    slots = {"QB": 1, "RB": 2.4, "WR": 3.45, "TE": 1.15, "DST": 1}

    def ownership(lean: float) -> pd.Series:
        own = pd.Series(0.0, index=df.index)
        for pos, n in slots.items():
            m = df["position"] == pos
            w = np.exp(1.1 * (value[m] - value[m].mean()) / value[m].std()) * df.loc[m, "mean"] ** 1.3
            w = w * np.exp(lean * df.loc[m, "salary"] / 1000)
            own[m] = 100 * n * w / w.sum()
        return own

    lo, hi = -3.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (ownership(mid) / 100 * df["salary"]).sum() < 49300 else (lo, mid)
    df["ownership"] = ownership(lo).round(1)
    return df
