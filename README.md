# Neil's Fantasy Football

Two command-line tools for DraftKings classic (full-roster) NFL tournaments:

1. **Simulator** – turns mean / floor / ceiling projections into thousands of simulated slates.
2. **Lineup analyzer** – builds lineups and ranks them by expected ROI in a given contest.

## Setup

```
pip install -e ".[dev]"
```

## Quick start (made-up sample slate)

```
ffsim sample-slate data/sample/projections.csv
ffsim simulate data/sample/projections.csv
ffsim field out/simulation.npz
ffsim lineups out/simulation.npz
```

`ffsim simulate` runs 50,000 slates by default, and `ffsim lineups` then takes about four and a half minutes; `--sims 20000` roughly halves that at the cost of noisier ROI figures. It writes the 150-lineup portfolio to `out/portfolio.csv` and every graded lineup to `out/lineups.csv`. Without `--contest` it uses `contests/flea_flicker.json`.

## Projections file

A CSV with one row per player. Column names are flexible (`proj`, `own%`, `opp` and so on are recognised).

| Column      | Required | Notes                                                     |
|-------------|----------|-----------------------------------------------------------|
| `name`      | yes      |                                                           |
| `position`  | yes      | QB, RB, WR, TE, DST                                       |
| `team`      | yes      |                                                           |
| `opponent`  | yes      | Used to group players into games                          |
| `salary`    | yes      | DraftKings salary                                         |
| `mean`      | yes      | Projected DraftKings points                               |
| `floor`     | no       | Treated as the 10th percentile unless configured          |
| `ceiling`   | no       | Treated as the 90th percentile unless configured          |
| `ownership` | for `lineups` | Projected ownership, percent. If the file has `Large Field` and `Small Field` columns, large field is used; `--ownership "Small Field"` picks the other |
| `rush_share` | no      | For quarterbacks: share of projected points from rushing (0.35 or 35). Blank means a typical passer (0.12) |

## Using a DraftKings entries export

Reserve your entries on DraftKings, then download the entries file ("DKEntries.csv") from the lineup upload page. It lists your entries and every player on the slate with their ID, salary and game. Keep it in `data/raw/`, which is not committed.

```
ffsim dk-slate data/raw/DKEntries.csv
ffsim simulate projections.csv --dk data/raw/DKEntries.csv
ffsim lineups out/simulation.npz --entries data/raw/DKEntries.csv
```

- `dk-slate` lists the contests you have entered and saves the slate's players.
- With `--dk`, projections only need `name` and `mean` (plus floor, ceiling and ownership). Players are matched to DraftKings by name; position, team, opponent and salary come from DraftKings. Projected players it cannot match are listed and dropped.
- With `--entries`, `lineups` also writes `out/dk_upload.csv`, ready to upload: one portfolio lineup per entry in the contest, best first. The contest is matched by name; use `--contest-id` to pick another.

## How the simulator works

- Each player gets a skewed distribution (different spread below and above its centre, cut off at zero) fitted so its mean, floor and ceiling match the projections.
- Players in the same game are correlated by role: a quarterback with their own pass catchers, opposing offenses with each other, a running back with their own defense, and offenses against the opposing defense. Players in different games are independent.
- A quarterback with a `rush_share` above the typical 0.12 is tied less tightly to his and the opposing pass catchers, in proportion to how much less of his scoring comes from passing.
- `out/simulation_report.csv` compares the simulated mean, floor and ceiling with the inputs for every player.

This simulates fantasy scores with game structure. It is not a play-by-play simulation. The correlation values in `src/ffsim/settings.py` are approximate starting points, not fitted to data here.

## How the lineup analyzer works

- **Opponents:** samples a field of lineups, one for every other entry in the contest (`--field 20000` is faster but makes ROI figures noisier: two 20,000-lineup samples moved a 150-lineup set's ROI by 20 points). See "The opponent field" below.
- **Candidates:** a wide search. An optimizer builds a few hundred lineups, each the best for a blend of the mean projections and one simulated slate. Then 200,000 lineups are sampled around the players those lineups favour, screened against part of the field, and re-sampled around the survivors. The best 5,000 go on to full grading.
- **Grading:** in every simulated slate each candidate is ranked against the field and paid from the contest's payout table. Duplicated lineups split prizes. Grading uses slates the search never saw, so a lineup is never graded on a slate it was built or screened on.
- **Portfolio:** 150 lineups chosen as a set from the top 2,000 (see "Choosing the portfolio"), subject to a per-player exposure cap and a limit on shared players. Saved to `out/portfolio.csv`.

### Choosing the portfolio

Expected winnings add up across lineups, so taking the top lineups one by one always has the highest expected ROI. It also tends to pick lineups that win or lose together. By default the portfolio is instead chosen to maximise the expected growth of a bankroll: each lineup is valued by what it adds in the slates where the lineups already chosen are not winning.

- `--bankroll DOLLARS` sets the bankroll (default: 20 times the cost of the entries). A smaller bankroll buys more diversification at the cost of expected ROI; a larger one moves toward the plain top lineups.
- `--bankroll 0` takes the top lineups one by one.

Each run prints both versions side by side: expected ROI, chance of profit, median return, chance of a 5x return, and chance of losing half the entry cost.

### The opponent field

ROI is only as good as the simulated opponents, so the field is built to hit explicit targets, all in the `field` section of `src/ffsim/settings.py`:

| Target                         | Default                                         |
|--------------------------------|-------------------------------------------------|
| Player ownership               | Your projected ownership                        |
| QB with no / one / two-plus of their own WRs and TEs | 12.6% / 51.9% / 35.5%     |
| Stacks that add an opposing WR or TE | 37%                                       |
| Flex split between RB, WR, TE  | Implied by your ownership totals per position   |
| Salary left unspent            | 47% use the whole cap, 23% leave $100, 23% leave $200-400, 7% more (average $137) |
| Defense facing the lineup's own QB | Rare (about 1%)                             |

The stacking, bring-back and salary figures are measured from the 8 November 2026 $5 Flea Flicker (118,855 entries), one contest so far.

`ffsim field out/simulation.npz` prints what the sampled field actually looks like on each of these, plus the players whose field ownership is furthest from your projection. Add `--out out/field.csv` to save the lineups.

Ownership and stacking targets can conflict (a 30%-owned quarterback forces ownership onto their receivers); when they do, the stack mix wins and the report shows the ownership gap.

The field is also made of three kinds of builder, tuned by replaying that contest so the simulated field's spread of lineup quality and its duplicated lineups match the real one:

| Builder   | Share | How they build                                                        |
|-----------|-------|-----------------------------------------------------------------------|
| Casual    | 25%   | Loosely tied to ownership, more off-the-board players, fewer stacks   |
| Typical   | 68%   | Follows ownership                                                     |
| Optimizer | 7%    | Each lineup is the highest-projected of 40 draws; these are the lineups that get duplicated |

Replayed at full size (118,855 lineups), the simulated field against the real one:

| Measure                                         | Real  | Simulated |
|-------------------------------------------------|-------|-----------|
| Shares its lineup with another entry            | 8.5%  | 7.0%      |
| Duplicated share, top tenth by projection       | 39.2% | 43.7%     |
| Spread (sd) of projected lineup totals          | 7.92  | 7.79      |
| Projected total, 1st / 99th percentile vs average | -21.6 / +14.6 | -20.1 / +13.8 |

Known gaps:

- **One contest.** Everything above is fitted to a single week, so it has not yet been tested on a contest it was not tuned to.
- **Flex split follows the ownership projections**, which that week overstated running backs and understated receivers (simulated flex 59% RB / 9% WR; real 40% / 24%).
- **No repeated entries.** Some real entrants enter one lineup many times (up to 150); the simulated field does not.

Output columns worth knowing:

| Column           | Meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| `adj_roi_pct`    | ROI estimate used for ranking (see below)                               |
| `roi_pct`        | Raw simulated ROI                                                       |
| `roi_se_pct`     | Standard error of the raw ROI                                           |
| `win_pct`, `top01_pct`, `top1_pct`, `cash_pct` | How often the lineup finishes first, top 0.1%, top 1%, in the money |
| `expected_dupes` | Expected number of opponents entering the identical lineup              |

### Reading the ROI numbers

- Raw tournament ROI is noisy: it swings on a handful of simulated first places. `adj_roi_pct` steadies it using the lineup's finish rates, which are far less noisy. More simulations (`ffsim simulate --sims 50000`) tighten both.
- The simulated field only knows ownership. Real opponents also optimise, so real ROI will be lower than shown. Use the numbers to compare lineups, not as a profit forecast.

## Contests

A contest is a JSON file with `entry_fee`, `entries` and either an explicit `payouts` table (`contests/sample_small_gpp.json`) or `auto` settings that generate a typical top-heavy curve (`contests/typical_gpp.json`: $20 entry, 100,000 entries). `contests/flea_flicker.json` is the real payout table of the $5 Flea Flicker. The other two are illustrative; copy the real payout table from the contest page for real use. If only the top prizes are known, a `tail` block (`to`, `total`, `start`, `min_prize`) estimates the lower places.

## Other commands

- `ffsim diff old.csv new.csv` lists players whose projection moved between two files.
- `--config my.json` on `simulate` or `lineups` overrides any default in `src/ffsim/settings.py`, for example `{"floor_percentile": 0.15, "ceiling_percentile": 0.85}`.

## Grading your own lineups

```
ffsim grade my_lineups.csv out/simulation.npz
```

Grades lineups you supply against the same simulated slates and opponent field: expected ROI, finish rates and expected duplicates for each, and how the whole set performs together. The file can be a DraftKings upload or entries file, or any CSV with nine players per row, given as DraftKings IDs, `Name (ID)` or names. Unreadable or illegal lineups are listed and skipped. Results go to `out/graded_lineups.csv` in the order given.

Lineups from elsewhere get an unbiased grade. Lineups this tool built are graded partly on slates it searched on, so expect their figure here to be a little flattering, though lower than the figure printed when they were chosen.

## During the games: live ranking and late swaps

Download the contest standings while games are in progress (for example at 2pm, 3pm and 3:30pm before the 4pm games), then:

```
ffsim live standings.csv --entries data/raw/DKEntries.csv --projections latest_projections.csv
```

Every opponent's real lineup and live score is in that file, so no simulated field is used. Only the points still to come are simulated: full projections for games not yet started, and a share for games in progress. How much of each game is left is worked out from the player-minutes remaining DraftKings reports for each entry.

It prints, and saves to `out/live/`:

- **Your lineups ranked by expected prize**, with rank and points now, players and salary still to start, projected points to come, projected final score, and the chance of a top-1% finish and of cashing.
- **Late swaps:** for each lineup, the best swap among players whose game has not started, in a slot their position fits and within the salary cap. A swap is found on half the simulations and recommended only if it also clearly gains on the other half; the rest are listed as within the noise.
- **`dk_swap_upload.csv`**: the changed lineups, ready to upload to DraftKings.

Your lineups are found by the entry IDs in your entries export (or `--user NAME`). A run takes about three minutes; `--sims 20000` is steadier and slower. Limits: a player in a game in progress is projected from his pre-game numbers scaled by time left, not from how his game is going, and swaps only replace a player within his own roster slot.

## Checking the field against a real contest

After a contest locks, DraftKings lets entrants download its standings, which list every lineup entered. Keep these in `data/raw/contests/` (not committed; they contain other entrants' usernames).

```
ffsim contest-report data/raw/contests/standings.csv --slate that-weeks-slate.csv
```

Single-game captain-mode contests are read too, with their own measures (duplicated lineups, and with a slate file, team splits, salary and captain position); the lineup tools themselves only build full-roster lineups.

It reports, overall and by how many entries each entrant has: flex use, duplicated lineups, stacking, salary used and average score, plus actual ownership. Compare it with `ffsim field` for the same slate. `--slate` is a CSV with name, team, opponent and salary for that week (the output of `ffsim dk-slate`, or a projections file); without it, stacking and salary are skipped.

## Historical results

`python scripts/fetch_rotoguru.py 2017 2025` downloads weekly DraftKings points and salaries per player from RotoGuru into `data/history/dk_results.csv`, for fitting correlations and outcome shapes. It has not been run against the live site: the site is plain HTTP, which the cloud session that wrote it cannot reach.

## Tests

```
pytest
```

## Not built yet

- Checking the field against real contest results (DraftKings lets entrants download a contest's standings, which list every lineup).
- Fetching projections on a schedule and reporting changes automatically.
- A DraftKings upload file (needs DraftKings player IDs).
- Showdown slates and FanDuel.

## Layout

| Path          | Contents                                   |
|---------------|--------------------------------------------|
| `src/ffsim/`  | The tools                                  |
| `contests/`   | Contest definitions                        |
| `data/`       | Projections; `data/raw/` is not committed  |
| `notes/`      | Strategy and league notes                  |
| `tests/`      | Automated tests                            |
