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

`ffsim lineups` takes a little over two minutes. It writes the 150-lineup portfolio to `out/portfolio.csv` and every graded lineup to `out/lineups.csv`. Without `--contest` it uses `contests/flea_flicker.json`.

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
| `floor`     | no       | Treated as the 15th percentile unless configured          |
| `ceiling`   | no       | Treated as the 85th percentile unless configured          |
| `ownership` | for `lineups` | Projected ownership, percent                         |

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
- `out/simulation_report.csv` compares the simulated mean, floor and ceiling with the inputs for every player.

This simulates fantasy scores with game structure. It is not a play-by-play simulation. The correlation values in `src/ffsim/settings.py` are approximate starting points, not fitted to data here.

## How the lineup analyzer works

- **Opponents:** samples a field of lineups (20,000 by default). See "The opponent field" below.
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
| QB with no / one / two-plus of their own WRs and TEs | 17% / 45% / 38%           |
| Stacks that add an opposing WR or TE | 40%                                       |
| Flex split between RB, WR, TE  | Implied by your ownership totals per position   |
| Median salary used             | $49,700 (none below $48,000)                    |
| Defense facing the lineup's own QB | Rare                                        |

`ffsim field out/simulation.npz` prints what the sampled field actually looks like on each of these, plus the players whose field ownership is furthest from your projection. Add `--out out/field.csv` to save the lineups.

The stacking and salary defaults are estimates for large tournaments, not measured from contest data. Ownership and stacking targets can conflict (a 30%-owned quarterback forces ownership onto their receivers); when they do, the stack mix wins and the report shows the ownership gap.

Known gaps: the field has no notion of sharper and weaker entrants, of one entrant's 150 correlated lineups, or of late-swap. It also produces almost no duplicated lineups, where real large fields duplicate popular builds.

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
- `--config my.json` on `simulate` or `lineups` overrides any default in `src/ffsim/settings.py`, for example `{"floor_percentile": 0.1, "ceiling_percentile": 0.9}`.

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
