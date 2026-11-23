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

This simulates fantasy scores with game structure. It is not a play-by-play simulation. The correlation values in `src/ffsim/settings.py` have been checked against real DraftKings scores from every game from 2024 to mid-2026 (707 games, scores rebuilt from play-by-play). The skill-position values were already within about 0.05 of the real figures and were left alone. Those involving a defence were not, and are now set from the data: a quarterback and his own defence move slightly against each other (-0.14 in real games, where the old value gave +0.02), a running back gains much less from his own defence than assumed (0.05, not 0.19), a quarterback is hurt less by the opposing defence (-0.28, not -0.39), the two defences move against each other (-0.17), and so do the two lead running backs (-0.10). After the change the simulator is within 0.03 of real games on average across 32 pairs of roles. One gap remains: in a quarterback's best 5% of games the opposing quarterback and pass catchers do better than the simulator has them (about 0.65 standard deviations for the opposing quarterback, against 0.41 here and 0.50 for the play-by-play simulator). That rests on 67 games.

## How the lineup analyzer works

- **Opponents:** samples a field of lineups, one for every other entry in the contest (`--field 20000` is faster but makes ROI figures noisier: two 20,000-lineup samples moved a 150-lineup set's ROI by 20 points). See "The opponent field" below.
- **Candidates:** a wide search. An optimizer builds a few hundred lineups, each the best for a blend of the mean projections and one simulated slate. Then 200,000 lineups are sampled around the players those lineups favour, screened against part of the field, and re-sampled around the survivors. The best 5,000 go on to full grading.
- **Grading:** in every simulated slate each candidate is ranked against the field and paid from the contest's payout table. Duplicated lineups split prizes. Grading uses slates the search never saw, so a lineup is never graded on a slate it was built or screened on.
- **Portfolio:** 150 lineups chosen as a set from the top 2,000 (see "Choosing the portfolio"), subject to a per-player exposure cap and a limit on shared players. Saved to `out/portfolio.csv`.

### Choosing the portfolio

Expected winnings add up across lineups, so taking the top lineups one by one always has the highest expected ROI. It also tends to pick lineups that win or lose together. By default the portfolio is instead chosen to maximise the expected growth of a bankroll: each lineup is valued by what it adds in the slates where the lineups already chosen are not winning.

- `--bankroll DOLLARS` sets the bankroll (default: 20 times the cost of the entries). A smaller bankroll buys more diversification at the cost of expected ROI; a larger one moves toward the plain top lineups.
- `--bankroll 0` takes the top lineups one by one.
- `--exposure "Player Name=1.0"` gives one player his own exposure limit (here, none), leaving the cap on everyone else. Repeat it for more players.

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

**Ownership is treated as an estimate.** The provider's projected ownership misses: across the 8 and 15 November contests, players projected at 1-3% landed about 50% above or below that, narrowing to about 20% for players above 25%, with no lean either way. A lineup's ROI moves by about 20 points across ownership that plausible, so lineups are graded against four opponent fields (`field.ownership_scenarios`), each built from ownership shaken by those amounts, with the simulated slates shared out between them. Tested on both weeks, ranking candidates this way agreed better with their ranking against the real ownership (rank correlation 0.80, against 0.70 from the projected ownership alone). It adds about five minutes to a build. The shaking also lifts every lineup's ROI by about 14 points, which is now calibrated out (see "Reading the ROI numbers"). Set it to 1 to grade against the projected ownership as given.

The stacking, bring-back and salary figures are measured from the 8 November 2026 $5 Flea Flicker (118,855 entries) and held up on the 15 November one.

`ffsim field out/simulation.npz` prints what the sampled field actually looks like on each of these, plus the players whose field ownership is furthest from your projection. Add `--out out/field.csv` to save the lineups.

Ownership and stacking targets can conflict (a 30%-owned quarterback forces ownership onto their receivers); when they do, the stack mix wins and the report shows the ownership gap.

The field is made of the four kinds of real entrant, by how many entries they play, with each kind's measured share and stacking habits. Each kind mixes lineups guided by ownership with a smaller share of optimizer-style lineups (the highest-projected of 40 draws), which make most of the duplicates. Fitted on the 8 and 15 November contests, with each week's field built from that week's actual ownership so the provider's ownership misses do not get in the way:

| Kind of entrant | Share | Ownership-guided sharpness | Optimizer-style share |
|---|---|---|---|
| Single entry | 13% | 0.65 (loosest) | 9% |
| 2-19 entries | 27% | 0.70 | 14% |
| 20-149 entries | 21% | 0.85 | 18% |
| 150 entries | 38% | 0.95 | 14% |

Each kind then lands close to its real counterpart. On 15 November, the share of lineups with Emmett Johnson (66% owned) by kind, real against simulated: 50/49%, 56/56%, 70/69%, 77/75%; duplicated lineups 4.9/3.5%, 9.9/6.7%, 12.4/10.6%, 9.4/9.7%. An earlier version used three builder types (casual, typical, optimizer) whose extremes only averaged out to the right field: the casual type rostered Johnson in 24% of lineups and the optimizer type in 100%, with most duplicates coming from that one type.

Building around games (`field.pairing`): a player facing the lineup's own defense is about a quarter as likely to be picked, and running backs are about twice as likely when they play for the lineup's defense, its quarterback or the quarterback's opponent. A third pass catcher in a double stack is also favoured. Without these the field put a player against its own defense in 21% of lineups (real: 8%) and a running back with his own defense in 9-12% (real: 15-18%).

Replayed at full size, the simulated field against the real one:

| Measure | Nov 8 real | Nov 8 simulated | Nov 15 real | Nov 15 simulated |
|---|---|---|---|---|
| A player facing own defense | 8.2% | 8.0% | 8.4% | 8.5% |
| RB with own defense | 18.2% | 18.9% | 14.6% | 14.2% |
| QB with own RB / opposing RB | 16.1% / 18.2% | 15.1% / 18.1% | 19.7% / 19.1% | 20.6% / 19.2% |
| Spread (sd) of projected lineup totals | 7.42 | 6.77 | 8.33 | 7.85 |
| Spread (sd) of lineup ownership (sum of log ownership) | 3.49 | 3.22 | 3.31 | 3.19 |
| Shares its lineup with another entry | 8.5% | 4.4% | 9.6% | 8.4% |
| In a group of 5+ identical lineups | 2.1% | 0.7% | 2.3% | 2.4% |
| Cash line / top 1% (scored on real results) | 138.6 / 174.3 | 138.8 / 175.1 | 180.3 / 229.4 | 178.3 / 224.1 |

Known gaps:

- **Two contests.** Everything above is fitted to two weeks.
- **Flex split follows the ownership projections.** With actual ownership it matches; when the provider overstates running backs, so does the field.
- **No repeated entries.** Some real entrants enter one lineup many times (one entered the same lineup 150 times on 8 November); the simulated field does not. Duplicates on 8 November stay low under every setting tried, which looks specific to that slate.
- **Too few very weak lineups.** The bottom 1% of simulated lineups project 117-119 points against 113-115 in the real fields; these rarely cash, so it matters little.
- **The top end depends on the week.** When one game turns into a shootout, the real field has more lineups built around it than the simulated one (15 November: top 1% at 229.4 real, 225 simulated); on 8 November the simulated top end was slightly higher.

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
- **Calibrated to the rake.** All entries together win the prize pool, so the average entry must lose the rake (-15.9% in the Flea Flicker). Grading against the shaken-ownership fields broke that: replaying about 20,000 real entries from each of the 8, 15 and 22 November contests, the average real lineup graded at +0.1%, -3.3% and +2.3%, and a typical simulated lineup at about -2%. The shaking spreads the field's chalk around, so every lineup meets fewer duplicates than in a real contest, chalky lineups most (+15 to +20 points, against +5 to +9 for the least-owned fifth). Against one unshaken field the replay landed at -14.8%, -17.1% and -11.2%, with top-1% rates of 0.98%, 0.94% and 1.01%. So `lineups` and `grade` now also grade 1,500 typical lineups drawn fresh from the unshaken field, and divide every prize by whatever factor puts them back at the rake (the `Calibration:` line in the output, usually prizes scaled down by about 12%). This rescales ROI and prize figures only; the ranking, the picks and the finish rates are unchanged.
- **Judge a portfolio on fresh slates.** A portfolio is picked as the best of thousands of lineups on one set of simulated slates, so its ROI on those slates flatters it (by 15-25 points in past checks). `ffsim week` therefore simulates the slate again with another seed and grades the portfolio, and your own lineups, on those fresh slates. That figure is the one to go by.
- **What the replay says about the ranking.** Lineups the tool rates poorly really do finish poorly. Among real lineups it rated above the median, though, the higher-rated ones did no better over those three slates (top 10% by predicted ROI: 1.4% predicted top-1% rate, 1.0% actual). Most of its preference there is for lower ownership at the same projection, which paid on 15 November and lost badly on 22 November, when the chalk hit. Three slates cannot settle whether that preference is right.
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
- **Late swaps:** each lineup's players whose games have not started are rebuilt together, within the salary the lineup has left, so it can pay down at one position to pay up at another. Each open slot's alternatives are valued on their own first; the best few per slot, plus the best player at each lower price, are combined and every legal combination is graded. The best is found on half the simulations and recommended only if it also clearly gains on the other half; the rest are listed in `live_swaps.csv` with `made` false. `--swap-mode single` goes back to one-for-one swaps. On the 15 November 1:40pm export the full rebuild recommended changes to 22 lineups, 14 of them changing two or more players, for $13.66 of expected prizes against $9.14 from one-for-one swaps.
- **`dk_swap_upload.csv`**: the changed lineups, ready to upload to DraftKings.

DraftKings hides every player whose game has not started, in your lineups as well as opponents'. Pass the lineup file you uploaded with `--my-lineups` so your late-game players are known; entries that no longer match it (edited since) have theirs guessed and get no swap suggestions. Opponents' hidden players are filled in from projected ownership, so figures firm up as games start.

Your lineups are found by the entry IDs in your entries export (or `--user NAME`). A run takes about six minutes; `--sims 20000` is steadier and slower. A player in a game in progress is projected from his pre-game numbers scaled by time left, not from how his game is going. That was checked against 8,000 real player-games: how a player has done so far predicts almost nothing about the rest of his game (a weight of about 0.04 at halftime, against 0.91 on the pre-game expectation), so it is left out. Limits: a replacement goes into the same roster slot as the player he replaces (players are not moved between a position slot and the flex), and each lineup is changed on its own, without regard to your exposure across the set.

## Checking the field against a real contest

After a contest locks, DraftKings lets entrants download its standings, which list every lineup entered. Keep these in `data/raw/contests/` (not committed; they contain other entrants' usernames).

```
ffsim contest-report data/raw/contests/standings.csv --slate that-weeks-slate.csv
```

Single-game captain-mode contests are read too, with their own measures (duplicated lineups, and with a slate file, team splits, salary and captain position); the lineup tools themselves only build full-roster lineups.

It reports, overall and by how many entries each entrant has: flex use, duplicated lineups, stacking, salary used and average score, plus actual ownership. Compare it with `ffsim field` for the same slate. `--slate` is a CSV with name, team, opponent and salary for that week (the output of `ffsim dk-slate`, or a projections file); without it, stacking and salary are skipped.

## The weekly run in one command

```
ffsim week projections.csv --entries DKEntries.csv --my-lineups my_150.csv --exposure "Player Name=1.0"
```

For one projections file this keeps a dated copy (`data/raw/projections/DATE_vN.csv`), lists whose projection moved since the previous copy, simulates the slate with both simulators, builds a portfolio and DraftKings upload file from each, then simulates fresh slates with a different seed and grades the portfolio (`graded_portfolio_fresh.csv`) and any lineup files of yours on them, so both are judged on slates nothing was picked on. Results go to `out/week/classic/` and `out/week/pbp/`, with a short `summary.txt` and the full output in `run_log.txt`. `--engine classic` or `--engine pbp` runs one simulator only; the portfolio options of `ffsim lineups` (`--max-exposure`, `--exposure`, `--bankroll`, `--portfolio`) apply. Each opponent field is built once per run and reused, so a run with both simulators takes about 40-50 minutes.

## Play-by-play simulator (experimental)

A second simulator plays each game snap by snap, so that game flow (a trailing team passing more, shootouts) and the box score it produces are realistic by construction. Every passing yard is a receiving yard. It is optional; the default simulator is still the one described above.

```
python scripts/fetch_pbp.py 2024 2026                              # public nflverse play-by-play data, into data/raw/pbp/
ffsim pbp-check                                                     # league-average simulated games against real ones
ffsim simulate projections.csv --dk DKEntries.csv --engine pbp      # a slate, played snap by snap
```

The output file is the same as the default simulator's, so `lineups`, `grade` and `field` work on it unchanged.

How it works:

- **Play calling** is a model of pass or run by down, distance, field position, score and time left, plus fourth-down decisions (go, kick or punt) and field-goal accuracy by distance. All are fitted to the data. Each team's own lean toward the pass this season is added.
- **Play results** are drawn from real plays run in a similar situation, so yardage, sacks, turnovers and long plays occur in their real proportions.
- **The clock** runs off real amounts of time, depending on whether the clock stopped and whether the offence is hurrying or protecting a lead. Halves, overtime, kickoffs, punts, penalties, safeties and returns for touchdowns are all played out.
- **Players.** Each target and carry goes to a player. A team's split starts from this season's usage (recent games counting most; targets by depth, carries by field zone) and is adjusted, with the offence's overall quality, until the simulated averages are close to the projections. That is how injuries and role changes in the projections carry through.
- **Scoring** is DraftKings classic, including yardage bonuses and defences.
- **Players leaving games.** A running back, receiver or tight end can leave a game partway through (injury or benching), and his later targets and carries go to teammates (`pbp.exit_rate`: 13%, 11% and 9% of games). Fitted to real games from 2024 to mid-2026: starters projected for 8+ who vanish after halftime (real / simulated: RB 2.6/2.3%, WR 6.9/6.2%, TE 8.0/6.6%) and who score near zero (RB 1.3/0.9%, WR 3.2/3.8%, TE 3.0/3.9%). Without it almost no simulated running back ever had a dud (0.3% near zero).
- **Game plans.** A team's lean toward the pass also varies from one game to the next, beyond what the score and clock explain (`pbp.game_plan_sd` in the settings, in log-odds). The size is fitted so the spread of a team's pass share matches real games; it moves volume between the passing and running games without changing how much scoring swings.
- **Anchoring.** Each player's simulated results are finally scaled so his average equals his projection exactly. The spread of his outcomes, and how players move together, come from the simulated games; the supplied floors and ceilings are not used.

Checks so far:

| Check | Result |
|-------|--------|
| League-average games against real ones (points, plays, pass and rush volume) | Within 2% on each |
| Simulated 10th / 90th percentiles against the provider's floors / ceilings, players projected 10+ | WR 5.5 / 27.4 against 5.4 / 27.9; RB 7.3 / 28.1 against 6.8 / 29.2; QB 9.0 / 26.3 against 8.2 / 26.7 |
| Simulator's own player averages before anchoring | Within 0.3 points of the projections on average |
| Opposing QB's extra points when a QB has a top-2% game (real 6.7) | 4.6, against 3.4 for the default simulator |

`ffsim pbp-check` also replays every real game 30 times at the quality the betting line implied, and compares how much real and simulated games vary around that line (707 games, 2024 to week 10 of 2026):

| Spread (sd) around the line | Real | Simulated, no game plans | Simulated, with game plans |
|-------|------|------|------|
| Team points | 8.92 | 8.79 | 8.90 |
| Game total points | 12.84 | 12.89 | 13.00 |
| Winning margin | 12.36 | 11.98 | 12.18 |
| Share of plays that are dropbacks | 0.096 | 0.084 | 0.096 |
| Dropbacks | 7.90 | 6.87 | 7.46 |
| Rushes | 6.69 | 6.18 | 6.72 |
| Passing yards | 66.6 | 60.2 | 62.7 |

Scoring already swings as much as real games do, so there is no general "good day / bad day" effect on an offence's quality: adding one made teams beat the line by 14+ points half again as often as they really do. What was missing was variation in the play mix, which the game plans supply. A test fails if scoring ever swings more than 5% differently from real games, or if volume swings wider than real.

Links between offenses and the opposing defense were checked with both sides adjusted for the betting line and each player's season level: quarterback with the opposing defense -0.39 simulated against -0.34 real, lead running back -0.29 against -0.21, lead receiver -0.11 against -0.17. An earlier, cruder measure had suggested these were much too strong; they are not.

Known gaps: teammates at the same position move together less than in real games (lead receiver with second receiver -0.07 against +0.04), because receivers compete for targets in the simulator without sharing good days, and exits add to it; passing yardage still varies a little less than in real games (62.7 against 66.6); and every play is independent given the situation.

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
