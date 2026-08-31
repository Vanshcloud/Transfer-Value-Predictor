# Dataset card — the training table

The model cards describe two models. This describes the thing both are fitted
to, because a model card that documents a model and not its data documents
half a system.

Written by hand rather than generated, unlike the model cards, because most of
what matters here is *why* a column is absent — and an absent column cannot
generate its own explanation. Every count is reproducible with
`python scripts/build_features.py`, which prints them.

## Provenance

| | |
|---|---|
| Source | Kaggle `davidcariboo/player-scores` |
| Licence | CC0 1.0 (public domain dedication) |
| Upstream | A mirror of Transfermarkt, refreshed weekly |
| Retrieved by | `scripts/fetch_data.py`, re-downloading when older than 7 days |
| Files used | 9 of 10 |

**Transfermarkt is never scraped.** Its Terms of Use §11.1 prohibit both
automated access and machine-learning training on the content. The Kaggle
mirror is CC0 and is the only route this project uses. This is a legal
constraint, not a technical preference, and it is why there is no scraper in
this repository and no plan to add one.

## Shape

| | |
|---|---|
| Grain | one row per (`player_id`, `season`) |
| Rows | 85,966 |
| Players | 24,411 |
| Seasons | 2011–2024 |
| Features | 54 (+2 more in the prior-value variant) |
| Target | `market_value_in_eur` |
| Rows with a usable prior valuation | 61,522 (71.6%) |
| Unlabelled current-season rows | 8,709 (season 2025, prediction only) |

The two shipped variants are a row filter and a column list over this one
table, never a second build — see `select_variant`. Two pipelines are how
training and serving drift apart.

## How a row is constructed

1. Per-match appearances are stamped with a season (August–July, named for the
   August) and aggregated per player-season.
2. The row's **as-of date** is `max(1 July season boundary, last appearance)`.
   This is the moment the feature window closes.
3. The **label** is the first valuation recorded at or after the as-of date,
   within 365 days — a forward `merge_asof`, never an equality merge, because
   valuations are an irregular on-change event series rather than a schedule.
4. Context and performance features are joined. Club form joins **as of the
   row's own date**, not as a whole-season mean.
5. The prior valuation and the career-momentum block are lagged from the
   player's own earlier rows.

Every step is ordered so that no feature can be observed after its label. This
is enforced by `LeakageValidator` as a pipeline stage, not only by tests: a leak
stops a run rather than quietly improving a metric.

## What is deliberately not in it

**`clubs.csv` — the whole file.** Its interesting columns
(`total_market_value`, `squad_size`, `average_age`, `national_team_players`)
are *current* state. Joining today's squad value to a 2013 row states a fact
from 2026. Club strength comes from `club_games.csv` instead, which is dated
results.

**`contract_expiration_date`, `market_value_in_eur` and the `current_club_*`
family from `players.csv`.** Same reason, and `market_value_in_eur` in that
table is today's answer sitting one join away from the label. Banned by name in
`src/validation/leakage.CURRENT_STATE_COLUMNS`.

**`international_caps` and `international_goals`.** Career totals as of the last
refresh — current state wearing a historical-sounding name.

**`season` as a feature.** It is the split key. A model that learns "later
season → higher value" extrapolates a trend straight off the end of its
training range the moment it meets the test years.

**`club_home_attendance`.** Built, measured, rejected. Season 2020 was played
behind closed doors: mean attendance falls 14,769 → 2,752, null rate rises to
21.6%, and correlation with log value drops from +0.65 to +0.24. Adding it
costs €297,445 of pooled held-out MAE (p = 1e−65). For one season in fourteen
the column measures a pandemic rather than a club. **Any future feature drawn
from `games.csv` must be checked against season 2020 before it is believed.**

**`game_events.csv`.** 157 MB and 1.27M rows of goals, cards and substitutions
with minute stamps. Its goal and card counts are already in `appearances.csv`
at the granularity the model uses. Left unbuilt as a decision, recorded so it
is not mistaken for an oversight.

## Known biases and limitations

**Coverage begins in 2012.** `appearances.csv` has no row before 2012-07-03 and
`game_lineups.csv` none before 2013-07-02. Valuations reach back to 2000, so
40,339 label-bearing rows for 2003–2011 *can* be reconstructed from biography
and a prior valuation — they were, and only **7 of the 54 declared features
exist on such a row**: age, age², height, position, sub-position, foot and
citizenship. Adding all of them moved held-out MAE by 0.19% (p = 0.73). The
experiment ran against the 41-feature list of the time, where 34 of 41 were
missing; the career-momentum block makes such a row emptier, not fuller.
Career-length features are left-censored and capped at 10 seasons for the same
reason.

**The label is an appraisal, not a price.** Transfermarkt market values are a
community consensus. The model reproduces that consensus, including wherever it
is biased — by league visibility, by nationality, by whatever the community
over- and under-rates. `transfers.csv` carries real fees and
`--target transfer_fee` will train on them, but a fee exists only where a sale
happened: 8.5% coverage against the market value's 83%, and conditioned on the
outcome. It answers what a sold player cost, not what a player is worth.

**Seasons are August–July, and no single boundary is right for every league.**
The declared and derived season labels disagree on 6.27% of fixtures —
Brazilian Série A, the J1 League, Eliteserien, Allsvenskan, the K-League, MLS,
and every European qualifying round played in July. 19.6% of player-seasons
span the boundary, though only 2.02% of minutes are misfiled. An alternative
table indexed on the declared season was built and measured *worse*
(`plans/06-final-research-audit.md` §5).

**Competition coverage is Europe-weighted**, inherited from Transfermarkt's
own. A player outside the covered competitions has no row, and a player who
moves into them arrives with no history — which `seasons_observed` and the
`career_stage` error segment both surface rather than hide.

**Error grows with value in euros and shrinks in relative terms.** The target
spans four orders of magnitude. A single MAE describes neither end; read the
per-band breakdown in `reports/error_analysis.html`.

**Season 2020 is a structural break**, not an outlier. Closed-doors football
moved attendance, and plausibly more than attendance.

## Splits

Chronological, three ways, and the test seasons are touched once.

| | seasons | rows |
|---|---|---|
| Train | ≤ 2021 | 65,907 |
| Validation | 2022 | 6,573 |
| Test | ≥ 2023 | 13,486 |

Random and group-by-player splits exist in `src/models/splits.py` as
diagnostics only. The random split reads about 35% better and answers a
question nobody deploying this will ever ask.

## Audit status

Re-verified from the raw CSVs — not from the pipeline's own checker — in
`plans/06-final-research-audit.md`. Eleven properties checked; one real defect
found and fixed (club form built from matches played after the label, 1,049
rows), two flags cleared as artefacts of the checker.

Current state: 0 duplicate player-seasons, 0 rows where any date value
postdates the label, 0 rows where club form postdates the as-of date, prior
valuations traced to a real raw valuation on 61,522 of 61,522 rows.
