# Feasibility spike — measured, not assumed (2026-08-28)

Source: Kaggle `davidcariboo/player-scores` (CC0), files player_valuations.csv,
players.csv, appearances.csv. All numbers below produced locally on py3.12/pandas 3.0.5.

## Data shape
- player_valuations: 656,301 rows / 41,528 players / 2000-01-20 -> 2026-06-12
- appearances:       1,894,350 rows / 29,531 players
- v1 joined table:   37,025 rows / 16,995 players / seasons 2011-2024
  (season = Aug-Jul; label = first valuation after season end, 120d tolerance)
- with prior-season value: 20,030 rows

## Target
- EUR: median 500k, p95 10M, max 200M, min 10k, zero nulls/zeros
- skew raw 8.70 -> log1p 0.43   => TRAIN ON log1p(value). Non-negotiable.

## Null rates in v1 table
age 0.0% | position 0.0% | minutes 0.0% | height 1.2%
contract_expiration_date 39.5%  <-- and it is CURRENT contract, not per-season.
    Using it for a 2015 row is leakage from the future. Drop or re-derive.

## Split strategy — measured
Baseline feats (age, apps, goals, assists, minutes, cards, height, per90):
  RANDOM   R2 0.465  MAE EUR 3,277,047
  GROUP    R2 0.455  MAE EUR 3,243,155
  TEMPORAL R2 0.412  MAE EUR 5,140,804
+ prior-season market value:
  RANDOM   R2 0.826  MAE EUR 2,393,800
  GROUP    R2 0.830  MAE EUR 2,370,351
  TEMPORAL R2 0.809  MAE EUR 3,958,350

Conclusion: random-vs-group leakage is NOT the big risk here (one row per
player-season already). The real gap is TEMPORAL: ~60% worse EUR MAE.
Report temporal numbers as headline; anything else flatters the model.

prev_mv dominates (R2 0.45 -> 0.83). Ship BOTH variants and say so:
  - "performance-only"  = the interesting model (scouting/undervalued detection)
  - "with prior value"  = the accurate model (tracking/forecasting)
Publishing only the second is technically true and practically useless.
