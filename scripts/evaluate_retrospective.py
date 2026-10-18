#!/usr/bin/env python3
"""Rolling-origin backtest and paired comparisons for the frozen configuration.

    python scripts/evaluate_retrospective.py

Everything here is **retrospective**: the configuration it evaluates was
designed while the 2023+ seasons were being inspected (docs/EVALUATION.md),
so no number it prints is independent evidence. What it adds is structure
the headline lacks — a figure per season rather than one pooled number,
intervals that resample players rather than rows, and a comparison of the two
variants on the rows they share. It chooses nothing.

Writes reports/evaluation/retrospective.json and the per-row predictions it
was computed from (reports/evaluation/backtest_predictions.parquet, with its
SHA-256 in the JSON) so the analysis can be re-run without refitting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.inference import cluster_mean_interval, paired_difference  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.feature_engineering.build import (  # noqa: E402
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    select_variant,
)
from src.models.artifact import load_variant, predict  # noqa: E402
from src.models.provenance import sha256_file  # noqa: E402
from src.models.registry import MODEL_REGISTRY, build_pipeline  # noqa: E402
from src.models.weighting import fit_params  # noqa: E402
from src.pipelines.features import TRAINING_TABLE  # noqa: E402
from src.pipelines.train import VARIANTS  # noqa: E402
from src.pipelines.tune import _numeric_features  # noqa: E402
from src.storage.duckdb_store import DuckDBParquetStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

FIRST_ORIGIN = 2016
"""The earliest season scored: four seasons of history before it."""

PRIOR_AGE_BANDS = ((0, 60), (60, 180), (180, 365), (365, 100_000))


def backtest(table: pd.DataFrame, settings: object) -> pd.DataFrame:
    """Fit the frozen family and parameters on every season before *s*, score *s*."""
    model_dir = settings.paths.model_dir  # type: ignore[attr-defined]
    rows = []
    for variant, include_prior in VARIANTS.items():
        artifact = load_variant(model_dir, variant)
        frame, columns = select_variant(table, include_prior_value=include_prior)
        features = list(columns)
        for season in range(FIRST_ORIGIN, int(frame["season"].max()) + 1):
            train = frame[frame["season"] < season]
            scored = frame[frame["season"] == season]
            pipeline = build_pipeline(
                MODEL_REGISTRY[artifact.model_name],
                _numeric_features(columns),
                CATEGORICAL_FEATURES,
            )
            pipeline.set_params(**artifact.params)
            pipeline.fit(train[features], train[TARGET_COLUMN], **fit_params(train["season"]))
            rows.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "origin": season,
                        "player_id": scored["player_id"].to_numpy(),
                        "season": season,
                        "actual": scored[TARGET_COLUMN].to_numpy(float),
                        "predicted": pipeline.predict(scored[features]),
                    }
                )
            )
            print(f"  backtest {variant} {season}: {len(scored):,} rows", flush=True)
    return pd.concat(rows, ignore_index=True)


def mae_interval(group: pd.DataFrame) -> dict[str, object]:
    errors = np.abs(group["predicted"] - group["actual"]).to_numpy()
    interval = cluster_mean_interval(errors, group["player_id"].to_numpy())
    return {**interval.as_dict(), "r2": evaluate(group["actual"], group["predicted"]).r2}


def main() -> int:
    settings = load_settings()
    store = DuckDBParquetStore(settings.paths.processed_dir)
    table = store.read_table(TRAINING_TABLE)
    test_start = settings.split.test_start_season
    out = PROJECT_ROOT / "reports" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    predictions = backtest(table, settings)
    predictions_path = out / "backtest_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    result: dict[str, object] = {
        "status": "retrospective — not independent evidence; see docs/EVALUATION.md",
        "training_table_sha256": sha256_file(store.path_for(TRAINING_TABLE)),
        "backtest_predictions_sha256": sha256_file(predictions_path),
        "backtest": {},
        "shipped_test": {},
    }

    for variant, group in predictions.groupby("variant"):
        per_season = {str(season): mae_interval(rows) for season, rows in group.groupby("season")}
        pooled = {
            "pre_test": mae_interval(group[group["season"] < test_start]),
            "test_seasons": mae_interval(group[group["season"] >= test_start]),
        }
        result["backtest"][str(variant)] = {"per_season": per_season, "pooled": pooled}  # type: ignore[index]

    # The shipped models on the test seasons, with player-clustered intervals.
    shipped: dict[str, pd.DataFrame] = {}
    for variant, include_prior in VARIANTS.items():
        artifact = load_variant(settings.paths.model_dir, variant)
        frame, _ = select_variant(table, include_prior_value=include_prior)
        test = frame[frame["season"] >= test_start]
        shipped[variant] = pd.DataFrame(
            {
                "player_id": test["player_id"].to_numpy(),
                "season": test["season"].to_numpy(),
                "actual": test[TARGET_COLUMN].to_numpy(float),
                "predicted": predict(artifact, test),
                "prior_age_days": test.get(
                    "prev_value_age_days", pd.Series(np.nan, index=test.index)
                ).to_numpy(),
            }
        )
        result["shipped_test"][variant] = mae_interval(shipped[variant])  # type: ignore[index]

    # Paired: the same player-seasons, both models. Unpaired figures above.
    keys = ["player_id", "season"]
    both = shipped["performance_only"].merge(
        shipped["with_prior_value"], on=keys, suffixes=("_a", "_b")
    )
    difference = paired_difference(
        both["predicted_a"] - both["actual_a"],
        both["predicted_b"] - both["actual_b"],
        both["player_id"].to_numpy(),
    )
    result["paired_variants_on_shared_test_rows"] = {
        "a": "performance_only",
        "b": "with_prior_value",
        "meaning": "mean |error_a| - mean |error_b| in EUR; positive means performance_only errs more",
        **difference.as_dict(),
        "relative_to_a": difference.estimate
        / float(np.abs(both["predicted_a"] - both["actual_a"]).mean()),
    }

    # A newer prior or a better algorithm? Same family both variants; the
    # prior-value error by how old the prior was.
    prior = shipped["with_prior_value"]
    bands = {}
    for low, high in PRIOR_AGE_BANDS:
        rows = prior[(prior["prior_age_days"] >= low) & (prior["prior_age_days"] < high)]
        if len(rows) >= 50:
            bands[f"{low}-{high if high < 100_000 else 'inf'} days"] = mae_interval(rows)
    result["with_prior_value_error_by_prior_age"] = bands

    path = out / "retrospective.json"
    path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
