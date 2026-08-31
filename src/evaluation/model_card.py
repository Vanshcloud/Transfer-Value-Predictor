"""A model card: what this model is for, and where it should not be trusted.

Generated from the artifact rather than written by hand, so it cannot describe
a model that no longer exists. The limitations section is the point of the
document — a card that only lists metrics is a scoreboard, and the questions
that matter when someone is about to act on a prediction are what the model
was not shown and who it is worst at.
"""

from __future__ import annotations

from src.evaluation.error_analysis import ErrorAnalysis
from src.models.artifact import ModelArtifact

INTENDED_USE = {
    "performance_only": (
        "Estimate a player's market value from on-pitch performance and "
        "biography alone, with no prior valuation supplied. This is the "
        "scouting and undervaluation model: it is the one that can express an "
        "opinion that differs from the market, because it has never been told "
        "what the market thinks."
    ),
    "with_prior_value": (
        "Forecast how a player's *already known* market value will move. This "
        "is the tracking model. It is substantially more accurate and "
        "substantially less interesting: most of its skill comes from the "
        "prior valuation, so it cannot tell you the market is wrong."
    ),
}

OUT_OF_SCOPE = (
    "Setting or negotiating an actual transfer fee. Market value and transfer "
    "fee are different quantities; this model is trained on the former.",
    "Players outside the covered competitions, or below the appearance volume "
    "the training data contains.",
    "Any individual decision about a person's employment or compensation "
    "without a human reviewing the explanation alongside the number.",
)


def _limitations(artifact: ModelArtifact, analysis: ErrorAnalysis | None) -> list[str]:
    limitations = [
        "**Market value is an estimate, not a fact.** The labels come from "
        "Transfermarkt's community-maintained valuations. The model reproduces "
        "that consensus, including wherever it is biased — it does not "
        "independently observe what anyone would pay.",
        "**Error grows with value.** Errors are reported in EUR, and the "
        "target spans four orders of magnitude, so a mid-table MAE conceals "
        "much larger absolute misses at the top of the market. Read the "
        "per-band breakdown before trusting a number for an expensive player.",
        "**Coverage begins in 2012.** Appearance data starts 2012-07-03, so "
        "career-length features are left-censored and capped; a player whose "
        "career began earlier looks younger in career terms than they are.",
        "**Seasons are August to July.** Leagues on a spring-autumn calendar "
        "are split across that boundary and are represented less faithfully.",
        "**The model has never seen the season it is asked about.** That is "
        "deliberate, and it is why the reported error is roughly 60% worse "
        "than a random split would suggest. The reported number is the "
        "honest one.",
    ]

    if artifact.variant == "with_prior_value":
        limitations.append(
            "**This variant is anchored to the previous valuation.** It will "
            "track the market rather than challenge it, and it cannot produce "
            "a prediction at all for a player with no valuation history."
        )

    if analysis is not None:
        worst = sorted(analysis.segments_for("value_band"), key=lambda s: -s.mape)[:1]
        if worst:
            band = worst[0]
            limitations.append(
                f"**Weakest measured segment: value band {band.value}** — "
                f"MAPE {band.mape:.0%} over {band.n:,} rows."
            )
    return limitations


def build_model_card(
    artifact: ModelArtifact,
    *,
    analysis: ErrorAnalysis | None = None,
    top_features: int = 10,
) -> str:
    """Render a model card as Markdown."""
    importance = sorted(artifact.feature_importance.items(), key=lambda kv: -abs(kv[1]))[
        :top_features
    ]

    lines = [
        f"# Model card — {artifact.variant}",
        "",
        f"Generated {artifact.created_at} from `{artifact.variant}__{artifact.model_name}`.",
        "This file is written from the artifact, so it cannot describe a model that",
        "is no longer the one on disk.",
        "",
        "## What it does",
        "",
        INTENDED_USE.get(artifact.variant, "Predict player market value in EUR."),
        "",
        "## Not what it is for",
        "",
        *[f"- {item}" for item in OUT_OF_SCOPE],
        "",
        "## Model",
        "",
        f"| Family | `{artifact.model_name}` |",
        "|---|---|",
        f"| Hyperparameters | `{artifact.params or 'defaults'}` |",
        f"| Target | `{artifact.target_column}`, trained on `log1p`, reported in EUR |",
        f"| Features | {len(artifact.feature_columns)} |",
        f"| Seed | {artifact.seed} |",
        "",
        "## Data and split",
        "",
        f"- Rows: {artifact.dataset.get('rows', 'unknown'):,}",
        f"- Split: {artifact.split.get('strategy')} — train ≤"
        f"{artifact.split.get('train_end_season')}, "
        f"validation {artifact.split.get('validation_season')}, "
        f"test ≥{artifact.split.get('test_start_season')}",
        "- Source: Kaggle `davidcariboo/player-scores` (CC0). Transfermarkt is",
        "  never scraped; its terms prohibit both the method and this purpose.",
        "",
        "## Measured performance",
        "",
        "Test seasons, in EUR. These are held-out seasons the model never saw.",
        "",
        "| Metric | Validation | Test |",
        "|---|---|---|",
        f"| MAE | €{artifact.validation.mae:,.0f} | €{artifact.test.mae:,.0f} |",
        f"| RMSE | €{artifact.validation.rmse:,.0f} | €{artifact.test.rmse:,.0f} |",
        f"| R² | {artifact.validation.r2:.3f} | {artifact.test.r2:.3f} |",
        f"| MAPE | {artifact.validation.mape:.1%} | {artifact.test.mape:.1%} |",
        f"| Rows | {artifact.validation.n:,} | {artifact.test.n:,} |",
        "",
        *_interval_section(artifact),
        "## What it relies on",
        "",
        "| Feature | Importance |",
        "|---|---|",
        *[f"| `{name}` | {value:,.4g} |" for name, value in importance],
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in _limitations(artifact, analysis)],
        "",
        "## Leakage controls",
        "",
        "- Every feature is observed at or before the label date, enforced by",
        "  construction and asserted on all rows.",
        "- Current-state columns (`contract_expiration_date`, the",
        "  `current_club_*` family) never enter the feature matrix.",
        "- The prior valuation, where used, is explicitly lagged and named so.",
        "- Splits are re-checked for row and player overlap after every split.",
        "- Club form is joined as of the row's own date, so a match played after",
        "  the label was set cannot reach the features through an aggregate.",
        "",
    ]
    return "\n".join(lines)


def _interval_section(artifact: ModelArtifact) -> list[str]:
    """The prediction interval's nominal level next to the level it achieved.

    Both, or neither. A nominal 80% printed on its own is the claim; the
    measured figure is the evidence, and it is lower — the quantiles come from
    the validation season and are checked a season or more later, across a
    market that moves. Printing only the nominal number is how an interval
    comes to be trusted more than it has earned.
    """
    coverage = artifact.calibration.get("coverage") or {}
    if not coverage:
        return []
    return [
        "## Prediction intervals",
        "",
        f"| Nominal level | {coverage['level']:.0%} |",
        "|---|---|",
        f"| Measured coverage | {coverage['empirical']:.1%} |",
        f"| Median width | €{coverage['median_width_eur']:,.0f} |",
        f"| Measured on | {coverage['n']:,} test rows |",
        "",
        "Quantiles are taken from the validation season and the coverage above",
        "is measured on the test seasons, which neither the model nor the",
        "interval has seen. Measured coverage below the nominal level is the",
        "cost of predicting forward across seasons that are not exchangeable.",
        "",
    ]
