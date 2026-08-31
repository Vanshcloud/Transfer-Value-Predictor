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


_ASSUMPTIONS: tuple[str, ...] = (
    "**The future resembles the recent past.** Training seasons are weighted "
    "toward recent ones, but a structural break — a new financial-fair-play "
    "regime, another closed-doors season — is not something the model can "
    "anticipate. Season 2020 is already such a break in this data.",
    "**A player's competition is a proxy for the quality he faced.** The model "
    "never sees an opponent. It sees the league's historical value level and "
    "his club's results, which is an average standing in for a specific.",
    "**Minutes are earned, not assigned.** Availability is read as a signal of "
    "quality. A player kept out by injury and one kept out by his manager look "
    "the same here.",
    "**The valuation being predicted is set within a year of the as-of date.** "
    "Widen that horizon and the question changes; the label window is a "
    "modelling choice, recorded per row as `label_horizon_days`.",
)

_FAILURE_MODES: tuple[str, ...] = (
    "**Players with no history.** A first covered season has no lagged "
    "features and the model falls back on biography and current output. The "
    "`career_stage` breakdown in the error report isolates exactly these rows.",
    "**Sudden reputational moves.** A transfer saga, a tournament breakout or "
    "a long-term injury repriced the player faster than any season-level "
    "feature can register. The model is smooth; the market is not.",
    "**The very top of the market.** Above EUR 50M there are few comparable "
    "seasons and the absolute error is largest. Relative error is smallest "
    "there, which is a different statement and easy to confuse.",
    "**Leagues thinly represented in the panel.** Competition strength is an "
    "expanding historical mean, so a league with little history gets a null "
    "and the imputer's median instead of a level.",
    "**Anything outside the covered competitions.** No row exists, so no "
    "prediction is served rather than a guess being manufactured.",
)

_ETHICS: tuple[str, ...] = (
    "**This is decision support, not a decision.** Every response carries an "
    "explanation and an interval so a human can disagree with it. A valuation "
    "used to set a person's wages or transfer terms without that human is a "
    "misuse of the model.",
    "**The labels encode a community's opinion of people.** Transfermarkt "
    "valuations are crowd estimates, and any bias in that crowd — toward "
    "visible leagues, particular nationalities, particular styles — is "
    "reproduced faithfully. The model cannot correct a bias it is trained to "
    "imitate.",
    "**Nationality is a feature.** `country_of_citizenship` improves accuracy "
    "and is a protected attribute. It is retained because removing it does not "
    "remove the information (league and club are proxies) and does hide it. "
    "Stated plainly so the choice is reviewable rather than invisible.",
    "**No personal data beyond the public record.** Biography, appearances and "
    "public valuations only. Nothing here is scraped, and nothing about a "
    "player's private life is used or inferred.",
)


def _fairness_section(analysis: ErrorAnalysis | None) -> list[str]:
    """Measured error spread across leagues, rather than a promise of fairness.

    A fairness claim with no numbers behind it is decoration. What can honestly
    be shown is where the model is worse and by how much, so the segments are
    printed and the reader draws the conclusion.
    """
    lines = [
        "## Fairness",
        "",
        "The model is not audited against protected attributes as a classifier "
        "would be — there is no favourable outcome to allocate, only an "
        "estimate that is more or less accurate. What matters here is whether "
        "it is *reliably* accurate across groups, so the measured spread is "
        "reported instead of a fairness score.",
        "",
    ]
    if analysis is None:
        lines += ["No per-segment analysis was supplied for this card.", ""]
        return lines

    segments = sorted(analysis.segments_for("primary_competition_id"), key=lambda s: -s.mape)
    if not segments:
        lines += ["No competition-level segments met the minimum row count.", ""]
        return lines

    lines += [
        "Worst and best competitions by MAPE, over segments with enough rows " "to mean anything:",
        "",
        "| Competition | Rows | MAE | MAPE |",
        "|---|---|---|---|",
    ]
    for segment in segments[:3] + segments[-3:]:
        lines.append(
            f"| `{segment.value}` | {segment.n:,} | EUR {segment.mae:,.0f} | {segment.mape:.0%} |"
        )
    lines += [
        "",
        "A wide spread here means the model serves some leagues better than "
        "others, which is a property of how much history the panel holds for "
        "each. Read it before quoting one number for every competition.",
        "",
    ]
    return lines


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
        "deliberate, and it is why the reported error is roughly 45% worse "
        "than a split grouped by player would suggest. The reported number is "
        "the honest one.",
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
        "## Assumptions",
        "",
        *[f"- {item}" for item in _ASSUMPTIONS],
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in _limitations(artifact, analysis)],
        "",
        "## Failure modes",
        "",
        *[f"- {item}" for item in _FAILURE_MODES],
        "",
        *_fairness_section(analysis),
        "## Ethical considerations",
        "",
        *[f"- {item}" for item in _ETHICS],
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
