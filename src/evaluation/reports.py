"""Self-contained HTML reports, built from the structures the product uses.

Every figure is embedded as a base64 PNG, so a report is one file that can be
emailed or opened from anywhere. No CDN, no sidecar images, no JavaScript —
the failure mode of a report that needs a network is that it is blank in the
meeting where it matters.

These renderers consume :mod:`src.evaluation.error_analysis` and
:mod:`src.explainability.shap_explainer` rather than recomputing anything, so
the numbers in a report and the numbers an API returns come from one place.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.evaluation.error_analysis import ErrorAnalysis
from src.evaluation.metrics import Metrics
from src.explainability.shap_explainer import GlobalExplanation, PredictionExplanation
from src.models.artifact import ModelArtifact
from src.utils.logging import get_logger
from src.visualization import plots

logger = get_logger(__name__)

_STYLE = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.5rem; line-height: 1.55; }
h1 { margin-bottom: .25rem; } h2 { margin-top: 2.5rem; }
.sub { color: #666; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #8883; }
th { font-weight: 600; } td.num, th.num { text-align: right; }
img { max-width: 100%; height: auto; display: block; margin: 1rem 0; }
.note { border-left: 3px solid #8886; padding: .1rem 0 .1rem 1rem; color: #666; margin: 1.2rem 0; }
.cards { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.25rem 0; }
.card { border: 1px solid #8883; border-radius: 8px; padding: .8rem 1.1rem; min-width: 8.5rem; }
.card .label { font-size: .75rem; color: #666; text-transform: uppercase; letter-spacing: .04em; }
.card .value { font-size: 1.35rem; font-variant-numeric: tabular-nums; }
code { background: #8881; padding: .1rem .3rem; border-radius: 3px; }
"""


@dataclass(frozen=True)
class Report:
    """One rendered page."""

    name: str
    title: str
    body: str

    def to_html(self) -> str:
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(self.title)}</title><style>{_STYLE}</style></head>"
            f"<body>{self.body}</body></html>"
        )

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.html"
        path.write_text(self.to_html(), encoding="utf-8")
        logger.info("wrote %s (%.0f KB)", path.name, path.stat().st_size / 1024)
        return path


def _escape(value: object) -> str:
    return html.escape(str(value))


def _table(frame: pd.DataFrame, numeric_columns: set[str] | None = None) -> str:
    numeric = numeric_columns or set()
    head = "".join(
        f"<th class='{'num' if c in numeric else ''}'>{_escape(c)}</th>" for c in frame.columns
    )
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td class='{'num' if c in numeric else ''}'>{_escape(row[c])}</td>"
            for c in frame.columns
        )
        + "</tr>"
        for _, row in frame.iterrows()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _cards(pairs: list[tuple[str, str]]) -> str:
    cards = "".join(
        f"<div class='card'><div class='label'>{_escape(label)}</div>"
        f"<div class='value'>{_escape(value)}</div></div>"
        for label, value in pairs
    )
    return f"<div class='cards'>{cards}</div>"


def _metric_cards(metrics: Metrics) -> str:
    return _cards(
        [
            ("MAE", f"€{metrics.mae / 1e6:,.2f}M"),
            ("RMSE", f"€{metrics.rmse / 1e6:,.2f}M"),
            ("R²", f"{metrics.r2:.3f}"),
            ("MAPE", f"{metrics.mape:.1%}"),
            ("rows", f"{metrics.n:,}"),
        ]
    )


def _header(artifact: ModelArtifact, title: str, subtitle: str) -> str:
    return (
        f"<h1>{_escape(title)}</h1><p class='sub'>{_escape(subtitle)}</p>"
        f"<p class='sub'><code>{_escape(artifact.variant)}</code> · "
        f"<code>{_escape(artifact.model_name)}</code> · generated "
        f"{_escape(artifact.created_at[:19])}</p>"
    )


def _image(uri: str, alt: str) -> str:
    return f"<img src='{uri}' alt='{_escape(alt)}'>"


TEMPORAL_NOTE = (
    "<div class='note'>Every figure on this page is measured on the "
    "<strong>test seasons</strong> — seasons the model never saw during "
    "training or selection. A random split would report roughly 30% better "
    "EUR error and would not describe deployment.</div>"
)


def baseline_report(comparison: pd.DataFrame, artifact: ModelArtifact) -> Report:
    """Why this family was chosen, not just that it was."""
    body = [
        _header(artifact, "Model comparison", "Every family, on the same split and seed"),
        TEMPORAL_NOTE,
        "<h2>Leaderboard</h2>",
        "<p>Ranked by validation MAE in EUR, which is the selection metric. "
        "Cost columns are shown because accuracy alone does not explain a "
        "choice: a family that ties on error while taking twenty times as long "
        "to fit lost for a reason the metric column cannot show.</p>",
        _table(comparison, numeric_columns=set(comparison.columns) - {"model", "params"}),
        "<h2>Validation MAE by family</h2>",
        _image(
            plots.to_data_uri(
                plots.model_comparison(
                    comparison, "val MAE (EUR)", "Validation MAE — lower is better"
                )
            ),
            "validation MAE by family",
        ),
    ]
    return Report("baseline_report", "Model comparison", "".join(body))


def evaluation_report(artifact: ModelArtifact, analysis: ErrorAnalysis) -> Report:
    """Headline metrics and the two plots that show whether they can be trusted."""
    residuals = analysis.residuals
    body = [
        _header(artifact, "Evaluation", "Held-out test seasons, all figures in EUR"),
        TEMPORAL_NOTE,
        "<h2>Test metrics</h2>",
        _metric_cards(analysis.overall),
        "<h2>Predicted against actual</h2>",
        "<p>Log axes on both sides: the target spans €10k to €200M, and on "
        "linear axes this is one dense blob and a single distant dot.</p>",
        _image(
            plots.to_data_uri(
                plots.predicted_vs_actual(
                    residuals[artifact.target_column].to_numpy(dtype=float),
                    residuals["predicted"].to_numpy(dtype=float),
                )
            ),
            "predicted versus actual",
        ),
        "<h2>Residuals</h2>",
        "<p>Signed <em>predicted minus actual</em>, so a positive residual "
        "means the model overvalued the player.</p>",
        _image(
            plots.to_data_uri(
                plots.residual_distribution(residuals["residual"].to_numpy(dtype=float))
            ),
            "residual distribution",
        ),
        _image(
            plots.to_data_uri(
                plots.residuals_vs_actual(
                    residuals[artifact.target_column].to_numpy(dtype=float),
                    residuals["residual"].to_numpy(dtype=float),
                )
            ),
            "residuals against actual value",
        ),
    ]
    return Report("evaluation", "Evaluation", "".join(body))


def feature_importance_report(artifact: ModelArtifact) -> Report:
    """The model's own importances, which are not the same thing as SHAP."""
    ranked = sorted(artifact.feature_importance.items(), key=lambda kv: -abs(kv[1]))
    frame = pd.DataFrame(
        [{"feature": name, "importance": round(value, 4)} for name, value in ranked[:30]]
    )
    body = [
        _header(artifact, "Feature importance", "What the fitted model leans on"),
        "<div class='note'>These are the estimator's internal importances — "
        "split counts for a tree model, coefficients for a linear one. They "
        "say what the model <em>used</em>. The SHAP report says how much each "
        "feature actually <em>moved predictions</em>, which is the more useful "
        "question and does not always give the same order.</div>",
        _image(
            plots.to_data_uri(plots.feature_importance(artifact.feature_importance)),
            "feature importance",
        ),
        "<h2>Full ranking</h2>",
        _table(frame, numeric_columns={"importance"}),
    ]
    return Report("feature_importance", "Feature importance", "".join(body))


def shap_report(
    artifact: ModelArtifact,
    explanation: GlobalExplanation,
    examples: list[tuple[str, PredictionExplanation]],
) -> Report:
    """Global impact, then worked per-player examples."""
    ranked = explanation.ranked(30)
    frame = pd.DataFrame(
        [
            {
                "feature": name,
                "mean |SHAP|": round(value, 4),
                "mean SHAP": round(explanation.mean_shap[name], 4),
            }
            for name, value in ranked
        ]
    )

    body = [
        _header(artifact, "SHAP explanations", f"Sampled over {explanation.sample_size:,} rows"),
        "<div class='note'>SHAP values are additive in <strong>log space</strong>, "
        "because the model is trained on <code>log1p(EUR)</code>. They are "
        "<em>not</em> additive in euros — the same contribution is worth a "
        "different number of euros for a €500k player and a €90M one. The "
        "exact reading is multiplicative: a contribution of 0.34 multiplies "
        "the prediction by e<sup>0.34</sup> ≈ 1.41, whatever the player is "
        "worth.</div>",
        "<h2>Global feature impact</h2>",
        _image(plots.to_data_uri(plots.shap_summary(explanation.mean_abs_shap)), "SHAP summary"),
        _table(frame, numeric_columns={"mean |SHAP|", "mean SHAP"}),
    ]

    if examples:
        body.append("<h2>Worked examples</h2>")
        body.append(
            "<p>Individual predictions, decomposed. Red bars raised the "
            "prediction, blue lowered it. This is the same structure "
            "<code>POST /predict</code> returns.</p>"
        )
    for label, example in examples:
        top = example.top(12)
        body.append(f"<h3>{_escape(label)}</h3>")
        body.append(
            _cards(
                [
                    ("predicted", f"€{example.prediction_eur / 1e6:,.2f}M"),
                    ("baseline", f"€{example.base_value_eur / 1e6:,.2f}M"),
                ]
            )
        )
        body.append(
            _image(
                plots.to_data_uri(
                    plots.contribution_waterfall(
                        [c.feature for c in top],
                        [c.shap_value for c in top],
                        example.base_value_eur,
                        example.prediction_eur,
                    )
                ),
                f"contributions for {label}",
            )
        )

    return Report("shap_summary", "SHAP explanations", "".join(body))


def error_analysis_report(artifact: ModelArtifact, analysis: ErrorAnalysis) -> Report:
    """Where the model is wrong, and on whom."""
    segments = pd.DataFrame([s.as_dict() for s in analysis.segments])
    body = [
        _header(artifact, "Error analysis", "Where the error is concentrated"),
        TEMPORAL_NOTE,
        "<h2>Overall</h2>",
        _metric_cards(analysis.overall),
    ]

    for column, title in (
        ("value_band", "Error by market-value band"),
        ("age_band", "Error by age band"),
        ("position", "Error by position"),
        ("season", "Error by test season"),
    ):
        rows = segments[segments["segment"] == column] if not segments.empty else segments
        if rows.empty:
            continue
        body.append(f"<h2>{_escape(title)}</h2>")
        body.append(_image(plots.to_data_uri(plots.error_by_segment(rows, title)), title))
        display = rows[["value", "n", "mae_eur", "median_residual_eur", "mape"]].copy()
        display["mae_eur"] = display["mae_eur"].map(lambda v: f"€{v:,.0f}")
        display["median_residual_eur"] = display["median_residual_eur"].map(lambda v: f"€{v:,.0f}")
        display["mape"] = display["mape"].map(lambda v: f"{v:.1%}")
        body.append(
            _table(display, numeric_columns={"n", "mae_eur", "median_residual_eur", "mape"})
        )

    body.append("<h2>Largest misses</h2>")
    body.append(
        "<p>Signed, so these are genuinely the two opposite failure modes "
        "rather than one list sorted by magnitude.</p>"
    )
    for title, frame in (
        ("Most overvalued by the model", analysis.worst_overpredictions),
        ("Most undervalued by the model", analysis.worst_underpredictions),
    ):
        columns = [
            c
            for c in (
                "player_id",
                "season",
                "position",
                "age",
                artifact.target_column,
                "predicted",
                "residual",
            )
            if c in frame.columns
        ]
        display = frame[columns].copy()
        for money in (artifact.target_column, "predicted", "residual"):
            if money in display:
                display[money] = display[money].map(lambda v: f"€{v:,.0f}")
        if "age" in display:
            display["age"] = display["age"].map(lambda v: f"{v:.1f}")
        body.append(f"<h3>{_escape(title)}</h3>")
        body.append(_table(display, numeric_columns=set(columns) - {"position"}))

    return Report("error_analysis", "Error analysis", "".join(body))
