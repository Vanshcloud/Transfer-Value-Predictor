"""Matplotlib figures, returned as objects rather than written to disk.

Each function takes plain data and returns a ``Figure``. Nothing here reads a
model, a store or a config, and nothing calls ``plt.show`` — so the same
function serves the HTML report, a notebook, and anything Phase 10 wants to
export. ``matplotlib.use("Agg")

# matplotlib emits an INFO line every time an axis is given strings that parse
# as numbers, which season labels ("2023") always do. It is advice, not a
# problem, and it lands once per segment chart.
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)`` is set at import: these run headless, and the
default backend on macOS tries to open a window.
"""

from __future__ import annotations

import base64
import io
import logging

import matplotlib

matplotlib.use("Agg")

# matplotlib emits an INFO line every time an axis is given strings that parse
# as numbers, which season labels ("2023") always do. It is advice, not a
# problem, and it lands once per segment chart.
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

MILLION = 1e6


def _millions(axis: plt.Axes, which: str = "both") -> None:
    """Label money axes in millions. EUR 40000000 is unreadable on a tick."""
    formatter = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / MILLION:,.0f}M")
    if which in {"x", "both"}:
        axis.xaxis.set_major_formatter(formatter)
    if which in {"y", "both"}:
        axis.yaxis.set_major_formatter(formatter)


def predicted_vs_actual(actual: np.ndarray, predicted: np.ndarray) -> Figure:
    """The first plot anyone should look at.

    Log-log, because a linear pair of axes on a target spanning EUR 10k to
    EUR 200M is a dense blob against the origin and one dot in the corner.
    """
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(actual, predicted, s=6, alpha=0.25, edgecolors="none")

    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axis.plot(limits, limits, linestyle="--", linewidth=1, color="black", label="perfect")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("actual market value (EUR)")
    axis.set_ylabel("predicted market value (EUR)")
    axis.set_title("Predicted vs actual")
    axis.legend()
    figure.tight_layout()
    return figure


def residual_distribution(residuals: np.ndarray) -> Figure:
    """Signed residuals in EUR. Centred means unbiased; skewed means not."""
    figure, axis = plt.subplots(figsize=(7, 4))
    # Clipped for display only: a handful of EUR 100M misses would otherwise
    # set the x-range and compress everything worth seeing into one bar.
    limit = float(np.percentile(np.abs(residuals), 99))
    axis.hist(np.clip(residuals, -limit, limit), bins=60)
    axis.axvline(0, linestyle="--", linewidth=1, color="black")

    axis.set_xlabel("residual (predicted − actual, EUR, clipped at p99)")
    axis.set_ylabel("rows")
    axis.set_title("Residual distribution")
    _millions(axis, "x")
    figure.tight_layout()
    return figure


def residuals_vs_actual(actual: np.ndarray, residuals: np.ndarray) -> Figure:
    """Shows whether error grows with value. On this data it does."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(actual, residuals, s=6, alpha=0.25, edgecolors="none")
    axis.axhline(0, linestyle="--", linewidth=1, color="black")

    axis.set_xscale("log")
    axis.set_xlabel("actual market value (EUR, log scale)")
    axis.set_ylabel("residual (EUR)")
    axis.set_title("Residual against actual value")
    _millions(axis, "y")
    figure.tight_layout()
    return figure


def feature_importance(importances: dict[str, float], top_n: int = 20) -> Figure:
    """Horizontal bars, largest at the top, which is how people read a ranking."""
    ranked = sorted(importances.items(), key=lambda kv: -abs(kv[1]))[:top_n]
    names = [name for name, _ in ranked][::-1]
    values = [value for _, value in ranked][::-1]

    figure, axis = plt.subplots(figsize=(8, max(4, 0.32 * len(names))))
    axis.barh(names, values)
    axis.set_xlabel("importance")
    axis.set_title(f"Top {len(names)} features")
    figure.tight_layout()
    return figure


def shap_summary(mean_abs_shap: dict[str, float], top_n: int = 20) -> Figure:
    """Mean absolute SHAP per feature, in log space.

    Deliberately not a beeswarm: a bar chart of mean magnitude is the thing
    that survives being read quickly, and the beeswarm's per-row detail is
    already available per player through the explanation API.
    """
    ranked = sorted(mean_abs_shap.items(), key=lambda kv: -kv[1])[:top_n]
    names = [name for name, _ in ranked][::-1]
    values = [value for _, value in ranked][::-1]

    figure, axis = plt.subplots(figsize=(8, max(4, 0.32 * len(names))))
    axis.barh(names, values, color="#4c72b0")
    axis.set_xlabel("mean |SHAP| (log-EUR space)")
    axis.set_title(f"Feature impact on predictions — top {len(names)}")
    figure.tight_layout()
    return figure


def contribution_waterfall(
    features: list[str], shap_values: list[float], base_eur: float, prediction_eur: float
) -> Figure:
    """One prediction, explained. Red pushes the value up, blue pushes it down."""
    order = np.argsort(np.abs(shap_values))
    names = [features[i] for i in order]
    values = [shap_values[i] for i in order]
    colours = ["#c44e52" if v > 0 else "#4c72b0" for v in values]

    figure, axis = plt.subplots(figsize=(8, max(3.5, 0.34 * len(names))))
    axis.barh(names, values, color=colours)
    axis.axvline(0, linewidth=1, color="black")
    axis.set_xlabel("contribution (log-EUR); positive raises the prediction")
    axis.set_title(f"EUR {base_eur:,.0f} baseline  →  EUR {prediction_eur:,.0f} predicted")
    figure.tight_layout()
    return figure


def error_by_segment(segments: pd.DataFrame, title: str) -> Figure:
    """MAE per slice, with the row count on each bar.

    The count is on the bar because a segment MAE without its n invites
    someone to quote a number computed over thirty rows.
    """
    figure, axis = plt.subplots(figsize=(8, max(3, 0.42 * len(segments))))
    bars = axis.barh(segments["value"].astype(str), segments["mae_eur"])

    for bar, count in zip(bars, segments["n"], strict=True):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  n={count:,}",
            va="center",
            fontsize=8,
        )

    axis.set_xlabel("MAE (EUR)")
    axis.set_title(title)
    _millions(axis, "x")
    figure.tight_layout()
    return figure


def model_comparison(frame: pd.DataFrame, metric: str, title: str) -> Figure:
    """Families ranked on one metric."""
    ordered = frame.sort_values(metric, ascending=False)
    figure, axis = plt.subplots(figsize=(8, max(3, 0.4 * len(ordered))))
    axis.barh(ordered["model"], ordered[metric])
    axis.set_xlabel(metric)
    axis.set_title(title)
    figure.tight_layout()
    return figure


def to_data_uri(figure: Figure) -> str:
    """Encode a figure as a base64 PNG data URI, and close it.

    Data URIs rather than files beside the HTML: a report that is one file
    survives being emailed, attached or opened from a different directory,
    which is the whole point of generating it. Closing matters — matplotlib
    keeps every unclosed figure alive and a report builds dozens.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
