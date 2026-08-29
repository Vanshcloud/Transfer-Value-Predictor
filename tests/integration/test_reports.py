"""Report generation against the real models.

Marked ``integration``: needs both the training table and trained artifacts.
Skips otherwise, so CI on a clean checkout stays green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipelines.features import TRAINING_TABLE
from src.pipelines.report import build_reports
from src.storage.duckdb_store import DuckDBParquetStore
from src.utils.config import load_settings
from src.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.integration

EXPECTED_PAGES = {
    "baseline_report",
    "evaluation",
    "feature_importance",
    "shap_summary",
    "error_analysis",
}


@pytest.fixture(scope="module")
def bundles(tmp_path_factory: pytest.TempPathFactory) -> list[object]:
    store = DuckDBParquetStore(PROJECT_ROOT / "data" / "processed")
    models = PROJECT_ROOT / "models"
    if not store.has_table(TRAINING_TABLE) or not list(models.glob("*.joblib")):
        pytest.skip("run scripts/build_features.py and scripts/train_models.py first")

    output = tmp_path_factory.mktemp("out")
    return build_reports(
        store,
        load_settings().split,
        models,
        output / "reports",
        output / "docs",
    )


def _written(bundles: list[object]) -> list[Path]:
    return [path for bundle in bundles for path in bundle.written]  # type: ignore[attr-defined]


def test_every_expected_page_is_produced(bundles: list[object]) -> None:
    names = {path.stem.split("__")[0] for path in _written(bundles) if path.suffix == ".html"}
    assert names >= EXPECTED_PAGES


def test_both_variants_get_their_own_pages(bundles: list[object]) -> None:
    """Both variants render a page called "evaluation"; without a suffix the
    second would silently overwrite the first."""
    pages = [p.name for p in _written(bundles) if p.suffix == ".html"]
    assert "evaluation.html" in pages
    assert "evaluation__with_prior_value.html" in pages


def test_every_page_is_self_contained(bundles: list[object]) -> None:
    for path in _written(bundles):
        if path.suffix != ".html":
            continue
        content = path.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content, path.name
        # No CDN, no sidecar image, nothing that needs a network to render.
        assert "http://" not in content and "https://" not in content, path.name


def test_a_model_card_is_written_per_variant(bundles: list[object]) -> None:
    cards = [p for p in _written(bundles) if p.name.startswith("MODEL_CARD")]
    assert len(cards) == 2
    for card in cards:
        text = card.read_text(encoding="utf-8")
        assert "## Limitations" in text
        assert "Leakage controls" in text


def test_the_comparison_document_covers_every_family(bundles: list[object]) -> None:
    from src.models.registry import MODEL_REGISTRY

    document = next(p for p in _written(bundles) if p.name == "model_comparison.md")
    text = document.read_text(encoding="utf-8")

    from src.evaluation.comparison import DISPLAY_NAMES

    for name in MODEL_REGISTRY:
        assert DISPLAY_NAMES[name] in text, name


def test_the_comparison_reports_cost_alongside_accuracy(bundles: list[object]) -> None:
    document = next(p for p in _written(bundles) if p.name == "model_comparison.md")
    text = document.read_text(encoding="utf-8")
    assert "train (s)" in text
    assert "size (MB)" in text
    assert "predict (ms/1k)" in text


def test_shap_ran_on_the_real_model(bundles: list[object]) -> None:
    for bundle in bundles:
        assert bundle.explanation is not None  # type: ignore[attr-defined]
        assert bundle.explanation.sample_size > 0  # type: ignore[attr-defined]


def test_the_error_analysis_covers_the_whole_test_set(bundles: list[object]) -> None:
    for bundle in bundles:
        analysis = bundle.analysis  # type: ignore[attr-defined]
        assert analysis.overall.n > 1_000
        assert analysis.segments


def test_error_grows_with_market_value(bundles: list[object]) -> None:
    """A documented limitation, asserted rather than assumed.

    The model card claims absolute error concentrates at the top of the
    market. If that ever stops being true the card is wrong and should change.
    """
    analysis = bundles[0].analysis  # type: ignore[attr-defined]
    bands = {s.value: s.mae for s in analysis.segments_for("value_band")}
    assert bands["<1M"] < bands.get(">50M", float("inf"))
