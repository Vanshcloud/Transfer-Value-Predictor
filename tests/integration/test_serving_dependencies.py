"""The serving image must be able to load the artifacts that are shipped.

The API image installs `requirements-serve.txt`, which omits xgboost and
catboost — together roughly 700 MB, including 291 MB of CUDA libraries a CPU
inference path never opens. That is only safe while every shipped variant's
winning family is one the serve set installs.

It is a real coupling and it is invisible: a tuning run that happens to select
CatBoost produces an artifact that trains fine, scores fine, saves fine — and
then fails to unpickle inside the container, at startup, in production. This
test is where that gets caught instead.

Marked ``integration``: it reads the artifacts on disk, so it skips on a clean
checkout like every other test that needs a trained model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.utils.config import load_settings
from src.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.integration

SERVE = PROJECT_ROOT / "requirements-serve.txt"

# The pip package each model family needs in order to be *unpickled*, not
# merely fitted: a saved artifact is a scikit-learn Pipeline wrapping the
# estimator, so the estimator's package must be importable to load it.
PACKAGE_FOR_FAMILY = {
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "random_forest": "scikit-learn",
    "gradient_boosting": "scikit-learn",
    "linear": "scikit-learn",
    "ridge": "scikit-learn",
    "lasso": "scikit-learn",
    "elastic_net": "scikit-learn",
}


def serve_packages() -> set[str]:
    names = set()
    for line in SERVE.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)", line)
        if match:
            names.add(match.group(1).lower())
    return names


def sidecars() -> list[Path]:
    return sorted(load_settings().paths.model_dir.glob("*.json"))


@pytest.fixture(scope="module")
def shipped() -> list[Path]:
    found = sidecars()
    if not found:
        pytest.skip("no trained artifacts; run scripts/train_models.py")
    return found


def test_the_serve_set_can_load_every_shipped_artifact(shipped: list[Path]) -> None:
    installed = serve_packages()
    for path in shipped:
        family = json.loads(path.read_text())["model_name"]
        package = PACKAGE_FOR_FAMILY.get(family)
        assert package, f"{family} is not in PACKAGE_FOR_FAMILY; add it"
        assert package in installed, (
            f"{path.name} is a {family} artifact, which needs {package!r}, but "
            f"requirements-serve.txt does not install it. Either add it there "
            f"and re-run `uv pip compile requirements-serve.txt "
            f"--python-version 3.13 --output-file requirements-serve-lock.txt`, "
            f"or ship a family the serving image can load."
        )


def test_every_family_the_zoo_can_pick_is_accounted_for() -> None:
    """A new family added to the registry must declare what loading it needs."""
    from src.models.registry import MODEL_REGISTRY

    unknown = sorted(set(MODEL_REGISTRY) - set(PACKAGE_FOR_FAMILY))
    assert not unknown, f"add these to PACKAGE_FOR_FAMILY: {unknown}"


def test_the_artifacts_actually_load_in_this_process(shipped: list[Path]) -> None:
    """The sidecar says which family it is; this proves the joblib agrees."""
    from src.models.artifact import load

    for path in shipped:
        artifact = load(path.with_suffix(".joblib"))
        assert artifact.model_name == json.loads(path.read_text())["model_name"]
