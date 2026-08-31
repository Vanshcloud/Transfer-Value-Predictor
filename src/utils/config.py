"""Typed configuration: YAML defaults, overlaid by environment variables.

Precedence, lowest to highest: ``configs/config.yaml`` -> ``.env`` -> real
environment variables. Secrets live only in the environment, never in the YAML.

Pydantic v2 only. The v1 ``validator`` decorator, ``.dict()`` and the nested
``Config`` class are v1 idioms
that still run but emit deprecation warnings; the test suite runs with
``-W error::DeprecationWarning`` so they cannot creep back in.

``pydantic-settings`` would also solve this. It is deliberately not a
dependency: the overlay this project needs is an explicit map of six variables,
which is less code than configuring the library to do the same thing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.paths import PROJECT_ROOT, resolve

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

# Environment variable -> dotted path into the config tree. Explicit rather than
# inferred: a typo in a variable name should be visible here, not silently
# ignored at runtime.
ENV_OVERRIDES: dict[str, str] = {
    "DATA_DIR": "paths.data_dir",
    "MODEL_DIR": "paths.model_dir",
    "KAGGLE_DATASET": "kaggle.dataset",
    "API_HOST": "api.host",
    "API_PORT": "api.port",
    "LOG_LEVEL": "logging.level",
}

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class _Strict(BaseModel):
    """Base for every config section.

    ``extra="forbid"`` turns a mistyped YAML key into an error instead of a
    value that is silently ignored — the failure mode where a setting appears
    to be configured and is not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    """Where data and models live. Relative entries resolve against the root."""

    data_dir: Path = Path("data")
    model_dir: Path = Path("models")

    @field_validator("data_dir", "model_dir")
    @classmethod
    def _resolve_against_root(cls, value: Path) -> Path:
        return resolve(value)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample"


class KaggleConfig(_Strict):
    """Source dataset. Credentials come from the environment, never from here."""

    dataset: str
    files: tuple[str, ...]
    max_age_days: int = Field(default=7, ge=0)


class DataConfig(_Strict):
    """Season construction and label-join parameters."""

    season_start_month: int = Field(default=8, ge=1, le=12)
    label_tolerance_days: int = Field(default=365, gt=0)
    default_horizon_days: int = Field(default=120, gt=0)
    min_age: int = Field(default=15, ge=0)
    max_age: int = Field(default=45, ge=0)
    """The plausible age interval. Mirrored by
    :data:`~src.feature_engineering.build.PLAUSIBLE_RANGES`, which is what the
    prediction service actually enforces — the service is constructed without a
    config in tests, in the CLI and in a notebook, and a guard that exists only
    when a config was loaded is absent exactly when someone is experimenting.
    `tests/unit/test_config.py` asserts the two do not drift apart."""

    @field_validator("default_horizon_days")
    @classmethod
    def _horizon_within_tolerance(cls, value: int, info: Any) -> int:
        """A default horizon outside the labelled range would ask the model a
        question no training row answers."""
        tolerance = info.data.get("label_tolerance_days")
        if tolerance is not None and value > tolerance:
            raise ValueError(
                f"default_horizon_days ({value}) exceeds label_tolerance_days ({tolerance})"
            )
        return value

    @field_validator("max_age")
    @classmethod
    def _max_exceeds_min(cls, value: int, info: Any) -> int:
        minimum = info.data.get("min_age")
        if minimum is not None and value <= minimum:
            raise ValueError(f"max_age ({value}) must exceed min_age ({minimum})")
        return value


class SplitConfig(_Strict):
    """Three-way temporal split boundaries, by season."""

    train_end_season: int
    validation_season: int
    test_start_season: int

    @field_validator("validation_season")
    @classmethod
    def _validation_follows_train(cls, value: int, info: Any) -> int:
        train_end = info.data.get("train_end_season")
        if train_end is not None and value <= train_end:
            raise ValueError(
                f"validation_season ({value}) must follow train_end_season ({train_end})"
            )
        return value

    @field_validator("test_start_season")
    @classmethod
    def _test_follows_validation(cls, value: int, info: Any) -> int:
        validation = info.data.get("validation_season")
        if validation is not None and value <= validation:
            raise ValueError(
                f"test_start_season ({value}) must follow validation_season ({validation})"
            )
        return value


class HttpConfig(_Strict):
    """Outbound request policy, shared by every ingestion module."""

    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=5, ge=0)
    backoff_factor: float = Field(default=0.5, ge=0)
    user_agent: str = "transfer-value-predictor/0.1"
    min_request_interval_seconds: float = Field(default=0.0, ge=0)


class ApiConfig(_Strict):
    """Bind address for the FastAPI service."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, lt=65536)


class LoggingConfig(_Strict):
    """Logging verbosity and line format."""

    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    @field_validator("level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"unknown log level {value!r}; expected one of {sorted(_VALID_LOG_LEVELS)}"
            )
        return upper


class Settings(_Strict):
    """The fully resolved configuration for one process."""

    paths: PathsConfig
    kaggle: KaggleConfig
    data: DataConfig
    split: SplitConfig
    http: HttpConfig
    api: ApiConfig
    logging: LoggingConfig


def _assign(tree: dict[str, Any], dotted: str, value: str) -> None:
    """Set ``dotted`` (e.g. ``"api.port"``) to ``value`` inside ``tree``."""
    head, _, tail = dotted.partition(".")
    if not tail:
        tree[head] = value
        return
    branch = tree.setdefault(head, {})
    if not isinstance(branch, dict):  # pragma: no cover - malformed YAML
        raise TypeError(f"cannot descend into {head!r}: not a mapping")
    _assign(branch, tail, value)


def apply_env_overrides(tree: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    """Overlay environment variables onto a parsed config tree.

    Values arrive as strings; Pydantic coerces them to the declared field type,
    so ``API_PORT=9000`` becomes an int and a non-numeric value raises rather
    than silently producing a string port.
    """
    source = os.environ if env is None else env
    for variable, dotted in ENV_OVERRIDES.items():
        raw = source.get(variable)
        if raw is not None and raw != "":
            _assign(tree, dotted, raw)
    return tree


def load_settings(
    config_path: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    use_dotenv: bool = True,
) -> Settings:
    """Load, overlay and validate configuration.

    Args:
        config_path: YAML defaults. Falls back to ``configs/config.yaml``.
        env: Environment mapping to overlay. Defaults to the real environment;
            passing one explicitly keeps tests independent of the machine.
        use_dotenv: Load a ``.env`` file first. Disabled in tests so a
            developer's local ``.env`` cannot change the result.

    Returns:
        A validated, immutable :class:`Settings`.
    """
    if use_dotenv and env is None:
        # override=False: a real environment variable outranks the file, which
        # is what a deployment expects when it injects configuration.
        load_dotenv(PROJECT_ROOT / ".env", override=False)

    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        tree = yaml.safe_load(handle) or {}

    return Settings.model_validate(apply_env_overrides(tree, env))
