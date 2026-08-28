"""Configuration loading, precedence and validation.

Every test passes ``env=`` explicitly and ``use_dotenv=False``, so results do
not depend on the developer's machine or on a stray ``.env``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config import DEFAULT_CONFIG_PATH, Settings, apply_env_overrides, load_settings
from src.utils.paths import PROJECT_ROOT


def test_committed_config_is_valid() -> None:
    """configs/config.yaml must load — it is the shipped default."""
    settings = load_settings(use_dotenv=False, env={})
    assert settings.kaggle.dataset == "davidcariboo/player-scores"
    assert settings.data.season_start_month == 8
    assert settings.data.label_tolerance_days == 120


def test_relative_paths_resolve_against_project_root() -> None:
    settings = load_settings(use_dotenv=False, env={})
    assert settings.paths.data_dir == PROJECT_ROOT / "data"
    assert settings.paths.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert settings.paths.processed_dir == PROJECT_ROOT / "data" / "processed"


def test_absolute_path_is_left_alone() -> None:
    """A deployment pointing DATA_DIR at a mounted volume must not be rewritten."""
    settings = load_settings(use_dotenv=False, env={"DATA_DIR": "/mnt/volume"})
    assert settings.paths.data_dir == Path("/mnt/volume")


def test_env_overrides_yaml() -> None:
    """The precedence rule the plan requires: environment beats file."""
    baseline = load_settings(use_dotenv=False, env={})
    assert baseline.api.port == 8000

    overridden = load_settings(use_dotenv=False, env={"API_PORT": "9999"})
    assert overridden.api.port == 9999
    # and it is coerced to the declared type, not left as a string
    assert isinstance(overridden.api.port, int)


def test_empty_env_var_does_not_override() -> None:
    """An unset variable exported as "" is absence, not a request for ""."""
    settings = load_settings(use_dotenv=False, env={"LOG_LEVEL": ""})
    assert settings.logging.level == "INFO"


def test_env_override_of_wrong_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        load_settings(use_dotenv=False, env={"API_PORT": "not-a-port"})


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    """extra="forbid": a mistyped key must fail loudly, not be ignored."""
    tree = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    tree["api"]["prot"] = 8000  # typo for "port"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tree))

    with pytest.raises(ValidationError):
        load_settings(path, use_dotenv=False, env={})


def test_settings_are_immutable() -> None:
    settings = load_settings(use_dotenv=False, env={})
    with pytest.raises(ValidationError):
        settings.api.port = 1234  # type: ignore[misc]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("data", "max_age", 10),  # not greater than min_age (15)
        ("data", "season_start_month", 13),  # not a month
        ("split", "validation_season", 2020),  # not after train_end_season (2021)
        ("split", "test_start_season", 2021),  # not after validation_season (2022)
        ("api", "port", 70000),  # not a port
        ("logging", "level", "CHATTY"),  # not a level
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, section: str, key: str, value: object) -> None:
    tree = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    tree[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tree))

    with pytest.raises(ValidationError):
        load_settings(path, use_dotenv=False, env={})


def test_log_level_is_normalised_to_upper_case() -> None:
    settings = load_settings(use_dotenv=False, env={"LOG_LEVEL": "debug"})
    assert settings.logging.level == "DEBUG"


def test_apply_env_overrides_creates_nested_keys() -> None:
    tree: dict[str, object] = {}
    apply_env_overrides(tree, {"API_PORT": "8080"})
    assert tree == {"api": {"port": "8080"}}


def test_settings_sections_are_all_present() -> None:
    """A new section added to the YAML must also be declared on Settings."""
    tree = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    assert set(tree) == set(Settings.model_fields)
