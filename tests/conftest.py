"""Shared fixtures and the ``--run-slow`` switch for long training runs."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

import tensorgrad as tg

settings.register_profile(
    "default",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("thorough", max_examples=200, deadline=None)
settings.load_profile("default")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="also run tests marked slow"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow training run; enable with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _seed_global_rng() -> None:
    """Make every test independent of execution order."""
    tg.manual_seed(0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
