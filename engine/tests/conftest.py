import warnings

import pytest

from yieldengine.calibration import fit_calibrated
from yieldengine.models import model_factory
from yieldengine.simulator import FakePlant, default_schema


@pytest.fixture(scope="session")
def schema():
    return default_schema()


@pytest.fixture(scope="session")
def plant():
    return FakePlant(seed=0)


@pytest.fixture(scope="session")
def history(plant):
    return plant.history()


@pytest.fixture(scope="session")
def factory(schema):
    return model_factory(schema, "gp", n_restarts=0)


@pytest.fixture(scope="session")
def fitted(factory, history, schema):
    return fit_calibrated(factory, history, schema)


@pytest.fixture(scope="session")
def context(plant, history):
    return plant.context_now(history)
