import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runplot", action="store_true", default=False, help="run plotting tests"
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_plot = pytest.mark.skip(reason="need --runplot option to run")
    for item in items:
        if not config.getoption("--runslow") and "slow" in item.keywords:
            item.add_marker(skip_slow)
        if not config.getoption("--runplot") and "plot" in item.keywords:
            item.add_marker(skip_plot)
