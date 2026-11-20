import os

import pytest
import yaml

import scripts.final_il_evaluation as uncond_script
import scripts.run_conditioned_il_training as cond_script

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StopAfterLoader(Exception):
    pass


@pytest.mark.parametrize("module, setup_name", [
    (uncond_script, "setup_pure_gaussian_environment"),
    (cond_script, "setup_simulation_environment"),
])
def test_il_scripts_read_the_frozen_snapshot(monkeypatch, module, setup_name):
    # Training must use the same frozen returns as the evaluation, not a live download.
    with open(os.path.join(ROOT, "config.yaml")) as f:
        config = yaml.safe_load(f)
    seen = {}

    class FakeLoader:
        def __init__(self, tickers, start_date, end_date, cache_path=None):
            seen["cache_path"] = cache_path

        def fetch_data(self):
            raise StopAfterLoader

    monkeypatch.setattr(module, "MarketDataLoader", FakeLoader)
    with pytest.raises(StopAfterLoader):
        getattr(module, setup_name)(config)
    assert seen["cache_path"] == os.path.join(ROOT, config["data"]["snapshot_path"])
