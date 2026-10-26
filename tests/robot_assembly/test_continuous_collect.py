"""Continuous-collection validation must fail before allocating a GPU or output."""

import pytest

import castle_reasoning.data.assembly.continuous_collect as continuous_collect


@pytest.mark.parametrize(
    "settings",
    [
        {"worlds": 0},
        {"worlds": 1.5},
        {"attempts_per_target": 0},
        {"wall_seconds": 0},
        {"planning_seconds": -1},
        {"planning_workers": 0},
        {"planning_workers": 1.5},
        {"planning_watchdog_seconds": 0},
        {"planning_watchdog_seconds": float("nan")},
        {"physics_timestep": 0.003},
        {"contact_impedance": [0.99, 0.95]},
        {"replay_settle_seconds": -1},
        {"replay_settle_seconds": 1.5},
        {"line_search_threads": 0},
        {"line_search_threads": 64},
        {"line_search_threads": 128.0},
    ],
)
def test_invalid_settings_do_not_create_run_directory(tmp_path, settings):
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        continuous_collect.execute([{}], output, **settings)
    assert not output.exists()
