"""Full-collector validation and stage-state invariants before GPU qualification."""

import numpy as np
import pytest

import castle_reasoning.data.assembly.batched_collect as batched_collect
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.execution.session as session
import castle_reasoning.runtime.simulation.models as simulation_models


def test_saved_model_hash_uses_identical_serialized_bytes_without_reread(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path

    import mujoco

    model, _ = simulation_models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    reference = tmp_path / "reference.mjb"
    candidate = tmp_path / "candidate.mjb"
    mujoco.mj_saveModel(model, str(reference), None)
    expected = reference.read_bytes()
    read = Path.read_bytes

    def forbidden_read(*args, **kwargs):
        raise AssertionError("Model saving must not reread the file")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    digest = simulation_models.save_model(model, candidate)
    assert digest == hashlib.sha256(expected).hexdigest()
    assert read(candidate) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"worlds": 0},
        {"worlds": 1.5},
        {"planning_seconds": 0},
        {"wall_seconds": float("inf")},
        {"physics_timestep": 0.003},
    ],
)
def test_invalid_batch_configuration_creates_no_attempt_artifacts(tmp_path, kwargs):
    path = tmp_path / "attempts"
    with pytest.raises(ValueError):
        batched_collect.execute({}, path, "invalid", **kwargs)
    assert not path.exists()


@pytest.mark.parametrize(
    "actions,efforts",
    [
        (np.zeros((2, 7)), np.ones(2)),
        (np.zeros((2, 2, 7)), np.ones(2)),
        (np.zeros((2, 2, 7)), np.zeros((2, 2))),
        (np.full((2, 2, 7), np.nan), np.ones((2, 2))),
    ],
)
def test_bad_replay_shape_or_nonfinite_commands_fail_before_device_access(
    actions, efforts
):
    with pytest.raises(ValueError):
        batch_tracking.replay(None, actions, efforts)


def test_world_view_copies_state_and_rejects_in_stage_reset():
    from types import SimpleNamespace


    model, home = simulation_models.make_model([[0.02] * 3], [[0.425, 0, 0.01]])
    group = SimpleNamespace(
        native_models=[model, model],
        snapshot={"qpos": np.tile(home, (2, 1)), "qvel": np.zeros((2, model.nv))},
    )
    view = session.WorldCollector(group, 1)
    snapshot = view.engine.snapshot()
    view.reset(qpos=snapshot["qpos"][0], qvel=snapshot["qvel"][0])
    snapshot["qpos"][0, 0] += 0.1
    np.testing.assert_array_equal(group.snapshot["qpos"][1], home)
    with pytest.raises(ValueError, match="reset inside"):
        view.reset(qpos=snapshot["qpos"][0], qvel=snapshot["qvel"][0])


def test_world_view_selects_its_actual_model():
    from types import SimpleNamespace


    models = [
        simulation_models.make_model([size], [[0.425, 0, size[2] / 2]])[0]
        for size in ([0.02] * 3, [0.03, 0.04, 0.02])
    ]
    group = SimpleNamespace(native_models=models)
    assert session.WorldState(group, 1).native_model is models[1]
