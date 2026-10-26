"""Vectorized resets preserve parent physics and collector bookkeeping."""

import os
import time

import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import castle_reasoning.data.assembly.continuous_collect as continuous_collect
import castle_reasoning.data.assembly.publication as publication
import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.control.active as active
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.execution.preparation as preparation
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as simulation_models
import castle_reasoning.runtime.simulation.world_slots as world_slots

pytestmark = pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="GPU opt-in"
)


def reference_reset(controller, mask, qpos, qvel):
    """Parent 0bb807c transaction, including its intermediate forward pass.

    Full arrays combine original unselected state and selected requested state;
    assigning these is equivalent to the parent's masked copy kernel.
    """
    engine = controller.engine
    engine.reset(mask)
    for source, destination in ((qpos, engine.data.qpos), (qvel, engine.data.qvel)):
        values = destination.numpy()
        values[mask] = source[mask]
        destination.assign(values)
    actions = controller.action.numpy()
    actions[mask] = [0, 0, 0, 0, 0, 0, -1]
    controller.action.assign(actions)
    mjw.forward(engine.model, engine.data)
    wp.synchronize_device(engine.device)
    controller.sync_targets(mask)
    engine.assert_healthy()


def state(controller):
    engine = controller.engine
    return (
        engine.snapshot()
        | {
            key: getattr(engine.data, key).numpy()
            for key in ("time", "qacc_warmstart", "qacc", "xpos", "xquat", "act")
        }
        | {
            "target": controller.arm_target.numpy(),
            "action": controller.action.numpy(),
            "active": controller.active_worlds.numpy(),
            "forces": engine.model.actuator_forcerange.numpy(),
        }
    )


@pytest.mark.parametrize("mask", [[False, True, True], [True, True, True]])
def test_batched_reset_matches_parent_and_continuation(mask, monkeypatch):
    mask = np.array(mask)
    goals = np.array([[0.425, 0, 0.01]])
    model, home = simulation_models.make_model([[0.02] * 3], goals)
    controllers = []
    actions = np.zeros((3, 7), np.float32)
    actions[:, 5], actions[:, 6] = 0.005, -1
    for _ in range(2):
        collector = teacher.make_collector(model, goals, 3, 1, 17)
        pool = world_slots.ModelPool([model], [0, 0, 0])
        collector.engine.model = pool.model
        collector.engine.native_models = pool.native_models
        controller = collector.controller = active.ActivePathControl(collector)
        collector.reset(qpos=np.tile(home, (3, 1)), qvel=np.zeros((3, model.nv)))
        controller.step(actions, graph=False)
        controllers.append(controller)
    original = state(controllers[0])
    qpos, qvel = original["qpos"].copy(), original["qvel"].copy()
    qpos[mask, 5] += 0.03
    qvel[mask, 5] = 0.025
    reference_reset(controllers[0], mask, qpos, qvel)
    # The compact upload must never read filler rows for continuing worlds.
    qpos[~mask], qvel[~mask] = np.nan, np.nan
    forwards = []
    forward = mjw.forward

    def counted_forward(*args, **kwargs):
        forwards.append(1)
        return forward(*args, **kwargs)

    monkeypatch.setattr(mjw, "forward", counted_forward)
    world_slots.reset(controllers[1], mask, qpos=qpos, qvel=qvel)
    expected, actual = map(state, controllers)
    for key in actual:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-6, atol=1e-7, err_msg=key
        )

    for key in (
        "qpos",
        "qvel",
        "ctrl",
        "qacc_warmstart",
        "target",
        "action",
        "active",
        "forces",
        "episode",
        "random_u32",
        "time",
    ):
        np.testing.assert_array_equal(
            actual[key][~mask], original[key][~mask], err_msg=key
        )
    assert len(forwards) == 2
    world_slots.reset(controllers[1], np.zeros(3, bool), qpos=qpos, qvel=qvel)
    assert len(forwards) == 2
    # The resulting derived state must also produce the same physical motion.
    for _ in range(3):
        for controller in controllers:
            controller.step(actions, graph=False)
    expected, actual = map(state, controllers)
    for key in actual:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-6, atol=1e-7, err_msg=key
        )


@pytest.mark.parametrize("producer", [False, True])
def test_collection_batches_resets_while_a_segment_remains_pending(
    tmp_path, monkeypatch, producer
):
    goals = np.array([[0.425, 0, 0.01]])
    sizes = [[[width, 0.02, 0.02]] for width in (0.02, 0.022)]
    models, homes = zip(
        *(simulation_models.make_model(size, goals) for size in sizes), strict=True
    )
    pool = world_slots.ModelPool(models, [0, 0, 0])
    collector = teacher.make_collector(models[0], goals, 3, 1, 0)
    collector.engine.model, collector.engine.native_models = (
        pool.model,
        pool.native_models,
    )
    collector.controller = active.ActivePathControl(collector)
    collector.reset(qpos=np.tile(homes[0], (3, 1)), qvel=np.zeros((3, models[0].nv)))
    plans = [
        dict(sizes=size, target=goals.tolist(), target_id=f"T{i}")
        for i, size in enumerate(sizes)
    ]
    settings = dict(
        wall_seconds=30,
        attempts_per_target=1,
        seed=0,
        planning_seconds=1,
        planning_workers=1,
        planning_watchdog_seconds=120,
        physics_timestep=0.002,
        pause_planning_worlds=True,
    )
    model_ids = None
    if producer:
        (tmp_path / "models").mkdir()
        model_ids = [simulation_models.save_model(model, tmp_path / "models" / f"{i}.mjb")
                     for i, model in enumerate(models)]
        settings.update(preparation_mode="producer", preparation_workers=2,
                        skip_paused_preparation=True)
    with publication.SummaryPublisher(tmp_path / "summary.json") as writer:
        group = continuous_collect.Collection(
            collector,
            pool,
            plans,
            homes,
            tmp_path,
            settings,
            time.perf_counter(),
            writer,
            model_ids=model_ids,
        )
        publications, snapshots = [], []
        submit, snapshot = writer.submit, collector.engine.snapshot

        def record_summary(summary):
            publications.append(summary)
            return submit(summary)

        def record_snapshot():
            snapshots.append(1)
            return snapshot()

        monkeypatch.setattr(writer, "submit", record_summary)
        monkeypatch.setattr(collector.engine, "snapshot", record_snapshot)
        requests = {
            2: (0, 1, "settle", homes[1], np.zeros(models[1].nv)),
            0: (1, 0, "settle", homes[0], np.zeros(models[0].nv)),
        }
        returned = group.reset_slots(requests)
        assert len(publications) == len(snapshots) == 1
        assert group.reset_count == 2 and group.reset_batch_count == 1
        assert set(returned) == {0, 2}
        assert pool.assignments == [0, 0, 1]
        for world, index in ((0, 0), (2, 1)):
            np.testing.assert_array_equal(
                returned[world]["qpos"][0], homes[index].astype(np.float32)
            )
            assert group.runners[world].planner.scene.model is models[index]
        runners = group.runners.copy()
        if producer:
            # Slot one continues the model installed by the fixture's initial reset.
            group.preparation_sources[1] = preparation.Source(1, 0, 0, 1, model_ids[0])
        motions = []
        for world in range(3):
            home = homes[pool.assignments[world]]
            path = np.tile(home, (2, 1))
            path[1, 5] += 0.6 if world == 1 else 0
            motions.append(
                batch_tracking.Request(
                    manipulation.Segment("transit", path, np.zeros(2)),
                    collision.Scene(pool.native_models[world], home),
                    -1,
                    guard=False,
                )
            )
        if producer:
            motions = [preparation.prepare(r, group.snapshot["qpos"][w:w+1],
                                            group.preparation_sources[w])
                       for w, r in enumerate(motions)]
            epoch_before = group.preparation_sources.copy()
        result = group.advance(motions)
        assert result[0] is not None and result[1] is None and result[2] is not None
        continuing = group.segments.pending[1]
        offset = continuing.offset
        forces = collector.engine.model.actuator_forcerange.numpy()
        forces[:, 6] = [-0.5, 0.5]
        collector.engine.model.actuator_forcerange.assign(forces)
        before = state(collector.controller)
        reset_requests = {
            w: (a, m, "disassembly", q, v) for w, (a, m, _, q, v) in requests.items()
        }
        publications.clear()
        snapshots.clear()
        group.reset_slots(reset_requests)
        assert group.reset_count == 4 and group.reset_batch_count == 2
        if producer:
            assert group.preparation_sources[1] == epoch_before[1]
            assert all(group.preparation_sources[w].slot_epoch == epoch_before[w].slot_epoch + 1
                       for w in (0, 2))
        assert len(publications) == len(snapshots) == 1
        assert group.runners == runners
        assert group.segments.pending[1] is continuing and continuing.offset == offset
        after = state(collector.controller)
        for key in (
            "qpos",
            "qvel",
            "ctrl",
            "qacc_warmstart",
            "target",
            "action",
            "forces",
            "episode",
            "random_u32",
            "time",
        ):
            np.testing.assert_array_equal(after[key][1], before[key][1], err_msg=key)
        np.testing.assert_array_equal(after["forces"][0], models[0].actuator_forcerange)
        np.testing.assert_array_equal(after["forces"][2], models[1].actuator_forcerange)
        # Reject the entire transaction before even the valid sibling is reset.
        with pytest.raises(RuntimeError, match="unfinished segment"):
            group.reset_slots({0: requests[0], 1: requests[2]})
        np.testing.assert_array_equal(
            collector.engine.episode.numpy(), after["episode"]
        )
        assert group.reset_slots({}) == {}
        assert group.reset_count == 4 and len(publications) == 1
        if producer:
            with pytest.raises(ValueError, match="source"):
                group.advance(motions)
