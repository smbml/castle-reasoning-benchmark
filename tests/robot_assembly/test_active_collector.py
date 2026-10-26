"""Real construction simulator tests for paused slots and reference continuity."""

import os

import mujoco
import numpy as np
import pytest

import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.control.active as active
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.execution.batching as batching
import castle_reasoning.runtime.execution.segments as segments
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as models
import castle_reasoning.runtime.simulation.world_slots as world_slots

pytestmark = pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="Explicit GPU opt-in"
)


@pytest.mark.parametrize("guard_failure", [False, True])
@pytest.mark.parametrize("skip_paused", [False, True])
def test_pause_reset_resume_preserves_state_and_only_records_active_actions(
    guard_failure, skip_paused,
):
    goals = np.array([[0.425, 0, 0.01]])
    model, home = models.make_model([[0.02] * 3], goals)
    model.opt.timestep = 0.002
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.iterations = 1000
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    collector = teacher.make_collector(model, goals, 2, 1, 0)
    collector.engine.model.block_dim.linesearch_iterative = 128
    controller = collector.controller = active.ActivePathControl(collector)
    collector.reset(qpos=np.tile(home, (2, 1)), qvel=np.zeros((2, model.nv)))
    scene = collision.Scene(model, home)

    def request(angle, name="transit"):
        path = np.tile(home, (2, 1))
        path[1, 5] += angle
        return batch_tracking.Request(
            manipulation.Segment(name, path, np.zeros(2)), scene, -1, guard=False
        )

    short, long = request(0), request(0.6)
    if guard_failure:
        import dataclasses

        wrong = home.copy()
        wrong[14] += 0.02
        short = dataclasses.replace(short, guard=True, scene_reference=wrong)
    expected = batch_tracking.prepare([long], home[None])
    executor = segments.SegmentExecutor(
        controller, pause_planning_worlds=True, skip_paused_preparation=skip_paused
    )
    first = executor.advance([short, long])
    assert first[0] is not None and first[1] is None
    assert first[0][1]["failure"] == (2 if guard_failure else 0)
    assert executor.pending[1].offset == first[0][1]["executed_control_steps"]
    fields = ("qpos", "qvel", "ctrl", "act", "time", "qacc_warmstart")
    state = {key: getattr(collector.engine.data, key).numpy() for key in fields}
    target, action = controller.arm_target.numpy(), controller.action.numpy()
    limits = collector.engine.model.actuator_forcerange.numpy()
    wait = batch_tracking.Request(
        manipulation.hold_segment(state["qpos"][0], ticks=2, name="batch_wait"),
        scene,
        -1,
        guard=False,
        grip_effort=0.123,
    )
    while executor.pending[1] is not None:
        result = executor.advance([wait, long])
        assert result[0][0]["action"].shape[0] == 0
        assert result[0][1]["executed_control_steps"] == 0
        assert result[0][1]["paused_control_steps"] > 0
        for key in fields:
            np.testing.assert_array_equal(
                getattr(collector.engine.data, key).numpy()[0], state[key][0]
            )
        np.testing.assert_array_equal(controller.arm_target.numpy()[0], target[0])
        np.testing.assert_array_equal(controller.action.numpy()[0], action[0])
        np.testing.assert_array_equal(
            collector.engine.model.actuator_forcerange.numpy()[0], limits[0]
        )
    assert result[1][1]["control_steps"] == len(expected.references)
    assert result[1][1]["padding_control_steps"] == 0
    assert result[1][1]["failure"] == 0
    # Recycle the sibling while slot zero is paused, then resume slot zero.
    qpos = collector.engine.data.qpos.numpy()
    qvel = collector.engine.data.qvel.numpy()
    qpos[1], qvel[1] = home, 0
    world_slots.reset(controller, np.array([False, True]), qpos=qpos, qvel=qvel)
    for key in fields:
        np.testing.assert_array_equal(
            getattr(collector.engine.data, key).numpy()[0], state[key][0]
        )
    np.testing.assert_array_equal(controller.arm_target.numpy()[0], target[0])
    resumed = executor.advance([short, request(0, "batch_wait")])
    assert resumed[0][1]["control_steps"] == first[0][1]["control_steps"]
    assert resumed[1][1]["control_steps"] == 0
    assert collector.engine.data.time.numpy()[0] > state["time"][0]


def test_scheduler_advances_real_robot_while_sibling_has_no_plan():
    import threading

    goals = np.array([[0.425, 0, 0.01]])
    model, home = models.make_model([[0.02] * 3], goals)
    model.opt.timestep = 0.002
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.iterations = 1000
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    collector = teacher.make_collector(model, goals, 2, 1, 0)
    collector.engine.model.block_dim.linesearch_iterative = 128
    controller = collector.controller = active.ActivePathControl(collector)
    collector.reset(qpos=np.tile(home, (2, 1)), qvel=np.zeros((2, model.nv)))
    scene = collision.Scene(model, home)
    request = batch_tracking.Request(
        manipulation.hold_segment(home, ticks=12), scene, -1, guard=False
    )
    waiting = batch_tracking.Request(
        manipulation.hold_segment(home, ticks=2, name="batch_wait"),
        scene,
        -1,
        guard=False,
    )
    executor = segments.SegmentExecutor(controller, pause_planning_worlds=True)
    # Warm the graph before the event deadline; compilation is not scheduling.
    executor.advance([request, request])
    before = collector.engine.data.time.numpy().copy()
    advanced = threading.Event()

    def execute(requests):
        results = executor.advance(requests)
        if requests[0] is waiting:
            now = collector.engine.data.time.numpy()
            assert now[0] == before[0] and now[1] > before[1]
        return results

    scheduler = batching.SlotGroup(
        2,
        execute,
        lambda w: waiting,
        lambda requests: {w: None for w in requests},
        pause_planning_worlds=True,
    )

    def planning():
        assert advanced.wait(10), "Ready world could not progress during planning"
        scheduler.segment(0, request)
        scheduler.close(0)

    def ready():
        trace, event = scheduler.segment(1, request)
        assert len(trace["action"]) == event["requested_steps"]
        assert event["padding_control_steps"] == 0
        assert event["failure"] == 0
        advanced.set()
        scheduler.close(1)

    scheduler.run([planning, ready])
