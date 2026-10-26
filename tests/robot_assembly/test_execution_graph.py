"""GPU graph reuse preserves controls, guards, slot resets and exact lengths."""

import dataclasses
import os

import numpy as np
import pytest
import warp as wp

import castle_reasoning.data.assembly.teacher as teacher
import castle_reasoning.runtime.control.batch_tracking as batch_tracking
import castle_reasoning.runtime.control.tracking as tracking
import castle_reasoning.runtime.planning.collision as collision
import castle_reasoning.runtime.planning.manipulation as manipulation
import castle_reasoning.runtime.simulation.models as simulation_models
import castle_reasoning.runtime.simulation.world_slots as world_slots

pytestmark = pytest.mark.skipif(
    os.environ.get("CONSTRUCTION_GPU_TESTS") != "1", reason="GPU opt-in"
)


def test_graph_reuses_buffers_with_fresh_capture_physics_parity(monkeypatch):
    goals = np.array([[0.425, 0, 0.01]])
    models, homes = zip(
        *(simulation_models.make_model([[width, 0.02, 0.02]], goals) for width in (0.02, 0.022)),
        strict=True,
    )
    pool = world_slots.ModelPool(models, [0, 0])
    collector = teacher.make_collector(models[0], goals, 2, 1, 0)
    engine = collector.engine
    engine.model, engine.native_models = pool.model, pool.native_models
    collector.controller = controller = tracking.PathControl(collector)
    original_capture = wp.ScopedCapture
    captures = []

    def capture(*args, **kwargs):
        result = original_capture(*args, **kwargs)
        captures.append(result)
        return result

    monkeypatch.setattr(wp, "ScopedCapture", capture)
    scenarios = [
        (8, "hold"),
        (2, "contact"),
        (65, "literal"),
        (1, "hold"),
        (3, "tracking"),
        (2, "scene"),
        (2, "attachment"),
        (130, "deadline"),
        (1, "hold"),
        (1, "replacement"),
    ]
    for count, mode in scenarios:
        # Installing another compatible model mutates rows without new pointers.
        pool.select([1], [int(mode == "literal")])
        initial = np.array([homes[index] for index in pool.assignments])
        world_slots.reset(
            controller,
            np.ones(2, bool),
            qpos=initial,
            qvel=np.zeros((2, models[0].nv)),
        )
        requests = [
            batch_tracking.Request(
                manipulation.hold_segment(qpos, ticks=max(70, count + 10)),
                collision.Scene(model, qpos),
                -1,
                guard=False,
                monitor_scene=False,
            )
            for model, qpos in zip(pool.native_models, initial, strict=True)
        ]
        base = batch_tracking.prepare(requests, initial)
        prepared = dataclasses.replace(
            base,
            lengths=np.full(2, count),
            **{
                name: getattr(base, name)[:count].copy()
                for name in ("references", "bias", "efforts", "literal_actions")
            },
        )
        prepared.efforts[:, 0], prepared.efforts[:, 1] = 1.0, 3.0
        if mode == "contact":
            # Real table contact triggers one world's guard, not a fake kernel.
            prepared.guards[1] = 1
            prepared.contacts[1] = 1.0
        if mode == "literal":
            prepared.literal_mask[:] = 1
            prepared.literal_actions[:, :, 6] = -0.9
            prepared.literal_actions[:, 0, 0] = 0.002
        if mode == "tracking":
            prepared.references[:, 1, 0] += 0.2
        if mode == "scene":
            prepared.monitors[1] = 1
            prepared.scene_references[1, 14] += 0.01
        if mode == "attachment":
            prepared.attachments[1], prepared.parts[1] = 1, 0
            prepared.offsets[1] = [10, 0, 0]
        if mode == "replacement":
            with wp.ScopedDevice(engine.device):
                controller.arm_target = wp.clone(controller.arm_target)
        kwargs = {"deadline": 0} if mode == "deadline" else {}
        before = len(captures)
        reused = batch_tracking.execute_prepared(
            controller, requests, prepared, **kwargs
        )
        added = len(captures) - before
        assert added == int(count in (8, 65, 130) or mode == "replacement"), (
            count,
            mode,
            added,
        )
        cache = controller._prepared_execution
        del controller._prepared_execution
        world_slots.reset(
            controller,
            np.ones(2, bool),
            qpos=initial,
            qvel=np.zeros((2, models[0].nv)),
        )
        fresh = batch_tracking.execute_prepared(
            controller, requests, prepared, **kwargs
        )
        controller._prepared_execution = cache
        for (actual, event), (expected, reference) in zip(reused, fresh, strict=True):
            assert len(actual["action"]) == (64 if mode == "deadline" else count)
            for key in actual:
                np.testing.assert_allclose(
                    actual[key], expected[key], atol=1e-7, rtol=1e-6
                )
            for key in (
                "failure",
                "control_steps",
                "padding_control_steps",
                "device_contact_violations",
            ):
                assert event[key] == reference[key], (mode, key)
        if mode == "contact":
            assert reused[1][1]["failure"] == 4
            assert reused[1][1]["device_contact_violations"]
        elif mode in ("tracking", "scene", "attachment", "deadline"):
            assert (
                reused[1][1]["failure"]
                == {
                    "tracking": 1,
                    "scene": 2,
                    "attachment": 3,
                    "deadline": 5,
                }[mode]
            )
        else:
            assert all(event["failure"] == 0 for _, event in reused)
            assert all(not event["device_contact_violations"] for _, event in reused)
        np.testing.assert_array_equal(
            engine.model.actuator_forcerange.numpy()[:, 6], [[-1, 1], [-3, 3]]
        )
