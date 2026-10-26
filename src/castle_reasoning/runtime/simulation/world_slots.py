"""Runtime simulation world_slots."""

import dataclasses
from typing import Any

import mujoco_warp as mjw
import numpy as np
import warp as wp

import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.simulation.model_batch as model_batch


def _batch_fields(model, prefix=""):
    for field in dataclasses.fields(model):
        value = getattr(model, field.name)
        name = prefix + field.name
        if dataclasses.is_dataclass(value):
            yield from _batch_fields(value, name + ".")
        elif isinstance(value, wp.array):
            shape = getattr(field.type, "shape", ())
            if shape and shape[0] == "*" and value.size:
                yield name, model, field.name, value


class ModelPool:
    """Precompile a finite model bank; replace rows without invalidating graphs.

    Construct and install ``model`` before capturing controller graphs. All
    mutable model arrays have one row per slot, including initially equal rows.
    Shared topology is validated by the same schema checks as fixed batches.
    """

    @timing.measured('startup.model_bank')
    def __init__(self, models, assignments, device="cuda:0"):
        if not models or not assignments:
            raise ValueError("Nonempty models and slot assignments are required")
        if any(not 0 <= index < len(models) for index in assignments):
            raise ValueError("Invalid model assignment")
        self.models = list(models)
        self.assignments = list(assignments)
        self.native_models = [models[index] for index in assignments]
        self.device = wp.get_device(device)
        bank = model_batch.stack_models(models, device="cpu")
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(models[0])
            destinations = {
                name: (owner, field)
                for name, owner, field, _ in _batch_fields(self.model)
            }
            self.rows = []
            for name, _, _, source in _batch_fields(bank):
                values = source.numpy()
                selected = np.asarray(assignments) % len(values)
                destination = wp.array(values[selected], dtype=source.dtype)
                destination._is_batched = True
                owner, field = destinations[name]
                setattr(owner, field, destination)
                stored = wp.array(values, dtype=source.dtype)
                self.rows.append((destination, stored))

    def select(self, worlds, model_indices):
        """Restore selected model rows in place, including same-model resets."""
        worlds, model_indices = np.asarray(worlds), np.asarray(model_indices)
        if worlds.ndim != 1 or model_indices.shape != worlds.shape:
            raise ValueError("One model index per selected world is required")
        if not worlds.size:
            return
        if (
            worlds.dtype.kind not in "iu"
            or model_indices.dtype.kind not in "iu"
            or np.any((worlds < 0) | (worlds >= len(self.assignments)))
            or np.any((model_indices < 0) | (model_indices >= len(self.models)))
            or len(np.unique(worlds)) != len(worlds)
        ):
            raise ValueError("Invalid world or model index")
        with wp.ScopedDevice(self.device):
            selected = wp.array(worlds, dtype=int)
            indices = wp.array(model_indices, dtype=int)
            for destination, source in self.rows:
                width = int(np.prod(destination.shape[1:]))
                wp.launch(
                    _select_model_rows,
                    (len(worlds), width),
                    inputs=[selected, indices, source.reshape((source.shape[0], width)),
                            destination.reshape((destination.shape[0], width))],
                )
        # The owner's next reset uses this same device stream. No host read or
        # barrier is needed between model installation and the state reset.
        for world, model_index in zip(worlds, model_indices, strict=True):
            self.assignments[world] = int(model_index)
            self.native_models[world] = self.models[model_index]


@wp.kernel
def _select_model_rows(
    worlds: wp.array(dtype=int),
    indices: wp.array(dtype=int),
    source: wp.array2d(dtype=Any),
    destination: wp.array2d(dtype=Any),
):
    row, column = wp.tid()
    destination[worlds[row], column] = source[indices[row] % source.shape[0], column]


@wp.kernel
def _scatter_states(
    worlds: wp.array(dtype=int),
    source: wp.array2d(dtype=float),
    destination: wp.array2d(dtype=float),
):
    row, column = wp.tid()
    destination[worlds[row], column] = source[row, column]


@wp.kernel
def _clear_actions(mask: wp.array(dtype=bool), action: wp.array2d(dtype=float)):
    world, column = wp.tid()
    if mask[world]:
        action[world, column] = 0.0
        if column == 6:
            action[world, column] = -1.0


@timing.measured('reset.device')
def reset(controller, mask, *, qpos, qvel):
    """Reset only selected slots, preserving every continuing servo target."""
    engine = controller.engine
    mask = np.asarray(mask, dtype=bool)
    qpos, qvel = np.asarray(qpos, np.float32), np.asarray(qvel, np.float32)
    if mask.shape != (engine.nworld,):
        raise ValueError("One reset-mask entry per world is required")
    if qpos.shape != (engine.nworld, engine.native_model.nq) or qvel.shape != (
        engine.nworld,
        engine.native_model.nv,
    ):
        raise ValueError("Initial state must match the complete batch layout")
    if not np.isfinite(qpos[mask]).all() or not np.isfinite(qvel[mask]).all():
        raise ValueError("Selected initial states must be finite")
    if not mask.any():
        return
    with wp.ScopedDevice(engine.device):
        wp.copy(engine.reset_mask, wp.array(mask, dtype=bool))
        mjw.reset_data(engine.model, engine.data, reset=engine.reset_mask)
        engine.launch_reset_metadata(engine.reset_mask)
        # Preserve the parent's solver initialization. Omitting this forward
        # changes derived accelerations even in continuing worlds (parity test).
        mjw.forward(engine.model, engine.data)
        worlds = wp.array(np.flatnonzero(mask), dtype=int)
        for source, destination in ((qpos, engine.data.qpos), (qvel, engine.data.qvel)):
            selected = wp.array(source[mask], dtype=float)
            wp.launch(
                _scatter_states,
                selected.shape,
                inputs=[worlds, selected, destination],
            )
        wp.launch(
            _clear_actions,
            controller.action.shape,
            inputs=[engine.reset_mask, controller.action],
        )
        mjw.forward(engine.model, engine.data)
        controller.launch_sync_targets(engine.reset_mask)
        wp.synchronize_device(engine.device)
    engine.assert_healthy()
