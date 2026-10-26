"""Runtime simulation model_batch."""

import copy
import dataclasses

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


def pad_plan(plan, capacity):
    """Keep real parts as a prefix; inactive slots are isolated outside the workcell."""
    count = len(plan["sizes"])
    if not isinstance(capacity, int) or capacity < count:
        raise ValueError("Block capacity must include every real part.")
    result = copy.deepcopy(plan)
    # A plan saved by a collector is already padded; keep its real-part count.
    result["active_part_count"] = plan.get("active_part_count", count)
    for slot in range(count, capacity):
        position = [2.0 + 0.03 * slot, 0.0, 0.5]
        result["sizes"].append([0.01] * 3)
        result["target"].append(position.copy())
        result["parking"].append(position.copy())
    return result


def stack_models(models, device="cuda:0"):
    """Copy all differing batchable fields from compatible compiled variants.

    Convert through MJ-Warp's schema so transformed/derived arrays match its
    representation. Call before capturing any graphs that reference this model.
    """
    if not models:
        raise ValueError("At least one compiled model is required.")
    for name in ("nq", "nv", "nbody", "njnt", "ngeom"):
        if any(getattr(m, name) != getattr(models[0], name) for m in models[1:]):
            raise ValueError(f"Incompatible model topology: {name}")
    sleeping = any(m.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP for m in models)
    with wp.ScopedDevice("cpu"):
        sources = [mjw.put_model(model) for model in models]
    with wp.ScopedDevice(device):
        target = mjw.put_model(models[0])
        _merge_fields(target, sources, sleeping)
    return target


def _merge_fields(target, sources, sleeping, prefix=""):
    """Only explicitly batchable schema fields may differ between worlds."""
    for field in dataclasses.fields(sources[0]):
        values = [getattr(source, field.name) for source in sources]
        first = values[0]
        name = prefix + field.name
        if dataclasses.is_dataclass(first):
            _merge_fields(getattr(target, field.name), values, sleeping, name + ".")
        elif isinstance(first, wp.array):
            arrays = [value.numpy() for value in values]
            if all(np.array_equal(arrays[0], a) for a in arrays[1:]):
                continue
            shape = getattr(field.type, "shape", ())
            if not shape or shape[0] != "*":
                # MJ-Warp 3.12 reads dof_length only in its sleeping kernels.
                if name == "dof_length" and not sleeping:
                    continue
                raise ValueError(f"Incompatible unbatched field: {name}")
            if any(a.shape != arrays[0].shape for a in arrays[1:]):
                raise ValueError(f"Incompatible parameter shape: {name}")
            combined = wp.array(np.concatenate(arrays), dtype=first.dtype)
            combined._is_batched = True
            setattr(target, field.name, combined)
        elif not all(_shared_equal(first, value) for value in values[1:]):
            raise ValueError(f"Incompatible shared field: {name}")


def _shared_equal(left, right):
    """Compare shared topology, including tuples of Warp arrays by tree level."""
    if isinstance(left, wp.array):
        return np.array_equal(left.numpy(), right.numpy())
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _shared_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return np.array_equal(left, right)
