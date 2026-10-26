"""Runtime simulation engine."""

import dataclasses
import time
from typing import Any

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

OverflowType = mujoco_warp.OverflowType


@wp.func
def _mix_u32_device(value: wp.uint32) -> wp.uint32:
    value = value ^ (value >> wp.uint32(16))
    value = value * wp.uint32(0x7FEB352D)
    value = value ^ (value >> wp.uint32(15))
    value = value * wp.uint32(0x846CA68B)
    return value ^ (value >> wp.uint32(16))


@wp.func
def _counter_random_u32_device(
    seed: int, episode: int, world: int, stream: int
) -> wp.uint32:
    result = _mix_u32_device(wp.uint32(seed) ^ wp.uint32(0x9E3779B9))
    episode_key = _mix_u32_device(wp.uint32(episode) + wp.uint32(0x85EBCA6B))
    result = _mix_u32_device(result ^ episode_key)
    world_key = _mix_u32_device(wp.uint32(world) + wp.uint32(0xC2B2AE35))
    result = _mix_u32_device(result ^ world_key)
    stream_key = _mix_u32_device(wp.uint32(stream) + wp.uint32(0x27D4EB2F))
    return _mix_u32_device(result ^ stream_key)


@wp.kernel
def _increment_episodes(
    reset: wp.array(dtype=bool), episodes: wp.array(dtype=int)
):
    world = wp.tid()
    if reset[world]:
        episodes[world] = episodes[world] + 1


@wp.kernel
def _write_counter_random(
    reset: wp.array(dtype=bool),
    episodes: wp.array(dtype=int),
    global_world_ids: wp.array(dtype=int),
    seed: int,
    output: wp.array2d(dtype=wp.uint32),
):
    world, stream = wp.tid()
    if reset[world]:
        output[world, stream] = _counter_random_u32_device(
            seed, episodes[world], global_world_ids[world], stream
        )


@wp.kernel
def _masked_zero(reset: wp.array(dtype=bool), output: wp.array2d(dtype=float)):
    world, column = wp.tid()
    if reset[world]:
        output[world, column] = 0.0


@dataclasses.dataclass(frozen=True)
class BufferCapacities:
    """Explicit MJWarp dynamic-buffer capacity overrides."""

    nconmax: int | None = None
    naconmax: int | None = None
    njmax: int | None = None
    njmax_nnz: int | None = None
    nvmax: int | None = None

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}.")

    def make_data_kwargs(self) -> dict[str, int]:
        """Returns only explicitly configured `make_data` arguments."""
        return {
            name: value
            for name, value in dataclasses.asdict(self).items()
            if value is not None
        }


class SimulationOverflowError(RuntimeError):
    """Raised when any world reports an MJWarp overflow bit."""

    def __init__(self, report: list[dict[str, Any]]):
        self.report = report
        summary = ", ".join(
            f"world {record['global_world']}: {'|'.join(record['flags'])}"
            for record in report[:10]
        )
        if len(report) > 10:
            summary += f", and {len(report) - 10} more"
        super().__init__(f"MJWarp overflow invalidated the run ({summary}).")


def overflow_report(
    masks: np.ndarray, *, global_world_start: int = 0
) -> list[dict[str, Any]]:
    """Decodes nonzero per-world overflow masks into inspectable records."""
    masks = np.asarray(masks, dtype=np.int64)
    if masks.ndim != 1:
        raise ValueError(
            f"Overflow masks must be one-dimensional, got {masks.shape}."
        )
    records = []
    for local_world in np.flatnonzero(masks):
        mask = int(masks[local_world])
        flags = [flag.name for flag in OverflowType if mask & int(flag)]
        records.append(
            {
                "local_world": int(local_world),
                "global_world": global_world_start + int(local_world),
                "mask": mask,
                "flags": flags or ["UNKNOWN"],
            }
        )
    return records


def raise_for_overflow(
    masks: np.ndarray, *, global_world_start: int = 0
) -> None:
    """Raises when any overflow mask is nonzero."""
    report = overflow_report(masks, global_world_start=global_world_start)
    if report:
        raise SimulationOverflowError(report)


@dataclasses.dataclass(frozen=True)
class StepMetrics:
    """Synchronized warm-step timing for one engine or shard group."""

    seconds: float
    physics_steps: int
    world_steps: int
    used_graph: bool

    @property
    def world_steps_per_second(self) -> float:
        return self.world_steps / self.seconds


class MJWarpEngine:
    """Owns one homogeneous model, data batch, and optional CUDA graph."""

    def __init__(
        self,
        native_model: mujoco.MjModel,
        *,
        nworld: int,
        device: str,
        capacities: BufferCapacities | None = None,
        seed: int = 0,
        global_world_start: int = 0,
        random_streams: int = 8,
        observation_width: int = 1,
        task_state_width: int = 1,
        graph: bool = True,
        graph_physics_steps: int = 1,
    ):
        if nworld < 1:
            raise ValueError(f"nworld must be positive, got {nworld}.")
        if seed < 0 or seed > 0x7FFFFFFF:
            raise ValueError(
                "seed must fit a nonnegative signed 32-bit integer."
            )
        if random_streams < 1:
            raise ValueError("random_streams must be positive.")
        if observation_width < 1 or task_state_width < 1:
            raise ValueError("Device feature widths must be positive.")
        if graph_physics_steps < 1:
            raise ValueError("graph_physics_steps must be positive.")
        self.native_model = native_model
        self.nworld = nworld
        self.seed = seed
        self.global_world_start = global_world_start
        self.random_streams = random_streams
        self.device = wp.get_device(device)
        self.graph_enabled = graph
        self.graph_physics_steps = graph_physics_steps
        if graph and not self.device.is_cuda:
            raise ValueError("CUDA graph capture requires a CUDA device.")
        capacities = capacities or BufferCapacities()

        with wp.ScopedDevice(self.device):
            self.model = mujoco_warp.put_model(native_model)
            self.data = mujoco_warp.make_data(
                native_model,
                nworld=nworld,
                **capacities.make_data_kwargs(),
            )
            self.reset_mask = wp.zeros(nworld, dtype=bool)
            self.global_world_ids = wp.array(
                np.arange(
                    global_world_start,
                    global_world_start + nworld,
                    dtype=np.int32,
                ),
                dtype=int,
            )
            self.episode = wp.full(nworld, -1, dtype=int)
            self.random_u32 = wp.zeros(
                (nworld, random_streams), dtype=wp.uint32
            )
            self.observation = wp.zeros(
                (nworld, observation_width), dtype=float
            )
            self.task_state = wp.zeros((nworld, task_state_width), dtype=float)
            self.reset(np.ones(nworld, dtype=bool))
            self.graph = None
            self.cold_capture_seconds = 0.0
            if graph:
                started = time.perf_counter()
                with wp.ScopedCapture() as capture:
                    for _ in range(graph_physics_steps):
                        mujoco_warp.step(self.model, self.data)
                self.graph = capture.graph
                self.cold_capture_seconds = time.perf_counter() - started

    @classmethod
    def from_xml_string(cls, xml: str, **kwargs: Any) -> "MJWarpEngine":
        """Constructs an engine from a self-contained MJCF string."""
        return cls(mujoco.MjModel.from_xml_string(xml), **kwargs)

    def _device_array(self, value: np.ndarray, *, dtype: Any) -> wp.array:
        with wp.ScopedDevice(self.device):
            return wp.array(value, dtype=dtype)

    def reset(self, mask: np.ndarray) -> None:
        """Resets selected worlds and advances their semantic episode keys."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.nworld,):
            raise ValueError(
                "Reset mask must have shape "
                f"{(self.nworld,)}, got {mask.shape}."
            )
        with wp.ScopedDevice(self.device):
            wp.copy(self.reset_mask, self._device_array(mask, dtype=bool))
            mujoco_warp.reset_data(self.model, self.data, reset=self.reset_mask)
            self.launch_reset_metadata(self.reset_mask)
            mujoco_warp.forward(self.model, self.data)
            wp.synchronize_device(self.device)
        self.assert_healthy()

    def launch_reset_metadata(self, mask: wp.array) -> None:
        """Enqueues episode, RNG, and feature reset for a device mask."""
        wp.launch(
            _increment_episodes,
            dim=self.nworld,
            inputs=[mask, self.episode],
        )
        wp.launch(
            _write_counter_random,
            dim=(self.nworld, self.random_streams),
            inputs=[
                mask,
                self.episode,
                self.global_world_ids,
                self.seed,
                self.random_u32,
            ],
        )
        wp.launch(
            _masked_zero,
            dim=self.observation.shape,
            inputs=[mask, self.observation],
        )
        wp.launch(
            _masked_zero,
            dim=self.task_state.shape,
            inputs=[mask, self.task_state],
        )

    def set_state(self, *, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Copies a complete batched generalized state onto the device."""
        qpos = np.asarray(qpos, dtype=np.float32)
        qvel = np.asarray(qvel, dtype=np.float32)
        expected_qpos = (self.nworld, self.native_model.nq)
        expected_qvel = (self.nworld, self.native_model.nv)
        if qpos.shape != expected_qpos or qvel.shape != expected_qvel:
            raise ValueError(
                f"State shapes must be {expected_qpos} and {expected_qvel}, "
                f"got {qpos.shape} and {qvel.shape}."
            )
        with wp.ScopedDevice(self.device):
            wp.copy(self.data.qpos, self._device_array(qpos, dtype=float))
            wp.copy(self.data.qvel, self._device_array(qvel, dtype=float))
            mujoco_warp.forward(self.model, self.data)
            wp.synchronize_device(self.device)
        self.assert_healthy()

    def set_control(self, control: np.ndarray) -> None:
        """Copies one control row per world onto the device."""
        control = np.asarray(control, dtype=np.float32)
        expected = (self.nworld, self.native_model.nu)
        if control.shape != expected:
            raise ValueError(
                f"Control must have shape {expected}, got {control.shape}."
            )
        with wp.ScopedDevice(self.device):
            wp.copy(self.data.ctrl, self._device_array(control, dtype=float))

    def _launch_steps(self, steps: int, use_graph: bool) -> bool:
        """Launches physics and returns whether any graph launch was used."""
        with wp.ScopedDevice(self.device):
            if use_graph:
                if self.graph is None:
                    raise ValueError(
                        "Graph execution requested without a captured graph."
                    )
                graph_launches, direct_steps = divmod(
                    steps, self.graph_physics_steps
                )
                for _ in range(graph_launches):
                    wp.capture_launch(self.graph)
                for _ in range(direct_steps):
                    mujoco_warp.step(self.model, self.data)
                return graph_launches > 0
            else:
                for _ in range(steps):
                    mujoco_warp.step(self.model, self.data)
                return False

    def step(self, *, steps: int = 1, graph: bool | None = None) -> StepMetrics:
        """Advances physics, synchronizes, and fails closed on invalid state."""
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}.")
        use_graph = self.graph_enabled if graph is None else graph
        started = time.perf_counter()
        used_graph = self._launch_steps(steps, use_graph)
        wp.synchronize_device(self.device)
        seconds = time.perf_counter() - started
        self.assert_healthy()
        return StepMetrics(seconds, steps, steps * self.nworld, used_graph)

    def assert_healthy(self) -> None:
        """Fails on overflow or any nonfinite primary device state."""
        masks = self.data.overflow.numpy()
        raise_for_overflow(masks, global_world_start=self.global_world_start)
        arrays = {
            "qpos": self.data.qpos.numpy(),
            "qvel": self.data.qvel.numpy(),
            "ctrl": self.data.ctrl.numpy(),
            "observation": self.observation.numpy(),
            "task_state": self.task_state.numpy(),
        }
        nonfinite = [
            name
            for name, value in arrays.items()
            if not np.isfinite(value).all()
        ]
        if nonfinite:
            raise FloatingPointError(
                f"Nonfinite MJWarp device state in: {', '.join(nonfinite)}."
            )

    def contact_counts(self) -> np.ndarray:
        """Returns per-world counts from MJWarp's shared contact buffer."""
        contact_count = int(self.data.nacon.numpy()[0])
        if contact_count == 0:
            return np.zeros(self.nworld, dtype=np.int64)
        world_ids = self.data.contact.worldid.numpy()[:contact_count]
        return np.bincount(world_ids, minlength=self.nworld)[: self.nworld]

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copies a small inspection snapshot to the host."""
        wp.synchronize_device(self.device)
        return {
            "qpos": self.data.qpos.numpy(),
            "qvel": self.data.qvel.numpy(),
            "ctrl": self.data.ctrl.numpy(),
            "energy": self.data.energy.numpy(),
            "overflow": self.data.overflow.numpy(),
            "reset_mask": self.reset_mask.numpy(),
            "episode": self.episode.numpy(),
            "random_u32": self.random_u32.numpy(),
            "observation": self.observation.numpy(),
            "task_state": self.task_state.numpy(),
        }
