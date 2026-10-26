"""Runtime execution batching."""

import concurrent.futures
import contextlib
import dataclasses
import queue
import threading

import numpy as np

import castle_reasoning.runtime.diagnostics.timing as timing


@dataclasses.dataclass
class Call:
    world: int
    request: object
    finishing: bool
    result: concurrent.futures.Future
    trace_context: dict = dataclasses.field(default_factory=dict)


class StageGroup:
    """Run independent episodes with physical holds in terminated batch slots.

    Only the calling thread invokes execute/wait_request. Worker threads may
    perform independent native planning, but submit every GPU segment here.
    """

    def __init__(self, worlds, execute, wait_request):
        self.worlds = worlds
        self.execute = execute
        self.wait_request = wait_request
        self.calls = queue.Queue()
        self.lock = threading.Lock()
        self.outstanding = set()
        self.error = None
        self.on_error = None

    def _call(self, world, request, finishing):
        future = concurrent.futures.Future()
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Batch stage aborted") from self.error
            self.outstanding.add(future)
            timing.mark("coordinator.enqueue", slot=world)
            self.calls.put(Call(world, request, finishing, future, timing.current()))
        try:
            with timing.span("ipc.wait.coordinator", slot=world):
                return future.result()
        finally:
            with self.lock:
                self.outstanding.discard(future)

    def segment(self, world, request):
        return self._call(world, request, False)

    def finish(self, world):
        """End this world's episode without waiting for its siblings."""
        return self._call(world, None, True)

    def run(self, workers):
        if len(workers) != self.worlds or not workers:
            raise ValueError("One worker is required per simulator world")
        with concurrent.futures.ThreadPoolExecutor(self.worlds) as pool:
            futures = [pool.submit(worker) for worker in workers]
            try:
                self._drive(futures)
                return [future.result() for future in futures]
            except BaseException as error:
                if self.on_error is not None:
                    self.on_error()
                # Release blocked callers before joining worker threads. A worker
                # still planning sees this same error when it next submits work.
                with self.lock:
                    self.error = error
                    for future in self.outstanding:
                        if not future.done():
                            future.set_exception(error)
                raise

    def _drive(self, futures):
        pending = {}
        finished = set()
        while True:
            while len(pending) + len(finished) < self.worlds:
                for world, future in enumerate(futures):
                    if future.done():
                        future.result()  # Propagate an actual worker exception.
                        if world not in finished:
                            raise RuntimeError(
                                f"World {world} returned without finishing"
                            )
                try:
                    call = self.calls.get(timeout=0.05)
                except queue.Empty:
                    continue
                if (
                    not 0 <= call.world < self.worlds
                    or call.world in pending
                    or call.world in finished
                ):
                    raise RuntimeError("Invalid or duplicate world request")
                if call.finishing:
                    finished.add(call.world)
                    call.result.set_result([])
                else:
                    pending[call.world] = call
            if len(finished) == self.worlds:
                return
            requests = [
                self.wait_request(world)
                if world in finished
                else pending[world].request
                for world in range(self.worlds)
            ]
            results = self.execute(requests)
            if len(results) != self.worlds:
                raise RuntimeError("Executor returned the wrong number of worlds")
            for world, result in enumerate(results):
                if world in finished:
                    continue  # GPU owner accounts for post-terminal computation.
                call = pending[world]
                if result is None:
                    continue
                del pending[world]
                call.result.set_result(result)


class SlotGroup(StageGroup):
    """Use the GPU owner thread for both masked resets and shared execution.

    By default, active planners pause the batch until their next request.
    Opt-in masked execution freezes only the non-ready worlds. Finished episodes
    release their slots while CPU audits run.
    Pending segments in other slots retain their exact reference and cursor.
    """

    def __init__(self, worlds, execute, wait_request, reset_slots, *,
                 pause_planning_worlds=False):
        super().__init__(worlds, execute, wait_request)
        self.reset_slots = reset_slots
        self.pause_planning_worlds = pause_planning_worlds

    def segment(self, world, request):
        return self._call(world, ("segment", request), False)

    def finish(self, world):
        return self._call(world, ("finish", None), False)

    def reset(self, world, request):
        return self._call(world, ("reset", request), False)

    def close(self, world):
        return self._call(world, ("close", None), False)

    @timing.measured("coordinator")
    def _drive(self, futures):
        with contextlib.ExitStack() as waiting:
            return self._drive_traced(futures, waiting)

    def _drive_traced(self, futures, waiting):
        pending, resets, idle, closed = {}, {}, set(), set()
        waiting_active, batch = False, 0
        while len(closed) < self.worlds:
            for world, future in enumerate(futures):
                if future.done():
                    future.result()
                    if world not in closed:
                        raise RuntimeError(f"World {world} returned without closing")
            ready = self.pause_planning_worlds or len(pending) + len(idle) == self.worlds
            runnable = resets or (ready and pending)
            if not runnable and not waiting_active:
                missing = sorted(set(range(self.worlds)) - set(pending) - idle)
                waiting.enter_context(timing.span("coordinator.wait", missing=missing,
                                               pending=sorted(pending), idle=sorted(idle),
                                               closed=sorted(closed)))
                waiting_active = True
            try:
                call = self.calls.get(timeout=0 if runnable else 0.05)
            except queue.Empty:
                if resets:
                    waiting.close()
                    waiting_active = False
                    # Drain only available requests. A lone reset must not wait
                    # for unrelated planning or for every world to finish.
                    with timing.span("reset", slots=sorted(resets), size=len(resets)):
                        results = self.reset_slots({
                            world: call.request[1] for world, call in resets.items()
                        })
                    if results.keys() != resets.keys():
                        raise RuntimeError("Reset returned the wrong set of worlds")
                    for world, call in resets.items():
                        call.result.set_result(results[world])
                    resets.clear()
                    continue
                if not ready or not pending:
                    continue
            else:
                waiting.close()
                waiting_active = False
                world = call.world
                if (not 0 <= world < self.worlds or world in pending
                        or world in resets or world in closed):
                    raise RuntimeError("Invalid or duplicate slot request")
                kind, payload = call.request
                timing.mark("coordinator.dequeue", slot=world, kind=kind,
                            request_context=call.trace_context)
                if kind == "segment":
                    idle.discard(world)
                    pending[world] = call
                elif kind == "reset":
                    idle.discard(world)
                    resets[world] = call
                elif kind in ("finish", "close"):
                    idle.add(world)
                    if kind == "close":
                        closed.add(world)
                    call.result.set_result([])
                else:
                    raise RuntimeError(f"Unknown slot operation: {kind}")
                continue
            waiting.close()
            waiting_active = False
            requests = [
                self.wait_request(world) if world not in pending else pending[world].request[1]
                for world in range(self.worlds)
            ]
            with timing.span("batch", batch=batch, active=sorted(pending),
                             paused=sorted(set(range(self.worlds)) - set(pending))
                             if self.pause_planning_worlds else []):
                results = self.execute(requests)
            batch += 1
            if len(results) != self.worlds:
                raise RuntimeError("Executor returned the wrong number of worlds")
            for world, result in enumerate(results):
                if world in pending and result is not None:
                    pending.pop(world).result.set_result(result)


def handoff(snapshot, samples):
    """Restore each available episode's exact endpoint at a stage boundary."""
    result = {key: value.copy() for key, value in snapshot.items()}
    if len(samples) != len(result["qpos"]):
        raise ValueError("One episode slot per simulator world is required")
    for world, sample in enumerate(samples):
        if sample is not None:
            for key in ("qpos", "qvel"):
                result[key][world] = sample[key][-1]
    return result


def replay_commands(samples):
    """Pad only after termination; return commands, efforts and true lengths."""
    lengths = np.array(
        [0 if sample is None else len(sample["action"]) for sample in samples],
        dtype=np.int32,
    )
    if not len(lengths) or not lengths.max():
        raise ValueError("At least one nonempty replay episode is required")
    actions = np.zeros((int(lengths.max()), len(samples), 7), np.float32)
    efforts = np.ones(actions.shape[:2], np.float32)
    actions[:, :, 6] = -1
    for world, (sample, count) in enumerate(zip(samples, lengths, strict=True)):
        if not count:
            continue
        if len(sample["qpos"]) != count + 1 or len(sample["qvel"]) != count + 1:
            raise ValueError("Replay requires T actions and T+1 states")
        actions[:count, world] = sample["action"]
        efforts[:count, world] = sample["grip_effort"]
        actions[count:, world, 6] = sample["action"][-1, 6]
        efforts[count:, world] = sample["grip_effort"][-1]
    return actions, efforts, lengths
