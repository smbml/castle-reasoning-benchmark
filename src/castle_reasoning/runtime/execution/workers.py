"""Runtime execution workers."""

import dataclasses
import multiprocessing
import os
import traceback

import mujoco

import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.execution.preparation as preparation


class RemoteGroup:
    """A child sees one concrete state and requests all simulator work remotely."""

    def __init__(self, connection, model_path, snapshot, *, source=None,
                 permits=None, cancelled=None):
        self.connection = connection
        self.sequence = 0
        with timing.span("startup.load_model"):
            self.native_model = mujoco.MjModel.from_binary_path(str(model_path))
        self.native_models = [self.native_model]
        self.snapshot = snapshot
        self.stage = self
        self.planning_budget = None
        self.source, self.permits, self.cancelled = source, permits, cancelled

    def _exchange(self, kind, payload):
        if self.planning_budget is not None and self.planning_budget.active:
            raise RuntimeError("Release planner admission before requesting simulation")
        self.sequence += 1
        with timing.span("request", request=self.sequence, kind=kind,
                         part=getattr(payload, "part", None)):
            if kind == "segment":
                if self.source is not None:
                    payload = preparation.prepare(
                        payload, self.snapshot["qpos"], self.source,
                        permits=self.permits, cancelled=self.cancelled,
                        location="planner_process",
                    )
                payload = dataclasses.replace(payload, scene=None)
            with timing.span("ipc.send.request"):
                self.connection.send((kind, payload))
                payload = None  # Drop prepared arrays before waiting for execution.
            with timing.span("ipc.wait.simulation"):
                response = self.connection.recv()
                if self.source is None:
                    result, self.snapshot = response
                else:
                    result, self.snapshot, self.source = response
            timing.mark("ipc.result_received")
            return result

    def segment(self, world, request):
        # MjModel/MjData stay local. The parent uses the same saved geometry and
        # reconstructs its own Scene for reference preparation/contact rules.
        return self._exchange("segment", request)

    def finish(self, world):
        return self._exchange("finish", None)

    def restore(self, world, qpos, qvel):
        return self._exchange("restore", (qpos, qvel))


def _child(connection, worker, packet):
    # Spawn, never fork, a GPU-owning interpreter. Children perform native CPU
    # planning/auditing only and cannot create a CUDA context by accident.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    trace = packet.pop("_trace", None)
    if trace is not None:
        timing.start(trace["directory"], **trace["context"])
        connection.send(("trace_ready", os.getpid()))
    try:
        with timing.span("worker.child"):
            worker(connection, packet)
    except BaseException:
        if packet.get("preparation_cancelled") is not None:
            packet["preparation_cancelled"].set()
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()
        timing.stop()


# A spawned planner re-imports the collector stack (warp, gymnasium, dm_control,
# ...) for about 210 MB of private memory, which caps worlds per host. Forkserver
# children share these preloaded imports copy-on-write (about 66 MB private).
# Importing them initializes no CUDA context, and planners use explicit RNGs.
PLANNER_PRELOAD = ["numpy", "scipy", "mujoco", 'castle_reasoning.runtime.execution.session']


START_METHODS = ("spawn", "forkserver")


def planner_context(method="spawn"):
    """Multiprocessing context for planner processes and the objects they share."""
    if method not in START_METHODS:
        raise ValueError(f"Planner start method must be one of {START_METHODS}")
    context = multiprocessing.get_context(method)
    if method == "forkserver":
        context.set_forkserver_preload(PLANNER_PRELOAD)
    return context


def relay(world, group, scene, packet, worker, start_method="spawn"):
    """Relay one child's requests through the existing stage rendezvous."""
    context = planner_context(start_method)
    connection, child = context.Pipe()
    trace = timing.child_config(slot=world)
    if trace is not None:
        packet = packet | {"_trace": trace}
    process = context.Process(target=_child, args=(child, worker, packet))
    with timing.span("process.start", slot=world):
        process.start()
    child.close()
    sequence = 0
    try:
        if trace is not None:
            with timing.span("process.bootstrap", slot=world, child_pid=process.pid):
                kind, payload = connection.recv()
                if kind != "trace_ready":
                    raise RuntimeError(f"Planner tracing bootstrap failed: {kind}")
        while True:
            try:
                with timing.span("ipc.wait.child", slot=world):
                    kind, payload = connection.recv()
            except EOFError as error:
                raise RuntimeError(
                    f"World {world} planner exited without a result"
                ) from error
            if kind == "error":
                raise RuntimeError(f"World {world} planner failed:\n{payload}")
            if kind == "done":
                return payload
            sequence += 1
            with timing.span("request", slot=world, request=sequence, kind=kind,
                             part=getattr(payload, "part", None)):
                timing.mark("ipc.request_received")
                if kind == "segment":
                    request = dataclasses.replace(payload, scene=scene)
                    result = group.stage.segment(world, request)
                elif kind == "restore":
                    result = group.restore(world, *payload)
                elif kind == "finish":
                    result = [
                        (dataclasses.replace(request, scene=None, preparation=None), value)
                        for request, value in group.stage.finish(world)
                    ]
                else:
                    raise RuntimeError(f"Unknown planner request: {kind}")
                with timing.span("ipc.snapshot_copy"):
                    snapshot = {
                        key: value[world : world + 1].copy()
                        for key, value in group.snapshot.items()
                    }
                with timing.span("ipc.send.result"):
                    response = (result, snapshot)
                    if packet.get("preparation_source") is not None:
                        response += (group.preparation_sources[world],)
                    connection.send(response)
                payload = request = None
    except BaseException:
        if packet.get("preparation_cancelled") is not None:
            packet["preparation_cancelled"].set()
        raise
    finally:
        connection.close()
        with timing.span("process.join", slot=world):
            process.join(timeout=1)
            if process.is_alive():
                timing.mark("process.terminated", child_pid=process.pid)
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
